"""Dataset builder for the Phase-0 residual ML research layer.

Produces the fundamental training observation -- PLAYER x GAMEWEEK -- from the existing
Quant backtest, asof-safely. This is the minimal dataset: identification, the Quant
prediction Q(x), the realised label y, the residual r = y - Q(x), and asof provenance.
Rolling feature columns are added in a second pass by feature_engineering.add_features(),
kept separate so this module stays small, auditable, and independently leakage-checkable.

Leakage discipline (see LEAKAGE_PROTOCOL.md): for each walk-forward step (season, gw) the
builder enters fpl_quant.backtest.asof_scope(), which shadows the three fact tables to
strictly-pre-deadline rows. Q(x) is read from ep_outputs (already produced asof the deadline
by the backtest). The label event_points is read from main.fact_player_season_stats at
gw == target AFTER the asof window closes -- it is the realised outcome, used only as the
target, never as a feature.

A clone of this repo carries no ingested DB, so build_dataset() raises a clear error if no
backtest steps exist, and the pipeline is provable via the synthetic fixtures in
tests/test_ml_dataset_builder.py.
"""

from __future__ import annotations

from typing import Iterator

import duckdb
import pandas as pd

from . import contract as C
from .leakage_checks import assert_minimal_dataset_invariants

# Late imports of fpl_quant internals (path bootstrap in __init__ makes these resolve).
# These are the SAME asof-safe helpers the Quant backtest itself uses -- not a parallel path.
from fpl_quant import backtest as bt
from fpl_quant import monte_carlo as mc


# ============================================================
# Step discovery
# ============================================================

def list_backtest_steps(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """One row per walk-forward step that produced an ep_outputs run, with provenance.

    Reads backtest_gameweek_steps (written by bt.run_gameweek_step). A step with
    ep_model_version IS NULL produced no prediction (e.g. an unfittable early gameweek) and is
    excluded -- there is no Q(x) to learn a residual against.
    """
    rows = con.execute(
        """
        SELECT season, gameweek, ep_model_version, mm_model_version, data_asof
        FROM backtest_gameweek_steps
        WHERE ep_model_version IS NOT NULL
        ORDER BY season, gameweek
        """
    ).fetchdf()
    if rows.empty:
        return []
    return rows.to_dict("records")


def seasons_available(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Distinct seasons that have at least one backtest step, in chronological order."""
    rows = con.execute(
        "SELECT DISTINCT season FROM backtest_gameweek_steps WHERE ep_model_version IS NOT NULL"
    ).fetchall()
    return sorted((r[0] for r in rows), key=C.season_sort_key)


# ============================================================
# Quant prediction extraction (player x fixture -> player x gw)
# ============================================================

def _quant_predictions_for_step(
    con: duckdb.DuckDBPyConnection, ep_model_version: int
) -> pd.DataFrame:
    """Q(x) for every player the Quant model gave a fixture in this step, reduced to player x gw.

    ep_outputs is player x fixture. For Phase 0 a player must have exactly one fixture in the
    gameweek (DGWs are skipped at the step level by the caller). Multiple fixtures for one player
    would be a DGW the caller failed to skip -- raised loudly rather than silently aggregated,
    because no aggregation semantics are defined (the same boundary expected_points.py itself
    draws for v1).
    """
    df = con.execute(
        """
        SELECT o.player_uid, o.fixture_match_id, o.ep_total,
               m.home_team_uid, m.away_team_uid, m.kickoff_time
        FROM ep_outputs o
        JOIN fact_match m ON m.match_id = o.fixture_match_id
        WHERE o.model_version = ?
        """,
        [ep_model_version],
    ).fetchdf()
    if df.empty:
        return df
    dup = df.groupby("player_uid").size()
    bad = dup[dup > 1].index.tolist()
    if bad:
        raise LeakageError(
            f"player(s) {bad} have >1 fixture in ep_model_version {ep_model_version} -- a DGW "
            "that should have been skipped before reaching the dataset builder"
        )
    df = df.rename(columns={"ep_total": C.COL_QUANT_PRED})
    return df


def _team_and_fixture_context(
    con: duckdb.DuckDBPyConnection, df: pd.DataFrame, season: str
) -> pd.DataFrame:
    """Attach team_uid, opponent_uid, home_away, position to each player's prediction row.

    Reuses monte_carlo._team_of_for_fixture() -- the SAME team-resolution helper the Quant
    backtest's own score_gameweek() uses (backtest.py). Team attribution is identification, not
    a feature value, so coupling to this established helper is consistent with existing code
    rather than a new resolution invented here.
    """
    if df.empty:
        return df

    # dim_player.position is static identity (knowable pre-deadline); not shadowed by asof_scope.
    pos = dict(con.execute(
        "SELECT player_uid, position FROM dim_player"
    ).fetchall())
    df[C.COL_POSITION] = df["player_uid"].map(pos)

    team_of: dict[str, str] = {}
    for match_id, home, away in df[["fixture_match_id", "home_team_uid", "away_team_uid"]].drop_duplicates().itertuples(index=False, name=None):
        team_of.update(mc._team_of_for_fixture(con, home, away, season))

    def _side(row: pd.Series) -> tuple[str | None, str | None, str | None]:
        team = team_of.get(row["player_uid"])
        if team is None:
            return None, None, None
        if team == row["home_team_uid"]:
            return team, row["away_team_uid"], "home"
        return team, row["home_team_uid"], "away"

    sides = df.apply(_side, axis=1, result_type="expand")
    sides.columns = [C.COL_TEAM_UID, C.COL_OPPONENT_UID, C.COL_HOME_AWAY]
    return pd.concat([df.drop(columns=["home_team_uid", "away_team_uid"]), sides], axis=1)


# ============================================================
# Label extraction (post-asof, realised outcome only)
# ============================================================

def _labels_for_step(
    con: duckdb.DuckDBPyConnection, season: str, gameweek: int, player_uids: list[str]
) -> pd.DataFrame:
    """The realised outcome y = event_points for each player in the target gameweek.

    Read from main.fact_player_season_stats (NOT the asof-shadowed temp table, which truncates
    the in-progress season to gw < target). event_points is the per-gameweek FPL score -- the
    label -- fetched only here, only as the target. A player with no row (did not appear in
    the gameweek snapshot) gets no observation: there is no realised outcome to learn against.
    """
    if not player_uids:
        return pd.DataFrame(columns=["player_uid", C.COL_ACTUAL])
    placeholders = ", ".join(["?"] * len(player_uids))
    return con.execute(
        f"""
        SELECT player_uid, event_points AS {C.COL_ACTUAL}
        FROM main.fact_player_season_stats
        WHERE season = ? AND gw = ? AND player_uid IN ({placeholders})
        """,
        [season, gameweek, *player_uids],
    ).fetchdf()


# ============================================================
# Public entrypoint
# ============================================================

class LeakageError(RuntimeError):
    """Raised when the built dataset violates a leakage invariant. Aborts before training."""


def _iter_steps(steps: list[dict], seasons: tuple[str, ...] | None) -> Iterator[dict]:
    for s in steps:
        if seasons is not None and s["season"] not in seasons:
            continue
        yield s


def build_minimal_dataset(
    con: duckdb.DuckDBPyConnection, seasons: tuple[str, ...] | None = None
) -> pd.DataFrame:
    """Build the minimal player x gameweek dataset: identifiers + Q(x) + y + residual + provenance.

    Walks every backtest step in chronological order. DGW gameweeks are skipped (recorded in
    the returned DataFrame's `skip_log` attribute) -- consistent with the M3/M7 v1 scope
    boundary. The result is leakage-checked before return; a LeakageError aborts the build.
    """
    steps = list_backtest_steps(con)
    if not steps:
        raise RuntimeError(
            "no backtest_gameweek_steps with an ep_model_version found -- the DuckDB has no "
            "ingested Quant backtest yet. Run scripts/run_ingestion.py + scripts/run_backtest.py "
            "first, or test via the synthetic fixtures in tests/test_ml_dataset_builder.py."
        )

    skip_log: list[dict] = []
    frames: list[pd.DataFrame] = []

    for step in _iter_steps(steps, seasons):
        season = step["season"]
        gw = int(step["gameweek"])
        ep_mv = int(step["ep_model_version"])
        mm_mv = step["mm_model_version"]
        data_asof = step["data_asof"]

        if bt.has_double_gameweek(con, season, gw):
            skip_log.append({"season": season, "gameweek": gw, "reason": "double_gameweek"})
            continue

        deadline = bt.gameweek_deadline(con, season, gw)
        if deadline is None:
            skip_log.append({"season": season, "gameweek": gw, "reason": "no_deadline"})
            continue

        # ---- asof window: shadow fact tables to strictly-pre-deadline rows ----
        with bt.asof_scope(con, season, gw):
            quant = _quant_predictions_for_step(con, ep_mv)
            if quant.empty:
                skip_log.append({"season": season, "gameweek": gw, "reason": "no_ep_outputs"})
                continue
            quant = _team_and_fixture_context(con, quant, season)
            player_uids = quant["player_uid"].tolist()

        # ---- label: realised outcome, fetched from main.* post-asof ----
        labels = _labels_for_step(con, season, gw, player_uids)
        if labels.empty:
            skip_log.append({"season": season, "gameweek": gw, "reason": "no_realised_outcomes"})
            continue

        row = quant.merge(labels, on="player_uid", how="inner")
        if row.empty:
            continue

        row[C.COL_SEASON] = season
        row[C.COL_GAMEWEEK] = gw
        row[C.COL_EP_MODEL_VERSION] = ep_mv
        row[C.COL_PRED_TIMESTAMP] = pd.Timestamp(data_asof) if data_asof is not None else pd.Timestamp(deadline)
        row[C.COL_RESIDUAL] = row[C.COL_ACTUAL] - row[C.COL_QUANT_PRED]
        # Stash mm_model_version for the minutes-feature pass (M2 probabilities are a permitted
        # asof prediction, not a realised outcome).
        row["_mm_model_version"] = int(mm_mv) if mm_mv is not None else pd.NA
        frames.append(row)

    if not frames:
        raise RuntimeError(
            "dataset build produced zero observations across all backtest steps -- check that "
            "realised outcomes (fact_player_season_stats.event_points) were ingested for the "
            "backtested gameweeks."
        )

    df = pd.concat(frames, ignore_index=True)
    # Provenance + asof invariants are enforced before any feature work or training.
    assert_minimal_dataset_invariants(df, con, skip_log)

    df.attrs["skip_log"] = skip_log
    df.attrs["n_steps_used"] = len(frames)
    df.attrs["seasons"] = sorted(df[C.COL_SEASON].unique().tolist(), key=C.season_sort_key)
    return df


def build_dataset(
    con: duckdb.DuckDBPyConnection,
    seasons: tuple[str, ...] | None = None,
    with_features: bool = True,
) -> pd.DataFrame:
    """Build the full player x gw dataset (minimal + features). Saves parquet + csv.

    Parquet is written via DuckDB (built-in, no pyarrow dependency); CSV via pandas. This is
    the only write the ML layer performs -- it writes to research/ml/data/, never to the
    production schema.
    """
    df = build_minimal_dataset(con, seasons=seasons)
    if with_features:
        from . import feature_engineering as fe  # late import: avoids circular import

        df = fe.add_features(con, df)

    _save_dataset(df)
    return df


def _save_dataset(df: pd.DataFrame) -> None:
    df.to_csv(C.DATASET_CSV, index=False)
    # DuckDB writes parquet natively; register the frame and COPY out. No pyarrow needed.
    con = duckdb.connect()
    try:
        con.register("_ml_dataset", df)
        con.execute(f"COPY _ml_dataset TO '{C.DATASET_PARQUET}' (FORMAT PARQUET)")
    finally:
        con.close()
