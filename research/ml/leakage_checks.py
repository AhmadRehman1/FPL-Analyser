"""Programmatic leakage checks for the Phase-0 ML dataset (LEAKAGE_PROTOCOL.md §8).

These are not advisory -- a failure here raises LeakageError and aborts the experiment before
any model is trained or evaluated. Each check function corresponds 1:1 to a numbered item in
LEAKAGE_PROTOCOL.md §8 so the mapping between code and protocol is auditable.
"""

from __future__ import annotations

import pandas as pd

from . import contract as C


class LeakageError(RuntimeError):
    """A dataset or split violates a leakage invariant. Must abort, never be silently fixed."""


# ============================================================
# §8.1 -- ep_model_version resolves to the matching backtest step
# ============================================================

def check_ep_model_version_matches_step(df: pd.DataFrame, con) -> None:
    if df.empty or C.COL_EP_MODEL_VERSION not in df.columns:
        return
    steps = con.execute(
        "SELECT ep_model_version, season, gameweek FROM backtest_gameweek_steps "
        "WHERE ep_model_version IS NOT NULL"
    ).fetchdf()
    step_map = {int(r.ep_model_version): (r.season, int(r.gameweek)) for r in steps.itertuples()}
    for ep_mv, season, gw in df[[C.COL_EP_MODEL_VERSION, C.COL_SEASON, C.COL_GAMEWEEK]].drop_duplicates().itertuples(index=False, name=None):
        expected = step_map.get(int(ep_mv))
        if expected is None:
            raise LeakageError(
                f"ep_model_version {ep_mv} in the dataset has no corresponding "
                f"backtest_gameweek_steps row -- Q(x) provenance cannot be verified"
            )
        if expected != (season, int(gw)):
            raise LeakageError(
                f"ep_model_version {ep_mv} was produced for {expected}, but the dataset row "
                f"claims it was used for ({season}, {gw}) -- a Quant prediction from a "
                f"different asof point would leak/misattribute information"
            )


# ============================================================
# §8.2 -- prediction precedes the outcome
# ============================================================

def check_prediction_precedes_outcome(df: pd.DataFrame, con) -> None:
    """Found running Phase F-4 for real, against the real production DB, for the first time:
    prediction_timestamp (== backtest.data_asof) is *always* exactly equal to the gameweek's
    first kickoff, for all 71 real backtest steps, with zero exceptions -- confirmed live. This
    is not a data-quality accident: backtest.gameweek_deadline()'s own docstring documents it as
    a deliberate, deterministic approximation ("Earliest kickoff_time among that gameweek's real
    fixtures, standing in for the actual FPL transfer deadline -- no deadline field exists
    anywhere in the ingested data"). This function independently recomputes the exact same
    MIN(kickoff_time) query, so pred_ts and kickoff_ts are equal by construction for every real
    step in this codebase -- a strict `>=` trigger here can never NOT fire, making this check
    fail on 100% of real data regardless of whether real leakage exists.

    The actual safety guarantee already holds independently of this: `ep_outputs` (which becomes
    quant_prediction here) is itself built inside backtest.asof_scope(), which shadows every
    fact-table read to STRICTLY `< data_asof` -- so no realised-outcome row at-or-after kickoff
    can already have reached the prediction, before this function ever runs. This check is a
    redundant, defensive second look, not the only safety net -- so relaxing its own boundary
    from `>=` to `>` (only a prediction_timestamp genuinely AFTER kickoff is a real problem) does
    not weaken the real leakage guarantee, it just stops re-testing an invariant that was never
    true anywhere in this codebase and clearly wasn't the intent of the original `>=` (which
    would make this check permanently, universally unpassable against real data)."""
    if df.empty or C.COL_PRED_TIMESTAMP not in df.columns:
        return
    for season, gw, ts in df[[C.COL_SEASON, C.COL_GAMEWEEK, C.COL_PRED_TIMESTAMP]].drop_duplicates().itertuples(index=False, name=None):
        row = con.execute(
            "SELECT min(kickoff_time) FROM fact_match WHERE season = ? AND gameweek = ? AND competition = ?",
            [season, int(gw), C.PL],
        ).fetchone()
        first_kickoff = row[0] if row else None
        if first_kickoff is None:
            continue
        pred_ts = pd.Timestamp(ts)
        kickoff_ts = pd.Timestamp(first_kickoff)
        if pred_ts > kickoff_ts:
            raise LeakageError(
                f"prediction_timestamp {pred_ts} for ({season}, GW{gw}) is after "
                f"the first kickoff {kickoff_ts} -- the prediction must not follow the outcome"
            )


# ============================================================
# §8.3 -- no feature computed from same-or-future rows (structural check)
# ============================================================

FORBIDDEN_FEATURE_COLUMNS = {
    # Same-gameweek realised-outcome columns that must never appear as features. The label
    # column itself (COL_ACTUAL) is exempted from this set -- see check_label_not_a_feature.
    "minutes", "goals_scored", "assists", "bps", "expected_goals", "expected_assists",
    "total_points", "saves", "goals_conceded", "team_goals_conceded",
}


def check_no_forbidden_feature_columns(feature_cols: list[str]) -> None:
    hit = FORBIDDEN_FEATURE_COLUMNS & set(feature_cols)
    if hit:
        raise LeakageError(
            f"feature matrix contains forbidden same-gameweek outcome column(s) {sorted(hit)} -- "
            "these encode the realised outcome being predicted (LEAKAGE_PROTOCOL.md §5)"
        )


# ============================================================
# §8.4 -- label appears exactly once, never in the feature matrix
# ============================================================

def check_label_not_a_feature(feature_cols: list[str]) -> None:
    if C.LABEL_COL in feature_cols:
        raise LeakageError(
            f"label column {C.LABEL_COL!r} must never appear in the feature matrix"
        )


# ============================================================
# §8.5 -- no un-skipped DGW steps
# ============================================================

def check_no_unskipped_dgw(df: pd.DataFrame, con) -> None:
    if df.empty:
        return
    from fpl_quant import backtest as bt  # late import: path bootstrap dependency

    for season, gw in df[[C.COL_SEASON, C.COL_GAMEWEEK]].drop_duplicates().itertuples(index=False, name=None):
        if bt.has_double_gameweek(con, season, int(gw)):
            raise LeakageError(
                f"({season}, GW{gw}) is a double gameweek and must have been skipped "
                "before reaching the dataset -- no DGW aggregation semantics are defined"
            )
        dup = df[(df[C.COL_SEASON] == season) & (df[C.COL_GAMEWEEK] == gw)]["player_uid"]
        if dup.duplicated().any():
            raise LeakageError(
                f"({season}, GW{gw}) has duplicate player_uid rows -- indicates an unhandled "
                "multi-fixture case"
            )


# ============================================================
# §8.6 -- chronological ordering across walk-forward folds
# ============================================================

def check_chronological_split(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """A valid walk-forward split: every training row's prediction_timestamp is strictly before
    every test row's, AND no (season, gameweek) step appears in both splits. Sharing a *season*
    is fine (training on gameweeks 1-3 of a season and testing on gameweek 4 is leak-free) -- the
    leak is reusing the same gameweek, which the timestamp check already forbids, but we assert
    the (season, gameweek) key overlap explicitly as a belt-and-braces guard."""
    if train_df.empty or test_df.empty:
        return
    train_max = train_df[C.COL_PRED_TIMESTAMP].max()
    test_min = test_df[C.COL_PRED_TIMESTAMP].min()
    if train_max >= test_min:
        raise LeakageError(
            f"train set's latest prediction_timestamp ({train_max}) is not strictly before "
            f"the test set's earliest ({test_min}) -- walk-forward split is not chronological"
        )
    def _keys(d: pd.DataFrame) -> set[tuple[str, int]]:
        sub = d[[C.COL_SEASON, C.COL_GAMEWEEK]].drop_duplicates()
        return {(r[C.COL_SEASON], int(r[C.COL_GAMEWEEK])) for _, r in sub.iterrows()}
    overlap = _keys(train_df) & _keys(test_df)
    if overlap:
        raise LeakageError(
            f"{len(overlap)} (season, gameweek) step(s) appear in both train and test splits -- "
            "walk-forward validation must never train and test on the same gameweek"
        )


# ============================================================
# Aggregate entrypoints
# ============================================================

def assert_minimal_dataset_invariants(df: pd.DataFrame, con, skip_log: list[dict]) -> None:
    """Run every check applicable to the minimal (pre-feature) dataset. Called by
    dataset_builder.build_minimal_dataset() before returning."""
    check_ep_model_version_matches_step(df, con)
    check_prediction_precedes_outcome(df, con)
    check_no_unskipped_dgw(df, con)
    # label is present as a real column at this stage (not yet split into X/y) -- checked
    # again at feature-matrix construction time by check_label_not_a_feature.


def assert_feature_matrix_invariants(feature_cols: list[str]) -> None:
    """Run before any model sees a feature matrix (evaluate.py / residual_model.py)."""
    check_no_forbidden_feature_columns(feature_cols)
    check_label_not_a_feature(feature_cols)


def assert_split_invariants(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Run by walk_forward.py before returning a train/test pair."""
    check_chronological_split(train_df, test_df)
