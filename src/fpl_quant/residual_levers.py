"""The residual pipeline behind the residual-fitted levers of docs/plans/2026-10_beat_the_template.md
(L1 crowd prior, L2 band bias, L3 per-player residual).

feature_rows() is the one place crowd features are read, in every harness (walk-forward, season
sim, live), so a feature means the same thing wherever a lever applies it (Req 11). Features
come from row gw = G - lag of fact_player_season_stats -- lag 1 by default, the latest completed
gameweek before G's deadline (the plan's A2 decides whether that row is knowable in time; its
fallback is lag 2).

residual_rows() is what `run_walkforward.py --export-residuals` writes as residual_features.csv
(Req 12): one row per player-gameweek a walk-forward step projected and FPL scored, with the
realized points, the projection before any promoted lever's adjustment, and the features. The
lever fits (scripts/fit_residual_lever.py) read that file. The pre-adjustment projection is
ep_total minus every lever's share (ep_crowd_adj for L1, ep_band_adj for L2, ep_player_adj for L3,
each zero when off), so a lever is always fitted to the model's own error; crowd_adj, band_adj and
player_adj carry the adjustments the run applied.
"""

import json
import math
from pathlib import Path

import duckdb

PL = "Premier League"
FEATURES = ("own_pct", "log_sbp", "xfer_pct")
PRICE_BANDS = (("<5.0", 5.0), ("5.0-7.0", 7.0), ("7.0-9.0", 9.0), ("9.0+", math.inf))
RESIDUAL_COLUMNS = (
    "season", "gw", "player_uid", "position", "price", "price_band", "realized_points", "ep_pre_adjustment",
    *FEATURES, "crowd_adj", "band_adj", "player_adj",
)


def price_band(price: float | None) -> str | None:
    if price is None:
        return None
    return next(band for band, upper in PRICE_BANDS if price < upper)


def feature_rows(con: duckdb.DuckDBPyConnection, season: str, gameweek: int, lag: int = 1) -> dict[str, dict]:
    """{player_uid: {"own_pct", "log_sbp", "xfer_pct"}} from row gw = gameweek - lag of `season`.

    - own_pct: the player's ownership percentile within his position (percent_rank, 0 to 1),
      over that row's players with an ownership figure;
    - log_sbp: log(1 + selected_by_percent);
    - xfer_pct: the percentile of transfers_in_event - transfers_out_event among every player in
      that row with both figures, or None when he lacks one.

    A player with no row at gw = gameweek - lag, or no ownership in it, is absent (no feature row
    means no adjustment). Empty for gameweek - lag < 1. Reads fact_player_season_stats by bare
    name, so inside asof_scope() for G it sees rows before G only."""
    source_gw = gameweek - lag
    if source_gw < 1:
        return {}
    rows = con.execute(
        """
        WITH r AS (
            SELECT s.player_uid, dp.position, s.selected_by_percent AS sbp,
                   s.transfers_in_event - s.transfers_out_event AS net_xfer
            FROM fact_player_season_stats s JOIN dim_player dp ON dp.player_uid = s.player_uid
            WHERE s.season = ? AND s.gw = ?
        ),
        x AS (
            -- among every player in the row with both transfer figures, owned or not
            SELECT player_uid, percent_rank() OVER (ORDER BY net_xfer) AS xfer_pct
            FROM r WHERE net_xfer IS NOT NULL
        ),
        owned AS (SELECT * FROM r WHERE sbp IS NOT NULL)
        SELECT owned.player_uid, percent_rank() OVER (PARTITION BY owned.position ORDER BY owned.sbp) AS own_pct,
               ln(1 + greatest(owned.sbp, 0)) AS log_sbp, x.xfer_pct
        FROM owned LEFT JOIN x USING (player_uid)
        """,
        [season, source_gw],
    ).fetchall()
    return {uid: {"own_pct": own, "log_sbp": log_sbp, "xfer_pct": xfer} for uid, own, log_sbp, xfer in rows}


def residual_rows(con: duckdb.DuckDBPyConnection, backtest_run_id: int, lag: int = 1) -> list[dict]:
    """Req 12's rows for one walk-forward run (RESIDUAL_COLUMNS): every player-gameweek with a
    projection at the step (ep_gameweek_outputs, so a double gameweek is summed and a blank one
    drops out) and a realized FPL score. price is the latest before the deadline, as the
    optimizer saw it; features are feature_rows() for that step. Ordered by season, gw, player."""
    steps = con.execute(
        "SELECT season, gameweek, ep_model_version FROM backtest_gameweek_steps "
        "WHERE backtest_run_id = ? AND ep_model_version IS NOT NULL ORDER BY season, gameweek",
        [backtest_run_id],
    ).fetchall()
    out = []
    for season, gw, ep_mv in steps:
        features = feature_rows(con, season, gw, lag=lag)
        rows = con.execute(
            """
            WITH price AS (
                SELECT player_uid, now_cost FROM fact_player_season_stats
                WHERE season = ? AND gw < ? AND now_cost IS NOT NULL
                QUALIFY row_number() OVER (PARTITION BY player_uid ORDER BY gw DESC) = 1
            )
            SELECT o.player_uid, dp.position, price.now_cost, f.event_points,
                   o.ep_total - o.ep_crowd_adj - o.ep_band_adj - o.ep_player_adj,
                   o.ep_crowd_adj, o.ep_band_adj, o.ep_player_adj
            FROM ep_gameweek_outputs o
            JOIN dim_player dp ON dp.player_uid = o.player_uid
            JOIN fact_player_season_stats f ON f.player_uid = o.player_uid AND f.season = ? AND f.gw = ?
            LEFT JOIN price ON price.player_uid = o.player_uid
            WHERE o.model_version = ? AND f.event_points IS NOT NULL
            ORDER BY o.player_uid
            """,
            [season, gw, season, gw, ep_mv],
        ).fetchall()
        for uid, position, price, realized, ep_pre, crowd_adj, band_adj, player_adj in rows:
            feat = features.get(uid, {})
            out.append({
                "season": season, "gw": gw, "player_uid": uid, "position": position, "price": price,
                "price_band": price_band(price), "realized_points": realized, "ep_pre_adjustment": ep_pre,
                **{name: feat.get(name) for name in FEATURES}, "crowd_adj": crowd_adj, "band_adj": band_adj,
                "player_adj": player_adj,
            })
    return out


def write_residual_csv(con: duckdb.DuckDBPyConnection, backtest_run_id: int, path, lag: int = 1) -> int:
    """Writes residual_rows() as CSV (empty cells for missing values); returns the row count."""
    import csv

    rows = residual_rows(con, backtest_run_id, lag=lag)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(RESIDUAL_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
    return len(rows)


# ============================================================
# L1, the crowd prior (docs/plans/2026-10_beat_the_template.md): a per-gameweek additive EP
# adjustment from a ridge regression of residuals (realized - pre-adjustment EP) on FEATURES,
# fitted per fold and applied by crowd_adjustments(), the one function every harness calls
# (through expected_points.run(), which floors an adjusted projection at zero).
# ============================================================

CROWD_PRIOR_FAMILY = "crowd_prior_params"
CROWD_PRIOR_KEYS = ("coeff_date", "coeff_hash", "ridge_alpha", "min_rows", "clip", "lag")
COEFF_DIR_NAME = "recalibration"
# A2 (docs/reports/2026-10_beat_the_template.md) was inconclusive: row G-1 isn't reliably settled
# by G's deadline, so a lever's crowd features read row G-2 or earlier, never G-1.
MIN_FEATURE_LAG = 2


def fold_for(season: str) -> str:
    """The coefficient fold that scores or replays `season`: each historical season uses the
    other one's fit, so no season is scored with coefficients fitted on its own residuals; later
    seasons (2026-27, live) use the both-season fit."""
    if season == "2024-2025":
        return "2025-2026"
    if season == "2025-2026":
        return "2024-2025"
    return "both"


def fit_ridge(rows: list[dict], features: tuple[str, ...] = FEATURES, alpha: float = 10.0,
              min_rows: int = 2000) -> dict | None:
    """Ridge on standardized features, no intercept: y = realized - ep_pre_adjustment. Standardizing
    centres each feature on the fold, so the fitted adjustment averages zero over it -- a mean
    shift by position and band is L2's job, not L1's. None when fewer than min_rows rows carry every
    feature."""
    import numpy as np

    usable = [r for r in rows if all(r.get(f) not in (None, "") for f in features)
              and r.get("realized_points") not in (None, "") and r.get("ep_pre_adjustment") not in (None, "")]
    if len(usable) < min_rows:
        return None
    x = np.array([[float(r[f]) for f in features] for r in usable])
    y = np.array([float(r["realized_points"]) - float(r["ep_pre_adjustment"]) for r in usable])
    means, sds = x.mean(axis=0), x.std(axis=0)
    sds[sds == 0] = 1.0
    z = (x - means) / sds
    beta = np.linalg.solve(z.T @ z + alpha * np.eye(len(features)), z.T @ (y - y.mean()))
    return {"n": len(usable), "means": means.round(8).tolist(), "sds": sds.round(8).tolist(),
            "beta": beta.round(8).tolist(), "resid_mean": round(float(y.mean()), 6)}


def fit_crowd_prior(rows: list[dict], alpha: float = 10.0, min_rows: int = 2000,
                    features: tuple[str, ...] = FEATURES) -> dict:
    """{fold: fit | None} for the folds fold_for() names: each season's own rows, and both."""
    by_season = {s: [r for r in rows if r.get("season") == s] for s in ("2024-2025", "2025-2026")}
    return {
        "2024-2025": fit_ridge(by_season["2024-2025"], features, alpha, min_rows),
        "2025-2026": fit_ridge(by_season["2025-2026"], features, alpha, min_rows),
        "both": fit_ridge(by_season["2024-2025"] + by_season["2025-2026"], features, alpha, min_rows),
    }


def adjustment(fit: dict, values: dict, features: tuple[str, ...], clip: float) -> float | None:
    """One player's adjustment from a fold's fit, clipped to +-clip; None without every feature."""
    if any(values.get(f) is None for f in features):
        return None
    total = sum(b * (float(values[f]) - m) / s for f, b, m, s in zip(features, fit["beta"], fit["means"], fit["sds"]))
    return max(-clip, min(clip, total))


def crowd_adjustments(con: duckdb.DuckDBPyConnection, season: str, deadline_gameweek: int, coeffs: dict,
                      clip: float, lag: int) -> dict[str, float]:
    """{player_uid: adjustment, pts/GW} for the gameweek being planned, from the fold that scores
    `season` and feature_rows() at that deadline minus `lag`. Empty when the fold wasn't fitted
    (fewer than min_rows); a player without a full feature row gets no entry (no adjustment).
    Every planning-horizon gameweek of one deadline gets the same adjustment: the features are
    what was knowable at that deadline."""
    if lag < MIN_FEATURE_LAG:
        raise ValueError(f"crowd features at lag {lag}: A2 failed, so they read row G-{MIN_FEATURE_LAG} or earlier")
    folds = coeffs.get("folds") or {}
    if fold_for(season) not in folds:
        raise KeyError(f"coefficient file has no {fold_for(season)!r} fold for {season}")
    fit = folds[fold_for(season)]
    if fit is None:
        print(f"[crowd_prior] {season} GW{deadline_gameweek}: the {fold_for(season)} fold was not fitted "
              "(fewer than min_rows), no adjustment")
        return {}
    features = tuple(coeffs.get("features") or FEATURES)
    out = {}
    for uid, values in feature_rows(con, season, deadline_gameweek, lag=lag).items():
        adj = adjustment(fit, values, features, clip)
        if adj is not None:
            out[uid] = adj
    return out


COEFF_ROOT = Path(__file__).resolve().parents[2] / "data" / COEFF_DIR_NAME


def coeff_path(coeff_date: int, lever: str = "crowd_prior") -> Path:
    """data/recalibration/<lever>_coeffs_<YYYYMMDD>.json, written by scripts/fit_residual_lever.py
    (lever crowd_prior for L1, band_bias for L2)."""
    return COEFF_ROOT / f"{lever}_coeffs_{int(coeff_date)}.json"


def coeff_hash(coeff_date: int, lever: str = "crowd_prior") -> int:
    """The coefficient file's content hash: the first 48 bits of its sha256, exact as a double."""
    import hashlib

    return int(hashlib.sha256(coeff_path(coeff_date, lever).read_bytes()).hexdigest()[:12], 16)


def crowd_prior_bundle(coeff_date: int) -> dict[str, float]:
    """The crowd_prior_params values for one coefficient file: its date and content hash, and the
    alpha, min_rows, clip and feature lag it was fitted and is applied with (read from the file
    itself, so the param version and the coefficients can't disagree)."""
    coeffs = json.loads(coeff_path(coeff_date).read_text())
    return {"coeff_date": int(coeff_date), "coeff_hash": coeff_hash(coeff_date), "ridge_alpha": coeffs["alpha"],
            "min_rows": coeffs["min_rows"], "clip": coeffs["clip"], "lag": coeffs["lag"]}


def crowd_prior_version(con: duckdb.DuckDBPyConnection, coeff_date: int, effective_date: str) -> int:
    """The crowd_prior_params version for this coefficient file (an existing one if it matches)."""
    from . import params as params_mod

    return params_mod.get_or_create_bundle_version(con, CROWD_PRIOR_FAMILY, crowd_prior_bundle(coeff_date), effective_date)


def resolve_crowd_prior(con: duckdb.DuckDBPyConnection, version: int | None) -> dict | None:
    """The crowd_prior_params bundle with its coefficient file loaded, or None when off."""
    from . import params as params_mod

    if version is None:
        return None
    values = {k: params_mod.resolve_param(con, CROWD_PRIOR_FAMILY, k, version)[0] for k in CROWD_PRIOR_KEYS}
    if coeff_hash(values["coeff_date"]) != int(values["coeff_hash"]):
        raise ValueError(f"{coeff_path(values['coeff_date'])} changed since crowd_prior_params v{version} was minted")
    coeffs = json.loads(coeff_path(values["coeff_date"]).read_text())
    return {"coeffs": coeffs, "clip": float(values["clip"]), "lag": int(values["lag"])}


# ============================================================
# L2, the band bias (docs/plans/2026-10_beat_the_template.md, Phase 6): each (position, price band)
# cell's mean residual (realized - the projection before any lever) in the fold that scores the
# season, shrunk toward zero by n / (n + pseudo_rows). Applied by band_adjustments() through
# expected_points.run(), the same way as L1: split across a double gameweek by EP, never taking a
# fixture's EP below zero, every planning-horizon gameweek of one deadline the same.
# ============================================================

BAND_BIAS_FAMILY = "band_bias_params"
BAND_BIAS_KEYS = ("coeff_date", "coeff_hash", "pseudo_rows")
BAND_BIAS_REQUIRED = ("realized_points", "ep_pre_adjustment", "position", "price_band")


def band_key(position: str, band: str) -> str:
    return f"{position}|{band}"


def fit_band_cells(rows: list[dict], pseudo_rows: float = 500.0) -> dict[str, dict]:
    """{"position|band": {"n", "mean_resid", "adj"}} over the rows that carry every BAND_BIAS_REQUIRED
    field: adj = mean_resid * n / (n + pseudo_rows)."""
    acc: dict[str, list[float]] = {}
    for r in rows:
        if any(r.get(k) in (None, "") for k in BAND_BIAS_REQUIRED):
            continue
        cell = acc.setdefault(band_key(r["position"], r["price_band"]), [0, 0.0])
        cell[0] += 1
        cell[1] += float(r["realized_points"]) - float(r["ep_pre_adjustment"])
    out = {}
    for key, (n, total) in sorted(acc.items()):
        mean = total / n
        out[key] = {"n": int(n), "mean_resid": round(mean, 6), "adj": round(mean * n / (n + pseudo_rows), 6)}
    return out


def fit_band_bias(rows: list[dict], pseudo_rows: float = 500.0) -> dict:
    """{fold: cells} for the folds fold_for() names: each season's own rows, and both."""
    by_season = {s: [r for r in rows if r.get("season") == s] for s in ("2024-2025", "2025-2026")}
    return {
        "2024-2025": fit_band_cells(by_season["2024-2025"], pseudo_rows),
        "2025-2026": fit_band_cells(by_season["2025-2026"], pseudo_rows),
        "both": fit_band_cells(by_season["2024-2025"] + by_season["2025-2026"], pseudo_rows),
    }


def band_adjustments(con: duckdb.DuckDBPyConnection, season: str, deadline_gameweek: int,
                     coeffs: dict) -> dict[str, float]:
    """{player_uid: adjustment, pts/GW} for the gameweek being planned: the cell of the player's
    position and price band, from the fold that scores `season`. The price is his latest before the
    deadline (the price the optimizer sees, and the one residual_rows() bands by). A player with no
    price yet, or in a cell the fold has no rows for, gets no entry (no adjustment)."""
    folds = coeffs.get("folds") or {}
    if fold_for(season) not in folds:
        raise KeyError(f"coefficient file has no {fold_for(season)!r} fold for {season}")
    cells = folds[fold_for(season)] or {}
    rows = con.execute(
        """
        WITH price AS (
            SELECT player_uid, now_cost FROM fact_player_season_stats
            WHERE season = ? AND gw < ? AND now_cost IS NOT NULL
            QUALIFY row_number() OVER (PARTITION BY player_uid ORDER BY gw DESC) = 1
        )
        SELECT price.player_uid, dp.position, price.now_cost
        FROM price JOIN dim_player dp ON dp.player_uid = price.player_uid
        """,
        [season, deadline_gameweek],
    ).fetchall()
    out = {}
    for uid, position, price in rows:
        cell = cells.get(band_key(position, price_band(price)))
        if cell is not None:
            out[uid] = float(cell["adj"])
    return out


def band_bias_bundle(coeff_date: int) -> dict[str, float]:
    """The band_bias_params values for one coefficient file: its date, content hash and pseudo_rows."""
    coeffs = json.loads(coeff_path(coeff_date, "band_bias").read_text())
    return {"coeff_date": int(coeff_date), "coeff_hash": coeff_hash(coeff_date, "band_bias"),
            "pseudo_rows": coeffs["pseudo_rows"]}


def band_bias_version(con: duckdb.DuckDBPyConnection, coeff_date: int, effective_date: str) -> int:
    """The band_bias_params version for this coefficient file (an existing one if it matches)."""
    from . import params as params_mod

    return params_mod.get_or_create_bundle_version(con, BAND_BIAS_FAMILY, band_bias_bundle(coeff_date), effective_date)


def resolve_band_bias(con: duckdb.DuckDBPyConnection, version: int | None) -> dict | None:
    """The band_bias_params bundle with its coefficient file loaded, or None when off."""
    from . import params as params_mod

    if version is None:
        return None
    values = {k: params_mod.resolve_param(con, BAND_BIAS_FAMILY, k, version)[0] for k in BAND_BIAS_KEYS}
    if coeff_hash(values["coeff_date"], "band_bias") != int(values["coeff_hash"]):
        raise ValueError(f"{coeff_path(values['coeff_date'], 'band_bias')} changed since band_bias_params "
                         f"v{version} was minted")
    return {"coeffs": json.loads(coeff_path(values["coeff_date"], "band_bias").read_text())}



# ============================================================
# L3, the per-player residual (docs/plans/2026-10_beat_the_template.md, Phase 7): an eligible
# player's own mean residual over the gameweeks just before the deadline, realized - the projection
# made at each of those gameweeks' own deadlines (before L3), shrunk toward zero by
# n / (n + pseudo_gws). No fit and no coefficient file: the window, shrinkage, eligibility and lag
# are the plan's declared values. Applied through expected_points.run() like L1 and L2.
#
# The history is the caller's: each harness passes the projection it made at each earlier deadline
# (projection_history() reads one from the EP model versions that projected them), because only the
# harness knows which run is its own -- the walk-forward's earlier steps, the season sim's earlier
# plan runs, live's committed deadline-time snapshots.
# ============================================================

PLAYER_RESIDUAL_FAMILY = "player_residual_params"
PLAYER_RESIDUAL_KEYS = ("window", "pseudo_gws", "min_minutes", "lag")
# the plan's declared L3 values; lag 2 because A2 failed, so row G-1 (where a gameweek's realized
# points live) is never read before G's deadline
PLAYER_RESIDUAL_V1 = {"window": 6, "pseudo_gws": 6, "min_minutes": 900, "lag": 2}
# GW1 never enters a window: neither the season sim (it starts at GW2) nor live (no snapshot before
# 2026-27's GW1 deadline) has a deadline-time GW1 projection, so the walk-forward doesn't use one
FIRST_HISTORY_GAMEWEEK = 2


def player_residual_window(deadline_gameweek: int, window: int, lag: int) -> list[int]:
    """The gameweeks whose residuals the deadline's adjustment averages: the `window` gameweeks
    ending at deadline - lag, none before FIRST_HISTORY_GAMEWEEK."""
    last = deadline_gameweek - lag
    return list(range(max(FIRST_HISTORY_GAMEWEEK, last - window + 1), last + 1))


def projection_history(con: duckdb.DuckDBPyConnection, ep_versions: dict[int, int]) -> dict[int, dict[str, float]]:
    """{gameweek: {player_uid: the projection before L3}} from {gameweek: the EP model version that
    projected it at its deadline}: ep_gameweek_outputs.ep_total - ep_player_adj (a double gameweek
    summed, a blank one absent)."""
    out = {}
    for gw, mv in sorted(ep_versions.items()):
        out[int(gw)] = {uid: float(ep) for uid, ep in con.execute(
            "SELECT player_uid, ep_total - ep_player_adj FROM ep_gameweek_outputs WHERE model_version = ?", [mv],
        ).fetchall()}
    return out


def _previous_season(season: str) -> str:
    start = int(season[:4])
    return f"{start - 1}-{start}"


def player_residual_adjustments(con: duckdb.DuckDBPyConnection, season: str, deadline_gameweek: int,
                                history: dict[int, dict[str, float]], params: dict) -> dict[str, float]:
    """{player_uid: adjustment, pts/GW} for the gameweek being planned. Over the window's gameweeks
    the history projected, a player's residual is his FPL score that gameweek minus his projection;
    adj = mean residual * n / (n + pseudo_gws), n the gameweeks with both. Only players with at least
    min_minutes Premier League minutes in the previous season plus this one up to deadline - lag
    are adjusted. A player with no such gameweek gets no entry (no adjustment)."""
    window, pseudo, lag = int(params["window"]), float(params["pseudo_gws"]), int(params["lag"])
    if lag < MIN_FEATURE_LAG:
        raise ValueError(f"player residual lag {lag} < {MIN_FEATURE_LAG}: row G-1 is not knowable at G's deadline (A2)")
    gws = [gw for gw in player_residual_window(deadline_gameweek, window, lag) if gw in history]
    if not gws:
        return {}
    eligible = {uid for uid, mins in con.execute(
        """
        SELECT s.player_uid, sum(coalesce(s.minutes_played, 0))
        FROM fact_player_match_stats s JOIN fact_match m ON m.match_id = s.match_id
        WHERE m.competition = ? AND (m.season = ? OR (m.season = ? AND m.gameweek <= ?))
        GROUP BY s.player_uid
        """,
        [PL, _previous_season(season), season, deadline_gameweek - lag],
    ).fetchall() if mins >= float(params["min_minutes"])}
    ph = ",".join("?" * len(gws))
    realized = con.execute(
        f"SELECT player_uid, gw, event_points FROM fact_player_season_stats "
        f"WHERE season = ? AND gw IN ({ph}) AND event_points IS NOT NULL",
        [season, *gws],
    ).fetchall()
    acc: dict[str, list[float]] = {}
    for uid, gw, points in realized:
        projected = history[gw].get(uid)
        if uid in eligible and projected is not None:
            cell = acc.setdefault(uid, [0, 0.0])
            cell[0] += 1
            cell[1] += float(points) - projected
    # the mean residual (total / n) shrunk by n / (n + pseudo_gws)
    return {uid: total / (n + pseudo) for uid, (n, total) in acc.items()}


def player_residual_version(con: duckdb.DuckDBPyConnection, effective_date: str,
                            values: dict | None = None) -> int:
    """The player_residual_params version for these values (the declared PLAYER_RESIDUAL_V1 by
    default; an existing version if it matches)."""
    from . import params as params_mod

    return params_mod.get_or_create_bundle_version(
        con, PLAYER_RESIDUAL_FAMILY, dict(values or PLAYER_RESIDUAL_V1), effective_date,
    )


def resolve_player_residual(con: duckdb.DuckDBPyConnection, version: int | None) -> dict | None:
    """The player_residual_params bundle, or None when off."""
    from . import params as params_mod

    if version is None:
        return None
    return {k: params_mod.resolve_param(con, PLAYER_RESIDUAL_FAMILY, k, version)[0] for k in PLAYER_RESIDUAL_KEYS}


def history_for_deadline(con: duckdb.DuckDBPyConnection, ep_versions: dict[int, int], deadline_gameweek: int,
                         params: dict) -> dict[int, dict[str, float]]:
    """L3's history for one deadline from a harness's {gameweek: the EP model version that projected
    it at its own deadline}: projection_history() over the gameweeks in the deadline's window."""
    gws = set(player_residual_window(deadline_gameweek, int(params["window"]), int(params["lag"])))
    return projection_history(con, {gw: mv for gw, mv in ep_versions.items() if gw in gws})


def walk_forward_ep_versions(con: duckdb.DuckDBPyConnection, backtest_run_id: int, season: str) -> dict[int, int]:
    """{gameweek: ep_model_version} of this walk-forward run's steps so far in `season`: each step
    projects its own gameweek at its own deadline, so these are the walk-forward's L3 history."""
    return dict(con.execute(
        "SELECT gameweek, ep_model_version FROM backtest_gameweek_steps "
        "WHERE backtest_run_id = ? AND season = ? AND ep_model_version IS NOT NULL",
        [backtest_run_id, season],
    ).fetchall())
