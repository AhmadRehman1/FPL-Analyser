"""Phase 1A/1B calibration diagnostics: honest, from-raw-rows recomputation of EP calibration
residuals (price-band/position slices, component decomposition, GW-clustered uncertainty) and
the selection-residual / "optimiser curse" test.

Why this module exists rather than reusing `backtest_metrics` directly: `backtest.score_gameweek()`
already writes segment-level calibration metrics, but each stored value is a MEAN OF PER-STEP
MEANS (one row per gameweek, averaged again across gameweeks by whoever reads it) -- not
sample-size-weighted, and with no uncertainty interval. This module always recomputes from
`ep_outputs` + `fact_player_season_stats` raw player-gameweek rows instead, and never trusts a
pre-aggregated JSON/metrics-table number as ground truth (the two are legitimately different
aggregations of the same underlying data -- see docs/reports/2026-09_calibration_captain_headline_audit.md
for why that discrepancy matters and shouldn't be papered over).

Price-band boundaries are asserted exactly once: `backtest._price_band()` is the single source
of truth (research/ml/baselines.py imports it rather than re-deriving the same four cut points;
scripts/diagnose_ep_calibration.py's SQL-side CASE expressions are a separate, pre-existing
duplication this module does not touch, since consolidating a live nightly-reporting script is
out of scope for a new diagnostics module -- see the docs note above).

Component-residual formulas (r_app/r_goals/r_assists/r_cs/r_other, ep_other) are a deliberate
mirror of the identical inline computation in `backtest.score_gameweek()` (lines ~807-819 as of
this writing) -- not re-derived independently, to avoid the two ever silently disagreeing about
what "the goals component" means. `tests/test_calibration_diagnostics.py`'s reconciliation test
is the guard against drift between the two copies.
"""

from __future__ import annotations

import math
import random
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from . import backtest as bt
from . import expected_points as ep

REPO_ROOT = Path(__file__).resolve().parents[2]

# The five components a step's ep_total/realized_pts residual decomposes into, matching
# backtest.score_gameweek()'s own comp_resid dict keys exactly.
COMPONENTS = ("goals", "assists", "appearance", "cleansheet", "other")


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None


# ============================================================
# Pure statistics helpers -- no DB, fully unit-testable.
# ============================================================

def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _mae(xs: list[float]) -> float | None:
    return sum(abs(x) for x in xs) / len(xs) if xs else None


def _rmse(xs: list[float]) -> float | None:
    return math.sqrt(sum(x * x for x in xs) / len(xs)) if xs else None


def _median(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def cluster_bootstrap_ci(
    rows: list[dict], value_fn, cluster_key_fn, *, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05,
) -> tuple[float, float] | None:
    """95% CI for mean(value_fn(row)) via a cluster bootstrap over cluster_key_fn(row) (normally
    (season, gameweek)) -- resamples whole clusters with replacement rather than individual rows,
    so within-gameweek correlation (players in the same GW share fixture-level shocks) is
    respected instead of treated as independent. Deterministic for a fixed seed. Returns None
    when fewer than 2 clusters exist (not enough structure for a meaningful interval -- reporting
    a fabricated interval from 1 cluster would be worse than reporting nothing)."""
    clusters: dict = {}
    for r in rows:
        clusters.setdefault(cluster_key_fn(r), []).append(value_fn(r))
    cluster_values = list(clusters.values())
    n = len(cluster_values)
    if n < 2:
        return None
    rng = random.Random(seed)
    boot_means = []
    for _ in range(n_boot):
        pooled = [v for idx in (rng.randrange(n) for _ in range(n)) for v in cluster_values[idx]]
        if pooled:
            boot_means.append(sum(pooled) / len(pooled))
    if not boot_means:
        return None
    boot_means.sort()
    lo = int((alpha / 2) * len(boot_means))
    hi = min(int((1 - alpha / 2) * len(boot_means)), len(boot_means) - 1)
    return boot_means[lo], boot_means[hi]


def summarize_residuals(rows: list[dict], *, n_boot: int = 2000, seed: int = 0) -> dict:
    """count, mean/mae/rmse/median residual, GW-clustered 95% CI on the mean. Never fabricates
    a value for an empty slice -- all fields are None when rows is empty, not 0.0."""
    resids = [r["residual"] for r in rows]
    return {
        "n": len(rows),
        "mean_resid": _mean(resids),
        "mae": _mae(resids),
        "rmse": _rmse(resids),
        "median_resid": _median(resids),
        "mean_resid_ci95_gw_clustered": cluster_bootstrap_ci(
            rows, lambda r: r["residual"], lambda r: (r["season"], r["gameweek"]), n_boot=n_boot, seed=seed,
        ) if rows else None,
    }


# ============================================================
# Phase 1A: raw-row fetch + price-band/position calibration report.
# ============================================================

def fetch_calibration_rows(
    con: duckdb.DuckDBPyConnection, backtest_run_id: int, *, tier: str | None = "mature",
    scoring_params_version: int = 1,
) -> tuple[list[dict], dict]:
    """Long-form player-gameweek residual rows for every scored step of `backtest_run_id`
    (default tier='mature' only, matching the report_history convention that mature-tier is
    the honest headline slice -- cold/warm carry known MLE-instability/less-evidence caveats
    documented in the README's M7 section). Pass tier=None for every tier pooled together.

    scoring_params_version defaults to 1 because backtest.run() takes it as a call-time argument
    and never persists it onto backtest_runs -- every existing caller (scripts/run_backtest.py)
    has only ever passed 1, so this default matches real usage rather than guessing; pass it
    explicitly if a run used something else.

    Returns (rows, missing_data_counts). Each row: player_uid, season, gameweek, tier, position,
    price_band, now_cost, realized, predicted, residual, components (name -> (realized, predicted)).
    """
    step_filter = " AND tier = ?" if tier else ""
    params = [backtest_run_id] + ([tier] if tier else [])
    steps = con.execute(
        f"SELECT season, gameweek, tier, ep_model_version FROM backtest_gameweek_steps "
        f"WHERE backtest_run_id = ? AND ep_model_version IS NOT NULL{step_filter}",
        params,
    ).fetchall()

    rows: list[dict] = []
    missing_realized_or_predicted = 0
    missing_price = 0
    missing_position = 0

    for season, gameweek, step_tier, ep_mv in steps:
        ep_rows = con.execute(
            "SELECT o.player_uid, dp.position, o.fixture_match_id, o.ep_clean_sheet, o.ep_goals, "
            "o.ep_assists, o.ep_total, o.ep_appearance "
            "FROM ep_outputs o JOIN dim_player dp ON dp.player_uid = o.player_uid WHERE o.model_version = ?",
            [ep_mv],
        ).fetchall()
        event_points_of = dict(con.execute(
            "SELECT player_uid, event_points FROM fact_player_season_stats "
            "WHERE season = ? AND gw = ? AND event_points IS NOT NULL",
            [season, gameweek],
        ).fetchall())
        price_of = dict(con.execute(
            "SELECT player_uid, now_cost FROM fact_player_season_stats "
            "WHERE season = ? AND gw = ? AND now_cost IS NOT NULL",
            [season, gameweek],
        ).fetchall())

        for player_uid, position, match_id, ep_cs, ep_g, ep_a, ep_total, ep_app in ep_rows:
            realized_pts = event_points_of.get(player_uid)
            if realized_pts is None or ep_total is None:
                missing_realized_or_predicted += 1
                continue
            now_cost = price_of.get(player_uid)
            if now_cost is None:
                missing_price += 1
            if position is None:
                missing_position += 1

            # Mirrors backtest.score_gameweek()'s inline component computation exactly -- see
            # module docstring. `ep_other`/`r_other` are residual-of-total by construction, so
            # the five components always sum back to `realized - predicted` algebraically.
            outcome = bt._realized_player_match_outcome(con, player_uid, match_id)
            cs_pts = ep._sm(con, "clean_sheet_points", scoring_params_version, position)
            goal_pts = ep._sm(con, "goal_points", scoring_params_version, position)
            assist_pts = ep._sm(con, "assist_points", scoring_params_version)
            mins = outcome["minutes_played"]
            r_app = 2.0 if mins >= 60 else (1.0 if mins >= 1 else 0.0)
            r_goals = outcome["goals"] * (goal_pts or 0.0)
            r_assists = outcome["assists"] * (assist_pts or 0.0)
            r_cs = (cs_pts or 0.0) if (outcome["team_goals_conceded"] == 0 and mins >= 60) else 0.0
            r_other = realized_pts - r_app - r_goals - r_assists - r_cs
            ep_other = ep_total - ep_app - ep_g - ep_a - ep_cs

            rows.append({
                "player_uid": player_uid, "season": season, "gameweek": gameweek, "tier": step_tier,
                "position": position, "price_band": bt._price_band(now_cost), "now_cost": now_cost,
                "realized": realized_pts, "predicted": ep_total, "residual": realized_pts - ep_total,
                "components": {
                    "goals": (r_goals, ep_g), "assists": (r_assists, ep_a),
                    "appearance": (r_app, ep_app), "cleansheet": (r_cs, ep_cs), "other": (r_other, ep_other),
                },
            })

    missing = {
        "missing_realized_or_predicted": missing_realized_or_predicted,
        "missing_price": missing_price,
        "missing_position": missing_position,
    }
    return rows, missing


def price_band_calibration_report(
    con: duckdb.DuckDBPyConnection, backtest_run_id: int, *, tier: str | None = "mature",
    scoring_params_version: int = 1, n_boot: int = 2000, seed: int = 0,
) -> dict:
    """Phase 1A deliverable: overall + by-price-band + by-position + position-x-band + component
    decomposition by band, all recomputed from raw rows (never trusts backtest_metrics), with
    GW-clustered 95% CIs. Returns a versioned, long-form artefact (includes every row, not just
    the aggregates) -- see scripts/run_calibration_diagnostics.py for persisting this to disk
    with the full run-ID/git-SHA/timestamp/seasons/seed header the project's artefact-versioning
    convention requires."""
    rows, missing = fetch_calibration_rows(
        con, backtest_run_id, tier=tier, scoring_params_version=scoring_params_version,
    )
    bands = sorted({r["price_band"] for r in rows})
    positions = sorted({r["position"] for r in rows if r["position"] is not None})

    by_band = {b: summarize_residuals([r for r in rows if r["price_band"] == b], n_boot=n_boot, seed=seed) for b in bands}
    by_position = {p: summarize_residuals([r for r in rows if r["position"] == p], n_boot=n_boot, seed=seed) for p in positions}
    by_position_band = {}
    for p in positions:
        for b in bands:
            sliced = [r for r in rows if r["position"] == p and r["price_band"] == b]
            if sliced:
                by_position_band[f"{p}|{b}"] = summarize_residuals(sliced, n_boot=n_boot, seed=seed)

    components_by_band: dict = {}
    for b in bands:
        band_rows = [r for r in rows if r["price_band"] == b]
        comp_out = {}
        for key in COMPONENTS:
            comp_resids = [r["components"][key][0] - r["components"][key][1] for r in band_rows]
            comp_out[key] = {"n": len(comp_resids), "mean_resid": _mean(comp_resids)}
        components_by_band[b] = comp_out

    return {
        "artefact": "price_band_calibration_report",
        "backtest_run_id": backtest_run_id, "tier": tier, "scoring_params_version": scoring_params_version,
        "n_boot": n_boot, "seed": seed, "git_sha": _git_sha(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall": summarize_residuals(rows, n_boot=n_boot, seed=seed),
        "by_price_band": by_band, "by_position": by_position, "by_position_price_band": by_position_band,
        "components_by_price_band": components_by_band,
        "missing_data_counts": missing,
        "rows": rows,
    }


# ============================================================
# Phase 1B: selection-residual / "optimiser curse" diagnostic.
# ============================================================

def compute_curse(
    selected_residuals: list[float], selected_bands: list, pool_band_mean_resid: dict,
) -> dict:
    """Pure sign-convention core (no DB). `selected_bands[i]` is the comparison-pool key for
    `selected_residuals[i]` -- a price_band string, or a (position, price_band) tuple for the
    position-matched variant. `pool_band_mean_resid` must have an entry for every band actually
    present in `selected_bands` (raises KeyError otherwise -- a silent 0.0 default would be
    exactly the kind of fabricated result the mission rules forbid).

    sel_resid   = mean residual over the selected players.
    band_resid  = sum over bands of (selected share in that band) x (pool's own mean residual
                  in that band) -- "what you'd expect from an average player at these same
                  prices/positions."
    curse       = band_resid - sel_resid.  Sign convention: selection UNDERPERFORMS its matched
                  peers (sel_resid more negative than what the bands alone would predict) =>
                  curse > 0. Verified by tests/test_calibration_diagnostics.py's deterministic
                  fixture.
    """
    n = len(selected_residuals)
    if n == 0 or len(selected_bands) != n:
        return {"n": 0, "sel_resid": None, "band_resid": None, "curse": None, "curse_sum_scale": None}
    sel_resid = sum(selected_residuals) / n
    band_counts = Counter(selected_bands)
    band_resid = sum((count / n) * pool_band_mean_resid[band] for band, count in band_counts.items())
    curse = band_resid - sel_resid
    return {"n": n, "sel_resid": sel_resid, "band_resid": band_resid, "curse": curse, "curse_sum_scale": curse * n}


def _band_key(row: dict, matched: str):
    return row["price_band"] if matched == "price_band" else (row["position"], row["price_band"])


def selection_curse_for_step(
    con: duckdb.DuckDBPyConnection, so_run_id: int, step_rows: list[dict], *,
    matched: str = "price_band", selection: str = "xi",
) -> dict | None:
    """One (season, gameweek) step's curse, all four (matched x selection) combinations left
    explicit to the caller rather than silently picked for them -- see
    selection_curse_report()'s docstring for what each combination means.

    matched: 'price_band' or 'position_price_band' (position_price_matched_curse).
    selection: 'xi' (selected-XI, the mission's headline definition) or 'squad' (selected-15).

    Returns None when the step has no squad_optimizer selection, or none of its selected
    players appear in step_rows (e.g. a blank/degenerate step) -- never a fabricated curse.
    """
    flag_col = "in_xi" if selection == "xi" else "in_squad"
    selected_uids = [
        uid for (uid,) in con.execute(
            f"SELECT player_uid FROM squad_optimizer_selections WHERE run_id = ? AND {flag_col}", [so_run_id],
        ).fetchall()
    ]
    if not selected_uids:
        return None
    row_of = {r["player_uid"]: r for r in step_rows}
    pool_by_band: dict = {}
    for r in step_rows:
        pool_by_band.setdefault(_band_key(r, matched), []).append(r["residual"])
    pool_band_mean = {band: _mean(vals) for band, vals in pool_by_band.items()}

    resids, bands, missing = [], [], 0
    for uid in selected_uids:
        r = row_of.get(uid)
        if r is None:
            missing += 1
            continue
        resids.append(r["residual"])
        bands.append(_band_key(r, matched))
    if not resids:
        return None
    result = compute_curse(resids, bands, pool_band_mean)
    result["missing_selected_players"] = missing
    return result


def selection_curse_report(
    con: duckdb.DuckDBPyConnection, backtest_run_id: int, *, tier: str | None = "mature",
    scoring_params_version: int = 1, n_boot: int = 2000, seed: int = 0,
) -> dict:
    """Full robustness sweep across the 2x2 (matched x selection) grid the mission asks for,
    over every scored step of `backtest_run_id` that has a squad_optimizer selection. Reports
    per-step curse values plus cross-GW aggregate stats (mean/median/GW-clustered CI/sign-
    consistency), never a single-step or small-sample claim of curse ~ 0."""
    rows, missing = fetch_calibration_rows(con, backtest_run_id, tier=tier, scoring_params_version=scoring_params_version)
    by_step: dict = {}
    for r in rows:
        by_step.setdefault((r["season"], r["gameweek"]), []).append(r)

    so_run_of = dict(con.execute(
        "SELECT (season || ':' || gameweek), so_run_id FROM backtest_gameweek_steps "
        "WHERE backtest_run_id = ? AND so_run_id IS NOT NULL", [backtest_run_id],
    ).fetchall())

    variants = [(m, s) for m in ("price_band", "position_price_band") for s in ("xi", "squad")]
    per_variant: dict = {key: [] for key in variants}
    n_steps_with_selection = 0
    for (season, gameweek), step_rows in sorted(by_step.items()):
        so_run_id = so_run_of.get(f"{season}:{gameweek}")
        if so_run_id is None:
            continue
        n_steps_with_selection += 1
        for matched, selection in variants:
            result = selection_curse_for_step(con, so_run_id, step_rows, matched=matched, selection=selection)
            if result is not None and result["curse"] is not None:
                per_variant[(matched, selection)].append({
                    "season": season, "gameweek": gameweek, **result,
                })

    def _aggregate(entries: list[dict]) -> dict:
        curses = [e["curse"] for e in entries]
        if not curses:
            return {"n_gameweeks": 0, "mean_curse": None, "median_curse": None,
                     "curse_ci95_gw_clustered": None, "frac_positive": None}
        positive = sum(1 for c in curses if c > 0)
        return {
            "n_gameweeks": len(curses),
            "mean_curse": _mean(curses),
            "median_curse": _median(curses),
            "curse_ci95_gw_clustered": cluster_bootstrap_ci(
                entries, lambda e: e["curse"], lambda e: (e["season"], e["gameweek"]), n_boot=n_boot, seed=seed,
            ),
            "frac_positive": positive / len(curses),
        }

    return {
        "artefact": "selection_curse_report",
        "backtest_run_id": backtest_run_id, "tier": tier, "scoring_params_version": scoring_params_version,
        "n_boot": n_boot, "seed": seed, "git_sha": _git_sha(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_steps_scored": len(by_step), "n_steps_with_selection": n_steps_with_selection,
        "missing_data_counts": missing,
        "variants": {
            f"{matched}__{selection}": {"per_gameweek": per_variant[(matched, selection)], **_aggregate(per_variant[(matched, selection)])}
            for matched, selection in variants
        },
    }
