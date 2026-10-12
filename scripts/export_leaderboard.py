"""Review B2 / roadmap Feature 7: publishes the model's real backtested edge (or lack of it)
over three baselines a manager could follow with none of this project's M1-M6 machinery -- the
change that converts the architecture from a disclosure ("65 of 71 parameters still invented")
into evidence. See backtest.beats_baseline() (Phase B2 hardening) for the actual computation;
this script runs backtest.run_season_simulation() to get a real model trajectory (the same call
scripts/run_season_simulation.py itself makes) and then beats_baseline() against it, reshaping
both into the PWA-facing JSON that script's own console-only report doesn't produce.

Deliberately does NOT also run report_season_simulation_sensitivity() (the lambda/guardrail-cap
sweep run_season_simulation.py additionally does) -- the dashboard leaderboard only needs the
beats_baseline comparison, and the sweep is a materially more expensive, separate research
question (see that function's own docstring).

Two like-for-like rows follow the baselines (docs/plans/2026-10_beat_the_template.md, Req 3):
"model (oracle)", the walk-forward's fresh hit-free model squad on this season, and "ownership
template (available)", the availability-gated template it is judged against. Both are oracles,
like every baseline here; each oracle row carries oracle_model_beats_baseline_by beside the
evolving model's model_beats_baseline_by.

Usage (from repo root):
    PYTHONPATH=src python scripts/export_leaderboard.py [start_gameweek] [end_gameweek]

With no end_gameweek, walks GW2 -> the last gameweek FPL itself marks finished (a season
simulation can only score gameweeks with real, settled results). Early in a season that window
is too short to be meaningful: below MIN_GAMEWEEK_SPAN completed gameweeks the script writes an
explicit {"status": "insufficient_data", ...} payload and exits 0 -- a real "not enough season
yet" state the PWA surfaces as such, NOT a silent empty section and NOT a hard failure.

Real MIQP solve + Monte Carlo simulation per gameweek in the window -- a real time cost, same as
scripts/run_season_simulation.py's own steps, not free. Deliberately NOT wired into
scheduled_pipeline.yml's twice-daily cadence for that reason; runs as its own decoupled job in
.github/workflows/weekly_backtest.yml (parallel to the multi-hour M7 backtest, not gated behind
it, so a backtest timeout can't starve this feed -- the failure mode before this).
"""

import inspect
import json
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import app_export as ax  # noqa: E402
from fpl_quant import backtest, db  # noqa: E402

TARGET_SEASON = "2026-2027"
START_GAMEWEEK = 2
# Fallback only, used when the live bootstrap-static fetch to resolve the last finished
# gameweek fails (e.g. the Claude Code sandbox's network policy blocks fantasy.premierleague.com
# -- same caveat as every other real-fetch script here). A real CI run resolves this live.
END_GAMEWEEK_FALLBACK = 2
# A season-long simulation over fewer than this many completed gameweeks isn't a meaningful
# "model vs baselines" comparison yet -- below it, write an explicit insufficient_data payload
# rather than a thin/noisy leaderboard or a hard error.
MIN_GAMEWEEK_SPAN = 3
DASHBOARD_DIR = REPO_ROOT / "data" / "dashboard"
RECALIBRATION_SEED_DIR = REPO_ROOT / "data" / "recalibration"


def _resolve_end_gameweek() -> int:
    try:
        resolved = ax.last_finished_event(ax.fetch_bootstrap_static())
    except Exception as e:  # noqa: BLE001 -- best-effort; an explicit CLI arg always overrides this
        print(f"::warning::export_leaderboard: last-finished-gameweek fetch failed ({e}) -- "
              f"defaulting end_gameweek to GW{END_GAMEWEEK_FALLBACK}.")
        return END_GAMEWEEK_FALLBACK
    if resolved is None:
        print(f"::warning::export_leaderboard: bootstrap-static reports no finished gameweek -- "
              f"defaulting end_gameweek to GW{END_GAMEWEEK_FALLBACK}.")
        return END_GAMEWEEK_FALLBACK
    return resolved


def _write(payload: dict, data_asof: str) -> Path:
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DASHBOARD_DIR / f"leaderboard_{data_asof}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    # Stable-named copy, same "PWA needs a fixed, predictable path" convention as
    # app_team_<id>.json -- a static site can't discover today's date-embedded filename on its
    # own. The dated file stays too, as the historical/audit record.
    (DASHBOARD_DIR / "leaderboard_latest.json").write_text(json.dumps(payload, indent=2))
    return out_path


# Roadmap P1 item (Track B, docs/plans/2026-08_roadmap_plan.md): recalibratable families
# (xi/rho/rho_residual/adjustment/shrinkage/fact_multiplier/lambda/kappa_tc) resolve from the
# git-committed confirmed-seed files via backtest.active_recalibratable_versions() -- see that
# function's own docstring. Every other family below isn't one recalibrate() can produce a new
# version for, so those stay hardcoded literals, same set scripts/run_season_simulation.py uses.
def _param_versions(active: dict) -> dict:
    return dict(
        xi_params_version=active["xi_params_version"], rho_params_version=active["rho_params_version"],
        decay_params_version=1, adjustment_params_version=active["adjustment_params_version"],
        shrinkage_params_version=active["shrinkage_params_version"],
        fact_multiplier_params_version=active["fact_multiplier_params_version"],
        scoring_params_version=1, bps_params_version=1, tau_params_version=1,
        rho_residual_params_version=active["rho_residual_params_version"], corr_params_version=1,
        lambda_params_version=active["lambda_params_version"], guardrail_params_version=1,
        horizon_params_version=1, transfer_cost_params_version=1,
        wildcard_threshold_params_version=1, free_hit_threshold_params_version=1,
        kappa_tc_params_version=active["kappa_tc_params_version"],
        captain_risk_params_version=active["captain_risk_params_version"],
        rate_shrinkage_params_version=active["rate_shrinkage_params_version"],
        rate_prior_params_version=active["rate_prior_params_version"],
        minutes_bounds_params_version=active["minutes_bounds_params_version"],
        minutes_start_prior_params_version=active["minutes_start_prior_params_version"],
        team_strength_guard_params_version=active["team_strength_guard_params_version"],
        current_season_role_params_version=active["current_season_role_params_version"],
        current_season_minutes_params_version=active["current_season_minutes_params_version"],
        sub_rate_params_version=active["sub_rate_params_version"],
    )

# beats_baseline() only takes the M1-M3 subset of PARAM_VERSIONS (the baselines never touch
# M5/M6) -- see run_season_simulation.py's own identical filter.
_BASELINE_VERSION_KEYS = (
    "xi_params_version", "rho_params_version", "decay_params_version", "adjustment_params_version",
    "shrinkage_params_version", "fact_multiplier_params_version", "scoring_params_version",
    "bps_params_version", "tau_params_version", "minutes_bounds_params_version",
    "rate_shrinkage_params_version", "minutes_start_prior_params_version",
    "team_strength_guard_params_version", "rate_prior_params_version",
    "current_season_role_params_version", "current_season_minutes_params_version", "sub_rate_params_version",
)

_BASELINE_DISPLAY_NAMES = {
    "recent_points": "recent-points baseline",
    "ownership_popularity": "price/ownership baseline",
    # real points weighted by ownership: far above FPL's own published average
    "crowd": "synthetic EO crowd (not FPL's average)",
}

# The like-for-like rows (docs/plans/2026-10_beat_the_template.md, Req 3). Every baseline above
# is an oracle -- a fresh, hit-free squad each gameweek -- while the model row is the evolving
# manager. These two put an oracle model beside them: the walk-forward's own fresh model squad
# on this season, and the availability-gated ownership template it is judged against. Both are
# is_baseline rows, appended after the others, so the evolving model stays the first
# non-baseline row the app reads.
_ORACLE_ROWS = {
    "model_oracle": "model (oracle)",
    "template_gated": "ownership template (available)",
}
ORACLE_RUN_NOTES = "export_leaderboard.py: the model (oracle) and gated-template rows"


def oracle_weeks(con, param_versions: dict, season: str, start_gw: int, end_gw: int) -> dict[str, dict[int, float]]:
    """{"model_oracle": {gw: pts}, "template_gated": {gw: pts}}: for each gameweek in the window,
    the walk-forward step itself (backtest.run_gameweek_step with this export's own live
    versions and the gated template), scored as the walk-forward scores it
    (backtest.score_gameweek: auto-subs, the solve's vice). A gameweek a row can't score is
    missing from that row's map."""
    # the walk-forward's solve: no EO term (the live from-scratch solve's template anchor is a
    # live-only setting, docs/plans/2026-10_beat_the_template.md "Current State")
    step_params = set(inspect.signature(backtest.run_gameweek_step).parameters) - {"ownership_params_version"}
    kwargs = {k: v for k, v in param_versions.items() if k in step_params}
    run_id = con.execute(
        "INSERT INTO backtest_runs (warm_up_gameweeks, notes, steps_planned, seasons) VALUES (0, ?, ?, ?) "
        "RETURNING backtest_run_id",
        [ORACLE_RUN_NOTES, end_gw - start_gw + 1, json.dumps([season])],
    ).fetchone()[0]
    out: dict[str, dict[int, float]] = {"model_oracle": {}, "template_gated": {}}
    for gw in range(start_gw, end_gw + 1):
        if not backtest.has_fittable_history(con, season, gw):
            continue
        template = backtest.run_gameweek_step(
            con, run_id, season, gw, run_monte_carlo=False, ownership_params_version=None,
            template_baseline=True, **kwargs,
        )
        ep_mv, mm_mv, ts_mv, so_run_id = con.execute(
            "SELECT ep_model_version, mm_model_version, ts_model_version, so_run_id FROM backtest_gameweek_steps "
            "WHERE backtest_run_id = ? AND season = ? AND gameweek = ?", [run_id, season, gw],
        ).fetchone()
        backtest.score_gameweek(
            con, run_id, season, gw, ep_mv, mm_mv, ts_mv, param_versions["scoring_params_version"],
            so_run_id=so_run_id, ownership_params_version=1, template=template,
        )
        metrics = dict(con.execute(
            "SELECT metric_name, metric_value FROM backtest_metrics WHERE backtest_run_id = ? AND season = ? "
            "AND gameweek = ? AND metric_name IN ('model_squad_realized_points', 'template_squad_realized_points')",
            [run_id, season, gw],
        ).fetchall())
        if "model_squad_realized_points" in metrics:
            out["model_oracle"][gw] = metrics["model_squad_realized_points"]
        if "template_squad_realized_points" in metrics:
            out["template_gated"][gw] = metrics["template_squad_realized_points"]
    return out


def _gap(ref: dict[int, float], row: dict[int, float]) -> tuple[float, float]:
    """(ref's total, ref - row) over the gameweeks both have."""
    gws = [gw for gw in row if gw in ref]
    ref_total = sum(ref[gw] for gw in gws)
    return ref_total, ref_total - sum(row[gw] for gw in gws)


def _weekly(points: dict[int, float]) -> dict[str, float]:
    return {str(gw): round(pts, 1) for gw, pts in sorted(points.items())}


def build_rows(model_weekly: dict[int, float], baselines: dict[str, dict[int, float]],
               oracle: dict[str, dict[int, float]], window_gameweeks: int,
               model_hits: dict[int, float] | None = None) -> list[dict]:
    """The leaderboard rows: the evolving model first, then every baseline (the three existing
    ones, then the model (oracle) and gated-template rows). model_beats_baseline_by is always the
    evolving model's total minus the row's, over the gameweeks the row has; the oracle rows and
    every oracle baseline also carry oracle_model_beats_baseline_by, the model (oracle) total
    minus the row's on the same terms. Every row carries weekly_points ({gw: points}); the model
    row also weekly_hits, the transfer hits its gross points don't deduct (the plan's Phase 3
    pairs it per gameweek with the live team)."""
    rows = [{
        "name": "FPL-Analyser (model)", "total_points": round(sum(model_weekly.values()), 1),
        "n_gameweeks": len(model_weekly), "is_baseline": False,
        "weekly_points": _weekly(model_weekly), "weekly_hits": _weekly(model_hits or {}),
    }]
    model_oracle = oracle.get("model_oracle") or {}
    entries = [(_BASELINE_DISPLAY_NAMES[k], k, baselines[k]) for k in _BASELINE_DISPLAY_NAMES]
    entries += [(_ORACLE_ROWS[k], k, oracle.get(k) or {}) for k in _ORACLE_ROWS]
    for name, kind, weekly in entries:
        model_same, beats_by = _gap(model_weekly, weekly)
        row = {
            "name": name, "kind": kind, "total_points": round(sum(weekly.values()), 1), "n_gameweeks": len(weekly),
            "n_gameweeks_skipped": window_gameweeks - len(weekly),
            "model_total_points_same_gameweeks": round(model_same, 1),
            "model_beats_baseline_by": round(beats_by, 1), "is_baseline": True,
            "weekly_points": _weekly(weekly),
        }
        if kind != "model_oracle":
            row["oracle_model_beats_baseline_by"] = round(_gap(model_oracle, weekly)[1], 1)
        rows.append(row)
    return rows


def main() -> None:
    start_gw = int(sys.argv[1]) if len(sys.argv) > 1 else START_GAMEWEEK
    end_gw = int(sys.argv[2]) if len(sys.argv) > 2 else _resolve_end_gameweek()
    data_asof = date.today().isoformat()

    completed_span = end_gw - start_gw + 1
    if completed_span < MIN_GAMEWEEK_SPAN:
        payload = {
            "data_asof": data_asof, "season": TARGET_SEASON, "status": "insufficient_data",
            "start_gameweek": start_gw, "end_gameweek": end_gw, "walked_gameweeks": 0,
            "completed_gameweeks": max(completed_span, 0), "min_gameweeks": MIN_GAMEWEEK_SPAN,
            "rows": [], "honest_losses": [],
        }
        out_path = _write(payload, data_asof)
        print(f"[export_leaderboard] only {max(completed_span, 0)} completed gameweek(s) in "
              f"GW{start_gw}-{end_gw} (need {MIN_GAMEWEEK_SPAN}) -- wrote insufficient_data payload to {out_path}")
        return

    con = db.connect()
    active = backtest.active_recalibratable_versions(RECALIBRATION_SEED_DIR)
    param_versions = _param_versions(active)

    result = backtest.run_season_simulation(con, TARGET_SEASON, start_gw, end_gw, **param_versions)
    print(f"[run_season_simulation] {len(result['weekly_points'])} gameweeks scored")

    beats = backtest.beats_baseline(
        con, TARGET_SEASON, start_gw, end_gw,
        model_gameweeks=result["gameweeks"], model_weekly_points=result["weekly_points"],
        ownership_params_version=1,
        **{k: v for k, v in param_versions.items() if k in _BASELINE_VERSION_KEYS},
    )

    oracle = oracle_weeks(con, param_versions, TARGET_SEASON, start_gw, end_gw)
    print(f"[oracle rows] model (oracle) {len(oracle['model_oracle'])} GWs, gated template "
          f"{len(oracle['template_gated'])} GWs")

    model_n = beats["model"]["n_gameweeks"]
    rows = build_rows(
        dict(zip(result["gameweeks"], result["weekly_points"])),
        {key: beats[key]["weekly"] for key in _BASELINE_DISPLAY_NAMES}, oracle, end_gw - start_gw + 1,
        model_hits=dict(zip(result["gameweeks"], result["weekly_hits"])),
    )
    # Reports a baseline win exactly as visibly as a model win -- a leaderboard that only shows
    # wins is not a leaderboard (same disclosure discipline backtest.beats_baseline() itself
    # documents). The model (oracle) row is the model itself, so it is no baseline to lose to.
    honest_losses = [
        row["name"] for row in rows[1:] if row["kind"] != "model_oracle" and row["model_beats_baseline_by"] < 0
    ]

    payload = {
        "data_asof": data_asof, "season": TARGET_SEASON, "status": "ok",
        "start_gameweek": start_gw, "end_gameweek": end_gw,
        "walked_gameweeks": model_n, "rows": rows, "honest_losses": honest_losses,
    }

    out_path = _write(payload, data_asof)
    print(f"[export_leaderboard] wrote {out_path}")
    for row in rows:
        print(f"  {row['name']}: total_points={row['total_points']:.1f} over {row['n_gameweeks']} gameweeks")
    if honest_losses:
        print(f"  honest_losses: {honest_losses}")

    con.close()


if __name__ == "__main__":
    main()
