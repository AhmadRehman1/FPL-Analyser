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
    )

# beats_baseline() only takes the M1-M3 subset of PARAM_VERSIONS (the baselines never touch
# M5/M6) -- see run_season_simulation.py's own identical filter.
_BASELINE_VERSION_KEYS = (
    "xi_params_version", "rho_params_version", "decay_params_version", "adjustment_params_version",
    "shrinkage_params_version", "fact_multiplier_params_version", "scoring_params_version",
    "bps_params_version", "tau_params_version",
)

_BASELINE_DISPLAY_NAMES = {
    "recent_points": "recent-points baseline",
    "ownership_popularity": "price/ownership baseline",
    "crowd": "crowd-average baseline",
}


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

    model_n = beats["model"]["n_gameweeks"]
    rows = [{
        "name": "FPL-Analyser (model)", "total_points": round(beats["model"]["total_points"], 1),
        "n_gameweeks": model_n, "is_baseline": False,
    }]
    for key, display_name in _BASELINE_DISPLAY_NAMES.items():
        b = beats[key]
        rows.append({
            "name": display_name, "total_points": round(b["total_points"], 1), "n_gameweeks": b["n_gameweeks"],
            "n_gameweeks_skipped": b["n_gameweeks_skipped"],
            "model_total_points_same_gameweeks": round(b["model_total_points_same_gameweeks"], 1),
            "model_beats_baseline_by": round(b["model_beats_baseline_by"], 1), "is_baseline": True,
        })
    # Reports a baseline win exactly as visibly as a model win -- a leaderboard that only shows
    # wins is not a leaderboard (same disclosure discipline backtest.beats_baseline() itself
    # documents).
    honest_losses = [
        _BASELINE_DISPLAY_NAMES[key] for key in _BASELINE_DISPLAY_NAMES if beats[key]["model_beats_baseline_by"] < 0
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
