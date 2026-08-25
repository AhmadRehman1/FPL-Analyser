"""M8+M7: real season-long simulation -- ONE evolving manager (bootstrapped from a real M5
squad, evolved forward by real M8 transfer_planner.run()/apply_recommendation() decisions
every gameweek), not a fresh from-scratch squad every step. See backtest.run_season_simulation()
and the README's "Season-long objective" design notes for why this exists and how it differs
from scripts/run_backtest.py's own walk-forward loop.

Usage (from repo root):
    .venv/Scripts/python scripts/run_season_simulation.py

Requires the real project database already built (scripts/run_ingestion.py). Runs one season
simulation over START_GAMEWEEK..END_GAMEWEEK at the live pinned lambda/cap, then a real
lambda/concentration-cap sensitivity sweep (report_season_simulation_sensitivity()) scored on
season_cumulative_metrics() -- both real evidence from a real run, not assumed. Each gameweek
step is a genuine SCIP MIQP solve plus a Monte Carlo simulation (same per-step cost as
scripts/run_backtest.py's own steps), so a wide gameweek window or grid is a real time cost,
not free -- size START_GAMEWEEK/END_GAMEWEEK/the grids to what you're willing to wait for.

Also runs backtest.beats_baseline() over the SAME window (Phase B2 hardening): does the whole
M1-M6 architecture actually outperform picking by recent form, by ownership popularity, or the
ownership-weighted average manager -- measured, not just asserted. This is the real evidence
the README's own "65 of 71 parameters still invented" transparency section was missing: an
honest architecture disclosure says what's invented, but says nothing about whether the
machinery pays off against something simpler. A negative result here (the model losing to a
baseline) is exactly as worth printing as a positive one -- this script does not filter or
spin either way.
"""

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import backtest, db  # noqa: E402

TARGET_SEASON = "2026-2027"
START_GAMEWEEK = 2
END_GAMEWEEK = 6
RECALIBRATION_SEED_DIR = REPO_ROOT / "data" / "recalibration"


# Roadmap P1 item (Track B, docs/plans/2026-08_roadmap_plan.md): recalibratable families
# (xi/rho/rho_residual/adjustment/shrinkage/fact_multiplier/lambda/kappa_tc) resolve from the
# git-committed confirmed-seed files via backtest.active_recalibratable_versions() -- see that
# function's own docstring. Every other family below isn't one recalibrate() can produce a new
# version for, so those stay hardcoded literals.
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


def main() -> None:
    con = db.connect()
    active = backtest.active_recalibratable_versions(RECALIBRATION_SEED_DIR)
    PARAM_VERSIONS = _param_versions(active)

    t0 = time.time()
    result = backtest.run_season_simulation(con, TARGET_SEASON, START_GAMEWEEK, END_GAMEWEEK, **PARAM_VERSIONS)
    print(f"[run_season_simulation] {time.time() - t0:.1f}s -> {len(result['weekly_points'])} gameweeks scored")
    print(f"  weekly_points={result['weekly_points']}")
    print(f"  actions={result['actions']}")
    if result["skipped_dgw_gameweeks"]:
        print(f"  skipped (DGW): {result['skipped_dgw_gameweeks']}")

    metrics = backtest.season_cumulative_metrics(result["weekly_points"])
    print(f"  total_points={metrics['total_points']:.1f} mean={metrics['mean_points']:.1f} "
          f"realized_sharpe={metrics['realized_sharpe']:.3f} max_drawdown={metrics['max_drawdown']:.1f}")

    t0 = time.time()
    baseline_version_keys = (
        "xi_params_version", "rho_params_version", "decay_params_version", "adjustment_params_version",
        "shrinkage_params_version", "fact_multiplier_params_version", "scoring_params_version",
        "bps_params_version", "tau_params_version",
    )
    beats = backtest.beats_baseline(
        con, TARGET_SEASON, START_GAMEWEEK, END_GAMEWEEK,
        model_gameweeks=result["gameweeks"], model_weekly_points=result["weekly_points"],
        ownership_params_version=1,
        **{k: v for k, v in PARAM_VERSIONS.items() if k in baseline_version_keys},
    )
    print(f"[beats_baseline] {time.time() - t0:.1f}s")
    print(f"  model: total_points={beats['model']['total_points']:.1f} over {beats['model']['n_gameweeks']} gameweeks")
    for name in ("recent_points", "ownership_popularity", "crowd"):
        b = beats[name]
        verdict = "beats" if b["model_beats_baseline_by"] > 0 else "LOSES TO" if b["model_beats_baseline_by"] < 0 else "ties"
        print(
            f"  {name}: total_points={b['total_points']:.1f} over {b['n_gameweeks']} gameweeks "
            f"({b['n_gameweeks_skipped']} skipped) -- model {verdict} it by {b['model_beats_baseline_by']:.1f} "
            f"over the {b['n_gameweeks']} comparable gameweek(s)"
        )

    # lambda=0.0 is not a valid grid candidate for this harness -- see
    # report_season_simulation_sensitivity()'s own docstring (it always goes through the real,
    # divergence-checked squad_optimizer.run(), which structurally can't compare lambda=0
    # against itself).
    t0 = time.time()
    sensitivity = backtest.report_season_simulation_sensitivity(
        con, TARGET_SEASON, START_GAMEWEEK, END_GAMEWEEK, PARAM_VERSIONS,
        lambda_grid=(0.10, 0.15, 0.20, 0.30, 0.50), guardrail_cap_grid=(2, 3, 4, 5),
    )
    print(f"[report_season_simulation_sensitivity] {time.time() - t0:.1f}s")
    for lam, m in sensitivity["lambda"].items():
        print(f"  lambda={lam}: total={m['total_points']:.1f} sharpe={m['realized_sharpe']:.3f} drawdown={m['max_drawdown']:.1f}")
    for cap, m in sensitivity["guardrail_cap"].items():
        print(f"  cap={cap}: total={m['total_points']:.1f} sharpe={m['realized_sharpe']:.3f} drawdown={m['max_drawdown']:.1f}")

    con.close()


if __name__ == "__main__":
    main()
