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

# Every param family below is currently seeded at version=1 by scripts/run_ingestion.py.
PARAM_VERSIONS = dict(
    xi_params_version=1, rho_params_version=1,
    decay_params_version=1, adjustment_params_version=1, shrinkage_params_version=1, fact_multiplier_params_version=1,
    scoring_params_version=1, bps_params_version=1, tau_params_version=1,
    rho_residual_params_version=1, corr_params_version=1,
    lambda_params_version=1, guardrail_params_version=1,
    horizon_params_version=1, transfer_cost_params_version=1,
    wildcard_threshold_params_version=1, free_hit_threshold_params_version=1, kappa_tc_params_version=1,
    transfer_accept_threshold_params_version=1,
)


def main() -> None:
    con = db.connect()

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
