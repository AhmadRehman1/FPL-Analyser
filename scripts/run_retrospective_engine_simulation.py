"""Track E, Phase E-2: run the engine's own 2025-26 season, from GW2, BLIND -- all 18 required
parameter-version kwargs hardcoded to 1, never resolved via active_recalibratable_versions().
See docs/plans/2026-08_retrospective_validation_and_ml_decision_plan.md for the full design and
why "blind" matters here specifically: Track B's automated recalibration has already promoted
several of these families to newer versions, fit in part against 2025-26's own real outcomes --
using those *currently-active* versions to simulate 2025-26 would give the engine hindsight it
never actually had at the time. This is the same class of leakage asof_scope() exists elsewhere
in this codebase to prevent, just at the parameter-version layer instead of the data layer.

**Correction found building this phase (not assumed in the original plan draft): GW1 is not
actually bootstrappable, for any season.** squad_optimizer.fetch_candidate_pool() requires a
real `now_cost` snapshot per player, read from fact_player_season_stats -- inside asof_scope(),
that table is shadowed to `gw < start_gameweek`. Verified live against the real DB:
`SELECT count(*) FROM fact_player_season_stats WHERE season='2025-2026' AND gw < 1` returns
zero -- there is no pre-season price row for any season in this schema, so a GW1 bootstrap asof-
sees no prices at all and squad_optimizer.run() fails with "candidate pool has only 0 priced
players." This is exactly why the existing, real scripts/run_season_simulation.py already
defaults to START_GAMEWEEK = 2, not 1 -- this script now matches that same, pre-existing
convention rather than assuming GW1 was reachable. The engine's own strategy is therefore
simulated from GW2 (the earliest gameweek with real, asof-safe pricing available), not a true
GW1 -- reflected in the methodology disclaimer this script's own output feeds into Phase E-3.

_blind_param_versions() below adapts scripts/run_season_simulation.py's own _param_versions()
function verbatim, but replaces every active[...] lookup (the 8 families that are normally
recalibratable) with a literal 1 -- so it's the same 18-name shape that script already proves is
correct and complete (verified against run_season_simulation()'s own 18-kwarg signature), just
never resolving any of them from the live recalibration state.

Usage (from repo root):
    .venv/Scripts/python scripts/run_retrospective_engine_simulation.py
"""

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import backtest, db  # noqa: E402

TARGET_SEASON = "2025-2026"
START_GAMEWEEK = 2  # GW1 is not bootstrappable -- see module docstring for the verified reason
END_GAMEWEEK = 38
OUTPUT_FILE = REPO_ROOT / "data" / "retrospective" / "2025-2026_engine_simulation.json"


def _blind_param_versions() -> dict:
    """Every one of the 18 kwargs run_season_simulation() requires, hardcoded to version 1 --
    both the 8 families that are normally recalibratable (xi, rho, rho_residual, adjustment,
    shrinkage, fact_multiplier, lambda, kappa_tc -- RECALIBRATABLE_VERSION_ARGS' own 8 keys,
    src/fpl_quant/backtest.py:1510-1524) and the 10 that are never recalibrated anywhere in this
    codebase (decay, scoring, bps, tau, corr, guardrail, horizon, transfer_cost,
    wildcard_threshold, free_hit_threshold -- always 1 even in scripts/run_season_simulation.py's
    own real, active-version-resolving call). Deliberately does NOT call
    backtest.active_recalibratable_versions() for any of the 8 -- that's the whole point."""
    return dict(
        xi_params_version=1, rho_params_version=1,
        decay_params_version=1, adjustment_params_version=1,
        shrinkage_params_version=1,
        fact_multiplier_params_version=1,
        scoring_params_version=1, bps_params_version=1, tau_params_version=1,
        rho_residual_params_version=1, corr_params_version=1,
        lambda_params_version=1, guardrail_params_version=1,
        horizon_params_version=1, transfer_cost_params_version=1,
        wildcard_threshold_params_version=1, free_hit_threshold_params_version=1,
        kappa_tc_params_version=1,
    )


def main() -> None:
    param_versions = _blind_param_versions()
    assert len(param_versions) == 18, f"expected 18 kwargs, got {len(param_versions)}"

    print(f"[run_retrospective_engine_simulation] simulating {TARGET_SEASON} GW{START_GAMEWEEK}-"
          f"{END_GAMEWEEK}, blind (all 18 param versions = 1): {param_versions}")

    con = db.connect()
    t0 = time.time()
    result = backtest.run_season_simulation(
        con, TARGET_SEASON, START_GAMEWEEK, END_GAMEWEEK, **param_versions,
    )
    elapsed = time.time() - t0

    total_points = sum(result["weekly_points"])
    print(f"[run_retrospective_engine_simulation] done in {elapsed:.1f}s ({elapsed / 60:.1f} min): "
          f"{len(result['weekly_points'])} gameweeks scored, total={total_points}, "
          f"skipped_dgw_gameweeks={result['skipped_dgw_gameweeks']}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "target_season": TARGET_SEASON,
        "start_gameweek": START_GAMEWEEK,
        "end_gameweek": END_GAMEWEEK,
        "param_versions_used": param_versions,
        "blind_simulation": True,
        "final_state_version": result["final_state_version"],
        "weekly_points": result["weekly_points"],
        "gameweeks": result["gameweeks"],
        "total_points": total_points,
        "actions": result["actions"],
        "skipped_dgw_gameweeks": result["skipped_dgw_gameweeks"],
        "wall_clock_seconds": elapsed,
        "run_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    OUTPUT_FILE.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[run_retrospective_engine_simulation] cached to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
