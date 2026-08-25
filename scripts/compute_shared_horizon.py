"""Real perf fix, not a new feature: the scheduled pipeline's four heaviest per-account steps
(run_transfer_planner_for_real_squad.py excluded -- see below) each independently call
transfer_planner.compute_horizon_ep() for the SAME multi-gameweek expected-points+uncertainty
forecast -- account-agnostic (team_strength/minutes_model/scoring are gameweek-and-account-
independent snapshots), so grade_squad.py, explain_my_move.py, and run_scenarios.py were
recomputing byte-identical results up to 6+ times per pipeline run (twice per script per
account, once for their own throwaway single-gameweek bootstrap lookup and once again inside
decision_engine.recommend_best_move()'s own transfer_planner.run() call). Measured, not
hypothesized: this is the concrete root cause behind the "Grade squad + explain best move"
(44 min) and "Run bench what-if scenarios" (3h42m) pipeline steps in a real run (#17) against
this repo's own ~20-25-minute documented estimate for the WHOLE pipeline.

This script computes that shared horizon exactly ONCE and writes it to a plain JSON file (see
transfer_planner.save_horizon_ep_versions()) that the three scripts above load instead of
recomputing, via the FPL_SHARED_HORIZON_FILE environment variable (unset/missing file -> those
scripts fall back to computing it themselves, unchanged from before this script existed -- this
is a pure opt-in, not a required dependency).

run_transfer_planner_for_real_squad.py is deliberately NOT wired to this cache: it calls
tp.run() with rho_residual_params_version=1, while the three scripts above (and this one) all
use version=2 -- a real, pre-existing discrepancy between this script's own param versions,
not something to silently paper over by forcing it onto a cache computed with a different
parameter version.

Usage (from repo root):
    PYTHONPATH=src python scripts/compute_shared_horizon.py <event> [output_path]

<event> is the current gameweek (same convention as every other real-squad script here --
the horizon computed covers event+1 onward). output_path defaults to
/tmp/fpl_shared_horizon_ep_versions.json (outside the repo -- this is an ephemeral, per-run
pipeline artifact, never committed data).
"""

import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import db, params as params_mod, transfer_planner as tp  # noqa: E402

TARGET_SEASON = "2026-2027"
DEFAULT_OUTPUT_PATH = Path("/tmp/fpl_shared_horizon_ep_versions.json")

# Must match grade_squad.py / explain_my_move.py / run_scenarios.py's own PARAM_VERSIONS
# exactly (all three already agree on these five) -- a mismatch here would silently hand
# those scripts a horizon computed under the wrong parameter version.
HORIZON_PARAMS_VERSION = 1
SCORING_PARAMS_VERSION = 1
BPS_PARAMS_VERSION = 1
TAU_PARAMS_VERSION = 1
RHO_RESIDUAL_PARAMS_VERSION = 2
CORR_PARAMS_VERSION = 1


def main() -> None:
    if len(sys.argv) not in (2, 3):
        raise SystemExit(f"usage: {sys.argv[0]} <event> [output_path]")
    current_event = int(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) == 3 else DEFAULT_OUTPUT_PATH
    plan_for_gameweek = current_event + 1

    con = db.connect()
    tp.seed_v1_params(con)

    ts_mv = con.execute("SELECT max(model_version) FROM team_strength_model_versions").fetchone()[0]
    mm_mv = con.execute("SELECT max(model_version) FROM minutes_model_versions").fetchone()[0]
    if ts_mv is None or mm_mv is None:
        raise SystemExit("no team_strength/minutes model versions found -- run scripts/run_ingestion.py first")

    horizon_gameweeks, _ = params_mod.resolve_param(con, "planning_horizon_params", "horizon_gameweeks", HORIZON_PARAMS_VERSION)

    horizon_ep_versions = tp.compute_horizon_ep(
        con, date.today(), TARGET_SEASON, plan_for_gameweek, ts_mv, mm_mv, int(horizon_gameweeks),
        SCORING_PARAMS_VERSION, BPS_PARAMS_VERSION, TAU_PARAMS_VERSION,
        RHO_RESIDUAL_PARAMS_VERSION, CORR_PARAMS_VERSION,
    )
    if plan_for_gameweek not in horizon_ep_versions:
        raise SystemExit(f"no fixtures found for {TARGET_SEASON} GW{plan_for_gameweek} -- cannot compute a shared horizon")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tp.save_horizon_ep_versions(output_path, horizon_ep_versions)
    print(f"[compute_shared_horizon] GW{plan_for_gameweek}+ horizon ({len(horizon_ep_versions)} gameweeks) -> wrote {output_path}")
    for gw, (ep_mv, un_mv) in sorted(horizon_ep_versions.items()):
        print(f"  GW{gw}: ep_model_version={ep_mv}, uncertainty_model_version={un_mv}")

    con.close()


if __name__ == "__main__":
    main()
