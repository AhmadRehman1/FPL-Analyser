"""Roadmap Feature 9: elite-manager tracking, done transparently. For each configured elite
manager's latest real transfer, shows the model's own recommendation for that manager's
squad vs what they actually did, plus a real, computed divergence reason. Writes
data/dashboard/elite_divergence_<asof>.json.

Usage (from repo root):
    PYTHONPATH=src python scripts/track_elite.py [event]

<event> defaults to bootstrap-static's own current gameweek. Reads the configurable elite
manager list from data/elite_managers.json (empty by default -- see that file's own comment).

Same network-blocked-in-sandbox caveat as every other real-squad script in this project.
"""

import json
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import app_export as ax  # noqa: E402
from fpl_quant import backtest, db, elite_tracking as et, ingest_fpl_entry_picks as ifp, transfer_planner as tp  # noqa: E402

TARGET_SEASON = "2026-2027"
DASHBOARD_DIR = REPO_ROOT / "data" / "dashboard"
ELITE_MANAGERS_PATH = REPO_ROOT / "data" / "elite_managers.json"
RECALIBRATION_SEED_DIR = REPO_ROOT / "data" / "recalibration"

# Roadmap P1 item (Track B, docs/plans/2026-08_roadmap_plan.md): rho_residual_params_version/
# lambda_params_version/kappa_tc_params_version resolve from the git-committed confirmed-seed
# files, not a hardcoded literal -- see backtest.active_recalibratable_versions()'s own
# docstring. Resolved inside main() (below), not at import time, for consistency with the rest
# of the sweep, and after the empty-elite_managers early-return so the no-op common case (this
# project's own default) doesn't pay for a file read it never uses.
def _param_versions(active: dict) -> dict:
    return dict(
        horizon_params_version=1, scoring_params_version=1, bps_params_version=1, tau_params_version=1,
        rho_residual_params_version=active["rho_residual_params_version"], corr_params_version=1,
        transfer_cost_params_version=1,
        lambda_params_version=active["lambda_params_version"], guardrail_params_version=1,
        wildcard_threshold_params_version=1,
        free_hit_threshold_params_version=1, kappa_tc_params_version=active["kappa_tc_params_version"],
    )


def main() -> None:
    con = db.connect()
    tp.seed_v1_params(con)

    elite_managers = et.load_elite_managers(ELITE_MANAGERS_PATH)
    if not elite_managers:
        print(f"[track_elite] no elite managers configured in {ELITE_MANAGERS_PATH} -- nothing to track")
        con.close()
        return
    PARAM_VERSIONS = _param_versions(backtest.active_recalibratable_versions(RECALIBRATION_SEED_DIR))

    if len(sys.argv) > 1:
        current_event = int(sys.argv[1])
    else:
        bootstrap = ax.fetch_bootstrap_static()
        current_event = ax.current_event(bootstrap)
        if current_event is None:
            raise SystemExit("no current gameweek reported by bootstrap-static -- pass one explicitly")

    ts_mv = con.execute("SELECT max(model_version) FROM team_strength_model_versions").fetchone()[0]
    mm_mv = con.execute("SELECT max(model_version) FROM minutes_model_versions").fetchone()[0]
    if ts_mv is None or mm_mv is None:
        raise SystemExit("no team_strength/minutes model versions found -- run scripts/run_ingestion.py first")

    calibration_asof_date = date.today()
    run_kwargs = {"ts_model_version": ts_mv, "mm_model_version": mm_mv, **PARAM_VERSIONS}
    element_names = ifp.fetch_bootstrap_elements()

    divergences = et.build_elite_divergence(
        con, elite_managers, calibration_asof_date, TARGET_SEASON, current_event,
        ifp.fetch_entry_picks, element_names, run_kwargs,
    )

    data_asof = calibration_asof_date.isoformat()
    payload = {"data_asof": data_asof, "gw": current_event, "managers": divergences}

    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DASHBOARD_DIR / f"elite_divergence_{data_asof}.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    # Stable-named copy, same "PWA needs a fixed, predictable path" convention as
    # app_team_<id>.json -- a static site can't discover today's date-embedded filename on
    # its own. The dated file stays too, as the historical/audit record.
    (DASHBOARD_DIR / "elite_divergence_latest.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[track_elite] {len(divergences)}/{len(elite_managers)} managers tracked -> wrote {out_path}")
    for d in divergences:
        flag = "DIVERGED" if d["diverged"] else "matched"
        print(f"  {d['name']}: actual={d['actual_move']} model={d['model_move']} [{flag}]")

    con.close()


if __name__ == "__main__":
    main()
