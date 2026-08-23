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
from fpl_quant import db, elite_tracking as et, ingest_fpl_entry_picks as ifp, transfer_planner as tp  # noqa: E402

TARGET_SEASON = "2026-2027"
DASHBOARD_DIR = REPO_ROOT / "data" / "dashboard"
ELITE_MANAGERS_PATH = REPO_ROOT / "data" / "elite_managers.json"

PARAM_VERSIONS = dict(
    horizon_params_version=1, scoring_params_version=1, bps_params_version=1, tau_params_version=1,
    rho_residual_params_version=2, corr_params_version=1, transfer_cost_params_version=1,
    lambda_params_version=1, guardrail_params_version=1, wildcard_threshold_params_version=1,
    free_hit_threshold_params_version=1, kappa_tc_params_version=1,
)


def main() -> None:
    con = db.connect()
    tp.seed_v1_params(con)

    elite_managers = et.load_elite_managers(ELITE_MANAGERS_PATH)
    if not elite_managers:
        print(f"[track_elite] no elite managers configured in {ELITE_MANAGERS_PATH} -- nothing to track")
        con.close()
        return

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
    print(f"[track_elite] {len(divergences)}/{len(elite_managers)} managers tracked -> wrote {out_path}")
    for d in divergences:
        flag = "DIVERGED" if d["diverged"] else "matched"
        print(f"  {d['name']}: actual={d['actual_move']} model={d['model_move']} [{flag}]")

    con.close()


if __name__ == "__main__":
    main()
