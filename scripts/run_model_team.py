"""Advance + score the model-managed team, then write its public track-record panel.

The model's own team (its from-scratch GW1 squad, then its real weekly transfer/captain/chip
decisions), tracked live and scored against the FPL overall average -- the conversion proof
for the Track Record page (docs/BUSINESS_PLAN.md P0). See src/fpl_quant/model_team.py.

Runs in scheduled_pipeline.yml right after run_ingestion.py (needs the freshly-ingested DB:
the GW1 squad_optimizer solve + fact_player_season_stats.event_points for realised scoring)
and needs open internet for the FPL bootstrap-static field averages.

Usage (from repo root):
    PYTHONPATH=src python scripts/run_model_team.py [current_event]
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import app_export as ax  # noqa: E402
from fpl_quant import backtest as bt  # noqa: E402
from fpl_quant import db  # noqa: E402
from fpl_quant import model_team  # noqa: E402

SEASON = "2026-2027"
STATE_DIR = REPO_ROOT / "data" / "model_team"
DASHBOARD_DIR = REPO_ROOT / "data" / "dashboard"
RECALIBRATION_SEED_DIR = REPO_ROOT / "data" / "recalibration"


def main() -> None:
    con = db.connect()
    active = bt.active_recalibratable_versions(RECALIBRATION_SEED_DIR)

    bootstrap = ax.fetch_bootstrap_static()
    current_event = (
        int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else ax.current_event(bootstrap)
    )
    if current_event is None:
        raise SystemExit("bootstrap-static reports no current gameweek and none was passed")

    field_average_by_gw = {
        ev["id"]: ev["average_entry_score"]
        for ev in bootstrap.get("events", [])
        if ev.get("average_entry_score") is not None and ev.get("finished")
    }

    state = model_team.advance(
        con, current_event=current_event, state_dir=STATE_DIR, active_versions=active, season=SEASON,
    )
    realized = model_team.realize(con, STATE_DIR, season=SEASON)
    print(f"[model_team] advanced to GW{state['current_gameweek']}; {len(state['ledger'])} ledger rows; "
          f"realized {realized['realized']} newly-scored gameweeks")

    summary = model_team.build_summary(con, STATE_DIR, field_average_by_gw)
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    (DASHBOARD_DIR / "app_model_team.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[dashboard] {DASHBOARD_DIR / 'app_model_team.json'}")
    if summary.get("ready"):
        print(f"  total realised: {summary['total_realized_points']} pts over {summary['n_gameweeks_scored']} scored GWs; "
              f"vs field: {summary['total_vs_field']}")


if __name__ == "__main__":
    main()
