"""Roadmap Feature 1: per-gameweek point-projection table + captaincy ranking with
confidence bands. Writes data/dashboard/projections_<data_asof>.json for the PWA.

Needs a real scripts/run_ingestion.py run to already have happened (uses the latest real
team_strength_model_version/minutes_model_version it left behind, the same "gameweek-agnostic
snapshot" ts/mm pair transfer_planner.compute_horizon_ep() reuses across the whole horizon).

Usage (from repo root):
    PYTHONPATH=src python scripts/export_projections.py [start_gameweek] [n_gameweeks]

Defaults to a single-gameweek table for TARGET_GAMEWEEK if no arguments are given.
"""

import dataclasses
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import db, projections as proj  # noqa: E402

TARGET_SEASON = "2026-2027"
TARGET_GAMEWEEK = 1
DASHBOARD_DIR = REPO_ROOT / "data" / "dashboard"

# Every one of these is currently seeded at version=1 by scripts/run_ingestion.py -- kept
# explicit here (not re-derived) to match this project's own no-invented-defaults discipline.
PARAM_VERSIONS = dict(
    scoring_params_version=1, bps_params_version=1, tau_params_version=1,
    rho_residual_params_version=2, corr_params_version=1,
)


def _latest_model_version(con, table: str) -> int:
    row = con.execute(f"SELECT max(model_version) FROM {table}").fetchone()
    if row is None or row[0] is None:
        raise SystemExit(f"no rows in {table} -- run scripts/run_ingestion.py first")
    return row[0]


def main() -> None:
    start_gw = int(sys.argv[1]) if len(sys.argv) > 1 else TARGET_GAMEWEEK
    n_gameweeks = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    gameweeks = list(range(start_gw, start_gw + n_gameweeks))

    con = db.connect()
    ts_mv = _latest_model_version(con, "team_strength_model_versions")
    mm_mv = _latest_model_version(con, "minutes_model_versions")

    rows = proj.build_projections(
        con, calibration_asof_date=date.today(), target_season=TARGET_SEASON, gameweeks=gameweeks,
        ts_model_version=ts_mv, mm_model_version=mm_mv, **PARAM_VERSIONS,
    )
    captain_ranking = proj.build_captain_ranking(rows, gw=gameweeks[0])

    data_asof = date.today().isoformat()
    payload = {
        "data_asof": data_asof,
        "model_version": rows[0].provenance.model_version if rows else None,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "captain_ranking": captain_ranking,
        "players": [dataclasses.asdict(r) for r in rows],
    }

    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DASHBOARD_DIR / f"projections_{data_asof}.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[export_projections] {len(rows)} players, gameweeks={gameweeks} -> wrote {out_path}")
    if captain_ranking:
        top = captain_ranking[0]
        print(f"  captain: {top['name']} (ep={top['ep']:.2f}, [{top['ci_low']:.2f}, {top['ci_high']:.2f}])")

    con.close()


if __name__ == "__main__":
    main()
