"""Roadmap Feature 1: per-gameweek point-projection table + captaincy ranking with
confidence bands. Writes data/dashboard/projections_<data_asof>.json for the PWA.

Needs a real scripts/run_ingestion.py run to already have happened (uses the latest real
team_strength_model_version/minutes_model_version it left behind, the same "gameweek-agnostic
snapshot" ts/mm pair transfer_planner.compute_horizon_ep() reuses across the whole horizon).

Usage (from repo root):
    PYTHONPATH=src python scripts/export_projections.py [start_gameweek] [n_gameweeks]

Defaults to a PLANNER_HORIZON_GAMEWEEKS-wide table starting at bootstrap-static's own current
gameweek if no arguments are given -- wide enough for the PWA's own future-gameweek planner
(see index.html's Planner sheet) to have real data to navigate, matching the ~5-8 gameweek
forward window real FPL chip-timing strategy guidance itself uses (a wildcard's payoff is
judged over "the next five to eight gameweeks," not a single week -- see the roadmap's own
chip-timing notes).

Also resolves each player's real FPL bootstrap-static element id (a live fetch, see
_resolve_element_ids()'s own docstring for why) -- projections.py's own player_uid is this
project's internal identity, meaningless to the PWA on its own, which is built entirely around
FPL's numeric element id (see app_export.py's own module docstring on why those are two
different identity spaces). The live fetch is best-effort: a failure is caught and disclosed
(fpl_element_id left absent on every row, never guessed) rather than blocking the whole export,
since projections are still useful without it -- only the planner's squad-join needs the id.
"""

import dataclasses
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import app_export as ax  # noqa: E402
from fpl_quant import db, ingest_fpl_entry_picks as ifp, projections as proj  # noqa: E402

TARGET_SEASON = "2026-2027"
TARGET_GAMEWEEK = 1
PLANNER_HORIZON_GAMEWEEKS = 8
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


def _fetch_element_names() -> dict[int, str]:
    """Isolated live fetch (see module docstring) -- best-effort: any failure (this sandbox's
    own network policy blocks fantasy.premierleague.com, same caveat as every other real-fetch
    script here) returns {} rather than raising, since the projections table itself is still
    real and useful without fpl_element_id resolved -- only the PWA's planner squad-join needs it."""
    try:
        return ifp.fetch_bootstrap_elements()
    except Exception as e:  # noqa: BLE001 -- best-effort: a live-fetch failure must not block the whole export
        print(f"::warning::export_projections: bootstrap-static fetch failed ({e}) -- fpl_element_id omitted from every row.")
        return {}


def main() -> None:
    con = db.connect()

    if len(sys.argv) > 1:
        start_gw = int(sys.argv[1])
    else:
        try:
            start_gw = ax.current_event(ax.fetch_bootstrap_static()) or TARGET_GAMEWEEK
        except Exception as e:  # noqa: BLE001 -- best-effort default; an explicit CLI arg always overrides this
            print(f"::warning::export_projections: could not determine the current gameweek live ({e}) -- defaulting to GW{TARGET_GAMEWEEK}.")
            start_gw = TARGET_GAMEWEEK
    n_gameweeks = int(sys.argv[2]) if len(sys.argv) > 2 else PLANNER_HORIZON_GAMEWEEKS
    gameweeks = list(range(start_gw, start_gw + n_gameweeks))

    ts_mv = _latest_model_version(con, "team_strength_model_versions")
    mm_mv = _latest_model_version(con, "minutes_model_versions")

    rows = proj.build_projections(
        con, calibration_asof_date=date.today(), target_season=TARGET_SEASON, gameweeks=gameweeks,
        ts_model_version=ts_mv, mm_model_version=mm_mv, **PARAM_VERSIONS,
    )
    captain_ranking = proj.build_captain_ranking(rows, gw=gameweeks[0])
    element_id_by_uid = proj.resolve_element_ids(con, TARGET_SEASON, _fetch_element_names())

    data_asof = date.today().isoformat()
    player_rows = []
    for r in rows:
        row = dataclasses.asdict(r)
        row["fpl_element_id"] = element_id_by_uid.get(r.player_uid)
        player_rows.append(row)
    payload = {
        "data_asof": data_asof,
        "model_version": rows[0].provenance.model_version if rows else None,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gameweeks": gameweeks,
        "captain_ranking": captain_ranking,
        "players": player_rows,
    }

    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DASHBOARD_DIR / f"projections_{data_asof}.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    # A dated file alone can't be discovered by a static-site PWA that doesn't know today's
    # date in advance -- also write a stable-named copy at a fixed, predictable path, same
    # "always-overwritten" convention as app_team_<id>.json/real_squad_<id>.json. The dated
    # file is kept too, as the historical/audit record.
    (DASHBOARD_DIR / "projections_latest.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[export_projections] {len(rows)} players, gameweeks={gameweeks} -> wrote {out_path}")
    if captain_ranking:
        top = captain_ranking[0]
        print(f"  captain: {top['name']} (ep={top['ep']:.2f}, [{top['ci_low']:.2f}, {top['ci_high']:.2f}])")

    con.close()


if __name__ == "__main__":
    main()
