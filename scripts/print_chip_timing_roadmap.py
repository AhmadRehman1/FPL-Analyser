"""First-half-of-season chip-timing roadmap for both real FPL accounts this project tracks,
computed from fixture_swing.py's real rolling swing scores (Dixon-Coles fixture-difficulty
deltas from THIS run's real ts_model_version, not invented dates).

For each gameweek in FIRST_HALF_GAMEWEEKS, averages each account's own squad's teams' swing
scores (negative = an easier-than-average near-term run for that team, per fixture_swing.py's
own sign convention) into one number for that gameweek -- the single easiest window across the
first half is flagged as a bench_boost/triple_captain candidate (several of the squad's teams
enjoying good fixtures at once); the toughest is flagged as a wildcard-consideration window
(several teams hitting a difficult patch simultaneously). A real, computed signal from today's
actual squad composition, not a locked-in plan -- transfers made along the way will change
which teams are actually in the squad by the time any of these gameweeks arrive.

Needs the same freshly-ingested database run_ingestion.py just built (same real-network-access
caveat as run_transfer_planner_for_real_squad.py's own module docstring: only runs somewhere
with open internet, not this project's own dev sandbox).

Usage (from repo root, same job as run_ingestion.py):
    PYTHONPATH=src python scripts/print_chip_timing_roadmap.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import db, fixture_swing as fs, ingest_fpl_entry_picks as ifp, ingest_workbook as iw  # noqa: E402

TARGET_SEASON = "2026-2027"
TS_MODEL_VERSION = 1
# GW1-2 already have their own real, computed transfer plan (run_transfer_planner_for_real_squad.py);
# this roadmap covers the rest of the first half. 19 is a reasonable midpoint for a 38-gameweek
# season -- not FPL's own real confirmed winter-break gameweek for 2026-27, which this project
# doesn't have visibility into beyond what's already in its ingested fixture data.
FIRST_HALF_GAMEWEEKS = range(3, 20)
DASHBOARD_DIR = REPO_ROOT / "data" / "dashboard"

ACCOUNTS = [
    {"entry_id": 7139944, "event": 1, "label": "ChatGPT template team"},
    {"entry_id": 1305242, "event": 1, "label": "Main account"},
]


def _account_team_uids(con, entry_id: int, event: int, team_by_player: dict[str, str]) -> set[str]:
    element_names = ifp.fetch_bootstrap_elements()
    picks = ifp.fetch_entry_picks(entry_id, event)
    if not picks:
        return set()
    uids = set()
    for p in picks:
        name = element_names.get(p["element"])
        player_uid = iw._resolve_player(con, name, TARGET_SEASON)
        if player_uid and player_uid in team_by_player:
            uids.add(team_by_player[player_uid])
    return uids


def main() -> None:
    con = db.connect()
    team_names = {r[0]: r[1] for r in con.execute("SELECT team_uid, canonical_name FROM dim_team").fetchall()}
    team_by_player = fs.team_uid_by_player(con, TARGET_SEASON)

    roadmap = {
        "target_season": TARGET_SEASON,
        "first_half_gameweeks": [FIRST_HALF_GAMEWEEKS.start, FIRST_HALF_GAMEWEEKS.stop - 1],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "accounts": [],
    }

    for account in ACCOUNTS:
        print(f"[chip_timing] {account['label']} (entry_id={account['entry_id']})...")
        squad_team_uids = _account_team_uids(con, account["entry_id"], account["event"], team_by_player)
        print(f"  {len(squad_team_uids)} distinct clubs in squad")

        weekly = []
        for gw in FIRST_HALF_GAMEWEEKS:
            scores = fs.swing_scores_by_team(con, TARGET_SEASON, gw, TS_MODEL_VERSION)
            squad_swings = [
                scores[t].swing_score for t in squad_team_uids
                if t in scores and scores[t].swing_score is not None
            ]
            avg_swing = sum(squad_swings) / len(squad_swings) if squad_swings else None
            weekly.append({
                "gameweek": gw,
                "avg_squad_swing": round(avg_swing, 3) if avg_swing is not None else None,
                "n_teams_with_data": len(squad_swings),
            })
            print(f"  GW{gw}: avg_squad_swing={weekly[-1]['avg_squad_swing']} ({weekly[-1]['n_teams_with_data']} teams)")

        valid = [w for w in weekly if w["avg_squad_swing"] is not None]
        best_window = min(valid, key=lambda w: w["avg_squad_swing"]) if valid else None
        worst_window = max(valid, key=lambda w: w["avg_squad_swing"]) if valid else None

        roadmap["accounts"].append({
            "entry_id": account["entry_id"],
            "label": account["label"],
            "squad_clubs": sorted(team_names.get(t, t) for t in squad_team_uids),
            "weekly_swing": weekly,
            "best_bench_boost_triple_captain_window": best_window,
            "toughest_window_wildcard_candidate": worst_window,
        })

    con.close()

    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DASHBOARD_DIR / "chip_timing_roadmap.json"
    out_path.write_text(json.dumps(roadmap, indent=2))
    print(f"\n[chip_timing] roadmap written to {out_path}")


if __name__ == "__main__":
    main()
