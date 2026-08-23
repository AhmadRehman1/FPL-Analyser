"""App conversion, Phase 1: writes the real, live JSON the 5-screen PWA (Home/Team/Transfers/
Fixtures/Leagues) reads at data/dashboard/app_*.json. Separate from run_ingestion.py's DuckDB
pipeline entirely -- see src/fpl_quant/app_export.py's own module docstring for why (this is a
pass-through/derivation of FPL's own live API, not a modeled output).

Usage (from repo root):
    PYTHONPATH=src python scripts/export_app_data.py <entry_id>:<event>:<label> [<entry_id>:<event>:<label> ...]

<event> is the gameweek each entry's CURRENT squad is set for -- same convention as
run_transfer_planner_for_real_squad.py. Multiple accounts share one bootstrap-static/fixtures
fetch (written once as app_players.json/app_fixtures.json); each account gets its own
app_team_<id>.json/app_profile_<id>.json/app_leagues_<id>.json.

Same network-blocked-in-sandbox caveat as run_transfer_planner_for_real_squad.py: only runs
somewhere with open internet (a real machine, or the scheduled GitHub Actions workflow).
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import app_export as ax  # noqa: E402

DASHBOARD_DIR = REPO_ROOT / "data" / "dashboard"

# FPL's own league_type code for a private/user-created classic league (a friend group), as
# opposed to "s" for a system-wide league (Overall, country leagues -- millions of entries,
# no meaningful standings table to fetch). A real, stable field on entry/<id>/'s own
# leagues.classic rows, not invented here.
USER_LEAGUE_TYPE = "x"
MAX_LEAGUES_PER_ACCOUNT = 5
LEAGUE_STANDINGS_PAGE_SIZE_PAGES = 1  # first page only (top ~50) -- enough for a "you + rivals" table


def _parse_account_arg(arg: str) -> tuple[int, int, str]:
    parts = arg.split(":", 2)
    if len(parts) != 3:
        raise SystemExit(f"bad account arg {arg!r}, expected entry_id:event:label")
    entry_id, event, label = parts
    return int(entry_id), int(event), label


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: {sys.argv[0]} <entry_id>:<event>:<label> [...]")
    accounts = [_parse_account_arg(a) for a in sys.argv[1:]]

    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

    print("[fetch] bootstrap-static + fixtures (shared across accounts)...")
    bootstrap = ax.fetch_bootstrap_static()
    fixtures = ax.fetch_fixtures()
    total_players = bootstrap.get("total_players")

    player_directory = ax.build_player_directory(bootstrap)
    (DASHBOARD_DIR / "app_players.json").write_text(json.dumps(
        {"generated_at": ax.generated_at(), "players": player_directory}, indent=2,
    ))
    print(f"[write] app_players.json ({len(player_directory)} players)")

    fixtures_out = ax.build_fixtures_by_gameweek(bootstrap, fixtures)
    fixtures_out["generated_at"] = ax.generated_at()
    (DASHBOARD_DIR / "app_fixtures.json").write_text(json.dumps(fixtures_out, indent=2))
    print(f"[write] app_fixtures.json ({len(fixtures_out['gameweeks'])} gameweeks)")

    for entry_id, event, label in accounts:
        print(f"\n[fetch] entry_id={entry_id} ({label})...")
        entry_summary = ax.fetch_entry_summary(entry_id)
        history = ax.fetch_entry_history(entry_id)
        picks_payload = ax.fetch_entry_picks(entry_id, event)
        if picks_payload is None:
            print(f"  [skip] no picks recorded yet for entry_id={entry_id} at event={event}")
            continue
        live_payload = ax.fetch_event_live(event)

        team = ax.build_team_snapshot(entry_summary, picks_payload, live_payload, player_directory)
        team["label"] = label
        team["free_transfers"] = ax.compute_free_transfers(history.get("current", []), history.get("chips", []))
        team["generated_at"] = ax.generated_at()
        (DASHBOARD_DIR / f"app_team_{entry_id}.json").write_text(json.dumps(team, indent=2))
        print(f"  [write] app_team_{entry_id}.json")

        profile = ax.build_profile(entry_summary, history)
        profile["label"] = label
        profile["generated_at"] = ax.generated_at()
        (DASHBOARD_DIR / f"app_profile_{entry_id}.json").write_text(json.dumps(profile, indent=2))
        print(f"  [write] app_profile_{entry_id}.json")

        user_leagues = [
            league for league in entry_summary.get("leagues", {}).get("classic", [])
            if league.get("league_type") == USER_LEAGUE_TYPE
        ][:MAX_LEAGUES_PER_ACCOUNT]
        standings_by_league = {
            league["id"]: ax.fetch_league_standings(league["id"])
            for league in user_leagues
        }
        leagues = ax.build_leagues(entry_summary, standings_by_league, total_players)
        leagues["label"] = label
        leagues["generated_at"] = ax.generated_at()
        (DASHBOARD_DIR / f"app_leagues_{entry_id}.json").write_text(json.dumps(leagues, indent=2))
        print(f"  [write] app_leagues_{entry_id}.json ({len(user_leagues)} user leagues fetched)")


if __name__ == "__main__":
    main()
