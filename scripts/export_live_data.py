"""Final-third roadmap item 1: writes data/dashboard/app_live_<entry_id>.json while real
fixtures are actually being played -- provisional bonus points and a sample-based live rank
estimate. Exits immediately (no writes, no extra fetches) when nothing is live right now, so the
tighter-cadence live_tracking.yml workflow that calls this every ~10 minutes across matchday
windows stays cheap outside actual play.

Roadmap Feature 6: also appends each account's rank estimate to an append-only
data/dashboard/live_rank_<entry_id>_<gw>.json snapshot series every time this runs while live
(see app_export.append_live_rank_snapshot()) -- the PWA's live rank trajectory line.
mini_league_pos is always None here: no specific mini-league is configured anywhere in this
project to track a position within, a real, disclosed gap rather than a fabricated number.
A live fetch that fails even after B7's retry/backoff falls back to that account's own
last-known snapshot, marked stale -- never a fabricated rank, and never crashes the whole
poll cycle over one account's fetch failure.

Usage (from repo root):
    PYTHONPATH=src python scripts/export_live_data.py <entry_id>:<label> [<entry_id>:<label> ...]

<event> isn't passed in -- unlike export_app_data.py/run_transfer_planner_for_real_squad.py, this
always operates on bootstrap-static's own current gameweek (there's no "read from a specific
past gameweek" use case for a live tracker).

Same network-blocked-in-sandbox caveat as every other live-fetch script in this project.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import app_export as ax  # noqa: E402
from fpl_quant import live_tracking as lt  # noqa: E402

DASHBOARD_DIR = REPO_ROOT / "data" / "dashboard"

# How many rival entries to sample for the live-rank estimate, centered on the account's own
# last-known overall rank (one standings page = 50 entries). Kept modest: this runs on a tight
# cadence during live windows, and ~50-100 extra fetches per poll is already the same order of
# magnitude as this project's other rival-sampling code (ingest_rival_squad_sample.py's own
# n_entries=200, but that runs once per gameweek, not every ~10 minutes).
RIVAL_SAMPLE_PAGES = 2


def _rival_sample_points(league_id: int, center_rank: int | None, event: int, live: dict) -> list[int]:
    page_center = max(1, round(center_rank / 50)) if center_rank else 1
    pages = sorted({max(1, page_center + delta) for delta in range(-(RIVAL_SAMPLE_PAGES // 2), RIVAL_SAMPLE_PAGES - RIVAL_SAMPLE_PAGES // 2)})
    sample_points = []
    for page in pages:
        standings = ax.fetch_league_standings(league_id, page)
        for r in standings.get("standings", {}).get("results", []):
            picks_payload = ax.fetch_entry_picks(r["entry"], event)
            if not picks_payload:
                continue
            rows = lt.build_live_squad_rows(picks_payload.get("picks", []), live)
            sample_points.append(lt.compute_live_squad_total(rows, provisional_bonus_by_id={})["total"])
    return sample_points


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: {sys.argv[0]} <entry_id>:<label> [...]")
    accounts = []
    for arg in sys.argv[1:]:
        entry_id, label = arg.split(":", 1)
        accounts.append((int(entry_id), label))

    bootstrap = ax.fetch_bootstrap_static()
    event = ax.current_event(bootstrap)
    if event is None:
        print("[skip] no current gameweek reported by bootstrap-static")
        return

    fixtures = ax.fetch_fixtures(event)
    if not lt.is_any_fixture_live(fixtures):
        print(f"[skip] no fixture is currently live for GW{event}")
        return

    print(f"[live] GW{event} has a live fixture -- fetching live data...")
    live = ax.fetch_event_live(event)
    provisional_bonus = lt.compute_provisional_bonus(bootstrap, fixtures, live)
    player_directory = ax.build_player_directory(bootstrap)

    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for entry_id, label in accounts:
        try:
            picks_payload = ax.fetch_entry_picks(entry_id, event)
            if not picks_payload:
                print(f"  [skip] entry_id={entry_id}: no picks recorded for GW{event} yet")
                continue
            entry_summary = ax.fetch_entry_summary(entry_id)

            team = ax.build_team_snapshot(entry_summary, picks_payload, live, player_directory)
            live_total = lt.compute_live_squad_total(team["squad"], provisional_bonus)

            sample_points = _rival_sample_points(
                ax.FPL_OVERALL_LEAGUE_ID, entry_summary.get("summary_overall_rank"), event, live,
            )
            rank_estimate = lt.estimate_live_rank(live_total["total"], sample_points, bootstrap.get("total_players"))
        except ax.UpstreamUnavailableError as e:
            # Review Feature 6 / B7: a live fetch failed even after retry/backoff -- fall back
            # to this account's own last-known rank (never fabricate one) and mark the
            # snapshot stale, rather than crashing the whole poll cycle over one account.
            print(f"  [stale] entry_id={entry_id}: live fetch failed ({e}) -- falling back to last-known rank")
            last = ax.last_known_rank_snapshot(DASHBOARD_DIR, entry_id, event)
            if last is None:
                print(f"  [skip] entry_id={entry_id}: no prior snapshot to fall back to -- nothing to append")
                continue
            ax.append_live_rank_snapshot(
                DASHBOARD_DIR, entry_id, event, ts=now_iso, overall_rank=last["overall_rank"],
                mini_league_pos=last["mini_league_pos"], live_points=last["live_points"],
                stale=True, data_asof=now_iso[:10],
            )
            continue

        out = {
            "entry_id": entry_id, "label": label, "gameweek": event,
            "schema_version": 2,
            "is_live": True,
            "live_total": live_total["total"], "players": live_total["players"],
            "rank_estimate": rank_estimate,
            "fixtures": lt.build_live_fixture_rows(fixtures, bootstrap),
            "events": lt.build_live_event_rows(bootstrap, live, fixtures),
            "generated_at": ax.generated_at(),
        }
        (DASHBOARD_DIR / f"app_live_{entry_id}.json").write_text(json.dumps(out, indent=2))
        print(f"  [write] app_live_{entry_id}.json (live_total={live_total['total']}, "
              f"rank_estimate sample_size={rank_estimate['sample_size']})")

        if rank_estimate["estimated_rank"] is not None:
            ax.append_live_rank_snapshot(
                DASHBOARD_DIR, entry_id, event, ts=now_iso, overall_rank=rank_estimate["estimated_rank"],
                mini_league_pos=None, live_points=live_total["total"], stale=False, data_asof=now_iso[:10],
            )
            print(f"  [live_rank] appended snapshot for entry_id={entry_id} (overall_rank={rank_estimate['estimated_rank']})")
        else:
            print(f"  [skip] entry_id={entry_id}: no rank estimate yet (sample_size={rank_estimate['sample_size']}) -- not appending")


if __name__ == "__main__":
    main()
