"""App conversion, Phase 1: real, live data for the 5-screen PWA (Home/Team/Transfers/Fixtures/
Leagues) that scripts/export_app_data.py writes to data/dashboard/. Deliberately separate from
the M0-M9 quant pipeline and its DuckDB store -- everything here is either a direct pass-through
of FPL's own public API (player directory, fixtures, entry summary/history, live gameweek
points, classic league standings) or a small, real, documented derivation of it (free transfers
remaining). No modeled/estimated numbers live in this module; those come from the existing
real_squad_<entry_id>.json / chip_timing_roadmap.json snapshots the quant pipeline already
produces, which the frontend reads separately.

Same real limitation as ingest_fpl_entry_picks.py's own module docstring: this sandbox's network
policy blocks fantasy.premierleague.com entirely, so the shapes assumed here come from that API's
own long-stable public documentation and the many open-source FPL tools built against it, not an
independently verified live fetch in THIS environment. Every fetch_* function isolates the one
`requests.get` call and accepts an injectable `payload=`/`pages=` for tests, matching this
project's established convention (ingest_understat.py, ingest_fpl_entry_picks.py).
"""

from datetime import datetime, timezone
from typing import Any

import requests

FPL_API_BASE = "https://fantasy.premierleague.com/api"

POSITION_LABELS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

# FPL's own chip "name" codes (as returned by entry/history's chips[].name and
# entry/event/picks' active_chip) -- a stable, publicly documented set, not invented here.
CHIP_DISPLAY_NAMES = {
    "wildcard": "Wildcard",
    "freehit": "Free Hit",
    "3xc": "Triple Captain",
    "bboost": "Bench Boost",
}

# FPL's own well-known "Overall" classic league id (see ingest_fpl_entry_picks.py's own
# FPL_OVERALL_LEAGUE_ID) -- excluded from the per-league standings fetch (tens of millions of
# entries, no meaningful "top of the table" to show) but still surfaced via the entry's own
# summary_overall_rank as a rank-of-N-million tile.
FPL_OVERALL_LEAGUE_ID = 314


# ============================================================
# fetch -- isolated network I/O, the one surface real verification and test mocking touch
# ============================================================

def _fetch_json(url: str) -> Any:
    """Return type is genuinely payload-dependent (an object for most endpoints, an array for
    /fixtures/) -- callers annotate their own expected shape (see fetch_fixtures below)."""
    resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    return resp.json()


def fetch_bootstrap_static(*, payload: dict | None = None) -> dict:
    if payload is None:
        payload = _fetch_json(f"{FPL_API_BASE}/bootstrap-static/")
    return payload


def fetch_fixtures(event: int | None = None, *, payload: list[dict] | None = None) -> list[dict]:
    if payload is None:
        url = f"{FPL_API_BASE}/fixtures/" + (f"?event={event}" if event is not None else "")
        payload = _fetch_json(url)
    return payload


def fetch_entry_summary(entry_id: int, *, payload: dict | None = None) -> dict:
    if payload is None:
        payload = _fetch_json(f"{FPL_API_BASE}/entry/{entry_id}/")
    return payload


def fetch_entry_history(entry_id: int, *, payload: dict | None = None) -> dict:
    if payload is None:
        payload = _fetch_json(f"{FPL_API_BASE}/entry/{entry_id}/history/")
    return payload


def fetch_entry_picks(entry_id: int, event: int, *, payload: dict | None = None) -> dict | None:
    """The full picks payload (picks + entry_history + active_chip), unlike
    ingest_fpl_entry_picks.fetch_entry_picks() which only keeps the picks list for the quant
    pipeline's own narrower need. None (not {}) on a real 404 -- same "no picks recorded yet"
    distinction that module's docstring explains."""
    if payload is None:
        try:
            payload = _fetch_json(f"{FPL_API_BASE}/entry/{entry_id}/event/{event}/picks/")
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                return None
            raise
    return payload


def fetch_event_live(event: int, *, payload: dict | None = None) -> dict:
    if payload is None:
        payload = _fetch_json(f"{FPL_API_BASE}/event/{event}/live/")
    return payload


def fetch_league_standings(league_id: int, page: int = 1, *, payload: dict | None = None) -> dict:
    if payload is None:
        payload = _fetch_json(f"{FPL_API_BASE}/leagues-classic/{league_id}/standings/?page_standings={page}")
    return payload


# ============================================================
# shared build -- player directory + fixtures, same for every account
# ============================================================

def build_player_directory(bootstrap: dict) -> list[dict]:
    """One row per real FPL player, current-season totals only (no history) -- the source for
    both the Transfers screen's browse/search list and the Player Stats detail sheet's header
    stats. Everything here is a direct bootstrap-static field, renamed/rescaled (now_cost is
    tenths of a million), never derived or estimated. news/news_added are the game's own official
    injury/suspension blurb (e.g. "Knock - 75% chance of playing"), not scraped or guessed."""
    teams_by_id = {t["id"]: t["short_name"] for t in bootstrap.get("teams", [])}
    team_names_by_id = {t["id"]: t["name"] for t in bootstrap.get("teams", [])}
    out = []
    for e in bootstrap.get("elements", []):
        out.append({
            "id": e["id"],
            "web_name": e.get("web_name", ""),
            "full_name": f"{e.get('first_name', '')} {e.get('second_name', '')}".strip(),
            "team": teams_by_id.get(e.get("team")),
            "team_name": team_names_by_id.get(e.get("team")),
            "position": POSITION_LABELS.get(e.get("element_type")),
            "price": round(e.get("now_cost", 0) / 10, 1),
            "total_points": e.get("total_points", 0),
            "event_points": e.get("event_points", 0),
            "form": float(e["form"]) if e.get("form") not in (None, "") else None,
            "points_per_game": float(e["points_per_game"]) if e.get("points_per_game") not in (None, "") else None,
            "selected_by_percent": float(e["selected_by_percent"]) if e.get("selected_by_percent") not in (None, "") else None,
            "minutes": e.get("minutes", 0),
            "goals_scored": e.get("goals_scored", 0),
            "assists": e.get("assists", 0),
            "clean_sheets": e.get("clean_sheets", 0),
            "goals_conceded": e.get("goals_conceded", 0),
            "bonus": e.get("bonus", 0),
            "yellow_cards": e.get("yellow_cards", 0),
            "red_cards": e.get("red_cards", 0),
            "ict_index": float(e["ict_index"]) if e.get("ict_index") not in (None, "") else None,
            "status": e.get("status"),
            "chance_of_playing_next_round": e.get("chance_of_playing_next_round"),
            "news": e.get("news") or None,
            "news_added": e.get("news_added"),
            "transfers_in_event": e.get("transfers_in_event", 0),
            "transfers_out_event": e.get("transfers_out_event", 0),
        })
    return out


def build_fixtures_by_gameweek(bootstrap: dict, fixtures: list[dict]) -> dict:
    """Every real scheduled fixture, grouped by gameweek then (client-side) by kickoff date.
    team_h_difficulty/team_a_difficulty are FPL's own published FDR (1-5), not modeled here --
    this project's own fixture_swing.py signal is a separate, model-derived thing surfaced on
    the chip-timing roadmap card, not conflated with this real/official number."""
    teams_by_id = {t["id"]: {"name": t["name"], "short_name": t["short_name"]} for t in bootstrap.get("teams", [])}
    events_by_id = {ev["id"]: ev for ev in bootstrap.get("events", [])}

    by_gw: dict[int, list[dict]] = {}
    for f in fixtures:
        gw = f.get("event")
        if gw is None:
            continue
        home = teams_by_id.get(f.get("team_h"), {})
        away = teams_by_id.get(f.get("team_a"), {})
        by_gw.setdefault(gw, []).append({
            "id": f["id"],
            "kickoff_time": f.get("kickoff_time"),
            "finished": f.get("finished", False),
            "home": {
                "name": home.get("name"), "short_name": home.get("short_name"),
                "difficulty": f.get("team_h_difficulty"), "score": f.get("team_h_score"),
            },
            "away": {
                "name": away.get("name"), "short_name": away.get("short_name"),
                "difficulty": f.get("team_a_difficulty"), "score": f.get("team_a_score"),
            },
        })

    gameweeks = []
    for gw in sorted(by_gw):
        ev = events_by_id.get(gw, {})
        gameweeks.append({
            "gameweek": gw,
            "deadline_time": ev.get("deadline_time"),
            "is_current": ev.get("is_current", False),
            "fixtures": sorted(by_gw[gw], key=lambda x: x["kickoff_time"] or ""),
        })
    return {"gameweeks": gameweeks}


def build_price_watch(player_directory: list[dict], top_n: int = 8) -> dict:
    """Today's biggest real transfer-momentum movers -- net transfers in/out this event, straight
    from bootstrap-static. Deliberately NOT a price-change prediction: FPL has never published
    its price-change algorithm, so any specific "will rise/fall" threshold would be an unverified
    guess dressed up as a fact. This only ever reports the real signal (who the field is actually
    buying/selling right now) and leaves the call to the reader."""
    with_momentum = [
        {**p, "net_transfers": p["transfers_in_event"] - p["transfers_out_event"]}
        for p in player_directory
        if p["transfers_in_event"] or p["transfers_out_event"]
    ]
    risers = sorted((p for p in with_momentum if p["net_transfers"] > 0), key=lambda p: -p["net_transfers"])[:top_n]
    fallers = sorted((p for p in with_momentum if p["net_transfers"] < 0), key=lambda p: p["net_transfers"])[:top_n]
    keep = ("id", "web_name", "team", "team_name", "position", "price", "selected_by_percent", "net_transfers")
    return {
        "risers": [{k: p[k] for k in keep} for p in risers],
        "fallers": [{k: p[k] for k in keep} for p in fallers],
    }


# ============================================================
# free transfers -- a real, publicly documented FPL rule (not a model estimate): 1 FT is
# granted for GW2 onward, unused FT carries over (capped at 5), a wildcard/free-hit gameweek's
# transfers don't consume or reset the count. Computed from the entry's own real transfer
# history, not assumed -- unlike transfer_planner.bootstrap_from_real_squad()'s deliberate
# free_transfers_available=1 placeholder (see that function's own docstring), this is the
# manager's actual current count.
# ============================================================

def compute_free_transfers(history_current: list[dict], chips: list[dict]) -> int:
    chip_events = {c["event"] for c in chips if c.get("name") in ("wildcard", "freehit")}
    available = 1
    for gw_row in sorted(history_current, key=lambda r: r["event"]):
        event = gw_row["event"]
        if event in chip_events:
            # a wildcard/free-hit gameweek's transfers are free and don't touch the count
            available = min(available + 1, 5)
            continue
        used = gw_row.get("event_transfers", 0) or 0
        available = min(max(available - used, 0) + 1, 5)
    return available


# ============================================================
# per-account build -- Team, Home, Profile, Leagues
# ============================================================

def build_team_snapshot(
    entry_summary: dict, picks_payload: dict, live_payload: dict, player_directory: list[dict],
) -> dict:
    """The 'My Team' pitch view + Home's team summary: real current picks, each player's
    actual live points for this gameweek (from event/<gw>/live/, not the model), captain/vice,
    bank/value/points from the picks payload's own entry_history block."""
    players_by_id = {p["id"]: p for p in player_directory}
    live_by_id = {el["id"]: el.get("stats", {}) for el in live_payload.get("elements", [])}
    hist = picks_payload.get("entry_history", {}) or {}

    squad = []
    for pick in picks_payload.get("picks", []):
        player = players_by_id.get(pick["element"], {})
        stats = live_by_id.get(pick["element"], {})
        squad.append({
            "player_id": pick["element"],
            "web_name": player.get("web_name"),
            "team": player.get("team"),
            "team_name": player.get("team_name"),
            "position": player.get("position"),
            "price": player.get("price"),
            "squad_position": pick["position"],
            "in_xi": pick["position"] <= 11,
            "multiplier": pick.get("multiplier", 1),
            "is_captain": bool(pick.get("is_captain")),
            "is_vice_captain": bool(pick.get("is_vice_captain")),
            "event_points": stats.get("total_points", 0),
            "minutes": stats.get("minutes", 0),
        })

    return {
        "entry_id": entry_summary.get("id"),
        "team_name": entry_summary.get("name"),
        "manager_name": f"{entry_summary.get('player_first_name', '')} {entry_summary.get('player_last_name', '')}".strip(),
        "gameweek": hist.get("event"),
        "gameweek_points": hist.get("points"),
        "gameweek_points_on_bench": hist.get("points_on_bench"),
        "bank": round((hist.get("bank") or 0) / 10, 1),
        "team_value": round((hist.get("value") or 0) / 10, 1),
        "active_chip": picks_payload.get("active_chip"),
        "squad": squad,
    }


def build_profile(entry_summary: dict, history: dict) -> dict:
    """Real season summary + chip usage. Chips not present in history['chips'] are reported as
    'available' -- a fact about what's been used so far, not a claim about this season's total
    chip allowance (which changes by rule and isn't asserted here)."""
    current = history.get("current", [])
    chips_used = history.get("chips", [])
    used_names = {c["name"] for c in chips_used}
    best_gw = max(current, key=lambda r: r.get("points", 0), default=None)
    best_rank = min(
        (r["overall_rank"] for r in current if r.get("overall_rank") is not None),
        default=None,
    )

    chips = []
    for name, display in CHIP_DISPLAY_NAMES.items():
        used = next((c for c in chips_used if c["name"] == name), None)
        chips.append({
            "chip_type": name,
            "display_name": display,
            "used": name in used_names,
            "used_gameweek": used["event"] if used else None,
        })

    recent_history = [
        {
            "gameweek": r["event"], "points": r.get("points"),
            "overall_rank": r.get("overall_rank"), "rank": r.get("rank"),
        }
        for r in current[-6:]
    ]

    return {
        "entry_id": entry_summary.get("id"),
        "team_name": entry_summary.get("name"),
        "manager_name": f"{entry_summary.get('player_first_name', '')} {entry_summary.get('player_last_name', '')}".strip(),
        "total_points": entry_summary.get("summary_overall_points"),
        "overall_rank": entry_summary.get("summary_overall_rank"),
        "best_gameweek": {"gameweek": best_gw["event"], "points": best_gw["points"]} if best_gw else None,
        "best_overall_rank": best_rank,
        "chips": chips,
        "recent_history": recent_history,
    }


def build_league_ownership(
    standings_payload: dict, entry_picks_by_id: dict[int, dict | None], player_directory: list[dict],
) -> dict:
    """Real ownership/captaincy intelligence for one private league -- who's captaining what and
    who owns whom, computed directly from each rival's own real picks for this gameweek (public
    information the game itself already shows for any entry; the same category of fetch
    ingest_fpl_entry_picks.py already does at global-sample scale, just pointed at one private
    league's own much smaller entry list instead of a 200-entry sample of the Overall league).

    entry_picks_by_id: {entry_id: picks_payload or None} -- a rival with no picks yet for this
    event is silently excluded from the count, the same "no picks recorded" distinction
    fetch_entry_picks() draws elsewhere; this keeps pct_of_league honest (denominator is who was
    actually counted, not the league's full roster)."""
    players_by_id = {p["id"]: p for p in player_directory}
    results = standings_payload.get("standings", {}).get("results", [])

    ownership_counts: dict[int, int] = {}
    captains = []
    n_counted = 0
    for r in results:
        payload = entry_picks_by_id.get(r["entry"])
        if not payload:
            continue
        n_counted += 1
        for pick in payload.get("picks", []):
            if pick.get("multiplier", 0) > 0:
                ownership_counts[pick["element"]] = ownership_counts.get(pick["element"], 0) + 1
            if pick.get("is_captain"):
                captains.append({
                    "entry_id": r["entry"], "entry_name": r.get("entry_name"), "player_name": r.get("player_name"),
                    "player_id": pick["element"], "web_name": players_by_id.get(pick["element"], {}).get("web_name"),
                })

    top_owned = sorted(ownership_counts.items(), key=lambda item: -item[1])[:10]
    most_owned = [
        {
            "player_id": pid, "web_name": players_by_id.get(pid, {}).get("web_name"), "n_owners": n,
            "pct_of_league": round(n / n_counted * 100, 1) if n_counted else None,
        }
        for pid, n in top_owned
    ]

    return {"n_entries_sampled": n_counted, "most_owned": most_owned, "captains": captains}


def build_leagues(
    entry_summary: dict, standings_by_league: dict[int, dict], total_players: int | None,
    ownership_by_league: dict[int, dict] | None = None,
) -> dict:
    """Every real classic league the manager belongs to (from their own entry summary), plus
    a real standings table for whichever of those leagues were fetched (standings_by_league --
    the caller decides which, typically excluding the global Overall league). The Overall league
    itself is still surfaced as a rank/of-N-million tile using entry_summary's own numbers,
    without fetching its (multi-million-row) standings. ownership_by_league (optional): each
    fetched league's build_league_ownership() result, attached onto its table."""
    classic = entry_summary.get("leagues", {}).get("classic", [])
    tiles = []
    tables = []
    for league in classic:
        league_id = league["id"]
        if league_id == FPL_OVERALL_LEAGUE_ID:
            tiles.append({
                "league_id": league_id, "name": league.get("name"),
                "rank": entry_summary.get("summary_overall_rank"), "total_entries": total_players,
            })
            continue
        tiles.append({
            "league_id": league_id, "name": league.get("name"),
            "rank": league.get("entry_rank") or league.get("entry_last_rank"),
            "total_entries": None,
        })
        standings_payload = standings_by_league.get(league_id)
        if not standings_payload:
            continue
        results = standings_payload.get("standings", {}).get("results", [])
        tables.append({
            "league_id": league_id,
            "name": league.get("name"),
            "standings": [
                {
                    "rank": r["rank"], "last_rank": r.get("last_rank"),
                    "entry_id": r["entry"], "entry_name": r.get("entry_name"),
                    "player_name": r.get("player_name"), "total": r.get("total"),
                    "is_you": r["entry"] == entry_summary.get("id"),
                }
                for r in results
            ],
            "ownership": (ownership_by_league or {}).get(league_id),
        })
    return {"tiles": tiles, "tables": tables}


def generated_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def current_event(bootstrap: dict) -> int | None:
    """The real gameweek FPL itself currently flags as in-progress (bootstrap-static's own
    events[].is_current) -- None if the season hasn't started yet or every event is finished.
    Shared by export_live_data.py and the on-demand per-manager workflow, which both need "what
    gameweek is a manager's CURRENT squad set for" without a hardcoded number."""
    for ev in bootstrap.get("events", []):
        if ev.get("is_current"):
            return ev["id"]
    return None
