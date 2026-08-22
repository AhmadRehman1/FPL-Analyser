"""Priority 10 Phase A: a real, bounded sample of rival squads from FPL's own public API --
see docs/priority10_field_simulator_design.md for the full design (Phase A only is built
here; the joint-simulation engine (Phase B) and rank-distribution output (Phase C) are
explicitly NOT part of this module).

Unlike every other "field/rival" signal built in this project so far (EO, captaincy-
concentration, field-covariance), this doesn't need to model or estimate anything -- FPL's own
public API (leagues-classic standings + entry picks) exposes real individual managers' picks
directly. No auth required; per-entry-per-gameweek picks are public information the game
itself displays.

Deliberately minimal fields stored: entry_id (an opaque numeric id, needed only for row
identity/dedup), player picks, captain flag, and public league rank at sample time -- no
manager name, team name, or any other personally-identifying field, even though the source API
exposes them. See fact_rival_squad_sample's own schema comment.

Same real limitation as Understat (Priority 7a): this sandbox's own network policy blocks
fantasy.premierleague.com entirely (confirmed via a direct connection attempt -- 403 policy
denial from the outbound proxy, not a guess), so the real API response shapes assumed here
come from this API's own extensive, years-stable public documentation and the many open-source
FPL tools built against it, NOT independently verified against a live fetch in THIS
environment. _fetch_json() is isolated specifically so verification is a one-line swap; every
other function here is exercised by this module's own tests against hand-built fixtures
faithful to the documented real shapes.

Deliberately NOT wired into scripts/run_ingestion.py's default flow: sampling even a modest
n_entries means that many real HTTP requests to FPL's own API per gameweek, which needs real
rate-limiting/caching discipline (fetch once per gameweek, never re-fetch on every report run)
that a bare "run on every ingestion" wiring wouldn't respect. A separate, deliberately
lower-cadence caller is the right shape for this -- see the design doc's Phase A framing.
"""

from datetime import datetime

import duckdb
import requests

from . import ingest_workbook as iw

FPL_API_BASE = "https://fantasy.premierleague.com/api"
# FPL's own well-known "Overall" classic league id -- not invented here, a stable public
# constant widely used across the FPL API community (fplreview, the various open-source `fpl`
# Python packages, etc. all key off 314 for the game-wide Overall league).
FPL_OVERALL_LEAGUE_ID = 314


# ============================================================
# fetch -- isolated network I/O, the one surface real verification and test mocking touch
# ============================================================

def _fetch_json(url: str) -> dict:
    resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    return resp.json()


# ============================================================
# bootstrap-static: element id -> the name string _resolve_player() matches against
# ============================================================

def fetch_bootstrap_elements(*, payload: dict | None = None) -> dict[int, str]:
    """payload can be injected directly (tests do this, matching ingest_understat.py's own
    html= injection convention) instead of fetching live."""
    if payload is None:
        payload = _fetch_json(f"{FPL_API_BASE}/bootstrap-static/")
    return {
        e["id"]: f"{e.get('first_name', '')} {e.get('second_name', '')}".strip()
        for e in payload.get("elements", [])
    }


# ============================================================
# leagues-classic standings: top-N entries by rank, paginated
# ============================================================

def fetch_top_entries(league_id: int, n_entries: int, *, pages: list[dict] | None = None) -> list[dict]:
    """Top n_entries by rank from a classic league's standings, paginating the real API's own
    fixed-size-per-page shape. pages can be injected directly for tests instead of fetching
    live. Stops early once n_entries is reached or the league genuinely runs out of pages --
    never over-fetches beyond what was asked for."""
    entries: list[dict] = []
    page_num = 1
    while len(entries) < n_entries:
        if pages is not None:
            if page_num > len(pages):
                break
            page = pages[page_num - 1]
        else:
            page = _fetch_json(f"{FPL_API_BASE}/leagues-classic/{league_id}/standings/?page_standings={page_num}")
        results = page.get("standings", {}).get("results", [])
        if not results:
            break
        for r in results:
            entries.append({"entry_id": r["entry"], "rank": r["rank"]})
            if len(entries) >= n_entries:
                break
        if not page.get("standings", {}).get("has_next"):
            break
        page_num += 1
    return entries


# ============================================================
# entry picks for one gameweek
# ============================================================

def fetch_entry_picks(entry_id: int, event: int, *, payload: dict | None = None) -> list[dict] | None:
    """None (not []) when this entry has no picks recorded for this event yet (e.g. joined the
    game after this gameweek -- the real API 404s in that case) -- a genuinely different claim
    from "picked an empty squad." Only a 404 is caught and turned into None; any other real API
    failure still surfaces loudly rather than being silently swallowed."""
    if payload is None:
        try:
            payload = _fetch_json(f"{FPL_API_BASE}/entry/{entry_id}/event/{event}/picks/")
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                return None
            raise
    return payload.get("picks")


# ============================================================
# ingest -- lands in fact_rival_squad_sample. Not a fact_raw/reconcile two-step (a live
# multi-endpoint fetch, not a locally-provided file) -- same shape ingest_understat.py already
# established for this project's other live-fetch source.
# ============================================================

def ingest_rival_squad_sample(
    con: duckdb.DuckDBPyConnection, season: str, event: int, ingested_date: datetime,
    *, league_id: int = FPL_OVERALL_LEAGUE_ID, n_entries: int = 200,
    element_names: dict[int, str] | None = None, entries: list[dict] | None = None,
    entry_picks_by_id: dict[int, list[dict] | None] | None = None,
) -> dict:
    """Top-level orchestration. element_names/entries/entry_picks_by_id can each be injected
    directly (tests do this, and so can a real caller with its own already-fetched data)
    instead of a live fetch. Idempotent per (season, event): real historical picks for an
    already-sampled gameweek never change, so a second call for the same one is a genuine
    no-op -- unlike this project's other "latest snapshot wins" model-version tables, there is
    no newer version of the past to overwrite with."""
    already = con.execute(
        "SELECT 1 FROM fact_rival_squad_sample WHERE season = ? AND event = ? LIMIT 1", [season, event],
    ).fetchone()
    if already:
        return {"status": "unchanged", "entries_sampled": 0, "picks_inserted": 0, "entries_skipped": 0}

    if element_names is None:
        element_names = fetch_bootstrap_elements()
    if entries is None:
        entries = fetch_top_entries(league_id, n_entries)

    picks_inserted, entries_skipped = 0, 0
    for entry in entries:
        entry_id = entry["entry_id"]
        picks = entry_picks_by_id.get(entry_id) if entry_picks_by_id is not None else fetch_entry_picks(entry_id, event)
        if not picks:
            entries_skipped += 1
            continue
        for p in picks:
            player_uid = iw._resolve_player(con, element_names.get(p["element"]), season)
            if not player_uid:
                continue
            con.execute(
                "INSERT INTO fact_rival_squad_sample (entry_id, season, event, player_uid, is_captain, "
                "multiplier, league_rank, _ingested_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
                [
                    entry_id, season, event, player_uid, bool(p.get("is_captain")),
                    int(p.get("multiplier") or 0), entry.get("rank"), ingested_date,
                ],
            )
            picks_inserted += 1

    return {
        "status": "ingested", "entries_sampled": len(entries) - entries_skipped,
        "picks_inserted": picks_inserted, "entries_skipped": entries_skipped,
    }


# ============================================================
# basic descriptive access -- a sanity-check view of what was actually sampled, same spirit as
# every other ingest module's own row-count reporting. NOT Phase C's rank-distribution work --
# no simulation, no joint outcomes, just counting what's already in the table.
# ============================================================

def most_owned_players(con: duckdb.DuckDBPyConnection, season: str, event: int, top_n: int = 10) -> list[dict]:
    rows = con.execute(
        "SELECT s.player_uid, dp.canonical_name, count(*) AS n_owners, "
        "sum(CASE WHEN s.is_captain THEN 1 ELSE 0 END) AS n_captains "
        "FROM fact_rival_squad_sample s JOIN dim_player dp ON dp.player_uid = s.player_uid "
        "WHERE s.season = ? AND s.event = ? AND s.multiplier > 0 "
        "GROUP BY s.player_uid, dp.canonical_name ORDER BY n_owners DESC, s.player_uid LIMIT ?",
        [season, event, top_n],
    ).fetchall()
    return [
        {"player_uid": uid, "name": name, "n_owners": n_owners, "n_captains": n_captains}
        for uid, name, n_owners, n_captains in rows
    ]
