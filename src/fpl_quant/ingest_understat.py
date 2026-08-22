"""Priority 7a: Understat season-cumulative xG/xA/npxG/xGChain/xGBuildup ingestion -- a
second, independent xG source alongside FPL-Core-Insights' own expected_goals/expected_assists
(already in fact_player_season_stats since M0).

Understat has no official public API. This fetches its real league season page and extracts
the `playersData` JSON Understat's own frontend embeds directly in a <script> tag -- the same
approach the long-standing community `understat`/`understatapi` packages use, based on
Understat's publicly documented, years-stable page format. That page-structure assumption is
NOT independently verified against a live fetch in THIS environment: this sandbox's own
network policy blocks understat.com (confirmed via a direct connection attempt -- 403 policy
denial from the outbound proxy, not a guess). Verify parse_understat_players_html() against a
real fetched page (e.g. from a CI runner with open internet, or locally) before relying on
this in production. _fetch_understat_html() is isolated specifically so that verification is a
one-line swap, not a rewrite -- every other function here is exercised by this module's own
tests against a hand-built fixture faithful to Understat's documented real structure.

Scope: season-cumulative totals via ONE request per season to the league page (which already
carries every player's season totals), not per-match scraping across hundreds of fixtures --
a deliberately tractable v1, same "scoped v1, flagged extension" pattern this project already
used for the set-piece evidence uplift (see expected_points.py) before Priority 7b extended it.

Deliberately informational, never silently blended into ep_goals/ep_assists: two
independently-fitted goal-rate estimates (this one and FPL-Core-Insights' own) combined
without a principled, backtested weighting rule would be exactly the double-counting risk this
project's conventions warn against everywhere else -- see explain_player_xg_signal()'s own
caveat. xGChain/xGBuildup have no existing analog anywhere in this project at all -- a
genuinely new playmaking-involvement signal, also surfaced informationally here rather than
woven into the points model in this first pass.

Understat's own JS escapes the embedded JSON with \\xHH byte-hex sequences rather than
standard JSON string escapes (a real, well-known Understat quirk, not a guess) --
_decode_understat_json() reverses that before json.loads().
"""

import json
import re
from datetime import datetime
from hashlib import sha256

import duckdb
import requests

from . import ingest_workbook as iw

UNDERSTAT_LEAGUE_URL = "https://understat.com/league/EPL/{season_start_year}"


# ============================================================
# fetch -- isolated network I/O, the one function real verification and test mocking touch
# ============================================================

def _fetch_understat_html(url: str) -> str:
    resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    return resp.text


# ============================================================
# parse
# ============================================================

def _decode_understat_json(js_string_literal: str) -> list[dict]:
    """Understat embeds playersData as JSON.parse('...') with \\xHH byte-hex escapes for
    every non-ASCII byte -- decode those to real bytes, then UTF-8 decode, then parse as JSON."""
    raw_bytes = re.sub(
        r"\\x([0-9A-Fa-f]{2})", lambda m: chr(int(m.group(1), 16)), js_string_literal,
    ).encode("latin-1")
    return json.loads(raw_bytes.decode("utf-8"))


def parse_understat_players_html(html: str) -> list[dict]:
    """Extracts the playersData JSON.parse('...') block from a real Understat league season
    page. Returns [] (not an error) if the page structure doesn't match -- a page-layout
    change should be visible downstream as "zero players ingested," not a crash mid-pipeline."""
    match = re.search(r"var\s+playersData\s*=\s*JSON\.parse\('(.+?)'\)", html)
    if not match:
        return []
    try:
        return _decode_understat_json(match.group(1))
    except (ValueError, UnicodeDecodeError):
        return []


# ============================================================
# ingest -- lands directly in fact_understat_player_season (no fact_raw/reconcile two-step:
# this is a live fetch, not a locally-provided file, so ingest_csv.py's file-scanning pattern
# doesn't apply). Still writes a real fact_raw_ingestion_log row for the same idempotency and
# provenance guarantee every other ingest module gets -- source_relpath holds the URL fetched,
# source_file_hash a hash of the raw HTML, so a byte-identical re-fetch is a genuine no-op.
# ============================================================

def ingest_league_season(
    con: duckdb.DuckDBPyConnection, season: str, season_start_year: int, ingested_date: datetime,
    *, html: str | None = None,
) -> dict:
    """html can be injected directly (tests do this, and so can a real caller with its own
    already-fetched page) instead of fetching live."""
    url = UNDERSTAT_LEAGUE_URL.format(season_start_year=season_start_year)
    if html is None:
        html = _fetch_understat_html(url)

    content_hash = sha256(html.encode("utf-8")).hexdigest()
    already = con.execute(
        "SELECT batch_id FROM fact_raw_ingestion_log WHERE raw_table_name = 'fact_understat_player_season' "
        "AND source_file_hash = ?", [content_hash],
    ).fetchone()
    if already:
        return {"status": "unchanged", "inserted": 0, "skipped": 0}

    players = parse_understat_players_html(html)
    inserted, skipped = 0, 0
    for p in players:
        name = p.get("player_name")
        player_uid = iw._resolve_player(con, name, season)
        if not player_uid:
            skipped += 1
            continue
        con.execute(
            "INSERT INTO fact_understat_player_season (player_uid, season, understat_player_id, "
            "source_player_name, games, minutes, goals, assists, xg, npxg, xa, xgchain, xgbuildup, "
            "shots, key_passes, _ingested_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (player_uid, season) DO UPDATE SET "
            "understat_player_id = excluded.understat_player_id, source_player_name = excluded.source_player_name, "
            "games = excluded.games, minutes = excluded.minutes, goals = excluded.goals, assists = excluded.assists, "
            "xg = excluded.xg, npxg = excluded.npxg, xa = excluded.xa, xgchain = excluded.xgchain, "
            "xgbuildup = excluded.xgbuildup, shots = excluded.shots, key_passes = excluded.key_passes, "
            "_ingested_at = excluded._ingested_at",
            [
                player_uid, season, str(p.get("id")), name,
                int(p.get("games") or 0), int(p.get("time") or 0), int(p.get("goals") or 0), int(p.get("assists") or 0),
                float(p.get("xG") or 0.0), float(p.get("npxG") or 0.0), float(p.get("xA") or 0.0),
                float(p.get("xGChain") or 0.0), float(p.get("xGBuildup") or 0.0),
                int(p.get("shots") or 0), int(p.get("key_passes") or 0), ingested_date,
            ],
        )
        inserted += 1

    con.execute(
        "INSERT INTO fact_raw_ingestion_log (raw_table_name, season, source_relpath, source_file_hash, row_count) "
        "VALUES ('fact_understat_player_season', ?, ?, ?, ?)",
        [season, url, content_hash, inserted],
    )
    return {"status": "ingested", "inserted": inserted, "skipped": skipped}


# ============================================================
# M9 adapter -- per this project's established integration pattern ("each of M0-M8 exposes
# its own explain()-style interface... M9 does not reach into other modules' internals
# directly," reporting.py's own module docstring), this is the ONLY function reporting.py
# reads Understat data through.
# ============================================================

def explain_player_xg_signal(con: duckdb.DuckDBPyConnection, player_uid: str, season: str) -> dict | None:
    """Understat's real numbers for one player, plus a side-by-side comparison against
    FPL-Core-Insights' own expected_goals_per_90/expected_assists_per_90 (that table's LATEST
    row for the season) -- informational only, this function never writes to or otherwise
    influences ep_outputs. None when no Understat row exists for this player/season (not every
    FPL player necessarily resolves against Understat's own EPL-only roster -- a player who
    only featured for a promoted team's prior Championship season, for instance)."""
    row = con.execute(
        "SELECT games, minutes, xg, npxg, xa, xgchain, xgbuildup, shots, key_passes "
        "FROM fact_understat_player_season WHERE player_uid = ? AND season = ?",
        [player_uid, season],
    ).fetchone()
    if row is None:
        return None
    games, minutes, xg, npxg, xa, xgchain, xgbuildup, shots, key_passes = row

    fci_row = con.execute(
        "SELECT expected_goals_per_90, expected_assists_per_90 FROM fact_player_season_stats "
        "WHERE player_uid = ? AND season = ? ORDER BY gw DESC LIMIT 1", [player_uid, season],
    ).fetchone()
    fci_xg_per_90, fci_xa_per_90 = fci_row if fci_row else (None, None)

    per_90 = (lambda total: total / minutes * 90) if minutes else (lambda total: None)
    return {
        "player_uid": player_uid, "season": season, "games": games, "minutes": minutes,
        "shots": shots, "key_passes": key_passes,
        "understat_xg_per_90": per_90(xg), "understat_npxg_per_90": per_90(npxg), "understat_xa_per_90": per_90(xa),
        "xgchain_per_90": per_90(xgchain), "xgbuildup_per_90": per_90(xgbuildup),
        "fpl_core_insights_xg_per_90": fci_xg_per_90, "fpl_core_insights_xa_per_90": fci_xa_per_90,
        "caveat": (
            "informational second opinion only -- Understat's xG model is independently fitted "
            "from FPL-Core-Insights' own expected_goals/expected_assists and is never blended "
            "into ep_goals/ep_assists (no principled combination rule exists for two "
            "independently-fitted rate estimates). xGChain/xGBuildup have no existing analog "
            "elsewhere in this project and are surfaced here as a new playmaking-involvement "
            "signal, not yet woven into the points model."
        ),
    }
