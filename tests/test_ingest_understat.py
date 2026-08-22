import json
from datetime import datetime

import pytest

from fpl_quant import ingest_understat as iu


def _hex_escape(text: str) -> str:
    """Mirrors Understat's own real embedding format: every byte of the UTF-8-encoded JSON
    escaped as \\xHH, wrapped in JSON.parse('...') -- not a simplified stand-in encoding."""
    return "".join(f"\\x{b:02x}" for b in text.encode("utf-8"))


def _fixture_html(players: list[dict]) -> str:
    escaped = _hex_escape(json.dumps(players))
    return f"""<!doctype html><html><body>
<script>
var datesData = JSON.parse('\\x5B\\x5D');
var playersData = JSON.parse('{escaped}');
var teamsData = JSON.parse('\\x7B\\x7D');
</script>
</body></html>"""


_ALAN_ROW = {
    "id": "1234", "player_name": "Alan Test", "games": "10", "time": "900",
    "goals": "5", "assists": "2", "xG": "4.5", "npxG": "4.0", "xA": "1.8",
    "xGChain": "6.2", "xGBuildup": "3.1", "shots": "20", "key_passes": "15",
}


# ============================================================
# _decode_understat_json / parse_understat_players_html
# ============================================================

def test_decode_understat_json_round_trips_hex_escaped_payload():
    payload = [{"a": 1, "name": "Ünïcödé Tëst"}]
    escaped = _hex_escape(json.dumps(payload))
    assert iu._decode_understat_json(escaped) == payload


def test_parse_understat_players_html_extracts_real_shaped_page():
    html = _fixture_html([_ALAN_ROW])
    players = iu.parse_understat_players_html(html)
    assert len(players) == 1
    assert players[0]["player_name"] == "Alan Test"
    assert players[0]["xG"] == "4.5"


def test_parse_understat_players_html_returns_empty_list_for_missing_variable():
    assert iu.parse_understat_players_html("<html>no playersData here</html>") == []


def test_parse_understat_players_html_returns_empty_list_for_malformed_payload():
    html = "<script>var playersData = JSON.parse('\\xzz\\xzz');</script>"
    assert iu.parse_understat_players_html(html) == []


# ============================================================
# ingest_league_season
# ============================================================

def _seed_resolvable_player(con, name, normalized, player_uid, season="2025-2026"):
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, 'Forward')", [player_uid, name])
    con.execute(
        "INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) "
        "VALUES (?, ?, '1', ?, ?)", [name, normalized, season, player_uid],
    )


def test_ingest_league_season_resolves_and_inserts(con):
    _seed_resolvable_player(con, "Alan Test", "alan test", "p_alan")
    html = _fixture_html([_ALAN_ROW])
    result = iu.ingest_league_season(con, "2025-2026", 2025, datetime(2026, 8, 10), html=html)
    assert result == {"status": "ingested", "inserted": 1, "skipped": 0}

    row = con.execute(
        "SELECT player_uid, games, minutes, xg, npxg, xa, xgchain, xgbuildup, shots, key_passes "
        "FROM fact_understat_player_season WHERE season = '2025-2026'"
    ).fetchone()
    assert row == ("p_alan", 10, 900, 4.5, 4.0, 1.8, 6.2, 3.1, 20, 15)


def test_ingest_league_season_skips_unresolvable_players(con):
    html = _fixture_html([_ALAN_ROW])  # no player_alias row seeded at all
    result = iu.ingest_league_season(con, "2025-2026", 2025, datetime(2026, 8, 10), html=html)
    assert result == {"status": "ingested", "inserted": 0, "skipped": 1}
    assert con.execute("SELECT count(*) FROM fact_understat_player_season").fetchone()[0] == 0


def test_ingest_league_season_is_idempotent_on_identical_html(con):
    _seed_resolvable_player(con, "Alan Test", "alan test", "p_alan")
    html = _fixture_html([_ALAN_ROW])
    iu.ingest_league_season(con, "2025-2026", 2025, datetime(2026, 8, 10), html=html)
    second = iu.ingest_league_season(con, "2025-2026", 2025, datetime(2026, 8, 11), html=html)
    assert second == {"status": "unchanged", "inserted": 0, "skipped": 0}
    assert con.execute("SELECT count(*) FROM fact_understat_player_season").fetchone()[0] == 1


def test_ingest_league_season_upserts_on_changed_html(con):
    _seed_resolvable_player(con, "Alan Test", "alan test", "p_alan")
    iu.ingest_league_season(con, "2025-2026", 2025, datetime(2026, 8, 10), html=_fixture_html([_ALAN_ROW]))

    updated_row = {**_ALAN_ROW, "xG": "9.9"}
    result = iu.ingest_league_season(con, "2025-2026", 2025, datetime(2026, 8, 17), html=_fixture_html([updated_row]))
    assert result["status"] == "ingested"

    xg = con.execute("SELECT xg FROM fact_understat_player_season WHERE player_uid = 'p_alan'").fetchone()[0]
    assert xg == pytest.approx(9.9)


# ============================================================
# explain_player_xg_signal -- M9 adapter
# ============================================================

def test_explain_player_xg_signal_none_when_no_understat_row(con):
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p1', 'P1', 'Forward')")
    assert iu.explain_player_xg_signal(con, "p1", "2025-2026") is None


def test_explain_player_xg_signal_computes_per_90_rates_and_fci_comparison(con):
    _seed_resolvable_player(con, "Alan Test", "alan test", "p_alan")
    iu.ingest_league_season(con, "2025-2026", 2025, datetime(2026, 8, 10), html=_fixture_html([_ALAN_ROW]))
    con.execute(
        "INSERT INTO fact_player_season_stats (player_uid, season, gw, expected_goals_per_90, "
        "expected_assists_per_90, _ingested_at) VALUES ('p_alan', '2025-2026', 5, 0.5, 0.3, current_timestamp)"
    )

    result = iu.explain_player_xg_signal(con, "p_alan", "2025-2026")
    assert result["games"] == 10
    assert result["understat_xg_per_90"] == pytest.approx(4.5 / 900 * 90)
    assert result["understat_xa_per_90"] == pytest.approx(1.8 / 900 * 90)
    assert result["xgchain_per_90"] == pytest.approx(6.2 / 900 * 90)
    assert result["fpl_core_insights_xg_per_90"] == pytest.approx(0.5)
    assert result["fpl_core_insights_xa_per_90"] == pytest.approx(0.3)
    assert "never blended" in result["caveat"]
