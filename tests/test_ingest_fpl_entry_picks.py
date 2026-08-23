from datetime import datetime

import pytest
import requests

from fpl_quant import ingest_fpl_entry_picks as ifp


def _seed_resolvable_player(con, name, normalized, player_uid, season="2025-2026"):
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, 'Forward')", [player_uid, name])
    con.execute(
        "INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) "
        "VALUES (?, ?, '1', ?, ?)", [name, normalized, season, player_uid],
    )


# ============================================================
# fetch_bootstrap_elements
# ============================================================

def test_fetch_bootstrap_elements_builds_full_name_map():
    payload = {"elements": [
        {"id": 1, "first_name": "Alan", "second_name": "Test", "web_name": "Test"},
        {"id": 2, "first_name": "Bo", "second_name": "Example", "web_name": "Example"},
    ]}
    names = ifp.fetch_bootstrap_elements(payload=payload)
    assert names == {1: "Alan Test", 2: "Bo Example"}


def test_fetch_bootstrap_elements_empty_for_no_elements():
    assert ifp.fetch_bootstrap_elements(payload={"elements": []}) == {}


# ============================================================
# fetch_top_entries -- pagination
# ============================================================

def test_fetch_top_entries_stops_once_n_reached_within_one_page():
    pages = [{"standings": {"has_next": True, "results": [
        {"entry": 100, "rank": 1}, {"entry": 101, "rank": 2}, {"entry": 102, "rank": 3},
    ]}}]
    entries = ifp.fetch_top_entries(314, 2, pages=pages)
    assert entries == [{"entry_id": 100, "rank": 1}, {"entry_id": 101, "rank": 2}]


def test_fetch_top_entries_paginates_across_multiple_pages():
    pages = [
        {"standings": {"has_next": True, "results": [{"entry": 100, "rank": 1}, {"entry": 101, "rank": 2}]}},
        {"standings": {"has_next": False, "results": [{"entry": 102, "rank": 3}]}},
    ]
    entries = ifp.fetch_top_entries(314, 3, pages=pages)
    assert [e["entry_id"] for e in entries] == [100, 101, 102]


def test_fetch_top_entries_stops_when_league_runs_out_before_n():
    pages = [{"standings": {"has_next": False, "results": [{"entry": 100, "rank": 1}]}}]
    entries = ifp.fetch_top_entries(314, 50, pages=pages)
    assert entries == [{"entry_id": 100, "rank": 1}]


def test_fetch_top_entries_empty_league():
    entries = ifp.fetch_top_entries(314, 10, pages=[{"standings": {"has_next": False, "results": []}}])
    assert entries == []


# ============================================================
# fetch_entry_picks
# ============================================================

def test_fetch_entry_picks_returns_the_picks_list():
    payload = {"picks": [{"element": 1, "is_captain": True, "multiplier": 2}]}
    picks = ifp.fetch_entry_picks(100, 5, payload=payload)
    assert picks == [{"element": 1, "is_captain": True, "multiplier": 2}]


def test_fetch_entry_picks_none_when_payload_has_no_picks_key():
    assert ifp.fetch_entry_picks(100, 5, payload={}) is None


# ============================================================
# ingest_rival_squad_sample
# ============================================================

def _standard_scenario(con):
    _seed_resolvable_player(con, "Alan Test", "alan test", "p_alan")
    _seed_resolvable_player(con, "Bo Example", "bo example", "p_bo")
    element_names = {1: "Alan Test", 2: "Bo Example"}
    entries = [{"entry_id": 100, "rank": 1}, {"entry_id": 101, "rank": 2}]
    entry_picks_by_id = {
        100: [{"element": 1, "is_captain": True, "multiplier": 2}, {"element": 2, "is_captain": False, "multiplier": 1}],
        101: [{"element": 2, "is_captain": True, "multiplier": 2}],
    }
    return element_names, entries, entry_picks_by_id


def test_ingest_rival_squad_sample_inserts_resolved_picks(con):
    element_names, entries, entry_picks_by_id = _standard_scenario(con)
    result = ifp.ingest_rival_squad_sample(
        con, "2025-2026", 5, datetime(2026, 8, 10),
        element_names=element_names, entries=entries, entry_picks_by_id=entry_picks_by_id,
    )
    assert result == {"status": "ingested", "entries_sampled": 2, "picks_inserted": 3, "entries_skipped": 0}

    rows = con.execute(
        "SELECT entry_id, player_uid, is_captain, multiplier, league_rank FROM fact_rival_squad_sample "
        "WHERE season = '2025-2026' AND event = 5 ORDER BY entry_id, player_uid"
    ).fetchall()
    assert rows == [
        (100, "p_alan", True, 2, 1),
        (100, "p_bo", False, 1, 1),
        (101, "p_bo", True, 2, 2),
    ]


def test_ingest_rival_squad_sample_skips_entries_with_no_picks(con):
    element_names, entries, _ = _standard_scenario(con)
    entry_picks_by_id = {100: None, 101: []}
    result = ifp.ingest_rival_squad_sample(
        con, "2025-2026", 5, datetime(2026, 8, 10),
        element_names=element_names, entries=entries, entry_picks_by_id=entry_picks_by_id,
    )
    assert result == {"status": "ingested", "entries_sampled": 0, "picks_inserted": 0, "entries_skipped": 2}


def test_ingest_rival_squad_sample_skips_unresolvable_players(con):
    # no player_alias rows seeded at all -- neither element resolves
    entries = [{"entry_id": 100, "rank": 1}]
    entry_picks_by_id = {100: [{"element": 1, "is_captain": True, "multiplier": 2}]}
    result = ifp.ingest_rival_squad_sample(
        con, "2025-2026", 5, datetime(2026, 8, 10),
        element_names={1: "Alan Test"}, entries=entries, entry_picks_by_id=entry_picks_by_id,
    )
    assert result == {"status": "ingested", "entries_sampled": 1, "picks_inserted": 0, "entries_skipped": 0}
    assert con.execute("SELECT count(*) FROM fact_rival_squad_sample").fetchone()[0] == 0


def test_ingest_rival_squad_sample_idempotent_for_an_already_sampled_gameweek(con):
    element_names, entries, entry_picks_by_id = _standard_scenario(con)
    ifp.ingest_rival_squad_sample(
        con, "2025-2026", 5, datetime(2026, 8, 10),
        element_names=element_names, entries=entries, entry_picks_by_id=entry_picks_by_id,
    )
    second = ifp.ingest_rival_squad_sample(
        con, "2025-2026", 5, datetime(2026, 8, 17),
        element_names=element_names, entries=entries, entry_picks_by_id=entry_picks_by_id,
    )
    assert second == {"status": "unchanged", "entries_sampled": 0, "picks_inserted": 0, "entries_skipped": 0}
    assert con.execute("SELECT count(*) FROM fact_rival_squad_sample").fetchone()[0] == 3


def test_ingest_rival_squad_sample_different_gameweek_is_not_blocked_by_idempotency(con):
    element_names, entries, entry_picks_by_id = _standard_scenario(con)
    ifp.ingest_rival_squad_sample(
        con, "2025-2026", 5, datetime(2026, 8, 10),
        element_names=element_names, entries=entries, entry_picks_by_id=entry_picks_by_id,
    )
    result = ifp.ingest_rival_squad_sample(
        con, "2025-2026", 6, datetime(2026, 8, 17),
        element_names=element_names, entries=entries, entry_picks_by_id=entry_picks_by_id,
    )
    assert result["status"] == "ingested"
    assert result["picks_inserted"] == 3


# ============================================================
# most_owned_players
# ============================================================

def test_most_owned_players_ranks_by_ownership_count(con):
    element_names, entries, entry_picks_by_id = _standard_scenario(con)
    ifp.ingest_rival_squad_sample(
        con, "2025-2026", 5, datetime(2026, 8, 10),
        element_names=element_names, entries=entries, entry_picks_by_id=entry_picks_by_id,
    )
    top = ifp.most_owned_players(con, "2025-2026", 5)
    assert top[0]["player_uid"] == "p_bo"
    assert top[0]["n_owners"] == 2
    assert top[0]["n_captains"] == 1
    assert top[1]["player_uid"] == "p_alan"
    assert top[1]["n_owners"] == 1


def test_most_owned_players_excludes_zero_multiplier_bench_picks(con):
    _seed_resolvable_player(con, "Alan Test", "alan test", "p_alan")
    ifp.ingest_rival_squad_sample(
        con, "2025-2026", 5, datetime(2026, 8, 10),
        element_names={1: "Alan Test"}, entries=[{"entry_id": 100, "rank": 1}],
        entry_picks_by_id={100: [{"element": 1, "is_captain": False, "multiplier": 0}]},
    )
    assert ifp.most_owned_players(con, "2025-2026", 5) == []


# ============================================================
# _fetch_json -- retry/backoff (Phase B hardening, same mechanism as app_export._fetch_json)
# ============================================================

class _FakeResponse:
    def __init__(self, status_code, json_payload=None, headers=None):
        self.status_code = status_code
        self._json_payload = json_payload
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error", response=self)

    def json(self):
        return self._json_payload


def test_fetch_json_retries_on_429_then_succeeds(monkeypatch):
    responses = [_FakeResponse(429), _FakeResponse(200, {"ok": True})]
    monkeypatch.setattr(ifp.requests, "get", lambda url, timeout, headers: responses.pop(0))
    monkeypatch.setattr(ifp.time, "sleep", lambda s: None)
    assert ifp._fetch_json("https://example.test/x") == {"ok": True}


def test_fetch_json_does_not_retry_on_404(monkeypatch):
    calls = []

    def fake_get(url, timeout, headers):
        calls.append(url)
        return _FakeResponse(404)

    monkeypatch.setattr(ifp.requests, "get", fake_get)
    monkeypatch.setattr(ifp.time, "sleep", lambda s: (_ for _ in ()).throw(AssertionError("should not sleep/retry on a 404")))
    with pytest.raises(requests.HTTPError):
        ifp._fetch_json("https://example.test/x")
    assert len(calls) == 1
