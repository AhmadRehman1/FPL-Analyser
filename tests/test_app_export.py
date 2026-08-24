import json

import pytest
import requests

from fpl_quant import app_export as ax


# ============================================================
# build_player_directory
# ============================================================

def test_build_player_directory_maps_and_rescales_fields():
    bootstrap = {
        "teams": [{"id": 1, "name": "Arsenal", "short_name": "ARS"}],
        "elements": [{
            "id": 10, "web_name": "Saka", "first_name": "Bukayo", "second_name": "Saka",
            "team": 1, "element_type": 3, "now_cost": 101, "total_points": 55, "event_points": 8,
            "form": "5.2", "points_per_game": "6.1", "selected_by_percent": "42.3", "minutes": 900,
            "goals_scored": 4, "assists": 6, "clean_sheets": 3, "goals_conceded": 5, "bonus": 9,
            "yellow_cards": 1, "red_cards": 0, "ict_index": "88.4", "status": "a",
            "chance_of_playing_next_round": 100,
        }],
    }
    [row] = ax.build_player_directory(bootstrap)
    assert row["team"] == "ARS"
    assert row["position"] == "MID"
    assert row["price"] == 10.1
    assert row["form"] == 5.2
    assert row["selected_by_percent"] == 42.3


def test_build_player_directory_carries_official_news_fields():
    bootstrap = {"teams": [{"id": 1, "name": "Arsenal", "short_name": "ARS"}], "elements": [{
        "id": 1, "web_name": "Saliba", "first_name": "William", "second_name": "Saliba", "team": 1,
        "element_type": 2, "now_cost": 60, "status": "d", "news": "Knock - 75% chance of playing",
        "news_added": "2026-08-20T10:00:00Z", "chance_of_playing_next_round": 75,
    }]}
    [row] = ax.build_player_directory(bootstrap)
    assert row["status"] == "d"
    assert row["news"] == "Knock - 75% chance of playing"
    assert row["chance_of_playing_next_round"] == 75


def test_build_player_directory_reports_no_news_as_none_not_empty_string():
    bootstrap = {"teams": [], "elements": [{
        "id": 1, "web_name": "Fit", "first_name": "A", "second_name": "B", "team": None,
        "element_type": 1, "now_cost": 45, "status": "a", "news": "",
    }]}
    [row] = ax.build_player_directory(bootstrap)
    assert row["news"] is None


def test_build_player_directory_handles_blank_numeric_strings():
    bootstrap = {"teams": [], "elements": [{
        "id": 1, "web_name": "New", "first_name": "New", "second_name": "Signing", "team": None,
        "element_type": 2, "now_cost": 40, "form": "", "points_per_game": "", "selected_by_percent": "",
        "ict_index": "",
    }]}
    [row] = ax.build_player_directory(bootstrap)
    assert row["form"] is None
    assert row["team"] is None


# ============================================================
# build_price_watch
# ============================================================

def _player(id, web_name, transfers_in_event=0, transfers_out_event=0, **kw):
    base = {
        "id": id, "web_name": web_name, "team": "ARS", "team_name": "Arsenal", "position": "MID",
        "price": 8.0, "selected_by_percent": 20.0,
        "transfers_in_event": transfers_in_event, "transfers_out_event": transfers_out_event,
    }
    base.update(kw)
    return base


def test_build_price_watch_splits_risers_and_fallers_by_net_momentum():
    directory = [
        _player(1, "Riser", transfers_in_event=500_000, transfers_out_event=10_000),
        _player(2, "Faller", transfers_in_event=5_000, transfers_out_event=300_000),
        _player(3, "Quiet", transfers_in_event=0, transfers_out_event=0),
    ]
    out = ax.build_price_watch(directory, top_n=5)
    assert [p["web_name"] for p in out["risers"]] == ["Riser"]
    assert out["risers"][0]["net_transfers"] == 490_000
    assert [p["web_name"] for p in out["fallers"]] == ["Faller"]
    assert out["fallers"][0]["net_transfers"] == -295_000


def test_build_price_watch_respects_top_n_and_sort_order():
    directory = [
        _player(1, "Small riser", transfers_in_event=10_000),
        _player(2, "Big riser", transfers_in_event=100_000),
    ]
    out = ax.build_price_watch(directory, top_n=1)
    assert [p["web_name"] for p in out["risers"]] == ["Big riser"]


def test_build_price_watch_empty_when_no_transfer_activity():
    directory = [_player(1, "Static")]
    out = ax.build_price_watch(directory)
    assert out == {"risers": [], "fallers": []}


# ============================================================
# build_fixtures_by_gameweek
# ============================================================

def test_build_fixtures_by_gameweek_groups_and_sorts():
    bootstrap = {
        "teams": [{"id": 1, "name": "Arsenal", "short_name": "ARS"}, {"id": 2, "name": "Chelsea", "short_name": "CHE"}],
        "events": [{"id": 3, "deadline_time": "2026-09-04T17:30:00Z", "is_current": False}],
    }
    fixtures = [
        {"id": 100, "event": 3, "kickoff_time": "2026-09-05T15:00:00Z", "finished": False,
         "team_h": 2, "team_a": 1, "team_h_difficulty": 4, "team_a_difficulty": 2,
         "team_h_score": None, "team_a_score": None},
        {"id": 99, "event": 3, "kickoff_time": "2026-09-05T12:30:00Z", "finished": False,
         "team_h": 1, "team_a": 2, "team_h_difficulty": 2, "team_a_difficulty": 4,
         "team_h_score": None, "team_a_score": None},
    ]
    out = ax.build_fixtures_by_gameweek(bootstrap, fixtures)
    [gw] = out["gameweeks"]
    assert gw["gameweek"] == 3
    assert [f["id"] for f in gw["fixtures"]] == [99, 100]
    assert gw["fixtures"][0]["home"]["short_name"] == "ARS"
    assert gw["fixtures"][0]["home"]["difficulty"] == 2


def test_build_fixtures_by_gameweek_skips_fixtures_with_no_event():
    bootstrap = {"teams": [], "events": []}
    fixtures = [{"id": 1, "event": None, "team_h": 1, "team_a": 2}]
    out = ax.build_fixtures_by_gameweek(bootstrap, fixtures)
    assert out["gameweeks"] == []


# ============================================================
# compute_free_transfers
# ============================================================

def test_compute_free_transfers_starts_at_one_and_carries_over():
    history = [
        {"event": 2, "event_transfers": 0},
        {"event": 3, "event_transfers": 0},
    ]
    # GW2: 1 available, 0 used -> carries to 2 (capped concept starts here)
    # GW3: 2 available, 0 used -> carries to 3... but capped at 5 overall
    assert ax.compute_free_transfers(history, []) == 3


def test_compute_free_transfers_deducts_used_transfers():
    history = [
        {"event": 2, "event_transfers": 1},  # 1 available, uses 1 -> 0 left -> +1 next = 1
        {"event": 3, "event_transfers": 0},  # 1 available, 0 used -> 2
    ]
    assert ax.compute_free_transfers(history, []) == 2


def test_compute_free_transfers_caps_at_five():
    history = [{"event": e, "event_transfers": 0} for e in range(2, 12)]
    assert ax.compute_free_transfers(history, []) == 5


def test_compute_free_transfers_wildcard_gameweek_is_free_and_uncapped_by_usage():
    history = [
        {"event": 2, "event_transfers": 9},  # would normally crater the count
    ]
    chips = [{"name": "wildcard", "event": 2}]
    # wildcard event is skipped entirely (transfers don't count against the FT ledger)
    assert ax.compute_free_transfers(history, chips) == 2


def test_compute_free_transfers_ignores_gw1_row():
    # entry/<id>/history/'s real "current" array always includes a GW1 row once GW1 has been
    # played -- GW1 is squad selection, not a transfer-eligible gameweek, so it must never
    # grant a phantom rollover credit on top of the GW2 base of 1.
    history = [{"event": 1, "event_transfers": 0}]
    assert ax.compute_free_transfers(history, []) == 1


def test_compute_free_transfers_gw1_row_does_not_offset_later_gameweeks():
    history = [
        {"event": 1, "event_transfers": 0},
        {"event": 2, "event_transfers": 0},
    ]
    # GW2 unused -> carries to GW3: 2, not 3 (which the GW1-row bug used to produce)
    assert ax.compute_free_transfers(history, []) == 2


# ============================================================
# build_team_snapshot
# ============================================================

def test_build_team_snapshot_wires_live_points_and_captain():
    entry_summary = {"id": 1305242, "name": "Main account", "player_first_name": "A", "player_last_name": "B"}
    picks_payload = {
        "active_chip": None,
        "entry_history": {"event": 7, "points": 68, "points_on_bench": 6, "bank": 14, "value": 1018},
        "picks": [
            {"element": 1, "position": 1, "multiplier": 1, "is_captain": False, "is_vice_captain": False},
            {"element": 2, "position": 12, "multiplier": 1, "is_captain": False, "is_vice_captain": False},
            {"element": 3, "position": 4, "multiplier": 2, "is_captain": True, "is_vice_captain": False},
        ],
    }
    live_payload = {"elements": [
        {"id": 1, "stats": {"total_points": 6, "minutes": 90}},
        {"id": 3, "stats": {"total_points": 19, "minutes": 90}},
    ]}
    directory = [
        {"id": 1, "web_name": "Webb", "team": "HBC", "position": "GKP", "price": 4.5},
        {"id": 2, "web_name": "Iyer", "team": "HBC", "position": "GKP", "price": 4.0},
        {"id": 3, "web_name": "Ferrer", "team": "RWV", "position": "MID", "price": 11.2},
    ]
    snap = ax.build_team_snapshot(entry_summary, picks_payload, live_payload, directory)
    assert snap["bank"] == 1.4
    assert snap["team_value"] == 101.8
    assert snap["gameweek"] == 7
    starter = next(p for p in snap["squad"] if p["player_id"] == 1)
    bench = next(p for p in snap["squad"] if p["player_id"] == 2)
    captain = next(p for p in snap["squad"] if p["player_id"] == 3)
    assert starter["in_xi"] is True and starter["event_points"] == 6
    assert bench["in_xi"] is False
    assert captain["is_captain"] is True and captain["event_points"] == 19


# ============================================================
# build_profile
# ============================================================

def test_build_profile_reports_chip_usage_and_bests():
    entry_summary = {
        "id": 1305242, "name": "Main account", "player_first_name": "A", "player_last_name": "B",
        "summary_overall_points": 412, "summary_overall_rank": 128450,
    }
    history = {
        "current": [
            {"event": 5, "points": 89, "overall_rank": 42000},
            {"event": 6, "points": 40, "overall_rank": 60000},
        ],
        "chips": [{"name": "wildcard", "event": 4, "time": "2026-09-01T00:00:00Z"}],
    }
    profile = ax.build_profile(entry_summary, history)
    assert profile["best_gameweek"] == {"gameweek": 5, "points": 89}
    assert profile["best_overall_rank"] == 42000
    wc = next(c for c in profile["chips"] if c["chip_type"] == "wildcard")
    bb = next(c for c in profile["chips"] if c["chip_type"] == "bboost")
    assert wc["used"] is True and wc["used_gameweek"] == 4
    assert bb["used"] is False and bb["used_gameweek"] is None


# ============================================================
# build_league_ownership
# ============================================================

def test_build_league_ownership_counts_ownership_and_captains():
    standings_payload = {"standings": {"results": [
        {"rank": 1, "entry": 555, "entry_name": "Nair-Do-Wells", "player_name": "Priya Nair"},
        {"rank": 2, "entry": 556, "entry_name": "Obi Wan FC", "player_name": "Marcus Obi"},
        {"rank": 3, "entry": 557, "entry_name": "No Picks Yet", "player_name": "New Joiner"},
    ]}}
    entry_picks_by_id = {
        555: {"picks": [
            {"element": 9, "multiplier": 2, "is_captain": True},
            {"element": 4, "multiplier": 1, "is_captain": False},
            {"element": 99, "multiplier": 0, "is_captain": False},  # benched, excluded
        ]},
        556: {"picks": [
            {"element": 9, "multiplier": 1, "is_captain": False},
            {"element": 4, "multiplier": 2, "is_captain": True},
        ]},
        557: None,  # no picks recorded yet -- excluded from the count entirely
    }
    player_directory = [
        {"id": 9, "web_name": "Salah"}, {"id": 4, "web_name": "Saka"}, {"id": 99, "web_name": "Bench"},
    ]
    out = ax.build_league_ownership(standings_payload, entry_picks_by_id, player_directory)
    assert out["n_entries_sampled"] == 2
    assert out["most_owned"][0] == {"player_id": 9, "web_name": "Salah", "n_owners": 2, "pct_of_league": 100.0}
    captains_by_entry = {c["entry_id"]: c["web_name"] for c in out["captains"]}
    assert captains_by_entry == {555: "Salah", 556: "Saka"}


def test_build_league_ownership_zero_when_nobody_has_picks_yet():
    standings_payload = {"standings": {"results": [{"rank": 1, "entry": 1, "entry_name": "X", "player_name": "Y"}]}}
    out = ax.build_league_ownership(standings_payload, {1: None}, [])
    assert out == {"n_entries_sampled": 0, "most_owned": [], "captains": []}


# ============================================================
# build_leagues
# ============================================================

def test_build_leagues_separates_overall_tile_from_fetched_tables():
    entry_summary = {
        "id": 1305242, "summary_overall_rank": 128450,
        "leagues": {"classic": [
            {"id": 314, "name": "Overall"},
            {"id": 99, "name": "Mates & Rivals", "entry_rank": 3},
        ]},
    }
    standings_by_league = {99: {"standings": {"results": [
        {"rank": 1, "last_rank": 1, "entry": 555, "entry_name": "Nair-Do-Wells", "player_name": "Priya Nair", "total": 441},
        {"rank": 3, "last_rank": 4, "entry": 1305242, "entry_name": "Main account", "player_name": "A B", "total": 412},
    ]}}}
    out = ax.build_leagues(entry_summary, standings_by_league, total_players=9_800_000)
    overall_tile = next(t for t in out["tiles"] if t["league_id"] == 314)
    assert overall_tile["rank"] == 128450 and overall_tile["total_entries"] == 9_800_000
    [table] = out["tables"]
    you_row = next(r for r in table["standings"] if r["is_you"])
    assert you_row["entry_id"] == 1305242
    assert table["ownership"] is None  # not passed, so not fabricated


def test_current_event_finds_the_is_current_gameweek():
    bootstrap = {"events": [{"id": 6, "is_current": False}, {"id": 7, "is_current": True}, {"id": 8, "is_current": False}]}
    assert ax.current_event(bootstrap) == 7


def test_current_event_none_when_nothing_flagged_current():
    assert ax.current_event({"events": [{"id": 1, "is_current": False}]}) is None
    assert ax.current_event({"events": []}) is None


def test_build_leagues_attaches_ownership_when_provided():
    entry_summary = {"id": 1, "leagues": {"classic": [{"id": 99, "name": "Mates & Rivals"}]}}
    standings_by_league = {99: {"standings": {"results": []}}}
    ownership_by_league = {99: {"n_entries_sampled": 5, "most_owned": [], "captains": []}}
    out = ax.build_leagues(entry_summary, standings_by_league, None, ownership_by_league)
    assert out["tables"][0]["ownership"] == ownership_by_league[99]


# ============================================================
# _fetch_json -- retry/backoff (Phase B7 hardening)
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


def test_fetch_json_succeeds_on_first_try(monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        return _FakeResponse(200, {"ok": True})

    monkeypatch.setattr(ax.requests, "get", fake_get)
    result = ax._fetch_json("http://example.test")
    assert result == {"ok": True}
    assert len(calls) == 1


def test_fetch_json_retries_on_429_then_succeeds(monkeypatch):
    responses = [_FakeResponse(429), _FakeResponse(429), _FakeResponse(200, {"ok": True})]
    sleeps = []

    def fake_get(url, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr(ax.requests, "get", fake_get)
    monkeypatch.setattr(ax.time, "sleep", lambda s: sleeps.append(s))

    result = ax._fetch_json("http://example.test", max_attempts=4, base_backoff_seconds=1.0)
    assert result == {"ok": True}
    assert sleeps == [1.0, 2.0]  # exponential backoff: 1*2**0, 1*2**1


def test_fetch_json_retries_on_connection_error_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake_get(url, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise requests.ConnectionError("boom")
        return _FakeResponse(200, {"ok": True})

    monkeypatch.setattr(ax.requests, "get", fake_get)
    monkeypatch.setattr(ax.time, "sleep", lambda s: None)

    result = ax._fetch_json("http://example.test")
    assert result == {"ok": True}
    assert calls["n"] == 3


def test_fetch_json_raises_upstream_unavailable_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr(ax.requests, "get", lambda url, **kwargs: _FakeResponse(503))
    monkeypatch.setattr(ax.time, "sleep", lambda s: None)

    with pytest.raises(ax.UpstreamUnavailableError):
        ax._fetch_json("http://example.test", max_attempts=2)


def test_fetch_json_honors_retry_after_header(monkeypatch):
    responses = [_FakeResponse(429, headers={"Retry-After": "7"}), _FakeResponse(200, {"ok": True})]
    sleeps = []

    monkeypatch.setattr(ax.requests, "get", lambda url, **kwargs: responses.pop(0))
    monkeypatch.setattr(ax.time, "sleep", lambda s: sleeps.append(s))

    ax._fetch_json("http://example.test")
    assert sleeps == [7.0]


def test_fetch_json_does_not_retry_a_real_404(monkeypatch):
    """A genuine client error (not rate-limited, not transient) must still raise a plain
    requests.HTTPError immediately, unretried -- fetch_entry_picks()'s own 404-means-None
    handling depends on this exact behavior being unchanged."""
    calls = {"n": 0}

    def fake_get(url, **kwargs):
        calls["n"] += 1
        return _FakeResponse(404)

    monkeypatch.setattr(ax.requests, "get", fake_get)
    monkeypatch.setattr(ax.time, "sleep", lambda s: (_ for _ in ()).throw(AssertionError("must not sleep/retry on a real 404")))

    with pytest.raises(requests.HTTPError):
        ax._fetch_json("http://example.test")
    assert calls["n"] == 1


def test_fetch_entry_picks_still_returns_none_on_404(monkeypatch):
    """Integration check: the retry wrapper must not break fetch_entry_picks()'s own
    404-means-'no picks yet' contract."""
    monkeypatch.setattr(ax.requests, "get", lambda url, **kwargs: _FakeResponse(404))
    assert ax.fetch_entry_picks(123, 5) is None


# ============================================================
# append_live_rank_snapshot / last_known_rank_snapshot (Roadmap Feature 6)
# ============================================================

def test_append_live_rank_snapshot_creates_file_when_missing(tmp_path):
    payload = ax.append_live_rank_snapshot(
        tmp_path, 123, 5, ts="2026-08-23T15:00:00Z", overall_rank=1_000_000,
        mini_league_pos=3, live_points=12, stale=False, data_asof="2026-08-23",
    )
    assert payload["entry_id"] == 123
    assert payload["gw"] == 5
    assert payload["stale"] is False
    assert len(payload["snapshots"]) == 1
    assert payload["snapshots"][0] == {
        "ts": "2026-08-23T15:00:00Z", "overall_rank": 1_000_000, "mini_league_pos": 3, "live_points": 12,
    }
    on_disk = json.loads((tmp_path / "live_rank_123_5.json").read_text())
    assert on_disk == payload


def test_append_live_rank_snapshot_appends_without_mutating_prior_rows(tmp_path):
    ax.append_live_rank_snapshot(
        tmp_path, 123, 5, ts="2026-08-23T15:00:00Z", overall_rank=1_000_000,
        mini_league_pos=3, live_points=0, stale=False, data_asof="2026-08-23",
    )
    payload = ax.append_live_rank_snapshot(
        tmp_path, 123, 5, ts="2026-08-23T15:10:00Z", overall_rank=999_000,
        mini_league_pos=2, live_points=12, stale=False, data_asof="2026-08-23",
    )
    assert len(payload["snapshots"]) == 2
    assert payload["snapshots"][0] == {
        "ts": "2026-08-23T15:00:00Z", "overall_rank": 1_000_000, "mini_league_pos": 3, "live_points": 0,
    }
    assert payload["snapshots"][1]["overall_rank"] == 999_000


def test_append_live_rank_snapshot_rejects_non_positive_rank(tmp_path):
    with pytest.raises(ValueError):
        ax.append_live_rank_snapshot(
            tmp_path, 123, 5, ts="2026-08-23T15:00:00Z", overall_rank=0,
            mini_league_pos=None, live_points=0, stale=False, data_asof="2026-08-23",
        )
    with pytest.raises(ValueError):
        ax.append_live_rank_snapshot(
            tmp_path, 123, 5, ts="2026-08-23T15:00:00Z", overall_rank=-5,
            mini_league_pos=None, live_points=0, stale=False, data_asof="2026-08-23",
        )


def test_append_live_rank_snapshot_records_stale_flag(tmp_path):
    payload = ax.append_live_rank_snapshot(
        tmp_path, 123, 5, ts="2026-08-23T15:20:00Z", overall_rank=999_000,
        mini_league_pos=2, live_points=12, stale=True, data_asof="2026-08-23",
    )
    assert payload["stale"] is True


def test_last_known_rank_snapshot_none_when_no_file(tmp_path):
    assert ax.last_known_rank_snapshot(tmp_path, 123, 5) is None


def test_last_known_rank_snapshot_returns_the_most_recent(tmp_path):
    ax.append_live_rank_snapshot(
        tmp_path, 123, 5, ts="2026-08-23T15:00:00Z", overall_rank=1_000_000,
        mini_league_pos=3, live_points=0, stale=False, data_asof="2026-08-23",
    )
    ax.append_live_rank_snapshot(
        tmp_path, 123, 5, ts="2026-08-23T15:10:00Z", overall_rank=999_000,
        mini_league_pos=2, live_points=12, stale=False, data_asof="2026-08-23",
    )
    last = ax.last_known_rank_snapshot(tmp_path, 123, 5)
    assert last["overall_rank"] == 999_000
    assert last["ts"] == "2026-08-23T15:10:00Z"
