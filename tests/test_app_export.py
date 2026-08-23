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
