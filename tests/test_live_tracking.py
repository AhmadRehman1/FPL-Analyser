from fpl_quant import live_tracking as lt


# ============================================================
# is_any_fixture_live
# ============================================================

def test_is_any_fixture_live_true_when_started_and_not_finished():
    fixtures = [{"started": True, "finished": False}, {"started": False, "finished": False}]
    assert lt.is_any_fixture_live(fixtures) is True


def test_is_any_fixture_live_false_once_all_finished_or_not_started():
    fixtures = [{"started": True, "finished": True}, {"started": False, "finished": False}]
    assert lt.is_any_fixture_live(fixtures) is False


def test_is_any_fixture_live_false_when_empty():
    assert lt.is_any_fixture_live([]) is False


# ============================================================
# compute_provisional_bonus / _bonus_from_bps
# ============================================================

def _bootstrap(team_by_id):
    return {"elements": [{"id": pid, "team": team} for pid, team in team_by_id.items()]}


def _live(stats_by_id):
    return {"elements": [{"id": pid, "stats": stats} for pid, stats in stats_by_id.items()]}


def test_bonus_from_bps_clear_top_three():
    bonus = lt._bonus_from_bps({1: 40, 2: 30, 3: 20, 4: 10})
    assert bonus == {1: 3, 2: 2, 3: 1}


def test_bonus_from_bps_tie_for_first_skips_second_not_third():
    bonus = lt._bonus_from_bps({1: 40, 2: 40, 3: 20})
    assert bonus == {1: 3, 2: 3, 3: 1}


def test_bonus_from_bps_three_way_tie_for_first_awards_nothing_else():
    bonus = lt._bonus_from_bps({1: 40, 2: 40, 3: 40, 4: 10})
    assert bonus == {1: 3, 2: 3, 3: 3}


def test_bonus_from_bps_tie_for_second_skips_third():
    bonus = lt._bonus_from_bps({1: 50, 2: 30, 3: 30, 4: 10})
    assert bonus == {1: 3, 2: 2, 3: 2}


def test_bonus_from_bps_tie_for_third():
    bonus = lt._bonus_from_bps({1: 50, 2: 40, 3: 20, 4: 20})
    assert bonus == {1: 3, 2: 2, 3: 1, 4: 1}


def test_compute_provisional_bonus_only_for_started_unfinished_fixtures():
    bootstrap = _bootstrap({1: 10, 2: 10, 3: 20, 4: 20})
    fixtures = [
        {"team_h": 10, "team_a": 30, "started": True, "finished": False},   # live -> compute
        {"team_h": 20, "team_a": 40, "started": True, "finished": True},    # finished -> skip
    ]
    live = _live({
        1: {"bps": 40, "minutes": 90}, 2: {"bps": 20, "minutes": 90},
        3: {"bps": 99, "minutes": 90}, 4: {"bps": 1, "minutes": 90},
    })
    out = lt.compute_provisional_bonus(bootstrap, fixtures, live)
    # only players 1 and 2 actually feature in the live fixture (team 30 has no squad here) --
    # still both get awarded (1st=3, 2nd=2) since the rule doesn't require exactly 3 players
    assert out == {1: 3, 2: 2}
    assert 3 not in out and 4 not in out  # finished fixture's players excluded entirely


def test_compute_provisional_bonus_excludes_players_with_zero_minutes():
    bootstrap = _bootstrap({1: 10, 2: 10})
    fixtures = [{"team_h": 10, "team_a": 30, "started": True, "finished": False}]
    live = _live({1: {"bps": 40, "minutes": 90}, 2: {"bps": 999, "minutes": 0}})
    out = lt.compute_provisional_bonus(bootstrap, fixtures, live)
    assert out == {1: 3}


def test_compute_provisional_bonus_skips_fixtures_not_yet_started():
    bootstrap = _bootstrap({1: 10})
    fixtures = [{"team_h": 10, "team_a": 30, "started": False, "finished": False}]
    live = _live({1: {"bps": 40, "minutes": 0}})
    assert lt.compute_provisional_bonus(bootstrap, fixtures, live) == {}


# ============================================================
# build_live_squad_rows
# ============================================================

def test_build_live_squad_rows_maps_picks_to_live_stats():
    picks = [
        {"element": 1, "position": 1, "multiplier": 2},
        {"element": 2, "position": 12, "multiplier": 1},
    ]
    live = _live({1: {"total_points": 9}, 2: {"total_points": 3}})
    rows = lt.build_live_squad_rows(picks, live)
    assert rows[0] == {"player_id": 1, "in_xi": True, "multiplier": 2, "event_points": 9}
    assert rows[1] == {"player_id": 2, "in_xi": False, "multiplier": 1, "event_points": 3}


def test_build_live_squad_rows_defaults_to_zero_for_missing_live_stats():
    picks = [{"element": 99, "position": 1, "multiplier": 1}]
    rows = lt.build_live_squad_rows(picks, {"elements": []})
    assert rows[0]["event_points"] == 0


# ============================================================
# compute_live_squad_total
# ============================================================

def test_compute_live_squad_total_applies_captain_multiplier_and_bonus():
    squad = [
        {"player_id": 1, "web_name": "A", "in_xi": True, "multiplier": 2, "event_points": 6},
        {"player_id": 2, "web_name": "B", "in_xi": True, "multiplier": 1, "event_points": 4},
        {"player_id": 3, "web_name": "C", "in_xi": False, "multiplier": 1, "event_points": 100},
    ]
    out = lt.compute_live_squad_total(squad, provisional_bonus_by_id={1: 2})
    # player 1: (6 + 2 bonus) * 2 = 16; player 2: (4 + 0) * 1 = 4; bench excluded
    assert out["total"] == 20
    p1 = next(p for p in out["players"] if p["player_id"] == 1)
    assert p1["live_points"] == 8 and p1["contribution"] == 16 and p1["provisional_bonus"] == 2


def test_compute_live_squad_total_zero_bonus_when_not_in_provisional_dict():
    squad = [{"player_id": 5, "web_name": "X", "in_xi": True, "multiplier": 1, "event_points": 10}]
    out = lt.compute_live_squad_total(squad, provisional_bonus_by_id={})
    assert out["total"] == 10


# ============================================================
# estimate_live_rank
# ============================================================

def test_estimate_live_rank_basic_percentile_and_projection():
    out = lt.estimate_live_rank(your_points=50, sample_points=[40, 45, 60, 70], total_players=1_000_000)
    # beats 2 of 4 (40, 45) -> percentile 50%
    assert out["sample_size"] == 4
    assert out["percentile"] == 50.0
    assert out["estimated_rank"] == 500_000


def test_estimate_live_rank_handles_ties_as_half_credit():
    out = lt.estimate_live_rank(your_points=50, sample_points=[50, 50], total_players=100)
    assert out["percentile"] == 50.0


def test_estimate_live_rank_empty_sample_returns_none_not_fabricated():
    out = lt.estimate_live_rank(your_points=50, sample_points=[], total_players=1_000_000)
    assert out == {"sample_size": 0, "percentile": None, "estimated_rank": None}


def test_estimate_live_rank_no_total_players_still_gives_percentile():
    out = lt.estimate_live_rank(your_points=50, sample_points=[10, 20], total_players=None)
    assert out["percentile"] == 100.0
    assert out["estimated_rank"] is None


def test_estimate_live_rank_beating_the_whole_sample_clamps_to_rank_1_not_0():
    # percentile=1.0 -> (1-percentile)*total_players == 0, which isn't a real FPL rank
    # (ranks are 1-indexed) and is falsy, so a naive caller would silently drop it.
    out = lt.estimate_live_rank(your_points=100, sample_points=[10, 20, 30], total_players=1_000_000)
    assert out["percentile"] == 100.0
    assert out["estimated_rank"] == 1


# ============================================================
# build_live_fixture_rows
# ============================================================

_TEAMS = [
    {"id": 1, "name": "Arsenal", "short_name": "ARS"},
    {"id": 2, "name": "Chelsea", "short_name": "CHE"},
    {"id": 3, "name": "Liverpool", "short_name": "LIV"},
]


def test_build_live_fixture_rows_only_started_fixtures():
    fixtures = [
        {"id": 100, "team_h": 1, "team_a": 2, "started": True, "finished": False, "minute": 34,
         "team_h_score": 1, "team_a_score": 0, "kickoff_time": "2026-08-24T15:00:00Z"},
        {"id": 101, "team_h": 3, "team_a": 1, "started": False, "finished": False, "minute": None},
        {"id": 102, "team_h": 2, "team_a": 3, "started": True, "finished": True, "minute": 90,
         "team_h_score": 2, "team_a_score": 2, "kickoff_time": "2026-08-24T17:30:00Z"},
    ]
    rows = lt.build_live_fixture_rows(fixtures, {"teams": _TEAMS})
    # only the two started fixtures appear; the not-started one is dropped
    assert [r["id"] for r in rows] == [100, 102]
    assert rows[0]["home"]["short_name"] == "ARS"
    assert rows[0]["away"]["short_name"] == "CHE"
    assert rows[0]["home"]["score"] == 1
    assert rows[0]["away"]["score"] == 0
    assert rows[0]["started"] is True and rows[0]["finished"] is False
    assert rows[0]["minute"] == 34
    assert rows[1]["finished"] is True


def test_build_live_fixture_rows_empty_when_nothing_started():
    fixtures = [{"id": 99, "team_h": 1, "team_a": 2, "started": False, "finished": False}]
    assert lt.build_live_fixture_rows(fixtures, {"teams": _TEAMS}) == []


# ============================================================
# build_live_event_rows
# ============================================================

def _bs(elements):
    return {"elements": elements, "teams": _TEAMS}


def test_build_live_event_rows_goals_and_assists_skip_zero_minutes():
    bootstrap = _bs([
        {"id": 10, "web_name": "Saka", "team": 1},
        {"id": 11, "web_name": "Palmer", "team": 2},
        {"id": 12, "web_name": "Unused", "team": 1},
    ])
    fixtures = [
        {"id": 100, "team_h": 1, "team_a": 2, "started": True, "finished": False},
    ]
    live = {"elements": [
        {"id": 10, "stats": {"minutes": 34, "goals_scored": 1, "assists": 0, "own_goals": 0,
                             "penalties_saved": 0, "penalties_missed": 0}},
        {"id": 11, "stats": {"minutes": 34, "goals_scored": 0, "assists": 1, "own_goals": 0}},
        # 0 minutes -> excluded even though stats exist
        {"id": 12, "stats": {"minutes": 0, "goals_scored": 1, "assists": 0}},
    ]}
    rows = lt.build_live_event_rows(bootstrap, live, fixtures)
    events = {(r["player_id"], r["event"], r["count"], r["fixture_id"]) for r in rows}
    assert events == {(10, "goal", 1, 100), (11, "assist", 1, 100)}


def test_build_live_event_rows_penalty_events_labeled_correctly():
    bootstrap = _bs([{"id": 20, "web_name": "Kepa", "team": 1}])
    fixtures = [{"id": 200, "team_h": 1, "team_a": 2, "started": True, "finished": False}]
    live = {"elements": [{"id": 20, "stats": {"minutes": 90, "goals_scored": 0, "assists": 0,
                                              "own_goals": 0, "penalties_saved": 1,
                                              "penalties_missed": 1}}]}
    rows = lt.build_live_event_rows(bootstrap, live, fixtures)
    labels = sorted(r["event"] for r in rows)
    assert labels == ["pen_missed", "pen_saved"]


def test_build_live_event_rows_empty_when_no_live_fixtures():
    bootstrap = _bs([{"id": 10, "web_name": "Saka", "team": 1}])
    # nothing started -> no fixture mapping, but a scorer with minutes still surfaces (no fixture_id)
    fixtures = [{"id": 100, "team_h": 1, "team_a": 2, "started": False, "finished": False}]
    live = {"elements": [{"id": 10, "stats": {"minutes": 0, "goals_scored": 1, "assists": 0}}]}
    assert lt.build_live_event_rows(bootstrap, live, fixtures) == []


def test_build_live_event_rows_fixture_id_none_in_double_gameweek():
    # A team playing two started fixtures at once can't be attributed to one -> fixture_id None.
    bootstrap = _bs([{"id": 10, "web_name": "Saka", "team": 1}])
    fixtures = [
        {"id": 100, "team_h": 1, "team_a": 2, "started": True, "finished": False},
        {"id": 101, "team_h": 3, "team_a": 1, "started": True, "finished": False},
    ]
    live = {"elements": [{"id": 10, "stats": {"minutes": 45, "goals_scored": 1, "assists": 0}}]}
    rows = lt.build_live_event_rows(bootstrap, live, fixtures)
    assert len(rows) == 1
    assert rows[0]["fixture_id"] is None  # ambiguous -> no fixture attribution, scorer still listed
