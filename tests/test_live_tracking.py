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
