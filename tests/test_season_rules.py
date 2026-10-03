"""FPL rules by season (docs/reports/2026-10_open_issues.md, issue 1): 2024-25 was planned and
scored under 2025-26 rules everywhere but DefCon."""

from fpl_quant import backtest as bt
from fpl_quant import expected_points as ep
from fpl_quant import season_rules
from fpl_quant import transfer_planner as tp
from tests.test_backtest import _seed_plan_run_with_recommendations
from tests.test_monte_carlo import _seed_asymmetric_fixture
from tests.test_transfer_planner import _seed_minimal_squad_optimizer_run, _seed_transfer_plan_run_for_apply


def test_gameweek_19_is_the_first_halfs_last_week():
    assert season_rules.half_of(19) == 1 and season_rules.half_of(20) == 2
    assert season_rules.chip_window("2025-2026", "bench_boost", 19) == (1, 19)
    assert season_rules.chip_window("2025-2026", "bench_boost", 20) == (20, 38)


def test_2024_25_had_one_free_hit_bench_boost_and_triple_captain_but_two_wildcards():
    used_first_half = ["triple_captain", "bench_boost", "free_hit", "wildcard"]
    for chip in ("triple_captain", "bench_boost", "free_hit"):
        assert not season_rules.chip_available("2024-2025", chip, 25, used_first_half, [])
        assert season_rules.chip_available("2025-2026", chip, 25, used_first_half, [])
    assert season_rules.chip_available("2024-2025", "wildcard", 25, used_first_half, [])
    # a whole-season chip used after GW19 is spent for the first half's weeks too
    assert not season_rules.chip_available("2024-2025", "free_hit", 5, [], ["free_hit"])
    assert season_rules.chip_available("2024-2025", "wildcard", 5, [], ["wildcard"])


def test_rules_for_unknown_seasons_and_the_defcon_flag():
    assert season_rules.rules_for(None) is season_rules.RULES[max(season_rules.RULES)]
    assert season_rules.rules_for("2030-2031").defcon
    assert not season_rules.rules_for("2023-2024").defcon
    assert not ep.defcon_in_force("2024-2025") and ep.defcon_in_force("2025-2026")


def test_a_chip_played_in_gameweek_19_is_recorded_in_the_first_set(con):
    so_run, _, _ = _seed_minimal_squad_optimizer_run(con)
    state_version = tp.bootstrap_from_squad_optimizer_run(con, so_run)
    run_id = _seed_transfer_plan_run_for_apply(con, state_version, target_gameweek=19)
    new_state = tp.apply_recommendation(con, run_id, accept_chip="bench_boost")
    set1, set2 = con.execute(
        "SELECT chips_used_set1, chips_used_set2 FROM manager_state_versions WHERE state_version = ?", [new_state],
    ).fetchone()
    assert (set1, set2) == ('["bench_boost"]', "[]")


def test_decide_gameweek_action_follows_the_plan_seasons_chip_allowance(con):
    run_id = _seed_plan_run_with_recommendations(con, recommended_chips=("triple_captain",), target_gameweek=25)
    for season, expected in (("2024-2025", None), ("2025-2026", "triple_captain")):
        con.execute("UPDATE transfer_plan_runs SET target_season = ? WHERE run_id = ?", [season, run_id])
        _rank, chip = bt._decide_gameweek_action(
            con, run_id, {"triple_captain"}, set(), target_gameweek=25, accept_transfer_if_net_value_above=0.0,
        )
        assert chip == expected, season


def test_bonus_points_use_each_seasons_weights(con):
    """+1 BPS per 2 clearances, blocks and interceptions until 2025-26, per 3 from 2026-27; a
    keeper save was a flat 2 BPS in 2024-25 and 3 (inside the box) from 2025-26."""
    ts_mv, _mm, _ep, _sq = _seed_asymmetric_fixture(con)
    mean_minutes = ep._mean_minutes_by_bucket(con)
    seasons = ["2026-2027", "2025-2026"]

    def expected_bps(uid, position, team, season):
        return ep.compute_player_fixture_components(
            con, uid, position, team, "m1", 0.05, 0.10, 0.85, ts_mv, 1, 1, seasons, mean_minutes, target_season=season,
        )["expected_bps"]

    assert expected_bps("dfn", "Defender", "dog", "2025-2026") > expected_bps("dfn", "Defender", "dog", "2026-2027") + 0.5
    assert expected_bps("gkp", "Goalkeeper", "dog", "2024-2025") < expected_bps("gkp", "Goalkeeper", "dog", "2025-2026") - 0.5
