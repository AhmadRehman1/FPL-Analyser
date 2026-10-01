"""Double gameweeks: a team with two fixtures under one gameweek label. ep_outputs /
uncertainty_outputs stay one row per player per fixture; everything that wants the player's
gameweek reads the per-gameweek views (schema/0023_gameweek_views.sql), so a double counts both
fixtures instead of crashing a (player, run) key or keeping one of the two rows."""

import math
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fpl_quant import squad_optimizer as so  # noqa: E402
from fpl_quant import transfer_planner as tp  # noqa: E402
from test_transfer_planner import _seed_real_squad_optimizer_candidate_pool  # noqa: E402

SECOND_FIXTURE_MU = 2.0


def _add_second_fixture(con, horizon_ep_versions, gameweek=2, club="clubA", opponent="clubC"):
    """clubA plays twice in `gameweek`: a second fixture row for each of its players."""
    ep_mv, un_mv = horizon_ep_versions[gameweek]
    con.execute(
        "INSERT INTO fact_match (match_id, season, gameweek, home_team_uid, away_team_uid, finished, "
        "competition, kickoff_time, _ingested_at) VALUES ('m2', '2026-2027', ?, ?, ?, FALSE, "
        "'Premier League', '2026-08-27', current_timestamp)", [gameweek, club, opponent],
    )
    uids = [r[0] for r in con.execute(
        "SELECT DISTINCT player_uid FROM player_alias WHERE team_code = ?", [club],
    ).fetchall()]
    for uid in uids:
        con.execute(
            "INSERT INTO ep_outputs (model_version, player_uid, fixture_match_id, ep_appearance, ep_goals, "
            "ep_assists, ep_clean_sheet, ep_goals_conceded, ep_defcon, ep_bonus, ep_saves, ep_penalty_save, "
            "ep_cards, ep_own_goal, ep_total, expected_bps) VALUES (?, ?, 'm2', 0,0,0,0,0,0,0,0,0,0,0, ?, 5.0)",
            [ep_mv, uid, SECOND_FIXTURE_MU],
        )
        con.execute(
            "INSERT INTO uncertainty_outputs (model_version, player_uid, fixture_match_id, var_appearance, "
            "var_goals, var_assists, var_clean_sheet, var_goals_conceded, var_defcon, var_bonus, var_saves, "
            "var_total, skew, excess_kurtosis, quantile_05, quantile_25, quantile_75, quantile_95) "
            "VALUES (?, ?, 'm2', 0,0,0,0,0,0,0,0, 4.0, 0,0,0,0,0,0)", [un_mv, uid],
        )
    return uids


def test_gameweek_views_sum_a_double_and_pass_a_single_through(con):
    horizon_ep_versions, _holdings = _seed_real_squad_optimizer_candidate_pool(con)
    ep_mv, un_mv = horizon_ep_versions[2]
    single = {r[0]: r for r in con.execute(
        "SELECT player_uid, ep_total, var_total, quantile_05 FROM ep_outputs o "
        "JOIN uncertainty_outputs u USING (player_uid, fixture_match_id) WHERE o.model_version = ? AND u.model_version = ?",
        [ep_mv, un_mv],
    ).fetchall()}
    doubled = _add_second_fixture(con, horizon_ep_versions)

    ep = {r[0]: r[1:] for r in con.execute(
        "SELECT player_uid, n_fixtures, ep_total FROM ep_gameweek_outputs WHERE model_version = ?", [ep_mv],
    ).fetchall()}
    un = {r[0]: r[1:] for r in con.execute(
        "SELECT player_uid, n_fixtures, var_total, quantile_05, quantile_95 FROM uncertainty_gameweek_outputs "
        "WHERE model_version = ?", [un_mv],
    ).fetchall()}

    uid = doubled[0]
    assert ep[uid] == (2, pytest.approx(single[uid][1] + SECOND_FIXTURE_MU))
    n, var, q05, q95 = un[uid]
    assert n == 2 and var == pytest.approx(single[uid][2] + 4.0)
    mean = single[uid][1] + SECOND_FIXTURE_MU
    assert q05 == pytest.approx(mean - 1.6448536 * math.sqrt(var))
    assert q95 == pytest.approx(mean + 1.6448536 * math.sqrt(var))

    other = next(u for u in single if u not in doubled)
    assert ep[other] == (1, pytest.approx(single[other][1]))
    assert un[other][2] == pytest.approx(single[other][3])  # stored quantile, untouched


def test_candidate_pool_has_one_row_per_player_in_a_double(con):
    horizon_ep_versions, _holdings = _seed_real_squad_optimizer_candidate_pool(con)
    ep_mv, un_mv = horizon_ep_versions[2]
    before = {c["player_uid"]: c for c in so.fetch_candidate_pool(con, ep_mv, un_mv, "2026-2027")}
    doubled = _add_second_fixture(con, horizon_ep_versions)

    pool = so.fetch_candidate_pool(con, ep_mv, un_mv, "2026-2027")
    uids = [c["player_uid"] for c in pool]
    assert len(uids) == len(set(uids)) == len(before)
    by_uid = {c["player_uid"]: c for c in pool}
    for uid in doubled:
        assert by_uid[uid]["mu"] == pytest.approx(before[uid]["mu"] + SECOND_FIXTURE_MU)


def test_free_hit_solves_in_a_double_gameweek(con):
    """A real squad_optimizer.run() over a pool with a double: used to crash on the
    squad_optimizer_selections (player_uid, run_id) key."""
    horizon_ep_versions, holdings = _seed_real_squad_optimizer_candidate_pool(con)
    _add_second_fixture(con, horizon_ep_versions)
    so.seed_v1_params(con)
    tp.params_mod.write_param(con, "free_hit_gain_threshold_params", 1, "2026-08-12", "min_horizon_gain", value_numeric=1.5)

    result = tp.evaluate_free_hit(
        con, date(2026, 8, 24), "2026-2027", 2, holdings, horizon_ep_versions,
        lambda_params_version=1, guardrail_params_version=1, threshold_params_version=1,
    )
    assert "gain" in result


def test_monte_carlo_sums_a_players_two_fixtures_per_realization():
    import pandas as pd

    from fpl_quant import monte_carlo as mc

    def frame(uid, match_pts, states):
        n = len(match_pts)
        return pd.DataFrame({
            "model_version": 1, "player_uid": uid, "realization_index": list(range(n)), "minutes_state": states,
            **{c: [0.0] * n for c in mc._POINT_COLUMNS if c != "total_points"},
            "total_points": match_pts,
        })

    df = pd.concat([
        frame("dgw", [2.0, 6.0], ["0", "60plus"]),        # fixture 1
        frame("dgw", [3.0, 1.0], ["1_59", "1_59"]),       # fixture 2, same realizations
        frame("single", [5.0, 4.0], ["60plus", "0"]),
    ], ignore_index=True)

    out = mc._sum_double_gameweek_fixtures(df).set_index(["player_uid", "realization_index"])
    assert len(out) == 4
    assert out.loc[("dgw", 0), "total_points"] == 5.0
    assert out.loc[("dgw", 1), "total_points"] == 7.0
    assert out.loc[("dgw", 0), "minutes_state"] == "1_59"   # the higher of "0" and "1_59"
    assert out.loc[("dgw", 1), "minutes_state"] == "60plus"
    assert out.loc[("single", 1), "total_points"] == 4.0


def test_monte_carlo_leaves_a_single_gameweek_frame_untouched():
    import pandas as pd

    from fpl_quant import monte_carlo as mc

    df = pd.DataFrame({"model_version": [1], "player_uid": ["a"], "realization_index": [0], "minutes_state": ["0"],
                       **{c: [1.0] for c in mc._POINT_COLUMNS}})
    assert mc._sum_double_gameweek_fixtures(df) is df
