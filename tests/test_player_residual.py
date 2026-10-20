"""Lever L3, the per-player residual (docs/plans/2026-10_beat_the_template.md, Phase 7): each
eligible player's own mean residual over the gameweeks G-7..G-2, realized minus the projection made
at each of those deadlines, shrunk by n / (n + 6); the shared apply inside expected_points.run();
each harness's own history (the walk-forward's earlier steps, the season sim's earlier plan runs);
and the checks (leak, shrinkage, eligibility, paths, flags, control identity)."""

import sys
from pathlib import Path

import pytest

from fpl_quant import backtest as bt
from fpl_quant import expected_points as ep
from fpl_quant import residual_levers as rl
from fpl_quant import transfer_planner as tp
from tests.test_backtest import _SEASON_SIM_VERSIONS, _seed_season_simulation_league
from tests.test_band_bias import WALK
from tests.test_crowd_prior import _snapshots

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_season_sim_arm as arm  # noqa: E402
import run_walkforward as rw  # noqa: E402

SEASON = "2025-2026"
# the declared values except eligibility, so every player in the small test league is adjusted
OPEN = {**rl.PLAYER_RESIDUAL_V1, "min_minutes": 0}


def _version(con, values=OPEN):
    return rl.player_residual_version(con, rw.EXPERIMENT_EFFECTIVE_DATE, values)


def _realized(con, season, gw):
    return dict(con.execute(
        "SELECT player_uid, event_points FROM fact_player_season_stats WHERE season = ? AND gw = ? "
        "AND event_points IS NOT NULL", [season, gw],
    ).fetchall())


def _expected(con, season, deadline, history, values=OPEN):
    """The adjustment re-derived by hand: the residuals over the window's gameweeks in the history,
    their mean shrunk by n / (n + pseudo_gws)."""
    last = deadline - int(values["lag"])
    gws = [gw for gw in range(max(2, last - int(values["window"]) + 1), last + 1) if gw in history]
    resid: dict[str, list[float]] = {}
    for gw in gws:
        for uid, points in _realized(con, season, gw).items():
            if uid in history[gw]:
                resid.setdefault(uid, []).append(points - history[gw][uid])
    return {uid: sum(r) / len(r) * len(r) / (len(r) + values["pseudo_gws"]) for uid, r in resid.items()}


def _ep_rows(con, mv):
    return {uid: (total, adj) for uid, total, adj in con.execute(
        "SELECT player_uid, ep_total, ep_player_adj FROM ep_gameweek_outputs WHERE model_version = ?", [mv],
    ).fetchall()}


def _ep(con, season, gameweek, asof, ts, mm, version=None, history=None):
    with bt.asof_scope(con, season, gameweek):
        mv = ep.run(con, asof, season, gameweek, ts, mm, 1, 1, 1, player_residual_params_version=version,
                    player_residual_history=history, lookback_seasons=bt.lookback_seasons_for(season))
    return _ep_rows(con, mv), mv


def _flat_history(con, gws, projection=3.0):
    """Every player projected at `projection` in each of these gameweeks."""
    uids = [uid for (uid,) in con.execute("SELECT player_uid FROM dim_player ORDER BY player_uid").fetchall()]
    return {gw: {uid: projection for uid in uids} for gw in gws}


# ------------------------------------------------------------------ the adjustment


def test_window_ends_at_g_minus_lag_and_never_reaches_gw1():
    assert rl.player_residual_window(10, 6, 2) == [3, 4, 5, 6, 7, 8]
    assert rl.player_residual_window(8, 6, 2) == [2, 3, 4, 5, 6]
    assert rl.player_residual_window(4, 6, 2) == [2]
    assert rl.player_residual_window(3, 6, 2) == []


def test_adjustment_is_each_players_shrunk_mean_residual_over_the_window(con):
    _seed_season_simulation_league(con)
    history = _flat_history(con, [2, 3, 4, 5])
    history[3] = {uid: ep for uid, ep in history[3].items() if uid != "p1"}  # p1 unprojected in GW3: n = 2
    history[5] = {uid: 50.0 for uid in history[5]}  # GW5 = G-1: outside the window, never read
    adj = rl.player_residual_adjustments(con, SEASON, 6, history, OPEN)
    expected = _expected(con, SEASON, 6, history)
    assert set(adj) == set(expected) and len(adj) == 18
    for uid, value in adj.items():
        assert value == pytest.approx(expected[uid])
    # p1 by hand: two residuals, shrunk by 2 / (2 + 6)
    r2, r4 = _realized(con, SEASON, 2)["p1"] - 3.0, _realized(con, SEASON, 4)["p1"] - 3.0
    assert adj["p1"] == pytest.approx((r2 + r4) / 2 * 2 / 8)
    # an empty window, or a history that misses it, adjusts no one
    assert rl.player_residual_adjustments(con, SEASON, 3, history, OPEN) == {}
    assert rl.player_residual_adjustments(con, SEASON, 6, {5: history[5]}, OPEN) == {}


def test_rows_g_minus_1_and_g_are_never_read(con):
    # the leak test: GW G's and G-1's realized points (and G-1's projection) can't move G's
    # adjustment; G-2's can
    _seed_season_simulation_league(con)
    history = _flat_history(con, [2, 3, 4, 5, 6])
    before = rl.player_residual_adjustments(con, SEASON, 6, history, OPEN)
    con.execute("UPDATE fact_player_season_stats SET event_points = 40 WHERE season = ? AND gw IN (5, 6)", [SEASON])
    con.execute("UPDATE fact_player_match_stats SET minutes_played = 0 WHERE match_id LIKE 'm_2025-2026_5_%'")
    assert rl.player_residual_adjustments(con, SEASON, 6, history, OPEN) == before
    con.execute("UPDATE fact_player_season_stats SET event_points = 40 WHERE season = ? AND gw = 4", [SEASON])
    after = rl.player_residual_adjustments(con, SEASON, 6, history, OPEN)
    assert all(after[uid] > before[uid] for uid in before)


def test_only_players_with_900_premier_league_minutes_last_season_and_this_one_are_adjusted(con):
    # nailed players play 90 a week, fringe players none; 2024-25 gives 7 x 90 = 630, so at GW5's
    # deadline GW1-3 of 2025-26 make exactly 900 (GW4, row G-1, never counts)
    _seed_season_simulation_league(con)
    history = _flat_history(con, [2, 3, 4, 5])
    nailed = {uid for (uid,) in con.execute(
        "SELECT DISTINCT player_uid FROM fact_player_match_stats WHERE minutes_played = 90").fetchall()}
    v1 = rl.PLAYER_RESIDUAL_V1
    assert set(rl.player_residual_adjustments(con, SEASON, 5, history, v1)) == nailed
    assert rl.player_residual_adjustments(con, SEASON, 4, history, v1) == {}  # 810 minutes
    # a cup tie doesn't count: one 2025-26 GW1 match moved out of the league drops its players to 810
    con.execute("UPDATE fact_match SET competition = 'FA Cup' WHERE match_id = 'm_2025-2026_1_0'")
    cup = {uid for (uid,) in con.execute(
        "SELECT player_uid FROM fact_player_match_stats WHERE match_id = 'm_2025-2026_1_0'").fetchall()}
    assert set(rl.player_residual_adjustments(con, SEASON, 5, history, v1)) == nailed - cup
    # 2024-25 has no earlier season: its own minutes alone (5 x 90 by GW7) don't reach 900
    assert rl.player_residual_adjustments(con, "2024-2025", 7, _flat_history(con, [2, 3, 4, 5]), v1) == {}


def test_projection_history_reads_the_projection_before_l3(con):
    _seed_season_simulation_league(con)
    asof, ts, mm = _snapshots(con, SEASON, 3)
    rows, mv = _ep(con, SEASON, 3, asof, ts, mm)
    con.execute("UPDATE ep_outputs SET ep_total = ep_total + 1.25, ep_player_adj = 1.25 WHERE model_version = ?", [mv])
    history = rl.projection_history(con, {3: mv})
    assert set(history) == {3} and history[3] == {uid: pytest.approx(total) for uid, (total, _) in rows.items()}
    with pytest.raises(ValueError, match="lag"):
        rl.player_residual_adjustments(con, SEASON, 6, history, {**OPEN, "lag": 1})


# ------------------------------------------------------------------ the apply inside expected_points.run()


def test_player_residual_adds_the_adjustment_floored_at_zero_ep(con):
    _seed_season_simulation_league(con)
    v = _version(con)
    history = _flat_history(con, [2, 3])
    history[2]["p12"] = history[3]["p12"] = 60.0  # a fringe midfielder projected far above his score
    asof, ts, mm = _snapshots(con, SEASON, 5)
    control, _ = _ep(con, SEASON, 5, asof, ts, mm)
    lever, mv = _ep(con, SEASON, 5, asof, ts, mm, v, history)
    expected = _expected(con, SEASON, 5, history)
    assert expected["p12"] < -10
    for uid, (total, adj) in lever.items():
        pre = control[uid][0]
        want = max(expected.get(uid, 0.0), -max(pre, 0.0))  # one fixture each
        assert adj == pytest.approx(want) and total == pytest.approx(pre + want)
    assert lever["p12"][0] == pytest.approx(0.0)
    assert con.execute("SELECT player_residual_params_version FROM ep_model_versions WHERE model_version = ?",
                       [mv]).fetchone()[0] == v


def test_without_the_flag_ep_is_identical_and_the_version_needs_a_history(con):
    _seed_season_simulation_league(con)
    v = _version(con)
    asof, ts, mm = _snapshots(con, SEASON, 5)
    control, mv_control = _ep(con, SEASON, 5, asof, ts, mm)
    empty, _ = _ep(con, SEASON, 5, asof, ts, mm, v, {})
    assert empty == control
    assert con.execute("SELECT count(*) FROM ep_outputs WHERE model_version = ? AND ep_player_adj != 0.0",
                       [mv_control]).fetchone()[0] == 0
    # TypeError, so a planning horizon (which skips a gameweek on ValueError) can't swallow it
    with pytest.raises(TypeError, match="history"):
        _ep(con, SEASON, 5, asof, ts, mm, v, None)


def test_monte_carlo_summary_carries_the_player_adjustment(con):
    from fpl_quant import monte_carlo

    _seed_season_simulation_league(con)
    v = _version(con)
    run_id = con.execute("INSERT INTO backtest_runs (warm_up_gameweeks) VALUES (0) RETURNING backtest_run_id").fetchone()[0]
    bt.run_gameweek_step(con, run_id, SEASON, 5, n_antithetic_pairs=200, **WALK)
    ts, mm, ep_control, un, so = con.execute(
        "SELECT ts_model_version, mm_model_version, ep_model_version, un_model_version, so_run_id "
        "FROM backtest_gameweek_steps WHERE backtest_run_id = ?", [run_id]).fetchone()
    with bt.asof_scope(con, SEASON, 5) as deadline:
        asof = deadline.date()
        ep_lever = ep.run(con, asof, SEASON, 5, ts, mm, 1, 1, 1, player_residual_params_version=v,
                          player_residual_history=_flat_history(con, [2, 3]),
                          lookback_seasons=bt.lookback_seasons_for(SEASON))
        mcs = [monte_carlo.run(con, asof, so, mv, mm, ts, un, 1, 1, 1, n_antithetic_pairs=200,
                               season_priority=bt.lookback_seasons_for(SEASON)) for mv in (ep_control, ep_lever)]
    summary = [dict((uid, (mean, var)) for uid, mean, var in con.execute(
        "SELECT player_uid, mean_total, var_total FROM monte_carlo_player_summary WHERE model_version = ?", [mc]).fetchall())
        for mc in mcs]
    lever = _ep_rows(con, ep_lever)
    assert summary[0] and any(lever[uid][1] != 0.0 for uid in summary[0])
    for uid, (mean, var) in summary[0].items():
        assert summary[1][uid][0] == pytest.approx(mean + lever[uid][1]) and summary[1][uid][1] == pytest.approx(var)


# ------------------------------------------------------------------ each harness's history


def test_walk_forward_reads_its_own_earlier_steps_and_the_export_takes_l3_out(tmp_path):
    from fpl_quant import db

    con = db.connect(tmp_path / "w.duckdb")
    _seed_season_simulation_league(con)
    v = _version(con)
    run_id = con.execute("INSERT INTO backtest_runs (warm_up_gameweeks) VALUES (0) RETURNING backtest_run_id").fetchone()[0]
    other = con.execute("INSERT INTO backtest_runs (warm_up_gameweeks) VALUES (0) RETURNING backtest_run_id").fetchone()[0]
    for gw in (2, 4):
        bt.run_gameweek_step(con, run_id, SEASON, gw, n_antithetic_pairs=200, player_residual_params_version=v, **WALK)
    steps = dict(con.execute("SELECT gameweek, ep_model_version FROM backtest_gameweek_steps WHERE backtest_run_id = ?",
                             [run_id]).fetchall())
    # another run's steps are not this run's history
    con.execute("INSERT INTO backtest_gameweek_steps (backtest_run_id, season, gameweek, tier, data_asof, ep_model_version) "
                "VALUES (?, ?, 3, 'warm', now(), ?)", [other, SEASON, steps[4]])
    assert rl.walk_forward_ep_versions(con, run_id, SEASON) == steps
    # GW2's step has no window yet; GW4's averages GW2's residual against GW2's own step projection
    assert all(adj == 0.0 for _, adj in _ep_rows(con, steps[2]).values())
    expected = _expected(con, SEASON, 4, rl.projection_history(con, {2: steps[2]}))
    step4 = _ep_rows(con, steps[4])
    assert expected and any(adj != 0.0 for _, adj in step4.values())
    for uid, (total, adj) in step4.items():
        assert adj == pytest.approx(max(expected.get(uid, 0.0), -max(total - adj, 0.0)))
    rows = [r for r in rl.residual_rows(con, run_id, lag=2) if r["gw"] == 4]
    assert rows
    for r in rows:
        total, adj = step4[r["player_uid"]]
        assert r["player_adj"] == pytest.approx(adj) and r["ep_pre_adjustment"] == pytest.approx(total - adj)
    con.close()


def test_the_planner_horizon_carries_the_deadlines_adjustment(con):
    _seed_season_simulation_league(con)
    v = _version(con)
    history = _flat_history(con, [2, 3])
    expected = _expected(con, SEASON, 5, history)
    asof, ts, mm = _snapshots(con, SEASON, 5, schedule_horizon_gameweeks=2)
    with bt.asof_scope(con, SEASON, 5, schedule_horizon_gameweeks=2):
        horizon = tp.compute_horizon_ep(con, asof, SEASON, 5, ts, mm, 2, 1, 1, 1, 1, 1,
                                        player_residual_params_version=v, player_residual_history=history)
    assert set(horizon) == {5, 6}
    for ep_mv, _ in horizon.values():
        for uid, (total, adj) in _ep_rows(con, ep_mv).items():
            assert adj == pytest.approx(max(expected.get(uid, 0.0), -max(total - adj, 0.0)))
    with pytest.raises(TypeError):
        tp.compute_horizon_ep(con, asof, SEASON, 5, ts, mm, 1, 1, 1, 1, 1, 1, player_residual_params_version=v)


def test_season_sim_feeds_its_own_earlier_deadlines_into_l3(con):
    _seed_season_simulation_league(con)
    v = _version(con)
    bt.run_season_simulation(con, SEASON, start_gameweek=2, end_gameweek=5, n_antithetic_pairs=200,
                             player_residual_params_version=v, **_SEASON_SIM_VERSIONS)
    boot = con.execute("SELECT ep_model_version FROM squad_optimizer_runs WHERE target_season = ? AND target_gameweek = 2 "
                       "ORDER BY run_id DESC LIMIT 1", [SEASON]).fetchone()[0]
    plan = {gw: con.execute("SELECT run_id FROM transfer_plan_runs WHERE target_gameweek = ? ORDER BY run_id DESC LIMIT 1",
                            [gw]).fetchone()[0] for gw in (3, 4, 5)}
    own = {2: boot, 3: tp.plan_gameweek_versions(con, plan[3], 3)[0]}
    for gw, history_gws in ((4, [2]), (5, [2, 3])):
        expected = _expected(con, SEASON, gw, rl.projection_history(con, {g: own[g] for g in history_gws}))
        rows = _ep_rows(con, tp.plan_gameweek_versions(con, plan[gw], gw)[0])
        assert expected and any(adj != 0.0 for _, adj in rows.values())
        for uid, (total, adj) in rows.items():
            assert adj == pytest.approx(max(expected.get(uid, 0.0), -max(total - adj, 0.0)))
    # GW3's deadline has no window yet (G-2 = GW1 never counts)
    assert all(adj == 0.0 for _, adj in _ep_rows(con, own[3]).values())


# ------------------------------------------------------------------ flags (Req 6)


def test_player_residual_flag_mints_and_reuses_v1_in_both_scripts(con):
    assert "player_residual_params_version" not in rw._experiment_versions(con, rw._parse_args([]))
    out = rw._experiment_versions(con, rw._parse_args(["--player-residual"]))
    assert set(out) == {"player_residual_params_version"}
    assert rl.resolve_player_residual(con, out["player_residual_params_version"]) == rl.PLAYER_RESIDUAL_V1
    assert rw._experiment_versions(con, rw._parse_args(["--player-residual"])) == out
    versions, changed = arm.arm_versions(con, arm._parse_args(["--label", "x", "--player-residual"]))
    assert changed == out and versions["player_residual_params_version"] == out["player_residual_params_version"]
    control, _ = arm.arm_versions(con, arm._parse_args(["--label", "control"]))
    assert "player_residual_params_version" not in control
