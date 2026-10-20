"""Lever L3, the per-player residual (docs/plans/2026-10_beat_the_template.md, Phase 7): each player's
own mean residual over his last gameweeks of the season, shrunk by n / (n + pseudo_gws), read from the
harness's own deadline-time EP (the walk-forward's earlier steps of the same run, the season sim's own
plan runs); and the Req 6 / Req 11 checks (flags, control identity, leak, missing data, the
pre-L3 projection, one adjustment in every harness)."""

import sys
from pathlib import Path

import pytest

from fpl_quant import backtest as bt
from fpl_quant import expected_points as ep
from fpl_quant import residual_levers as rl
from fpl_quant import transfer_planner as tp
from tests.test_backtest import _SEASON_SIM_VERSIONS
from tests.test_band_bias import WALK
from tests.test_crowd_prior import _crowd_league, _snapshots

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_season_sim_arm as arm  # noqa: E402
import run_walkforward as rw  # noqa: E402

SEASON = "2025-2026"
# the test league is small: a low minutes floor keeps every nailed player eligible
LOW_FLOOR = {"window": 6, "pseudo_gws": 6, "min_minutes": 90}


def _run_id(con):
    return con.execute("INSERT INTO backtest_runs (warm_up_gameweeks) VALUES (0) RETURNING backtest_run_id").fetchone()[0]


def _step_mv(con, run_id, gameweek):
    return con.execute(
        "SELECT ep_model_version FROM backtest_gameweek_steps WHERE backtest_run_id = ? AND gameweek = ?",
        [run_id, gameweek],
    ).fetchone()[0]


def _ep_rows(con, mv):
    return {uid: (total, adj) for uid, total, adj in con.execute(
        "SELECT player_uid, ep_total, ep_player_adj FROM ep_gameweek_outputs WHERE model_version = ?", [mv],
    ).fetchall()}


def _expected(con, mv_by_gw, deadline, params=LOW_FLOOR):
    """The adjustment by hand: realized - (ep_total - ep_player_adj) for each history gameweek in the
    window, mean times n / (n + pseudo_gws), eligible players only."""
    minutes = rl.league_minutes(con, SEASON, deadline)
    acc: dict[str, list[float]] = {}
    for gw, mv in mv_by_gw.items():
        if not deadline - params["window"] <= gw < deadline:
            continue
        for uid, (total, adj) in _ep_rows(con, mv).items():
            row = con.execute("SELECT event_points FROM fact_player_season_stats WHERE player_uid = ? AND season = ? "
                              "AND gw = ?", [uid, SEASON, gw]).fetchone()
            if row is None or row[0] is None or minutes.get(uid, 0) < params["min_minutes"]:
                continue
            acc.setdefault(uid, []).append(row[0] - (total - adj))
    return {uid: sum(r) / len(r) * len(r) / (len(r) + params["pseudo_gws"]) for uid, r in acc.items()}


def _walk(con, gameweeks, version=None, run_id=None):
    run_id = run_id or _run_id(con)
    for gw in gameweeks:
        bt.run_gameweek_step(con, run_id, SEASON, gw, n_antithetic_pairs=200,
                             player_residual_params_version=version, **WALK)
    return run_id


# ------------------------------------------------------------------ the adjustment


def test_adjustment_is_the_shrunk_mean_residual_of_the_window(con):
    _crowd_league(con)
    run_id = _walk(con, [1, 2, 3])
    history = {gw: _step_mv(con, run_id, gw) for gw in (1, 2, 3)}
    adj = rl.player_residual_adjustments(con, SEASON, 4, history, **LOW_FLOOR)
    expected = _expected(con, history, 4)
    assert adj and set(adj) == set(expected)
    for uid, value in expected.items():
        assert adj[uid] == pytest.approx(value)
    # a window of 2 drops GW1
    narrow = rl.player_residual_adjustments(con, SEASON, 4, history, 2, 6, 90)
    assert narrow == pytest.approx(_expected(con, history, 4, {**LOW_FLOOR, "window": 2}))
    assert narrow != pytest.approx(adj)
    # n / (n + pseudo_gws): a heavier prior shrinks every adjustment toward zero
    heavy = rl.player_residual_adjustments(con, SEASON, 4, history, 6, 60, 90)
    assert all(abs(heavy[uid]) < abs(adj[uid]) or adj[uid] == 0 for uid in adj)


def test_the_minutes_floor_counts_this_season_before_the_deadline_and_the_last_one(con):
    _crowd_league(con)
    minutes = rl.league_minutes(con, SEASON, 3)
    by_hand = dict(con.execute(
        "SELECT s.player_uid, sum(s.minutes_played) FROM fact_player_match_stats s JOIN fact_match m USING (match_id) "
        "WHERE (m.season = ? AND m.gameweek < 3) OR m.season = '2024-2025' GROUP BY s.player_uid", [SEASON]).fetchall())
    assert minutes == by_hand
    run_id = _walk(con, [1, 2])
    history = {gw: _step_mv(con, run_id, gw) for gw in (1, 2)}
    floor = sorted(minutes.values())[len(minutes) // 2]
    adj = rl.player_residual_adjustments(con, SEASON, 3, history, 6, 6, floor)
    assert adj and all(minutes[uid] >= floor for uid in adj)
    assert any(m < floor for m in minutes.values())


def test_no_history_or_no_window_gameweek_means_no_adjustment_and_the_deadline_is_never_read(con):
    _crowd_league(con)
    run_id = _walk(con, [2, 3, 4])
    history = {gw: _step_mv(con, run_id, gw) for gw in (2, 3, 4)}
    assert rl.player_residual_adjustments(con, SEASON, 3, {}, **LOW_FLOOR) == {}
    # Req 11's leak test: step G never reads gameweek G's (or a later) EP or score
    before = rl.player_residual_adjustments(con, SEASON, 3, history, **LOW_FLOOR)
    assert before == pytest.approx(rl.player_residual_adjustments(con, SEASON, 3, {2: history[2]}, **LOW_FLOOR))
    con.execute("UPDATE fact_player_season_stats SET event_points = event_points + 15 WHERE season = ? AND gw >= 3",
                [SEASON])
    assert rl.player_residual_adjustments(con, SEASON, 3, history, **LOW_FLOOR) == pytest.approx(before)
    # another season's gameweek of the same number is never read
    con.execute("UPDATE fact_player_season_stats SET event_points = event_points + 15 WHERE season = '2024-2025'")
    assert rl.player_residual_adjustments(con, SEASON, 3, history, **LOW_FLOOR) == pytest.approx(before)


# ------------------------------------------------------------------ the apply and the harnesses


def test_the_walk_forward_reads_its_own_runs_earlier_steps_and_the_pre_l3_projection(con):
    _crowd_league(con)
    v = rl.player_residual_version(con, rw.EXPERIMENT_EFFECTIVE_DATE, LOW_FLOOR)
    other = _walk(con, [1, 2, 3])  # another run in the same DB: never read
    run_id = _walk(con, [1, 2, 3, 4], version=v)
    assert bt.player_residual_walk_history(con, run_id, SEASON, 4) == {
        gw: _step_mv(con, run_id, gw) for gw in (1, 2, 3)}
    assert set(bt.player_residual_walk_history(con, other, SEASON, 4)) == {1, 2, 3}
    history = bt.player_residual_walk_history(con, run_id, SEASON, 4)
    # GW2 and GW3 were adjusted too, so the residual must take ep_player_adj back out
    assert any(adj != 0 for _, adj in _ep_rows(con, history[3]).values())
    expected = _expected(con, history, 4)
    rows = _ep_rows(con, _step_mv(con, run_id, 4))
    assert expected
    for uid, (total, adj) in rows.items():
        want = expected.get(uid, 0.0)
        assert adj == pytest.approx(max(want, -max(total - adj, 0.0)))  # one fixture each: floored at zero
    assert con.execute("SELECT player_residual_params_version FROM ep_model_versions WHERE model_version = ?",
                       [_step_mv(con, run_id, 4)]).fetchone()[0] == v
    # GW1 has no earlier step: nothing to adjust
    assert all(adj == 0 for _, adj in _ep_rows(con, _step_mv(con, run_id, 1)).values())


def test_without_the_flag_ep_is_identical(con):
    _crowd_league(con)
    v = rl.player_residual_version(con, rw.EXPERIMENT_EFFECTIVE_DATE, LOW_FLOOR)
    asof, ts, mm = _snapshots(con, SEASON, 4)
    with bt.asof_scope(con, SEASON, 4):
        control = ep.run(con, asof, SEASON, 4, ts, mm, 1, 1, 1, lookback_seasons=bt.lookback_seasons_for(SEASON))
        empty = ep.run(con, asof, SEASON, 4, ts, mm, 1, 1, 1, player_residual_params_version=v,
                       player_residual_history={}, lookback_seasons=bt.lookback_seasons_for(SEASON))
    assert _ep_rows(con, empty) == _ep_rows(con, control)
    assert all(adj == 0 for _, adj in _ep_rows(con, control).values())


def test_the_planner_horizon_carries_the_deadlines_adjustment(con):
    _crowd_league(con)
    v = rl.player_residual_version(con, rw.EXPERIMENT_EFFECTIVE_DATE, LOW_FLOOR)
    run_id = _walk(con, [1, 2])
    history = {gw: _step_mv(con, run_id, gw) for gw in (1, 2)}
    expected = rl.player_residual_adjustments(con, SEASON, 3, history, **LOW_FLOOR)
    asof, ts, mm = _snapshots(con, SEASON, 3, schedule_horizon_gameweeks=2)
    with bt.asof_scope(con, SEASON, 3, schedule_horizon_gameweeks=2):
        horizon = tp.compute_horizon_ep(con, asof, SEASON, 3, ts, mm, 2, 1, 1, 1, 1, 1,
                                        player_residual_params_version=v, player_residual_history=history)
    assert set(horizon) == {3, 4} and expected
    for ep_mv, _ in horizon.values():
        for uid, (total, adj) in _ep_rows(con, ep_mv).items():
            assert adj == pytest.approx(max(expected.get(uid, 0.0), -max(total - adj, 0.0)))


def test_the_season_sim_replays_the_weeks_before_its_start_and_grows_its_own_history(con):
    _crowd_league(con)
    v = rl.player_residual_version(con, rw.EXPERIMENT_EFFECTIVE_DATE, LOW_FLOOR)
    before = con.execute("SELECT coalesce(max(model_version), 0) FROM ep_model_versions").fetchone()[0]
    result = bt.run_season_simulation(con, SEASON, start_gameweek=2, end_gameweek=4, n_antithetic_pairs=200,
                                      player_residual_params_version=v, **_SEASON_SIM_VERSIONS)
    new = con.execute(
        "SELECT model_version, player_residual_params_version, calibration_asof_date FROM ep_model_versions "
        "WHERE model_version > ? ORDER BY model_version", [before]).fetchall()
    # the first new version is GW1's replay, without L3; every later projection carries L3
    gw1_deadline = bt.gameweek_deadline(con, SEASON, 1).date()
    assert new[0][1] is None and new[0][2] == gw1_deadline
    assert all(pv == v for _, pv, _ in new[1:])
    # GW4's plan read GW1-3's deadline EP: its own GW4 projection carries their adjustment
    gw4_plan = next(a["plan_run_id"] for a in result["actions"] if a["gameweek"] == 4)
    gw3_plan = next(a["plan_run_id"] for a in result["actions"] if a["gameweek"] == 3)
    history = {1: new[0][0], 2: result_bootstrap_ep(con, result),
               3: tp.plan_gameweek_versions(con, gw3_plan, 3)[0]}
    expected = rl.player_residual_adjustments(con, SEASON, 4, history, **LOW_FLOOR)
    assert expected
    for uid, (total, adj) in _ep_rows(con, tp.plan_gameweek_versions(con, gw4_plan, 4)[0]).items():
        assert adj == pytest.approx(max(expected.get(uid, 0.0), -max(total - adj, 0.0)))


def result_bootstrap_ep(con, result):
    return con.execute("SELECT ep_model_version FROM squad_optimizer_runs WHERE run_id = ?",
                       [result["bootstrap_run_id"]]).fetchone()[0]


# ------------------------------------------------------------------ flags (Req 6)


def test_player_residual_flag_mints_and_reuses_the_declared_version_in_both_scripts(con):
    assert "player_residual_params_version" not in rw._experiment_versions(con, rw._parse_args([]))
    out = rw._experiment_versions(con, rw._parse_args(["--player-residual"]))
    assert set(out) == {"player_residual_params_version"}
    assert rl.resolve_player_residual(con, out["player_residual_params_version"]) == {
        "window": 6, "pseudo_gws": 6.0, "min_minutes": 900.0}
    assert rw._experiment_versions(con, rw._parse_args(["--player-residual"])) == out
    versions, changed = arm.arm_versions(con, arm._parse_args(["--label", "x", "--player-residual"]))
    assert changed == out and versions["player_residual_params_version"] == out["player_residual_params_version"]
    control, _ = arm.arm_versions(con, arm._parse_args(["--label", "control"]))
    assert "player_residual_params_version" not in control
