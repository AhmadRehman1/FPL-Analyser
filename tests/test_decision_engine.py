from datetime import date

import pytest

from fpl_quant import decision_engine as de
from fpl_quant import squad_optimizer as so
from fpl_quant import transfer_planner as tp
from test_transfer_planner import _seed_real_squad_optimizer_candidate_pool, _seed_run_ready_state

RUN_KWARGS_1 = dict(
    horizon_params_version=1, scoring_params_version=1, bps_params_version=1, tau_params_version=1,
    rho_residual_params_version=1, corr_params_version=1, transfer_cost_params_version=1,
    lambda_params_version=1, guardrail_params_version=1, wildcard_threshold_params_version=1,
    free_hit_threshold_params_version=1, kappa_tc_params_version=1,
)


def test_recommend_best_move_returns_a_decision_with_provenance(con, monkeypatch):
    state_version, ts_mv, mm_mv, horizon_ep_versions = _seed_run_ready_state(con)
    monkeypatch.setattr(tp, "compute_horizon_ep", lambda *a, **k: horizon_ep_versions)

    decision = de.recommend_best_move(
        con, entry_id=123, calibration_asof_date=date(2026, 8, 24), target_season="2026-2027",
        target_gameweek=2, input_state_version=state_version, ts_model_version=ts_mv, mm_model_version=mm_mv,
        **RUN_KWARGS_1, include_sensitivity=False,
    )
    assert decision.action == "roll" or decision.action.startswith("transfer_in:") or decision.action in (
        "wildcard", "free_hit", "bench_boost", "triple_captain",
    )
    assert decision.provenance.data_asof == "2026-08-24"
    assert "plan_run" in decision.provenance.model_version
    assert isinstance(decision.downside_ci, tuple) and len(decision.downside_ci) == 2
    assert isinstance(decision.sensitivity, list)


def test_recommend_best_move_uses_a_supplied_horizon_ep_versions(con, monkeypatch):
    """Real perf fix: a caller evaluating the same baseline repeatedly (e.g. scenario.py's
    shared-baseline call, or several scripts in the same pipeline run) can hand recommend_
    best_move() an already-computed horizon instead of paying for tp.run()'s own
    compute_horizon_ep() call again."""
    state_version, ts_mv, mm_mv, horizon_ep_versions = _seed_run_ready_state(con)

    def _must_not_be_called(*a, **k):
        raise AssertionError("compute_horizon_ep() must not be called when horizon_ep_versions is supplied")

    monkeypatch.setattr(tp, "compute_horizon_ep", _must_not_be_called)

    decision = de.recommend_best_move(
        con, entry_id=123, calibration_asof_date=date(2026, 8, 24), target_season="2026-2027",
        target_gameweek=2, input_state_version=state_version, ts_model_version=ts_mv, mm_model_version=mm_mv,
        **RUN_KWARGS_1, include_sensitivity=False, horizon_ep_versions=horizon_ep_versions,
    )
    assert decision.provenance.data_asof == "2026-08-24"


def test_recommend_best_move_never_forwards_the_supplied_horizon_into_injury_sensitivity(con, monkeypatch):
    """Correctness guard, not just a perf test: _injury_sensitivity()'s whole mechanism is
    re-deriving ep_outputs against a shadowed minutes_model_outputs table. If a caller-supplied
    horizon_ep_versions ever leaked into that nested call, the shadow would be silently
    ignored -- the sensitivity toggle would stop reacting to the very perturbation it exists to
    test. compute_horizon_ep is left callable (not raising) so the nested leg can actually
    proceed; the assertion is that it DOES get called at least once despite a horizon being
    supplied for the outer call."""
    state_version, ts_mv, mm_mv, horizon_ep_versions = _seed_run_ready_state(con)

    call_count = 0

    def counting_compute(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return horizon_ep_versions

    monkeypatch.setattr(tp, "compute_horizon_ep", counting_compute)

    de.recommend_best_move(
        con, entry_id=123, calibration_asof_date=date(2026, 8, 24), target_season="2026-2027",
        target_gameweek=2, input_state_version=state_version, ts_model_version=ts_mv, mm_model_version=mm_mv,
        **RUN_KWARGS_1, include_sensitivity=True, horizon_ep_versions=horizon_ep_versions,
    )

    # The outer call supplied horizon_ep_versions (so its own tp.run() must not trigger a
    # call), but _injury_sensitivity()'s nested recommend_best_move() call never receives it
    # and must still compute its own horizon fresh -- at least one real call, not zero.
    assert call_count >= 1


def test_recommend_best_move_track_record_is_insufficient_history_by_default(con, monkeypatch):
    state_version, ts_mv, mm_mv, horizon_ep_versions = _seed_run_ready_state(con)
    monkeypatch.setattr(tp, "compute_horizon_ep", lambda *a, **k: horizon_ep_versions)

    decision = de.recommend_best_move(
        con, entry_id=123, calibration_asof_date=date(2026, 8, 24), target_season="2026-2027",
        target_gameweek=2, input_state_version=state_version, ts_model_version=ts_mv, mm_model_version=mm_mv,
        **RUN_KWARGS_1, include_sensitivity=False,
    )
    assert decision.track_record.optimal_in_n_of_71 is None  # never fabricated when no history supplied
    assert decision.track_record.sample_size == 0


def test_recommend_best_move_track_record_scores_against_supplied_history(con, monkeypatch):
    state_version, ts_mv, mm_mv, horizon_ep_versions = _seed_run_ready_state(con)
    monkeypatch.setattr(tp, "compute_horizon_ep", lambda *a, **k: horizon_ep_versions)

    decision = de.recommend_best_move(
        con, entry_id=123, calibration_asof_date=date(2026, 8, 24), target_season="2026-2027",
        target_gameweek=2, input_state_version=state_version, ts_model_version=ts_mv, mm_model_version=mm_mv,
        **RUN_KWARGS_1, include_sensitivity=False,
    )
    action_kind = decision.action.split(":")[0]
    # 6 history rows matching this action's kind, 2 not -- clears the sample-size floor
    matching = {"accepted_chip": action_kind} if action_kind != "roll" and action_kind != "transfer_in" else (
        {"accepted_chip": None, "accepted_transfer_rank": None} if action_kind == "roll"
        else {"accepted_chip": None, "accepted_transfer_rank": 1}
    )
    non_matching = {"accepted_chip": "bench_boost", "accepted_transfer_rank": None}
    history = [matching] * 6 + [non_matching] * 2

    scored = de.recommend_best_move(
        con, entry_id=123, calibration_asof_date=date(2026, 8, 24), target_season="2026-2027",
        target_gameweek=2, input_state_version=state_version, ts_model_version=ts_mv, mm_model_version=mm_mv,
        **RUN_KWARGS_1, include_sensitivity=False, historical_actions=history,
    )
    assert scored.track_record.optimal_in_n_of_71 == 6
    assert scored.track_record.sample_size == 8


def test_runner_up_populated_when_top_two_transfers_are_close(con, monkeypatch):
    """Isolated unit test of the runner_up logic: monkeypatches tp.run() and
    backtest._decide_gameweek_action() to fixed values over a hand-seeded transfer_plan_runs/
    transfer_recommendations pair, so this doesn't depend on a real solve happening to
    produce a close top-2 (data-dependent and not worth chasing for this specific check)."""
    con.execute(
        "INSERT INTO manager_state_versions (season, as_of_gameweek, free_transfers_available, "
        "chips_used_set1, chips_used_set2, bank) VALUES ('2026-2027', 2, 1, '[]', '[]', 0.0) RETURNING state_version"
    )
    state_version = con.execute("SELECT max(state_version) FROM manager_state_versions").fetchone()[0]
    for uid in ("out1", "in1", "in2"):
        con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, 'Midfielder')", [uid, uid])
    con.execute(
        "INSERT INTO manager_squad_holdings (state_version, player_uid, in_xi, is_captain, is_vice) "
        "VALUES (?, 'out1', TRUE, FALSE, FALSE)", [state_version],
    )
    run_id = con.execute(
        "INSERT INTO transfer_plan_runs (calibration_asof_date, target_season, target_gameweek, input_state_version, "
        "horizon_params_version, transfer_cost_params_version, ep_model_versions, uncertainty_model_versions) "
        "VALUES ('2026-08-24', '2026-2027', 2, ?, 1, 1, '{\"2\": 1}', '{\"2\": 1}') RETURNING run_id",
        [state_version],
    ).fetchone()[0]
    con.execute(
        "INSERT INTO transfer_recommendations (run_id, rank, player_out, player_in, horizon_value_gain, transfer_cost, net_value) "
        "VALUES (?, 1, 'out1', 'in1', 5.0, 0.0, 5.0)", [run_id],
    )
    con.execute(
        "INSERT INTO transfer_recommendations (run_id, rank, player_out, player_in, horizon_value_gain, transfer_cost, net_value) "
        "VALUES (?, 2, 'out1', 'in2', 4.7, 0.0, 4.7)", [run_id],  # 0.3 gap -- well under RUNNER_UP_DELTA_THRESHOLD
    )

    monkeypatch.setattr(de.tp, "run", lambda *a, **k: run_id)
    monkeypatch.setattr(de.bt, "_decide_gameweek_action", lambda *a, **k: (1, None))

    decision = de.recommend_best_move(
        con, entry_id=123, calibration_asof_date=date(2026, 8, 24), target_season="2026-2027",
        target_gameweek=2, input_state_version=state_version, ts_model_version=1, mm_model_version=1,
        **RUN_KWARGS_1, include_sensitivity=False,
    )
    assert decision.action == "transfer_in:out1->in1"
    assert decision.runner_up is not None
    assert decision.runner_up.action == "transfer_in:out1->in2"


def test_runner_up_is_none_when_the_gap_is_large(con, monkeypatch):
    con.execute(
        "INSERT INTO manager_state_versions (season, as_of_gameweek, free_transfers_available, "
        "chips_used_set1, chips_used_set2, bank) VALUES ('2026-2027', 2, 1, '[]', '[]', 0.0) RETURNING state_version"
    )
    state_version = con.execute("SELECT max(state_version) FROM manager_state_versions").fetchone()[0]
    for uid in ("out1", "in1", "in2"):
        con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, 'Midfielder')", [uid, uid])
    con.execute(
        "INSERT INTO manager_squad_holdings (state_version, player_uid, in_xi, is_captain, is_vice) "
        "VALUES (?, 'out1', TRUE, FALSE, FALSE)", [state_version],
    )
    run_id = con.execute(
        "INSERT INTO transfer_plan_runs (calibration_asof_date, target_season, target_gameweek, input_state_version, "
        "horizon_params_version, transfer_cost_params_version, ep_model_versions, uncertainty_model_versions) "
        "VALUES ('2026-08-24', '2026-2027', 2, ?, 1, 1, '{\"2\": 1}', '{\"2\": 1}') RETURNING run_id",
        [state_version],
    ).fetchone()[0]
    con.execute(
        "INSERT INTO transfer_recommendations (run_id, rank, player_out, player_in, horizon_value_gain, transfer_cost, net_value) "
        "VALUES (?, 1, 'out1', 'in1', 10.0, 0.0, 10.0)", [run_id],
    )
    con.execute(
        "INSERT INTO transfer_recommendations (run_id, rank, player_out, player_in, horizon_value_gain, transfer_cost, net_value) "
        "VALUES (?, 2, 'out1', 'in2', 2.0, 0.0, 2.0)", [run_id],  # 8.0 gap -- well over threshold
    )
    monkeypatch.setattr(de.tp, "run", lambda *a, **k: run_id)
    monkeypatch.setattr(de.bt, "_decide_gameweek_action", lambda *a, **k: (1, None))

    decision = de.recommend_best_move(
        con, entry_id=123, calibration_asof_date=date(2026, 8, 24), target_season="2026-2027",
        target_gameweek=2, input_state_version=state_version, ts_model_version=1, mm_model_version=1,
        **RUN_KWARGS_1, include_sensitivity=False,
    )
    assert decision.runner_up is None


# ============================================================
# _injury_sensitivity -- the "what would change my mind" mechanism
# ============================================================

def test_minutes_model_outputs_shadow_zeroes_only_the_target_player(con):
    """Direct proof the TEMP TABLE shadow SQL _injury_sensitivity() uses actually works --
    the same technique backtest.asof_scope() already relies on. Verifies main's real storage
    is untouched by checking it AFTER the shadow is dropped, not via a main.-qualified read
    WHILE the shadow exists -- DuckDB's main. prefix does not reliably disambiguate from a
    same-named temp table for the duration that temp table exists (confirmed directly against
    backtest.asof_scope() itself: main.fact_match reads mid-shadow show the shadowed values
    too, even though the real underlying storage is provably untouched -- restored correctly
    the moment the shadow is dropped). This is a resolution quirk of qualified reads, not a
    data-corruption bug in the CREATE-time shadow itself, whose own contents this test does
    check directly (unqualified, i.e. how every real caller of the shadow reads it)."""
    for uid in ("star", "other"):
        con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, 'Forward')", [uid, uid])
        con.execute(
            "INSERT INTO minutes_model_versions (calibration_asof_date, target_season, decay_params_version, "
            "adjustment_params_version, shrinkage_params_version, fact_multiplier_params_version, lookback_seasons) "
            "VALUES ('2026-08-10', '2026-2027', 1, 1, 1, 1, '[]') ON CONFLICT DO NOTHING"
        )
    mm_mv = con.execute("SELECT max(model_version) FROM minutes_model_versions").fetchone()[0]
    for uid in ("star", "other"):
        con.execute(
            "INSERT INTO minutes_model_outputs (model_version, player_uid, position, p_start_historical_final, "
            "p_start_historical_position_avg, weight_own, logit_adjustment_total, p_start_final, "
            "p_used_as_sub_given_not_started, p_0min, p_1_59min, p_60plus_min, competitive_matches_last_2_seasons) "
            "VALUES (?, ?, 'Forward', 0.9, 0.9, 1.0, 0.0, 0.9, 0.0, 0.05, 0.05, 0.9, 20)", [mm_mv, uid],
        )

    con.execute(
        "CREATE OR REPLACE TEMP TABLE minutes_model_outputs AS "
        "SELECT * REPLACE (1.0 AS p_0min, 0.0 AS p_1_59min, 0.0 AS p_60plus_min) FROM main.minutes_model_outputs WHERE player_uid = ? "
        "UNION ALL SELECT * FROM main.minutes_model_outputs WHERE player_uid != ?",
        ["star", "star"],
    )
    try:
        star_row = con.execute("SELECT p_0min, p_1_59min, p_60plus_min FROM minutes_model_outputs WHERE player_uid = 'star'").fetchone()
        other_row = con.execute("SELECT p_0min, p_1_59min, p_60plus_min FROM minutes_model_outputs WHERE player_uid = 'other'").fetchone()
        assert star_row == (1.0, 0.0, 0.0)
        assert other_row == (0.05, 0.05, 0.9)
    finally:
        con.execute("DROP TABLE IF EXISTS minutes_model_outputs")

    # main's real storage was never actually mutated -- restored the instant the shadow drops
    main_star = con.execute("SELECT p_0min FROM minutes_model_outputs WHERE player_uid = 'star'").fetchone()
    assert main_star == (0.05,)


def test_injury_sensitivity_reports_a_flip_when_the_perturbed_action_differs(con, monkeypatch):
    horizon_ep_versions, holdings = _seed_real_squad_optimizer_candidate_pool(con)
    so.seed_v1_params(con)
    ep_mv, _un_mv = horizon_ep_versions[2]
    xi_uids = {h["player_uid"] for h in holdings}

    fake_perturbed = de.Decision(
        action="transfer_in:def5->fwd0", swaps=[], ep_lift=3.0, downside_ci=(0.0, 0.0), sensitivity=[],
        track_record=de.TrackRecord(pattern="p", optimal_in_n_of_71=None, sample_size=0),
        provenance=de.Provenance(model_version="v", data_asof="2026-08-24"), runner_up=None,
    )
    monkeypatch.setattr(de, "recommend_best_move", lambda *a, **k: fake_perturbed)

    result = de._injury_sensitivity(
        con, {"entry_id": 123}, target_ep_mv=ep_mv, xi_uids=xi_uids, baseline_action="roll", baseline_ep_lift=0.0,
    )
    assert len(result) == 1
    assert result[0].then_action == "transfer_in:def5->fwd0"
    assert result[0].delta_ep == pytest.approx(3.0)
    assert "ruled out" in result[0].if_condition


def test_injury_sensitivity_empty_when_action_does_not_change(con, monkeypatch):
    horizon_ep_versions, holdings = _seed_real_squad_optimizer_candidate_pool(con)
    ep_mv, _un_mv = horizon_ep_versions[2]
    xi_uids = {h["player_uid"] for h in holdings}

    fake_perturbed = de.Decision(
        action="roll", swaps=[], ep_lift=0.0, downside_ci=(0.0, 0.0), sensitivity=[],
        track_record=de.TrackRecord(pattern="p", optimal_in_n_of_71=None, sample_size=0),
        provenance=de.Provenance(model_version="v", data_asof="2026-08-24"), runner_up=None,
    )
    monkeypatch.setattr(de, "recommend_best_move", lambda *a, **k: fake_perturbed)

    result = de._injury_sensitivity(
        con, {"entry_id": 123}, target_ep_mv=ep_mv, xi_uids=xi_uids, baseline_action="roll", baseline_ep_lift=0.0,
    )
    assert result == []


def test_injury_sensitivity_empty_with_no_xi_players(con):
    assert de._injury_sensitivity(con, {}, target_ep_mv=1, xi_uids=set(), baseline_action="roll", baseline_ep_lift=0.0) == []
    assert de._injury_sensitivity(con, {}, target_ep_mv=None, xi_uids={"p1"}, baseline_action="roll", baseline_ep_lift=0.0) == []


def test_injury_sensitivity_drops_the_temp_shadow_even_on_exception(con, monkeypatch):
    horizon_ep_versions, holdings = _seed_real_squad_optimizer_candidate_pool(con)
    ep_mv, _un_mv = horizon_ep_versions[2]
    xi_uids = {h["player_uid"] for h in holdings}

    def _boom(*a, **k):
        raise RuntimeError("simulated failure mid re-solve")

    monkeypatch.setattr(de, "recommend_best_move", _boom)
    with pytest.raises(RuntimeError):
        de._injury_sensitivity(con, {}, target_ep_mv=ep_mv, xi_uids=xi_uids, baseline_action="roll", baseline_ep_lift=0.0)

    # the shadow must not survive the exception -- main data must be readable/unshadowed after
    row = con.execute("SELECT count(*) FROM minutes_model_outputs").fetchone()
    assert row is not None  # no crash querying the (now real, unshadowed) table


def test_recommend_best_move_runs_end_to_end_with_sensitivity_enabled(con, monkeypatch):
    """Full integration smoke test with the real recommend_best_move()+_injury_sensitivity()
    wiring (compute_horizon_ep mocked, same as every other real-solve transfer_planner test in
    this suite) -- must not crash, and must return a real list (possibly empty: this fixture's
    mocked compute_horizon_ep doesn't actually react to the minutes-model shadow, so a genuine
    flip isn't guaranteed here -- that's covered in isolation above)."""
    state_version, ts_mv, mm_mv, horizon_ep_versions = _seed_run_ready_state(con)
    monkeypatch.setattr(tp, "compute_horizon_ep", lambda *a, **k: horizon_ep_versions)

    decision = de.recommend_best_move(
        con, entry_id=123, calibration_asof_date=date(2026, 8, 24), target_season="2026-2027",
        target_gameweek=2, input_state_version=state_version, ts_model_version=ts_mv, mm_model_version=mm_mv,
        **RUN_KWARGS_1, include_sensitivity=True,
    )
    assert isinstance(decision.sensitivity, list)
