from datetime import date

import pytest

from fpl_quant import decision_engine as de
from fpl_quant import scenario as scen
from fpl_quant import transfer_planner as tp
from fpl_quant.errors import InvalidScenarioError
from test_transfer_planner import _seed_run_ready_state

RUN_KWARGS_1 = dict(
    horizon_params_version=1, scoring_params_version=1, bps_params_version=1, tau_params_version=1,
    rho_residual_params_version=1, corr_params_version=1, transfer_cost_params_version=1,
    lambda_params_version=1, guardrail_params_version=1, wildcard_threshold_params_version=1,
    free_hit_threshold_params_version=1, kappa_tc_params_version=1,
)


def _base_state(state_version, ts_mv, mm_mv):
    return dict(
        entry_id=123, calibration_asof_date=date(2026, 8, 24), target_season="2026-2027",
        target_gameweek=2, input_state_version=state_version, ts_model_version=ts_mv, mm_model_version=mm_mv,
        **RUN_KWARGS_1,
    )


# ============================================================
# validate_scenario
# ============================================================

def test_validate_scenario_rejects_unknown_kind(con):
    with pytest.raises(InvalidScenarioError, match="unknown scenario kind"):
        scen.validate_scenario(scen.Scenario(kind="teleport"), con)


def test_validate_scenario_rejects_unknown_player_uid(con):
    with pytest.raises(InvalidScenarioError, match="unknown player_uid"):
        scen.validate_scenario(scen.Scenario(kind="injury", player_uid="nobody"), con)


def test_validate_scenario_injury_requires_player_uid(con):
    with pytest.raises(InvalidScenarioError, match="needs player_uid"):
        scen.validate_scenario(scen.Scenario(kind="injury"), con)


def test_validate_scenario_lineup_change_requires_starting(con):
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p1', 'P1', 'Forward')")
    with pytest.raises(InvalidScenarioError, match="needs starting"):
        scen.validate_scenario(scen.Scenario(kind="lineup_change", player_uid="p1"), con)


def test_validate_scenario_price_change_requires_delta(con):
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p1', 'P1', 'Forward')")
    with pytest.raises(InvalidScenarioError, match="needs delta"):
        scen.validate_scenario(scen.Scenario(kind="price_change", player_uid="p1"), con)


def test_validate_scenario_price_change_rejects_impossible_negative_price(con):
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p1', 'P1', 'Forward')")
    con.execute(
        "INSERT INTO fact_player_season_stats (player_uid, season, gw, now_cost, _ingested_at) "
        "VALUES ('p1', '2026-2027', 1, 5.0, current_timestamp)"
    )
    with pytest.raises(InvalidScenarioError, match="impossible"):
        scen.validate_scenario(scen.Scenario(kind="price_change", player_uid="p1", delta=-10.0), con)


def test_validate_scenario_price_change_accepts_a_legal_delta(con):
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p1', 'P1', 'Forward')")
    con.execute(
        "INSERT INTO fact_player_season_stats (player_uid, season, gw, now_cost, _ingested_at) "
        "VALUES ('p1', '2026-2027', 1, 5.0, current_timestamp)"
    )
    scen.validate_scenario(scen.Scenario(kind="price_change", player_uid="p1", delta=0.1), con)  # must not raise


def test_validate_scenario_dgw_swing_requires_known_team_uids(con):
    with pytest.raises(InvalidScenarioError, match="unknown team_uid"):
        scen.validate_scenario(scen.Scenario(kind="dgw_swing", team_uids=["nope"], add_gws=[10]), con)


def test_validate_scenario_dgw_swing_requires_add_gws(con):
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('team_a', 'A')")
    with pytest.raises(InvalidScenarioError, match="needs add_gws"):
        scen.validate_scenario(scen.Scenario(kind="dgw_swing", team_uids=["team_a"]), con)


# ============================================================
# apply_scenario
# ============================================================

def test_apply_scenario_baseline_matches_a_direct_no_scenario_call(con, monkeypatch):
    """Idempotence: apply_scenario()'s own baseline leg must be the SAME recommendation a
    direct recommend_best_move() call (no scenario at all) produces."""
    state_version, ts_mv, mm_mv, horizon_ep_versions = _seed_run_ready_state(con)
    monkeypatch.setattr(tp, "compute_horizon_ep", lambda *a, **k: horizon_ep_versions)
    base_state = _base_state(state_version, ts_mv, mm_mv)

    direct = de.recommend_best_move(con, **base_state, include_sensitivity=False)

    scenario = scen.Scenario(kind="price_change", player_uid="def0", delta=0.0)  # delta=0 -- a pure no-op perturbation
    result = scen.apply_scenario(con, base_state, scenario)

    assert result.baseline_decision.action == direct.action
    assert result.baseline_decision.ep_lift == pytest.approx(direct.ep_lift)


def test_apply_scenario_never_leaks_into_param_versions(con, monkeypatch):
    state_version, ts_mv, mm_mv, horizon_ep_versions = _seed_run_ready_state(con)
    monkeypatch.setattr(tp, "compute_horizon_ep", lambda *a, **k: horizon_ep_versions)
    base_state = _base_state(state_version, ts_mv, mm_mv)

    before = con.execute("SELECT param_family, param_version, param_key, dimensions, value_numeric FROM param_versions ORDER BY 1,2,3,4").fetchall()
    scen.apply_scenario(con, base_state, scen.Scenario(kind="injury", player_uid="def0"))
    after = con.execute("SELECT param_family, param_version, param_key, dimensions, value_numeric FROM param_versions ORDER BY 1,2,3,4").fetchall()
    assert before == after


def test_apply_scenario_injury_drops_the_shadow_after_returning(con, monkeypatch):
    state_version, ts_mv, mm_mv, horizon_ep_versions = _seed_run_ready_state(con)
    monkeypatch.setattr(tp, "compute_horizon_ep", lambda *a, **k: horizon_ep_versions)
    base_state = _base_state(state_version, ts_mv, mm_mv)

    scen.apply_scenario(con, base_state, scen.Scenario(kind="injury", player_uid="def0"))
    # a plain, unqualified query must succeed against the real (unshadowed) table afterward
    row = con.execute("SELECT count(*) FROM minutes_model_outputs").fetchone()
    assert row is not None


def test_apply_scenario_price_change_drops_the_shadow_after_returning(con, monkeypatch):
    state_version, ts_mv, mm_mv, horizon_ep_versions = _seed_run_ready_state(con)
    monkeypatch.setattr(tp, "compute_horizon_ep", lambda *a, **k: horizon_ep_versions)
    base_state = _base_state(state_version, ts_mv, mm_mv)

    scen.apply_scenario(con, base_state, scen.Scenario(kind="price_change", player_uid="def0", delta=0.5))
    row = con.execute("SELECT count(*) FROM fact_player_season_stats").fetchone()
    assert row is not None


def test_apply_scenario_returns_flipped_and_delta_ep(con, monkeypatch):
    state_version, ts_mv, mm_mv, horizon_ep_versions = _seed_run_ready_state(con)
    monkeypatch.setattr(tp, "compute_horizon_ep", lambda *a, **k: horizon_ep_versions)
    base_state = _base_state(state_version, ts_mv, mm_mv)

    result = scen.apply_scenario(con, base_state, scen.Scenario(kind="injury", player_uid="def0"))
    assert isinstance(result.flipped, bool)
    assert result.flipped == (result.perturbed_decision.action != result.baseline_decision.action)
    assert result.delta_ep == pytest.approx(result.perturbed_decision.ep_lift - result.baseline_decision.ep_lift)


def test_apply_scenario_raises_on_unknown_player(con, monkeypatch):
    state_version, ts_mv, mm_mv, horizon_ep_versions = _seed_run_ready_state(con)
    monkeypatch.setattr(tp, "compute_horizon_ep", lambda *a, **k: horizon_ep_versions)
    base_state = _base_state(state_version, ts_mv, mm_mv)

    with pytest.raises(InvalidScenarioError):
        scen.apply_scenario(con, base_state, scen.Scenario(kind="injury", player_uid="totally-not-a-player"))


def test_apply_scenario_dgw_swing_raises_not_implemented(con, monkeypatch):
    state_version, ts_mv, mm_mv, horizon_ep_versions = _seed_run_ready_state(con)
    monkeypatch.setattr(tp, "compute_horizon_ep", lambda *a, **k: horizon_ep_versions)
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('clubA', 'A') ON CONFLICT DO NOTHING")
    base_state = _base_state(state_version, ts_mv, mm_mv)

    with pytest.raises(NotImplementedError):
        scen.apply_scenario(con, base_state, scen.Scenario(kind="dgw_swing", team_uids=["clubA"], add_gws=[10]))
