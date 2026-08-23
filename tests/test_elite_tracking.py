import json
from datetime import date

import pytest

from fpl_quant import decision_engine as de
from fpl_quant import elite_tracking as et
from fpl_quant import expected_points as ep_mod
from fpl_quant.errors import MissingModelVersionError
from test_transfer_planner import _seed_real_squad_optimizer_candidate_pool

RUN_KWARGS = dict(
    ts_model_version=1, mm_model_version=1,
    horizon_params_version=1, scoring_params_version=1, bps_params_version=1, tau_params_version=1,
    rho_residual_params_version=1, corr_params_version=1, transfer_cost_params_version=1,
    lambda_params_version=1, guardrail_params_version=1, wildcard_threshold_params_version=1,
    free_hit_threshold_params_version=1, kappa_tc_params_version=1,
)

# Matches _seed_real_squad_optimizer_candidate_pool's own player order: 2 GK, 6 DEF, 6 MID, 4 FWD.
ELEMENT_NAMES = {
    1: "gk0", 2: "gk1", 3: "def0", 4: "def1", 5: "def2", 6: "def3", 7: "def4", 8: "def5",
    9: "mid0", 10: "mid1", 11: "mid2", 12: "mid3", 13: "mid4", 14: "mid5",
    15: "fwd0", 16: "fwd1", 17: "fwd2", 18: "fwd3",
}


def _picks(xi_elements, bench_elements, captain_element):
    picks = [
        {"element": el, "position": i, "is_captain": el == captain_element, "is_vice_captain": False}
        for i, el in enumerate(xi_elements, start=1)
    ]
    picks += [
        {"element": el, "position": i, "is_captain": False, "is_vice_captain": False}
        for i, el in enumerate(bench_elements, start=len(xi_elements) + 1)
    ]
    return picks


def _fake_decision(action, in_uid=None):
    swaps = [de.Swap(out_player_uid="unused", in_player_uid=in_uid, delta_ep=1.0, reason="test")] if in_uid else []
    return de.Decision(
        action=action, swaps=swaps, ep_lift=1.0, downside_ci=(0.0, 0.0), sensitivity=[],
        track_record=de.TrackRecord(pattern="p", optimal_in_n_of_71=None, sample_size=0),
        provenance=de.Provenance(model_version="fake_v1", data_asof="2026-08-24"), runner_up=None,
    )


# ============================================================
# load_elite_managers -- pure file IO, no hardcoded entry_ids anywhere in code
# ============================================================

def test_load_elite_managers_missing_file_returns_empty(tmp_path):
    assert et.load_elite_managers(tmp_path / "does_not_exist.json") == []


def test_load_elite_managers_reads_a_real_file(tmp_path):
    path = tmp_path / "elite_managers.json"
    path.write_text(json.dumps({"managers": [{"entry_id": 42, "name": "Someone"}]}))
    assert et.load_elite_managers(path) == [{"entry_id": 42, "name": "Someone"}]


def test_elite_manager_list_is_genuinely_configurable(tmp_path):
    path = tmp_path / "elite_managers.json"
    path.write_text(json.dumps({"managers": [{"entry_id": 111}, {"entry_id": 222, "name": "Two"}]}))
    managers = et.load_elite_managers(path)
    assert [m["entry_id"] for m in managers] == [111, 222]
    assert managers[1]["name"] == "Two"


# ============================================================
# compute_actual_move -- pure set-difference logic over raw FPL element ids
# ============================================================

def test_compute_actual_move_is_roll_when_squad_is_unchanged():
    picks = _picks([1, 2, 3], [4], captain_element=1)
    move = et.compute_actual_move(picks, picks)
    assert move.players_out == []
    assert move.players_in == []


def test_compute_actual_move_single_transfer():
    previous = _picks([1, 2, 3], [4], captain_element=1)
    current = _picks([1, 2, 5], [4], captain_element=1)  # 3 -> 5
    move = et.compute_actual_move(previous, current)
    assert move.players_out == [3]
    assert move.players_in == [5]


def test_compute_actual_move_multi_transfer():
    previous = _picks([1, 2, 3], [4], captain_element=1)
    current = _picks([1, 6, 7], [4], captain_element=1)  # 2->6, 3->7
    move = et.compute_actual_move(previous, current)
    assert move.players_out == [2, 3]
    assert move.players_in == [6, 7]


# ============================================================
# build_elite_divergence -- real DB, real bootstrap_from_real_squad element-id -> player_name
# -> player_uid resolution (this feature's own new logic); compute_horizon_ep and
# recommend_best_move are monkeypatched, matching this project's established convention for
# these tests (see test_decision_engine.py's own tp.compute_horizon_ep mocking) -- a hand-built
# fixture can cheaply provide a fixed {gw: (ep_mv, un_mv)} and a fixed "what the model would
# recommend", and the real solve quality of transfer_planner/decision_engine is already covered
# by their own test suites, not re-proven here.
# ============================================================

def test_build_elite_divergence_flags_diverged_with_a_reason(con, monkeypatch):
    horizon_ep_versions, _holdings = _seed_real_squad_optimizer_candidate_pool(con, target_gameweek=2)
    monkeypatch.setattr(et.tp, "compute_horizon_ep", lambda *a, **k: horizon_ep_versions)
    monkeypatch.setattr(et.de, "recommend_best_move", lambda *a, **k: _fake_decision("transfer_in:mid2->fwd2", in_uid="fwd2"))

    previous_picks = _picks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [12, 13, 14, 15], captain_element=1)
    current_picks = _picks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16], [12, 13, 14, 15], captain_element=1)  # 11 (mid2) -> 16 (fwd1)

    def fetch_picks(entry_id, event):
        return previous_picks if event == 1 else current_picks

    out = et.build_elite_divergence(
        con, [{"entry_id": 999, "name": "Elite One"}], date(2026, 8, 24), "2026-2027", 2,
        fetch_picks, ELEMENT_NAMES, dict(RUN_KWARGS),
    )
    assert len(out) == 1
    row = out[0]
    assert row["entry_id"] == 999
    assert row["actual_move"] == "transfer_in:mid2->fwd1"
    assert row["model_move"] == "transfer_in:mid2->fwd2"
    assert row["diverged"] is True
    assert row["divergence_reason"] is not None
    assert row["provenance"] == {"model_version": "fake_v1", "data_asof": "2026-08-24"}


def test_build_elite_divergence_no_divergence_when_actual_matches_model(con, monkeypatch):
    horizon_ep_versions, _holdings = _seed_real_squad_optimizer_candidate_pool(con, target_gameweek=2)
    monkeypatch.setattr(et.tp, "compute_horizon_ep", lambda *a, **k: horizon_ep_versions)
    monkeypatch.setattr(et.de, "recommend_best_move", lambda *a, **k: _fake_decision("transfer_in:mid2->fwd1", in_uid="fwd1"))

    previous_picks = _picks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [12, 13, 14, 15], captain_element=1)
    current_picks = _picks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16], [12, 13, 14, 15], captain_element=1)  # mid2 -> fwd1, same as model

    def fetch_picks(entry_id, event):
        return previous_picks if event == 1 else current_picks

    out = et.build_elite_divergence(
        con, [{"entry_id": 999, "name": "Elite One"}], date(2026, 8, 24), "2026-2027", 2,
        fetch_picks, ELEMENT_NAMES, dict(RUN_KWARGS),
    )
    assert out[0]["diverged"] is False
    assert out[0]["divergence_reason"] is None


def test_build_elite_divergence_is_deterministic(con, monkeypatch):
    horizon_ep_versions, _holdings = _seed_real_squad_optimizer_candidate_pool(con, target_gameweek=2)
    monkeypatch.setattr(et.tp, "compute_horizon_ep", lambda *a, **k: horizon_ep_versions)
    monkeypatch.setattr(et.de, "recommend_best_move", lambda *a, **k: _fake_decision("roll"))

    previous_picks = _picks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [12, 13, 14, 15], captain_element=1)

    def fetch_picks(entry_id, event):
        return previous_picks

    managers = [{"entry_id": 999, "name": "Elite One"}]
    out1 = et.build_elite_divergence(con, managers, date(2026, 8, 24), "2026-2027", 2, fetch_picks, ELEMENT_NAMES, dict(RUN_KWARGS))
    out2 = et.build_elite_divergence(con, managers, date(2026, 8, 24), "2026-2027", 2, fetch_picks, ELEMENT_NAMES, dict(RUN_KWARGS))
    assert out1 == out2


def test_build_elite_divergence_skips_a_failing_manager_without_crashing(con, monkeypatch, capsys):
    horizon_ep_versions, _holdings = _seed_real_squad_optimizer_candidate_pool(con, target_gameweek=2)
    monkeypatch.setattr(et.tp, "compute_horizon_ep", lambda *a, **k: horizon_ep_versions)
    monkeypatch.setattr(et.de, "recommend_best_move", lambda *a, **k: _fake_decision("roll"))

    good_picks = _picks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [12, 13, 14, 15], captain_element=1)

    def fetch_picks(entry_id, event):
        if entry_id == 111:  # the "bad" manager: fetch itself blows up
            raise RuntimeError("simulated fetch failure")
        return good_picks

    managers = [{"entry_id": 111, "name": "Broken"}, {"entry_id": 999, "name": "Elite One"}]
    out = et.build_elite_divergence(con, managers, date(2026, 8, 24), "2026-2027", 2, fetch_picks, ELEMENT_NAMES, dict(RUN_KWARGS))
    assert len(out) == 1
    assert out[0]["entry_id"] == 999
    captured = capsys.readouterr()
    assert "::warning::elite_tracking" in captured.out
    assert "111" in captured.out


def test_build_elite_divergence_skips_when_picks_are_missing(con, monkeypatch, capsys):
    horizon_ep_versions, _holdings = _seed_real_squad_optimizer_candidate_pool(con, target_gameweek=2)
    monkeypatch.setattr(et.tp, "compute_horizon_ep", lambda *a, **k: horizon_ep_versions)

    def fetch_picks(entry_id, event):
        return None  # e.g. no picks recorded yet for this event

    out = et.build_elite_divergence(
        con, [{"entry_id": 999, "name": "Elite One"}], date(2026, 8, 24), "2026-2027", 2,
        fetch_picks, ELEMENT_NAMES, dict(RUN_KWARGS),
    )
    assert out == []
    assert "::warning::elite_tracking" in capsys.readouterr().out


def test_build_elite_divergence_raises_when_current_event_has_no_fixtures(con):
    ep_mod.seed_v1_params(con)  # far enough to reach the real "no fixtures for GW99" check
    with pytest.raises(MissingModelVersionError):
        et.build_elite_divergence(
            con, [{"entry_id": 999, "name": "Elite One"}], date(2026, 8, 24), "2026-2027", 99,
            lambda entry_id, event: None, ELEMENT_NAMES, dict(RUN_KWARGS),
        )
