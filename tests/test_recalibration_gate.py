"""recalibration_gate: every rejection case, plus review_recalibration.py --confirm going through it."""

import importlib.util
import json
from pathlib import Path

import pytest

from fpl_quant import backtest as bt
from fpl_quant import params
from fpl_quant import recalibration_gate as gate

_SPEC = importlib.util.spec_from_file_location(
    "review_recalibration", Path(__file__).resolve().parents[1] / "scripts" / "review_recalibration.py"
)
review = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(review)


def _run(con):
    return con.execute("INSERT INTO backtest_runs (warm_up_gameweeks) VALUES (0) RETURNING backtest_run_id").fetchone()[0]


def _propose(con, run_id, *, family="risk_aversion_params", key="lambda_value", old=0.15, new=0.1,
             metric="realized_sharpe", before=1.0, after=1.2, hold_before=1.0, hold_after=1.2, grid=None,
             new_params_version=None):
    params.write_param(con, family, 1, "2026-08-10", key, value_numeric=old)
    return bt.propose_recalibration(
        con, run_id, family, key, new, metric, before, after, old_params_version=1,
        holdout_metric_before=hold_before, holdout_metric_after=hold_after, grid_values=grid,
        new_params_version=new_params_version,
    )


# ---- purged K-fold ----

def test_purged_kfold_tests_every_step_once_and_embargoes_neighbours():
    steps = list(range(20))
    splits = gate.purged_kfold_splits(steps, k=5, embargo=2)
    assert len(splits) == 5
    assert sorted(s for _, test in splits for s in test) == steps
    train, test = splits[2]  # test = 8..11
    assert test == [8, 9, 10, 11]
    assert not set(train) & set(range(6, 14))
    assert set(train) == set(range(0, 6)) | set(range(14, 20))


def test_cv_holdout_scores_only_uses_training_steps_to_choose():
    steps = list(range(10))
    # candidate "b" is better everywhere -> every fold picks it
    scores = {"a": {s: 1.0 for s in steps}, "b": {s: 0.5 for s in steps}}
    out = gate.cv_holdout_scores(scores, steps, "a", lower_is_better=True, k=5, embargo=1)
    assert out["holdout_metric_before"] == pytest.approx(1.0)
    assert out["holdout_metric_after"] == pytest.approx(0.5)
    assert set(out["fold_choices"]) == {"b"}


def test_cv_holdout_scores_exposes_an_in_sample_fluke():
    # "b" wins only on the first half; a fold tuned on the other half keeps "a"
    steps = list(range(10))
    scores = {"a": {s: 1.0 for s in steps}, "b": {s: (0.0 if s < 5 else 1.5) for s in steps}}
    out = gate.cv_holdout_scores(scores, steps, "a", lower_is_better=True, k=2, embargo=0)
    # fold 1 tests 0..4 after tuning on 5..9 (picks a), fold 2 tests 5..9 after tuning on 0..4 (picks b)
    assert out["holdout_metric_after"] > out["holdout_metric_before"]


# ---- the gate ----

def test_a_clean_proposal_passes(con, tmp_path):
    pid = _propose(con, _run(con))
    assert gate.gate_reasons(con, pid, tmp_path) == []


def test_no_held_out_score_is_refused(con, tmp_path):
    pid = _propose(con, _run(con), hold_before=None, hold_after=None)
    reasons = gate.gate_reasons(con, pid, tmp_path)
    assert any("no held-out score" in r for r in reasons)


def test_in_sample_gain_that_vanishes_out_of_sample_is_refused(con, tmp_path):
    pid = _propose(con, _run(con), before=1.0, after=1.5, hold_before=1.0, hold_after=0.9)
    assert any("not an improvement" in r for r in gate.gate_reasons(con, pid, tmp_path))


def test_below_the_floor_is_refused(con, tmp_path):
    pid = _propose(con, _run(con), hold_before=1.0, hold_after=1.005)
    assert any("noise floor" in r for r in gate.gate_reasons(con, pid, tmp_path))


def test_floor_is_a_versioned_parameter(con, tmp_path):
    gate.seed_gate_params(con)
    assert gate.resolve_min_relative_improvement(con, 1) == pytest.approx(0.01)
    params.write_param(con, gate.GATE_FAMILY, 2, "2026-09-29", "min_relative_improvement", value_numeric=0.5)
    pid = _propose(con, _run(con), hold_before=1.0, hold_after=1.2)
    assert gate.gate_reasons(con, pid, tmp_path, gate_params_version=1) == []
    assert any("noise floor" in r for r in gate.gate_reasons(con, pid, tmp_path, gate_params_version=2))


def test_unchanged_value_is_refused(con, tmp_path):
    pid = _propose(con, _run(con), old=0.15, new=0.15)
    assert any("value unchanged" in r for r in gate.gate_reasons(con, pid, tmp_path))


def test_rho_hat_unchanged_is_refused_but_needs_no_held_out(con, tmp_path):
    run_id = _run(con)
    same = bt.propose_recalibration(con, run_id, "correlation_params", "rho_residual", 0.0, "rho_hat", 0.0, 0.0)
    moved = bt.propose_recalibration(con, run_id, "correlation_params", "rho_residual", 0.05, "rho_hat", 0.15, 0.05)
    assert any("unchanged" in r for r in gate.gate_reasons(con, same, tmp_path))
    assert gate.gate_reasons(con, moved, tmp_path) == []


def test_collision_with_a_confirmed_proposal_is_refused(con, tmp_path):
    run_id = _run(con)
    first = _propose(con, run_id, new=0.1)
    con.execute("UPDATE recalibration_proposals SET status = 'confirmed' WHERE proposal_id = ?", [first])
    version = con.execute("SELECT new_params_version FROM recalibration_proposals WHERE proposal_id = ?", [first]).fetchone()[0]
    # a second lineage claims the same version for a different value
    second = bt.propose_recalibration(
        con, run_id, "risk_aversion_params", "lambda_value", 0.3, "realized_sharpe", 1.0, 1.5,
        old_params_version=1, holdout_metric_before=1.0, holdout_metric_after=1.5, new_params_version=version,
    )
    assert any("collision" in r for r in gate.gate_reasons(con, second, tmp_path))


def test_collision_with_param_versions_is_refused(con, tmp_path):
    run_id = _run(con)
    params.write_param(con, "risk_aversion_params", 1, "2026-08-10", "lambda_value", value_numeric=0.15)
    params.write_param(con, "risk_aversion_params", 2, "2026-09-01", "lambda_value", value_numeric=0.05)
    pid = bt.propose_recalibration(
        con, run_id, "risk_aversion_params", "lambda_value", 0.1, "realized_sharpe", 1.0, 1.5,
        old_params_version=1, holdout_metric_before=1.0, holdout_metric_after=1.5, new_params_version=2,
    )
    assert any("already holds 0.05" in r for r in gate.gate_reasons(con, pid, tmp_path))


def test_collision_with_a_committed_seed_file_is_refused(con, tmp_path):
    pid = _propose(con, _run(con), new=0.1)
    version = con.execute("SELECT new_params_version FROM recalibration_proposals WHERE proposal_id = ?", [pid]).fetchone()[0]
    (tmp_path / "seeds_7.json").write_text(json.dumps({"proposals": [{
        "proposal_id": 42, "param_family": "risk_aversion_params", "param_key": "lambda_value", "dimensions": None,
        "new_params_version": version, "new_value": 0.05, "status": "confirmed",
    }]}))
    assert any("seeds_7.json" in r for r in gate.gate_reasons(con, pid, tmp_path))


def test_grid_edge_winner_is_refused(con, tmp_path):
    pid = _propose(con, _run(con), new=0.3, grid=(0.05, 0.1, 0.15, 0.3))
    assert any("edge of the searched grid" in r for r in gate.gate_reasons(con, pid, tmp_path))
    inner = _propose(con, _run(con), new=0.1, grid=(0.05, 0.1, 0.15, 0.3))
    assert gate.gate_reasons(con, inner, tmp_path) == []


def test_the_real_k_minutes_450_to_900_promotion_would_now_be_refused(con, tmp_path):
    # seeds_1.json, 2026-09-09: 1.1456995 -> 1.1453660 (0.029%), 900 = top of the grid
    pid = _propose(
        con, _run(con), family="rate_shrinkage_params", key="k_minutes", old=450.0, new=900.0,
        metric="ep_total_calibration_mae", before=1.1456995269035073, after=1.1453659985877347,
        hold_before=1.1456995269035073, hold_after=1.1453659985877347,
        grid=(150.0, 250.0, 350.0, 450.0, 600.0, 900.0),
    )
    reasons = gate.gate_reasons(con, pid, tmp_path)
    assert any("noise floor" in r for r in reasons)
    assert any("edge of the searched grid" in r for r in reasons)
    assert bt.evaluate_and_promote_proposal(con, pid, tmp_path)["action"] == "held"


# ---- recalibrate() records held-out + grid for k_minutes ----

def test_recalibrate_records_held_out_score_and_grid_for_rate_shrinkage(con, monkeypatch, tmp_path):
    params.write_param(con, "rate_shrinkage_params", 1, "2026-08-10", "k_minutes", value_numeric=450.0)
    run_id = _run(con)
    steps = [("2025-2026", gw) for gw in range(1, 11)]
    for season, gw in steps:
        con.execute(
            "INSERT INTO backtest_gameweek_steps (backtest_run_id, season, gameweek, tier, data_asof) "
            "VALUES (?, ?, ?, 'mature', '2025-08-01')", [run_id, season, gw],
        )

    def fake_refit(con, eval_steps, ep_model_version_by_step, k_minutes_grid=None, **kwargs):
        per_step = {k: {s: (1.0 if k == 450.0 else 0.8) for s in eval_steps} for k in k_minutes_grid}
        grid = {k: {"ep_total_calibration_mae": (1.0 if k == 450.0 else 0.8)} for k in k_minutes_grid}
        return {"best_k_minutes": 250.0, "grid": grid, "per_step_scores": per_step}

    monkeypatch.setattr(bt, "refit_rate_shrinkage", fake_refit)
    monkeypatch.setattr(bt, "_model_version_map", lambda con, run_id, col: {s: 1 for s in steps})
    ids = bt.recalibrate(
        con, run_id,
        current_xi_version=1, current_rho_version=1, current_rho_residual_version=1,
        current_minutes_versions={}, current_lambda_version=1, guardrail_cap=3.0,
        minutes_param_grids=[], refit_xi_rho_flag=False, refit_rho_residual_flag=False,
        refit_minutes_flag=False, refit_lambda_flag=False,
        current_rate_shrinkage_version=1, refit_rate_shrinkage_flag=True,
        rate_shrinkage_k_grid=(150.0, 250.0, 450.0),
    )
    row = con.execute(
        "SELECT holdout_metric_before, holdout_metric_after, grid_min, grid_max FROM recalibration_proposals "
        "WHERE proposal_id = ?", [ids[0]],
    ).fetchone()
    assert row[0] == pytest.approx(1.0) and row[1] == pytest.approx(0.8)
    assert row[2:] == (150.0, 450.0)


# ---- review_recalibration.py --confirm has no bypass ----

def test_manual_confirm_goes_through_the_gate(con, tmp_path, monkeypatch):
    monkeypatch.setattr(review, "SEED_DIR", tmp_path)
    run_id = _run(con)
    bad = _propose(con, run_id, old=0.15, new=0.15)
    with pytest.raises(review.GateRefused):
        review.set_status(con, bad, "confirmed", "someone")
    assert con.execute("SELECT status FROM recalibration_proposals WHERE proposal_id = ?", [bad]).fetchone() == ("pending",)

    good = _propose(con, run_id, new=0.1)
    review.set_status(con, good, "confirmed", "someone")
    assert con.execute("SELECT status FROM recalibration_proposals WHERE proposal_id = ?", [good]).fetchone() == ("confirmed",)


def test_manual_reject_is_never_blocked(con, tmp_path, monkeypatch):
    monkeypatch.setattr(review, "SEED_DIR", tmp_path)
    pid = _propose(con, _run(con), old=0.15, new=0.15)
    review.set_status(con, pid, "rejected", "someone")
    assert con.execute("SELECT status FROM recalibration_proposals WHERE proposal_id = ?", [pid]).fetchone() == ("rejected",)
