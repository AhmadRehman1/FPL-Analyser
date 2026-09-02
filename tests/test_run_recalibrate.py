"""scripts/run_recalibrate.py -- the per-stage recalibration runner that replaced
weekly_backtest.yml's monolithic (always-cancelled) job. The refit techniques themselves are
tested in tests/test_backtest.py; this covers the stage->flag wiring and the proposals dump."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_recalibrate as rr  # noqa: E402
from fpl_quant import backtest as bt  # noqa: E402


def test_every_stage_maps_to_a_real_recalibrate_flag():
    import inspect
    params = inspect.signature(bt.recalibrate).parameters
    assert set(rr.STAGE_FLAGS) == {"xi_rho", "rho_residual", "kappa_tc", "minutes", "lambda"}
    for flag in rr.STAGE_FLAGS.values():
        assert flag in params, flag
        assert params[flag].annotation is bool


def test_dump_proposals_writes_a_review_shape(con, tmp_path, monkeypatch):
    monkeypatch.setattr(rr, "PROPOSALS_JSON", tmp_path / "proposals.json")
    run_id = con.execute("INSERT INTO backtest_runs (warm_up_gameweeks) VALUES (0) RETURNING backtest_run_id").fetchone()[0]
    con.execute(
        "INSERT INTO recalibration_proposals (backtest_run_id, param_family, param_key, dimensions, "
        "old_params_version, new_params_version, old_value, new_value, metric_name, metric_before, metric_after) "
        "VALUES (?, 'model_decay_params', 'xi', NULL, 1, 2, 0.0018, 0.003, 'log_likelihood', -1200.0, -1150.0)",
        [run_id],
    )
    con.execute(
        "INSERT INTO recalibration_proposals (backtest_run_id, param_family, param_key, dimensions, "
        "old_params_version, new_params_version, old_value, new_value, metric_name, metric_before, metric_after) "
        "VALUES (?, 'risk_aversion_params', 'lambda_value', ?, 1, 2, 0.15, 0.05, 'realized_sharpe', 3.5, 4.3)",
        [run_id, json.dumps({"scope": "global"})],
    )

    n = rr._dump_proposals(con, run_id)

    assert n == 2
    payload = json.loads((tmp_path / "proposals.json").read_text())
    assert payload["backtest_run_id"] == run_id and payload["n_proposals"] == 2
    xi = next(p for p in payload["proposals"] if p["param_key"] == "xi")
    assert xi["old_value"] == 0.0018 and xi["new_value"] == 0.003
    assert xi["metric_delta"] == 50.0 and xi["status"] == "pending"
    lam = next(p for p in payload["proposals"] if p["param_key"] == "lambda_value")
    assert lam["dimensions"] == {"scope": "global"}
