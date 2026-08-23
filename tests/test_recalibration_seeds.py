import json
from pathlib import Path

import pytest

from fpl_quant import backtest as bt

REPO_ROOT = Path(__file__).resolve().parents[1]


def _seed_backtest_run(con, notes="test run"):
    return con.execute(
        "INSERT INTO backtest_runs (warm_up_gameweeks, notes) VALUES (0, ?) RETURNING backtest_run_id", [notes]
    ).fetchone()[0]


def test_write_recalibration_seeds_writes_shape(con, tmp_path):
    backtest_run_id = _seed_backtest_run(con)
    p1 = bt.propose_recalibration(
        con, backtest_run_id, "model_decay_params", "xi", 0.003,
        metric_name="neg_log_likelihood", metric_before=100.0, metric_after=95.0, old_params_version=None,
    )
    p2 = bt.propose_recalibration(
        con, backtest_run_id, "risk_aversion_params", "lambda_value", 0.20,
        metric_name="realized_sharpe", metric_before=0.0, metric_after=1.2, old_params_version=None,
    )

    out_path = bt.write_recalibration_seeds(con, backtest_run_id, [p1, p2], out_dir=tmp_path)

    assert out_path == tmp_path / f"seeds_{backtest_run_id}.json"
    data = json.loads(out_path.read_text())
    assert data["backtest_run_id"] == backtest_run_id
    assert "generated_at" in data
    assert set(data["seeds"]) == {"model_decay_params:xi", "risk_aversion_params:lambda_value"}
    xi_seed = data["seeds"]["model_decay_params:xi"]
    assert xi_seed["value"] == pytest.approx(0.003)
    assert xi_seed["status"] == "pending"
    assert xi_seed["param_family"] == "model_decay_params"
    assert xi_seed["param_key"] == "xi"


def test_write_recalibration_seeds_raises_on_no_proposals(con, tmp_path):
    backtest_run_id = _seed_backtest_run(con)
    with pytest.raises(ValueError, match="no proposal_ids given"):
        bt.write_recalibration_seeds(con, backtest_run_id, [], out_dir=tmp_path)


def test_load_confirmed_recalibration_seeds_empty_for_missing_file(tmp_path):
    assert bt.load_confirmed_recalibration_seeds(tmp_path / "nope.json") == {}


def test_load_confirmed_recalibration_seeds_excludes_pending(con, tmp_path):
    """The central Gate G1 guarantee: a seed marked 'pending' -- exactly what
    write_recalibration_seeds() always produces immediately after recalibrate() -- must NEVER
    be loaded as an active default, however plausible the fresher number looks."""
    backtest_run_id = _seed_backtest_run(con)
    p1 = bt.propose_recalibration(
        con, backtest_run_id, "model_decay_params", "xi", 0.999,
        metric_name="neg_log_likelihood", metric_before=100.0, metric_after=1.0, old_params_version=None,
    )
    seed_path = bt.write_recalibration_seeds(con, backtest_run_id, [p1], out_dir=tmp_path)

    confirmed = bt.load_confirmed_recalibration_seeds(seed_path)
    assert confirmed == {}


def test_load_confirmed_recalibration_seeds_returns_confirmed_entries(tmp_path):
    seed_path = tmp_path / "seeds_1.json"
    seed_path.write_text(json.dumps({
        "backtest_run_id": 1,
        "seeds": {
            "model_decay_params:xi": {
                "param_family": "model_decay_params", "param_key": "xi", "dimensions": None,
                "value": 0.005, "params_version": 2, "status": "confirmed",
            },
            "risk_aversion_params:lambda_value": {
                "param_family": "risk_aversion_params", "param_key": "lambda_value", "dimensions": None,
                "value": 0.20, "params_version": 2, "status": "pending",
            },
        },
    }))

    confirmed = bt.load_confirmed_recalibration_seeds(seed_path)
    assert confirmed == {("model_decay_params", "xi"): {"value": 0.005, "params_version": 2}}


def test_resolve_confirmed_seed_or_warn_returns_seed_silently_when_present(tmp_path, capsys):
    seed_path = tmp_path / "seeds.json"
    seed_path.write_text(json.dumps({
        "seeds": {
            "model_decay_params:xi": {
                "param_family": "model_decay_params", "param_key": "xi", "dimensions": None,
                "value": 0.005, "params_version": 2, "status": "confirmed",
            },
        },
    }))
    result = bt.resolve_confirmed_seed_or_warn(
        seed_path, "model_decay_params", "xi", fallback_value=0.0018, fallback_params_version=1,
    )
    assert result == {"value": 0.005, "params_version": 2}
    assert capsys.readouterr().out == ""


def test_resolve_confirmed_seed_or_warn_falls_back_and_warns_when_missing(tmp_path, capsys):
    result = bt.resolve_confirmed_seed_or_warn(
        tmp_path / "nope.json", "model_decay_params", "xi", fallback_value=0.005, fallback_params_version=2,
    )
    assert result == {"value": 0.005, "params_version": 2}
    out = capsys.readouterr().out
    assert "::warning::backtest.resolve_confirmed_seed_or_warn" in out
    assert "falling back" in out


def test_resolve_confirmed_seed_or_warn_falls_back_and_warns_when_only_pending(tmp_path, capsys):
    seed_path = tmp_path / "seeds.json"
    seed_path.write_text(json.dumps({
        "seeds": {
            "model_decay_params:xi": {
                "param_family": "model_decay_params", "param_key": "xi", "dimensions": None,
                "value": 0.999, "params_version": 3, "status": "pending",
            },
        },
    }))
    result = bt.resolve_confirmed_seed_or_warn(
        seed_path, "model_decay_params", "xi", fallback_value=0.005, fallback_params_version=2,
    )
    assert result == {"value": 0.005, "params_version": 2}  # NOT the pending 0.999
    assert "::warning::backtest.resolve_confirmed_seed_or_warn" in capsys.readouterr().out


def test_committed_confirmed_v2_seed_file_is_well_formed_and_matches_run_ingestion():
    """The real, committed seed file run_ingestion.py actually loads from -- see
    data/recalibration/seeds_confirmed_v2.json. Guards against the file drifting out of sync
    with the xi=0.005/rho_residual=0.0 values README documents as the real, already-confirmed
    commit-7bf7604 activation."""
    seed_path = REPO_ROOT / "data" / "recalibration" / "seeds_confirmed_v2.json"
    assert seed_path.exists()
    confirmed = bt.load_confirmed_recalibration_seeds(seed_path)
    assert confirmed == {
        ("model_decay_params", "xi"): {"value": 0.005, "params_version": 2},
        ("correlation_params", "rho_residual"): {"value": 0.0, "params_version": 2},
    }
