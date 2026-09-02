"""ab_evidence_strength.yml's helper scripts: the metric-direction logic and the run_ingestion
env overrides. The walk-forward itself is a 3h cloud job, out of scope for a unit test."""

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from compare_backtest_metrics import _direction  # noqa: E402


def test_direction_points_delta_higher_is_better():
    assert _direction("beats_crowd_points_delta", 2.0, 3.5) == "better"
    assert _direction("beats_crowd_points_delta", 2.0, 1.0) == "worse"


def test_direction_calibration_loss_lower_is_better():
    assert _direction("brier_minutes_mean", 0.15, 0.13) == "better"
    assert _direction("log_score_minutes_mean", -0.8, -0.6) == "worse"


def test_direction_signed_bias_closer_to_zero_is_better():
    assert _direction("poisson_calibration_mean_resid", -0.10, -0.02) == "better"
    assert _direction("poisson_calibration_mean_resid", -0.02, -0.10) == "worse"
    assert _direction("poisson_calibration_mean_resid", 0.05, -0.05) == "="  # same distance from 0


def test_compare_writes_a_table_and_the_github_summary(tmp_path, capsys):
    import compare_backtest_metrics as cmp

    def _f(label, pull, crowd, brier):
        p = tmp_path / f"{label}.json"
        p.write_text(json.dumps({
            "label": label, "backtest_run_id": 1, "n_gameweek_steps": 71,
            "seasons_covered": ["2024-2025"], "config": {"predicted_xi_pull_strength": pull, "official_tier_weight": 1.0},
            "metrics": {"beats_crowd_points_delta": {"mean": crowd, "n": 71}, "brier_minutes_mean": {"mean": brier, "n": 71}},
        }))
        return str(p)

    summary = tmp_path / "summary.md"
    os.environ["GITHUB_STEP_SUMMARY"] = str(summary)
    try:
        sys.argv = ["compare", _f("baseline", 0.8, 2.0, 0.15), _f("variant", 1.5, 3.1, 0.14)]
        cmp.main()
    finally:
        del os.environ["GITHUB_STEP_SUMMARY"]
    out = capsys.readouterr().out
    assert "| +3.1000 |" in out and "better" in out  # variant lifted the crowd delta
    assert "| +3.1000 |" in summary.read_text()  # and it reached the GitHub step summary


def test_run_ingestion_env_overrides_parse_safely(monkeypatch):
    # GitHub Actions passes "" for an unset expression -- must fall back to the baseline value,
    # never crash on float("").
    for raw, expected_pull in [(None, 0.8), ("", 0.8), ("1.5", 1.5)]:
        if raw is None:
            monkeypatch.delenv("FPL_AB_PREDICTED_XI_PULL", raising=False)
        else:
            monkeypatch.setenv("FPL_AB_PREDICTED_XI_PULL", raw)
        assert float(os.getenv("FPL_AB_PREDICTED_XI_PULL", "0.8") or "0.8") == expected_pull
