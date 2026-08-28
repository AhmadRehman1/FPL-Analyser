"""scripts/summarize_ml_experiment_results.py -- run against a real (synthetic-DB) experiment
output and confirm it doesn't crash and reads every artifact it claims to summarize."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from research.ml import contract as C
from research.ml.experiment import run_experiment

_SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "summarize_ml_experiment_results.py"


def _load_summarize_module():
    spec = importlib.util.spec_from_file_location("summarize_ml_experiment_results", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summarize_script_runs_cleanly_against_real_experiment_output(seeded_db, monkeypatch, tmp_path, capsys):
    for attr, fname in [
        ("BASELINE_METRICS_JSON", "baseline_metrics.json"),
        ("BASELINE_PREDICTIONS_PARQUET", "baseline_predictions.parquet"),
        ("MODEL_COMPARISON_CSV", "model_comparison.csv"),
        ("RESIDUAL_ANALYSIS_CSV", "residual_analysis.csv"),
        ("HIGH_DISAGREEMENT_CSV", "high_disagreement_cases.csv"),
        ("CALIBRATION_CSV", "calibration.csv"),
        ("FEATURE_IMPORTANCE_CSV", "feature_importance.csv"),
        ("STABILITY_CSV", "feature_stability.csv"),
        ("IMPROVEMENT_CSV", "improvement.csv"),
        ("ENSEMBLE_CSV", "ensemble.csv"),
        ("SEASON_POINTS_CSV", "season_points.csv"),
        ("EXPERIMENT_MANIFEST_JSON", "experiment_manifest.json"),
        ("COMPUTE_RUNTIME_CSV", "compute_runtime.csv"),
        ("BOOTSTRAP_CI_JSON", "bootstrap_ci.json"),
        ("SLICED_MODEL_COMPARISON_CSV", "sliced_model_comparison.csv"),
    ]:
        monkeypatch.setattr(C, attr, tmp_path / fname)

    run_experiment(seasons=["2024-2025", "2025-2026"], con=seeded_db, random_seed=42)

    module = _load_summarize_module()
    monkeypatch.setattr(module, "C", C)  # the script imports contract itself -- point it at the same patched module
    module.main()

    out = capsys.readouterr().out
    assert "RUN METADATA" in out
    assert "HEADLINE METRICS" in out
    assert "BOOTSTRAP CONFIDENCE INTERVALS" in out
    assert "SLICING CHECK" in out
    assert "DECISION INPUT SUMMARY" in out


def test_summarize_script_exits_cleanly_with_a_clear_message_when_artifacts_are_missing(tmp_path, monkeypatch):
    module = _load_summarize_module()
    fake_missing_path = tmp_path / "does_not_exist.json"
    monkeypatch.setattr(module.C, "EXPERIMENT_MANIFEST_JSON", fake_missing_path)
    try:
        module.main()
        assert False, "expected SystemExit"
    except SystemExit as e:
        assert "Missing real experiment output" in str(e)
