"""scripts/append_ml_run_to_history.py -- run against real (synthetic-DB) experiment output
twice, confirming the tracked history CSV accumulates rows and the trend summary compares
consecutive runs correctly."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from research.ml import contract as C
from research.ml.experiment import run_experiment

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "append_ml_run_to_history.py"

_ARTIFACTS = [
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
    ("FEATURE_IMPORTANCE_LIGHTGBM_CSV", "feature_importance_lightgbm.csv"),
    ("STABILITY_LIGHTGBM_CSV", "feature_stability_lightgbm.csv"),
    ("FEATURE_IMPORTANCE_XGBOOST_CSV", "feature_importance_xgboost.csv"),
    ("STABILITY_XGBOOST_CSV", "feature_stability_xgboost.csv"),
]


def _load_module():
    spec = importlib.util.spec_from_file_location("append_ml_run_to_history", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_real_experiment(seeded_db, monkeypatch, tmp_path, seed=42):
    for attr, fname in _ARTIFACTS:
        monkeypatch.setattr(C, attr, tmp_path / fname)
    run_experiment(seasons=["2024-2025", "2025-2026"], con=seeded_db, random_seed=seed)


def test_build_history_row_raises_clear_error_when_no_real_run_exists(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "EXPERIMENT_MANIFEST_JSON", tmp_path / "experiment_manifest.json")
    monkeypatch.setattr(C, "BOOTSTRAP_CI_JSON", tmp_path / "bootstrap_ci.json")
    module = _load_module()
    monkeypatch.setattr(module, "C", C)
    with pytest.raises(SystemExit, match="Missing real experiment output"):
        module.build_history_row()


def test_append_row_creates_then_accumulates(seeded_db, monkeypatch, tmp_path):
    _run_real_experiment(seeded_db, monkeypatch, tmp_path)
    history_csv = tmp_path / "weekly_quality_history.csv"
    monkeypatch.setattr(C, "RESULTS_HISTORY_DIR", tmp_path)
    monkeypatch.setattr(C, "RESULTS_HISTORY_CSV", history_csv)
    module = _load_module()
    monkeypatch.setattr(module, "C", C)

    row1 = module.build_history_row()
    history1 = module.append_row(row1)
    assert len(history1) == 1
    assert history_csv.exists()
    assert set(module.HISTORY_COLUMNS) <= set(pd.read_csv(history_csv).columns)

    # a second real run appends a second row rather than overwriting the first
    _run_real_experiment(seeded_db, monkeypatch, tmp_path, seed=43)
    row2 = module.build_history_row()
    history2 = module.append_row(row2)
    assert len(history2) == 2
    on_disk = pd.read_csv(history_csv)
    assert len(on_disk) == 2


def test_print_trend_summary_first_run_has_no_comparison(seeded_db, monkeypatch, tmp_path, capsys):
    _run_real_experiment(seeded_db, monkeypatch, tmp_path)
    monkeypatch.setattr(C, "RESULTS_HISTORY_DIR", tmp_path)
    monkeypatch.setattr(C, "RESULTS_HISTORY_CSV", tmp_path / "weekly_quality_history.csv")
    module = _load_module()
    monkeypatch.setattr(module, "C", C)

    history = module.append_row(module.build_history_row())
    module.print_trend_summary(history)
    captured = capsys.readouterr()
    assert "No prior run to compare against yet" in captured.out


def test_print_trend_summary_second_run_compares_to_first(seeded_db, monkeypatch, tmp_path, capsys):
    _run_real_experiment(seeded_db, monkeypatch, tmp_path, seed=42)
    monkeypatch.setattr(C, "RESULTS_HISTORY_DIR", tmp_path)
    monkeypatch.setattr(C, "RESULTS_HISTORY_CSV", tmp_path / "weekly_quality_history.csv")
    module = _load_module()
    monkeypatch.setattr(module, "C", C)
    module.append_row(module.build_history_row())

    _run_real_experiment(seeded_db, monkeypatch, tmp_path, seed=43)
    history = module.append_row(module.build_history_row())
    module.print_trend_summary(history)
    captured = capsys.readouterr()
    assert "Season points: quant" in captured.out
    assert "->" in captured.out


def test_main_end_to_end(seeded_db, monkeypatch, tmp_path, capsys):
    _run_real_experiment(seeded_db, monkeypatch, tmp_path)
    monkeypatch.setattr(C, "RESULTS_HISTORY_DIR", tmp_path)
    monkeypatch.setattr(C, "RESULTS_HISTORY_CSV", tmp_path / "weekly_quality_history.csv")
    module = _load_module()
    monkeypatch.setattr(module, "C", C)

    module.main()

    assert (tmp_path / "weekly_quality_history.csv").exists()
    captured = capsys.readouterr()
    assert "Logged run 1 of the tracked quality history" in captured.out
