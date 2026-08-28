"""scripts/narrate_ml_results.py -- run its numeric-summary builder against a real
(synthetic-DB) experiment output, and its main() with a fake Ollama `chat()` (no real Ollama
server or network access is used or required)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from research.ml import contract as C
from research.ml.experiment import run_experiment
from research.ml.ollama_client import OllamaResponse

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "narrate_ml_results.py"


def _load_narrate_module():
    spec = importlib.util.spec_from_file_location("narrate_ml_results", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _run_real_experiment(seeded_db, monkeypatch, tmp_path):
    for attr, fname in _ARTIFACTS:
        monkeypatch.setattr(C, attr, tmp_path / fname)
    run_experiment(seasons=["2024-2025", "2025-2026"], con=seeded_db, random_seed=42)


def test_build_numeric_summary_contains_real_numbers(seeded_db, monkeypatch, tmp_path):
    _run_real_experiment(seeded_db, monkeypatch, tmp_path)
    module = _load_narrate_module()
    monkeypatch.setattr(module, "C", C)

    summary = module.build_numeric_summary()
    assert "quant_lightgbm" in summary or "quant" in summary
    assert "Bootstrap 95%" in summary
    assert "Season manager simulation" in summary
    # Never invents a number -- everything printed traces back to a real source file, so this
    # is a smoke check that the function actually ran real aggregation, not a stub.
    assert "n/a" not in summary or "MAE" in summary


def test_build_numeric_summary_raises_clear_error_when_no_real_run_exists(monkeypatch, tmp_path):
    for attr, fname in _ARTIFACTS:
        monkeypatch.setattr(C, attr, tmp_path / fname)  # none of these exist yet
    module = _load_narrate_module()
    monkeypatch.setattr(module, "C", C)
    with pytest.raises(SystemExit, match="Missing real experiment output"):
        module.build_numeric_summary()


def test_main_writes_narrative_draft_with_disclaimer(seeded_db, monkeypatch, tmp_path, capsys):
    _run_real_experiment(seeded_db, monkeypatch, tmp_path)
    monkeypatch.setattr(C, "RESULTS_DIR", tmp_path)
    module = _load_narrate_module()
    monkeypatch.setattr(module, "C", C)
    monkeypatch.setattr(
        module, "chat",
        lambda system_prompt, user_prompt, model, host: OllamaResponse(text="Draft summary text.", model="llama3.1"),
    )
    monkeypatch.setattr("sys.argv", ["narrate_ml_results.py"])

    module.main()

    out_path = tmp_path / "narrative_draft.md"
    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")
    assert "AI-generated, for human review only" in content
    assert "Draft summary text." in content
    assert "REPORT.md" in content
    captured = capsys.readouterr()
    assert "Draft summary text." in captured.out


def test_main_print_only_skips_writing_file(seeded_db, monkeypatch, tmp_path, capsys):
    _run_real_experiment(seeded_db, monkeypatch, tmp_path)
    monkeypatch.setattr(C, "RESULTS_DIR", tmp_path)
    module = _load_narrate_module()
    monkeypatch.setattr(module, "C", C)
    monkeypatch.setattr(
        module, "chat",
        lambda system_prompt, user_prompt, model, host: OllamaResponse(text="Draft summary text.", model="llama3.1"),
    )
    monkeypatch.setattr("sys.argv", ["narrate_ml_results.py", "--print-only"])

    module.main()

    assert not (tmp_path / "narrative_draft.md").exists()


def test_main_surfaces_ollama_unavailable_error_as_system_exit(seeded_db, monkeypatch, tmp_path):
    _run_real_experiment(seeded_db, monkeypatch, tmp_path)
    module = _load_narrate_module()
    monkeypatch.setattr(module, "C", C)

    def _raise(*a, **k):
        raise module.OllamaUnavailableError("Cannot reach a local Ollama server -- start it with `ollama serve`.")
    monkeypatch.setattr(module, "chat", _raise)
    monkeypatch.setattr("sys.argv", ["narrate_ml_results.py"])

    with pytest.raises(SystemExit, match="ollama serve"):
        module.main()
