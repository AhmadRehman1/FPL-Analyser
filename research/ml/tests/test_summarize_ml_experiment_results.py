"""scripts/summarize_ml_experiment_results.py -- run against a real (synthetic-DB) experiment
output and confirm it doesn't crash and reads every artifact it claims to summarize."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

from research.ml import contract as C
from research.ml.experiment import run_experiment

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "summarize_ml_experiment_results.py"


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


def test_summarize_script_runs_as_a_real_subprocess_from_repo_root():
    # Regression test for a real bug: the script only added REPO_ROOT/"src" to sys.path, never
    # REPO_ROOT itself, so its own documented invocation (`python scripts/summarize_ml_
    # experiment_results.py` from repo root) raised `ModuleNotFoundError: No module named
    # 'research'` immediately -- `research.ml.__init__.py`'s own path bootstrap never gets a
    # chance to run, since it fires only once `research.ml` is already importable. The other
    # tests in this file load the script in-process via importlib, which bypasses sys.path
    # entirely and could not have caught this. This test runs the real subprocess invocation
    # instead. This repo's real research/ml/results/ is gitignored and normally absent, so the
    # expected outcome is the script's own "Missing real experiment output" SystemExit message
    # -- but the one invariant this test actually exists to guard is "no ModuleNotFoundError",
    # regardless of whether a real run happens to have populated results/ locally.
    result = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH)],
        cwd=_REPO_ROOT, capture_output=True, text=True, timeout=30,
    )
    assert "ModuleNotFoundError" not in result.stderr, result.stderr
    if result.returncode != 0:
        assert "Missing real experiment output" in result.stderr
    else:
        assert "RUN METADATA" in result.stdout


def test_summarize_script_exits_cleanly_with_a_clear_message_when_artifacts_are_missing(tmp_path, monkeypatch):
    module = _load_summarize_module()
    fake_missing_path = tmp_path / "does_not_exist.json"
    monkeypatch.setattr(module.C, "EXPERIMENT_MANIFEST_JSON", fake_missing_path)
    try:
        module.main()
        assert False, "expected SystemExit"
    except SystemExit as e:
        assert "Missing real experiment output" in str(e)
