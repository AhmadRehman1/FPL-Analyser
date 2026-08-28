"""End-to-end experiment orchestrator test. Runs run_experiment against the synthetic DB and
verifies every required artifact is produced with sensible shape.

The results-dir path constants are monkeypatched on the `research.ml.contract` module, so this
test imports the module (not individual constants) and reads the attributes back at assertion
time -- a `from research.ml.contract import MODEL_COMPARISON_CSV` would bind the pre-patch value
and silently pass against a stale directory from a prior smoke run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from research.ml import contract as C
from research.ml.experiment import run_experiment
from research.ml.residual_model import lightgbm_available, sklearn_available

# (contract attribute name, filename) for every artifact the experiment must produce.
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
]


def test_run_experiment_produces_all_artifacts(seeded_db, monkeypatch, tmp_path):
    # redirect the results dir to a temp location so the test is hermetic
    for attr, fname in _ARTIFACTS:
        monkeypatch.setattr(C, attr, tmp_path / fname)

    payload = run_experiment(seasons=["2024-2025", "2025-2026"], con=seeded_db, random_seed=42)

    # every artifact file exists -- read the (now-monkeypatched) attribute back from the module
    for attr, _ in _ARTIFACTS:
        path = getattr(C, attr)
        assert Path(path).exists(), f"missing artifact {attr} at {path}"

    # manifest is valid json with required keys
    manifest = json.loads(Path(C.EXPERIMENT_MANIFEST_JSON).read_text())
    assert manifest["dataset_rows"] > 0
    assert manifest["dataset_seasons"] == ["2024-2025", "2025-2026"]
    assert len(manifest["walk_forward_folds"]) >= 1
    assert manifest["git_commit"]  # git commit recorded for reproducibility
    # gameweek walk-forward: many folds, first/last test step recorded, season points present
    assert manifest["fold_mode"] == "gameweek"
    assert manifest["n_walk_forward_folds"] > 1
    assert manifest["first_test_step"]["season"] == "2024-2025"
    assert manifest["last_test_step"]["season"] == "2025-2026"
    assert "quant_manager" in manifest["season_points"]
    assert "ml_manager" in manifest["season_points"]

    # comparison table includes the quant baseline + at least one ML model
    comp = pd.read_csv(C.MODEL_COMPARISON_CSV)
    assert "quant" in set(comp["model"])
    assert "quant_linear" in set(comp["model"])
    # Track F, R8: quant_lightgbm must appear whenever lightgbm is actually installed --
    # this environment has it, so its absence here would be a real regression, not a skip.
    if lightgbm_available():
        assert "quant_lightgbm" in set(comp["model"])
    if sklearn_available():
        assert "quant_gbm" in set(comp["model"])

    # season points table compares the quant manager vs the ML manager
    sp = pd.read_csv(C.SEASON_POINTS_CSV)
    assert "quant_prediction" in set(sp["signal"])
    assert "ml_prediction" in set(sp["signal"])
    assert (sp["total_points"] >= 0).all()

    # ensemble table has a best weight in the allowed grid
    ens = pd.read_csv(C.ENSEMBLE_CSV)
    assert ens["best_w"].iloc[0] in {0.0, 0.25, 0.5, 0.75, 1.0}

    # Track F, R10: per-model compute/runtime and bootstrap CIs are real, non-empty artifacts
    # whenever the corresponding model actually ran, not just empty placeholder files.
    runtime = pd.read_csv(C.COMPUTE_RUNTIME_CSV)
    assert "quant_linear" in set(runtime["model"])
    assert (runtime["fit_predict_seconds"] >= 0).all()
    bootstrap = json.loads(Path(C.BOOTSTRAP_CI_JSON).read_text())
    assert "quant_linear" in bootstrap
    assert "statistically_credible_improvement" in bootstrap["quant_linear"]
    if lightgbm_available():
        assert "quant_lightgbm" in bootstrap
        assert "quant_lightgbm" in set(runtime["model"])
    # R11: manifest carries the bootstrap CI results directly, so Phase F-4 doesn't have to
    # re-derive them from bootstrap_ci.json separately.
    assert manifest["bootstrap_ci"] == bootstrap
    assert manifest["lightgbm_available"] == lightgbm_available()

    # payload returned to caller matches what was written
    assert set(payload.keys()) >= {"manifest", "comparison", "improvement", "bootstrap_ci"}


def test_run_experiment_survives_a_lightgbm_runtime_failure_not_just_import_failure(seeded_db, monkeypatch, tmp_path):
    """R14: 'IF LightGBM fails to install OR RUN...' -- a Critique Engine pass on Phase F-2
    found the first version only handled the install-missing case (ImportError at construction
    time); a genuine runtime failure inside fit() (lightgbm's native library has a real history
    of environment-specific issues, e.g. Windows OpenMP/threading problems) propagated raw and
    aborted the whole multi-fold run, losing quant/quant_linear/quant_gbm results too -- not
    just the LightGBM arm. This test simulates exactly that (a non-ImportError exception from
    inside fit(), not lightgbm being absent) and asserts the rest of the pipeline still
    completes -- the gap the previous, narrower "unavailable" tests did not cover."""
    for attr, fname in _ARTIFACTS:
        monkeypatch.setattr(C, attr, tmp_path / fname)

    if not lightgbm_available():
        pytest.skip("lightgbm is not installed in this environment; the real .fit() runtime-failure path cannot be exercised")

    # Patch the underlying LGBMRegressor.fit itself -- NOT LightGBMResidualModel -- so the real
    # class's own try/except (the actual fix under test) is what has to do the converting to
    # ResidualModelUnavailableError. Replacing the whole class, as an earlier version of this
    # test did, bypasses that logic entirely and tests nothing real.
    import lightgbm

    def _always_fails(self, X, y, *a, **k):
        raise RuntimeError("simulated native-library failure -- not an ImportError")

    monkeypatch.setattr(lightgbm.LGBMRegressor, "fit", _always_fails)

    payload = run_experiment(seasons=["2024-2025", "2025-2026"], con=seeded_db, random_seed=42)

    comp = pd.read_csv(C.MODEL_COMPARISON_CSV)
    assert "quant" in set(comp["model"])
    assert "quant_linear" in set(comp["model"])
    assert "quant_lightgbm" not in set(comp["model"])  # the failing arm is genuinely absent...
    assert not comp.empty  # ...but nothing else was lost
    assert payload["manifest"]["lightgbm_available"] is True  # import succeeded; the *fit* failed
