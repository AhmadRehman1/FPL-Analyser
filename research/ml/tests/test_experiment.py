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

from research.ml import contract as C
from research.ml.experiment import run_experiment

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

    # season points table compares the quant manager vs the ML manager
    sp = pd.read_csv(C.SEASON_POINTS_CSV)
    assert "quant_prediction" in set(sp["signal"])
    assert "ml_prediction" in set(sp["signal"])
    assert (sp["total_points"] >= 0).all()

    # ensemble table has a best weight in the allowed grid
    ens = pd.read_csv(C.ENSEMBLE_CSV)
    assert ens["best_w"].iloc[0] in {0.0, 0.25, 0.5, 0.75, 1.0}

    # payload returned to caller matches what was written
    assert set(payload.keys()) >= {"manifest", "comparison", "improvement"}
