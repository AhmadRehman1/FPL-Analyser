"""Baseline-model tests: Quant baseline metrics, historical baseline, and per-slice breakdowns."""

from __future__ import annotations

import numpy as np
import pytest

from research.ml import contract as C
from research.ml.baselines import (
    compute_metrics,
    historical_baseline_predictions,
    metrics_to_dict,
    quant_predictions,
    save_baseline_metrics,
    sliced_metrics,
)
from research.ml.dataset_builder import build_dataset


def test_compute_metrics_basic():
    actual = np.array([3.0, 5.0, 2.0, 0.0, 4.0])
    pred = np.array([3.0, 4.0, 2.0, 1.0, 4.0])
    m = compute_metrics(pred, actual)
    assert m.mae == pytest.approx(np.mean(np.abs(pred - actual)))
    assert m.rmse == pytest.approx(np.sqrt(np.mean((pred - actual) ** 2)))
    assert m.n == 5
    assert -1.0 <= m.correlation <= 1.0


def test_compute_metrics_zero_error_perfect():
    actual = np.array([2.0, 4.0, 6.0])
    m = compute_metrics(actual.copy(), actual)
    assert m.mae == 0.0
    assert m.rmse == 0.0


def test_quant_predictions_match_dataset(seeded_db):
    df = build_dataset(seeded_db, with_features=True)
    pred = quant_predictions(df)
    assert len(pred) == len(df)
    assert np.allclose(pred, df["quant_prediction"].to_numpy())


def test_historical_baseline_predictions_length(seeded_db):
    df = build_dataset(seeded_db, with_features=True)
    pred = historical_baseline_predictions(df)
    assert len(pred) == len(df)
    # historical baseline is clipped at 0
    assert (pred >= 0).all()


def test_save_baseline_metrics_writes_artifacts(seeded_db, tmp_path, monkeypatch):
    df = build_dataset(seeded_db, with_features=True)
    # redirect the results dir to a temp path
    monkeypatch.setattr("research.ml.contract.BASELINE_METRICS_JSON", tmp_path / "baseline_metrics.json")
    monkeypatch.setattr("research.ml.contract.BASELINE_PREDICTIONS_PARQUET", tmp_path / "baseline_predictions.parquet")
    payload = save_baseline_metrics(df)
    assert (tmp_path / "baseline_metrics.json").exists()
    assert (tmp_path / "baseline_predictions.parquet").exists()
    assert "overall" in payload
    assert payload["overall"]["n"] == len(df)


def test_sliced_metrics_breaks_down_by_position(seeded_db):
    df = build_dataset(seeded_db, with_features=True)
    sliced = sliced_metrics(df, quant_predictions(df))
    assert C.COL_POSITION in sliced
    pos_slices = sliced[C.COL_POSITION]
    assert len(pos_slices) > 0
    assert {"mae", "n"} <= set(metrics_to_dict(next(iter(pos_slices.values()))).keys())
