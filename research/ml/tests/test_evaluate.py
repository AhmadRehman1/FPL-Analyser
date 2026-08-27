"""Evaluate-module tests: comparison rows, improvement rows, residual analysis, disagreement,
calibration, ensemble weight selection, and stability table."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.ml.dataset_builder import build_dataset
from research.ml.evaluate import (
    calibration,
    ensemble_predictions,
    evaluate_ensemble,
    high_disagreement_cases,
    improvement_rows,
    model_comparison_rows,
    residual_analysis,
    stability_table,
)
from research.ml.feature_engineering import feature_columns
from research.ml.residual_model import LinearResidualModel, Preprocessor
from research.ml.walk_forward import default_folds


def _fit_and_predict(train_df, test_df):
    feats = feature_columns()
    pp = Preprocessor().fit(train_df, feats)
    Xtr, names = pp.transform(train_df)
    m = LinearResidualModel().fit(Xtr, train_df["residual"].to_numpy())
    Xte, _ = pp.transform(test_df)
    ml_residual = m.predict(Xte)
    ml_pred = test_df["quant_prediction"].to_numpy() + ml_residual
    return ml_pred


def test_model_comparison_rows(seeded_db):
    df = build_dataset(seeded_db, with_features=True)
    folds = default_folds(df)
    f = folds[0]
    ml_pred = _fit_and_predict(f.train_df, f.test_df)
    rows = model_comparison_rows(f.name, f.test_season, f.test_df, {
        "quant": f.test_df["quant_prediction"].to_numpy(),
        "quant_linear": ml_pred,
    })
    assert len(rows) == 2
    assert {r["model"] for r in rows} == {"quant", "quant_linear"}
    assert all(r["n"] == len(f.test_df) for r in rows)


def test_improvement_rows(seeded_db):
    df = build_dataset(seeded_db, with_features=True)
    folds = default_folds(df)
    f = folds[0]
    ml_pred = _fit_and_predict(f.train_df, f.test_df)
    comp = pd.DataFrame(model_comparison_rows(f.name, f.test_season, f.test_df, {
        "quant": f.test_df["quant_prediction"].to_numpy(),
        "quant_linear": ml_pred,
    }))
    imp = improvement_rows(comp)
    assert len(imp) > 0
    assert set(imp.columns) >= {"season", "model", "metric", "improvement", "improvement_pct"}
    # improvement = quant_error - ml_error (positive means ML is better)
    row = imp[imp["metric"] == "mae"].iloc[0]
    assert row["improvement"] == pytest.approx(row["quant_error"] - row["ml_error"])


def test_residual_analysis(seeded_db):
    df = build_dataset(seeded_db, with_features=True)
    pred = df["quant_prediction"].to_numpy()
    res = residual_analysis(df, pred)
    assert len(res) > 0
    assert set(res.columns) >= {"dimension", "slice", "n", "mean_error", "median_error", "std_error"}


def test_high_disagreement_cases(seeded_db):
    df = build_dataset(seeded_db, with_features=True)
    quant = df["quant_prediction"].to_numpy()
    ml = quant + np.random.RandomState(0).normal(0, 1, len(df))
    cases = high_disagreement_cases(df, quant, ml, top_n=10)
    assert len(cases) <= 10
    assert set(cases.columns) >= {"quant_prediction", "ml_prediction", "difference", "quant_error", "ml_error"}
    # sorted by absolute difference descending
    assert cases["difference"].abs().is_monotonic_decreasing


def test_calibration(seeded_db):
    df = build_dataset(seeded_db, with_features=True)
    pred = df["quant_prediction"].to_numpy()
    cal = calibration(df, pred, "quant")
    assert len(cal) > 0
    assert set(cal.columns) == {"model", "bucket", "mean_predicted", "mean_actual", "n"}


def test_ensemble_predictions_linear_combination():
    quant = np.array([4.0, 6.0, 2.0])
    ml = np.array([5.0, 3.0, 2.0])
    # w=0 -> pure ML, w=1 -> pure quant
    assert np.allclose(ensemble_predictions(quant, ml, 0.0), ml)
    assert np.allclose(ensemble_predictions(quant, ml, 1.0), quant)
    assert np.allclose(ensemble_predictions(quant, ml, 0.5), 0.5 * quant + 0.5 * ml)


def test_evaluate_ensemble_selects_best_train_weight(seeded_db):
    df = build_dataset(seeded_db, with_features=True)
    folds = default_folds(df)
    f = folds[0]
    ml_train = _fit_and_predict(f.train_df, f.train_df)
    ml_test = _fit_and_predict(f.train_df, f.test_df)
    result = evaluate_ensemble(f.train_df, f.test_df, ml_train, ml_test)
    assert "best_w" in result
    assert result["best_w"] in {0.0, 0.25, 0.5, 0.75, 1.0}
    assert len(result["grid"]) == 5
    assert "test_pred" in result
    assert len(result["test_pred"]) == len(f.test_df)


def test_stability_table(seeded_db):
    df = build_dataset(seeded_db, with_features=True)
    folds = default_folds(df)
    f = folds[0]
    feats = feature_columns()
    pp = Preprocessor().fit(f.train_df, feats)
    Xtr, names = pp.transform(f.train_df)
    m = LinearResidualModel().fit(Xtr, f.train_df["residual"].to_numpy())
    fi1 = m.feature_importance(names)
    fi2 = m.feature_importance(names)
    st = stability_table([fi1, fi2])
    assert len(st) > 0
    assert set(st.columns) == {"feature", "mean_importance", "std_importance", "cv"}
