"""Residual-model tests: Preprocessor fit/transform, LinearResidualModel (numpy ridge fallback
when sklearn absent), GradientBoostingResidualModel graceful failure without sklearn, and
feature importance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.ml.dataset_builder import build_dataset
from research.ml.feature_engineering import feature_columns
from research.ml.residual_model import (
    GradientBoostingResidualModel,
    LinearResidualModel,
    Preprocessor,
    ResidualModelUnavailableError,
    sklearn_available,
)


def test_preprocessor_fit_transform_shapes(seeded_db):
    df = build_dataset(seeded_db, with_features=True)
    feats = feature_columns()
    pp = Preprocessor().fit(df, feats)
    X, names = pp.transform(df)
    assert X.shape[0] == len(df)
    assert len(names) == X.shape[1]
    # no NaNs remain after imputation
    assert not np.isnan(X).any()


def test_preprocessor_fit_on_train_transform_test_consistent(seeded_db):
    df = build_dataset(seeded_db, with_features=True)
    train = df[df["season"] == "2024-2025"]
    test = df[df["season"] == "2025-2026"]
    feats = feature_columns()
    pp = Preprocessor().fit(train, feats)
    Xtr, names_tr = pp.transform(train)
    Xte, names_te = pp.transform(test)
    assert names_tr == names_te  # same feature order
    assert Xtr.shape[1] == Xte.shape[1]


def test_linear_model_fit_predict(seeded_db):
    df = build_dataset(seeded_db, with_features=True)
    train = df[df["season"] == "2024-2025"]
    test = df[df["season"] == "2025-2026"]
    feats = feature_columns()
    pp = Preprocessor().fit(train, feats)
    Xtr, names = pp.transform(train)
    m = LinearResidualModel().fit(Xtr, train["residual"].to_numpy())
    Xte, _ = pp.transform(test)
    pred = m.predict(Xte)
    assert len(pred) == len(test)
    assert np.isfinite(pred).all()


def test_linear_model_feature_importance(seeded_db):
    df = build_dataset(seeded_db, with_features=True)
    train = df[df["season"] == "2024-2025"]
    feats = feature_columns()
    pp = Preprocessor().fit(train, feats)
    Xtr, names = pp.transform(train)
    m = LinearResidualModel().fit(Xtr, train["residual"].to_numpy())
    fi = m.feature_importance(names)
    assert set(fi.columns) == {"feature", "importance", "direction"}
    assert len(fi) == len(names)


def test_linear_model_uses_numpy_fallback_when_no_sklearn(monkeypatch):
    # force the sklearn import to fail inside fit
    import sys
    monkeypatch.setitem(sys.modules, "sklearn.linear_model", None)
    df = pd.DataFrame({
        "x1": np.random.RandomState(0).randn(50),
        "x2": np.random.RandomState(1).randn(50),
    })
    X = df[["x1", "x2"]].to_numpy()
    y = 2.0 * df["x1"].to_numpy() - 1.0 * df["x2"].to_numpy() + 0.1
    m = LinearResidualModel().fit(X, y)
    assert m._backend is None  # fell back to numpy closed-form
    pred = m.predict(X)
    assert np.isfinite(pred).all()


def test_gbm_raises_when_sklearn_unavailable(monkeypatch):
    if sklearn_available():
        pytest.skip("sklearn is installed in this environment; cannot test the unavailable path")
    with pytest.raises(ResidualModelUnavailableError):
        GradientBoostingResidualModel().fit(np.zeros((3, 2)), np.zeros(3))
