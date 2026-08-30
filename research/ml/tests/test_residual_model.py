"""Residual-model tests: Preprocessor fit/transform, LinearResidualModel (numpy ridge fallback
when sklearn absent), GradientBoostingResidualModel graceful failure without sklearn,
LightGBMResidualModel (the primary nonlinear challenger -- Track F,
docs/plans/2026-08_retrospective_validation_and_ml_decision_plan.md, R8/R11), and feature
importance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.ml.dataset_builder import build_dataset
from research.ml.feature_engineering import feature_columns
from research.ml.residual_model import (
    GradientBoostingResidualModel,
    LightGBMResidualModel,
    LinearResidualModel,
    Preprocessor,
    ResidualModelUnavailableError,
    XGBoostResidualModel,
    lightgbm_available,
    sklearn_available,
    xgboost_available,
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


def test_preprocessor_one_hot_encodes_position(seeded_db):
    # position is an approved static-identity feature (EXISTING_MODEL_AUDIT.md §9,
    # LEAKAGE_PROTOCOL.md §4) that was attached to every dataset row but never actually listed
    # in feature_columns() until now -- must be one-hot encoded alongside status.
    df = build_dataset(seeded_db, with_features=True)
    feats = feature_columns()
    pp = Preprocessor().fit(df, feats)
    _, names = pp.transform(df)
    assert any(n.startswith("position=") for n in names)
    assert "status" in pp.categorical_cols
    assert "position" in pp.categorical_cols


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


# ============================================================
# LightGBMResidualModel -- the primary nonlinear challenger (Track F, R8/R11): the arm the
# ship/no-ship decision is governed by. Kept alongside, not instead of, quant_gbm above.
# ============================================================

def test_lightgbm_model_fit_predict(seeded_db):
    if not lightgbm_available():
        pytest.skip("lightgbm is not installed in this environment")
    df = build_dataset(seeded_db, with_features=True)
    train = df[df["season"] == "2024-2025"]
    test = df[df["season"] == "2025-2026"]
    feats = feature_columns()
    pp = Preprocessor().fit(train, feats)
    Xtr, names = pp.transform(train)
    m = LightGBMResidualModel().fit(Xtr, train["residual"].to_numpy())
    Xte, _ = pp.transform(test)
    pred = m.predict(Xte)
    assert len(pred) == len(test)
    assert np.isfinite(pred).all()


def test_lightgbm_model_feature_importance(seeded_db):
    if not lightgbm_available():
        pytest.skip("lightgbm is not installed in this environment")
    df = build_dataset(seeded_db, with_features=True)
    train = df[df["season"] == "2024-2025"]
    feats = feature_columns()
    pp = Preprocessor().fit(train, feats)
    Xtr, names = pp.transform(train)
    y = train["residual"].to_numpy()
    m = LightGBMResidualModel().fit(Xtr, y)
    fi = m.feature_importance(Xtr, y, names)
    assert set(fi.columns) == {"feature", "importance", "direction"}
    assert len(fi) == len(names)


def test_lightgbm_default_hyperparameters_are_regularised():
    # exhaustive gameweek walk-forward means some folds train on very little data -- the
    # defaults must regularise (L1/L2, row/column subsampling) on top of the existing
    # max_depth cap, or an early small fold could memorise noise.
    m = LightGBMResidualModel()
    assert m.reg_alpha > 0
    assert m.reg_lambda > 0
    assert 0 < m.subsample < 1
    assert 0 < m.colsample_bytree < 1


def test_lightgbm_objective_is_configurable_and_defaults_to_l2(seeded_db):
    # the class default stays "regression" (L2) so nothing else that constructs it changes
    # behaviour; the experiment orchestrator opts into "regression_l1" explicitly.
    assert LightGBMResidualModel().objective == "regression"
    if not lightgbm_available():
        pytest.skip("lightgbm is not installed in this environment")
    train = build_dataset(seeded_db, with_features=True)
    pp = Preprocessor().fit(train, feature_columns())
    Xtr, _ = pp.transform(train)
    y = train["residual"].to_numpy()
    m = LightGBMResidualModel(objective="regression_l1").fit(Xtr, y)
    assert m._model.get_params()["objective"] == "regression_l1"
    assert m.predict(Xtr).shape == (len(train),)


def test_lightgbm_predict_before_fit_raises():
    m = LightGBMResidualModel()
    with pytest.raises(ResidualModelUnavailableError):
        m.predict(np.zeros((3, 2)))


def test_lightgbm_raises_when_unavailable(monkeypatch):
    if not lightgbm_available():
        pytest.skip("lightgbm is not installed in this environment; the real ImportError path already covers this")
    # Force the lazy `from lightgbm import LGBMRegressor` inside fit() to fail, the same
    # technique test_linear_model_uses_numpy_fallback_when_no_sklearn already uses for sklearn
    # -- proves R14's fallback path (a real LightGBM failure must not crash the whole
    # experiment) actually raises the documented, catchable exception type.
    import sys
    monkeypatch.setitem(sys.modules, "lightgbm", None)
    with pytest.raises(ResidualModelUnavailableError):
        LightGBMResidualModel().fit(np.random.RandomState(0).randn(20, 3), np.random.RandomState(1).randn(20))


def test_lightgbm_feature_importance_raises_when_sklearn_unavailable(seeded_db, monkeypatch):
    """feature_importance() reuses sklearn's permutation_importance (see module docstring for
    why) -- a separate, real dependency from lightgbm itself, so its own unavailable-path needs
    its own test rather than assuming lightgbm's own ResidualModelUnavailableError test covers it."""
    if not lightgbm_available():
        pytest.skip("lightgbm is not installed in this environment")
    df = build_dataset(seeded_db, with_features=True)
    train = df[df["season"] == "2024-2025"]
    feats = feature_columns()
    pp = Preprocessor().fit(train, feats)
    Xtr, names = pp.transform(train)
    y = train["residual"].to_numpy()
    m = LightGBMResidualModel().fit(Xtr, y)
    import sys
    monkeypatch.setitem(sys.modules, "sklearn.inspection", None)
    with pytest.raises(ResidualModelUnavailableError):
        m.feature_importance(Xtr, y, names)


# ============================================================
# XGBoostResidualModel -- the R13 independent-implementation confirmation arm (mirrors
# LightGBMResidualModel's own test coverage above; this model is informational only and does
# not govern R11's ship/no-ship decision).
# ============================================================

def test_xgboost_model_fit_predict(seeded_db):
    if not xgboost_available():
        pytest.skip("xgboost is not installed in this environment")
    df = build_dataset(seeded_db, with_features=True)
    train = df[df["season"] == "2024-2025"]
    test = df[df["season"] == "2025-2026"]
    feats = feature_columns()
    pp = Preprocessor().fit(train, feats)
    Xtr, names = pp.transform(train)
    m = XGBoostResidualModel().fit(Xtr, train["residual"].to_numpy())
    Xte, _ = pp.transform(test)
    pred = m.predict(Xte)
    assert len(pred) == len(test)
    assert np.isfinite(pred).all()


def test_xgboost_model_feature_importance(seeded_db):
    if not xgboost_available():
        pytest.skip("xgboost is not installed in this environment")
    df = build_dataset(seeded_db, with_features=True)
    train = df[df["season"] == "2024-2025"]
    feats = feature_columns()
    pp = Preprocessor().fit(train, feats)
    Xtr, names = pp.transform(train)
    y = train["residual"].to_numpy()
    m = XGBoostResidualModel().fit(Xtr, y)
    fi = m.feature_importance(Xtr, y, names)
    assert set(fi.columns) == {"feature", "importance", "direction"}
    assert len(fi) == len(names)


def test_xgboost_default_hyperparameters_are_regularised():
    # same regularisation rationale as LightGBMResidualModel's own equivalent test -- exhaustive
    # gameweek walk-forward means some folds train on very little data.
    m = XGBoostResidualModel()
    assert m.reg_alpha > 0
    assert m.reg_lambda > 0
    assert 0 < m.subsample < 1
    assert 0 < m.colsample_bytree < 1


def test_xgboost_predict_before_fit_raises():
    m = XGBoostResidualModel()
    with pytest.raises(ResidualModelUnavailableError):
        m.predict(np.zeros((3, 2)))


def test_all_tree_arms_objective_is_configurable_and_defaults_to_l2(seeded_db):
    # class defaults stay L2 so nothing that constructs these unexpectedly changes; the
    # experiment orchestrator opts every tree arm into L1 explicitly (aligns loss w/ the MAE metric).
    assert GradientBoostingResidualModel().loss == "squared_error"
    assert XGBoostResidualModel().objective == "reg:squarederror"
    train = build_dataset(seeded_db, with_features=True)
    pp = Preprocessor().fit(train, feature_columns())
    Xtr, _ = pp.transform(train)
    y = train["residual"].to_numpy()
    if sklearn_available():
        g = GradientBoostingResidualModel(loss="absolute_error").fit(Xtr, y)
        assert g._model.loss == "absolute_error"
        assert g.predict(Xtr).shape == (len(train),)
    if xgboost_available():
        x = XGBoostResidualModel(objective="reg:absoluteerror").fit(Xtr, y)
        assert x._model.get_params()["objective"] == "reg:absoluteerror"
        assert x.predict(Xtr).shape == (len(train),)


def test_xgboost_raises_when_unavailable(monkeypatch):
    if not xgboost_available():
        pytest.skip("xgboost is not installed in this environment; the real ImportError path already covers this")
    import sys
    monkeypatch.setitem(sys.modules, "xgboost", None)
    with pytest.raises(ResidualModelUnavailableError):
        XGBoostResidualModel().fit(np.random.RandomState(0).randn(20, 3), np.random.RandomState(1).randn(20))


def test_xgboost_feature_importance_raises_when_sklearn_unavailable(seeded_db, monkeypatch):
    """feature_importance() reuses sklearn's permutation_importance -- a separate, real
    dependency from xgboost itself, so its own unavailable-path needs its own test (same
    reasoning as LightGBMResidualModel's equivalent test above)."""
    if not xgboost_available():
        pytest.skip("xgboost is not installed in this environment")
    df = build_dataset(seeded_db, with_features=True)
    train = df[df["season"] == "2024-2025"]
    feats = feature_columns()
    pp = Preprocessor().fit(train, feats)
    Xtr, names = pp.transform(train)
    y = train["residual"].to_numpy()
    m = XGBoostResidualModel().fit(Xtr, y)
    import sys
    monkeypatch.setitem(sys.modules, "sklearn.inspection", None)
    with pytest.raises(ResidualModelUnavailableError):
        m.feature_importance(Xtr, y, names)
