"""Evaluate-module tests: comparison rows, improvement rows, residual analysis, disagreement,
calibration, ensemble weight selection, and stability table."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.ml.dataset_builder import build_dataset
from research.ml.evaluate import (
    bootstrap_ci,
    bootstrap_ci_for_model_improvement,
    calibration,
    ensemble_predictions,
    evaluate_ensemble,
    high_disagreement_cases,
    improvement_rows,
    model_comparison_rows,
    residual_analysis,
    sliced_comparison_rows,
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


def test_sliced_comparison_rows_covers_every_model_and_every_dimension(seeded_db):
    """Track F, R11: the per-slice check that gates the ship/no-ship decision needs
    quant_lightgbm's own numbers per slice, not just Quant's -- confirm this function actually
    produces rows for every model passed in, across every slicing dimension
    baselines.slice_columns() defines (position/price/minutes/fixture-difficulty/ownership)."""
    df = build_dataset(seeded_db, with_features=True)
    folds = default_folds(df)
    f = folds[0]
    ml_pred = _fit_and_predict(f.train_df, f.test_df)
    rows = sliced_comparison_rows(f.name, f.test_season, f.test_df, {
        "quant": f.test_df["quant_prediction"].to_numpy(),
        "quant_linear": ml_pred,
    })
    assert rows  # non-empty -- a real dataset always has at least one row per slice
    models_present = {r["model"] for r in rows}
    assert models_present == {"quant", "quant_linear"}
    dimensions_present = {r["dimension"] for r in rows}
    assert len(dimensions_present) > 1  # more than one slicing dimension actually got covered
    # every row carries real, finite metrics -- not a placeholder/empty slice
    assert all(r["n"] > 0 for r in rows)


def test_sliced_comparison_rows_uses_the_same_slice_definitions_as_the_quant_baseline(seeded_db):
    """The slice boundaries (price bands, minutes bands, ...) must be identical to whatever
    save_baseline_metrics()'s own Quant-only sliced_metrics() call already uses -- otherwise a
    'quant_lightgbm improved in the 5.0-7.0 price band' claim and the existing baseline report's
    own price-band numbers would silently be sliced two different ways."""
    df = build_dataset(seeded_db, with_features=True)
    folds = default_folds(df)
    f = folds[0]
    from research.ml.baselines import sliced_metrics
    quant_pred = f.test_df["quant_prediction"].to_numpy()
    direct = sliced_metrics(f.test_df, quant_pred)
    direct_slice_keys = {(dim, str(k)) for dim, by_slice in direct.items() for k in by_slice}
    rows = sliced_comparison_rows(f.name, f.test_season, f.test_df, {"quant": quant_pred})
    row_slice_keys = {(r["dimension"], str(r["slice"])) for r in rows}
    assert row_slice_keys == direct_slice_keys


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


# ============================================================
# bootstrap_ci / bootstrap_ci_for_model_improvement (Track F, R10/R11:
# docs/plans/2026-08_retrospective_validation_and_ml_decision_plan.md) -- the ship/no-ship
# decision must be confidence-interval-based, not a point estimate, so these need real coverage
# against synthetic cases with a KNOWN answer, not just "it runs without crashing."
# ============================================================

def test_bootstrap_ci_zero_variance_gives_a_point_interval():
    """Every value identical -> every bootstrap resample is the same multiset -> the interval
    collapses to a single point. A known, exact answer, not just a plausible-looking range."""
    values = np.full(50, 7.0)
    result = bootstrap_ci(values, n_resamples=200, random_state=0)
    assert result["point_estimate"] == pytest.approx(7.0)
    assert result["ci_low"] == pytest.approx(7.0)
    assert result["ci_high"] == pytest.approx(7.0)
    assert result["n"] == 50


def test_bootstrap_ci_all_positive_values_excludes_zero():
    """A clearly, consistently positive signal (small noise around a large positive mean) must
    produce a CI entirely above zero -- the exact condition R11's ship criterion checks for."""
    rng = np.random.RandomState(1)
    values = 5.0 + rng.normal(0, 0.2, size=200)  # mean ~5, tiny noise -- unambiguous
    result = bootstrap_ci(values, n_resamples=1000, random_state=2)
    assert result["ci_low"] > 0
    assert result["point_estimate"] == pytest.approx(5.0, abs=0.2)


def test_bootstrap_ci_noisy_signal_straddling_zero_includes_zero():
    """A signal with no real effect (mean ~0, real noise) must NOT produce a CI excluding
    zero -- this is the case R11 exists to correctly say "no-ship" on, distinct from the
    all-positive case above."""
    rng = np.random.RandomState(3)
    values = rng.normal(0, 5.0, size=30)  # mean ~0, large noise relative to n
    result = bootstrap_ci(values, n_resamples=1000, random_state=4)
    assert result["ci_low"] <= 0 <= result["ci_high"]


def test_bootstrap_ci_reproducible_with_same_random_state():
    values = np.random.RandomState(9).normal(2, 1, size=40)
    r1 = bootstrap_ci(values, n_resamples=500, random_state=42)
    r2 = bootstrap_ci(values, n_resamples=500, random_state=42)
    assert r1 == r2


def test_bootstrap_ci_empty_values_returns_nan_not_a_crash():
    result = bootstrap_ci(np.array([]), n_resamples=100)
    assert result["n"] == 0
    assert np.isnan(result["point_estimate"])
    assert np.isnan(result["ci_low"])
    assert np.isnan(result["ci_high"])


def test_bootstrap_ci_for_model_improvement_flags_a_consistent_winner_as_credible():
    """A model that beats quant by a real, consistent margin on every fold -> statistically
    credible improvement (entire CI above zero)."""
    folds = [f"gw{i}" for i in range(20)]
    rows = []
    rng = np.random.RandomState(5)
    for f in folds:
        quant_mae = 2.0 + rng.normal(0, 0.05)
        rows.append({"fold": f, "model": "quant", "mae": quant_mae})
        rows.append({"fold": f, "model": "quant_lightgbm", "mae": quant_mae - 0.3})  # consistently 0.3 better
    df = pd.DataFrame(rows)
    result = bootstrap_ci_for_model_improvement(df, "quant_lightgbm", metric="mae", n_resamples=1000, random_state=6)
    assert result["statistically_credible_improvement"] is True
    assert result["point_estimate"] == pytest.approx(0.3, abs=0.05)
    assert result["model"] == "quant_lightgbm"
    assert result["metric"] == "mae"


def test_bootstrap_ci_for_model_improvement_does_not_flag_a_noisy_near_tie():
    """A model whose per-fold improvement is essentially noise around zero (sometimes better,
    sometimes worse, no real edge) must NOT be flagged credible -- this is the exact "a positive
    point estimate isn't enough" case R11 exists to guard against."""
    folds = [f"gw{i}" for i in range(15)]
    rows = []
    rng = np.random.RandomState(7)
    for f in folds:
        quant_mae = 2.0
        noisy_improvement = rng.normal(0, 0.5)  # no real signal, large fold-to-fold noise
        rows.append({"fold": f, "model": "quant", "mae": quant_mae})
        rows.append({"fold": f, "model": "quant_gbm", "mae": quant_mae - noisy_improvement})
    df = pd.DataFrame(rows)
    result = bootstrap_ci_for_model_improvement(df, "quant_gbm", metric="mae", n_resamples=1000, random_state=8)
    assert result["statistically_credible_improvement"] is False


def test_bootstrap_ci_for_model_improvement_only_uses_folds_present_in_both():
    """A fold where the model ran but quant's row is missing (or vice versa) must be excluded
    from the comparison, not silently treated as zero improvement."""
    df = pd.DataFrame([
        {"fold": "gw1", "model": "quant", "mae": 2.0},
        {"fold": "gw1", "model": "quant_lightgbm", "mae": 1.5},
        {"fold": "gw2", "model": "quant", "mae": 2.0},
        # gw2 has no quant_lightgbm row -- must not be treated as a 0-improvement fold
        {"fold": "gw3", "model": "quant", "mae": 2.0},
        {"fold": "gw3", "model": "quant_lightgbm", "mae": 1.5},
    ])
    result = bootstrap_ci_for_model_improvement(df, "quant_lightgbm", metric="mae", n_resamples=100, random_state=1)
    assert result["n"] == 2  # only gw1 and gw3, not gw2
