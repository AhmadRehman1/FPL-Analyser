"""Evaluation: model comparison, improvement, residual analysis, disagreement, calibration,
feature importance, stability, and the ensemble experiment (spec §11-19).

This module turns a trained residual model's predictions into the CSV artifacts the spec
requires. It never trains anything itself (that lives in residual_model + experiment) and it
never reads the test set's label except to compute metrics -- the test set is genuinely
out-of-sample within each walk-forward fold.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import baselines as B
from . import contract as C


# ============================================================
# Model comparison (spec §11)
# ============================================================

def model_comparison_rows(
    fold_name: str, test_season: str, df_test: pd.DataFrame,
    predictions: dict[str, np.ndarray],
) -> list[dict]:
    """One row per (season, model) with MAE/RMSE/bias/correlation. `predictions` maps a model
    name (quant / quant_linear / quant_gbm / historical) to its corrected point predictions
    for the test fold."""
    actual = df_test[C.COL_ACTUAL].to_numpy(dtype=float)
    rows = []
    for model_name, pred in predictions.items():
        m = B.compute_metrics(pred, actual)
        rows.append({
            "fold": fold_name, "season": test_season, "model": model_name,
            "mae": m.mae, "rmse": m.rmse, "bias": m.bias, "correlation": m.correlation,
            "rank_correlation": m.rank_correlation, "n": m.n,
        })
    return rows


# ============================================================
# Improvement (spec §12)
# ============================================================

def improvement_rows(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Improvement = baseline_error - ml_error, per season per model. Positive = ML helps.

    Multiple folds per season (e.g. gameweek walk-forward) are aggregated to a season-level
    mean error before the comparison, so improvement is reported per season per model."""
    if comparison_df.empty:
        return pd.DataFrame()
    metrics = [c for c in ("mae", "rmse", "bias", "pearson", "spearman") if c in comparison_df.columns]
    agg = comparison_df.groupby(["season", "model"])[metrics].mean().reset_index()
    base = agg[agg["model"] == "quant"].set_index("season")
    out = []
    for _, row in agg.iterrows():
        if row["model"] == "quant":
            continue
        b = base.loc[row["season"]] if row["season"] in base.index else None
        if b is None:
            continue
        for metric in ("mae", "rmse"):
            improvement = b[metric] - row[metric]
            pct = (improvement / b[metric] * 100.0) if b[metric] != 0 and not math.isnan(b[metric]) else float("nan")
            out.append({
                "season": row["season"], "model": row["model"], "metric": metric,
                "quant_error": b[metric], "ml_error": row[metric],
                "improvement": improvement, "improvement_pct": pct,
            })
    return pd.DataFrame(out)


# ============================================================
# Residual analysis (spec §13)
# ============================================================

def residual_analysis(df: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    """Where is the Quant model systematically wrong? Mean/median/std of the residual by slice."""
    df = df.copy()
    df["_err"] = pred - df[C.COL_ACTUAL].to_numpy(dtype=float)
    slices = B.slice_columns(df)
    rows = []
    for dim, series in slices.items():
        for key in series.dropna().unique():
            sub = df[series == key]
            err = sub["_err"]
            rows.append({
                "dimension": dim, "slice": str(key), "n": len(sub),
                "mean_error": float(err.mean()) if len(err) else float("nan"),
                "median_error": float(err.median()) if len(err) else float("nan"),
                "std_error": float(err.std()) if len(err) > 1 else float("nan"),
            })
    return pd.DataFrame(rows)


# ============================================================
# Disagreement analysis (spec §14)
# ============================================================

def high_disagreement_cases(df: pd.DataFrame, quant_pred: np.ndarray, ml_pred: np.ndarray, top_n: int = 200) -> pd.DataFrame:
    """Cases where ML and Quant disagree most, sorted by absolute disagreement, with each
    side's error so we can see who was right."""
    out = df[C.IDENTIFIER_COLS + [C.COL_ACTUAL]].copy()
    out["quant_prediction"] = quant_pred
    out["ml_prediction"] = ml_pred
    out["difference"] = ml_pred - quant_pred
    out["quant_error"] = quant_pred - out[C.COL_ACTUAL]
    out["ml_error"] = ml_pred - out[C.COL_ACTUAL]
    out = out.reindex(out["difference"].abs().sort_values(ascending=False).index)
    return out.head(top_n).reset_index(drop=True)


# ============================================================
# Calibration (spec §17)
# ============================================================

CALIBRATION_BUCKETS = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 1e9)]


def calibration(df: pd.DataFrame, pred: np.ndarray, model_name: str) -> pd.DataFrame:
    """Does a prediction of 6 actually average 6 realized points? Bucket by predicted value."""
    actual = df[C.COL_ACTUAL].to_numpy(dtype=float)
    out = []
    for lo, hi in CALIBRATION_BUCKETS:
        mask = (pred >= lo) & (pred < hi)
        if mask.sum() == 0:
            continue
        out.append({
            "model": model_name,
            "bucket": f"{lo:g}-{hi:g}" if hi < 1e9 else f"{lo:g}+",
            "mean_predicted": float(pred[mask].mean()),
            "mean_actual": float(actual[mask].mean()),
            "n": int(mask.sum()),
        })
    return pd.DataFrame(out)


# ============================================================
# Ensemble (spec §16)
# ============================================================

def ensemble_predictions(quant: np.ndarray, ml: np.ndarray, w: float) -> np.ndarray:
    """Final = w*Quant + (1-w)*ML. Weight is NOT fit on the test set (spec §16)."""
    return w * quant + (1.0 - w) * ml


def evaluate_ensemble(train_df: pd.DataFrame, test_df: pd.DataFrame, ml_train: np.ndarray, ml_test: np.ndarray) -> dict:
    """Test fixed 75/25, 50/50, and a learned weight (fit on TRAINING residuals only).
    Returns the best weight's test-set predictions, so the caller can include it in the
    comparison table."""
    quant_train = train_df[C.COL_QUANT_PRED].to_numpy(dtype=float)
    quant_test = test_df[C.COL_QUANT_PRED].to_numpy(dtype=float)
    actual_train = train_df[C.COL_ACTUAL].to_numpy(dtype=float)
    actual_test = test_df[C.COL_ACTUAL].to_numpy(dtype=float)

    def train_mae_for(w: float) -> float:
        p = ensemble_predictions(quant_train, ml_train, w)
        return B.compute_metrics(p, actual_train).mae

    candidates = [0.0, 0.25, 0.5, 0.75, 1.0]
    grid = [{"w": w, "train_mae": train_mae_for(w)} for w in candidates]
    best = min(grid, key=lambda d: d["train_mae"])
    pred = ensemble_predictions(quant_test, ml_test, best["w"])
    test_mae = B.compute_metrics(pred, actual_test).mae
    return {"best_w": best["w"], "train_mae": best["train_mae"], "test_mae": test_mae, "grid": grid, "test_pred": pred}


# ============================================================
# Bootstrap confidence intervals (Track F, R10/R11 --
# docs/plans/2026-08_retrospective_validation_and_ml_decision_plan.md): the ship/no-ship
# decision must be governed by a confidence interval, not a point estimate, so "quant_lightgbm's
# MAE improved by 0.03" alone is never enough to ship it.
# ============================================================

def bootstrap_ci(
    values: np.ndarray, *, n_resamples: int = 1000, confidence: float = 0.95,
    statistic=np.mean, random_state: int | None = None,
) -> dict:
    """Percentile bootstrap CI for `statistic` applied to `values`. `values` is expected to be
    one row per walk-forward FOLD, not per observation -- players within the same gameweek
    share the same underlying match outcomes (goals, cards, bonus) and are not independent, so
    bootstrapping at the observation level would understate real uncertainty. Resampling at
    fold granularity treats each gameweek's result as one (much closer to) independent draw,
    consistent with this module's own existing fold-level "improvement" reporting
    (improvement_rows above)."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return {
            "point_estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
            "n_resamples": n_resamples, "confidence": confidence, "n": 0,
        }
    rng = np.random.RandomState(random_state)
    point = float(statistic(values))
    resampled = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        resampled[i] = statistic(values[idx])
    alpha = 1.0 - confidence
    ci_low = float(np.percentile(resampled, 100 * (alpha / 2)))
    ci_high = float(np.percentile(resampled, 100 * (1 - alpha / 2)))
    return {
        "point_estimate": point, "ci_low": ci_low, "ci_high": ci_high,
        "n_resamples": n_resamples, "confidence": confidence, "n": n,
    }


def bootstrap_ci_for_model_improvement(
    comparison_df: pd.DataFrame, model_name: str, *, metric: str = "mae",
    n_resamples: int = 1000, confidence: float = 0.95, random_state: int | None = None,
) -> dict:
    """Bootstrap CI on `model_name`'s per-fold improvement over the quant baseline
    (quant_error - ml_error per fold; positive = model_name helps), resampled at fold
    granularity -- see bootstrap_ci's own docstring for why. `statistically_credible_improvement`
    is True only when the ENTIRE interval sits above zero (ci_low > 0) -- a positive point
    estimate alone is exactly the "point estimate, not a confidence interval" failure mode R11
    exists to rule out."""
    quant = comparison_df[comparison_df["model"] == "quant"].set_index("fold")[metric]
    model = comparison_df[comparison_df["model"] == model_name].set_index("fold")[metric]
    common_folds = quant.index.intersection(model.index)
    improvements = (quant.loc[common_folds] - model.loc[common_folds]).to_numpy(dtype=float)
    result = bootstrap_ci(improvements, n_resamples=n_resamples, confidence=confidence, random_state=random_state)
    result["model"] = model_name
    result["metric"] = metric
    result["statistically_credible_improvement"] = result["n"] > 0 and result["ci_low"] > 0
    return result


# ============================================================
# Feature stability (spec §20)
# ============================================================

def stability_table(importances_by_fold: list[pd.DataFrame]) -> pd.DataFrame:
    """A feature important in one season but irrelevant in every other is suspicious. Compute
    per-feature mean/coeff-var of importance across folds, plus how often it ranks top-K."""
    if not importances_by_fold:
        return pd.DataFrame()
    joined = None
    for i, fi in enumerate(importances_by_fold):
        col = pd.DataFrame({"feature": fi["feature"], f"imp_{i}": fi["importance"].to_numpy()})
        joined = col if joined is None else joined.merge(col, on="feature", how="outer")
    vals = joined.drop(columns=["feature"]).to_numpy(dtype=float)
    means = np.nanmean(vals, axis=1)
    stds = np.nanstd(vals, axis=1)
    cv = np.where(means > 1e-12, stds / means, float("nan"))
    out = pd.DataFrame({"feature": joined["feature"], "mean_importance": means, "std_importance": stds, "cv": cv})
    return out.sort_values("mean_importance", ascending=False).reset_index(drop=True)
