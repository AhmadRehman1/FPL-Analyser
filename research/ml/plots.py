"""Plots for the Phase-0 experiment (spec §28). matplotlib is not a dependency of this repo,
so every plotting function lazy-imports it and no-ops (returning a note) when it is absent --
the CSV artifacts from evaluate.py are always produced and are the primary record; plots are a
secondary view. No plot may obscure uncertainty (spec §28)."""

from __future__ import annotations


import numpy as np
import pandas as pd

from . import contract as C


def _matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def _save(fig, name: str) -> str | None:
    path = C.PLOTS_DIR / f"{name}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return str(path)


def plot_prediction_vs_actual(df: pd.DataFrame, pred: np.ndarray, model_name: str) -> str | None:
    plt = _matplotlib()
    if plt is None:
        return None
    actual = df[C.COL_ACTUAL].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pred, actual, alpha=0.25, s=8)
    lim = [min(pred.min(), actual.min()), max(pred.max(), actual.max())]
    ax.plot(lim, lim, "r--", linewidth=1, label="perfect")
    ax.set_xlabel(f"{model_name} predicted points")
    ax.set_ylabel("actual points")
    ax.set_title(f"{model_name}: prediction vs actual")
    ax.legend()
    return _save(fig, f"pred_vs_actual_{model_name}")


def plot_residual_distribution(df: pd.DataFrame, pred: np.ndarray, model_name: str) -> str | None:
    plt = _matplotlib()
    if plt is None:
        return None
    err = pred - df[C.COL_ACTUAL].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(err, bins=50)
    ax.axvline(0.0, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel("error (predicted - actual)")
    ax.set_ylabel("count")
    ax.set_title(f"{model_name}: residual distribution")
    return _save(fig, f"residual_dist_{model_name}")


def plot_mae_by_season(comparison: pd.DataFrame) -> str | None:
    plt = _matplotlib()
    if plt is None or comparison.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot = comparison.pivot(index="season", columns="model", values="mae")
    pivot.plot.bar(ax=ax)
    ax.set_ylabel("MAE")
    ax.set_title("MAE by season and model")
    ax.legend(title="model")
    return _save(fig, "mae_by_season")


def plot_models_comparison(comparison: pd.DataFrame, metric: str = "rmse") -> str | None:
    plt = _matplotlib()
    if plt is None or comparison.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot = comparison.pivot(index="season", columns="model", values=metric)
    pivot.plot(ax=ax, marker="o")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric}: Quant vs ML vs Ensemble")
    return _save(fig, f"models_{metric}")


def plot_calibration(calib: pd.DataFrame) -> str | None:
    plt = _matplotlib()
    if plt is None or calib.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 6))
    for model_name, sub in calib.groupby("model"):
        ax.plot(sub["mean_predicted"], sub["mean_actual"], marker="o", label=model_name)
    lim = [0, max(calib["mean_predicted"].max(), calib["mean_actual"].max())]
    ax.plot(lim, lim, "k--", linewidth=1, label="perfect")
    ax.set_xlabel("mean predicted")
    ax.set_ylabel("mean actual")
    ax.set_title("calibration")
    ax.legend()
    return _save(fig, "calibration")


def plot_feature_importance(fi: pd.DataFrame, model_name: str, top: int = 20) -> str | None:
    plt = _matplotlib()
    if plt is None or fi.empty:
        return None
    sub = fi.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.3 * len(sub))))
    ax.barh(sub["feature"], sub["importance"])
    ax.set_xlabel("importance")
    ax.set_title(f"{model_name}: feature importance")
    return _save(fig, f"feature_importance_{model_name}")
