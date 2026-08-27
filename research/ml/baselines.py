"""Baselines and metrics for the Phase-0 residual ML experiment.

Two baselines, deliberately not cherry-picked:

1. Existing Quant model -- ep_total as-is. This is the baseline to beat. NOT optimised during the
   experiment (LEAKAGE_PROTOCOL.md §6 / spec §7): its predictions are read asof-safe from the
   existing backtest and used unchanged.

2. Simple historical baseline -- predict future points from a prior rolling average of the
   player's own FPL points. Exists to answer "does the Quant model add value over the
   dumbest possible time-series forecast?" (spec §8). If Quant does not beat this, the entire
   ML question is moot.

Metrics (spec §7/§12): MAE, RMSE, median absolute error, mean signed error (bias), Pearson
correlation, and rank correlation. Reported overall AND sliced by position, price band,
minutes band, fixture difficulty, ownership, gameweek, and season -- a model that improves one
slice but loses badly in others is not declared successful (spec §12).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import contract as C


# ============================================================
# Metrics
# ============================================================

@dataclass(frozen=True)
class MetricSet:
    mae: float
    rmse: float
    medae: float
    bias: float
    correlation: float
    rank_correlation: float
    n: int


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    rho, _ = spearmanr(a, b)
    return float(rho) if np.isfinite(rho) else float("nan")


def compute_metrics(pred: np.ndarray, actual: np.ndarray) -> MetricSet:
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(actual)
    pred, actual = pred[mask], actual[mask]
    if len(actual) == 0:
        return MetricSet(float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), 0)
    err = pred - actual
    return MetricSet(
        mae=float(np.mean(np.abs(err))),
        rmse=float(np.sqrt(np.mean(err ** 2))),
        medae=float(np.median(np.abs(err))),
        bias=float(np.mean(err)),
        correlation=_safe_corr(pred, actual),
        rank_correlation=_safe_spearman(pred, actual),
        n=len(actual),
    )


def metrics_to_dict(m: MetricSet) -> dict:
    return {"mae": m.mae, "rmse": m.rmse, "medae": m.medae, "bias": m.bias,
            "correlation": m.correlation, "rank_correlation": m.rank_correlation, "n": m.n}


# ============================================================
# Prediction framings
# ============================================================

def quant_predictions(df: pd.DataFrame) -> np.ndarray:
    """Baseline 1: the existing Quant model, unchanged. Q(x) as recorded asof the deadline."""
    return df[C.COL_QUANT_PRED].to_numpy(dtype=float)


def historical_baseline_predictions(df: pd.DataFrame, window: int = 5) -> np.ndarray:
    """Baseline 2: predict each gameweek's points with the player's mean prior FPL points.

    Uses the leakage-safe rolling_points_{window} feature (mean of the player's event_points over
    the last `window` gameweeks, strictly before the target gw -- built by feature_engineering).
    Players with no prior history fall back to their position's mean rolling points, then to 0 --
    an explicit, disclosed fallback, not a silent drop of new players (spec §9: 'do not silently
    drop large portions of the dataset').
    """
    col = f"rolling_points_{window}"
    if col not in df.columns:
        raise ValueError(f"feature column {col!r} not present -- run feature_engineering.add_features first")
    fallback_pos = df.groupby(C.COL_POSITION)[col].transform("mean")
    pred = df[col].fillna(fallback_pos).fillna(0.0)
    # A historical baseline must never be negative -- FPL points can be, but a naive mean of
    # past points is a non-negative expectation for a typical player; clipping at 0 is the
    # disclosed simplification, matching the spec's "simple historical average" intent.
    return pred.clip(lower=0.0).to_numpy(dtype=float)


# ============================================================
# Slicing
# ============================================================

def _price_band(v: float) -> str:
    if pd.isna(v):
        return "unknown"
    if v < 5.0:
        return "<5.0"
    if v < 7.0:
        return "5.0-7.0"
    if v < 9.0:
        return "7.0-9.0"
    return "9.0+"


def _minutes_band(v: float) -> str:
    if pd.isna(v):
        return "unknown"
    if v < 0.3:
        return "low_start"
    if v < 0.7:
        return "mid_start"
    return "high_start"


def _ownership_band(v: float) -> str:
    if pd.isna(v):
        return "unknown"
    if v < 5.0:
        return "<5%"
    if v < 20.0:
        return "5-20%"
    return "20%+"


def _fd_band(v: float) -> str:
    if pd.isna(v):
        return "unknown"
    if v < -1.5:
        return "easy"
    if v < 0:
        return "medium"
    return "hard"


def slice_columns(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Build the categorical slicing columns used for sliced metrics. All derived from
    pre-prediction features (asof-safe), so slicing never leaks the outcome."""
    out: dict[str, pd.Series] = {
        C.COL_POSITION: df[C.COL_POSITION].fillna("unknown"),
        C.COL_SEASON: df[C.COL_SEASON],
        C.COL_GAMEWEEK: df[C.COL_GAMEWEEK].astype(str),
    }
    if "now_cost" in df.columns:
        out["price_band"] = df["now_cost"].apply(_price_band)
    if "p_start_final" in df.columns:
        out["minutes_band"] = df["p_start_final"].apply(_minutes_band)
    if "selected_by_percent" in df.columns:
        out["ownership_band"] = df["selected_by_percent"].apply(_ownership_band)
    if "fixture_difficulty" in df.columns:
        out["fixture_difficulty_band"] = df["fixture_difficulty"].apply(_fd_band)
    return out


def sliced_metrics(df: pd.DataFrame, pred: np.ndarray) -> dict[str, dict[str, MetricSet]]:
    """Metrics per slice for every slicing dimension. Returns {dimension: {slice_value: MetricSet}}."""
    actual = df[C.COL_ACTUAL].to_numpy(dtype=float)
    slices = slice_columns(df)
    out: dict[str, dict[str, MetricSet]] = {}
    for dim, series in slices.items():
        out[dim] = {}
        for key in series.dropna().unique():
            mask = (series == key).to_numpy()
            out[dim][str(key)] = compute_metrics(pred[mask], actual[mask])
    return out


# ============================================================
# Persistence
# ============================================================

def save_baseline_metrics(df: pd.DataFrame) -> dict:
    """Compute + persist the Quant baseline metrics (overall + sliced) and predictions.
    Writes baseline_metrics.json and baseline_predictions.parquet (via DuckDB, no pyarrow)."""
    import duckdb

    pred = quant_predictions(df)
    overall = compute_metrics(pred, df[C.COL_ACTUAL].to_numpy(dtype=float))
    sliced = sliced_metrics(df, pred)
    payload = {
        "model": "quant_baseline",
        "overall": metrics_to_dict(overall),
        "sliced": {dim: {k: metrics_to_dict(v) for k, v in slices.items()} for dim, slices in sliced.items()},
        "n_observations": int(len(df)),
    }
    C.BASELINE_METRICS_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    pred_df = df[C.IDENTIFIER_COLS + [C.COL_ACTUAL]].copy()
    pred_df["quant_prediction"] = pred
    con = duckdb.connect()
    try:
        con.register("_baseline_preds", pred_df)
        con.execute(f"COPY _baseline_preds TO '{C.BASELINE_PREDICTIONS_PARQUET}' (FORMAT PARQUET)")
    finally:
        con.close()
    return payload
