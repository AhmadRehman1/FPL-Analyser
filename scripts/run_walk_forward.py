#!/usr/bin/env python3
"""Leakage-safe, parallel walk-forward parameter search for FPL data.

Example:
python scripts/run_walk_forward.py --trials 100000 --workers 0
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from fpl_quant.historical_data import load_player_gameweeks

TARGET = "total_points"
BASE_FEATURES = [
    "minutes", "xP", "expected_goals", "expected_assists",
    "influence", "creativity", "threat", "ict_index", "selected",
    "transfers_in", "transfers_out", "value",
]


def prepare_frame(frame: pd.DataFrame, window: int) -> tuple[pd.DataFrame, list[str]]:
    data = frame.copy()
    data = data.dropna(subset=["kickoff_time", TARGET, "element"])
    data = data.sort_values(["element", "kickoff_time"])
    available = [c for c in BASE_FEATURES if c in data.columns]
    for column in available:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0.0)
        data[f"{column}_roll{window}"] = (
            data.groupby("element", sort=False)[column]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )
    features = [f"{c}_roll{window}" for c in available]
    data = data.dropna(subset=features + [TARGET])
    data["period"] = data["kickoff_time"].dt.to_period("W").astype(str)
    return data, features


def fit_predict_ridge(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float) -> np.ndarray:
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale == 0] = 1.0
    x_train = (x_train - mean) / scale
    x_test = (x_test - mean) / scale
    design = np.c_[np.ones(len(x_train)), x_train]
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y_train)
    return np.c_[np.ones(len(x_test)), x_test] @ coefficients


def evaluate_trial(trial: tuple[int, float], raw: pd.DataFrame, min_train_periods: int) -> dict:
    window, alpha = trial
    data, features = prepare_frame(raw, window)
    periods = list(data["period"].drop_duplicates())
    errors = []
    for i in range(min_train_periods, len(periods)):
        train = data[data["period"].isin(periods[:i])]
        test = data[data["period"] == periods[i]]
        if len(train) < 100 or test.empty:
            continue
        prediction = fit_predict_ridge(
            train[features].to_numpy(float), train[TARGET].to_numpy(float),
            test[features].to_numpy(float), alpha,
        )
        errors.append(np.mean((prediction - test[TARGET].to_numpy(float)) ** 2))
    return {"window": window, "alpha": alpha, "folds": len(errors), "mse": float(np.mean(errors)) if errors else float("inf")}


def make_trials(n_trials: int, seed: int) -> list[tuple[int, float]]:
    rng = np.random.default_rng(seed)
    windows = np.array([2, 3, 4, 5, 6, 8, 10, 12])
    alphas = np.logspace(-4, 3, 250)
    grid = list(itertools.product(windows, alphas))
    if n_trials <= len(grid):
        choices = rng.choice(len(grid), size=n_trials, replace=False)
    else:
        choices = rng.choice(len(grid), size=n_trials, replace=True)
    return [grid[i] for i in choices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", default=["2023-24", "2024-25"])
    parser.add_argument("--data-dir", default="data/historical")
    parser.add_argument("--output-dir", default="data/outputs")
    parser.add_argument("--trials", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=0, help="0 uses all local CPU cores")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-train-periods", type=int, default=8)
    args = parser.parse_args()

    raw = load_player_gameweeks(args.data_dir, args.seasons)
    trials = make_trials(args.trials, args.seed)
    workers = args.workers or (os.cpu_count() or 1)
    evaluator = partial(evaluate_trial, raw=raw, min_train_periods=args.min_train_periods)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(evaluator, trials, chunksize=max(1, len(trials) // (workers * 20))))

    results = pd.DataFrame(rows).sort_values("mse", kind="stable")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results.to_parquet(output / "walk_forward_trials.parquet", index=False)
    results.to_csv(output / "walk_forward_trials.csv", index=False)
    best = results.iloc[0].to_dict()
    (output / "walk_forward_best.json").write_text(json.dumps(best, indent=2))
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
