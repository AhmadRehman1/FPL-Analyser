"""Phase-0 experiment orchestrator (spec §23): the single script that runs the whole pipeline
offline, outside the conversational context. Run it and inspect the results.

    python -m research.ml.experiment
    python -m research.ml.experiment --seasons 2024-2025 2025-2026
    python -m research.ml.experiment --runs 50            # 24/7-style loop: as many
                                                          # simulations as possible, each
                                                          # with a different seed, rolling log

Steps (spec §23): load dataset, construct walk-forward splits, train models, generate
predictions, calculate metrics, simulate an FPL manager's season points, save results, save
experiment manifest.

Walk-forward: by default every historical gameweek with prior training data becomes an
out-of-sample test point (exhaustive expanding gameweek window) -- the maximum number of
simulations. Use --fold-mode season for a coarser one-fold-per-season view.

Exit behaviour: if scikit-learn is unavailable, the gradient-boosting model is skipped with a
clear message and every other result (Quant baseline, historical baseline, linear residual
model, ensemble over quant+linear, season points) still completes -- a partial run is not a
failed run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import baselines as B
from . import contract as C
from . import evaluate as ev
from . import plots
from . import season_sim as sim
from . import walk_forward as wf
from .dataset_builder import build_dataset
from .leakage_checks import assert_feature_matrix_invariants
from .residual_model import (
    GradientBoostingResidualModel,
    LightGBMResidualModel,
    LinearResidualModel,
    Preprocessor,
    ResidualModelUnavailableError,
    lightgbm_available,
    sklearn_available,
)
from .feature_engineering import feature_columns


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(C.REPO_ROOT),
            capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _fit_and_predict_linear(train_df: pd.DataFrame, test_df: pd.DataFrame, feat_cols: list[str], seed: int):
    pp = Preprocessor().fit(train_df, feat_cols)
    Xtr, names = pp.transform(train_df)
    Xte, _ = pp.transform(test_df)
    ytr = train_df[C.COL_RESIDUAL].to_numpy(dtype=float)
    model = LinearResidualModel(random_state=seed).fit(Xtr, ytr)
    resid_train = model.predict(Xtr)
    resid_test = model.predict(Xte)
    return resid_train, resid_test, model, pp, names


def _fit_and_predict_gbm(train_df: pd.DataFrame, test_df: pd.DataFrame, feat_cols: list[str], seed: int):
    pp = Preprocessor().fit(train_df, feat_cols)
    Xtr, names = pp.transform(train_df)
    Xte, _ = pp.transform(test_df)
    ytr = train_df[C.COL_RESIDUAL].to_numpy(dtype=float)
    model = GradientBoostingResidualModel(random_state=seed).fit(Xtr, ytr)
    resid_train = model.predict(Xtr)
    resid_test = model.predict(Xte)
    return resid_train, resid_test, model, pp, Xte, names


def _fit_and_predict_lightgbm(train_df: pd.DataFrame, test_df: pd.DataFrame, feat_cols: list[str], seed: int):
    pp = Preprocessor().fit(train_df, feat_cols)
    Xtr, names = pp.transform(train_df)
    Xte, _ = pp.transform(test_df)
    ytr = train_df[C.COL_RESIDUAL].to_numpy(dtype=float)
    model = LightGBMResidualModel(random_state=seed).fit(Xtr, ytr)
    resid_train = model.predict(Xtr)
    resid_test = model.predict(Xte)
    return resid_train, resid_test, model, pp, names


def run_experiment(
    seasons: tuple[str, ...] | None = None,
    con=None,
    random_seed: int = 42,
    fold_mode: str = "gameweek",
) -> dict:
    """Runs the full Phase-0 pipeline. `con` may be an already-open DuckDB connection (used by
    tests with a synthetic in-memory DB); if None, opens the repo's default DB read-only."""
    owns_con = con is None
    if owns_con:
        con = C.connect(read_only=True)

    try:
        df = build_dataset(con, seasons=seasons, with_features=True)
    finally:
        if owns_con:
            con.close()

    feat_cols = [c for c in feature_columns() if c in df.columns]
    assert_feature_matrix_invariants(feat_cols)

    B.save_baseline_metrics(df)

    folds = wf.default_folds(df, fold_mode=fold_mode)
    comparison_rows: list[dict] = []
    residual_rows: list[pd.DataFrame] = []
    disagreement_frames: list[pd.DataFrame] = []
    calibration_frames: list[pd.DataFrame] = []
    linear_importances: list[pd.DataFrame] = []
    gbm_importances: list[pd.DataFrame] = []
    lightgbm_importances: list[pd.DataFrame] = []
    ensemble_rows: list[dict] = []
    runtime_rows: list[dict] = []
    sklearn_ok = sklearn_available()
    lightgbm_ok = lightgbm_available()

    # per-gameweek prediction frames, collected so we can simulate a full season manager
    per_gw_frames: list[pd.DataFrame] = []
    # pooled out-of-sample predictions per model, for bootstrap CIs computed once over the
    # whole run rather than per fold (see evaluate.bootstrap_ci_rows). Actuals are pooled per
    # model too (not once globally) because quant_gbm/quant_lightgbm can be absent from a fold
    # (dependency unavailable, or a fit failure) -- pooling a shared actual list would misalign
    # predictions and actuals for any model that skipped a fold.
    pooled_predictions: dict[str, list[np.ndarray]] = {}
    pooled_actual_by_model: dict[str, list[np.ndarray]] = {}

    for fold in folds:
        train_df, test_df = fold.train_df, fold.test_df
        quant_test = test_df[C.COL_QUANT_PRED].to_numpy(dtype=float)
        historical_test = B.historical_baseline_predictions(test_df)

        predictions: dict[str, np.ndarray] = {
            "quant": quant_test,
            "historical_baseline": historical_test,
        }

        # ---- linear residual model ----
        t0 = time.perf_counter()
        resid_train_lin, resid_test_lin, lin_model, _pp, feat_names = _fit_and_predict_linear(train_df, test_df, feat_cols, random_seed)
        runtime_rows.append({"fold": fold.name, "model": "quant_linear", "fit_predict_seconds": time.perf_counter() - t0})
        ml_pred_lin = quant_test + resid_test_lin
        predictions["quant_linear"] = ml_pred_lin
        linear_importances.append(lin_model.feature_importance(feat_names))

        # ---- LightGBM residual model (Track F primary nonlinear challenger, optional) ----
        ml_pred_lightgbm = None
        resid_train_lightgbm = None
        if lightgbm_ok:
            try:
                t0 = time.perf_counter()
                resid_train_lightgbm, resid_test_lightgbm, lightgbm_model, _pp_lgbm, lgbm_names = _fit_and_predict_lightgbm(train_df, test_df, feat_cols, random_seed)
                runtime_rows.append({"fold": fold.name, "model": "quant_lightgbm", "fit_predict_seconds": time.perf_counter() - t0})
                ml_pred_lightgbm = quant_test + resid_test_lightgbm
                predictions["quant_lightgbm"] = ml_pred_lightgbm
                lightgbm_importances.append(lightgbm_model.feature_importance(lgbm_names))
            except ResidualModelUnavailableError as exc:
                print(f"[experiment] skipping LightGBM for fold {fold.name}: {exc}", file=sys.stderr)

        # ---- gradient boosting residual model (sklearn quant_gbm -- bonus/informational only,
        # per R16: never governs the ship/no-ship decision, LightGBM above does) ----
        ml_pred_gbm = None
        if sklearn_ok:
            try:
                t0 = time.perf_counter()
                resid_train_gbm, resid_test_gbm, gbm_model, pp_gbm, Xte_gbm, gbm_names = _fit_and_predict_gbm(train_df, test_df, feat_cols, random_seed)
                runtime_rows.append({"fold": fold.name, "model": "quant_gbm", "fit_predict_seconds": time.perf_counter() - t0})
                ml_pred_gbm = quant_test + resid_test_gbm
                predictions["quant_gbm"] = ml_pred_gbm
                gbm_importances.append(
                    gbm_model.feature_importance(Xte_gbm, test_df[C.COL_ACTUAL].to_numpy(dtype=float) - quant_test, gbm_names)
                )
            except ResidualModelUnavailableError as exc:
                print(f"[experiment] skipping gradient boosting for fold {fold.name}: {exc}", file=sys.stderr)

        # ---- ensemble (quant + best available ML model), weight learned on train only.
        # Priority LightGBM > quant_gbm > linear: LightGBM is the primary nonlinear challenger
        # (R11), so when it's available the ensemble and season-manager sim are built from it,
        # not from the informational-only quant_gbm arm. ----
        if ml_pred_lightgbm is not None:
            ml_for_ensemble = ml_pred_lightgbm
            resid_train_for_ensemble = resid_train_lightgbm
        elif ml_pred_gbm is not None:
            ml_for_ensemble = ml_pred_gbm
            resid_train_for_ensemble = resid_train_gbm
        else:
            ml_for_ensemble = ml_pred_lin
            resid_train_for_ensemble = resid_train_lin
        ml_train_for_ensemble = train_df[C.COL_QUANT_PRED].to_numpy(dtype=float) + resid_train_for_ensemble
        ens = ev.evaluate_ensemble(train_df, test_df, ml_train_for_ensemble, ml_for_ensemble)
        predictions["ensemble"] = ens["test_pred"]
        ensemble_rows.append({"fold": fold.name, "season": fold.test_season, **{k: v for k, v in ens.items() if k != "test_pred"}})

        comparison_rows.extend(ev.model_comparison_rows(fold.name, fold.test_season, test_df, predictions))
        residual_rows.append(ev.residual_analysis(test_df, quant_test).assign(fold=fold.name, season=fold.test_season))
        disagreement_frames.append(
            ev.high_disagreement_cases(test_df, quant_test, ml_for_ensemble).assign(fold=fold.name)
        )
        fold_actual = test_df[C.COL_ACTUAL].to_numpy(dtype=float)
        for model_name, pred in predictions.items():
            calibration_frames.append(ev.calibration(test_df, pred, model_name).assign(fold=fold.name))
            pooled_predictions.setdefault(model_name, []).append(pred)
            pooled_actual_by_model.setdefault(model_name, []).append(fold_actual)

        # collect this gameweek's rows + the ensemble prediction for the season manager sim
        gw_frame = test_df.copy()
        gw_frame[C.COL_ML_PRED] = ml_for_ensemble
        per_gw_frames.append(gw_frame)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(C.MODEL_COMPARISON_CSV, index=False)

    improvement_df = ev.improvement_rows(comparison_df)
    improvement_df.to_csv(C.IMPROVEMENT_CSV, index=False)

    residual_df = pd.concat(residual_rows, ignore_index=True) if residual_rows else pd.DataFrame()
    residual_df.to_csv(C.RESIDUAL_ANALYSIS_CSV, index=False)

    disagreement_df = pd.concat(disagreement_frames, ignore_index=True) if disagreement_frames else pd.DataFrame()
    disagreement_df.to_csv(C.HIGH_DISAGREEMENT_CSV, index=False)

    calibration_df = pd.concat(calibration_frames, ignore_index=True) if calibration_frames else pd.DataFrame()
    calibration_df.to_csv(C.CALIBRATION_CSV, index=False)

    fi_df = pd.concat(linear_importances, ignore_index=True) if linear_importances else pd.DataFrame()
    fi_df.to_csv(C.FEATURE_IMPORTANCE_CSV, index=False)

    stability_df = ev.stability_table(linear_importances)
    stability_df.to_csv(C.STABILITY_CSV, index=False)

    fi_lightgbm_df = pd.concat(lightgbm_importances, ignore_index=True) if lightgbm_importances else pd.DataFrame()
    fi_lightgbm_df.to_csv(C.FEATURE_IMPORTANCE_LIGHTGBM_CSV, index=False)

    lightgbm_stability_df = ev.stability_table(lightgbm_importances)
    lightgbm_stability_df.to_csv(C.STABILITY_LIGHTGBM_CSV, index=False)

    pd.DataFrame(ensemble_rows).to_csv(C.ENSEMBLE_CSV, index=False)

    pd.DataFrame(runtime_rows).to_csv(C.RUNTIME_CSV, index=False)

    # ---- bootstrap confidence intervals (R10/R11): computed once over pooled out-of-sample
    # predictions per model, not per fold (see evaluate.bootstrap_ci_rows's docstring) ----
    bootstrap_rows: list[dict] = []
    for model_name, preds_list in pooled_predictions.items():
        pooled_pred = np.concatenate(preds_list)
        pooled_act = np.concatenate(pooled_actual_by_model[model_name])
        for r in ev.bootstrap_ci(pooled_pred, pooled_act, random_state=random_seed):
            bootstrap_rows.append({"model": model_name, **r})
    bootstrap_df = pd.DataFrame(bootstrap_rows)
    bootstrap_df.to_csv(C.BOOTSTRAP_CI_CSV, index=False)

    # ---- season manager simulation: how many real points would each signal score? ----
    if per_gw_frames:
        season_pool = pd.concat(per_gw_frames, ignore_index=True)
        signal_cols = {C.COL_QUANT_PRED: C.COL_QUANT_PRED, C.COL_ML_PRED: C.COL_ML_PRED}
        if "historical_baseline" in comparison_df["model"].unique() if not comparison_df.empty else False:
            pass
        season_pts = sim.season_points_table(season_pool, signal_cols)
        season_pts.to_csv(C.SEASON_POINTS_CSV, index=False)
    else:
        season_pts = pd.DataFrame()

    # ---- plots (best-effort; no-op if matplotlib is unavailable) ----
    if not comparison_df.empty:
        plots.plot_mae_by_season(comparison_df)
        plots.plot_models_comparison(comparison_df, metric="rmse")
    if not calibration_df.empty:
        plots.plot_calibration(calibration_df)
    if not fi_df.empty:
        plots.plot_feature_importance(fi_df, "quant_linear")
    last_fold = folds[-1]
    plots.plot_prediction_vs_actual(last_fold.test_df, last_fold.test_df[C.COL_QUANT_PRED].to_numpy(dtype=float), "quant")
    plots.plot_residual_distribution(last_fold.test_df, last_fold.test_df[C.COL_QUANT_PRED].to_numpy(dtype=float), "quant")

    # headline points: total season points for the ML-augmented manager vs the Quant manager
    ml_points = float(season_pts.loc[season_pts["signal"] == C.COL_ML_PRED, "total_points"].sum()) if not season_pts.empty else 0.0
    quant_points = float(season_pts.loc[season_pts["signal"] == C.COL_QUANT_PRED, "total_points"].sum()) if not season_pts.empty else 0.0

    manifest = {
        "git_commit": _git_commit(),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_rows": int(len(df)),
        "dataset_seasons": df.attrs.get("seasons", []),
        "skip_log": df.attrs.get("skip_log", []),
        "fold_mode": fold_mode,
        "n_walk_forward_folds": len(folds),
        "first_test_step": {"season": folds[0].test_season, "gameweek": folds[0].test_gameweek},
        "last_test_step": {"season": folds[-1].test_season, "gameweek": folds[-1].test_gameweek},
        "walk_forward_folds": [
            {"name": f.name, "train_seasons": list(f.train_seasons), "test_season": f.test_season,
             "test_gameweek": f.test_gameweek, "n_train": len(f.train_df), "n_test": len(f.test_df)}
            for f in folds
        ],
        "feature_columns": feat_cols,
        "sklearn_available": sklearn_ok,
        "lightgbm_available": lightgbm_ok,
        "models": (
            ["quant", "historical_baseline", "quant_linear"]
            + (["quant_lightgbm"] if lightgbm_ok else [])
            + (["quant_gbm"] if sklearn_ok else [])
            + ["ensemble"]
        ),
        # the model whose predictions feed the ensemble and season-manager sim (R11: LightGBM
        # is the primary nonlinear challenger; quant_gbm is bonus/informational only, per R16)
        "primary_ml_model": "quant_lightgbm" if lightgbm_ok else ("quant_gbm" if sklearn_ok else "quant_linear"),
        "random_seed": random_seed,
        "season_points": {"quant_manager": quant_points, "ml_manager": ml_points,
                          "ml_beats_quant": ml_points > quant_points},
    }
    C.EXPERIMENT_MANIFEST_JSON.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    return {
        "manifest": manifest,
        "comparison": comparison_df,
        "improvement": improvement_df,
        "season_points": season_pts,
    }


def _append_run_log(summary: dict) -> None:
    """Append one row per loop run to the rolling experiment_runs.csv."""
    row = pd.DataFrame([summary])
    if C.RUN_LOG_CSV.exists():
        existing = pd.read_csv(C.RUN_LOG_CSV)
        pd.concat([existing, row], ignore_index=True).to_csv(C.RUN_LOG_CSV, index=False)
    else:
        row.to_csv(C.RUN_LOG_CSV, index=False)


def run_loop(runs: int, seasons: tuple[str, ...] | None, fold_mode: str, base_seed: int = 42) -> dict:
    """Run the experiment `runs` times, each with a distinct seed. Each run writes its artifacts
    to a timestamped subdir under results/runs/ and a summary row is appended to the rolling
    experiment_runs.csv. Returns the best run by ML manager season points."""
    best = None
    for i in range(runs):
        seed = base_seed + i
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = C.RUNS_DIR / f"run_{i:04d}_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)
        # redirect artifact paths to this run's subdir
        orig = {a: getattr(C, a) for a in [
            "MODEL_COMPARISON_CSV", "IMPROVEMENT_CSV", "RESIDUAL_ANALYSIS_CSV",
            "HIGH_DISAGREEMENT_CSV", "CALIBRATION_CSV", "FEATURE_IMPORTANCE_CSV",
            "FEATURE_IMPORTANCE_LIGHTGBM_CSV", "STABILITY_CSV", "STABILITY_LIGHTGBM_CSV",
            "ENSEMBLE_CSV", "BOOTSTRAP_CI_CSV", "RUNTIME_CSV", "EXPERIMENT_MANIFEST_JSON",
            "SEASON_POINTS_CSV",
        ]}
        for a in orig:
            setattr(C, a, run_dir / orig[a].name)
        try:
            result = run_experiment(seasons=seasons, con=None, random_seed=seed, fold_mode=fold_mode)
            m = result["manifest"]
            summary = {
                "run_index": i, "seed": seed, "timestamp_utc": ts,
                "fold_mode": fold_mode, "n_folds": m["n_walk_forward_folds"],
                "dataset_rows": m["dataset_rows"],
                "quant_manager_points": m["season_points"]["quant_manager"],
                "ml_manager_points": m["season_points"]["ml_manager"],
                "ml_beats_quant": m["season_points"]["ml_beats_quant"],
                "run_dir": str(run_dir),
            }
            _append_run_log(summary)
            if best is None or summary["ml_manager_points"] > best["ml_manager_points"]:
                best = summary
            print(f"[loop] run {i + 1}/{runs} seed={seed} -> ML {summary['ml_manager_points']:.1f} "
                  f"vs Quant {summary['quant_manager_points']:.1f} pts "
                  f"({'ML wins' if summary['ml_beats_quant'] else 'quant holds'})")
        finally:
            for a in orig:
                setattr(C, a, orig[a])
    if best is None:
        return {"runs": 0}
    best["runs"] = runs
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase-0 FPL Quant residual ML experiment.")
    parser.add_argument("--seasons", nargs="*", default=None, help="restrict to these seasons (default: all with backtest steps)")
    parser.add_argument("--fold-mode", choices=["gameweek", "season"], default="gameweek",
                        help="walk-forward granularity (default: gameweek = exhaustive, one test gw at a time)")
    parser.add_argument("--runs", type=int, default=1,
                        help="loop this many times with different seeds (24/7-style: as many simulations as possible)")
    parser.add_argument("--base-seed", type=int, default=42)
    args = parser.parse_args()
    seasons = tuple(args.seasons) if args.seasons else None

    if args.runs > 1:
        best = run_loop(args.runs, seasons, args.fold_mode, args.base_seed)
        print(json.dumps(best, indent=2, default=str))
        return

    result = run_experiment(seasons=seasons, random_seed=args.base_seed, fold_mode=args.fold_mode)
    print(json.dumps(result["manifest"], indent=2, default=str))
    if not result["comparison"].empty:
        print("\n=== model comparison ===")
        print(result["comparison"].to_string(index=False))
    if not result["season_points"].empty:
        print("\n=== season manager points ===")
        print(result["season_points"].to_string(index=False))


if __name__ == "__main__":
    main()
