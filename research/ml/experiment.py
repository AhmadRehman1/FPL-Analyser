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
from contextlib import contextmanager
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
    XGBoostResidualModel,
    lightgbm_available,
    sklearn_available,
    xgboost_available,
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
    return resid_train, resid_test, model, pp, Xte, names


def _fit_and_predict_xgboost(train_df: pd.DataFrame, test_df: pd.DataFrame, feat_cols: list[str], seed: int):
    pp = Preprocessor().fit(train_df, feat_cols)
    Xtr, names = pp.transform(train_df)
    Xte, _ = pp.transform(test_df)
    ytr = train_df[C.COL_RESIDUAL].to_numpy(dtype=float)
    model = XGBoostResidualModel(random_state=seed).fit(Xtr, ytr)
    resid_train = model.predict(Xtr)
    resid_test = model.predict(Xte)
    return resid_train, resid_test, model, pp, Xte, names


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
    ensemble_rows: list[dict] = []
    lightgbm_importances: list[pd.DataFrame] = []
    xgboost_importances: list[pd.DataFrame] = []  # R13: independent-implementation confirmation arm
    runtime_rows: list[dict] = []  # R10: per-model fit+predict wall-clock time, per fold
    sliced_rows: list[dict] = []  # R11: per-model, per-slice metrics, every fold
    sklearn_ok = sklearn_available()
    lightgbm_ok = lightgbm_available()
    xgboost_ok = xgboost_available()

    # per-gameweek prediction frames, collected so we can simulate a full season manager
    per_gw_frames: list[pd.DataFrame] = []

    for fold in folds:
        train_df, test_df = fold.train_df, fold.test_df
        quant_test = test_df[C.COL_QUANT_PRED].to_numpy(dtype=float)
        historical_test = B.historical_baseline_predictions(test_df)

        predictions: dict[str, np.ndarray] = {
            "quant": quant_test,
            "historical_baseline": historical_test,
        }

        # ---- linear residual model ----
        _t0 = time.perf_counter()
        resid_train_lin, resid_test_lin, lin_model, _pp, feat_names = _fit_and_predict_linear(train_df, test_df, feat_cols, random_seed)
        runtime_rows.append({"fold": fold.name, "model": "quant_linear", "fit_predict_seconds": time.perf_counter() - _t0})
        ml_pred_lin = quant_test + resid_test_lin
        predictions["quant_linear"] = ml_pred_lin
        linear_importances.append(lin_model.feature_importance(feat_names))

        # ---- gradient boosting residual model (optional, bonus/informational -- R16: does
        # not factor into the R11 ship/no-ship decision, which is governed by LightGBM below) ----
        ml_pred_gbm = None
        if sklearn_ok:
            try:
                _t0 = time.perf_counter()
                resid_train_gbm, resid_test_gbm, gbm_model, pp_gbm, Xte_gbm, gbm_names = _fit_and_predict_gbm(train_df, test_df, feat_cols, random_seed)
                runtime_rows.append({"fold": fold.name, "model": "quant_gbm", "fit_predict_seconds": time.perf_counter() - _t0})
                ml_pred_gbm = quant_test + resid_test_gbm
                predictions["quant_gbm"] = ml_pred_gbm
                gbm_importances.append(
                    gbm_model.feature_importance(Xte_gbm, test_df[C.COL_ACTUAL].to_numpy(dtype=float) - quant_test, gbm_names)
                )
            except ResidualModelUnavailableError as exc:
                print(f"[experiment] skipping gradient boosting for fold {fold.name}: {exc}", file=sys.stderr)

        # ---- LightGBM residual model (optional; R8/R11 -- the PRIMARY nonlinear challenger,
        # the sole arm the ship/no-ship decision is governed by) ----
        ml_pred_lightgbm = None
        resid_train_lightgbm = None
        if lightgbm_ok:
            try:
                _t0 = time.perf_counter()
                resid_train_lightgbm, resid_test_lightgbm, lightgbm_model, pp_lgb, Xte_lgb, lgb_names = _fit_and_predict_lightgbm(train_df, test_df, feat_cols, random_seed)
                runtime_rows.append({"fold": fold.name, "model": "quant_lightgbm", "fit_predict_seconds": time.perf_counter() - _t0})
                ml_pred_lightgbm = quant_test + resid_test_lightgbm
                predictions["quant_lightgbm"] = ml_pred_lightgbm
                lightgbm_importances.append(
                    lightgbm_model.feature_importance(Xte_lgb, test_df[C.COL_ACTUAL].to_numpy(dtype=float) - quant_test, lgb_names)
                )
            except ResidualModelUnavailableError as exc:
                # R14: a real LightGBM failure must not block the rest of the pipeline --
                # quant/historical/linear/quant_gbm results for this fold still complete.
                print(f"[experiment] skipping LightGBM for fold {fold.name}: {exc}", file=sys.stderr)

        # ---- XGBoost residual model (optional; R13 -- an independent-implementation
        # confirmation of the LightGBM result, informational only, does NOT govern R11) ----
        ml_pred_xgboost = None
        if xgboost_ok:
            try:
                _t0 = time.perf_counter()
                resid_train_xgb, resid_test_xgb, xgb_model, pp_xgb, Xte_xgb, xgb_names = _fit_and_predict_xgboost(train_df, test_df, feat_cols, random_seed)
                runtime_rows.append({"fold": fold.name, "model": "quant_xgboost", "fit_predict_seconds": time.perf_counter() - _t0})
                ml_pred_xgboost = quant_test + resid_test_xgb
                predictions["quant_xgboost"] = ml_pred_xgboost
                xgboost_importances.append(
                    xgb_model.feature_importance(Xte_xgb, test_df[C.COL_ACTUAL].to_numpy(dtype=float) - quant_test, xgb_names)
                )
            except ResidualModelUnavailableError as exc:
                # R14: same treatment as the other optional arms -- a real XGBoost failure must
                # not block the rest of the pipeline.
                print(f"[experiment] skipping XGBoost for fold {fold.name}: {exc}", file=sys.stderr)

        # ---- ensemble (quant + best available ML model), weight learned on train only.
        # Priority: LightGBM (primary nonlinear challenger) > XGBoost (R13 confirmation arm,
        # also a strong nonlinear challenger in its own right) > quant_gbm > linear -- this is
        # the "best available ML signal" convenience column for season_points.csv, separate
        # from R11's ship/no-ship criterion, which reads the quant_lightgbm row directly. ----
        if ml_pred_lightgbm is not None:
            ml_for_ensemble = ml_pred_lightgbm
            resid_train_for_ensemble = resid_train_lightgbm
        elif ml_pred_xgboost is not None:
            ml_for_ensemble = ml_pred_xgboost
            resid_train_for_ensemble = resid_train_xgb
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
        sliced_rows.extend(ev.sliced_comparison_rows(fold.name, fold.test_season, test_df, predictions))
        residual_rows.append(ev.residual_analysis(test_df, quant_test).assign(fold=fold.name, season=fold.test_season))
        disagreement_frames.append(
            ev.high_disagreement_cases(test_df, quant_test, ml_for_ensemble).assign(fold=fold.name)
        )
        for model_name, pred in predictions.items():
            calibration_frames.append(ev.calibration(test_df, pred, model_name).assign(fold=fold.name))

        # collect this gameweek's rows + the ensemble prediction for the season manager sim
        gw_frame = test_df.copy()
        gw_frame[C.COL_ML_PRED] = ml_for_ensemble
        per_gw_frames.append(gw_frame)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(C.MODEL_COMPARISON_CSV, index=False)

    improvement_df = ev.improvement_rows(comparison_df)
    improvement_df.to_csv(C.IMPROVEMENT_CSV, index=False)

    pd.DataFrame(runtime_rows).to_csv(C.COMPUTE_RUNTIME_CSV, index=False)
    pd.DataFrame(sliced_rows).to_csv(C.SLICED_MODEL_COMPARISON_CSV, index=False)

    # ---- bootstrap confidence intervals (R10/R11): per-fold improvement over quant, resampled
    # at fold granularity (see evaluate.bootstrap_ci's own docstring for why). quant_lightgbm's
    # entry here is what R11's ship/no-ship decision is actually governed by; the others are
    # reported for completeness, not decision-relevant on their own (R16 for quant_gbm). ----
    bootstrap_ci_by_model: dict[str, dict] = {}
    if not comparison_df.empty:
        for candidate_model in ("quant_linear", "quant_gbm", "quant_lightgbm", "quant_xgboost"):
            if candidate_model in comparison_df["model"].unique():
                bootstrap_ci_by_model[candidate_model] = ev.bootstrap_ci_for_model_improvement(
                    comparison_df, candidate_model, metric="mae", n_resamples=1000,
                    confidence=0.95, random_state=random_seed,
                )
    C.BOOTSTRAP_CI_JSON.write_text(json.dumps(bootstrap_ci_by_model, indent=2, default=str), encoding="utf-8")

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

    # quant_lightgbm's own feature importance/stability -- it was being computed per fold
    # (lightgbm_importances above) but never persisted anywhere, unlike quant_linear's. R11's
    # ship/no-ship decision is governed by this arm, so its feature importance matters more
    # than the linear model's and should be inspectable, not silently discarded.
    fi_lightgbm_df = pd.concat(lightgbm_importances, ignore_index=True) if lightgbm_importances else pd.DataFrame()
    fi_lightgbm_df.to_csv(C.FEATURE_IMPORTANCE_LIGHTGBM_CSV, index=False)

    stability_lightgbm_df = ev.stability_table(lightgbm_importances)
    stability_lightgbm_df.to_csv(C.STABILITY_LIGHTGBM_CSV, index=False)

    # R13: quant_xgboost's own feature importance/stability -- same treatment as quant_lightgbm's,
    # persisted separately so the independent-implementation confirmation arm's numbers are
    # inspectable on their own footing.
    fi_xgboost_df = pd.concat(xgboost_importances, ignore_index=True) if xgboost_importances else pd.DataFrame()
    fi_xgboost_df.to_csv(C.FEATURE_IMPORTANCE_XGBOOST_CSV, index=False)

    stability_xgboost_df = ev.stability_table(xgboost_importances)
    stability_xgboost_df.to_csv(C.STABILITY_XGBOOST_CSV, index=False)

    pd.DataFrame(ensemble_rows).to_csv(C.ENSEMBLE_CSV, index=False)

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
        "xgboost_available": xgboost_ok,
        # quant_lightgbm is the primary nonlinear challenger and the sole arm R11's ship/no-ship
        # decision is governed by (docs/plans/2026-08_retrospective_validation_and_ml_decision_plan.md);
        # quant_gbm and quant_xgboost (R13's independent-implementation confirmation arm) are
        # kept, but bonus/informational only -- all listed here, none implies another decides
        # anything on its own.
        "models": (
            ["quant", "historical_baseline", "quant_linear"]
            + (["quant_gbm"] if sklearn_ok else [])
            + (["quant_lightgbm"] if lightgbm_ok else [])
            + (["quant_xgboost"] if xgboost_ok else [])
            + ["ensemble"]
        ),
        "random_seed": random_seed,
        "season_points": {"quant_manager": quant_points, "ml_manager": ml_points,
                          "ml_beats_quant": ml_points > quant_points},
        # R10/R11: confidence-interval-based, not point-estimate -- statistically_credible_
        # improvement is True only when the entire 95% CI of per-fold MAE improvement sits
        # above zero. R11's ship/no-ship decision reads bootstrap_ci["quant_lightgbm"] here.
        "bootstrap_ci": bootstrap_ci_by_model,
        "compute_runtime_seconds_by_model": (
            pd.DataFrame(runtime_rows).groupby("model")["fit_predict_seconds"].agg(["sum", "mean", "count"]).to_dict("index")
            if runtime_rows else {}
        ),
    }
    C.EXPERIMENT_MANIFEST_JSON.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    return {
        "manifest": manifest,
        "comparison": comparison_df,
        "improvement": improvement_df,
        "season_points": season_pts,
        "bootstrap_ci": bootstrap_ci_by_model,
    }


def _append_run_log(summary: dict) -> None:
    """Append one row per loop run to the rolling experiment_runs.csv."""
    row = pd.DataFrame([summary])
    if C.RUN_LOG_CSV.exists():
        existing = pd.read_csv(C.RUN_LOG_CSV)
        pd.concat([existing, row], ignore_index=True).to_csv(C.RUN_LOG_CSV, index=False)
    else:
        row.to_csv(C.RUN_LOG_CSV, index=False)


@contextmanager
def redirect_results_to(run_dir):
    """Temporarily redirect every per-run result-artifact path (contract.PER_RUN_RESULT_ATTRS)
    into `run_dir`, restoring the originals on exit -- so a loop/24-7 caller's successive
    run_experiment() calls each get their own subdirectory instead of silently overwriting each
    other's detailed output in place. Single source of truth shared by run_loop() below and
    run_continuous.run_forever(), so they can no longer drift out of sync with each other or
    with a newly added result artifact the way they previously did (see PER_RUN_RESULT_ATTRS's
    own comment for the real history)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    orig = {a: getattr(C, a) for a in C.PER_RUN_RESULT_ATTRS}
    for a in orig:
        setattr(C, a, run_dir / orig[a].name)
    try:
        yield
    finally:
        for a in orig:
            setattr(C, a, orig[a])


def run_loop(runs: int, seasons: tuple[str, ...] | None, fold_mode: str, base_seed: int = 42) -> dict:
    """Run the experiment `runs` times, each with a distinct seed. Each run writes its artifacts
    to a timestamped subdir under results/runs/ and a summary row is appended to the rolling
    experiment_runs.csv. Returns the best run by ML manager season points."""
    best = None
    for i in range(runs):
        seed = base_seed + i
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = C.RUNS_DIR / f"run_{i:04d}_{ts}"
        with redirect_results_to(run_dir):
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
