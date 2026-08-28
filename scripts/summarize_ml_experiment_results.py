"""Track F, Phase F-4 helper: read the real experiment output (experiment_manifest.json,
model_comparison.csv, improvement.csv, bootstrap_ci.json, compute_runtime.csv,
sliced_model_comparison.csv, ensemble.csv, season_points.csv) and print a single, structured
summary -- so filling in research/ml/REPORT.md's real numbers is a matter of transcribing this
output, not manually cross-referencing six separate files by hand.

This script does NOT write REPORT.md itself -- the report's prose (what the numbers *mean*,
whether ML found real signal, why or why not) is a real analytical judgment call each time,
not something to template-generate. It only assembles the numbers accurately.

Usage (from repo root, after `python -m research.ml.experiment` has produced real output):
    .venv/Scripts/python scripts/summarize_ml_experiment_results.py
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from research.ml import contract as C  # noqa: E402

GOVERNING_MODEL = "quant_lightgbm"  # R11: the ship/no-ship decision is governed by this arm alone


def _fmt(x, nd=3) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def main() -> None:
    missing = [p for p in [
        C.EXPERIMENT_MANIFEST_JSON, C.MODEL_COMPARISON_CSV, C.IMPROVEMENT_CSV,
        C.BOOTSTRAP_CI_JSON, C.COMPUTE_RUNTIME_CSV, C.SLICED_MODEL_COMPARISON_CSV,
        C.ENSEMBLE_CSV, C.SEASON_POINTS_CSV,
    ] if not Path(p).exists()]
    if missing:
        raise SystemExit(
            "Missing real experiment output: " + ", ".join(str(m) for m in missing) +
            "\nRun `python -m research.ml.experiment` for real first."
        )

    manifest = json.loads(C.EXPERIMENT_MANIFEST_JSON.read_text(encoding="utf-8"))
    comparison = pd.read_csv(C.MODEL_COMPARISON_CSV)
    improvement = pd.read_csv(C.IMPROVEMENT_CSV)
    bootstrap = json.loads(C.BOOTSTRAP_CI_JSON.read_text(encoding="utf-8"))
    runtime = pd.read_csv(C.COMPUTE_RUNTIME_CSV)
    sliced = pd.read_csv(C.SLICED_MODEL_COMPARISON_CSV)
    season_points = pd.read_csv(C.SEASON_POINTS_CSV)

    print("=" * 70)
    print("RUN METADATA (REPORT.md §10)")
    print("=" * 70)
    print(f"git_commit: {manifest['git_commit']}")
    print(f"run_timestamp_utc: {manifest['run_timestamp_utc']}")
    print(f"dataset_rows: {manifest['dataset_rows']}")
    print(f"dataset_seasons: {manifest['dataset_seasons']}")
    print(f"fold_mode: {manifest['fold_mode']}  n_walk_forward_folds: {manifest['n_walk_forward_folds']}")
    print(f"first_test_step: {manifest['first_test_step']}  last_test_step: {manifest['last_test_step']}")
    print(f"skip_log: {manifest['skip_log']}")
    print(f"sklearn_available: {manifest['sklearn_available']}  lightgbm_available: {manifest['lightgbm_available']}")
    print(f"models: {manifest['models']}")

    print()
    print("=" * 70)
    print("3.1 HEADLINE METRICS -- aggregate (mean across all folds) per model")
    print("=" * 70)
    agg = comparison.groupby("model")[["mae", "rmse", "bias", "correlation", "rank_correlation", "n"]].mean(numeric_only=True)
    print(agg.round(3).to_string())

    print()
    print("=" * 70)
    print("3.1b BOOTSTRAP CONFIDENCE INTERVALS (MAE improvement vs Quant, fold-resampled)")
    print("=" * 70)
    for model, r in bootstrap.items():
        marker = "  <-- GOVERNS R11's DECISION" if model == GOVERNING_MODEL else ""
        print(f"{model}: point={_fmt(r['point_estimate'])}  CI=[{_fmt(r['ci_low'])}, {_fmt(r['ci_high'])}]  "
              f"n={r['n']}  statistically_credible={r['statistically_credible_improvement']}{marker}")
    if GOVERNING_MODEL not in bootstrap:
        print(f"*** {GOVERNING_MODEL} not present in bootstrap_ci.json -- lightgbm_available={manifest['lightgbm_available']} ***")

    print()
    print("=" * 70)
    print("3.1c COMPUTE / RUNTIME")
    print("=" * 70)
    print(runtime.groupby("model")["fit_predict_seconds"].agg(["sum", "mean", "count"]).round(3).to_string())

    print()
    print("=" * 70)
    print("3.2 IMPROVEMENT vs QUANT BASELINE, per season")
    print("=" * 70)
    print(improvement.to_string(index=False))

    print()
    print("=" * 70)
    print("3.4 SEASON MANAGER POINTS")
    print("=" * 70)
    print(season_points.to_string(index=False))

    print()
    print("=" * 70)
    print(f"8. SLICING CHECK -- {GOVERNING_MODEL} vs quant, aggregated across all folds")
    print("=" * 70)
    if GOVERNING_MODEL in set(sliced["model"]):
        gov = sliced[sliced["model"] == GOVERNING_MODEL].groupby(["dimension", "slice"])["mae"].mean()
        base = sliced[sliced["model"] == "quant"].groupby(["dimension", "slice"])["mae"].mean()
        joined = pd.DataFrame({"quant_mae": base, f"{GOVERNING_MODEL}_mae": gov}).dropna()
        joined["improvement"] = joined["quant_mae"] - joined[f"{GOVERNING_MODEL}_mae"]
        joined["worse_than_quant"] = joined["improvement"] < 0
        print(joined.round(3).to_string())
        n_worse = int(joined["worse_than_quant"].sum())
        print(f"\n{n_worse} of {len(joined)} slices where {GOVERNING_MODEL} is WORSE than quant "
              f"(a real slicing regression, if any, must be named in REPORT.md §8 before any 'ship' verdict).")
    else:
        print(f"{GOVERNING_MODEL} not present in sliced_model_comparison.csv -- cannot run this check.")

    print()
    print("=" * 70)
    print("9. DECISION INPUT SUMMARY")
    print("=" * 70)
    if GOVERNING_MODEL in bootstrap:
        credible = bootstrap[GOVERNING_MODEL]["statistically_credible_improvement"]
        print(f"{GOVERNING_MODEL} statistically_credible_improvement (bootstrap CI excludes zero): {credible}")
        print("Per R11, 'ship' requires this AND the slicing check above showing no regression.")
        print("This script does not make the final call -- record the real decision in REPORT.md §9 yourself.")
    else:
        print(f"{GOVERNING_MODEL} unavailable this run -- R14 applies, no ship/no-ship call is possible; "
              f"document why LightGBM was unavailable in REPORT.md §9.1.")


if __name__ == "__main__":
    main()
