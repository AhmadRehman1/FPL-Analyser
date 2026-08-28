"""Append one real experiment run's headline numbers to the tracked, persistent quality-trend
ledger (`research/ml/results_history/weekly_quality_history.csv`) -- the only way to actually
answer "is the model improving over time?" as more real gameweeks accrue week over week,
since a single run's `results/` directory is gitignored and each Sunday's GitHub Actions build
artifact expires after 90 days in isolation from every other week's.

This is a narrow, deliberate exception to research/ml/README.md's "does not commit anything
back to the repo" stance (also `.github/workflows/ml_experiment.yml`'s own header): it persists
only real numbers already computed by a real `python -m research.ml.experiment` run --
`quant_lightgbm`'s and `quant_xgboost`'s bootstrap CI (point estimate, bounds, whether the
interval excludes zero) and the season-manager points -- never REPORT.md's prose or its §9
decision, which stays exactly as much a human judgement call as it always was
(scripts/summarize_ml_experiment_results.py's own docstring makes the same distinction).

Usage (from repo root, after `python -m research.ml.experiment` has produced real output):
    python scripts/append_ml_run_to_history.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from research.ml import contract as C  # noqa: E402

GOVERNING_MODEL = "quant_lightgbm"  # R11
CONFIRMATION_MODEL = "quant_xgboost"  # R13

HISTORY_COLUMNS = [
    "run_timestamp_utc", "git_commit", "dataset_rows", "fold_mode",
    "quant_lightgbm_point_estimate", "quant_lightgbm_ci_low", "quant_lightgbm_ci_high",
    "quant_lightgbm_credible",
    "quant_xgboost_point_estimate", "quant_xgboost_ci_low", "quant_xgboost_ci_high",
    "quant_xgboost_credible",
    "quant_manager_points", "ml_manager_points", "ml_beats_quant",
]


def _model_ci_fields(bootstrap: dict, model_name: str) -> dict:
    r = bootstrap.get(model_name)
    if r is None:
        return {
            f"{model_name}_point_estimate": float("nan"), f"{model_name}_ci_low": float("nan"),
            f"{model_name}_ci_high": float("nan"), f"{model_name}_credible": None,
        }
    return {
        f"{model_name}_point_estimate": r["point_estimate"], f"{model_name}_ci_low": r["ci_low"],
        f"{model_name}_ci_high": r["ci_high"], f"{model_name}_credible": r["statistically_credible_improvement"],
    }


def build_history_row() -> dict:
    """Reads the just-completed real run's manifest + bootstrap CI and assembles one row. Raises
    a clear error (not a crash) if no real run has happened yet."""
    if not Path(C.EXPERIMENT_MANIFEST_JSON).exists() or not Path(C.BOOTSTRAP_CI_JSON).exists():
        raise SystemExit(
            f"Missing real experiment output: {C.EXPERIMENT_MANIFEST_JSON}, {C.BOOTSTRAP_CI_JSON}\n"
            "Run `python -m research.ml.experiment` for real first."
        )
    manifest = json.loads(C.EXPERIMENT_MANIFEST_JSON.read_text(encoding="utf-8"))
    bootstrap = json.loads(C.BOOTSTRAP_CI_JSON.read_text(encoding="utf-8"))
    sp = manifest["season_points"]
    row = {
        "run_timestamp_utc": manifest["run_timestamp_utc"],
        "git_commit": manifest["git_commit"],
        "dataset_rows": manifest["dataset_rows"],
        "fold_mode": manifest["fold_mode"],
        "quant_manager_points": sp["quant_manager"],
        "ml_manager_points": sp["ml_manager"],
        "ml_beats_quant": sp["ml_beats_quant"],
    }
    row.update(_model_ci_fields(bootstrap, GOVERNING_MODEL))
    row.update(_model_ci_fields(bootstrap, CONFIRMATION_MODEL))
    return row


def append_row(row: dict) -> pd.DataFrame:
    """Appends `row` to the tracked history CSV (creating it with a header if it doesn't exist
    yet) and returns the full, updated history as a DataFrame."""
    C.RESULTS_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    new_row_df = pd.DataFrame([row], columns=HISTORY_COLUMNS)
    if Path(C.RESULTS_HISTORY_CSV).exists():
        existing = pd.read_csv(C.RESULTS_HISTORY_CSV)
        history = pd.concat([existing, new_row_df], ignore_index=True)
    else:
        history = new_row_df
    history.to_csv(C.RESULTS_HISTORY_CSV, index=False)
    return history


def print_trend_summary(history: pd.DataFrame) -> None:
    """Prints a short, human-readable "is it getting better" delta vs the previous logged run,
    if one exists -- so a human doesn't have to open the CSV just to get the headline signal."""
    print(f"Logged run {len(history)} of the tracked quality history ({C.RESULTS_HISTORY_CSV}).")
    if len(history) < 2:
        print("No prior run to compare against yet -- this is the first entry.")
        return
    prev, curr = history.iloc[-2], history.iloc[-1]
    for model in (GOVERNING_MODEL, CONFIRMATION_MODEL):
        prev_point, curr_point = prev.get(f"{model}_point_estimate"), curr.get(f"{model}_point_estimate")
        prev_credible, curr_credible = prev.get(f"{model}_credible"), curr.get(f"{model}_credible")
        if pd.isna(prev_point) or pd.isna(curr_point):
            print(f"{model}: unavailable in a prior or current run -- cannot compare.")
            continue
        delta = curr_point - prev_point
        direction = "improved" if delta > 0 else ("worsened" if delta < 0 else "unchanged")
        print(
            f"{model}: MAE-improvement point estimate {direction} by {abs(delta):.4f} "
            f"({prev_point:.4f} -> {curr_point:.4f}); credible {prev_credible} -> {curr_credible}"
        )
    print(
        f"Season points: quant {prev['quant_manager_points']:.1f} -> {curr['quant_manager_points']:.1f}, "
        f"ml {prev['ml_manager_points']:.1f} -> {curr['ml_manager_points']:.1f}"
    )


def main() -> None:
    row = build_history_row()
    history = append_row(row)
    print_trend_summary(history)


if __name__ == "__main__":
    main()
