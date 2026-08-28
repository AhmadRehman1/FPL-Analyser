"""Optional, local-only convenience: turn the Phase-0 ML experiment's real numeric results
(research/ml/results/*) into a plain-English draft paragraph using a local LLM served by
Ollama (https://ollama.com) -- so reading `research/ml/REPORT.md`'s six-plus result files by
hand isn't the only way to get a sense of what a real run found.

This is explicitly a DRAFT for a human to read and correct, not an analytical judgement:
research/ml/README.md and scripts/summarize_ml_experiment_results.py are both deliberate that
"what the numbers mean" is a real judgement call each time, not something to template-generate
-- an LLM narrating the same numbers doesn't change that. The output file this script writes
says so at the top, and this script never edits REPORT.md itself.

Requires nothing beyond this repo's existing `requests` dependency (requirements.txt) PLUS a
separately-installed, separately-running local Ollama server -- see the "Running locally with
Ollama" section of research/ml/README.md. If Ollama isn't reachable, this fails with one clear
actionable message; it never falls back to a remote API (nothing in this project should send
FPL data or model results to a third-party service without the user explicitly choosing to).

Usage (from repo root, after `python -m research.ml.experiment` has produced real output):
    python scripts/narrate_ml_results.py
    python scripts/narrate_ml_results.py --model qwen2.5:14b --host http://localhost:11434
    python scripts/narrate_ml_results.py --print-only   # skip writing results/narrative_draft.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Same two-path bootstrap scripts/summarize_ml_experiment_results.py uses and documents: the
# top-level `research` package (for `research.ml`) and `src` (for research.ml's own
# `from fpl_quant import ...`) both need to be importable before anything below runs.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from research.ml import contract as C  # noqa: E402
from research.ml.ollama_client import DEFAULT_HOST, DEFAULT_MODEL, OllamaUnavailableError, chat  # noqa: E402

GOVERNING_MODEL = "quant_lightgbm"  # R11
CONFIRMATION_MODEL = "quant_xgboost"  # R13

SYSTEM_PROMPT = """You are narrating the results of a machine-learning experiment for a \
technical reader who already understands the setup. You will be given real, precomputed \
numbers -- headline error metrics, a bootstrap confidence interval, season-simulation points, \
and a slice-regression count.

Rules, followed strictly:
- Use ONLY the numbers given to you. Never invent, estimate, or round a number you were not \
given.
- Write 150-250 words, plain English, neutral tone -- describe what the numbers show, not what \
you think should be done about it.
- The experiment's own rule: a model only counts as a statistically credible improvement when \
its ENTIRE bootstrap confidence interval sits above zero. If told the interval crosses zero, \
say plainly that the result is not statistically credible, even if the point estimate is \
positive -- do not soften this.
- Do not issue a ship/no-ship recommendation or any other decision. End by stating plainly that \
this is a draft summary for a human to check against the real numbers, not the project's actual \
decision.
"""


def _fmt(x, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def build_numeric_summary() -> str:
    """Reads the same real result files scripts/summarize_ml_experiment_results.py reads and
    assembles a compact, numbers-only text block -- deliberately small (not raw CSVs pasted in
    full) since it is the entire factual basis the model is allowed to narrate from."""
    required_files = (
        C.EXPERIMENT_MANIFEST_JSON, C.MODEL_COMPARISON_CSV, C.BOOTSTRAP_CI_JSON,
        C.SLICED_MODEL_COMPARISON_CSV, C.SEASON_POINTS_CSV,
    )
    missing = [p for p in required_files if not Path(p).exists()]
    if missing:
        raise SystemExit(
            "Missing real experiment output: " + ", ".join(str(m) for m in missing) +
            "\nRun `python -m research.ml.experiment` for real first."
        )

    manifest = json.loads(C.EXPERIMENT_MANIFEST_JSON.read_text(encoding="utf-8"))
    comparison = pd.read_csv(C.MODEL_COMPARISON_CSV)
    bootstrap = json.loads(C.BOOTSTRAP_CI_JSON.read_text(encoding="utf-8"))
    sliced = pd.read_csv(C.SLICED_MODEL_COMPARISON_CSV)
    season_points = pd.read_csv(C.SEASON_POINTS_CSV)

    lines: list[str] = []
    lines.append(f"Dataset: {manifest['dataset_rows']} rows across seasons {manifest['dataset_seasons']}.")
    lines.append(f"Walk-forward folds: {manifest['n_walk_forward_folds']} ({manifest['fold_mode']} mode).")

    lines.append("\nHeadline MAE/RMSE per model, averaged across all folds:")
    agg = comparison.groupby("model")[["mae", "rmse", "bias"]].mean(numeric_only=True)
    for model, row in agg.iterrows():
        lines.append(f"- {model}: MAE={_fmt(row['mae'])}, RMSE={_fmt(row['rmse'])}, bias={_fmt(row['bias'])}")

    lines.append("\nBootstrap 95% confidence intervals on MAE improvement vs the Quant baseline (fold-resampled):")
    for model in (GOVERNING_MODEL, CONFIRMATION_MODEL, "quant_gbm", "quant_linear"):
        if model in bootstrap:
            r = bootstrap[model]
            role = "governs the ship/no-ship decision" if model == GOVERNING_MODEL else (
                "independent-implementation confirmation, informational only" if model == CONFIRMATION_MODEL
                else "informational only"
            )
            lines.append(
                f"- {model} ({role}): point estimate={_fmt(r['point_estimate'])}, "
                f"CI=[{_fmt(r['ci_low'])}, {_fmt(r['ci_high'])}], "
                f"statistically_credible_improvement={r['statistically_credible_improvement']}"
            )

    if GOVERNING_MODEL in set(sliced["model"]):
        gov = sliced[sliced["model"] == GOVERNING_MODEL].groupby(["dimension", "slice"])["mae"].mean()
        base = sliced[sliced["model"] == "quant"].groupby(["dimension", "slice"])["mae"].mean()
        joined = pd.DataFrame({"quant_mae": base, "model_mae": gov}).dropna()
        n_worse = int((joined["model_mae"] > joined["quant_mae"]).sum())
        lines.append(
            f"\nSlicing check: {GOVERNING_MODEL} is worse than the Quant baseline in "
            f"{n_worse} of {len(joined)} slices (position/price/minutes/fixture/ownership/gw/season)."
        )

    if not season_points.empty:
        lines.append("\nSeason manager simulation -- total points across the walk-forward run:")
        for _, row in season_points.iterrows():
            lines.append(f"- {row['signal']}: {row['total_points']:.1f} points")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model tag to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Ollama server URL (default: {DEFAULT_HOST})")
    parser.add_argument("--print-only", action="store_true", help="print to stdout only, skip writing results/narrative_draft.md")
    args = parser.parse_args()

    numeric_summary = build_numeric_summary()
    print("=" * 70)
    print("NUMERIC SUMMARY FED TO THE MODEL (verbatim, nothing else)")
    print("=" * 70)
    print(numeric_summary)
    print()

    try:
        response = chat(SYSTEM_PROMPT, numeric_summary, model=args.model, host=args.host)
    except OllamaUnavailableError as exc:
        raise SystemExit(str(exc)) from exc

    print("=" * 70)
    print(f"DRAFT NARRATIVE (model={response.model}, via Ollama at {args.host})")
    print("=" * 70)
    print(response.text)

    if not args.print_only:
        out_path = C.RESULTS_DIR / "narrative_draft.md"
        out_path.write_text(
            "# ML experiment -- draft narrative (AI-generated, for human review only)\n\n"
            f"Generated locally by Ollama, model `{response.model}`. This is a draft summary of "
            "the real numbers in this results/ directory, not the project's actual decision -- "
            "see REPORT.md §9 for that. Check every number below against the source CSVs/JSON "
            "before trusting or sharing it.\n\n"
            "---\n\n"
            f"{response.text}\n",
            encoding="utf-8",
        )
        print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
