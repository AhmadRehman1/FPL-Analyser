# research/ml — Phase-0 ML Residual Experiment

A standalone, drop-in research engine that asks: **can a machine-learning residual model
predict the errors of the existing FPL Quant model (`ep_total`) more accurately than the Quant
model alone, using only information available at prediction time?**

This does **not** replace the Quant model and does **not** modify live production
recommendations. The Quant model is the baseline to beat. A negative result is a successful
research result.

## Running

```bash
# from the repo root, with the repo's venv active
python -m research.ml.experiment            # all ingested seasons, exhaustive gameweek walk-forward
python -m research.ml.experiment --seasons 2024-2025 2025-2026
python -m research.ml.experiment --runs 50          # loop 50x with fresh seeds, rolling log
python -m research.ml.run_continuous               # 24/7: loop forever, never stop
```

The experiment builds the player×gameweek dataset from the existing DuckDB (populated by
`scripts/run_ingestion.py` + `scripts/run_backtest.py`), runs chronological walk-forward
folds, fits a linear residual model, a LightGBM residual model (the primary nonlinear
challenger, if `lightgbm` is installed -- see `requirements-research.txt`), an XGBoost residual
model (an independent gradient-boosting implementation kept as a cross-check on the LightGBM
result -- informational only, if `xgboost` is installed), and an sklearn gradient-boosting
model (also informational only, if `scikit-learn` is installed), simulates an FPL manager's
season points, and writes every artifact to `research/ml/results/`.

The default walk-forward mode is **exhaustive gameweek**: every historical gameweek with prior
training data becomes an out-of-sample test point (one gameweek at a time, re-training on all
prior rows) -- the maximum number of simulations. Use `--fold-mode season` for a coarser
one-fold-per-season view.

`run_continuous.py` is the 24/7 engine: it loops the experiment forever with fresh seeds,
appends a row to the rolling `results/experiment_runs.csv` each run, and tracks the best
FPL-manager points found. It never crashes on a single run failure (it backs off and retries).
Requires the repo's DuckDB to be populated first (the FPL API must be reachable -- it is
blocked from some sandboxed environments, so run this on a machine with open internet).

## Automation (`.github/workflows/ml_experiment.yml`)

The experiment runs for real, automatically, every Sunday (05:00 UTC) via GitHub Actions --
the same open-internet runner `scheduled_pipeline.yml` and `weekly_backtest.yml` already use,
which has neither restriction a sandboxed Claude Code Remote session has (no populated DB
until ingested, `fantasy.premierleague.com` blocked by policy). It restores the most recent
`scheduled_pipeline.yml` run's cached DB (falling back to a fresh ingestion only if none
exists yet), runs `python -m research.ml.experiment` for real, posts
`scripts/summarize_ml_experiment_results.py`'s structured summary to the run's job summary, and
uploads the full `results/` directory as a 90-day build artifact. It does not write
`REPORT.md`'s decision prose -- that is a real analytical judgment call each run, made by a
human reading the job summary/artifact, not something to template-generate unattended. A
manual run (with an optional cheaper `--fold-mode season` override) is available via
`workflow_dispatch`.

It also — the one deliberate exception to "doesn't commit anything back" — appends this run's
real bootstrap-CI numbers and season points (nothing else) to the tracked
`results_history/weekly_quality_history.csv` ledger below, and commits that one file. This is
also GitHub Actions' free-cloud compute doing double duty: the same weekly run that produces
the numbers is what persists them, at no cost beyond what already runs every Sunday. Finally,
it best-effort narrates the run with a local model via Ollama (installed fresh on the runner
each time, `continue-on-error: true` so this never turns a real run red) — see "Optional:
narrating results locally with Ollama" below; the output lands in the uploaded `results/`
artifact as `narrative_draft.md`, same disclaimer as a local run.

## Is the model actually improving over time?

`results_history/weekly_quality_history.csv` is a tracked (not gitignored), append-only ledger
— one row per real `python -m research.ml.experiment` run, written by
`scripts/append_ml_run_to_history.py` and populated automatically by the Sunday workflow above.
Unlike `results/experiment_runs.csv` (a rolling log of `--runs N`/`run_continuous.py`'s own
seed sweeps, reset by every fresh loop, gitignored) this is the one place that accumulates real
numbers across real calendar weeks as the season's data grows, so a genuine trend — not just one
run's point estimate — is actually visible:

| Column | Meaning |
|---|---|
| `run_timestamp_utc`, `git_commit`, `dataset_rows`, `fold_mode` | provenance for that row |
| `quant_lightgbm_point_estimate`/`_ci_low`/`_ci_high`/`_credible` | R11's governing bootstrap CI that week |
| `quant_xgboost_point_estimate`/`_ci_low`/`_ci_high`/`_credible` | R13's confirmation-arm bootstrap CI that week |
| `quant_manager_points`, `ml_manager_points`, `ml_beats_quant` | that week's season-simulation result |

Run it by hand after any real experiment run: `python scripts/append_ml_run_to_history.py` — it
prints a short delta against the previous logged row (point estimate up/down, credible flag
changed, season points moved) so you don't have to open the CSV just to get the headline
signal. This still doesn't decide anything on its own (R11's actual ship/no-ship threshold is
unchanged — the entire 95% CI must exclude zero) — it just makes "has this been trending toward
that, or away from it, over the last N weeks" an answerable question instead of a guess.

## Artifacts (`results/`)

| File | Contents |
|------|----------|
| `baseline_metrics.json` | Quant baseline metrics, overall + sliced |
| `baseline_predictions.parquet` | Quant predictions joined to actuals |
| `model_comparison.csv` | MAE/RMSE/bias/correlation per model per fold |
| `improvement.csv` | Improvement vs the Quant baseline, per metric |
| `residual_analysis.csv` | Where the Quant model is systematically wrong, by slice |
| `high_disagreement_cases.csv` | Cases where ML and Quant disagree most |
| `calibration.csv` | Predicted vs realised, bucketed by predicted value |
| `feature_importance.csv` | Per-fold feature importance |
| `feature_stability.csv` | Mean / coefficient-of-variation of importance across folds |
| `ensemble.csv` | Best ensemble weight (fit on training only) |
| `bootstrap_ci.csv` | Bootstrap confidence interval (default 1,000 resamples, 95%) per metric per model, pooled across all walk-forward folds |
| `runtime.csv` | Fit+predict wall-clock seconds per model per fold |
| `feature_importance_lightgbm.csv`, `feature_stability_lightgbm.csv` | LightGBM gain-based feature importance, per-fold and stability across folds |
| `feature_importance_xgboost.csv`, `feature_stability_xgboost.csv` | XGBoost permutation feature importance, per-fold and stability across folds (R13 confirmation arm) |
| `season_points.csv` | FPL-manager season points: Quant signal vs ML signal |
| `experiment_manifest.json` | Git commit, timestamp, dataset shape, fold mode, skip log |
| `experiment_runs.csv` | Rolling log of every loop run (seed, points, best-so-far) |
| `runs/` | One timestamped subdir per loop run, each with full artifacts |
| `plots/` | PNG charts (secondary; CSVs are the primary record) |

See `REPORT.md` for the report template to fill in after a real run.

## No data leakage

Every feature is assembled through the repo's own `backtest.asof_scope()` shadow, so no
realised-outcome column can reach a prediction row. Programmatic checks in
`leakage_checks.py` abort the build on any violation:

- `ep_model_version` resolves to the matching backtest step
- prediction timestamp strictly precedes the first kickoff
- no forbidden same-gameweek outcome columns in the feature matrix
- the label never appears as a feature
- double-gameweeks are skipped before reaching the dataset
- train and test never share a season; train precedes test chronologically

See `LEAKAGE_PROTOCOL.md` for the full protocol and `EXISTING_MODEL_AUDIT.md` for how the
existing Quant model is used unchanged.

## Tests

```bash
python -m pytest research/ml/tests/ -q
```

The test suite uses a synthetic 2-season DuckDB seeder (`tests/conftest.py`) — no real data
or network access required.

## Optional: narrating results locally with Ollama

Every model here (linear, `quant_gbm`, `quant_lightgbm`, `quant_xgboost`) is a classical
tabular regressor — plain Python/CPU code (`pandas`/`numpy`/`scikit-learn`/`lightgbm`/`xgboost`).
None of it runs "in" an LLM, and Ollama is not required to run `python -m research.ml.experiment`
or anything else in this directory.

What Ollama *is* useful for here: turning the real numeric results in `results/` into a
plain-English draft paragraph, entirely locally (no data leaves your laptop). This is a
separate, optional convenience script — it is never called by `experiment.py` and never affects
`REPORT.md`'s actual decision, which stays a human judgement call (see that file's §9).

**Model choice for a 4GB-VRAM laptop GPU** (e.g. an RTX 3050 Ti, paired with a 12th-gen mobile
i7): the default, `phi4-mini` (3.8B, ~2.5GB at Ollama's default quantization), fully fits in
VRAM with headroom for context, so inference stays fast. This script only narrates numbers it
is already handed — it never does open-ended reasoning — so a small model is enough for the
job; a larger one buys little here. If you want to try a heavier model anyway (somewhat richer
prose, noticeably slower — a Q4 7-8B model like `qwen2.5:7b-instruct` or `llama3.1:8b` is
~4.5-5GB, so it won't fully fit a 4GB card and Ollama will offload part of it to CPU), pass
`--model`:

```bash
# 1. Install Ollama (https://ollama.com/download) and pull a model once:
ollama pull phi4-mini

# 2. Start the server (if it isn't already running as a background service):
ollama serve

# 3. After a real `python -m research.ml.experiment` run has produced results/, narrate them:
python scripts/narrate_ml_results.py
# or with a different model / a remote Ollama host:
python scripts/narrate_ml_results.py --model qwen2.5:7b-instruct --host http://localhost:11434
```

The draft is printed to stdout and written to `results/narrative_draft.md`, headed with an
"AI-generated, for human review only" disclaimer. It narrates only the numbers it is given
(headline MAE/RMSE, `quant_lightgbm`'s bootstrap CI and R13's `quant_xgboost` confirmation-arm
CI, the slicing-regression count, season-manager points) — it is instructed never to invent a
number or issue its own ship/no-ship call. See `research/ml/ollama_client.py` for the (thin,
`requests`-based, no new dependency) HTTP client.
