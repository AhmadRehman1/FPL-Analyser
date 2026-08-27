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
folds, fits a linear residual model (and an optional gradient-boosting model if sklearn is
installed), simulates an FPL manager's season points, and writes every artifact to
`research/ml/results/`.

The default walk-forward mode is **exhaustive gameweek**: every historical gameweek with prior
training data becomes an out-of-sample test point (one gameweek at a time, re-training on all
prior rows) -- the maximum number of simulations. Use `--fold-mode season` for a coarser
one-fold-per-season view.

`run_continuous.py` is the 24/7 engine: it loops the experiment forever with fresh seeds,
appends a row to the rolling `results/experiment_runs.csv` each run, and tracks the best
FPL-manager points found. It never crashes on a single run failure (it backs off and retries).
Requires the repo's DuckDB to be populated first (the FPL API must be reachable -- it is
blocked from some sandboxed environments, so run this on a machine with open internet).

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
