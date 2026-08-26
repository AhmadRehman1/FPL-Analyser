# Phase-0 ML Residual Experiment — Report

> **Status:** TEMPLATE — not yet run against production data. Fill every section below with the
> numbers produced by `python -m research.ml.experiment` against a real DuckDB populated by the
> ingestion + backtest pipeline. A negative result is a successful research result (spec §30):
> do not frame a non-improvement as a failure.

## 1. Question

Can a machine-learning residual model predict the errors of the existing FPL Quant model
(`ep_total`) more accurately than the Quant model alone, **using only information available at
prediction time**?

The existing Quant model is the baseline to beat. This experiment does **not** replace it and
does **not** modify live production recommendations.

## 2. Method

- **Dataset:** player × gameweek rows. Features = prior rolling stats, minutes-model
  probabilities, fixture context, team/opponent form — all assembled through the repo's own
  `backtest.asof_scope()` shadow so no realised-outcome column can reach a prediction row
  (see [LEAKAGE_PROTOCOL.md](./LEAKAGE_PROTOCOL.md) §5/§8).
- **Label:** `event_points` (per-gameweek realised points), never the cumulative `total_points`.
- **Residual:** `residual = event_points − quant_prediction`. The ML model predicts this residual;
  the final prediction is `quant_prediction + predicted_residual`.
- **Validation:** chronological walk-forward. By default every historical gameweek with prior
  training data is a separate out-of-sample test point (exhaustive expanding-window gameweek
  walk-forward -- the maximum number of historical simulations); a coarser one-fold-per-season
  mode is also available. No `train_test_split`, no shuffling, no row from a later gameweek
  ever appearing in training (spec §4, enforced by `leakage_checks.check_chronological_split`).
- **Models:** (1) Linear / Ridge residual model (closed-form numpy fallback when sklearn is
  absent); (2) optional Gradient Boosting residual model (only if sklearn is available). No neural
  networks, no large hyperparameter search (spec §3).
- **Baselines:** the Quant model unchanged, and a simple historical rolling-mean baseline. If the
  Quant model does not beat the dumb historical baseline, the ML question is moot (spec §8).
- **Ensemble:** `final = w·Quant + (1−w)·ML`, weight fit on **training** residuals only over the
  grid {0, 0.25, 0.5, 0.75, 1.0}. If ML does not help out-of-sample, the learned weight is 0
  (pure Quant) — "do no harm" (spec §16).
- **Season simulation:** a simplified greedy manager picks a starting XI each gameweek from a
  prediction signal (Quant vs ML-augmented), under position/club-balance constraints, and the
  captain's actual points are doubled. Total season points per signal are compared — this asks
  the question that actually matters (fantasy points), not just prediction error. This is a
  research proxy only; it does not call the production optimizer or touch live recommendations.
- **24/7 / repeated runs:** `experiment.py --runs N` and `run_continuous.py` repeat the whole
  pipeline with a fresh random seed each time, appending every run to a rolling log and tracking
  the best ML-manager points found — "as many walk-forward simulations as possible."

## 3. Results

### 3.1 Headline metrics (out-of-sample, walk-forward)

| Season | Model | MAE | RMSE | Bias | Pearson ρ | Rank ρ | n |
|--------|-------|-----|------|------|-----------|--------|---|
|        | quant (baseline) |  |  |  |  |  |  |
|        | quant_linear |  |  |  |  |  |  |
|        | quant_gbm |  |  |  |  |  |  |
|        | historical_baseline |  |  |  |  |  |  |
|        | ensemble (best w=) |  |  |  |  |  |  |

> Source: `results/model_comparison.csv`

### 3.2 Improvement vs the Quant baseline

| Season | Model | Metric | Quant error | ML error | Improvement | Improvement % |
|--------|-------|--------|-------------|-----------|-------------|---------------|
|        |       |        |             |           |             |               |

> Source: `results/improvement.csv`. Improvement = quant_error − ml_error; positive = ML helps.

### 3.3 Ensemble weight selected

- Best `w` (fit on training only): ___
- Train MAE at best `w`: ___ → Test MAE at best `w`: ___
- If `w = 0.0`: ML did not improve out-of-sample; the Quant model alone is recommended.

> Source: `results/ensemble.csv`

### 3.4 Season manager points -- the metric that actually matters

| Season | Signal | Total points | Gameweeks |
|--------|--------|---------------|-----------|
|        | quant_prediction |  |  |
|        | ml_prediction |  |  |

> Source: `results/season_points.csv`. A manager using the ML-augmented signal must score more
> real points than the Quant-only manager across multiple runs/seeds -- a lower MAE that doesn't
> translate into more points is not a reason to integrate.

## 4. Where the Quant model is wrong

`results/residual_analysis.csv` breaks the Quant residual down by position, price band, minutes
band, fixture difficulty, ownership, gameweek, and season. Summarise the slices where the Quant
model is systematically biased (large mean/median error), and whether the ML model corrected them.

## 5. Disagreement

`results/high_disagreement_cases.csv` lists the cases where ML and Quant disagree most, with each
side's error so we can see who was right. Summarise whether ML was right on its high-disagreement
calls.

## 6. Calibration

`results/calibration.csv` buckets predictions by predicted value and compares mean predicted to
mean realised. Summarise whether predictions of N actually average N realised points.

## 7. Feature importance & stability

`results/feature_importance.csv` (per-fold) and `results/feature_stability.csv` (mean / coefficient
of variation across folds). A feature important in one season but irrelevant in every other is
suspicious — note any such features here.

## 8. Slicing check (spec §12)

A model that improves one slice but loses badly in others is **not** declared successful. Confirm
the improvement holds (or document where it does not) across: position, price band, minutes band,
fixture difficulty, ownership, gameweek, and season.

## 9. Decision

- [ ] **ML improves out-of-sample on the primary metric (MAE) and the improvement holds across
      slices** → recommend a controlled integration (separate code path, shadow-only at first).
- [ ] **ML does not improve, or only improves a slice while degrading others** → do not integrate.
      The existing Quant model stands. Document why ML failed to find signal.

### 9.1 If not integrating — why?

(e.g. residual is dominated by genuinely unpredictable variance; features leak no usable signal;
the linear model is capacity-limited and GBM overfits the small sample; etc.)

## 10. Reproducibility

- Git commit: _(recorded in `results/experiment_manifest.json` → `git_commit`)_
- Run timestamp (UTC): _(recorded in `experiment_manifest.json` → `run_timestamp_utc`)_
- Dataset rows / seasons: _(recorded in `experiment_manifest.json`)_
- Skip log (DGW / no-deadline / no-ep-outputs steps): _(recorded in `experiment_manifest.json` →
  `skip_log`)_
- Reproduce: `python -m research.ml.experiment`

## 11. Absolute rule — leakage

Every feature in this experiment was verified asof-safe by the programmatic checks in
`research/ml/leakage_checks.py`, which abort the build on any violation. No realised-outcome
column (minutes, goals, assists, bps, expected_goals, total_points, …) may appear in the feature
matrix; the label never appears as a feature; train and test never share a season; the prediction
timestamp strictly precedes the first kickoff. See [LEAKAGE_PROTOCOL.md](./LEAKAGE_PROTOCOL.md).
