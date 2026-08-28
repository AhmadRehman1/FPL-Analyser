# Phase-0 ML Residual Experiment — Report

> **Status:** TEMPLATE — not yet run against production data. Fill every section below with the
> numbers produced by `python -m research.ml.experiment` against a real DuckDB populated by the
> ingestion + backtest pipeline. A negative result is a successful research result (spec §30):
> do not frame a non-improvement as a failure.
>
> **Track F update** (`docs/plans/2026-08_retrospective_validation_and_ml_decision_plan.md`):
> this template now includes a fourth model arm, `quant_lightgbm` — the **primary** nonlinear
> challenger and the **only** arm the ship/no-ship decision in §9 is governed by (R11). The
> original `quant_gbm` arm (sklearn's `HistGradientBoostingRegressor`) is kept and still reported
> throughout, but demoted to informational/bonus only (R16) — its numbers are real and worth
> reading, but they do not decide anything on their own. Every results table below also expects
> a 95%-confidence bootstrap interval alongside each point estimate (R10) — "positive on
> average" is not sufficient for §9's decision; the interval must exclude zero (R11).
>
> **Phase F-4 (the real run against the production database) has not been executed yet** in any
> session that has touched this file so far — every session so far has been sandboxed, with no
> populated `db/fpl_quant_v2.duckdb` and no outbound access to the live FPL API
> (`fantasy.premierleague.com` returns a proxy 403). Run `python -m research.ml.experiment` on a
> machine with the populated production DB and open internet to produce the real numbers and fill
> every `___`/blank cell below — do not fabricate them.
>
> **Pre-run model improvements** (found while auditing the pipeline ahead of Phase F-4, since a
> real run wasn't possible here either): (1) `position` — approved as a feature source in
> `EXISTING_MODEL_AUDIT.md` §9 and `LEAKAGE_PROTOCOL.md` §4 from the start — was attached to
> every dataset row but never actually listed in `feature_columns()`, so no model (linear,
> `quant_gbm`, or `quant_lightgbm`) ever saw it; now one-hot encoded alongside `status`. A
> goalkeeper's and a forward's point distributions differ enormously, so this was costing every
> arm real signal, not just LightGBM. (2) `LightGBMResidualModel`'s hyperparameters had no L1/L2
> regularisation or row/column subsampling — reasonable for large datasets, but exhaustive
> gameweek walk-forward means the earliest folds train on a few hundred rows, where that risks
> memorising noise. Added `reg_alpha=reg_lambda=0.1`, `subsample=colsample_bytree=0.8` on top of
> the existing `max_depth=4` cap — a one-time, principled choice of more conservative defaults,
> not a hyperparameter search (spec §3 forbids the latter only). (3) `lightgbm_importances` was
> being computed per fold but never persisted anywhere (unlike the linear model's own
> `feature_importance.csv`/`feature_stability.csv`) — added `feature_importance_lightgbm.csv` /
> `feature_stability_lightgbm.csv` so the primary decision-governing arm's feature importance is
> actually inspectable, not silently discarded. All three are covered by `research/ml/tests/`.
>
> **Bug fix (season_sim.py):** `select_starting_xi()`'s position-balance constraints
> (`_POS_MIN`/`_POS_MAX`) were keyed by FPL's short position codes (`"GK"`/`"DEF"`/`"MID"`/
> `"FWD"`), which never actually appear in this repo's data — every position value everywhere
> else (`contract.POSITIONS`, `dim_player.position`, the `position` column `dataset_builder`
> attaches) is the full word (`"Goalkeeper"`, etc.). `if pos not in _POS_MIN: continue` was
> therefore true for every real row, so the greedy position/club-aware selection loop never
> selected anyone, and every single simulated gameweek — for the entire history of this module —
> silently fell through to the "backfill" path: a position-blind top-11-by-predicted-value pick
> respecting only the 3-per-club cap, not a legal FPL formation (it could and did pick e.g. 3
> goalkeepers and 2 midfielders in one lineup). This directly invalidated `results/
> season_points.csv` and the manifest's `season_points` block — the headline "which signal scores
> more real points" metric §3.4 below reports, and the number R11's decision-impact criterion
> weighs. `season_sim.py`'s own dedicated test file (`test_season_sim.py`) didn't catch this
> because its mock fixture used the same wrong short codes, consistently with the bug, so it
> tested the mock's self-consistency rather than the real integration. Fixed: `_POS_MIN`/
> `_POS_MAX` and the goalkeeper special-case now use the real position strings; the test fixture
> was corrected to match `contract.POSITIONS`; a new regression test
> (`test_select_starting_xi_respects_real_fpl_position_rules`) asserts the actual FPL formation
> invariant (exactly 1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD) rather than just XI size — verified to fail
> against the pre-fix code (3 GK / 2 MID) before confirming it passes against the fix.

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
  absent); (2) optional Gradient Boosting residual model, `quant_gbm` (only if sklearn is
  available — reported, but informational only, R16); (3) **LightGBM residual model,
  `quant_lightgbm`** (only if lightgbm is available — the primary nonlinear challenger; R11's
  ship/no-ship decision is governed by this arm alone). No neural networks, no large
  hyperparameter search (spec §3).
- **Bootstrap confidence intervals (R10/R11):** each ML arm's per-fold improvement over the Quant
  baseline (on MAE) is bootstrap-resampled at fold granularity — not per-observation, since
  players within one gameweek share match outcomes and aren't independent — 1,000 resamples, 95%
  interval (`evaluate.bootstrap_ci_for_model_improvement`, `results/bootstrap_ci.json`). A model
  only counts as a statistically credible improvement if the **entire interval** sits above
  zero — a positive point estimate alone is exactly the failure mode this check exists to rule
  out.
- **Compute/runtime (R10):** each model's real fit+predict wall-clock time is captured per fold
  (`results/compute_runtime.csv`), so §9's decision can weigh a real accuracy gain against its
  real compute cost, not just assume ML is free.
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
|        | quant_gbm *(informational only — R16)* |  |  |  |  |  |  |
|        | **quant_lightgbm *(governs §9's decision — R11)*** |  |  |  |  |  |  |
|        | historical_baseline |  |  |  |  |  |  |
|        | ensemble (best w=) |  |  |  |  |  |  |

> Source: `results/model_comparison.csv`

### 3.1b Bootstrap confidence intervals on MAE improvement (R10/R11)

| Model | Point estimate (MAE improvement) | 95% CI low | 95% CI high | n folds | Statistically credible? |
|-------|-----------------------------------|-----------|-----------|---------|--------------------------|
| quant_linear |  |  |  |  |  |
| quant_gbm *(informational only)* |  |  |  |  |  |
| **quant_lightgbm** |  |  |  |  |  |

> Source: `results/bootstrap_ci.json` (also embedded in `results/experiment_manifest.json` →
> `bootstrap_ci`). "Statistically credible" = `True` only when the entire 95% interval sits above
> zero. **This table's `quant_lightgbm` row is what §9's decision is actually based on.**

### 3.1c Compute/runtime (R10)

| Model | Total fit+predict seconds (all folds) | Mean seconds/fold | n folds |
|-------|----------------------------------------|--------------------|---------|
| quant_linear |  |  |  |
| quant_gbm |  |  |  |
| quant_lightgbm |  |  |  |

> Source: `results/compute_runtime.csv`, grouped by model.

### 3.2 Improvement vs the Quant baseline

| Season | Model | Metric | Quant error | ML error | Improvement | Improvement % |
|--------|-------|--------|-------------|-----------|-------------|---------------|
|        |       |        |             |           |             |               |

> Source: `results/improvement.csv`. Improvement = quant_error − ml_error; positive = ML helps.
> This table is the per-season point-estimate view; §3.1b's bootstrap CI is the one §9's decision
> actually reads.

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
> translate into more points is not a reason to integrate. Note (Track F): `ml_prediction` here
> reflects the experiment's ensemble-priority convention (LightGBM > `quant_gbm` > linear, whichever
> is available) — it is a "best available ML signal" view, not itself the §9 decision input.

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

`results/feature_importance.csv` (per-fold, `quant_linear`) and `results/feature_stability.csv`
(mean / coefficient of variation across folds). `results/feature_importance_lightgbm.csv` /
`results/feature_stability_lightgbm.csv` are the same for `quant_lightgbm` — the arm §9's
decision is actually governed by, so its own feature importance matters more than the linear
model's. A feature important in one season but irrelevant in every other is suspicious — note
any such features here.

## 8. Slicing check (spec §12)

A model that improves one slice but loses badly in others is **not** declared successful. Confirm
the improvement holds (or document where it does not) across: position, price band, minutes band,
fixture difficulty, ownership, gameweek, and season. **R11 requires this check specifically for
`quant_lightgbm`** before any "ship" verdict in §9 — a slicing regression there blocks shipping
even if the aggregate/bootstrap-CI numbers in §3.1b look credible.

> Source: `results/sliced_model_comparison.csv` — one row per (fold, model, dimension, slice),
> for every model, using the same slice definitions `baselines.sliced_metrics()`/
> `save_baseline_metrics()`'s own Quant-only report already uses (added for Track F; previously
> this breakdown only ever existed for the Quant baseline). Aggregate across folds per
> (model, dimension, slice) and compare `quant_lightgbm`'s MAE against `quant`'s in every slice
> before summarising here — a slice where `quant_lightgbm` is worse than Quant, even if the
> aggregate number in §3.1b looks good, is exactly what this check exists to catch.

## 9. Decision

**Governed by `quant_lightgbm` alone (R11)** — §3.1b's bootstrap CI for that arm, plus this
section's slicing check for that arm, decide the verdict below. `quant_gbm`'s numbers (real,
reported throughout this document) are informational only and do not factor into this checkbox
(R16) — do not let a `quant_gbm` result substitute for `quant_lightgbm`'s here even if the two
happen to point the same direction.

- [ ] **`quant_lightgbm` improves out-of-sample on the primary metric (MAE), the entire 95%
      bootstrap CI sits above zero (§3.1b), and the improvement holds across slices (§8)** →
      recommend a controlled integration (separate code path, shadow-only at first). Flag
      whether XGBoost should be added as a further, independent-implementation confirmation
      (R13 — only if this result is positive or borderline).
- [ ] **`quant_lightgbm` does not improve, the CI does not exclude zero, or it only improves a
      slice while degrading others** → do not integrate. The existing Quant model stands.
      Document why ML failed to find signal below.

### 9.1 If not integrating — why?

(e.g. residual is dominated by genuinely unpredictable variance; features leak no usable signal;
LightGBM overfits the small sample; the bootstrap CI crosses zero despite a positive point
estimate — name the actual reason once real numbers exist, don't leave this templated.)

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
