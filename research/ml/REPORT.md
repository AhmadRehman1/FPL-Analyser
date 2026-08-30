# Phase-0 ML Residual Experiment — Report

> **Status: SHIP (recommend shadow-only integration) — as of 2026-08-30, see §8b + §9.**
> §3–§8 document the L2-objective model (`git_commit` `45514ae`, run against `backtest_run_id=1`)
> and its multi-seed cloud confirmation (§8a); that model was **NO-SHIP** on one seed-stable
> per-slice regression (`ownership_band=20%+`). §8b then identifies the cause (L2 loss vs an MAE
> metric) and the fix (L1 objective on all tree arms), which removes the regression on every
> slice and lifts the MAE improvement to 0.611. §8b + §9 carry the current verdict; the older
> sections are the historical record and their numbers are the L2 model's, not the ship model's.
>
> **2026-08-30 — verdict: SHIP (recommend shadow-only integration). See §8c + §9.** The
> 2026-08-29 sweep (§8a) left NO-SHIP resting on one per-slice regression (`quant_lightgbm` worse
> than Quant on `ownership_band=20%+`, 5/5 seeds). §8b switched the training objective from L2 to
> L1 and that cleared it — but an external review (§8c) then showed L1 "won" on MAE by fitting the
> biased conditional *median* (final bias −0.48, worse player-ranking and worse manager-sim points
> than L2). The shipped answer is **Huber δ=4**: 0 regressing slices, near-L2 RMSE, best rank
> correlation, best season points among gate-clearing models. §8c also documents that **pure L2 is
> strictly better on value** but trips the one-slice gate — a carve-out call for the plan owner.
> §3–§8 document the original L2 model (historical); **§8c + §9 carry the current state.** All on
> `master`.
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
> **R13 resolved and now part of this report's real numbers:** `quant_xgboost`
> (`research/ml/residual_model.py::XGBoostResidualModel`) is the independent-implementation
> confirmation arm this report's §9 originally flagged as a follow-up ("Flag whether XGBoost should
> be added..."). As of this 2026-08-29 re-run `xgboost` is installed, so it ran for real and its
> numbers are reported alongside `quant_gbm` throughout — **informational only** (like `quant_gbm`,
> R16 — it never governs §9's decision, which remains `quant_lightgbm` alone per R11). Its value:
> two independently implemented gradient-boosting libraries can now be compared directly on whether
> there is out-of-sample residual signal, which is evidence about whether `quant_lightgbm`'s result
> is a real finding or an artifact of one library's defaults/splitting heuristic. Result: they
> agree closely everywhere — aggregate MAE within 0.006, CI both excluding zero with margin, and
> **both regress on the exact same 2 of 60 slices** (§8). The agreement cuts both ways: it
> corroborates the strong aggregate finding *and* corroborates the narrow per-slice weakness.
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
> actually inspectable, not silently discarded — **§7 below is filled from these two files, not
> the linear model's**, since the earlier version of this report cited the wrong file for the
> arm §9 is actually governed by (caught in blind review — see PR discussion). All three are
> covered by `research/ml/tests/`.
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
> weighs. Fixed upstream (see git history) before this report's real run — §3.4 below reflects
> the corrected, legal-formation numbers; an earlier run of this same experiment (before the fix
> landed in this checkout) produced different, invalid season-points numbers that were briefly
> transcribed into an intermediate draft of this report and have been replaced.

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
  ship/no-ship decision is governed by this arm alone); (4) XGBoost residual model,
  `quant_xgboost` (only if xgboost is available — the R13 independent-implementation confirmation
  arm, informational only like `quant_gbm`, R16). No neural networks, no large hyperparameter
  search (spec §3).
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
  grid {0, 0.25, 0.5, 0.75, 1.0}. If ML does not help out-of-sample, the learned weight is **1**
  (pure Quant, i.e. `(1−w)=0` weight on ML) — "do no harm" (spec §16). *(Corrected here: an
  earlier version of this line said "the learned weight is 0" — backwards given this section's
  own formula, where `w` is literally the coefficient **on Quant**. §3.3 reports the real result
  using this corrected convention.)*
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

Aggregated (mean) across all 70 walk-forward folds, both seasons combined — 52,323 real test-row
evaluations, first test step `2024-2025 GW3`, last `2025-2026 GW38`, zero skipped steps
(`skip_log` is empty).

| Model | MAE | RMSE | Bias | Pearson ρ | Rank ρ | n (total) |
|-------|-----|------|------|-----------|--------|---|
| quant (baseline) | 1.525 | 2.162 | 0.509 | 0.443 | 0.495 | 52,323 |
| quant_linear | 1.108 | 1.971 | -0.037 | 0.550 | 0.691 | 52,323 |
| quant_gbm *(informational only — R16)* | 1.012 | 1.937 | -0.035 | 0.565 | 0.707 | 52,323 |
| **quant_lightgbm *(governs §9's decision — R11)*** | **1.010** | **1.934** | **-0.035** | **0.567** | **0.709** | 52,323 |
| quant_xgboost *(informational only — R13 confirmation arm)* | 1.016 | 1.940 | -0.023 | 0.564 | 0.708 | 52,323 |
| historical_baseline | 1.053 | 2.108 | -0.008 | 0.497 | 0.702 | 52,323 |
| ensemble (best w=0 on Quant, i.e. 100% ML — see §3.3) | 1.010 | 1.934 | -0.035 | 0.567 | 0.709 | 52,323 |

Per-season breakdown:

| Season | Model | MAE | RMSE | Bias | Pearson ρ | Rank ρ | n |
|--------|-------|-----|------|------|-----------|--------|---|
| 2024-2025 | quant | 1.513 | 2.164 | 0.506 | 0.428 | 0.436 | 25,238 |
| 2024-2025 | quant_lightgbm | 1.019 | 1.937 | -0.018 | 0.561 | 0.704 | 25,238 |
| 2024-2025 | quant_xgboost *(informational)* | 1.030 | 1.951 | 0.002 | 0.556 | 0.702 | 25,238 |
| 2025-2026 | quant | 1.538 | 2.160 | 0.512 | 0.459 | 0.553 | 27,085 |
| 2025-2026 | quant_lightgbm | 1.001 | 1.930 | -0.053 | 0.574 | 0.715 | 27,085 |
| 2025-2026 | quant_xgboost *(informational)* | 1.002 | 1.930 | -0.047 | 0.573 | 0.715 | 27,085 |

> Source: `results/model_comparison.csv`. `quant_gbm`, `quant_lightgbm` and `quant_xgboost` track
> each other extremely closely throughout this report (aggregate MAE within 0.006 of each other) —
> all three are gradient-boosted trees over the same feature set, so this is expected, not a bug;
> R11 still governs off `quant_lightgbm` specifically, per the plan. `quant_xgboost` (independent
> library, independent implementation) landing within 0.006 MAE of `quant_lightgbm` is the R13
> confirmation signal: the residual signal is real, not a LightGBM-specific artifact.

### 3.1b Bootstrap confidence intervals on MAE improvement (R10/R11)

| Model | Point estimate (MAE improvement) | 95% CI low | 95% CI high | n folds | Statistically credible? |
|-------|-----------------------------------|-----------|-----------|---------|--------------------------|
| quant_linear | 0.417 | 0.397 | 0.432 | 70 | True |
| quant_gbm *(informational only)* | 0.513 | 0.498 | 0.525 | 70 | True |
| **quant_lightgbm** | **0.515** | **0.499** | **0.527** | **70** | **True** |
| quant_xgboost *(informational only — R13 confirmation arm)* | 0.509 | 0.492 | 0.522 | 70 | True |

> Source: `results/bootstrap_ci.json` (also embedded in `results/experiment_manifest.json` →
> `bootstrap_ci`). "Statistically credible" = `True` only when the entire 95% interval sits above
> zero. **This table's `quant_lightgbm` row is what §9's decision is actually based on** — the
> entire interval sits comfortably above zero (low bound 0.499 MAE, a ~33% relative improvement),
> satisfying R11's confidence-interval requirement. This half of R11 is met; §8 covers the other
> half. `quant_xgboost`'s interval (0.492–0.522) overlaps `quant_lightgbm`'s almost entirely — the
> R13 confirmation arm independently clears the same bar.

### 3.1c Compute/runtime (R10)

| Model | Total fit+predict seconds (all folds) | Mean seconds/fold | n folds |
|-------|----------------------------------------|--------------------|---------|
| quant_linear | 4.7s | 0.067s | 70 |
| quant_gbm | 25.4s | 0.363s | 70 |
| quant_lightgbm | 17.9s | 0.256s | 70 |
| quant_xgboost | 20.7s | 0.296s | 70 |

> Source: `results/compute_runtime.csv` (`compute_runtime_seconds_by_model` in the manifest),
> grouped by model. `quant_lightgbm` is both slightly more accurate than `quant_gbm` on raw MAE
> (1.010 vs 1.012) and ~30% cheaper to fit/predict (17.9s vs 25.4s total); `quant_xgboost` sits
> between them on cost (20.7s). Compute cost is not a concern for any arm at this scale
> (sub-second per fold); this table exists mainly to confirm ML isn't silently expensive.

### 3.2 Improvement vs the Quant baseline

| Season | Model | Metric | Quant error | ML error | Improvement | Improvement % |
|--------|-------|--------|-------------|-----------|-------------|---------------|
| 2024-2025 | quant_lightgbm | MAE | 1.513 | 1.019 | 0.493 | 32.6% |
| 2024-2025 | quant_lightgbm | RMSE | 2.164 | 1.937 | 0.227 | 10.5% |
| 2024-2025 | quant_gbm *(informational)* | MAE | 1.513 | 1.021 | 0.491 | 32.5% |
| 2024-2025 | quant_xgboost *(informational)* | MAE | 1.513 | 1.030 | 0.483 | 31.9% |
| 2025-2026 | quant_lightgbm | MAE | 1.538 | 1.001 | 0.536 | 34.9% |
| 2025-2026 | quant_lightgbm | RMSE | 2.160 | 1.930 | 0.230 | 10.6% |
| 2025-2026 | quant_gbm *(informational)* | MAE | 1.538 | 1.004 | 0.534 | 34.7% |
| 2025-2026 | quant_xgboost *(informational)* | MAE | 1.538 | 1.002 | 0.535 | 34.8% |

> Source: `results/improvement.csv`. Improvement = quant_error − ml_error; positive = ML helps.
> The improvement is consistent across both seasons (32.6% and 34.9% MAE improvement for
> `quant_lightgbm` respectively) — not a single-season artifact. Both informational arms
> (`quant_gbm`, `quant_xgboost`) land within ~1 percentage point of it in both seasons. This table
> is the per-season point-estimate view; §3.1b's bootstrap CI is the one §9's decision actually
> reads.

### 3.3 Ensemble weight selected

- Best `w` (fit on training only, `w` = weight on **Quant** per §2's formula): **0.0, in all
  70/70 folds, no exceptions** — the training-only grid search never once preferred any nonzero
  weight on Quant; it always preferred 100% ML.
- Mean train MAE at best `w`: 0.995 → mean test MAE at best `w`: 1.010 (small, expected
  train→test gap; no sign of gross overfitting).
- This is a consistent, unanimous signal in ML's favor across every fold, not a marginal or
  fold-dependent result.

> Source: `results/ensemble.csv`

### 3.4 Season manager points -- the metric that actually matters

| Season | Signal | Total points | Gameweeks |
|--------|--------|---------------|-----------|
| 2024-2025 | quant_prediction | 1,867.0 | 35 |
| 2024-2025 | ml_prediction | 2,211.0 | 35 |
| 2025-2026 | quant_prediction | 1,896.0 | 35 |
| 2025-2026 | ml_prediction | 2,054.0 | 35 |

> Source: `results/season_points.csv` (`season_points` in the manifest: `quant_manager` 3,763.0,
> `ml_manager` 4,265.0, `ml_beats_quant` true), from the corrected `season_sim.py` (see banner). A
> manager using the ML-augmented signal must score more real points than the Quant-only manager
> across multiple runs/seeds -- a lower MAE that doesn't translate into more points is not a reason
> to integrate. Here it does: **+344 points (+18.4%)** in 2024-25 and **+158 points (+8.3%)** in
> 2025-26 over a 35-gameweek season using this simplified greedy-manager proxy — consistent in
> direction and rough magnitude with the previous run (+312 / +146). Note (Track F):
> `ml_prediction` here reflects the experiment's ensemble-priority
> convention (LightGBM > `quant_xgboost` > `quant_gbm` > linear, whichever is available — see
> `experiment.py`) — since LightGBM was available and is the strongest arm throughout this run,
> this row **is** effectively the
> `quant_lightgbm` signal — but it is still a "best available ML signal" view structurally, not
> itself the §9 decision input (§3.1b/§8 are).

## 4. Where the Quant model is wrong

`results/residual_analysis.csv` (real, n-weighted mean error by position, `residual = event_points
− quant_prediction`, so positive = Quant under-predicts):

| Position | Mean error (Quant) | n |
|----------|---------------------|---|
| Defender | +0.793 | 17,247 |
| Goalkeeper | +0.559 | 5,762 |
| Midfielder | +0.356 | 23,394 |
| Forward | +0.246 | 5,920 |

Quant systematically **under-predicts defenders** most (by far the largest bias of the four
positions) and is closest to unbiased for forwards. This matches §6's calibration finding below
(Quant over-predicts the low-score bucket where most defenders' rows land) and is consistent with
§3.1's aggregate bias sign (+0.509, i.e. Quant under-predicts on average across all positions).

## 5. Disagreement

`results/high_disagreement_cases.csv`: 14,000 real high-disagreement rows (cases where
`quant_prediction` and `ml_prediction` differed most, `ml_prediction` = the `quant_lightgbm`
signal). **ML was closer to the actual outcome (`|ml_error| < |quant_error|`) in 87.6% of these
cases.** This is the single most direct evidence in this report that the ML model isn't just
shaving a fraction of a point off an already-good baseline — on the rows where the two models
actively disagree, ML is right almost 9 times out of 10.

## 6. Calibration

`results/calibration.csv`, n-weighted, mean predicted vs. mean actual per predicted-value bucket:

| Bucket | Quant: mean predicted | Quant: mean actual | quant_lightgbm: mean predicted | quant_lightgbm: mean actual |
|--------|------------------------|----------------------|-----------------------------------|--------------------------------|
| 0-2 | 1.138 | 0.531 | 0.574 | 0.584 |
| 2-4 | 2.811 | 2.570 | 2.819 | 2.903 |
| 4-6 | 4.439 | 3.931 | 4.525 | 4.165 |
| 6-8 | 6.162 *(n=3, noise)* | 1.000 *(n=3, noise)* | 6.812 | 6.180 |

Quant is badly miscalibrated in the largest bucket (0-2, ~37,500 rows): it predicts a mean of
1.14 against a real mean of 0.53 — **more than double** the realised value, i.e. Quant is
systematically too optimistic about players it expects to blank or nearly blank.
`quant_lightgbm`'s 0-2 bucket is well calibrated (0.574 predicted vs. 0.584 actual) — this single
bucket is most of both the aggregate MAE gap in §3.1 and the aggregate bias-sign flip (Quant
+0.509 → quant_lightgbm −0.035), since it holds the majority of all rows.

## 7. Feature importance & stability

`results/feature_stability_lightgbm.csv` (mean importance / coefficient of variation across all
70 folds, for `quant_lightgbm` — the arm §9's decision is actually governed by; an earlier draft
of this section incorrectly cited the linear model's `feature_stability.csv` instead, caught in
blind review before this report was finalized; `feature_importance_xgboost.csv`/
`feature_stability_xgboost.csv` now exist for the real `quant_xgboost` arm too — its top three are
the same three features in a different order, `p_start_final` / `rolling_points_3` / `p_60plus_min`,
also its most stable, CV 0.40–0.46), top real drivers:

| Feature | Mean importance | CV (std/mean) |
|---------|------------------|----------------|
| `p_60plus_min` | 0.0965 | 0.367 |
| `p_start_final` | 0.0903 | 0.301 |
| `rolling_points_3` | 0.0857 | 0.300 |
| `status=a` (available) | 0.0287 | 0.834 |
| `rolling_bps` | 0.0210 | 0.580 |
| `chance_of_playing_next_round` | 0.0163 | 0.951 |

The minutes-model probabilities (`p_60plus_min`, `p_start_final`) and short-term form
(`rolling_points_3`) dominate `quant_lightgbm`'s real importance, and are also its most *stable*
drivers (CV 0.30–0.37, the lowest three of the top six) — consistently the dominant signal across
folds, not a fold-specific artifact. The single-gameweek availability flag `status=a` and
`chance_of_playing_next_round` matter but are far less stable (CV 0.83–0.95) — expected, since
day-to-day fitness/rotation news is inherently gameweek-specific, not a persistent per-player
signal the model can lean on every fold. No feature showed the "important in one season,
irrelevant in every other" red flag this section exists to catch.

## 8. Slicing check (spec §12)

**Real result (this local run): 58 of 60 (dimension, slice) combinations show `quant_lightgbm`
improving on Quant. 2 of 60 regress — both named here in full, per R11/this section's own
requirement, not omitted. `quant_xgboost` (the R13 confirmation arm) regresses on the *same two
slices*. §8a's 5-seed cloud sweep then resolves which of the two is real (`ownership_band=20%+`)
and which is noise (`price_band=9.0+`):**

| Dimension | Slice | Quant MAE | quant_lightgbm MAE | Change (lgbm) | Change (xgboost) | n (real, non-trivial) |
|-----------|-------|-----------|----------------------|---------------|------------------|----|
| `ownership_band` | `20%+` (highly-owned players) | 3.135 | 3.200 | **−0.066 (worse)** | −0.096 (worse) | 1,149 |
| `price_band` | `9.0+` (premium-priced players) | 3.343 | 3.371 | **−0.028 (worse)** | −0.044 (worse) | 498 |

(In the prior run — `8a94bd2`, before this 2026-08-29 re-run — only `ownership_band=20%+` regressed
(−0.048); `price_band=9.0+` was flat there (+0.007). This re-run puts it back on the regressing
side. The two slices correlate heavily — premium-priced and highly-owned players overlap a great
deal — so this is one real, narrow underlying effect sitting close to the noise floor, not two
independent ones: the fresh run's own fold-to-fold variance is enough to move a near-zero effect
across the improve/regress line. That an *independently implemented* library, `quant_xgboost`,
regresses on both slices in the same run makes it more likely the effect is real than a
single-library artifact.)

Both regressions are real (1,149 and 498 real observations, not sampling noise) but narrow: small
in absolute magnitude (2.1% and 0.8% relative to the slice's own MAE), and set against Quant's own
MAE already being at its *worst* in exactly these slices (3.13 / 3.34 vs. an overall MAE of 1.53) —
i.e. these are among the hardest-to-predict player populations for every model, where variance is
dominated by genuinely unpredictable factors (rotation risk, blanks/hauls for premium attackers),
not a place any model is confidently right. Every other slice — all four positions, all three
minutes bands, all four fixture-difficulty bands, every gameweek 1–38, both seasons, and the lower
price/ownership bands — shows `quant_lightgbm` improving on Quant, several substantially (e.g.
`price_band=<5.0`: 1.346 → 0.689, a 0.657 improvement; `position=Defender`: 1.736 → 1.111, a 0.625
improvement; `ownership_band=unknown`: 1.521 → 1.002, a 0.518 improvement).

Practically, these two slices are exactly FPL's highest-stakes decision — captaincy
disproportionately involves highly-owned, premium-priced players — so this is not a corner case to
wave away even though it's numerically narrow and small. See §9 for how the plan's own
pre-committed rule for this exact situation applies.

> Source: `results/sliced_model_comparison.csv` — one row per (fold, model, dimension, slice),
> for every model. Aggregated n-weighted across folds per (model, dimension, slice), comparing
> `quant_lightgbm`'s (and `quant_xgboost`'s) MAE against `quant`'s in all 60 real (dimension,
> slice) combinations before summarising above.

### 8a. Multi-seed cloud confirmation (2026-08-30)

The experiment above ran locally against `backtest_run_id=1`. To separate a real per-slice
weakness from run-to-run fold-variance noise, it was re-run on GitHub Actions (`ubuntu-latest`,
unthrottled) against a **fresh** walk-forward built by `scripts/run_walkforward.py` — one baseline
run plus a 5-seed sweep (seeds 42–46), each a full 70-fold gameweek walk-forward. All six agree on
the aggregate: `quant_lightgbm` MAE improvement 0.517, CI [0.502, 0.530]; `ml_beats_quant` in
every run.

Per-slice regression, by seed (`quant_lightgbm` vs `quant`, n-weighted MAE; − = worse):

| seed | # slices `quant_lightgbm` regresses on | `ownership_band=20%+` Δ | `price_band=9.0+` Δ |
|------|----------------------------------------|-------------------------|---------------------|
| 42 | 1 | −0.052 | **+0.020** |
| 43 | 1 | −0.065 | **+0.013** |
| 44 | 2 | −0.085 | −0.037 |
| 45 | 2 | −0.065 | −0.016 |
| 46 | 2 | −0.073 | −0.002 |
| **pooled (n≈5.7k / 2.5k)** | — | **−0.068** | **−0.004** |

`quant_xgboost` regresses on **both** slices in all 5 seeds (pooled: `ownership_band=20%+` −0.107,
`price_band=9.0+` −0.101).

**Reading:**
- **`ownership_band=20%+`** (highly-owned players, n≈1,150/seed) — regresses in **5/5 seeds for
  `quant_lightgbm` and 5/5 for `quant_xgboost`**. This is a **real, seed-stable weakness**, not
  noise. It is the one slice §9's NO-SHIP now genuinely rests on.
- **`price_band=9.0+`** (premium-priced, n≈500/seed) — for the governing arm it *improves* in 2
  seeds and *regresses* in 3, pooling to ≈0. **Noise-dominated for `quant_lightgbm`.** It stays a
  real regression for `quant_xgboost` (informational only).

> Source: run artifacts `ml-experiment-results-33281320762` (baseline) and
> `ml-experiment-results-33281374600` (`experiment_runs.csv` + `runs/run_000{0..4}/`). Cloud
> pipeline: `.github/workflows/ml_experiment.yml` (PR #73).

### 8b. The L1 training objective clears the gate (2026-08-30)

§8a leaves NO-SHIP resting on exactly one thing: `quant_lightgbm` is worse than Quant on
`ownership_band=20%+` (highly-owned players) in 5/5 seeds. A short cloud iteration loop
(`scratchpad/ITERATION_LOG.md`, four experiments each measured against a fresh master baseline)
found the cause and the fix.

**Cause.** The experiment's primary metric is MAE, but every tree arm trained on **L2**
(`objective="regression"` / `loss="squared_error"` / `"reg:squarederror"`). L2 on the residual is
pulled by the rare double-digit haul, so the model over-corrects on the highest-variance
population — captained premiums, i.e. exactly the highly-owned slice. Adding *features* aimed at
that slice (recent xG/xA volume, PR #74) made it slightly **worse**, confirming the problem was
the loss, not missing signal.

**Fix.** Train every tree arm on **L1** (`regression_l1` / `absolute_error` /
`reg:absoluteerror`) — median-optimal, matches the metric, robust to the tail. Same category as
the existing `reg_alpha`/`subsample` defaults: a one-time principled choice, not a hyperparameter
search (spec §3). Branch `claude/ml-l1-all-arms` / `claude/ml-l1-defcon-ship`.

**Result** (cloud, fresh walk-forward, seeds 42–46 + canonical; runs `33309139236` /
`33309669402` for the LightGBM-only change, `33311976548` / `33312534544` for all-arms + the
`rolling_defcon` features):

| | L2 baseline (§8a) | **L1 (all arms) + defcon** |
|---|---|---|
| `quant_lightgbm` MAE improvement | 0.515  CI [0.501, 0.528] | **0.611  CI [0.594, 0.624]** |
| headline `quant_lightgbm` MAE (mean folds) | ~1.010 | **0.915** |
| **`quant_lightgbm` regressing slices** | **1** (5/5 seeds) | **0** (5/5 seeds) |
| `ownership_band=20%+` d (pooled) | **−0.068** | **+0.147** |
| `quant_gbm` regressing slices | 2 | **0** |
| `quant_xgboost` (R13 arm) regressing slices | 2 | **0** |
| ml season pts (greedy-manager proxy; quant = 3745) | 4178 | ~4086 |

**Every one of 60 slices improves, for every one of the three tree arms, in every one of 5
seeds.** The `ownership_band=20%+` regression that drove NO-SHIP is not just gone — that slice now
*improves* by ~0.15 MAE. The R13 independent-implementation confirmation arm (`quant_xgboost`),
which contradicted the cleared result under L2, now confirms it.

The one cost: the greedy-manager season-points proxy drops ~90–120 points versus L2 (still a
clear ML win over Quant, 4086 vs 3745). L1 gets the median right; the captain pick (`argmax`)
wants the upper tail. The `rolling_defcon` features recover ~25 of those points and don't disturb
the slice result. This is a genuine metric tension worth a follow-up (a ceiling-aware captain
signal for the manager sim), not a blocker.

### 8c. Correction — L1 gamed MAE; the shipped point forecast is Huber (2026-08-30, later)

An external review (Perplexity, prompt in `scratchpad/PERPLEXITY_PROMPT.md`) plus the L1 ship
run's own numbers exposed a real problem with §8b's framing: **MAE targets the conditional
*median*, and FPL points are so zero/2-heavy that the median sits ~0.5 below the mean.** The L1
model therefore "won" on MAE partly by predicting low everywhere:

- final-prediction **bias −0.48** (the L2 baseline is −0.04, essentially unbiased)
- **RMSE 2.05** — barely better than the L2 baseline's 1.93, while MAE dropped 40%
- **within-gameweek rank correlation 0.688** — *worse* than L2's 0.709; L1 is measurably worse at
  the thing XI selection actually needs, which is *ranking* players
- greedy-manager **season points 4,086 — below the L2 model's 4,178**

So L1 minimised the reporting metric while being the wrong functional for team selection. A
five-run sweep on the LightGBM point-forecast objective settles it (compare artifacts vs the L2
baseline `33303614405` and the L1 ship `33313576174`):

| objective | MAE | RMSE | bias | rank ρ | regressing slices | ml season pts |
|-----------|-----|------|------|--------|-------------------|---------------|
| pure L2 (`regression`) | 1.011 | **1.933** | **−0.03** | 0.709 | **1** (`ownership_band=20%+` −0.057) | **4,313** |
| **Huber δ=4** (shipped) | 0.967 | 1.941 | −0.17 | **0.710** | **0** | 4,208 |
| Huber δ=2 | 0.932 | 1.982 | −0.32 | 0.709 | 0 | 4,192 |
| L1 (`regression_l1`) | **0.915** | 2.048 | −0.48 | 0.688 | 0 | 4,086 |

The picture: **the per-slice regression is real and loss-dependent.** Pure L2 is the best mean
forecast on every value axis — near-zero bias, best RMSE, best season points (4,313) — but it
over-corrects the highest-variance slice (captained premiums, whose outcomes are near-random) and
dips 0.057 MAE below Quant there. L1 removes that by being biased low. **Huber δ=4 is the frontier
point**: L2 behaviour for `|residual| < 4` (a near-unbiased mean for the bulk of players), L1 only
for genuine hauls — 0 regressing slices, rank ρ and season points at/above the L2 baseline, RMSE
back to L2, residual bias −0.17 (closeable with a linear calibration step; tracked).

Shipped model (`experiment.py`, all on `master`): LightGBM point forecast on **Huber δ=4**;
`quant_xgboost` on **pseudo-Huber slope 4** (matched, so the R13 confirmation arm confirms the
same functional); `quant_gbm` unchanged; the q90 quantile captain-ceiling arm (§9.1) unchanged.

> Source runs: `33328436934` (Huber δ2), `33329089778` (Huber δ4), `33329778704` (pure L2
> control), `33332070357` (Huber δ4 re-confirm after the alpha-plumbing fix below).
>
> **Fix note (PR #82):** `LightGBMResidualModel` only forwarded `alpha` to LightGBM for
> `objective="quantile"`, so for one commit `master`'s `objective="huber", alpha=4.0` silently
> ran with LightGBM's default Huber delta (0.9 ≈ L1). Caught by comparing the consolidated ship
> run (`quant_lightgbm` bias −0.40) against `33329089778` (bias −0.17 — that branch's lineage
> carried the fix). One line: `if self.objective in ("quantile", "huber", "fair")`. Run
> `33332070357` re-confirms the δ4 numbers in the table: `quant_lightgbm` MAE 0.967, RMSE 1.94,
> bias −0.17, rank ρ 0.710, 0 regressing slices; `quant_xgboost` (matched pseudo-Huber) MAE
> 0.967, bias −0.17, 0 regressing slices.

## 9. Decision

**Governed by `quant_lightgbm` alone (R11)** — §3.1b's bootstrap CI for that arm, plus this
section's slicing check for that arm, decide the verdict below. `quant_gbm`'s and `quant_xgboost`'s
numbers (real, reported throughout this document) are informational only and do not factor into
this checkbox (R16) — do not let a `quant_gbm` or `quant_xgboost` result substitute for
`quant_lightgbm`'s here even when all three point the same direction (as they do in this run).

- [x] **`quant_lightgbm` improves out-of-sample on the primary metric, the entire 95%
      bootstrap CI sits above zero, and the improvement holds across all slices (§8c)** →
      recommend a controlled integration (separate code path, shadow-only at first). R13 is
      resolved *and consistent*: `quant_xgboost` (independent library, matched pseudo-Huber
      objective) clears the same bar.
- [ ] **`quant_lightgbm` does not improve, the CI does not exclude zero, or it only improves a
      slice while degrading others** → do not integrate. *(This was the verdict under the L2
      objective — see the history below.)*

**Verdict: SHIP — recommend a controlled, shadow-only integration** (a separate `ep_total_ml`
column alongside `ep_total`, never replacing it), with the **Huber δ=4** point-forecast model on
`master`. It clears the pre-committed per-slice gate — **0 regressing slices, every one of 60
slices improves** (§8c) — while being a genuine value improvement over both the L2 baseline and
the L1 model: near-L2 RMSE, best within-gameweek rank correlation (0.710), best greedy-manager
season points among gate-clearing models (4,208 vs the L2 baseline's 4,178), MAE 0.967 vs 1.011.

**One thing for the plan owner to decide.** §8c shows **pure L2 is strictly better on every
value metric** — bias −0.03, RMSE 1.93, season points 4,313 — but it dips 0.057 MAE (1.8%
relative) below Quant on the single `ownership_band=20%+` slice, a highly-owned population whose
gameweek outcomes are near-random and where Quant's own MAE is already its worst (~3.1). The
pre-committed rule (line 349; R11) is no-ship on *any* per-slice regression, which forces the
choice toward Huber. If the plan owner judges that one narrow, noise-dominated slice an
acceptable carve-out, pure L2 is the better model — this is exactly the "conditional ship with a
named carve-out" question §9.1 has flagged from the start. Huber δ=4 is what ships **under the
rule as written**; it is close to L2 on value and strictly better than the L1 model that §8b
originally proposed.

The human who owns
`docs/plans/2026-08_retrospective_validation_and_ml_decision_plan.md` still makes the actual
production-integration call and sets the shadow-period length.

**History (2026-08-28 → 08-30):** under an **L2** objective `quant_lightgbm` was worse than Quant
on `ownership_band=20%+` in 5/5 cloud seeds — a real per-slice regression → **NO-SHIP** under the
pre-committed rule. §8b then over-corrected: switching to **L1** removed the regression but §8c
shows L1 minimised MAE by fitting the biased conditional median (bias −0.48, worse ranking, worse
season points than L2). The shipped answer, **Huber δ=4**, is the frontier between the two.

### 9.1 The residual signal, and the one open tension

The aggregate case was never weak — 33–35% MAE improvement per season, CI clear of zero, 87.6%
win rate on the highest-disagreement rows, ML beating Quant on the manager-points proxy in every
run. §8b makes the per-slice picture match: the model is now better than Quant on every slice.

**Metric tension — addressed (PR #79, merged 2026-08-30):** the L1 objective costs ~90 points in
the greedy-manager season-points proxy versus L2, because L1 optimises the median while the
proxy's captain pick wants the upper tail. Fixed by decoupling the two decisions: a **q90
quantile residual arm** (`LightGBMResidualModel(objective="quantile", alpha=0.9)`) produces
`ml_ceiling = Q(x) + q90_resid`, and the ML manager now picks its **XI on the L1 median** but
its **captain on `ml_ceiling`**. The ceiling column never touches the MAE tables, the CI, the
slicing check, or this decision — season-sim captain pick only. Cloud run `33314060135`:
`ml_manager` recovers to **4157** (from 4086; L2 was 4178) with the MAE improvement and the 0
regressing slices unchanged. `rolling_defcon` features contribute a further ~25 points and are in
the ship model. Both are on `master`.

## 10. Reproducibility

- Git commit: `45514aecb8cd4652d69198921f5813288d1e77c9`
- Run timestamp (UTC): `2026-08-29T08:11:14Z`
- Walk-forward source: committed `backtest_run_id=1` (`scripts/run_backtest.py`, 71 gameweek steps)
- Dataset: 52,950 rows, seasons `['2024-2025', '2025-2026']`, gameweek-fold mode, 70 walk-forward
  folds (first test step `2024-2025 GW3`, last `2025-2026 GW38`), 52,323 test-row evaluations
- Skip log (DGW / no-deadline / no-ep-outputs steps): empty — no steps were skipped in this run
- `sklearn_available: true`, `lightgbm_available: true`, `xgboost_available: true` — all three
  optional arms (`quant_gbm`, `quant_lightgbm`, `quant_xgboost`) ran for real; this is not a
  degraded/partial run (R14's fallback path was not exercised)
- `random_seed: 42`
- Reproduce: `python -m research.ml.experiment` (run from repo root)

**Cloud re-runs (2026-08-30), §8a:**
- Baseline: Actions run `33281320762`, branch `claude/cloud-ml-pipeline` (`0efa684`), fresh
  walk-forward via `scripts/run_walkforward.py`. `quant_lightgbm` MAE improvement 0.517
  CI [0.502, 0.530]; season points ml 4356 / quant 3750.
- 5-seed sweep: Actions run `33281374600`, seeds 42–46, artifact `experiment_runs.csv` +
  `runs/run_000{0..4}/`. All five: `ml_beats_quant = true`; `ownership_band=20%+` regresses for
  `quant_lightgbm` in 5/5.
- Reproduce on cloud: `gh workflow run ml_experiment.yml --ref <branch> [-f runs=N]`
  (`.github/workflows/ml_experiment.yml`, PR #73).

**Ship model (2026-08-30), §8c — the current recommendation (all on `master`):**
- Iteration loop: `scratchpad/ITERATION_LOG.md`. Reference runs: L2 baseline `33303614405`
  (commit `a941953`), L1 ship `33313576174` (5-seed).
- §8b (L1) runs: `33309139236` / `33309669402` (LightGBM-only) → `33311976548` (all arms) →
  `33312534544` (+ `rolling_defcon`). Merged as PRs #78/#79.
- §8c objective sweep: `33328436934` (Huber δ2), `33329089778` (Huber δ4), `33329778704` (pure
  L2 control). PRs #80/#81; #82 fixed the LightGBM alpha plumbing (see §8c fix note);
  `33332070357` re-confirms the shipped Huber δ4 numbers.
- Shipped `experiment.py`: LightGBM point forecast `objective="huber", alpha=4.0`;
  `quant_xgboost` `objective="reg:pseudohubererror", huber_slope=4.0` (matched); `quant_gbm`
  unchanged (`loss="absolute_error"`); `rolling_defcon_{3,5,10}` features; q90 quantile
  captain-ceiling arm (§9.1). Class objective defaults all stay L2. Full `research/ml/tests/` green.
- Compare-vs-baselines helper: `scratchpad/compare_run.py <run_id> <label>` (MAE / RMSE / bias /
  rank corr / per-season season points / per-slice / last-10-folds held-out check).
- Reproduce: `python -m research.ml.experiment` on `master`, or `gh workflow run
  ml_experiment.yml --ref master -f runs=5`.

## 11. Absolute rule — leakage

Every feature in this experiment was verified asof-safe by the programmatic checks in
`research/ml/leakage_checks.py`, which abort the build on any violation. No realised-outcome
column (minutes, goals, assists, bps, expected_goals, total_points, …) may appear in the feature
matrix; the label never appears as a feature; train and test never share a season; the prediction
timestamp strictly precedes the first kickoff. See [LEAKAGE_PROTOCOL.md](./LEAKAGE_PROTOCOL.md).
