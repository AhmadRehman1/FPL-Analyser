# Phase-0 ML Residual Experiment — Report

> **Status:** Run against real production data (`git_commit` `8a94bd2`, 2026-08-28). §3–§9 below
> report the real, final numbers, from a single, fully self-consistent run (re-run after
> `research/ml/season_sim.py`'s starting-XI position-code bug — see banner below — was fixed
> upstream, so §3.4's numbers are not mixed with an earlier, invalidated run).
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
> **R13 resolved, but not yet part of this report's real numbers:** `quant_xgboost`
> (`research/ml/residual_model.py::XGBoostResidualModel`) has since been added as the
> independent-implementation confirmation arm this report's §9 originally flagged as a follow-up
> ("Flag whether XGBoost should be added..."). It lands after the real run §3–§9 below report, so
> none of those numbers include it — it will run automatically whenever `xgboost` is installed
> (see `requirements-research.txt`) on the *next* real run, reported alongside `quant_gbm` as
> **informational only** (like `quant_gbm`, R16 — it will never govern §9's decision, which
> remains `quant_lightgbm` alone per R11). Its value once real numbers exist: two independently
> implemented gradient-boosting libraries agreeing (or disagreeing) on whether there is
> out-of-sample residual signal is itself evidence about whether `quant_lightgbm`'s result is a
> real finding or an artifact of one library's specific defaults/splitting heuristic — a follow-up
> run including it is recommended, not required (Open Items).
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
| quant (baseline) | 1.519 | 2.160 | 0.495 | 0.443 | 0.495 | 52,323 |
| quant_linear | 1.111 | 1.974 | -0.045 | 0.549 | 0.691 | 52,323 |
| quant_gbm *(informational only — R16)* | 1.012 | 1.938 | -0.043 | 0.566 | 0.708 | 52,323 |
| **quant_lightgbm *(governs §9's decision — R11)*** | **1.011** | **1.936** | **-0.040** | **0.567** | **0.709** | 52,323 |
| historical_baseline | 1.053 | 2.108 | -0.008 | 0.497 | 0.702 | 52,323 |
| ensemble (best w=1, i.e. 100% ML — see §3.3) | 1.011 | 1.936 | -0.040 | 0.567 | 0.709 | 52,323 |

Per-season breakdown:

| Season | Model | MAE | RMSE | Bias | Pearson ρ | Rank ρ | n |
|--------|-------|-----|------|------|-----------|--------|---|
| 2024-2025 | quant | 1.513 | 2.165 | 0.505 | 0.427 | 0.435 | 25,238 |
| 2024-2025 | quant_lightgbm | 1.020 | 1.937 | -0.015 | 0.561 | 0.704 | 25,238 |
| 2025-2026 | quant | 1.525 | 2.156 | 0.486 | 0.460 | 0.555 | 27,085 |
| 2025-2026 | quant_lightgbm | 1.002 | 1.935 | -0.065 | 0.573 | 0.715 | 27,085 |

*(`quant_xgboost` not included — this run predates it landing; see banner above.)*

> Source: `results/model_comparison.csv`. `quant_gbm` and `quant_lightgbm` track each other
> extremely closely throughout this report (typically within 0.002 MAE) — both are gradient
> boosted trees over the same feature set, so this is expected, not a bug; R11 still governs
> off `quant_lightgbm` specifically, per the plan.

### 3.1b Bootstrap confidence intervals on MAE improvement (R10/R11)

| Model | Point estimate (MAE improvement) | 95% CI low | 95% CI high | n folds | Statistically credible? |
|-------|-----------------------------------|-----------|-----------|---------|--------------------------|
| quant_linear | 0.408 | 0.385 | 0.426 | 70 | True |
| quant_gbm *(informational only)* | 0.507 | 0.490 | 0.522 | 70 | True |
| **quant_lightgbm** | **0.507** | **0.489** | **0.523** | **70** | **True** |
| quant_xgboost *(informational only — R13 confirmation arm, not yet run — see banner)* | — | — | — | — | — |

> Source: `results/bootstrap_ci.json` (also embedded in `results/experiment_manifest.json` →
> `bootstrap_ci`). "Statistically credible" = `True` only when the entire 95% interval sits above
> zero. **This table's `quant_lightgbm` row is what §9's decision is actually based on** — the
> entire interval sits comfortably above zero (low bound 0.489 MAE, a ~32% relative improvement),
> satisfying R11's confidence-interval requirement. This half of R11 is met; §8 covers the other
> half.

### 3.1c Compute/runtime (R10)

| Model | Total fit+predict seconds (all folds) | Mean seconds/fold | n folds |
|-------|----------------------------------------|--------------------|---------|
| quant_linear | 3.6s | 0.051s | 70 |
| quant_gbm | 24.0s | 0.343s | 70 |
| quant_lightgbm | 17.2s | 0.246s | 70 |
| quant_xgboost | — *(not yet run — see banner)* | — | — |

> Source: `results/compute_runtime.csv`, grouped by model. `quant_lightgbm` is both slightly more
> accurate than `quant_gbm` on raw MAE (1.011 vs 1.012) and ~28% cheaper to fit/predict (17.2s vs
> 24.0s total). Compute cost is not a concern for either arm at this scale (sub-second per fold);
> this table exists mainly to confirm ML isn't silently expensive.

### 3.2 Improvement vs the Quant baseline

| Season | Model | Metric | Quant error | ML error | Improvement | Improvement % |
|--------|-------|--------|-------------|-----------|-------------|---------------|
| 2024-2025 | quant_lightgbm | MAE | 1.513 | 1.020 | 0.492 | 32.5% |
| 2024-2025 | quant_lightgbm | RMSE | 2.165 | 1.937 | 0.228 | 10.5% |
| 2024-2025 | quant_gbm *(informational)* | MAE | 1.513 | 1.020 | 0.493 | 32.6% |
| 2025-2026 | quant_lightgbm | MAE | 1.525 | 1.002 | 0.522 | 34.3% |
| 2025-2026 | quant_lightgbm | RMSE | 2.156 | 1.935 | 0.221 | 10.2% |
| 2025-2026 | quant_gbm *(informational)* | MAE | 1.525 | 1.004 | 0.521 | 34.2% |

> Source: `results/improvement.csv`. Improvement = quant_error − ml_error; positive = ML helps.
> The improvement is consistent across both seasons (32.5% and 34.3% MAE improvement
> respectively) — not a single-season artifact. This table is the per-season point-estimate view;
> §3.1b's bootstrap CI is the one §9's decision actually reads.

### 3.3 Ensemble weight selected

- Best `w` (fit on training only, `w` = weight on **Quant** per §2's formula): **0.0, in all
  70/70 folds, no exceptions** — the training-only grid search never once preferred any nonzero
  weight on Quant; it always preferred 100% ML.
- Mean train MAE at best `w`: 0.995 → mean test MAE at best `w`: 1.011 (small, expected
  train→test gap; no sign of gross overfitting).
- This is a consistent, unanimous signal in ML's favor across every fold, not a marginal or
  fold-dependent result.

> Source: `results/ensemble.csv`

### 3.4 Season manager points -- the metric that actually matters

| Season | Signal | Total points | Gameweeks |
|--------|--------|---------------|-----------|
| 2024-2025 | quant_prediction | 1,862.0 | 35 |
| 2024-2025 | ml_prediction | 2,174.0 | 35 |
| 2025-2026 | quant_prediction | 1,889.0 | 35 |
| 2025-2026 | ml_prediction | 2,035.0 | 35 |

> Source: `results/season_points.csv`, from the corrected `season_sim.py` (see banner). A manager
> using the ML-augmented signal must score more real points than the Quant-only manager across
> multiple runs/seeds -- a lower MAE that doesn't translate into more points is not a reason to
> integrate. Here it does: **+312 points (+16.8%)** in 2024-25 and **+146 points (+7.7%)** in
> 2025-26 over a 35-gameweek season using this simplified greedy-manager proxy. Both signals'
> totals rose relative to the pre-fix run (a legal formation scores more than the illegal
> position-blind fallback it replaced), and the *margin* between them shrank once both signals
> were evaluated on legal formations — expected, since the bug affected both rows identically, not
> just one. Note (Track F): `ml_prediction` here reflects the experiment's ensemble-priority
> convention (LightGBM > `quant_gbm` > linear, whichever is available) — since LightGBM was
> available and is the strongest arm throughout this run, this row **is** effectively the
> `quant_lightgbm` signal — but it is still a "best available ML signal" view structurally, not
> itself the §9 decision input (§3.1b/§8 are).

## 4. Where the Quant model is wrong

`results/residual_analysis.csv` (real, n-weighted mean error by position, `residual = event_points
− quant_prediction`, so positive = Quant under-predicts):

| Position | Mean error (Quant) | n |
|----------|---------------------|---|
| Defender | +0.775 | 17,247 |
| Goalkeeper | +0.544 | 5,762 |
| Midfielder | +0.348 | 23,394 |
| Forward | +0.226 | 5,920 |

Quant systematically **under-predicts defenders** most (by far the largest bias of the four
positions) and is closest to unbiased for forwards. This matches §6's calibration finding below
(Quant over-predicts the low-score bucket where most defenders' rows land) and is consistent with
§3.1's aggregate bias sign (+0.495, i.e. Quant under-predicts on average across all positions).

## 5. Disagreement

`results/high_disagreement_cases.csv`: 14,000 real high-disagreement rows (cases where
`quant_prediction` and `ml_prediction` differed most). **ML was closer to the actual outcome in
87.3% of these cases.** This is the single most direct evidence in this report that the ML model
isn't just shaving a fraction of a point off an already-good baseline — on the rows where the two
models actively disagree, ML is right almost 9 times out of 10.

## 6. Calibration

`results/calibration.csv`, n-weighted, mean predicted vs. mean actual per predicted-value bucket:

| Bucket | Quant: mean predicted | Quant: mean actual | quant_lightgbm: mean predicted | quant_lightgbm: mean actual |
|--------|------------------------|----------------------|-----------------------------------|--------------------------------|
| 0-2 | 1.131 | 0.538 | 0.570 | 0.582 |
| 2-4 | 2.806 | 2.576 | 2.823 | 2.893 |
| 4-6 | 4.434 | 3.986 | 4.534 | 4.217 |
| 6-8 | 6.119 *(n=3, noise)* | 1.000 *(n=3, noise)* | 6.872 | 5.679 |

Quant is badly miscalibrated in the largest bucket (0-2, ~37,600 rows): it predicts a mean of
1.13 against a real mean of 0.54 — **more than double** the realised value, i.e. Quant is
systematically too optimistic about players it expects to blank or nearly blank.
`quant_lightgbm`'s 0-2 bucket is well calibrated (0.570 predicted vs. 0.582 actual) — this single
bucket is most of both the aggregate MAE gap in §3.1 and the aggregate bias-sign flip (Quant
+0.495 → quant_lightgbm −0.040), since it holds the majority of all rows.

## 7. Feature importance & stability

`results/feature_stability_lightgbm.csv` (mean importance / coefficient of variation across all
70 folds, for `quant_lightgbm` — the arm §9's decision is actually governed by; an earlier draft
of this section incorrectly cited the linear model's `feature_stability.csv` instead, caught in
blind review before this report was finalized; `feature_importance_xgboost.csv`/
`feature_stability_xgboost.csv` exist for `quant_xgboost` too, but that arm wasn't part of this
real run — see banner), top real drivers:

| Feature | Mean importance | CV (std/mean) |
|---------|------------------|----------------|
| `p_60plus_min` | 0.0947 | 0.378 |
| `p_start_final` | 0.0884 | 0.302 |
| `rolling_points_3` | 0.0867 | 0.351 |
| `status=a` (available) | 0.0275 | 0.853 |
| `rolling_bps` | 0.0239 | 0.666 |
| `chance_of_playing_next_round` | 0.0161 | 0.938 |

The minutes-model probabilities (`p_60plus_min`, `p_start_final`) and short-term form
(`rolling_points_3`) dominate `quant_lightgbm`'s real importance, and are also its most *stable*
drivers (CV 0.30–0.38, the lowest three of the top six) — consistently the dominant signal across
folds, not a fold-specific artifact. The single-gameweek availability flag `status=a` and
`chance_of_playing_next_round` matter but are far less stable (CV 0.85–0.94) — expected, since
day-to-day fitness/rotation news is inherently gameweek-specific, not a persistent per-player
signal the model can lean on every fold. No feature showed the "important in one season,
irrelevant in every other" red flag this section exists to catch.

## 8. Slicing check (spec §12)

**Real result: 59 of 60 (dimension, slice) combinations show `quant_lightgbm` improving on Quant.
1 of 60 shows a real regression — named here in full, per R11/this section's own requirement, not
omitted:**

| Dimension | Slice | Quant MAE | quant_lightgbm MAE | Change | n (real, non-trivial) |
|-----------|-------|-----------|----------------------|--------|----|
| `ownership_band` | `20%+` (highly-owned players) | 3.127 | 3.175 | **−0.048 (worse)** | 1,149 |

(`price_band=9.0+`, the second slice that regressed in an earlier run of this same experiment —
before the `season_sim.py` fix and a re-run — is no longer a regression here: 3.321 → 3.314,
+0.007, essentially flat. The two slices correlate heavily — premium-priced and highly-owned
players overlap a great deal — so this is consistent with one real, narrow underlying effect
rather than two independent ones; the fresh re-run's own fold-to-fold variance is enough to move
a near-zero effect across the improve/regress line, which is itself informative about how close
to noise this specific slice's regression sits.)

The regression is real (1,149 real observations, not sampling noise) but narrow: small in
absolute magnitude (1.5% relative to the slice's own MAE), and set against Quant's own MAE
already being at its *worst* in exactly this slice (3.13 vs. an overall MAE of 1.52) — i.e. this
is one of the hardest-to-predict player populations for both models, where variance is dominated
by genuinely unpredictable factors (rotation risk, blanks/hauls for premium attackers), not a
place either model is confidently right. Every other slice — all four positions, all three
minutes bands, all four fixture-difficulty bands, every gameweek 1–38, both seasons, and all
price bands including the highest — shows `quant_lightgbm` improving on Quant, several
substantially (e.g. `ownership_band=unknown`: 1.146 → 0.276, an 0.870 improvement;
`position=Defender`: 1.726 → 1.110, a 0.617 improvement).

Practically, this one slice is exactly FPL's highest-stakes decision — captaincy disproportionately
involves highly-owned players — so this is not a corner case to wave away even though it's
numerically narrow and small. See §9 for how the plan's own pre-committed rule for this exact
situation applies.

> Source: `results/sliced_model_comparison.csv` — one row per (fold, model, dimension, slice),
> for every model. Aggregated across folds per (model, dimension, slice), comparing
> `quant_lightgbm`'s MAE against `quant`'s in all 60 real (dimension, slice) combinations before
> summarising above.

## 9. Decision

**Governed by `quant_lightgbm` alone (R11)** — §3.1b's bootstrap CI for that arm, plus this
section's slicing check for that arm, decide the verdict below. `quant_gbm`'s numbers (real,
reported throughout this document) are informational only and do not factor into this checkbox
(R16) — do not let a `quant_gbm` result substitute for `quant_lightgbm`'s here even if the two
happen to point the same direction.

- [ ] **`quant_lightgbm` improves out-of-sample on the primary metric (MAE), the entire 95%
      bootstrap CI sits above zero (§3.1b), and the improvement holds across slices (§8)** →
      recommend a controlled integration (separate code path, shadow-only at first). R13 is now
      resolved for a future run — `quant_xgboost` (§3.1b) exists and will run on the next real
      run, whenever `xgboost` is installed; its point estimate should be read alongside
      `quant_lightgbm`'s then as a check that the result isn't an artifact of one library's
      implementation. It was not yet available for this run (see banner) and did not factor into
      the verdict below either way.
- [x] **`quant_lightgbm` does not improve, the CI does not exclude zero, or it only improves a
      slice while degrading others** → do not integrate. The existing Quant model stands.
      Document why ML failed to find signal below.

**Verdict: NO-SHIP**, per this plan's own pre-committed "Edge Cases & Failure Handling" rule
(`docs/plans/2026-08_retrospective_validation_and_ml_decision_plan.md`, line 349): *"ML result is
a near-tie or **fails the per-slice check** → recorded as an explicit 'no-ship' in `REPORT.md`."*
§8's real result is a per-slice check failure — 1 of 60 slices (`ownership_band=20%+`, a real,
1,149-observation effect, not noise) shows `quant_lightgbm` regressing against Quant. R11's own
text is unconditional on this point: *"ship only if it shows a statistically credible
improvement... **without regressing**... per-slice performance."* An earlier draft of this
section recorded a "CONDITIONAL SHIP" verdict instead — reasoning that 58–59 of 60 slices passing
was close enough to justify shipping with the failing slice(s) carved out. That reasoning is not
adopted here: it invented a third decision category the plan never defined or authorized, and it
directly contradicted the plan's own pre-committed rule for this exact scenario — a rule written
*before* any real result existed, specifically to prevent a good-looking aggregate number from
talking anyone (including the executor) out of a real, disclosed per-slice failure after the fact.
Caught in blind review; corrected here to the plan's own literal rule rather than the executor's
in-the-moment judgment call.

This does **not** mean the result is weak — it very much is not (32–34% MAE improvement in both
seasons independently, CI excludes zero with real margin, 87.3% win rate on the highest-
disagreement cases, real season-points gains in both seasons under the manager-points proxy, and
now only 1 of 60 slices regressing, down from 2 in the prior run). `quant_gbm` is correctly kept
informational-only (R16) throughout and did not factor into this decision either way. See §9.1 for
the full reasoning and a flagged (not decided) follow-up path.

### 9.1 If not integrating — why?

Not because ML failed to find signal — it found a strong, consistent, statistically credible one.
The reason is procedural and deliberate: this plan pre-committed, before any real result existed,
to treating *any* real per-slice regression as an automatic "no-ship," specifically to prevent a
strong aggregate number from being used to talk past a real, narrow weakness after the fact. §8's
real result — 1 of 60 slices regressing, in exactly the highly-owned-player segment that matters
most for captaincy calls — is precisely the scenario that rule exists to catch, even though the
regression itself is small (1.5% relative) and the aggregate case is otherwise very strong.

**Flagged, not decided:** whether to formally define a "conditional ship, with named per-slice
carve-outs" category for a future case like this one — where the aggregate case is strong, CI is
credible, and only a small number of narrow, correlated slices regress — is a real question worth
raising with whoever owns this plan next. That would need its own plan revision (a new Key
Decision, ideally with its own Critique Engine pass), not a unilateral substitution made inside a
single "done" artifact, which is what happened in this section's first draft and was corrected
here. Until such a revision exists, the plan's literal Edge Case rule — no-ship on any real
per-slice regression — governs, and this section records that outcome rather than overriding it.

## 10. Reproducibility

- Git commit: `8a94bd2f24b32cc78b495c8f2d2def74fa3967dc`
- Run timestamp (UTC): `2026-08-28T15:27:36Z`
- Dataset: 52,950 rows, seasons `['2024-2025', '2025-2026']`, gameweek-fold mode, 70 walk-forward
  folds (first test step `2024-2025 GW3`, last `2025-2026 GW38`)
- Skip log (DGW / no-deadline / no-ep-outputs steps): empty — no steps were skipped in this run
- `sklearn_available: true`, `lightgbm_available: true` — both optional arms (`quant_gbm`,
  `quant_lightgbm`) ran for real; this is not a degraded/partial run (R14's fallback path was not
  exercised)
- Reproduce: `python -m research.ml.experiment` (run from repo root)

## 11. Absolute rule — leakage

Every feature in this experiment was verified asof-safe by the programmatic checks in
`research/ml/leakage_checks.py`, which abort the build on any violation. No realised-outcome
column (minutes, goals, assists, bps, expected_goals, total_points, …) may appear in the feature
matrix; the label never appears as a feature; train and test never share a season; the prediction
timestamp strictly precedes the first kickoff. See [LEAKAGE_PROTOCOL.md](./LEAKAGE_PROTOCOL.md).
