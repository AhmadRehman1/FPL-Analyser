# 2026-09 evidence-gated audit: price-axis calibration, captain risk, headline/benchmark

Repository inspection + four parallel read-only agents (forecast/calibration, optimiser/captain,
backtest/benchmark, data/leakage/tests), run 2026-09-15/16 against branch
`claude/wire-squad-optimizer-recalibratable-params-into-backtest` at `00ff3ac` (merged from
`origin/master`; working tree clean apart from pre-existing untracked scratch files `db/`,
`docs/solver_swap_scope.md`, `scratchpad/`, `scripts/benchmark_m5_solver.py`). Mission: reproduce,
falsify or confirm three specific premises before changing any behaviour --
(1) price-axis miscalibration in the squad optimiser's EP inputs, (2) captain-variance
over-penalisation making elite captains objective-negative, (3) the headline walk-forward metric
being a fresh-solve oracle compared against a synthetic benchmark rather than a legal stateful
season. See each phase's own module docstrings/comments for direct pointers back here.

**Process note**: one audit agent ran `git checkout master && git pull` mid-run despite an
explicit read-only/no-branch-change brief. Caught via cross-check, confirmed zero damage (feature
branch untouched, no stray commits, all untracked files intact), branch restored. The one
detectable consequence was a workflow-inventory gap in that agent's own report (missing the one
workflow file that differs between branches); noted and corrected inline, no other findings
affected (every file the affected audits cite is identical between the two branches).

## Claim-by-claim verdicts

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| 1 | Price-axis miscalibration in EP feeding the optimiser | **CONFIRMED in last real data, UNRESOLVED post-fix** | Recomputed directly from raw `ep_outputs`/`fact_player_season_stats` rows (not the pre-aggregated `backtest_metrics`): mean residual -0.76 (<£5.0m) -> -0.30 (£5-7m) -> +0.71 (£7-9m) -> +0.58 (£9.0m+) across 15,779 player-gameweeks, `backtest_run_id=1` (the Aug-29 backup DB). That run **predates** two calibration fixes shipped 2026-09-05..09 (the "Lead A/B" fixes referenced in `docs/plans/2026-09_ep_attacker_defender_imbalance.md`). No walk-forward has been re-run since; the live local DB (`db/fpl_quant_v2.duckdb`) is completely empty (schema-only), and the committed `data/dashboard/app_track_record.json` got a fresher `generated_at` timestamp without a fresh underlying run (still `backtest_run_id=1`, `n_gameweek_steps=71`). Price only enters `squad_optimizer.py`'s MIQP as a budget constraint (`sum(price*x) <= BUDGET`), never the objective -- so any price-band EP bias directly skews the EP-per-budget-unit signal the optimiser is implicitly maximising, independent of intentional value-for-money logic. |
| 2 | Captain variance over-penalised, elite captains objective-negative | **REJECTED** | `squad_optimizer.py`'s MIQP risk term is exactly `Var(sum_i w_i*r_i)` with `w_i = xi_i + captain_i` -- the mathematically correct variance of the manager's real doubled-captain total, scaled by the *same* single `lambda` as the rest of squad risk. No separate captain-risk coefficient exists. Verified numerically: the closed-form `_captain_objective_component()` helper reproduces the real SCIP-solved objective exactly (diff 0.0 to 1e-5), and in a synthetic 180-player pool the single highest-EP player in the whole pool was the solver's own chosen captain (captain-objective-**positive**). A real `kappa_tc` risk-aversion coefficient exists but is a structurally separate mechanism (Triple Captain *chip* evaluation in `transfer_planner.py`), never touching weekly in-XI captaincy -- the likely source of the original suspicion is a naming collision between `risk_posture.py`'s `kappa_tc` and the unrelated `risk_posture_params` DB table `squad_optimizer.py` reads for its EO/ownership term. |
| 3 | Headline = fresh-solve oracle vs. synthetic benchmark, not a legal stateful season | **CONFIRMED, with an accidental mitigation** | `backtest.run()` genuinely re-solves `squad_optimizer` fresh (full £100m budget, no bank/transfers/hits/chip/captaincy state) every gameweek -- confirmed by the module's own comment and by `squad_optimizer.run()`'s signature taking no held-squad args. Its "crowd" benchmark (`backtest._avg_manager_benchmark_points()`) is explicitly a synthetic, EO-weighted proxy, not FPL's real `average_entry_score` (own docstring says so). **But** a real, tested, stateful `backtest.run_season_simulation()` also exists (bootstraps a squad, then calls real `transfer_planner.run()`/`apply_recommendation()` every week, with `manager_state_versions` genuinely carrying bank/FT/chips/captaincy forward), and a real live tracker (`model_team.py`) compares the model's actual 2026-27 team against FPL's real `bootstrap-static events[].average_entry_score`. The public `track-record.html` page never actually rendered the oracle headline in any of 67 committed revisions of `app_track_record.json`, purely by accident -- see the bug below -- so in practice what users saw was the real model-team card, not the misleading oracle number. |

## Bonus findings (not in the original three premises)

- **No leakage found.** `backtest.asof_scope()` (connection-scoped `TEMP TABLE` shadowing) correctly covers the price/EO/team-strength/minutes/EP query paths during a walk-forward step, with a regression test (`tests/test_backtest.py::test_bootstrap_bank_composes_with_asof_scope_never_leaking_a_later_price`) that demonstrates the counterfactual leak really happens when unshadowed. The real official `average_entry_score` is used *only* for live current-season tracking (`model_team.py`), never fetched to backfill a historical gameweek. Zero `_ingested_at`-based (wall-clock) filtering remains anywhere in `src/fpl_quant` (the two/three known-leaky functions this README already documented were confirmed deleted, nothing new crept back in).
- **No epistemic uncertainty exists anywhere** (only Poisson/Bernoulli *outcome* variance of a single fitted rate) -- confirms the mission's Phase 7 is a real, currently-unaddressed gap, and correctly the last phase in the mission's own ordering (gate on calibration/composition fixes first).
- **Recalibration coverage is much lower than the README/memory's stale "65 of 71" claim.** Live number as of this audit: 8 of 62 active parameters have ever been through `recalibrate()` (`data/dashboard/app_track_record.json`'s own `parameters_total`/`parameters_backtested`). The active-parameter universe itself shrank from 71 to 62 since that figure was last true; both numbers should be treated as live, not memorised.
- **Artefact versioning is fragmented.** `scripts/record_provenance.py` captures a real git-SHA/config manifest, but into a *separate* file never cross-referenced from `backtest_runs`, `recalibration_proposals`, or `rank_autopsy.json`. No `config_hash` field exists anywhere. `monte_carlo_run_versions.seed` is the one place a real deterministic seed is actually persisted.
- **Full test suite is green.** 1084/1084 Python (`pytest tests/ -q`, ~19min uncontended locally, corroborated by the last 10 real `tests.yml` CI runs: 8 success / 2 superseded-cancelled / 0 failures), 141/141 Node (`npm test`), ruff clean.

## Phase 3 (captain risk separation) -- explicitly skipped

Per the mission's own instruction ("if a phase is disproven, commit only useful diagnostics and
document why the behavioural change was skipped"): claim 2 above is rejected by direct numerical
verification against the real solver, so no `lambda_captain` role-separation code was written.
What shipped instead is the diagnostic that proved this (`tests/test_captain_objective_diagnostics.py`)
plus a real, small robustness fix it found along the way -- see below.

## What shipped this pass

All local commits only; nothing pushed, no PR opened, no default/live configuration changed,
per the mission's explicit gate ("pause before the first live/default configuration change").

1. **`src/fpl_quant/calibration_diagnostics.py`** (new) -- Phase 1A/1B diagnostics, always
   recomputed from raw `ep_outputs`/`fact_player_season_stats` rows, never from
   `backtest_metrics`' pre-aggregated (unweighted, no-CI) segment means:
   - `price_band_calibration_report()`: overall + price-band + position + position-x-band +
     five-component decomposition, with GW-clustered bootstrap 95% CIs (not a naive row-level
     bootstrap -- resamples whole gameweeks, respecting within-GW fixture-shock correlation) and
     explicit missing-data counts. Validated against the real Aug-29 backup DB: reproduces
     Audit A's independently hand-computed numbers exactly (n=15,779; band means/MAE/RMSE match
     to 4 decimal places).
   - `selection_curse_report()`: the Phase 1B "optimiser curse" diagnostic (`sel_resid`,
     `band_resid`, `curse = band_resid - sel_resid`), with all four required robustness variants
     (price-band vs. position-price-matched pool, XI vs. full-15-squad selection), sum-scale
     equivalents, and GW-clustered CIs. Sign convention unit-tested on a deterministic fixture
     per the mission's explicit instruction. **First-ever real measurement for this project**:
     run against the Aug-29 backup, the XI/price-band-matched curse is mean **-0.21**
     (median -0.28, 95% CI [-0.66, +0.27], 40% of 20 gameweeks positive) -- i.e. **no evidence of
     a real optimiser curse** on that (stale) run; the selected XI does not systematically
     underperform price-band-matched peers.
   - `scripts/run_calibration_diagnostics.py` persists both as versioned JSON (run ID, git SHA,
     seed, tier, gameweek range, generated_at) to `data/calibration_diagnostics/` -- not run for
     real this pass (the only available data is the stale pre-fix backup; the user explicitly
     chose "build tooling now, get fresh data separately" rather than commit a result that would
     look canonical but isn't from current code).
   - `tests/test_calibration_diagnostics.py`: 20 tests, all passing.
   - Consolidated the price-band boundary definition (`backtest._price_band()`, now NaN-safe) as
     the single source of truth; `research/ml/baselines.py` now imports it instead of maintaining
     an independent copy that had silently drifted on NaN handling.

2. **`tests/test_captain_objective_diagnostics.py`** (new, Phase 1C) -- 7 tests: hand-computed
   decomposition (including Audit B's own cross-covariance toy check), zero-lambda reduction to
   pure EP, real-SCIP-solve agreement on a synthetic pool with real cross-covariance, solver's
   own captain choice reproduced by the closed-form argmax, and a meta-test proving the agreement
   check has power (a deliberately wrong lambda genuinely fails it). **Found and fixed a real
   (if previously harmless) bug**: `squad_optimizer._captain_objective_component()`'s linear-EP
   term summed `mu_by_uid.values()` directly instead of iterating `xi_uids` explicitly (unlike
   its own risk term, which already did this correctly) -- silently wrong if ever called with a
   mu/var dict covering more than the fixed XI. Never wrong for the one existing caller
   (`captain_choice_with_differential`, which always pre-scopes its dict to the XI), but a real
   footgun for any diagnostic tool with a full candidate pool lying around -- exactly what this
   audit's own numeric verification needed. Fixed to sum over `xi_uids`; full existing
   `tests/test_squad_optimizer.py` suite (59 tests) still green.

3. **`scripts/export_track_record.py` + `scripts/run_report.py` + `src/fpl_quant/reporting.py`**
   (Phase 1D/2, bug fix) -- the real bug behind claim 3's "accidental mitigation": the nightly
   `export_track_record.py` job (the only writer of a real `backtest_run_id`) wrote the real
   headline only at the JSON's top level, never into `transparency_log.backtest` --
   `track-record.html` reads *exclusively* `transparency_log.backtest.headline`. Confirmed in
   every one of 67 committed revisions of `app_track_record.json`: real top-level `headline`,
   `null` nested one. Once corrupted, the twice-daily `run_report.py`'s own merge-preservation
   guard made it permanent: `if committed_bt:` treats a dict with `headline: None` as truthy, so
   it "preserved" the corruption on every subsequent run instead of ever fixing it.
   - Extracted `reporting.backtest_transparency_section()` (pure, reused by both the writer and
     the reader) and `run_report.py`'s `_merge_track_record_onto_committed()` (pure, was
     previously inline in `main()` and untested beyond the upstream boolean guard).
   - `export_track_record.py` now populates `transparency_log` for real, closing the root cause
     for every future nightly run.
   - `run_report.py`'s merge now self-heals: when the committed `transparency_log.backtest` is
     missing or has a null `headline`, it reconstructs from `existing_track_record`'s own real
     top-level fields instead of blindly preserving the corrupted copy. **Verified against the
     real currently-committed file** (read-only, nothing written): the self-heal correctly
     recovers the real headline (`beats_avg_manager_by_points_per_gw: -0.9444`, 70 scored
     gameweeks) that was stuck at `null` in every prior commit.
   - `tests/test_run_report.py`: 3 new tests (correctly-populated preserve, corrupted self-heal,
     honest-None-stays-None), extending the file's existing pure/DB-free testing style.
   - `tests/test_reporting.py`: extended the existing headline test with the new provenance field.

4. **Benchmark provenance metadata** (Phase 1D) -- `backtest.synthetic_crowd_benchmark_provenance()`
   (new, pure, additive) attaches `{benchmark_name, source, endpoint_or_artefact, as_of,
   gross_or_net_of_hits, stateful, oracle}` to `reporting._backtest_headline()`'s output. Changes
   no existing metric value; purely disclosive.

5. **`track-record.html` copy relabel** (Phase 1D) -- the walk-forward headline card used to read
   "Does the model beat the average manager?" / "the average FPL manager's score", identical
   phrasing to the *different*, genuinely-real comparison lower on the same page ("The model's
   own team" ... "vs the average manager", which uses FPL's actual `average_entry_score`). Now:
   "Does a fresh model squad beat a synthetic crowd benchmark?", explicit "not FPL's official
   average score" / "no bank, transfers, or hits carry between steps here" copy, KPI labels
   changed to "synthetic crowd benchmark", and a cross-reference to the real section. The real
   section's own copy was tightened the other way ("FPL's official average manager") for the same
   disambiguation. No JS test covered this text (checked); no test broken.

## What deliberately was NOT done this pass

- No fresh walk-forward run (user's explicit choice: validate tooling against the existing stale
  backup, get fresh post-fix numbers as a separate, later step -- a real walk-forward is an
  expensive, multi-hour job per the mission's own run-ladder gating).
- No promotion of `run_season_simulation()`/`beats_baseline()` to the default historical headline
  (currently only wired for the live 2026-27 season via `weekly_backtest.yml`'s
  `export_leaderboard.py`; wiring it for the two historical seasons is a bigger, separately-scoped
  lift the user did not ask for this pass).
- No Phase 4 (price-conditioned shrinkage anchors) -- gated on knowing whether claim 1 still holds
  post-fix, which requires the fresh walk-forward above.
- No Phase 5/6/7 (recalibration-loss redesign, active variance, epistemic uncertainty) -- all
  correctly gated behind the above per the mission's own phase ordering.
- Nothing pushed, no PR opened, no branch merged, no default parameter version changed.

## Test/lint status after all changes

- `pytest tests/test_calibration_diagnostics.py tests/test_captain_objective_diagnostics.py
  tests/test_run_report.py tests/test_reporting.py -q`: all green (129 tests across the four
  files touched/added this pass).
- `pytest tests/test_squad_optimizer.py -q`: 59/59 green (regression check on the
  `_captain_objective_component()` fix).
- `npm test`: 141/141 green (track-record.html copy change has no JS coverage; confirmed no
  existing test references the changed strings).
- `ruff check`: clean on every touched file.
- Full `pytest tests/ -q` (all ~1084 tests) not re-run in full this pass after the Phase 1D
  changes specifically (takes ~19min uncontended); the four directly-relevant suites above were
  run in full instead. Recommended before any push.
