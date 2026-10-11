# Plan: Beat the template, then push toward a top-100k season

One-line goal: the model's picks beat a legal ownership-template squad by a wide margin on a fair,
like-for-like backtest, and the live model team scores at the pace of FPL's actual top-100k
cutoff. Today the oracle model only roughly matches the template, and the live team sits about
level with FPL's average.

## Classification
Track: Performance. The end state is a number: squad points against the template, then season
points against the real top-100k cutoff. Unlike a classic performance job, the projections are
meant to change, so "frozen surfaces" means output formats and the live pipeline, not model
behaviour.

Parked secondary asks:
- Planner and chip re-tuning: Phase 9, only once G2 holds.
- Elite (top-10k) ownership as an input: Out of Scope, with triggers.

## Interview Ledger
- **Q1 crowd input:** ownership and transfer trends MAY feed the projections as a backtested prior
  (user, accepted recommendation).
- **Q2 promotion rule:** "rule R6", points-first (user, accepted recommendation):
  - paired squad points over both seasons with m − SE ≥ 0;
  - EP MAE ≤ control + 0.01 in each season;
  - then a season-simulation check with m ≥ 0 before going live.
- **Q3 finish line:** "beat template by a lot; we need to push to like top 100k overall" (user, free
  form). It is translated into G2–G4 below [A10].

Questions used: 3 of 14, plus the approval question.

## Goal & Success Criteria
- **G1 (fair yardsticks exist).**
  - **Oracle.** Every walk-forward summary reports, per gameweek, the model squad's realized points
    and a legal ownership-template squad's realized points. The template is built under the same
    budget, formation, club cap, captain/vice/bench and auto-sub rules, as a fresh hit-free squad
    each week, like the model's. `compare_arms.py --rule r6` judges arms on paired squad points.
  - **Live.** The live leaderboard gains a "model (oracle)" row, so its oracle template baselines
    are compared like for like (Req 3).
- **G2 (beat the template "by a lot").** On the two-season oracle walk-forward, the model squad
  beats the template squad by m ≥ +2.0 pts/GW with m − SE > 0 [A10: +2.0/GW ≈ 76 pts a season is
  the reading of "by a lot"]. Today's gap is measured in Phase 1. The only prior number is against
  the *synthetic* crowd: 1.15 pts/GW behind, pooled (verified:
  `data/dashboard/app_track_record.json` headline).
- **G3 (top-100k pace, backtest).** The evolving manager (transfers, hits, chips;
  `season_sim_arms.yml`) scores net pts/GW at least the real top-100k cutoff's pace in that season.
  - The run is one continuous squad: a single GW2–38 job, or chained chunks. A fresh squad at GW20
    is not allowed (Req 7).
  - The cutoff is each season's total points at overall rank 100,000 ÷ 38, measured in Phase 1
    from sampled managers' FPL `entry/{id}/history` `past` arrays [A1].
- **G4 (live).** The live model team's cumulative net points (`data/dashboard/app_model_team.json`
  `total_realized_points`, net of hits) reach the 2026-27 top-100k cutoff's cumulative points.
  - The cutoff is estimated each gameweek from sampled managers' `entry/{id}/history` `current`
    arrays near rank 100k (Req 4). It is tracked weekly after each promotion.
  - The model team today: 302 pts over GW1–5, +3 against FPL's average, with Free Hit and Triple
    Captain already used (verified: `app_model_team.json`).
- **Stop rule.**
  - A lever family is dropped after two arms in a row fail rule R6.
  - The programme pauses for owner review at any of these:
    - 10 lever arms have run in total;
    - mean template overlap reaches ≥ 9/11 (Req 9);
    - the backlog is empty with G2 unmet.

## Current State
- **The walk-forward squad is an oracle.** It is a fresh, hit-free squad every gameweek (verified:
  `src/fpl_quant/backtest.py` `real_average_benchmark_provenance()` "oracle": True). Against the
  *synthetic* crowd (real points weighted by ownership; `_avg_manager_benchmark_points`):
  - 2024-25: model 63.54 pts/GW, crowd 61.85;
  - 2025-26: model 60.16, crowd 64.15;
  - pooled: 1.15/GW behind.
  
  (verified: walk-forward control run 38167650902, `headline_by_season`; app_track_record headline)
  The gap to a *legal* template squad has never been measured on the backtest.
- **A legal template squad already exists in code.** `backtest._naive_candidate_pool(signal=
  "ownership")` builds it (mu = the latest `selected_by_percent`, read inside `asof_scope()`). It is
  solved by `squad_optimizer.solve(pool, {}, 0.0, guardrail_cap, captain_variance_multiplier=
  LIVE_CAPTAIN_VARIANCE_MULTIPLIER)` and scored by `_realized_solve_points()` with the solve's own
  vice and bench order. Only `backtest.beats_baseline()` uses it (verified: backtest.py
  `beats_baseline`, `_naive_candidate_pool`, `_realized_solve_points`).
- **The live leaderboard is not like for like.** `scripts/export_leaderboard.py` scores the
  *evolving* `run_season_simulation()` model (gross, real transfers) against *oracle* baselines,
  a fresh squad each week (verified: `export_leaderboard.py` lines 4–24, 148). Over 2026-27 GW2–5:
  - evolving model 241;
  - oracle ownership template 266;
  - "crowd-average baseline" 323.1, which is the synthetic EO crowd, far above FPL's real average
    for those weeks (81 + 51 + 69 + 48 = 249);
  
  (verified: `leaderboard_latest.json`, `app_model_team.json` `field_average`).
- **The live model team is about average.** 302 pts over GW1–5, `total_vs_field` +3, with Free Hit
  and Triple Captain used (verified: `app_model_team.json`).
- **The backtest season sim is far more generous than live.** The honest 2025-26 season-sim control
  was +8.1/GW over FPL's real average (verified: `docs/reports/2026-10_promoted_club_strength.md`,
  "Season sim" bullet). Each default chunk (`2-19`, `20-38`) starts from a fresh squad, which is a
  free wildcard at GW20 (verified: `season_sim_arms.yml` `chunks` input description).
- **The season sim does not repeat across CI jobs.** The same DB and params split at GW3 (Bench
  Boost played vs held): ±1.81/GW. The cause is not confirmed (verified:
  `promoted_club_strength.md`, the "does not repeat across CI jobs" bullet).
- **Ownership and transfers never reach the projections** (verified: no `selected_by_percent` or
  `transfers_in_event` in `expected_points.py` or `minutes_model.py`). The columns exist in
  `fact_player_season_stats` (verified: `schema/0001_core_schema.sql`;
  `schema/0014_priority4_transfer_momentum.sql`; ingested by `reconcile.py`). Historical
  coverage of the transfer columns is unverified [A3].
- **Projection bias by price band, pre-promotion** (both seasons pooled, realized − projected per
  player-GW, control run 38167650902 on 60d3e0e, before the own substitute rate went live,
  verified):

  | band | total | goals | assists | bonus/other |
  |---|---|---|---|---|
  | <£5.0m | −0.158 | | | |
  | £5.0–7.0m | +0.062 | | | |
  | £7.0–9.0m | +0.272 | | | |
  | £9.0m+ | +0.175 | −0.177 | +0.169 | +0.182 |

- **Captaincy, pre-promotion** (same run): 6.42 pts/GW before doubling; the P95 rule +0.39 ± 0.30;
  the best pick in hindsight 12.87.
- **Optimizer settings.**
  - Risk aversion λ is 0 live (verified: `data/recalibration/seeds_lambda_2026-10-01.json`).
  - Guardrails: captain not a goalkeeper (`squad_optimizer.py:395`), ≤3 XI players per club
    (`:405`), and a bench-quality floor (`bench_quality_params`) (verified: `squad_optimizer.py`).
- **The season-sim harness.**
  - `season_sim_arms.yml` takes arms as `"label: flags"`, a seasons input (default 2025-2026) and
    chunks (default `2-19,20-38`).
  - The job timeout is 358 min, of which the run step's limit is 345 (verified: that workflow).
  - `scripts/run_season_sim_arm.py` has its own flag parser and builds an `argparse.Namespace` for
    `run_walkforward._experiment_versions` (verified: that script, lines 44–74). A new lever flag
    therefore has to be added to both scripts.
- **The rank goal "overall rank top 100k" is already declared** in `data/dashboard/rank_autopsy.json`
  (verified). Its `model_optimal` subject is the from-scratch oracle pick, and `estimated_rank` is a
  single-gameweek projection (verified: `scripts/rank_autopsy.py` docstring; `rank_autopsy.json`
  limitations). `fact_rival_squad_sample` stores picks, captain and multiplier per sampled entry;
  it has no points, hits or history (verified: `schema/0017_priority10_phaseA_rival_squad_sample.sql`).
- **`branch_walkforward.yml` cancels duplicates.** It uses concurrency group
  `branch-walkforward-${ref}-${args}` with `cancel-in-progress: true`. Two dispatches with the same
  args on the same ref cancel the first (verified: that workflow).
- **Two-season pairing on squad points needs PR #260 merged.** PR #260 adds per-gameweek
  `squad_points` to the summary, because 2024-25 has no FPL average and so no `beats_real`. It also
  carries this plan file. Phase 0 checks both are on master.
- **The own substitute rate went live on 2026-10-11** (verified: merged PR #259). Every control here
  includes it.

## Scope (v1)
The projections and selection layer, judged on fair yardsticks:
1. Template and live yardsticks, and the top-100k cutoffs.
2. A reconciliation of backtest pace against live pace.
3. A zero-build batch of two near-miss arms.
4. A crowd prior from ownership and transfer momentum.
5. A position × price-band bias correction.
6. Conditional levers: per-player residuals, the optimizer's curse, crowd captaincy.
7. Live promotion of each passing lever.
8. Planner and chip re-tuning, only once G2 holds.

## Out of Scope & Parked Items
- **Elite (top-10k) ownership as a feature.** The rival sample exists only for 2026-27, so it
  cannot be backtested. Trigger: a full season of rival picks is stored.
- **New data sources: expert picks, articles, predicted line-ups.** The standing rule is that they
  must pass the backtest. Trigger: a lever round shows the gap is team news the crowd has and FPL
  data lacks.
- **A rewrite of the EP model's structure** (new likelihoods, ML models). Trigger: every lever here
  fails rule R6.
- **The app UI.** Projections keep their current fields; showing the crowd adjustment is a later
  front-end task.

## Approach
**Measure fairly first.**
- **Oracle against oracle.** Template-squad points join the walk-forward, paired per gameweek over
  both seasons. The template is model-independent, so an arm's template gap moves by exactly its
  gain over control.
- **Evolving against reality.** The season sim must run as one continuous squad, and the live team
  is judged on cumulative net points against the real top-100k cutoff.
- **Reconcile the two.** Before any lever, explain why the backtest season sim (+8.1/GW over
  average) is so far above live (+0.6/GW over 5 GWs).

**Then pull levers, one per arm, each declared before it runs.** Each is opt-in, behind its own
param family and flags in both `run_walkforward.py` and `run_season_sim_arm.py`:
- **Z (zero build).** Re-judge `--no-goal-anchor` and `--no-rate-price-anchor` under rule R6. In
  2025-26 alone they were +1.00 ± 0.83 (MAE +0.008) and +1.32 ± 1.01 (MAE +0.011) (verified:
  `docs/reports/2026-10_premium_and_form_returns.md`).
- **L1, crowd prior.** EP is adjusted by β·features (ownership and transfer momentum, interacted
  with price band).
  - The coefficients are *cross-fitted by season*: steps in 2025-26 use coefficients fitted on
    2024-25 residuals, and steps in 2024-25 use coefficients fitted on 2025-26 residuals.
  - Live and the season sim use coefficients fitted on both seasons, frozen in the param version.
    They are refitted only by a declared batch.
  - Residuals are realized − the *control run's pre-adjustment* EP per player-gameweek, extracted
    once from the Phase 1 control walk-forward DB. Every harness applies the same coefficient
    table through one shared function.
  - Using another season's relationship is look-ahead of the relationship only, not of any
    outcome. This is stated in the report. Live, where it matters, it is out of sample.
- **L2, band bias.** An additive mean residual per (position, price band), cross-fitted the same
  way, shrunk by n/(n + pseudo_rows). It differs from B3's failed band *conversion* (−2.27 ± 0.97
  in 2025-26, which scaled goal and assist rates by realized/xG ratios). L2 corrects the total EP
  residual directly. B3's failure is a stated risk.
- **L3, per-player residual (conditional).** A player's own mean residual over his last 6
  gameweeks before G, shrunk. It needs point-in-time EP history per harness (Phase 6 defines the
  source per harness, or drops L3).
- **L4, optimizer's curse (conditional).** Before the solve, a penalty μ′ = μ − κ·sd·(1 − s), with
  s = the player's ownership share within position (0–1) and sd from the uncertainty model.
- **L5, crowd captaincy (conditional).** After the solve, re-pick the captain from the XI's
  outfield players by argmax(EP + δ·EO/100). EO comes from `ownership.compute_eo_for_pool`. It is
  pre-screened as a free counterfactual in the summary (Req 2) before any arm.

L4 and L5 pull toward the template, so they are risk controls, not the route to "by a lot". The
margin over the template has to come from Z, L1–L3 and the planner levers. The template-collapse
guard (Req 9) applies to every arm.

**PROMOTE procedure** (runs as soon as any single arm passes rule R6; nothing is queued):
1. Season-sim check (Req 7) against control and a second control.
2. If m ≥ 0, apply the live-switch pattern (Req 8). Full suite green, then the PR, merge, the
   `scheduled_pipeline.yml` run, and a projections before/after diff.
3. Start the live kill-switch clock (Phase 10).

A promoted lever becomes part of every later control.

Executor's choice:
- internal function layout and new module names (follow the `fpl_quant/<topic>.py` pattern);
- the regression implementation (closed-form ridge in numpy is enough);
- how the coefficient table is stored (param rows or a seed JSON under `data/recalibration/`).

## Requirements
- **Req 1.** WHEN a walk-forward step is scored THE SYSTEM SHALL record
  `template_squad_realized_points` and `beats_template_points_delta` for that gameweek.
  - The template pool is built inside the same `asof_scope()` as the model's candidate pool.
  - It is solved and scored exactly as `beats_baseline` does.
  - Acceptance: unit tests (a known template squad; captain = the most-owned outfield XI player
    [A5]) and both seasons' summaries carry the field every gameweek.
  - The added runtime per run is measured and recorded.
- **Req 2.** WHEN a summary is written THE SYSTEM SHALL include, per gameweek, `squad_points`,
  `template_points` and a diagnostics block:
  - XI overlap count with the template XI;
  - captain realized points against the most-owned captain's;
  - a captain counterfactual `ep_plus_eo` (argmax EP + 1.0·EO/100 among outfield XI);
  - £ spent per price band for both XIs;
  - realized − projected for the model XI and for the template XI under the model's EP.
  
  Acceptance: unit tests on a seeded run.
- **Req 3.** WHEN `export_leaderboard.py` runs THE SYSTEM SHALL add a "model (oracle)" row: the
  from-scratch model pick each gameweek, scored like the baselines. `model_beats_baseline_by` is
  reported against it for the oracle baselines. The crowd-average row is labelled "synthetic EO
  crowd (not FPL's average)". Acceptance: a unit test and the next weekly backtest's
  `leaderboard_latest.json`.
- **Req 4.** WHEN the rival-sample workflow runs THE SYSTEM SHALL store each sampled entry's FPL
  `entry/{id}/history`:
  - `current`: per-GW points, `event_transfers_cost`, `total_points`, `overall_rank`;
  - `past`: per-season `total_points` and `rank`.
  
  From these it derives each season's top-100k cutoff (2024-25 and 2025-26 from `past`; 2026-27
  per GW from `current`) by interpolating total points at rank 100,000 across the sampled entries.
  Acceptance: a test on canned JSON; the report records the cutoffs and the number of entries
  bracketing rank 100k.
- **Req 5.** WHEN `compare_arms.py control arm --rule r6` runs THE SYSTEM SHALL return PASS only if:
  - (i) paired `squad_points` over every gameweek of every season gives m − SE ≥ 0;
  - (ii) each season's EP MAE ≤ control + 0.01.
  
  It also:
  - refuses on different cached DBs, different seasons, an incomplete run, or any season with
    fewer than 2 paired gameweeks;
  - prints both runs' template gap and mean template overlap, the number of changed gameweeks, and
    the SE.
  
  Acceptance: unit tests for PASS, each FAIL and each refusal.
- **Req 6.** WHEN a lever flag is given THE SYSTEM SHALL mint or reuse an immutable param version.
  Without the flag, the EP output is byte-identical to the control. Acceptance: a flag test per
  lever in both scripts, plus a control-identity test.
- **Req 7.** WHEN an arm passes rule R6 THE SYSTEM SHALL run `season_sim_arms.yml` for 2025-26 as
  one continuous squad: chunk `2-38` if one job fits in 345 min, else chained chunks resumed from
  the previous chunk's end state (squad, bank, free transfers, chips used).
  - The arms are `control:`, `control_b:` and `arm: <flags>`.
  - The check passes on paired net points, arm − control, m ≥ 0 over complete chunks only.
  - control_b − control is recorded as the noise floor.
  - Splitting a chip half into fresh-squad chunks is forbidden.
  
  Acceptance: the result and floor are in the report before the promotion PR.
- **Req 8.** WHEN a lever is promoted THE SYSTEM SHALL follow the live-switch pattern:
  - a v1 seed called from `backtest.materialize_confirmed_seeds()` and `scripts/run_ingestion.py`;
  - a `RECALIBRATABLE_VERSION_ARGS` entry;
  - every call site of the affected function passing it;
  - the `REQUIRED` lists in `tests/test_live_switch_wiring.py`;
  - a `--no-<lever>` flag in both scripts.
  
  Acceptance: the wiring test and the full suite pass.
- **Req 9 (template-collapse guard).** WHEN an arm is judged THE SYSTEM SHALL report its mean XI
  overlap with the template.
  - An arm that raises mean overlap by ≥ 1 player/GW while its template gap widens by less than its
    m is not promoted.
  - Mean overlap ≥ 9/11 stops the programme for owner review.
  
  Acceptance: printed by `--rule r6` and recorded per batch.
- **Req 10.** Every batch is declared in `docs/reports/2026-10_beat_the_template.md` (arms, flags,
  declared hyperparameter values, rule, harness) and committed BEFORE dispatch. Results are
  appended after. Acceptance: the git history shows the declaration commit precedes the run ids.
- **Req 11.** WHEN the crowd prior (or any residual-fitted lever) is on THE SYSTEM SHALL:
  - leave a player's EP unchanged when he has no feature row (never treat missing data as 0);
  - read ownership and transfers at step G from rows gw ≤ G−1 (gw ≤ G−2 if A2 fails);
  - apply the same coefficient table through one function in the walk-forward, the season sim and
    live, to every planning-horizon gameweek.
  
  Acceptance:
  - a leak test (changing row G's ownership does not change step G's EP);
  - a missing-data test;
  - a test that the three call paths produce identical adjustments for the same inputs.

## Key Decisions
- **Crowd information may feed projections** (user).
- **Rule R6** (user):
  - m − SE ≥ 0 on paired squad points over both seasons;
  - EP MAE ≤ control + 0.01 per season;
  - a season-sim check with m ≥ 0 before live.
  
  R5's target and breakout checks are reported only. A single passing arm goes straight to PROMOTE
  (the owner's standing instruction, "if an arm wins just promote it and merge"; A8).
- **The finish line is top-100k** (user), translated as G2 (≥ +2.0/GW over the template) [A10], G3
  (the season sim at the real top-100k cutoff pace) and G4 (live cumulative points at the cutoff).
- **The noise floor.** Per arm, the SE is sd(paired diff)/√74. It ranges from about 0.4 (an arm
  changing a few squads, D2) to about 1.2 (λ arms changing most squads; seeds_lambda note)
  [A6]. Each arm records its changed-GW count and SE. A null arm passes m − SE ≥ 0 with probability
  ≈ 16%, so total lever arms are capped at 10 before owner review.
- **Declared hyperparameters** (single values; one declared alternative at most, in the same batch):
  - **L1:** `ridge_alpha` 10 on standardized features. `min_rows` 2,000 player-GWs for a fit. The
    clip on the adjustment is ±1.5 pts/GW. `feature_set` v1:
    - ownership percentile within position;
    - log(1 + `selected_by_percent`);
    - last gameweek's net transfers ÷ max(owners, 1), clipped to [−1, 1];
    - each of the three × one-hot price band (4 bands).
  - **L2:** `pseudo_rows` 500.
  - **L3:** N = 6 gameweeks, shrink n/(n + 6), players with ≥ 900 league minutes in the current
    plus previous season.
  - **L4:** κ ∈ {0.25, 0.5} (two arms).
  - **L5:** δ = 1.0 point per 100% EO.
  
  These values are [assumed] defaults with no fitted basis. They are fixed before any run so they
  cannot be tuned after the results.
- **The template squad** is `ownership_popularity` from `beats_baseline`. Its captain is the solve's
  own pick (max mu, outfield) [A5].
- **The walk-forward harness for every lever arm** is `branch_walkforward.yml` with no `--seasons`
  (both seasons, 75 steps), dispatched with `actions_run_trigger` on the worktree branch's ref
  [A4].
  - Control and arm use different args, so they don't cancel each other. They are dispatched
    back-to-back.
  - Both are re-dispatched if their `db_cache_key` values differ.
- **Batches have at most 3 arms plus control.**

## Data & State Changes
- **New param families**, numeric and versioned: `crowd_prior_params` (coefficients per season fold
  plus the frozen live set, `ridge_alpha`, `min_rows`, `clip`, `feature_set`), `band_bias_params`,
  `player_residual_params`, `selection_shrink_params`, `captaincy_crowd_params`.
- **A new table `fact_entry_history`** (Req 4) via an additive `schema/00NN_entry_history.sql`
  migration: entry_id, season, gw (NULL for `past` rows), points, transfers_cost, total_points,
  overall_rank, _ingested_at. It follows the rival-sample retention rule
  (`purge_prior_season_rival_squad_sample` pattern).
- **Season-sim chaining state**, if needed (Req 7): a JSON artifact per chunk holding squad, bank,
  free transfers and chips used. There is no DB schema change.
- **New `backtest_metrics` names** (`template_squad_realized_points`, `beats_template_points_delta`,
  diagnostics) are additive.
- **Rollback:** every lever has a `--no-<lever>` flag and a live switch version. Removing the seed
  entry reverts live behaviour.

## Interfaces, Integrations & Credentials
- **GitHub Actions** via the GitHub MCP tools:
  - `actions_run_trigger` (method `run_workflow`) on `branch_walkforward.yml`,
    `season_sim_arms.yml`, `scheduled_pipeline.yml` and the rival-sample/rank workflow
    (`.github/workflows/rank_tracking.yml`, verified to exist by the
    `run_rival_sample_ingestion.py` docstring);
  - `actions_list` and `actions_get` for runs, logs (`ARGS:`, `DB_CACHE_KEY:` lines) and
    artifacts.
- **FPL's public API** from Actions only: `entry/{id}/history`. The sandbox cannot reach
  fantasy.premierleague.com (verified: `run_rival_sample_ingestion.py` docstring). Use the
  existing request pacing and caching discipline there. No credentials are needed.
- **Frozen surfaces:**
  - the JSON shapes of `data/dashboard/*.json` read by the app (only add keys or rows);
  - the walk-forward summary keys existing tools read (only add keys).

## Edge Cases & Failure Handling
- No ownership row for a player-GW → no adjustment (Req 11).
- A fit with fewer than `min_rows` rows → the lever is off for that fold, logged.
- Double gameweeks: the per-GW adjustment is split across a player's fixtures in proportion to
  each fixture's EP. Blank gameweeks: no fixture, no adjustment.
- A template solve status outside (`optimal`, `timelimit`) → None for that gameweek. Rule R6 pairs
  only gameweeks present in both runs.
- An incomplete walk-forward or season-sim chunk → refused or not judged. Re-run; never judge a
  partial run.
- An Actions job that dies before any step ran → re-run once. A second failure is real.
- FPL API pagination doesn't reach rank 100k → record the bands actually reached. If no entries
  bracket rank 100k, G3 and G4 use the nearest bracket and say so.

## Risks, Landmines & Adaptations
- **Point-in-time leakage through ownership** (a prior leak was fixed in #214, see the
  `seeds_lambda_2026-10-01.json` note) → Phase 1 verifies [A2]. Req 11 enforces gw ≤ G−1, with a
  fallback to gw ≤ G−2.
- **An unfair live yardstick** (evolving model vs oracle baselines; a synthetic crowd far above
  FPL's average) → Req 3 adds the oracle-model row. G4 uses real cumulative points against the real
  cutoff, not the leaderboard.
- **An inflated season-sim pace** (a fresh squad at GW20; +8.1 backtest vs about +0.6 live) → Req 7
  requires one continuous squad. Phase 2 reconciles backtest against live before any lever.
- **The season sim is nondeterministic** (±1.81/GW) → every season-sim batch has `control_b`, and
  the floor is recorded. Rule R6's m ≥ 0 check stands as the owner set it. The live kill-switch
  (Phase 10) is the real out-of-sample guard.
- **The noise floor varies by arm** (0.4–1.2) → record SE and changed GWs per arm. Bundle levers
  into a combined arm only when each is individually plausible (Phase 8). Never re-run an arm for a
  better draw.
- **Overfitting by reusing 74 gameweeks across many arms.** Only 2026-27 is out of sample: the
  season sim replays 2025-26 → declaration first (Req 10), a cap of 10 lever arms, cross-fitting for
  L1/L2, and the Phase 10 kill-switch.
- **The crowd prior turning the model into the template** → Req 9 thresholds. L4/L5 are classed as
  risk controls.
- **Hyperparameter fishing** → single declared values (Key Decisions).
- **Concurrent edits** by another session to `scripts/compare_arms.py`,
  `scripts/walkforward_summary.py` and `src/fpl_quant/backtest.py` → merge `origin/master` into the
  worktree branch before every PR. Never force-push over others' commits.
- **Compute.** A two-season walk-forward takes ~40 min (verified: D2 runs). The template solve adds
  per-step time, measured per Req 1. Season-sim jobs approach 345 min [A7].

## Assumptions Ledger
| ID | Assumption | Basis | Blast radius if wrong | Check |
|----|------------|-------|-----------------------|-------|
| A1 | Sampled managers' `entry/{id}/history` `past` arrays give per-season totals and ranks, enough to interpolate the rank-100k total for 2024-25 and 2025-26 | [assumed: FPL API shape, not verified this session] | G3 has no target | Phase 1, Req 4 test on a real response |
| A2 | `selected_by_percent` and transfer values in row gw = G−1 are knowable before G's deadline | [assumed] | L1 is leaky; fallback gw ≤ G−2 | Phase 1: snapshot timestamps vs deadlines, plus the Req 11 leak test |
| A3 | `transfers_in_event`/`transfers_out_event` are populated for 2024-25 and 2025-26 | [assumed] | L1 drops the momentum features | Phase 1: non-null counts per season in the cached DB |
| A4 | `workflow_dispatch` on the worktree branch's ref runs that branch's code | [assumed: standard GitHub behaviour] | Arms run master's code | Phase 1: the run's `head_sha` equals the branch head |
| A5 | The template captain is the most-owned outfield XI player | [assumed: `solve` maximises mu with the captain doubled; captain-not-GK constraint verified] | The template baseline is slightly off | Phase 1 unit test |
| A6 | The per-arm paired SE over 74 GWs is 0.4–1.2 | verified: D2 ±0.39 to ±0.74; seeds_lambda λ arms ±1.21 (n = 70) | Rule R6's power differs per arm | Recorded per arm |
| A7 | One GW2–38 season-sim job fits in 345 min, or chaining is needed | [assumed] | G3 is measured on a non-continuous squad | Phase 1: read past `season_sim_arms` job durations (`actions_list list_workflow_jobs`) |
| A8 | The executor has the owner's standing authority: merge green PRs, promote an arm that passes rule R6 and the season-sim check, ask the owner before merging any winner with m < 0 | (user) standing instructions this session | It would need to stop for approval | none; the owner can revoke |
| A9 | Walk-forward runs on the same commit, DB and args are deterministic (squads identical) | verified: D1 and D2 controls identical in 2025-26 | A noise floor is needed for the walk-forward too | If two controls ever differ, add a control_b to walk-forward batches |
| A10 | "By a lot" means ≥ +2.0 pts/GW over the template (≈ 76 pts a season) | [assumed: the owner said "by a lot"] | G2's bar is wrong | The owner can restate it at approval |
| A11 | The walk-forward can run season "2026-2027" for the kill-switch | [assumed] | The kill-switch needs the season sim (export_leaderboard path) instead | Phase 10: one dispatch with `--seasons 2026-2027` |

## Open Items (none blocking)
- **Elite ownership for 2026-27.** Proceed without it.
- **Showing the crowd adjustment in the app.** EP fields stay unchanged.

## Verification
- `PYTHONPATH=src python -m pytest tests/ -q` is green, and `ruff check .` and `mypy src/` are
  clean, before every PR.
- `PYTHONPATH=src python scripts/compare_arms.py control.json arm.json --rule r6` prints PASS/FAIL
  with (i) and (ii), the template gap, overlap, changed GWs and SE.
- Each batch's report section lists the declaration commit, run ids, the `ARGS:` mapping,
  `db_cache_key` and the verdict.
- Season-sim batches record control, control_b and the arm, the floor, and the completed chunks.
- Live: `projections_latest.json` before/after diff per promotion. `app_model_team.json` cumulative
  points against the 2026-27 cutoff per GW, from `fact_entry_history`.
- The owner confirms done from the report's headline table:
  - the model − template gap (G2);
  - season-sim pace against the real cutoff pace (G3);
  - live cumulative points against the live cutoff (G4).

## Build Phases
- [ ] Phase 0: Set up the worktree and confirm the prerequisites
      Done when: `git -C ../FPL-Analyser-btt status` shows branch `claude/beat-template`;
      `grep -n squad_points scripts/walkforward_summary.py` matches; and
      `docs/plans/2026-10_beat_the_template.md` exists, all in that worktree.
      Steps:
      - `git fetch origin master && git worktree add ../FPL-Analyser-btt -b claude/beat-template origin/master`
        (or the host's EnterWorktree).
      - If PR #260 (squad points and this plan) is not merged, stop and merge it first.
      - Create `docs/reports/2026-10_beat_the_template.md` with the rule R6 text, G1–G4, the
        declared hyperparameters and the stop rule, then commit.
      Covers: Req 10

- [ ] Phase 1: Build the yardsticks and measure the baselines
      Done when:
      - Req 1–5 tests pass; the PR is merged.
      - A both-season control walk-forward on the branch records the model − template gap with SE,
        the overlap and the diagnostics.
      - A1, A2, A3, A4 and A7 are resolved in the report.
      - The 2024-25 and 2025-26 top-100k cutoffs (pts/GW) are recorded.
      Steps:
      - Write failing tests first (Req 1–5, A5), then implement. Reuse `_naive_candidate_pool`,
        `squad_optimizer.solve` and `_realized_solve_points` as `beats_baseline` does. The control's
        EP must not change.
      - Add `fact_entry_history` and the history fetch to the rival-sample ingestion. Dispatch the
        rank-tracking workflow, then compute the cutoffs.
      - Check A2 and A3 with a diagnostic script run in Actions on the cached DB (pattern:
        `.github/workflows/diagnose_premium_returns.yml`).
      - Check A7 from past `season_sim_arms` job durations. If a GW2–38 job won't fit, implement
        chunk chaining (Req 7).
      - Dispatch the control walk-forward on the branch (args `""`) and confirm A4 from its
        `head_sha`.
      Covers: Req 1, Req 2, Req 3, Req 4, Req 5, Req 7 (chaining); checks: A1, A2, A3, A4, A5, A7
      Critique: important phase. After building, hand the result plus this phase's Done-when and
      Covers lines to a blind critic (a fresh sub-agent if available, else a fresh-eyes reread). The
      critic defaults to FAIL and returns PASS/FAIL, strengths, weaknesses and fixes. Apply blocker
      fixes and repeat, max 3 rounds. Done only on PASS.

- [ ] Phase 2: Reconcile backtest against live, decompose the gap, and declare the lever order
      Done when:
      - The report has the reconciliation table: 2025-26 season sim (continuous; control and
        control_b) vs the 2026-27 leaderboard sim vs the live team, each as net pts/GW over FPL's
        average and over the cutoff pace, with named causes for the difference.
      - The report has the decomposition table and an ordered lever queue derived by the rules
        below, committed before any lever run.
      Steps:
      - Dispatch `season_sim_arms.yml` with arms `control: ; control_b:` for 2025-26, continuous.
      - From the Phase 1 control summary, compute overlap, model-only vs template-only points, the
        captain gap and the `ep_plus_eo` counterfactual, £ per band, and the curse residuals.
      - **If model − template ≥ +2.0 with m − SE > 0:** G2 is met, so skip Phases 3–7 and go to
        Phase 9.
      - Otherwise the order is Z, then L1, then L2:
        - L3 only if £9.0m+ realized − projected is still ≥ +0.1 after the levers that passed.
        - L4 only if the model XI's realized − projected is ≤ −0.3 pts/player-GW and the template
          XI's is ≥ 0.
        - L5 only if the `ep_plus_eo` counterfactual beats the model's captain by ≥ +0.3 pts/GW
          with m − SE > 0.
      Covers: Req 7, Req 10
      Critique: important phase (it sets what "fair" means). Run the blind-critic loop as in
      Phase 1 on the reconciliation write-up.

- [ ] Phase 3: Run the zero-build batch (Z)
      Done when: the batch (control, `--no-goal-anchor`, `--no-rate-price-anchor`; both seasons) is
      declared, run, judged with `--rule r6` and recorded. Any PASS has completed PROMOTE.
      Steps:
      - Declare in the report, commit, then dispatch the three runs back-to-back on the branch.
      - Judge each arm against control. On PASS, run PROMOTE (season-sim check with control_b, then
        Req 8 and the live check).
      Covers: Req 5, Req 7, Req 8, Req 9, Req 10

- [ ] Phase 4: L1, the crowd prior
      Done when: declared, run, judged with `--rule r6` and recorded. A PASS has completed PROMOTE.
      A FAIL records the reason and counts towards the family stop rule.
      Steps:
      - Write failing tests first: the Req 11 leak, missing-data and identical-paths tests, the
        `min_rows` gate, cross-fit fold selection, and the flags in both scripts (Req 6).
      - Extract the residual table (realized − control pre-adjustment EP, plus features) from the
        Phase 1 control run's DB in Actions. Fit the 2024-25, 2025-26 and both-season coefficient
        sets with the declared values. Store them in `crowd_prior_params`.
      - Implement the shared apply function and `--crowd-prior` in both scripts.
      - Declare, commit, dispatch control and arm, judge, and PROMOTE on PASS.
      Covers: Req 6, Req 9, Req 10, Req 11
      Critique: important phase (leakage). Run the blind-critic loop on the fit and apply code.

- [ ] Phase 5: L2, the position × price-band bias correction
      Done when: declared, run, judged with `--rule r6` and recorded. A PASS has completed PROMOTE.
      Steps:
      - Write failing tests first: cross-fit only, shrinkage n/(n+500), Req 11 paths, the flag.
      - Implement `band_bias_params` from the same residual table. Declare control and
        `--band-bias`, plus `--crowd-prior --band-bias` if L1 passed and is not yet live. Dispatch,
        judge, PROMOTE on PASS.
      Covers: Req 6, Req 9, Req 10, Req 11

- [ ] Phase 6: L3, the per-player residual (only if Phase 2's rule queued it)
      Done when: declared, run, judged and recorded, or the report says why it was skipped. That
      includes the case where no harness can supply point-in-time EP history.
      Steps:
      - Name the EP-history source for each harness: walk-forward = earlier steps' `ep_outputs` in
        the same run; season sim = the sim's own per-GW EP, carried in the chaining state; live =
        each gameweek's deadline-time model version in the live DB. Drop L3 if any is missing.
      - Tests (leak, shrinkage, paths, flag), then declare, run, judge, PROMOTE on PASS.
      Covers: Req 6, Req 9, Req 10, Req 11

- [ ] Phase 7: L4 and L5, the risk controls (only those Phase 2's rule queued)
      Done when: each queued lever is declared, run, judged and recorded, or skipped with the
      reason.
      Steps:
      - L4: `selection_shrink_params`, μ′ = μ − κ·sd·(1 − s) before the solve. Tests: no effect at
        κ = 0, monotonic in sd, the flag. Two arms, κ 0.25 and 0.5.
      - L5: `captaincy_crowd_params`, a post-solve captain re-pick by argmax(EP + 1.0·EO/100) over
        outfield XI. Tests: re-pick only among XI, the vice unchanged unless it is the new captain,
        the flag.
      - Declare, run, judge (Req 9 applies strictly), PROMOTE on PASS.
      Covers: Req 6, Req 9, Req 10

- [ ] Phase 8: Combine levers that were individually plausible but not promoted
      Done when: either no such levers exist (recorded), or the combined arm is declared, run,
      judged, and promoted on PASS. On FAIL nothing changes, since every lever that passed alone is
      already live.
      Steps:
      - The candidates are arms with m > 0 that failed only m − SE ≥ 0. Combine them into one arm,
        declare, run against a fresh control (which includes all promoted levers), judge, PROMOTE on
        PASS.
      Covers: Req 5, Req 7, Req 10

- [ ] Phase 9: Re-tune the planner and chips under the new projections (only once G2 holds)
      Done when: for each declared season-sim batch (`control: ; control_b: ; arm:`, 2025-26,
      continuous), the report records a decision: promote if arm − control m − SE ≥ 0 on net
      pts/GW, else reject.
      Steps:
      - Use the existing season-sim flags, one per arm, at most 3 per batch: `--transfer-threshold`,
        `--multi-transfers`, `--chip-timing`, `--chip-wait`, `--chip-option-value`.
      - Declare, run, judge, and promote by Req 8.
      Covers: Req 7, Req 8, Req 10

- [ ] Phase 10: Track live, apply the kill-switch, and decide whether to continue
      Done when: for gameweek N (≥ 6 live gameweeks after the latest promotion) the report records:
      - the headline table (G2 gap, G3 pace vs cutoff, G4 cumulative vs cutoff);
      - for each promoted lever, its 2026-27 comparison against `--no-<lever>` (paired m ± SE over
        GW2–N), and the decision to keep it or roll it back (roll back if m + SE < 0);
      - the programme decision for GW N: continue the queue, pause for owner review (stop rule), or
        close.
      Steps:
      - After each weekly backtest and rank-tracking run, update the table from
        `app_model_team.json` and `fact_entry_history`.
      - Dispatch the kill-switch comparison: a walk-forward with `--seasons 2026-2027` [A11], or
        the season sim on 2026-27 if A11 fails.
      Covers: Req 4, Req 8, Req 10
