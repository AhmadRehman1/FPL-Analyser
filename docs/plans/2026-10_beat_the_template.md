# Plan: Beat the template, then push toward a top-100k season

One-line goal: the model's picks beat a legal ownership-template squad by a wide margin in the
backtest, and the live model team plays at top-100k overall pace, where today it trails the template.

## Classification
Track: Performance. The end state is a number: squad points against the template, then against
top-100k pace. Unlike a classic performance job, the projections are meant to change, so
"frozen surfaces" means output formats and the live pipeline, not model behaviour.

Parked secondary asks:
- Transfer-planner and chip strategy: Phase 10, only once projections beat the template.
- Elite (top-10k) ownership as an input: Out of Scope, with triggers.

## Interview Ledger
- Q1 crowd input → ownership and transfer trends MAY feed the projections as a backtested prior
  (user, accepted recommendation).
- Q2 promotion rule → R6, points-first (user, accepted recommendation): paired squad points over both
  seasons with m − SE ≥ 0; EP MAE ≤ control + 0.01 in each season; then a season-simulation check
  with m ≥ 0 before going live.
- Q3 finish line → "beat template by a lot; we need to push to like top 100k overall" (user, free
  form). It is translated into the measurable targets under Goal & Success Criteria.

Questions used: 3 of 14, plus the approval question.

## Goal & Success Criteria
- **G1 (yardstick exists).** Every walk-forward summary reports, per gameweek, the model squad's
  realized points and a legal ownership-template squad's realized points. The template squad is
  built under the same budget, formation, club cap, captain/vice and auto-sub rules.
  `compare_arms.py --rule r6` judges any arm on them.
- **G2 (beat the template).** The live model's walk-forward squad beats the template squad over both
  seasons: paired mean gap m ≥ +2.0 pts/GW, with m − SE > 0. Today it trails the synthetic crowd
  by 1.15 pts/GW (verified: `data/dashboard/app_track_record.json` headline, 74 GWs). The gap to
  the legal template squad is measured in Phase 1.
- **G3 (top-100k pace, backtest).** The evolving manager (transfers, hits, chips;
  `season_sim_arms.yml`) scores net pts/GW over 2025-26 at least FPL's real average plus the
  top-100k margin.
  - The margin is measured in Phase 1 from the live rival sample's 60k–140k band
    [A1: default +9 pts/GW over the average, net of hits].
  - FPL's 2025-26 average is 49.76 pts/GW (verified: walk-forward run 38167650902,
    `real_avg_manager_points`).
- **G4 (live).**
  - "price/ownership baseline" and "crowd-average baseline" leave the honest losses in
    `data/dashboard/leaderboard_latest.json`. Today both are there (verified: that file,
    2026-10-11).
  - The model team's estimated rank in `data/dashboard/rank_autopsy.json` trends into the top 100k
    over the rest of 2026-27.
- **Stop rule.**
  - A lever family is dropped after two arms in a row fail R6.
  - After every three batches, the executor writes a progress section in the report: gap to the
    template, distance to G3, remaining levers.
  - When the backlog is empty with G2 unmet, the executor stops and reports to the owner. It does
    not invent levers outside this plan.

## Current State
- **The walk-forward squad is an oracle.** It is a fresh, hit-free squad every gameweek (verified:
  `src/fpl_quant/backtest.py` `real_average_benchmark_provenance()`, "oracle": True). That makes
  it a free wildcard each week. Even so, it trails the template:
  - 2024-25: model 63.54 pts/GW against synthetic crowd 61.85.
  - 2025-26: model 60.16 against crowd 64.15.
  
  (verified: walk-forward control run 38167650902 summary, `headline_by_season`)
- **A legal template squad already exists in code.** `backtest._naive_candidate_pool(signal=
  "ownership")` builds it (mu = the latest `selected_by_percent`, read inside `asof_scope()`), and
  it is solved by `squad_optimizer.solve(lam=0)` and scored by `_realized_solve_points()`. It is
  used only by `backtest.beats_baseline()` for the live-season leaderboard (verified: backtest.py
  `beats_baseline`, `_naive_candidate_pool`, `_realized_solve_points`). Over GW2–5 of 2026-27 it
  scored 266 against the model's 241 (verified: `data/dashboard/leaderboard_latest.json`).
- **Ownership and transfers never reach the projections** (verified: no `selected_by_percent` or
  `transfers_in_event` in `src/fpl_quant/expected_points.py` or `minutes_model.py`).
  `fact_player_season_stats` holds `selected_by_percent` (verified: `schema/0001_core_schema.sql`)
  and `transfers_in_event`/`transfers_out_event` (verified: `schema/0014_priority4_transfer_momentum.sql`,
  ingested by `reconcile.py`). Historical coverage of the transfer columns is unverified [A3].
- **Projection bias by price band** (2024-25 and 2025-26 pooled, control run 38167650902,
  realized − projected per player-GW, verified):

  | band | total | goals | assists | other |
  |---|---|---|---|---|
  | <£5.0m | −0.158 | | | |
  | £5.0–7.0m | +0.062 | | | |
  | £7.0–9.0m | +0.272 | | | |
  | £9.0m+ | +0.175 | −0.177 | +0.169 | +0.182 |

  The 9.0m+ column split is goals, assists and bonus/other.
- **Captaincy** (2024-25 and 2025-26 pooled, verified, same run):
  - The captain scores 6.42 pts/GW before doubling.
  - The P95 upside rule would gain +0.39 ± 0.30.
  - The best pick in hindsight scores 12.87.
- **Optimizer settings.** Risk aversion λ is 0 live (verified: `data/recalibration/seeds_lambda_2026-10-01.json`).
  The only optimizer guardrail is ≤3 XI players per club (verified: `squad_optimizer.py:405`).
- **The season-simulation harness exists.**
  - `season_sim_arms.yml` takes arms as `"label: flags"`, a seasons input (default 2025-2026) and
    chunks (default `2-19,20-38`), with a job timeout of 345 minutes (verified: that workflow).
  - `scripts/run_season_sim_arm.py` has its own flag parser and builds an `argparse.Namespace` for
    `run_walkforward._experiment_versions` (verified: that script, lines 44–74). A new lever flag
    therefore has to be added to both scripts.
- **The rank goal "overall rank top 100k" is already declared** in `data/dashboard/rank_autopsy.json`
  (verified). `fact_rival_squad_sample` samples real squads in rank bands, including 60k–140k, for
  2026-27 only. FPL's API serves only current-season picks (verified: `rank_autopsy.json`
  limitations; `scripts/run_rival_sample_ingestion.py` docstring).
- **Two-season pairing on squad points needs PR #260 merged.** PR #260 adds per-gameweek
  `squad_points` to the summary, because 2024-25 has no FPL average and so no `beats_real`.
  Phase 1 checks it is on master.
- **The own substitute rate went live on 2026-10-11** (verified: merged PR #259). Every control in
  this programme includes it.

## Scope (v1)
The projections and selection layer, judged by the new yardstick. In order:
1. The template yardstick and diagnostics.
2. A crowd prior from ownership and transfer momentum.
3. A position × price-band bias correction.
4. Two conditional levers: elite de-compression and an optimizer's-curse correction.
5. A captaincy lever.
6. A combined arm, confirmed in the season simulation, then promoted live.

Then, only once G2 holds, planner and chip re-tuning under the new projections.

## Out of Scope & Parked Items
- **Elite (top-10k) ownership as a feature.** It exists only for 2026-27 live (rival sample), so it
  cannot be backtested. Trigger to revisit: a full season of rival samples exists.
- **New data sources: expert picks, articles, predicted line-ups.** The standing rule is that they
  must pass the backtest. Trigger: a lever round shows the gap is team news the crowd has and FPL
  data lacks.
- **A rewrite of the EP model's structure** (new likelihoods, ML models). Trigger: all levers here
  fail R6.
- **The app UI.** Projections keep their current fields. Showing the crowd adjustment is a later
  front-end task.

## Approach
**Measure first.** Make "model squad vs template squad, paired per gameweek over both seasons" the
headline. The template is model-independent, so every arm's gap shrinks exactly as much as it
beats control.

**Diagnose next.** Decompose the gap by:
- overlap between the model XI and the template XI;
- points from model-only picks against template-only picks;
- captain against the most-owned captain;
- spend per price band;
- the model XI's realized − projected (the optimizer's curse).

Pre-declared mappings then order the levers.

**Lever mechanics** (each opt-in, behind its own param family and flag, judged by R6 before
anything goes live):
- **L1, crowd prior.** At step G, regress per player-gameweek residuals (realized − EP) on
  ownership and transfer features, using only rows with gw < G. Add the shrunk fitted adjustment
  to EP.
- **L2, band bias correction.** The same idea with position × price-band dummies only.
- **L3, elite de-compression.** Each player's own recent residual, shrunk.
- **L4, optimizer's-curse correction.** Shrink a candidate's EP toward a crowd-informed prior in
  proportion to its uncertainty and its distance from the template.
- **L5, captaincy.** Choose the captain from EP plus captaincy ownership.

Executor's choice: internal function layout, feature encodings within each lever, and file names
for new modules (follow the existing `fpl_quant/<topic>.py` pattern).

## Requirements
- **R1.** WHEN a walk-forward step is scored THE SYSTEM SHALL record `template_squad_realized_points`
  and `beats_template_points_delta` for that gameweek.
  - The template squad is built from `_naive_candidate_pool(signal="ownership")` inside the same
    `asof_scope()` as the model's own candidate pool, and solved with `squad_optimizer.solve(...,
    lam=0.0, ...)` using the live captain variance multiplier.
  - It is scored by `_realized_solve_points()` after the scope exits.
  - Acceptance: a unit test where the template squad is known, and both seasons' summaries carry
    the field for every gameweek.
- **R2.** WHEN a summary is written THE SYSTEM SHALL include per gameweek `squad_points`,
  `template_points` and the diagnostics block. That block holds XI overlap count, captain
  vs most-owned-captain realized points, £ spent per price band for both XIs, and the model XI's
  realized − projected and the template XI's realized − projected (under the model's EP).
  Acceptance: unit tests on a seeded run.
- **R3.** WHEN `compare_arms.py control arm --rule r6` runs THE SYSTEM SHALL return PASS only if:
  - (i) paired `squad_points` over every gameweek of every season gives m − SE ≥ 0;
  - (ii) each season's EP MAE is ≤ control + 0.01.
  
  It refuses runs over different cached DBs, different seasons, an incomplete run, or any season
  with fewer than 2 paired gameweeks. It prints the template gap for both runs (informational).
  Acceptance: unit tests for PASS, each FAIL reason and each refusal.
- **R4.** WHEN a lever flag is given to `run_walkforward.py` or `run_season_sim_arm.py` THE SYSTEM
  SHALL mint or reuse an immutable param version for it. Without the flag, behaviour is
  byte-identical to the control. Acceptance: a flag test per lever, plus a test that the control
  path produces the same EP as before.
- **R5.** WHEN the crowd prior is on THE SYSTEM SHALL use only rows with gw < G at step G. It
  leaves a player's EP unchanged when he has no ownership row, and makes no adjustment until at
  least `min_rows` training rows exist. Acceptance: a leak test (changing row G's ownership does
  not change step G's EP) and a missing-data test.
- **R6.** WHEN an arm passes R6 on the walk-forward THE SYSTEM SHALL be confirmed by
  `season_sim_arms.yml` (2025-26, chunks 2–19 and 20–38): paired net points per gameweek of
  control against arm, with m ≥ 0, before any live promotion. Acceptance: the result is recorded
  in the report before the promotion PR.
- **R7.** WHEN a lever is promoted THE SYSTEM SHALL follow the live-switch pattern:
  - a v1 seed function called from `backtest.materialize_confirmed_seeds()` and
    `scripts/run_ingestion.py`;
  - a `RECALIBRATABLE_VERSION_ARGS` entry;
  - the version passed at every call site of the affected function;
  - the `REQUIRED` lists in `tests/test_live_switch_wiring.py`;
  - a `--no-<lever>` flag for later comparisons.
  
  Acceptance: `tests/test_live_switch_wiring.py` passes and the full suite is green.
- **R8.** Every batch is declared in `docs/reports/2026-10_beat_the_template.md` (arms, flags, rule,
  harness) and committed BEFORE its runs are dispatched. Results are appended after. Acceptance:
  the git history shows the declaration commit precedes the run ids recorded.

## Key Decisions
- **Crowd information may feed projections** (user).
- **The promotion rule is R6** (user):
  - points first: m − SE ≥ 0 on paired squad points over both seasons;
  - MAE guard ≤ control + 0.01 per season;
  - a season-sim check with m ≥ 0 before live.
  
  The R5 target-group and breakout checks are reported but do not decide, because R6 does not
  include them.
- **The finish line is top-100k pace** (user), translated as G2 (≥ +2 pts/GW over the template)
  and G3 (top-100k pace in the season sim) [A1, A2 for the numbers].
- **The template squad** is `ownership_popularity` from `beats_baseline` (verified: it exists and is
  scored with real auto-subs and vice). Its captain is the solve's own pick from the ownership mu:
  the most-owned player in the XI [A5].
- **The harness for every lever arm** is `branch_walkforward.yml` with no `--seasons` (both
  seasons, 75 steps), dispatched on the worktree branch's ref [A4]. Control and arm come from the
  same commit and the same `db_cache_key` (verified: the workflow prints `DB_CACHE_KEY` and
  `compare_arms.py` refuses mismatches).
- **Batch size is at most 3 arms plus control.** Each arm changes one lever, except the combined
  arm in Phase 8.

## Data & State Changes
- **New param families**, numeric and versioned like the existing ones: `crowd_prior_params`
  (e.g. `ridge_alpha`, `min_rows`, `feature_set`), `band_bias_params` (`pseudo_rows`),
  `elite_residual_params`, `selection_shrink_params`, `captaincy_crowd_params`.
- **No schema migration is required** for v1. If a lever needs a recipe column on
  `ep_model_versions` (as `conversion_params_version` did), add an `ALTER TABLE ... ADD COLUMN IF
  NOT EXISTS` migration under `schema/` with the next number. That is additive and needs no
  rollback beyond ignoring the column.
- **New `backtest_metrics` metric names:** `template_squad_realized_points`,
  `beats_template_points_delta`, and diagnostics. These are additive.

## Interfaces, Integrations & Credentials
- **GitHub Actions** via the GitHub MCP tools: `run_workflow` on `branch_walkforward.yml`,
  `season_sim_arms.yml` and `scheduled_pipeline.yml`; artifacts `walkforward-summary` and
  `season-sim-summary`; run logs for the `ARGS:` and `DB_CACHE_KEY:` lines.
- **No new external services and no secrets beyond those the workflows already use.**
- **Frozen surfaces:**
  - the JSON shapes of `data/dashboard/*.json` read by the app (`projections_latest.json`,
    `leaderboard_latest.json`, `app_model_team.json`);
  - the walk-forward summary keys existing tools read. Only add keys, never rename.

## Edge Cases & Failure Handling
- A player with no ownership row at step G → no crowd adjustment. Missing data is never a 0.
- Early-season steps with fewer than `min_rows` training rows → no adjustment, logged in the run.
- Double gameweeks: the fitted adjustment is per gameweek. It is split across a player's fixtures
  in proportion to each fixture's EP, so `ep_gameweek_outputs` sums match.
- A template solve with status not in (`optimal`, `timelimit`) → that gameweek's template points
  are None, and R6 pairs only gameweeks present in both runs, as `beats_baseline` does.
- An incomplete walk-forward (the time budget hit) → `compare_arms.py` refuses (existing
  behaviour). Re-run; never judge a partial run.
- A flaky Actions failure before any step ran → re-run once. A second failure is real.

## Risks, Landmines & Adaptations
- **Point-in-time leakage through ownership** (a prior leak was fixed in #214, see the λ seed note
  in `data/recalibration/seeds_lambda_2026-10-01.json`) → Phase 1 verifies [A2]: the
  `selected_by_percent` and transfer values in row gw = G−1 are knowable before G's deadline. Every
  crowd feature reads gw < G only, with an R5 leak test.
- **The crowd prior turns the model into the template.** Gains would then cap at the template
  level, short of G2 → it is fitted as a residual correction, not "pick by ownership". Phase 2's
  overlap metric watches whether the model XI converges to the template XI. L3–L5 and Phase 10
  carry the push beyond.
- **The noise floor.** A 74-gameweek paired SE is about 0.4 pts/GW [A6]. Levers worth < 0.4 can't
  pass alone → bundle small, individually plausible levers into the Phase 8 combined arm. Never
  re-run an arm hoping for a better draw.
- **Overfitting by reusing 74 gameweeks across many arms** → declare before running (R8), cap
  batches at 3 arms, and confirm out of sample in the season sim (R6) and on live 2026-27
  gameweeks (Phase 11).
- **Runtime and compute.** A two-season walk-forward takes ~40 min (verified: D2 runs, 21 and 39
  min). A season-sim chunk may approach the 345-min timeout [A7] → split chunks further (e.g.
  `2-10,11-19,20-29,30-38`) if a job times out.
- **Concurrent edits.** Another session may edit `scripts/compare_arms.py`,
  `scripts/walkforward_summary.py` or `src/fpl_quant/backtest.py` → merge `origin/master` into
  the worktree branch before every PR. Never force-push over others' commits.
- **The live projections change shape for app users** (the crowd adjustment moves EP) → only after
  R6 and the season-sim pass, with a live check comparing `projections_latest.json` before and
  after.

## Assumptions Ledger
| ID | Assumption | Basis | Blast radius if wrong | Check |
|----|------------|-------|-----------------------|-------|
| A1 | A top-100k season needs about +9 pts/GW over FPL's average, net of hits | [assumed: general FPL knowledge, not measured] | G3's number moves | Phase 1: mean GW points of the 60k–140k rival band vs the FPL average for the same 2026-27 GWs, from `fact_rival_squad_sample` |
| A2 | `selected_by_percent` and transfer values in row gw = G−1 are knowable before G's deadline | [assumed: the snapshot is taken after G−1's deadline] | L1 is leaky; any gain is fake | Phase 1: compare FPL-Core-Insights snapshot timestamps with deadlines, plus the R5 leak test |
| A3 | `transfers_in_event`/`transfers_out_event` are populated for 2024-25 and 2025-26 | [assumed: ingested by `reconcile.py`, coverage unknown] | L1 loses the momentum features | Phase 1: count non-null rows per season in the cached DB (a step in a diagnostic workflow run) |
| A4 | `workflow_dispatch` on the worktree branch's ref runs that branch's code | [assumed: standard GitHub Actions behaviour] | Arms would run master's code | Phase 1: dispatch the control on the branch and check the run's `head_sha` |
| A5 | The template captain is the most-owned XI player (`solve` picks the max-mu captain) | [assumed: `solve` maximises mu with captain doubling] | The template baseline is slightly off | Phase 1 unit test on a seeded pool |
| A6 | The paired SE over 74 GWs is about 0.4 pts/GW | verified on 2025-26 diffs plus the 2024-25 `beats_crowd` proxy (D2) | R6's power differs | Phase 1 measures it on the control-vs-control noise run |
| A7 | A season-sim chunk of about 18 GWs fits within 345 min | [assumed] | Jobs time out | Phase 1 season-sim control; split chunks if needed |
| A8 | The executor has the owner's standing authority: merge green PRs, promote a lever that passes R6 and the season sim | (user) standing instructions this session: "keep merging new prs", "if an arm wins just promote it and merge" | It would need to stop for approval | none: the owner can revoke |
| A9 | Two control runs on the same commit and DB give identical squads | partly verified: D1 and D2 controls matched in 2025-26 | The noise floor is mis-stated | Phase 1 runs two controls (noise run) |

## Open Items (none blocking)
- **Elite ownership for 2026-27.** Proceed without it; revisit when a full season is sampled.
- **Showing the crowd adjustment in the app.** Proceed with EP fields unchanged.

## Verification
- `PYTHONPATH=src python -m pytest tests/ -q` is green, and `ruff check .` and `mypy src/` are
  clean, before every PR.
- `PYTHONPATH=src python scripts/compare_arms.py control.json arm.json --rule r6` prints a PASS
  or FAIL with (i) and (ii), plus both runs' template gaps.
- Each batch's report section lists run ids, the `ARGS:` mapping, `db_cache_key` and the verdict.
- Live: after promotion, run `scheduled_pipeline.yml`, then diff the new
  `data/dashboard/projections_latest.json` against the pre-promotion copy, and read
  `leaderboard_latest.json` honest losses after the next weekly backtest.
- The owner confirms done by reading the programme report's headline table: model vs template
  gap, season-sim net pts/GW vs the FPL average and top-100k pace, and live rank trend.

## Build Phases
- [ ] Phase 0: Set up the worktree
      Done when: `git -C ../FPL-Analyser-btt status` shows branch `claude/beat-template` tracking
      origin, and master contains PR #260 (`grep -n squad_points scripts/walkforward_summary.py`
      matches).
      Steps:
      - `git fetch origin master && git worktree add ../FPL-Analyser-btt -b claude/beat-template origin/master`
        (or the host's EnterWorktree).
      - If PR #260 is not merged, stop and merge it first (it is the pairing R6 needs).
      - Create `docs/reports/2026-10_beat_the_template.md` with the R6 rule text and the finish
        line, then commit.
      Covers: R8

- [ ] Phase 1: Build the yardstick (template squad, diagnostics, `--rule r6`) and measure the baseline
      Done when:
      - the new tests pass;
      - two control runs on the branch (both seasons, same commit) complete;
      - `compare_arms.py ctrlA ctrlB --rule r6` shows m = 0.00 [A9], or the noise is recorded;
      - the report records the baseline gap to the template with its SE, the diagnostics table,
        the A1/A3 measurements, and a season-sim control's net pts/GW vs the FPL average.
      Steps:
      - Write failing tests first: template squad metrics on a seeded run (R1), summary fields and
        diagnostics (R2), `--rule r6` PASS/FAIL/refusals (R3), and the template captain [A5].
      - Implement R1–R3 by reusing `_naive_candidate_pool`, `squad_optimizer.solve` and
        `_realized_solve_points` exactly as `beats_baseline` does. Do not change control behaviour.
      - Verify A2 (snapshot timing) and A3 (transfer coverage) with a diagnostic script run in
        Actions on the cached DB (pattern: `.github/workflows/diagnose_premium_returns.yml`).
      - Measure A1 from `fact_rival_squad_sample` on the live DB: mean GW points of the 60k–140k
        band minus FPL's average for the same GWs. Write the derived G3 number into the report.
      - Dispatch `branch_walkforward.yml` twice with args `""` on ref `claude/beat-template` [A4],
        and `season_sim_arms.yml` with arms `control:` for 2025-26.
      - Open a PR, merge on green, and record the numbers.
      Covers: R1, R2, R3, R8; checks: A1, A2, A3, A4, A5, A6, A7, A9
      Critique: important phase. After building, hand the result plus this phase's Done-when and
      Covers lines to a blind critic (a fresh sub-agent if available, else a fresh-eyes reread).
      The critic defaults to FAIL and returns PASS/FAIL, strengths, weaknesses and fixes. Apply
      blocker fixes and repeat, max 3 rounds. Done only on PASS.

- [ ] Phase 2: Decompose the gap and declare the lever order
      Done when: the report has the decomposition table (overlap, model-only vs template-only
      points, captain gap, £ per band, curse residuals) and an ordered lever queue derived by the
      rules below, committed before any lever run.
      Steps:
      - Compute from the Phase 1 control summaries; do not run a new batch.
      - Order rule: L1 runs first, whatever the decomposition.
        - L2 next if the model XI spends less on £7.0m+ than the template XI by ≥ £3m on average.
        - L4 before L3 if the model XI's realized − projected is ≤ −0.3 pts/player-GW and the
          template XI's is ≥ 0.
        - L5 is included if the captain gap (most-owned captain minus model captain, realized) is
          ≥ +0.5 pts/GW.
      Covers: R8

- [ ] Phase 3: L1, the crowd prior
      Done when: the batch is declared, run, judged with `--rule r6`, and recorded. On PASS it
      queues for Phase 8; on FAIL the report records why and the lever family's fail count.
      Steps:
      - Write failing tests first: the R5 leak test, missing-ownership-no-change, `min_rows` gate,
        and the flag mapping in both scripts (R4).
      - Implement `crowd_prior_params` and the per-step fit. Ridge-shrunk linear residual model on
        player-gameweek rows with gw < G (this season, plus earlier loaded seasons), with features:
        - ownership percentile within position;
        - last gameweek's net transfers per owner;
        - each interacted with the price band.
        
        Apply it to EP per gameweek, split across fixtures.
      - Add `--crowd-prior` to `run_walkforward.py` and `run_season_sim_arm.py`.
      - Declare the batch in the report, commit, then dispatch control and the arm on the branch.
      Covers: R4, R5, R8
      Critique: important phase (leakage risk). Run the blind-critic loop as in Phase 1, focusing on
      the leak test.

- [ ] Phase 4: L2, the position × price-band bias correction
      Done when: declared, run, judged with `--rule r6`, and recorded.
      Steps:
      - Write failing tests first: gw < G only, shrinkage toward 0 with `pseudo_rows`, and the flag
        in both scripts.
      - Implement `band_bias_params`: per (position, band) the mean residual over rows with gw < G,
        times n / (n + pseudo_rows), added to EP.
      - Declare, commit, then dispatch control and `--band-bias` (and `--crowd-prior --band-bias`
        if L1 passed).
      Covers: R4, R8

- [ ] Phase 5: L3, elite de-compression (only if Phase 2 or the latest diagnostics still show
  £9.0m+ under-projected by ≥ +0.1 after the levers that passed)
      Done when: declared, run, judged with `--rule r6`, and recorded, or the report says why it
      was skipped.
      Steps:
      - Implement `elite_residual_params`: a player's own mean residual over his last N gameweeks
        with gw < G, shrunk by n / (n + k). Restricted to players with ≥ 900 minutes this season
        and the previous one.
      - Tests (leak, shrinkage, flag), then declare and run.
      Covers: R4, R8

- [ ] Phase 6: L4, the optimizer's-curse correction (only if the Phase 2 rule triggered it)
      Done when: declared, run, judged with `--rule r6`, and recorded, or skipped with the reason.
      Steps:
      - Implement `selection_shrink_params`: before the solve, mu' = mu − κ·sd·(1 − ownership share
        within position), κ from the param, sd from the existing uncertainty model.
      - Tests (no effect at κ = 0, monotonic in sd, flag), then declare and run with κ ∈ {0.25, 0.5}
        as two arms.
      Covers: R4, R8

- [ ] Phase 7: L5, captaincy with the crowd (only if the Phase 2 rule included it)
      Done when: declared, run, judged with `--rule r6`, and recorded, or skipped with the reason.
      Steps:
      - Implement `captaincy_crowd_params`: captain = argmax over XI of EP + δ·captaincy EO
        (`ownership.compute_eo_for_pool`), δ from the param.
      - Tests, then declare and run.
      Covers: R4, R8

- [ ] Phase 8: Combine the passing levers and confirm in the season simulation
      Done when: the combined arm passes `--rule r6` against a fresh control, AND
      `season_sim_arms.yml` (2025-26, both chunks) shows paired net points m ≥ 0 against control.
      Both are recorded with the season-sim net pts/GW against the FPL average and the G3 target.
      Steps:
      - Declare the combined arm (all levers that passed R6 alone), commit, then dispatch the
        walk-forward control and combined arm.
      - On PASS, dispatch `season_sim_arms.yml` with arms `control: ; combined: <flags>`.
      - If the season sim fails (m < 0), do not promote. Record it and return to the lever queue.
      Covers: R3, R6, R8
      Critique: important phase (it gates live promotion). Run the blind-critic loop on the
      verdict write-up against R6's exact text.

- [ ] Phase 9: Promote live and check
      Done when:
      - the promotion PR is merged with `tests/test_live_switch_wiring.py` green;
      - `scheduled_pipeline.yml` succeeds;
      - the report has a before/after diff of `projections_latest.json` (top-50 by EP, mean EP by
        price band).
      Steps:
      - Copy the current `data/dashboard/projections_latest.json` aside.
      - Implement R7 for each promoted lever (seed, `RECALIBRATABLE_VERSION_ARGS`, call sites,
        wiring test, `--no-<lever>` flags). Run the full suite.
      - Open the PR, merge on green, dispatch `scheduled_pipeline.yml`, then diff the projections.
      Covers: R7
      Critique: important phase (live, real users). Run the blind-critic loop on the wiring diff
      before merging.

- [ ] Phase 10: Re-tune the transfer planner and chips under the new projections (only once G2 holds)
      Done when: each declared season-sim batch is judged (paired net points m − SE ≥ 0 against
      control over 2025-26) and recorded. Promotions follow R7.
      Steps:
      - Use the existing season-sim flags (`--transfer-threshold`, `--multi-transfers`,
        `--chip-timing`, `--chip-wait`, `--chip-option-value`), one per arm, at most 3 per batch.
      - Declare, run and judge each batch.
      Covers: R6, R7, R8

- [ ] Phase 11: Track live and decide whether to continue
      Done when: the report's programme headline table is current (template gap, season-sim pace
      vs G3, leaderboard honest losses, `rank_autopsy.json` estimated rank), and the stop rule has
      been applied: continue with the queue, or stop and report to the owner.
      Steps:
      - After each weekly backtest and rank-tracking run, update the table.
      - If G2 and G3 hold and the live rank trends toward the top 100k, close the programme with a
        summary. If the queue is empty and G2 is unmet, stop and report.
      Covers: R8
