# Plan: premium and in-form attacking returns (round after the breakout fix)

One-line goal: fix the walk-forward's remaining attacking-return errors for £9.0m+ players and for
in-form established starters. Each arm changes one model setting (one param version), and is judged
against a rule declared in advance. A change goes live only if it passes. If none passes, the round
ends in a report and nothing changes live.

## Classification

Track: bug fix (diagnosis before build). The model is known to mis-predict two groups. The plan
measures where, tests the levers already in the code, and builds at most one new lever.
Parked secondary asks: Haaland's P(60+) Community Shield quirk (see Out of Scope).

## Interview Ledger

- Q1 scope → premiums **and** the in-form scoring-rate gap, under one diagnosis (accepted Recommended).
- Q2 promotion rule → R4 retargeted, called **rule R5** below (accepted Recommended).
- Q3 rule R5 check (iii) → the **sum of absolute component residuals**, not the band total (accepted
  Recommended, after batch B1 showed offsetting errors).
- Q4 approval → see closing turn.
Count: 4 of 14.

## Goal & Success Criteria

- **Done means one of two outcomes:**
  - (a) an arm passes rule R5, goes live, and the next scheduled pipeline's
    `projections_latest.json` shows the named players' EP changed;
  - (b) no arm passes, and the report records every arm with its numbers and why it failed.
- Every arm's target group is written in `docs/reports/2026-10_premium_and_form_returns.md`
  **before** its batch is dispatched.
- The in-form group is defined in code (`src/fpl_quant/form_groups.py`) from data available before
  each gameweek's deadline. Its constants are fixed in Phase 1, before any arm is judged on it.

## Current State

- **What is live.** A4 is live: the current-season start-rate blend and the P(60+|started) blend
  (verified: `src/fpl_quant/backtest.py:2608-2611`). The price-anchored rate prior is also live
  (verified: `backtest.py:2605`).
- **Where scoring rates come from.** Goal and assist rates come from FPL's own per-90
  `expected_goals` / `expected_assists` in `fact_player_season_stats`. They are shrunk toward a
  per-position line fitted on price (verified: `src/fpl_quant/expected_points.py:186-234`, `:390-398`).
  - 2025-26+ rows carry season-total `minutes` and `expected_goals`.
  - 2024-25 rows carry only per-90 rates with NULL minutes. Minutes come from
    `fact_player_match_stats` (verified: `expected_points.py:174-177`, `:251-254`).
- **How realized components are counted.** In the backtest, realized components use the match feed
  (`fact_player_match_stats`), not FPL's own assist count (verified: `backtest.py:535-552`,
  `:1040-1057`).
  - "Other" = event_points − appearance − goals − assists − clean sheet. So any assist FPL awards
    that the match feed lacks lands in "other".
  - A code comment calls match-feed assists "close to FPL's" (`expected_points.py:489`). That is
    unchecked.
- **Components are only stored as aggregates.** The backtest stores them only as per-gameweek segment
  means in `backtest_metrics`. `walkforward_summary.py`'s `price_band` reads those aggregates
  (verified: `backtest.py:1040-1057`, `scripts/walkforward_summary.py:309-314`). Per-player-step
  components are **not stored**, so a new group's components must be recomputed after the run.
  - Everything needed for that exists on the main tables after a run:
    - `ep_outputs` per fixture (`fixture_match_id`; verified: `schema/0004_m3_expected_points.sql:25`);
    - `ep_gameweek_outputs` sums;
    - `ep_model_versions.scoring_matrix_params_version` (same file, line 13);
    - `fact_player_match_stats`;
    - `fact_player_season_stats.event_points`.
  - `breakout_cuts()` already recomputes the total residual this way (verified:
    `walkforward_summary.py:224-279`).
- **Bonus data.** No measured-bonus column in `fact_player_match_stats` or `fact_player_season_stats`
  (verified: `schema/0001_core_schema.sql:94-138`). Raw CSV tables are named
  `raw_<season>_<slug>` (verified: `src/fpl_quant/ingest_csv.py:30-32`) and listed in
  `fact_raw_ingestion_log.raw_table_name` (verified: `src/fpl_quant/reconcile.py:24-28`).
- **Bonus arms (batch B1) are judged.** Cached DB `duckdb-38085065975`, commit e452e49, all
  `--seasons 2025-2026` (verified: runs 38099909460 control, 38099910627, 38099912176, mapped by the
  `ARGS:` log line):

  | 2025-26 | control | `--bps-calibration-k 450` | `--bps-tau 7` |
  |---|---|---|---|
  | squad pts vs control (m ± SE per GW) | — | −0.14 ± 0.87 | −1.03 ± 0.76 |
  | EP MAE | 1.0663 | 1.0658 | 1.0655 |
  | 9.0m+ residual: appearance / goals / assists / clean sheet / other | −0.004 / −0.267 / +0.227 / −0.007 / +0.158 | … / +0.166 | … / +0.030 |
  | 9.0m+ component sum (rule R5 limit 0.441) | 0.662 | 0.670 | 0.534 |
  | breakout residual (limit +0.182) | +0.108 | +0.111 | +0.107 |
  | captain pts/GW (forward share) | 5.89 (54%) | 5.84 (43%) | 5.57 (68%) |

  - Both fail rule R5. "…" means unchanged from control.
  - Premium goals are **over**-predicted, while assists and bonus are **under**-predicted. The errors
    partly cancel.
  - `--bps-tau 7` removed the bonus error, but it over-ranked premium forwards, whose goals are
    already over-predicted. Captain points fell.
- **Existing opt-in levers.** These flags already exist (verified: `scripts/run_walkforward.py:20-58`):
  `--finishing-prior-xg`, `--assist-prior-xa`, `--k-minutes`, `--rate-current-season-weight`,
  `--rate-season-decay`, `--no-rate-price-anchor`, `--bps-tau`, `--bps-calibration-k`.
  - **Finishing and assist ratios.** `finishing_ratios()` scales each player by his own
    (goals+prior)/(xG+prior) and (assists+prior)/(xA+prior). It uses FPL totals for 2025-26+ and
    match-feed counts for 2024-25 (verified: `expected_points.py:476-520`).
  - **Their history.** The finishing prior flipped sign between data windows (verified:
    `docs/reports/2026-10_open_issues.md` §5). The assist prior was flat (verified:
    `docs/reports/2026-10_live_path_diagnosis.md:204`). Both were measured before the price anchor
    and A4.
- **Which settings already reach the live path.** Only the rate settings: `rate_shrinkage_params`
  (`k_minutes`) and `rate_prior_params` (anchor, current-season weight, season decay). They are in
  `RECALIBRATABLE_VERSION_ARGS` and in `tests/test_live_switch_wiring.py`'s `REQUIRED` as RATE and
  PRIOR (verified: `backtest.py:2593`, `:2605`; `test_live_switch_wiring.py:42-55`).
  `projections.build_projections` takes only those two of the rate settings (verified:
  `src/fpl_quant/projections.py:61-76`). `finishing_skill_params` is read only by backtest,
  expected_points, monte_carlo and uncertainty. It is **not** on the live path (verified:
  `grep -rln finishing_skill_params_version src scripts`).
- **k_minutes history.** Live `k_minutes` is 900 (since 2026-09-09). The overnight report recommended
  rejecting that confirmation (verified: `docs/reports/2026-10_overnight_run.md:58`). In the
  rate-prior batch, `--k-minutes 450` gave +0.89 ± 0.49 pts/GW with no MAE gain (verified:
  `docs/reports/2026-10_rate_prior.md`).
- **Mbeumo (live, 2026-27 GW6).** Model goal rate 0.43/90 against this season's xG 0.61/90, a rates
  gap of about +0.61 pts/GW (verified: `docs/reports/2026-10_breakout_players.md:10-17`).
- **Breakout limit.** The breakout limit +0.182 is ⅔ × the before-fix breakout residual +0.272,
  using the declared `breakout` group (verified: `docs/reports/2026-10_breakout_players.md`, follow-up
  2 table, "limit +0.182").

## Scope (v1)

1. **Phase 1:** one data diagnostic, plus the in-form/cooling groups in code, with sizes fixed.
2. **Phase 2:**
   - a shared `realized_components()` helper;
   - per-group component blocks in `walkforward_summary.py`;
   - rule R5 in `compare_arms.py`;
   - if Phase 1 says so, FPL's own assist count for the realized split.
3. **Phase 3, batch B2:** five existing levers against a fresh control.
4. **Phase 4:** at most one new lever (batch B3), or the combined run if two B2 arms pass.
5. **Phase 5:** wire the winner into the live path if it isn't already, then promote it, or write the
   report and stop.

## Out of Scope & Parked Items

- **Haaland P(60+) / Community Shield quirk.** Parked. Control's 9.0m+ appearance residual is
  −0.004 (verified: run 38099909460). The trigger to come back is a 9.0m+ appearance residual above
  +0.05 in a later control.
- **Captaincy rule changes.** Parked. The captain gap follows the EP ranking
  (`docs/reports/2026-10_open_issues.md` §7).
- **Risk aversion `lambda`, chip policy, minutes model.** Not touched.
- **Expert or article evidence changing live projections.** Never (user rule). It may be shown in the
  report only.
- **2024-25 as a decider.** It is reported as the cold-start stress test, never decides (convention:
  `docs/reports/2026-10_open_issues.md`, issue 4 decision).
- **A second new lever in this round.** Only one; anything more is the next round.

## Approach

- **Measure, then test, then build.** B1 showed premium errors that partly cancel.
  - Phase 1 separates "the measurement is off" (match-feed vs FPL assists) from "the model is off"
    (FPL conversion by band), and fixes the in-form group.
  - Phase 2 makes the judge able to see any group's components.
  - B2 tests the five existing levers.
  - Only a cause that Phase 1 and B2 agree on earns the one new lever (B3).
- **Hypotheses**, each with what kills it:
  - **H1:** the price anchor over-extrapolates premium goals. *Phase 1 measures:* FPL goals/xG at
    9.0m+. *B2 tests:* the `--no-rate-price-anchor` arm. Killed if that arm doesn't lower the 9.0m+
    goals residual's size **and** FPL goals/xG at 9.0m+ ≥ 1.0. Verdict in Phase 3.
  - **H2:** FPL awards premiums more assists than xA implies. *Phase 1 measures:* FPL assists/xA by
    band. Killed if the 9.0m+ ratio ≤ 1.05. Verdict in Phase 1.
  - **H3:** part of 9.0m+ "other" is FPL-only assists, not bonus. *Phase 1 measures:* FPL vs
    match-feed assists at 9.0m+. Killed if they agree within 3%. **If not killed**, Phase 2 switches
    the realized assist count to FPL's own (Requirement RQ2b). Verdict in Phase 1.
  - **H4:** multi-season shrinkage drags in-form starters' rates down. *B2 measures:* control's
    `in_form` component sum. Killed if it is < 0.15. Verdict in Phase 3.
- **Executor's choice:** internal layout of `form_groups.py`, the diagnostic's JSON shape, helper
  names, and set-based SQL vs per-player loops (keep the summary under 10 minutes on CI).

## Requirements

- **RQ1. The diagnostic.** WHEN `scripts/diagnose_premium_returns.py` runs on the cached DB THE
  SYSTEM SHALL write JSON with:
  - (a) **Assist sources.** For 2025-26 league matches, per price band: FPL season assists (each
    player's last `fact_player_season_stats` row) vs the sum of match-feed assists. Give the ratio
    and the count of players who differ.
  - (b) **Conversion.** For 2025-26 (and 2024-25 reported), per price band, players with ≥ 900 league
    minutes: pooled FPL goals/xG and assists/xA.
  - (c) **Bonus column.** Every `raw_*` table named in `fact_raw_ingestion_log.raw_table_name` that
    has a column whose name contains `bonus`, with the column names.
  - (d) **Group sizes.** In-form/cooling sizes (declared and widened) per season, and the live
    season's current members.

  The band is set by the player's `now_cost` on his last row that season, using the existing band
  edges <5.0 / 5.0-7.0 / 7.0-9.0 / 9.0+. Acceptance: a synthetic-DB test checks (a), (b) and (d) by
  hand. The workflow artifact holds all four blocks.
- **RQ2. Realized components helper.** `realized_components(con, season, gw, ep_model_version,
  player_uids=None)` → `{player_uid: {appearance, goals, assists, cleansheet, other}}`.
  - It is moved out of the inline loop at `backtest.py:1040-1057`, and both that loop and the summary
    call it.
  - Inputs:
    - `ep_outputs` fixtures (`fixture_match_id`) and `ep_gameweek_outputs` sums for the version;
    - `fact_player_match_stats` per fixture;
    - scoring points via `ep._sm` with the version's `scoring_matrix_params_version`;
    - `event_points` from `fact_player_season_stats` (season, gw).

  Acceptance:
  - (i) existing backtest tests stay green, so the metrics are unchanged;
  - (ii) a synthetic test checks a recomputed `price_band=9.0+` block, aggregated as below, against
    the stored `ep_*_calibration_mean_resid:price_band=9.0+` metrics to 1e-9.

  **RQ2b.** (Only if H3 is not killed.) A `realized_assist_source` argument, `"fpl"`, takes each
  player's gameweek assists as the difference of his cumulative `fact_player_season_stats.assists`
  between gw G and his previous row. It applies to 2025-26+; 2024-25 keeps the match feed. It
  becomes the default for the backtest metrics and the summary. The B2 control then runs with it,
  so every later comparison is consistent.
- **RQ3. Summary blocks.** WHEN `walkforward_summary.py` summarises a run THE SYSTEM SHALL add:
  - `form_groups` → `{season: {in_form, in_form_widened, cooling, cooling_widened: {n_player_steps,
    mean_resid, components: {…5…}, component_abs_sum}}}`;
  - `component_abs_sum` on every `price_band` entry.

  How the numbers are aggregated:
  - Each component is the mean over gameweeks of that gameweek's mean over the group's player-steps.
    This matches how `price_band` aggregates (`_mean_metric`).
  - `component_abs_sum` = Σ |component|.

  Acceptance: a synthetic-DB test with hand-computed numbers.
- **RQ4. The in-form group** (`form_groups.classify(con, season, gw, deadline)`). A player-step
  (S, G) is IN_FORM when, using only rows dated before G's deadline:
  - (a) he started ≥ 3 of his club's last 4 league matches (the same counting as `breakout.py`);
  - (b) his this-season rate is high enough:
    - he has ≥ 360 league minutes in S;
    - his this-season (xG+xA)/90 is ≥ 1.30 × his earlier-season (xG+xA)/90;
    - "this season" is read from his latest `fact_player_season_stats` row with gw < G;
    - "earlier" comes from `ep._player_rate_pool(con, uid, earlier_loaded_seasons)`, which handles
      2024-25's per-90-only rows;
    - the earlier seasons must hold ≥ 900 minutes;
  - (c) he is not a goalkeeper, and not in `breakout` or `breakout_promoted` under the declared
    (unwidened) breakout rule.

  The other cases:
  - **COOLING:** the same, with ratio ≤ 1/1.30.
  - **Widened:** ratio 1.20 (and 1/1.20).
  - **No earlier loaded season:** returns None.
  - **Minimum group size:** 150 player-steps in 2025-26. Phase 1 records whether the declared or
    widened group is used, before any arm runs.

  Acceptance:
  - unit tests for each clause;
  - a test where the earlier season has NULL `minutes` (2024-25 shape) and the group is not empty;
  - a **leakage test**: rows with gw ≥ G change nothing.
- **RQ5. Rule R5 in `compare_arms.py`.** WHEN `compare_arms.py control.json arm.json --rule r5
  --target <price_band=9.0+ | in_form>` runs THE SYSTEM SHALL print PASS only if all hold on
  2025-26:
  - (i) paired squad points m ≥ −0.25 and m + SE ≥ 0;
  - (ii) EP MAE ≤ control + 0.002;
  - (iii) the target group's `component_abs_sum` ≤ ⅔ of control's;
  - (iv) the `breakout` group's mean residual ≤ +0.182.

  More behaviour:
  - For `in_form`, it uses the group Phase 1 fixed (declared or widened). It fails with a clear
    reason when the block is missing.
  - It refuses runs from different `db_cache_key`s, as today.
  - The existing default rule and `--baseline` keep working.

  Acceptance: tests for a pass, each failing check, the missing block, and the CLI flags.
- **RQ6. Promotion.** WHEN an arm passes rule R5 with m ≥ 0 THE SYSTEM SHALL go live through the
  live-switch pattern, with every live call site wired (Phase 5). WHEN it passes with m < 0, ask the
  owner before merging (user rule).
- **RQ7. Declaration.** Every arm's target group is in the report before its batch is dispatched.
  Every run passes `--seasons 2025-2026`, because `price_band` has no season split.

## Key Decisions

- **Rule R5** = R4's (i) and (ii), plus (iii) a component-sum cut of ≥ ⅓ on the arm's declared
  target, plus (iv) breakout residual ≤ +0.182. (user, Q2 + Q3)
- **Judging season:** 2025-26 decides; 2024-25 is reported only (verified: issue 4 convention).
- **In-form uses underlying stats (xG+xA), not output (G+A).** [assumed: less noisy, and it matches
  Mbeumo's case - if wrong: B2's in-form sum stays < 0.15 and H4 is killed. One re-definition on G+A
  is allowed, declared before any re-run.]
- **Thresholds:** 1.30 (widened 1.20) for in-form, a group of at least 150 player-steps, and a
  component sum of at least 0.15 before the in-form half is pursued. [assumed: sized for a few
  hundred player-steps in 2025-26 - if wrong: use the widening, else stop that half]
- **One target per arm, declared before results:**
  - **9.0m+:** finishing prior, assist prior, no-anchor.
  - **in_form:** recency, k_minutes.

  Recency is one `rate_prior_params` version holding two keys, so it counts as one setting.
  [assumed: each lever's mechanism points at one group - if wrong: both groups are reported, and
  targets are never changed after results]
- **Batch budget:** B2, then one B3. If two B2 arms pass, B3 is their combined run and the new lever
  waits for the next round. [assumed: proportionate to a +0.66-point error sum - if wrong: the owner
  extends]
- **`--k-minutes 450`** may be promoted if it passes rule R5, even though that reverses the
  2026-09-09 confirmation (verified: the overnight report recommended rejecting it).
- **`--no-rate-price-anchor`** passing would mean reverting the 2026-10-07 promotion. If it passes,
  that is promoted like any arm. If it fails, it still decides H1.

## Data & State Changes

- **No schema changes.**
  - Arm param versions come from `run_walkforward.py` flags (existing behaviour).
  - RQ2b changes how a metric is computed, not a table.
- **Promotion** creates or confirms a live version through `materialize_confirmed_seeds`. It also
  adds a `RECALIBRATABLE_VERSION_ARGS` entry when the family is new to the registry.
- **Rollback:** revert the promotion commit. The old versions stay in the param tables.

## Interfaces, Integrations & Credentials

- **GitHub Actions only.**
  - `branch_walkforward.yml` (`args` input, one run per arm, `walkforward-summary` artifact).
  - A new `diagnose_premium_returns.yml` copies `diagnose_breakouts.yml`'s cache-restore steps.
  - `scheduled_pipeline.yml` for the live check.
- **Credentials:** none beyond the existing workflows.
- **Fixed contract:** existing `walkforward_summary.json` keys that `compare_arms.py` reads must not
  change. New keys are only added.

## Edge Cases & Failure Handling

- **The in-form group is under 150 even widened** → that half stops; record it. Only 9.0m+ arms go
  on.
- **A batch spans more than one `db_cache_key`** → `compare_arms.py` refuses. Re-dispatch the whole
  batch.
- **`progress.complete` is false** → not judged. Re-dispatch that run once.
- **Two B2 arms pass** → B3 is their combined run. Promote the combined run if it passes; otherwise
  promote the single arm with the larger m.
- **No raw bonus column** → H3 is judged on assists alone, and "other" stays a residual.
- **The scheduled pipeline is late after promotion** (known cron delay) → dispatch it by hand.

## Risks, Landmines & Adaptations

- **Offsetting component errors** (found in B1) → check (iii) uses the component sum (Q3), and
  Phase 1 separates measurement error from model error.
- **Realized assists come from the match feed** → if Phase 1 finds a gap over 3% at 9.0m+, RQ2b
  switches to FPL's count before B2. That is a declared action, not a judgement call.
- **Two candidate levers aren't on the live path** (`finishing_skill_params`, any new lever) →
  Phase 5 has an explicit wiring step with a call-site list and a live-EP Done-when.
- **2024-25 has only per-90 rows** → in-form "earlier" rates come from `_player_rate_pool`, with a
  test of that shape.
- **Points noise** (SE ≈ 0.8 pts/GW over 37 GWs) → the points gate is a floor. The calibration and
  component gates carry the decision.
- **Selection on form** (regression to the mean) → the group uses pre-deadline underlying stats, a
  leakage test guards it, and the cooling group is reported so over-predicting faders shows.
- **Baselines move between DB caches** → each batch has its own control. Compare only within a
  batch.

## Assumptions Ledger

| ID | Assumption | Basis | Blast radius if wrong | Check |
|----|------------|-------|-----------------------|-------|
| A1 | Match-feed assists ≈ FPL assists | code comment `expected_points.py:489` | component split mis-attributed; wrong lever | Phase 1 RQ1(a) → RQ2b |
| A2 | A `raw_*` table may carry per-GW bonus | the source CSVs carry many FPL fields | bonus not separable from DefCon/saves | Phase 1 RQ1(c) |
| A3 | In-form = (xG+xA)/90 ≥ 1.30 × earlier | Mbeumo's live gap; underlying stats are steadier | empty or noisy group | Phase 1 sizes + tests |
| A4 | 150 player-steps is enough | breakout used 200 for a broader rule | judging on too few | widening once, else stop |
| A5 | 2025-26 shows the in-form gap seen live in 2026-27 | same model and data shape | H4 has nothing to fix | B2 control's in-form block |
| A6 | Each lever has one natural target | lever mechanisms | an arm helps the other group | both groups reported for every arm |
| A7 | Recomputed components match the stored metrics | same inputs, same formulas | judge disagrees with backtest | RQ2 acceptance (ii) |

## Open Items (none blocking)

- A narrow P(60+) arm that drops only friendlies and the Community Shield. Proceed without it unless
  a later control's 9.0m+ appearance residual > +0.05.

## Verification

- `PYTHONPATH=src python -m pytest tests/test_form_groups.py tests/test_compare_arms.py tests/test_walkforward_summary.py tests/test_diagnose_premium_returns.py -q`
- The full suite (`PYTHONPATH=src python -m pytest -q`), including `tests/test_live_switch_wiring.py`.
- `PYTHONPATH=src python scripts/compare_arms.py control.json arm.json --rule r5 --target in_form`
  prints the four checks.
- **The owner's check:** the report's B2/B3 tables. After a promotion, the next pipeline's
  `projections_latest.json` shows the new EP for Mbeumo and the 9.0m+ starters, with before and
  after recorded in the report.

## Build Phases

- [ ] Phase 1: Data diagnostic and the in-form groups
      Done when: the `diagnose_premium_returns.yml` artifact holds RQ1's four blocks. The report
      records the H2 and H3 verdicts and whether RQ2b applies. It also fixes the in-form group used
      for judging (declared or widened).
      Steps:
      - Write failing tests first (RQ1 synthetic DB, RQ4 clauses, NULL-minutes earlier season,
        leakage).
      - Add `src/fpl_quant/form_groups.py` (constants at the top, like `breakout.py`).
      - Add `scripts/diagnose_premium_returns.py` and the workflow. Merge, then dispatch.
      - Record the results and the decisions in `docs/reports/2026-10_premium_and_form_returns.md`.
      Covers: RQ1, RQ4; checks: A1, A2, A3, A4
- [ ] Phase 2: Components helper, summary blocks, rule R5
      Done when: the new tests pass, the full suite is green, and the PR is merged. The report
      records B1 and declares B2's arms and targets.
      Steps:
      - Write failing tests first: RQ2(ii), RQ3, RQ5, and RQ2b if it applies.
      - Move the loop at `backtest.py:1040-1057` into `realized_components()` and call it from
        there. If Phase 1 says so, add RQ2b.
      - Extend `walkforward_summary.py` with `form_groups` and `component_abs_sum`, and
        `compare_arms.py` with `--rule r5 --target`.
      Covers: RQ2, RQ2b, RQ3, RQ5, RQ7; checks: A7
      Critique: important phase - after building, hand the diff plus this phase's Done-when and
      Covers lines to a blind critic (fresh sub-agent if available, else a fresh-eyes reread).
      The critic defaults to FAIL and returns PASS/FAIL, strengths, weaknesses and fixes. Apply
      blocker fixes and repeat, max 3 rounds. Done only on PASS.
- [ ] Phase 3: Batch B2 - existing levers
      Done when: all 6 runs complete on one `db_cache_key` with `progress.complete`. Each arm is
      judged with `compare_arms.py --rule r5`, and the report has the table plus the H1 and H4
      verdicts.
      Steps:
      - Dispatch `branch_walkforward.yml` with `--seasons 2025-2026` plus:
        - nothing (control);
        - `--finishing-prior-xg 10` (target 9.0m+);
        - `--assist-prior-xa 30` (9.0m+);
        - `--no-rate-price-anchor` (9.0m+);
        - `--rate-current-season-weight 2 --rate-season-decay 0.5` (in_form);
        - `--k-minutes 450` (in_form).
      - Map runs to arms by the `ARGS:` log line, never by dispatch order.
      - Record both groups and the cooling group for every arm.
      Covers: RQ5, RQ7; checks: A5, A6
- [ ] Phase 4: One diagnosis-led lever, or the combined run
      Done when: the B3 batch (control + arm) is judged and recorded. Or: the report says why no
      single cause was named, and the phase is skipped.
      Steps:
      - If two B2 arms passed: B3 = control + their combined flags. Stop there.
      - Else if H1 or H2 holds: build the **pooled band conversion** lever as an opt-in
        `conversion_params` version with key `pseudo_xg`, flag `--band-conversion K` (K = 20
        declared).
        - Formula: for position p and price band b, over earlier loaded seasons' players in (p, b),
          `r_goals = (Σ FPL goals + K) / (Σ xG + K)` and `r_assists = (Σ FPL assists + K) / (Σ xA + K)`.
        - Applied after shrinkage: `e_goals *= r_goals`, `e_assists *= r_assists`.
        - How it differs from `--assist-prior-xa`: it pools a whole band, so one premium's small
          sample can't drive it, and it moves goals and assists together.
      - Else if B2's recency or k arm fails only (i), and only by m + SE < 0, B3 = that arm plus
        `--bps-tau 7`.
      - Write the failing test first, then the flag in `run_walkforward.py`. Declare the arm and
        target in the report before dispatch.
      Covers: RQ5, RQ7
- [ ] Phase 5: Wire, promote, or stop
      Done when: either (a) the winner is live, the wiring test is green, the PR is merged,
      `scheduled_pipeline.yml` has run, and the named players' EP changed in
      `projections_latest.json` (recorded); or (b) the report's conclusion lists every arm and why it
      failed, and the plan is ticked.
      Steps:
      - If m < 0, ask the owner before anything else.
      - Wire the winner's version:
        - **`k_minutes`, recency, or no-anchor:** already wired (RATE/PRIOR). Seed the new live
          version only.
        - **`finishing_skill_params` or `conversion_params`:**
          - add the kwarg to `expected_points.run`'s callers: `projections.build_projections`,
            `transfer_planner.compute_horizon_ep`, `transfer_planner.run`,
            `decision_engine.recommend_best_move`, `backtest.run`, `backtest.run_season_simulation`;
          - add the reviewed splat builders listed in `tests/test_live_switch_wiring.py`'s
            `REVIEWED_SPLATS`;
          - add a `REQUIRED` entry and a `RECALIBRATABLE_VERSION_ARGS` entry;
          - add a seed in `materialize_confirmed_seeds`.
      - Dispatch `scheduled_pipeline.yml`. Record before/after EP for Mbeumo and the 9.0m+ starters.
      - Tick this plan's phases and write the report's conclusion.
      Covers: RQ6
      Critique: important phase - after building, hand the diff plus this phase's Done-when and
      Covers lines to a blind critic (fresh sub-agent if available, else a fresh-eyes reread).
      The critic defaults to FAIL and returns PASS/FAIL, strengths, weaknesses and fixes. Apply
      blocker fixes and repeat, max 3 rounds. Done only on PASS.
