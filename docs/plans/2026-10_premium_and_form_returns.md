# Plan: premium and in-form attacking returns (round after the breakout fix)

One-line goal: fix the walk-forward's remaining attacking-return errors for £9.0m+ players and for
in-form established starters by changing one setting at a time and judging each change against a rule
declared in advance. A change goes live only if it passes; if none passes, this round ends in a
report and nothing changes live.

## Classification

Track: bug fix (diagnosis before build). The model is known to mis-predict two groups. The plan
measures where first, then tests the levers that already exist, then builds at most one new lever.
Parked secondary asks: Haaland's P(60+) Community Shield quirk (see Out of Scope).

## Interview Ledger

- Q1 scope → premiums **and** the in-form scoring-rate gap, under one diagnosis (accepted Recommended).
- Q2 promotion rule → R4 retargeted, called R5 below (accepted Recommended).
- Q3 R5 check (iii) → judged on the **sum of absolute component residuals**, not the band total
  (accepted Recommended, after batch B1 showed offsetting errors).
- Q4 approval → see closing turn.
Count: 4 of 14.

## Goal & Success Criteria

- Done means one of two outcomes:
  - **(a)** an arm passes R5, goes live through the live-switch pattern, and the next scheduled
    pipeline's projections carry it;
  - **(b)** no arm passes, and the report records every arm with its numbers and why it failed.
- Every arm's target group is written in `docs/reports/2026-10_premium_and_form_returns.md`
  **before** its results exist.
- The in-form group is defined in code (`src/fpl_quant/form_groups.py`) from data available before
  each gameweek's deadline, and its constants are fixed before any arm is judged on it.

## Current State

- **Live model:** A4 is live, meaning the current-season start-rate blend and the P(60+|started)
  blend (verified: `src/fpl_quant/backtest.py:2608-2612`). The price-anchored rate prior is also
  live (verified: same file, `rate_prior_params_version`).
- **Scoring rates come from FPL's own xG/xA.** Goal and assist rates come from per-90 `expected_goals`
  / `expected_assists` in `fact_player_season_stats`. They are shrunk toward a per-position rate that
  rises with price (verified: `src/fpl_quant/expected_points.py:186-234`, `:335`).
- **Measured assists come from a different source.** The backtest splits each gameweek's error into
  components using `fact_player_match_stats` (the match feed), not FPL's own assist count (verified:
  `backtest.py:535-547`, `:1040-1056`).
  - "Other" is the remainder: `event_points − appearance − goals − assists − clean sheet`. Any assist
    FPL awards that the match feed lacks therefore lands in "other", not "assists".
  - A code comment calls match-feed assists "close to FPL's" (`expected_points.py:489`). Nobody has
    checked that this session.
- **No measured-bonus column.** `fact_player_match_stats` and `fact_player_season_stats` have no
  bonus field (verified: `schema/0001_core_schema.sql:94-138`). The model's own `ep_bonus` is stored
  (verified: `schema/0023_gameweek_views.sql`).
- **Bonus arms (batch B1) are judged; both fail R5.** All three runs were on cached DB
  `duckdb-38085065975`, commit e452e49 (verified: runs 38099909460 control, 38099910627 and
  38099912176, each mapped to its arm by the `ARGS:` log line):

  | 2025-26 | control | `--bps-calibration-k 450` | `--bps-tau 7` |
  |---|---|---|---|
  | squad pts vs control (m ± SE per GW) | — | −0.14 ± 0.87 | −1.03 ± 0.76 |
  | EP MAE | 1.0663 | 1.0658 | 1.0655 |
  | 9.0m+ residual, appearance / goals / assists / clean sheet / other | −0.004 / −0.267 / +0.227 / −0.007 / +0.158 | same / same / same / same / +0.166 | same / same / same / same / +0.030 |
  | 9.0m+ component sum (R5 limit 0.441) | 0.662 | 0.670 | 0.534 |
  | breakout residual (limit +0.182) | +0.108 | +0.111 | +0.107 |
  | captain pts/GW (forward share) | 5.89 (54%) | 5.84 (43%) | 5.57 (68%) |

  - Reading the table: premiums' goals are **over**-predicted and their assists and bonus
    **under**-predicted, and these errors partly cancel.
  - `--bps-tau 7` removed the bonus error but over-ranked premium forwards, whose goals are already
    over-predicted. Captain points fell as a result.
- **Existing opt-in levers.** `scripts/run_walkforward.py` already has these flags (verified: its
  docstring, lines 20-58): `--finishing-prior-xg`, `--assist-prior-xa`, `--k-minutes`,
  `--rate-current-season-weight`, `--rate-season-decay`, `--bps-tau`, `--bps-calibration-k`.
  - **Finishing and assist ratios.** `finishing_ratios()` computes (goals+prior)/(xG+prior) and
    (assists+prior)/(xA+prior). For 2025-26 onward it reads FPL's own season totals; for 2024-25 it
    reads match-feed counts (verified: `expected_points.py:476-520`).
  - **Their history.** They were last measured before the price anchor and A4. The finishing prior
    flipped sign between data windows (verified: `docs/reports/2026-10_open_issues.md` §5). The
    assist prior was flat (verified: `docs/reports/2026-10_live_path_diagnosis.md:204`).
- **k_minutes.** Live `k_minutes` is 900 (since 2026-09-09). The overnight report recommended
  rejecting that confirmation (verified: `docs/reports/2026-10_overnight_run.md:58`). In the
  rate-prior batch, `--k-minutes 450` gave +0.89 ± 0.49 pts/GW with no MAE gain (verified:
  `docs/reports/2026-10_rate_prior.md`).
- **Mbeumo (live, 2026-27 GW6).** His model goal rate is 0.43/90 against this season's xG of
  0.61/90, a rates gap of about +0.61 pts/GW (verified: `docs/reports/2026-10_breakout_players.md:10-17`).
- **How arms run.** `branch_walkforward.yml` runs one arm per dispatch from the `args` input and
  uploads `walkforward_summary.json` with `db_cache_key` (verified: the workflow file).
  `scripts/compare_arms.py` judges R4 and the breakout follow-up rule (verified: that file).

## Scope (v1)

1. **One data diagnostic** on the cached DB:
   - FPL vs match-feed assists;
   - whether a raw per-gameweek bonus column exists;
   - FPL goals/xG and assists/xA conversion by price band;
   - in-form group sizes.
2. **The in-form and cooling groups** in code, reported per run by `walkforward_summary.py`.
3. **R5 in `compare_arms.py`** (`--rule r5 --target <group>`).
4. **Batch B2:** the four existing levers against a fresh control.
5. **At most one new lever, batch B3**, built only if steps 1 and 4 point at a single cause.
6. **Then one of:** promote the winner, or write the report and stop.

## Out of Scope & Parked Items

- **Haaland P(60+) / Community Shield quirk.** Parked. Control's 9.0m+ appearance residual is now
  −0.004 (verified: run 38099909460), so minutes is not where the premium error is. The trigger to
  return is a 9.0m+ appearance residual above +0.05 in a later control.
- **Captaincy rule changes.** Parked. The captain gap follows from EP ranking
  (`docs/reports/2026-10_open_issues.md` §7).
- **Risk aversion `lambda`, chip policy, minutes model.** Not touched this round.
- **Changing the live projections from article or expert evidence.** Never (user rule, earlier
  in this project). Expert consensus may be shown in the report only.
- **2024-25.** Reported as the cold-start stress test, never decides (convention from
  `docs/reports/2026-10_open_issues.md`, issue 4 decision).

## Approach

- **Measure, then test, then build.**
  - B1 showed premium errors that partly cancel. Phase 1 therefore separates "the measurement is
    off" (match-feed vs FPL assists) from "the model is off" (FPL conversion by band).
  - Phase 2 adds the in-form cut and R5.
  - B2 then tests the four levers already in the code: two aimed at premium conversion, two at
    in-form rates.
  - Only a cause that Phase 1 and B2 agree on earns a new lever (B3).
- **Hypotheses.** Each is ranked, with the evidence that would kill it:
  - **H1:** premium goals are over-predicted because the price anchor's goal slope exceeds how often
    premiums actually convert. Killed if the Phase 1 FPL goals/xG ratio at 9.0m+ is ≥ 1.0 *and* the
    finishing-prior arm doesn't lower the 9.0m+ goals residual.
  - **H2:** premium assists are under-predicted because FPL awards more assists than xA implies.
    Killed if FPL assists/xA at 9.0m+ ≤ 1.05.
  - **H3:** "other" at 9.0m+ is partly FPL-only assists, not bonus. Killed if FPL and match-feed
    assists agree within 3% at 9.0m+.
  - **H4:** in-form starters are under-predicted because multi-season shrinkage drags this season's
    rate down. Killed if control's in-form component sum is < 0.15 in 2025-26.
- **Executor's choice:** internal layout of `form_groups.py`, the diagnostic's JSON shape, and names
  of helper functions.

## Requirements

- **R1.** WHEN `scripts/diagnose_premium_returns.py` runs on the cached DB THE SYSTEM SHALL write JSON
  holding:
  - per price band, for 2025-26 league matches only: FPL season assists (last gameweek row per
    player) vs the sum of match-feed assists, with the ratio and the count of players whose counts
    differ;
  - per price band: FPL goals/xG and assists/xA (season totals, players with ≥ 900 minutes);
  - whether any `fact_raw_*` table carries a per-gameweek bonus column, and its name;
  - in-form and cooling group sizes per season, plus the live season's members.

  Acceptance: the workflow artifact contains all four blocks.
- **R2.** WHEN `walkforward_summary.py` summarises a run THE SYSTEM SHALL add `form_groups` →
  `{season: {in_form, cooling: {n_player_steps, mean_resid, components: {appearance, goals,
  assists, cleansheet, other}, component_abs_sum}}}`.
  - Each component is computed the same way as `price_band` (realized − predicted, as in
    `backtest.py:1040-1056`).
  - `price_band` gains `component_abs_sum` per band.

  Acceptance: a unit test on a synthetic DB checks the numbers by hand.
- **R3.** The in-form group (`form_groups.classify`): a player-step (season S, gameweek G) is IN_FORM
  when, using only data from before G's deadline:
  - (a) he started ≥ 3 of his club's last 4 league matches (the same counting as `breakout.py`);
  - (b) he has ≥ 360 league minutes in S, and his (xG+xA) per 90 in S is ≥ 1.30 × his (xG+xA) per 90
    over earlier loaded seasons. "In S" is read from the latest `fact_player_season_stats` row with
    gw < G. The earlier seasons must hold ≥ 900 minutes;
  - (c) he is not a goalkeeper and not in the breakout group.

  COOLING is the same with ratio ≤ 1/1.30. The minimum group size is 150 player-steps in 2025-26,
  with one declared widening (ratio 1.20). Acceptance: unit tests for each clause and the
  no-earlier-season case (returns None).
- **R4.** WHEN `compare_arms.py control.json arm.json --rule r5 --target <price_band=9.0+ | in_form>`
  runs THE SYSTEM SHALL print PASS only if all of these hold, comparing the arm with control on
  2025-26:
  - (i) paired squad points m ≥ −0.25 and m + SE ≥ 0;
  - (ii) EP MAE ≤ control + 0.002;
  - (iii) the target group's component sum ≤ ⅔ of control's;
  - (iv) breakout residual ≤ +0.182.

  It refuses runs from different `db_cache_key`s, as it does today. Acceptance: tests for a pass,
  for each failing check, and for a group missing from the summary.
- **R5.** WHEN an arm passes R5 with m ≥ 0 THE SYSTEM SHALL be promoted through the live-switch
  pattern. WHEN it passes with m < 0, the owner is asked before merging (user rule).
- **R6.** Every arm's target group is written in the report before its batch is dispatched.

## Key Decisions

- **Promotion rule R5** = R4's (i) and (ii), plus (iii) a component-sum cut of ≥ ⅓ on the arm's
  declared target group, plus (iv) breakout residual ≤ +0.182. (user, Q2 + Q3)
- **Judging season:** 2025-26 decides and 2024-25 is reported (verified: issue 4 convention).
- **In-form definition:** underlying (xG+xA) rather than output (goals+assists).
  [assumed: underlying form is less noisy and matches Mbeumo's case (xG 0.61 vs model 0.43) - if
  wrong: B2's in-form residual stays near 0 and H4 is killed; re-define once on G+A, declared before
  re-running]
- **Thresholds:** 1.30 for in-form, 150 player-steps minimum, ≥ 0.15 component sum to pursue the
  in-form half. [assumed: sized so 2025-26 gives a group of a few hundred player-steps - if wrong:
  the one declared widening, then stop]
- **One target per arm:** finishing and assist arms target 9.0m+, recency and k_minutes arms target
  in_form. [assumed: each lever's mechanism points at one group - if wrong: report the other group's
  numbers, never re-target after results]
- **Batch budget:** B2 and at most one B3. [assumed: two more batches is proportionate to a
  +0.66 / ~0.4-point error - if wrong: the owner can extend]
- **`--k-minutes 450`** may be promoted if it passes R5, even though it reverses the 2026-09-09
  confirmation (verified: overnight report recommended rejecting that confirmation).

## Data & State Changes

- **No schema changes.** New param versions are created only by `run_walkforward.py` flags
  (existing behaviour, `EXPERIMENT_EFFECTIVE_DATE`).
- **Promotion** changes the live default version through the same seeding as A4:
  - `materialize_confirmed_seeds`, plus `RECALIBRATABLE_VERSION_ARGS` when the family isn't
    already in it;
  - `rate_shrinkage_params` and `rate_prior_params` are already registered (verified:
    `backtest.py:2593`, `:2605`).
- **Rollback:** revert the promotion commit; the old versions stay in the param tables.

## Interfaces, Integrations & Credentials

- **GitHub Actions only.**
  - Dispatch `branch_walkforward.yml` with `args` (one run per arm) and read the
    `walkforward-summary` artifact.
  - A new `diagnose_premium_returns.yml` is copied from `diagnose_breakouts.yml`'s cache-restore
    steps.
- **Credentials:** none beyond the existing repo workflows.
- **Fixed contract:** `walkforward_summary.json` keys already read by `compare_arms.py` must not
  change. New keys are only added.

## Edge Cases & Failure Handling

- **Control's in-form group is under 150 even after widening** → the in-form half stops; record it.
- **Runs in a batch on different cached DBs** (`db_cache_key` differs) → `compare_arms.py` refuses;
  re-dispatch the whole batch.
- **A run hits `--max-minutes` before 38 steps** (`progress.complete` false) → not judged;
  re-dispatch once.
- **Two arms pass** → promote the one with larger m. Its partner runs combined with it in the next
  batch before any second promotion.
- **No raw bonus column** → H3 is judged from assists alone (FPL vs match-feed); the bonus split
  stays a residual.
- **The scheduled pipeline is late after promotion** (known cron delay) → dispatch it manually.

## Risks, Landmines & Adaptations

- **Offsetting component errors** (found in B1) → (iii) is judged on the component sum (Q3), and
  Phase 1 separates measurement error from model error before any new lever is built.
- **Measured "assists" come from the match feed, not FPL** → Phase 1's R1 check. If they differ, the
  component split is re-read with that in mind before choosing a B3 lever.
- **Points noise** (SE ≈ 0.8 pts/GW over 37 GWs) → the points gate is a floor, not proof. Calibration
  and component gates carry the decision.
- **Selection on form** (regression to the mean) → the group uses underlying stats before the
  deadline, and the cooling group is reported so an arm that over-predicts faders is visible.
- **Baselines move between DB caches** → each batch has its own control. Compare only within a batch.

## Assumptions Ledger

| ID | Assumption | Basis | Blast radius if wrong | Check |
|----|------------|-------|-----------------------|-------|
| A1 | Match-feed assists ≈ FPL assists | code comment `expected_points.py:489` | component split mis-attributes assists to "other"; wrong B3 lever | Phase 1 R1 |
| A2 | A raw per-GW bonus column may exist in `fact_raw_*` | the CSV source carries many FPL fields | bonus can't be split from DefCon/saves | Phase 1 R1 |
| A3 | In-form = (xG+xA)/90 ≥ 1.30 × earlier seasons | Mbeumo's live gap; less noise than G+A | empty or noisy group | Phase 2 tests, Phase 3 sizes |
| A4 | 150 player-steps is enough | breakout used 200 for a broader rule | judge on too few | widening once, else stop |
| A5 | 2025-26 shows the in-form gap seen live in 2026-27 | same model and data shape | H4 has nothing to fix in backtest | B2 control's in-form block |
| A6 | Each lever has one natural target group | lever mechanisms | an arm helps the "other" group | report both groups for every arm |

## Open Items (none blocking)

- A narrow P(60+) arm that excludes only friendlies and the Community Shield. Proceed without it
  unless a later control shows a 9.0m+ appearance residual > +0.05.

## Verification

- `PYTHONPATH=src python -m pytest tests/test_form_groups.py tests/test_compare_arms.py tests/test_walkforward_summary.py -q` passes.
- `tests/test_live_switch_wiring.py` passes after any promotion.
- `PYTHONPATH=src python scripts/compare_arms.py control.json arm.json --rule r5 --target in_form` prints the four checks.
- **The owner's check:** the report's B2/B3 tables. After a promotion, the next pipeline's
  `projections_latest.json` shows the named players' new EP (Mbeumo, the 9.0m+ starters).

## Build Phases

- [ ] Phase 1: Data diagnostic on the cached DB
      Done when: the `diagnose_premium_returns.yml` artifact holds R1's four blocks, and the report
      records the H1-H3 verdicts from it.
      Steps:
      - Write `scripts/diagnose_premium_returns.py`; its test runs on a synthetic DB first.
      - Add the workflow (copy `diagnose_breakouts.yml`'s restore steps). Merge, then dispatch.
      - Record the assist ratio, conversion ratios, bonus-column finding and group sizes in
        `docs/reports/2026-10_premium_and_form_returns.md`.
      Covers: R1; checks: A1, A2
- [ ] Phase 2: In-form groups, summary cut, R5 judge
      Done when: the new tests pass, the full suite stays green, and the PR is merged.
      Steps:
      - Write the failing tests first: `form_groups.classify` clauses, summary blocks, and R5's
        pass and each fail.
      - Add `src/fpl_quant/form_groups.py` (constants at the top, like `breakout.py`).
      - Extend `scripts/walkforward_summary.py` (`form_groups`, `component_abs_sum`) and
        `scripts/compare_arms.py` (`--rule r5 --target`).
      - The report records B1 (table above) and declares B2's arms and targets before dispatch.
      Covers: R2, R3, R4, R6; checks: A3, A4
      Critique: important phase - after building, hand the diff plus this phase's Done-when and
      Covers lines to a blind critic (fresh sub-agent if available, else a fresh-eyes reread).
      The critic defaults to FAIL and returns PASS/FAIL, strengths, weaknesses and fixes. Apply
      blocker fixes and repeat, max 3 rounds. Done only on PASS.
- [ ] Phase 3: Batch B2 - existing levers
      Done when: all 5 runs complete on one `db_cache_key`, each is judged with `compare_arms.py
      --rule r5`, and the report has the table.
      Steps:
      - Dispatch `--seasons 2025-2026` with: nothing (control); `--finishing-prior-xg 10` (target
        9.0m+); `--assist-prior-xa 30` (9.0m+); `--rate-current-season-weight 2
        --rate-season-decay 0.5` (in_form); `--k-minutes 450` (in_form).
      - Map runs to arms by the `ARGS:` log line, not by order.
      - Record both groups for every arm (A6), and the cooling group.
      Covers: R4, R6; checks: A5
- [ ] Phase 4: One diagnosis-led lever (only if Phase 1 + B2 name one cause)
      Done when: the lever ships as an opt-in flag with tests, its batch B3 (control + arm) is
      judged, and the result is recorded. Or: the report states why no single cause was named, and
      this phase is skipped.
      Steps:
      - Pick from the evidence, and declare the choice in the report before dispatch:
        - H1 holds → a goals-anchor fit on FPL goals rather than xG;
        - H2 or H3 holds → an assist-conversion factor on FPL assists;
        - B2's recency/k arm nearly passes → that arm plus `--bps-tau 7`.
      - Write the failing test first, then the flag in `run_walkforward.py`.
      Covers: R4, R6
- [ ] Phase 5: Promote or stop
      Done when: either (a) the winner is live (seeded, wiring test green, PR merged, pipeline run,
      live check recorded), or (b) the report's conclusion lists every arm and why it failed.
      Steps:
      - m ≥ 0: promote through the A4 pattern (`materialize_confirmed_seeds`, the registry,
        `tests/test_live_switch_wiring.py`). m < 0: ask the owner first.
      - Dispatch `scheduled_pipeline.yml`. Record the before/after EP for Mbeumo and the 9.0m+
        starters.
      - Tick this plan's phases.
      Covers: R5
      Critique: important phase - after building, hand the diff plus this phase's Done-when and
      Covers lines to a blind critic (fresh sub-agent if available, else a fresh-eyes reread).
      The critic defaults to FAIL and returns PASS/FAIL, strengths, weaknesses and fixes. Apply
      blocker fixes and repeat, max 3 rounds. Done only on PASS.
