# Plan: Fix the live decision layer (template anchor, planner, chips)

One-line goal: the points the model loses between its projections and the live team's score
(the template anchor, carrying a squad from week to week, chip timing) are measured on fair
harnesses, and each fix that passes goes live, so the live team closes on the top-100k pace.

Follows the owner review that `docs/plans/2026-10_beat_the_template.md` called for. Phase 9 of
that plan left G2 unmet (+9.92 vs M̄ 13.17) with its EP-lever queue exhausted (verified:
`docs/reports/2026-10_beat_the_template.md`, Phase 9). Its goals G2–G4, rule R6, the gated
template and Phase 11's live tracking all carry over unchanged.

## Classification
Track: Performance. The end state is a number: live net points against the top-100k cutoff.
Model projections stay frozen here; this plan changes decisions only.
Parked secondary asks: more EP-residual levers (Out of Scope); a structural projection track
(Phase 5's out-of-sample check decides whether to open one).

## Interview Ledger
- Q1 next round of work → the live decision layer first, with a smaller projection track for
  structural changes (user, accepted recommendation).
- Q2 template anchor → test κ = 0 and κ = 0.06 against the live κ = 0.12 and promote the
  winner by R6 (user, accepted recommendation).
- Standing instruction: "always answer the recommended choice" (user). Every later fork is
  decided by its recommended option and recorded here, with no further questions.

Questions used: 2 of 14.

## Goal & Success Criteria
- **S1 (anchor).** Batch K is judged by R6 against the live κ = 0.12. A passing weight is live
  in `scripts/run_ingestion.py`'s from-scratch solve, or the report records why 0.12 stays.
- **S2 (planner and chips).** Batch P1 (and P2 if declared) is judged by the Phase 10 rule:
  arm − control m − SE ≥ 0 on season-sim net pts/GW, 2025-26, one GW2–38 chunk, one runner.
  A passing arm is live in the model team's planner versions.
- **S3 (out of sample).** The walk-forward runs 2026-27's finished gameweeks, so the
  backtest → 2026-27 step (−9.60 ± 5.38 pts/GW vs FPL's average, unexplained; Phase 3) is
  measured on the walk-forward's own harness.
- **S4 (live).** At GW13 and GW19 the report records the live team against FPL's average and
  the top-100k cutoff, and each promoted change against its own off-switch (Phase 11 of the
  earlier plan).

## Current State
- **Live team:** 397 pts after GW7, −19 against FPL's average. Bench Boost, Free Hit and Triple
  Captain are all used in the first half. The GW7 Bench Boost scored 35 against a field of 59.
  The first-half Wildcard is left, with GW19 as its last week (verified:
  `data/dashboard/app_model_team.json`, `season_rules.py` per the open-issues status table).
- **Where the anchor reaches:** only `scripts/run_ingestion.py`'s from-scratch solve
  (`ownership_params_version=1, risk_posture_params_version=2`, κ = 0.12) (verified: that file,
  line 309). That solve is the model team's GW1 seed (`model_team.seed_squad`) and the weekly
  report's optimal squad (`scripts/run_report.py`). The planner's Wildcard and Free Hit rebuilds
  pass no anchor (verified: `transfer_planner.py` `squad_optimizer.run(` calls near lines 1343
  and 1482). Neither does the season sim.
  - So the anchor's live cost now sits in the GW1 seed (still being carried) and in the
    recommended squad the report publishes, not in the model team's future chip squads.
  - κ = 0.12 was the owner's explicit rank-safety call (verified: `squad_optimizer.py` comment
    above `seed_template_anchor_params`). The owner chose to re-test it (Q2).
- **Measured anchor cost:** −2.96 ± 1.49 pts/GW in the oracle walk-forward, against an older
  control (before Z-goal) (verified: report Phase 3, run 38269669897).
- **Planner arms already run** (2025-26, chunks GW2–19 and GW20–38, matrix jobs on different
  runners): `--multi-transfers` −0.35 ± 1.44, `--chip-option-value` −0.73 ± 2.01, both
  +0.68 ± 1.36 (verified: `docs/reports/2026-10_open_issues.md`, run 37201797073).
  - Those runs carry two flaws this plan removes:
    - the GW19 fresh-squad restart works as a free Wildcard;
    - runners differ by ±1.81 pts/GW on the same DB.
  - `season_sim_arms.yml` now has `same_runner` and accepts a `2-38` chunk (verified: that
    workflow's inputs).
- **Live planner settings:** `multi_transfer_params_version` and `chip_wait_params_version` are
  None, so the live team makes one transfer a week at most, never takes a hit, and plays chips at
  the first local maximum (verified: `forward_season_sim.py` active-versions block).

## Scope (v1)
Batch K (anchor), batch P1 (planner and chips), an optional P2 of at most 3 arms, the 2026-27
out-of-sample walk-forward, and the GW13 and GW19 reviews.

## Out of Scope & Parked Items
- **More EP-residual levers** (crowd prior, band bias, per-player residual variants): each failed
  R6 and moved the XI toward the template. Reopen only with a structural change to a sub-model,
  the way Z-goal changed the goal model.
- **A structural projection track:** opened only if Phase 5 shows the oracle edge holds in
  2026-27 while the live team lags. If the edge itself collapses in 2026-27, the projection
  model goes first, ahead of the planner.
- **Elite (top-10k) ownership as an input:** parked as before.
- **Re-seeding the live model team:** no. Its record stays one continuous squad.

## Approach
Each lever is one opt-in flag, declared in the report before its runs are dispatched, judged by
a fixed rule, and switched live only on a pass, with an off-switch kept for Phase 11.
- **Batch K (walk-forward, both seasons, `--template-lag 2`, three runs on one ref):**
  - `control`: no anchor, which is κ = 0;
  - `k012`: `--live-eo-anchor`, the live κ = 0.12;
  - `k006`: `--eo-anchor-kappa 0.06`.
- **K's rule:** each of κ = 0 and κ = 0.06 is judged against `k012` with
  `compare_arms.py --rule r6 <k012 run> <arm run>`.
  - R6(i): m − SE ≥ 0 on paired squad points.
  - R6(ii): EP MAE within +0.01 in each season. The anchor never changes EP, so the MAE must
    match `k012`'s exactly. A difference means the run is broken and is not judged.
  - R6's season-sim check does not apply. The season sim never passes the anchor, so its
    control and arm would be identical; this is recorded rather than run.
  - If both pass, the higher m wins. If neither passes, κ = 0.12 stays.
- **Batch P1 (season sim, 2025-26, chunk `2-38`, `same_runner`, five runs, the workflow's cap):**
  - `control`;
  - `control_b`: a second control, for the noise check;
  - `multi`: `--multi-transfers`;
  - `chipov`: `--chip-option-value`;
  - `both`: `--multi-transfers --chip-option-value`.
- **P1's rule:**
  - First, control vs control_b must differ by 0.00 on every gameweek; otherwise the batch isn't
    judged.
  - Then each arm against `control`: promote on m − SE ≥ 0 over GW2–38 net points.
  - If several pass, the highest m wins. If `both` wins, both flags go live together.
  - Planner arms used: 3 of a cap of 6.
- **P2** is declared only after P1, from P1's per-week traces, with at most 3 arms (for example a
  Wildcard threshold or a Bench Boost floor, if the traces show those decisions losing points).
  Its rule is the same as P1's.

## Requirements
- **R1.** WHEN `run_walkforward.py --eo-anchor-kappa K` runs, THE SYSTEM SHALL solve with
  ownership_params v1 and the risk_posture_params version holding posture 'protect' and
  eo_weight_kappa K, minting it if it doesn't exist. K = 0.12 resolves to v2, K ≤ 0 is refused,
  and the flag can't be combined with `--live-eo-anchor`. Check: `tests/test_run_walkforward.py`.
- **R2.** WHEN a batch is dispatched, THE REPORT SHALL already hold its declaration: runs, args,
  rule, ref. Check: the declaring commit predates the run ids.
- **R3.** WHEN a lever passes, THE SYSTEM SHALL switch it live in one commit that names the run
  ids, and the report SHALL name its off-switch. Check: the diff and the report.
- **R4.** WHEN the walk-forward runs `--seasons 2026-2027`, THE SYSTEM SHALL walk ≥ 1 finished
  2026-27 gameweek. Check: a test, plus the run's `progress` block.

## Key Decisions
- Judge κ against the live 0.12, not against no-anchor: the question is whether to move off what
  is live (user, Q2).
- Season sim chunk `2-38` on one runner, so there is no free-Wildcard restart and no runner noise
  [assumed: one GW2–38 run takes about 40 min, so five fit in the 345-min step; basis: GW2–19
  chunks took 17–20 min (run 37397424563) - if wrong: split into two same_runner batches, each
  with its own control].
- The live κ = 0 switch passes no ownership or posture version at all, which is the code's
  "no anchor" path (verified: `squad_optimizer.run` requires both or neither).
- A promoted planner flag goes live from the next deadline. Chips already played stay played.

## Data & State Changes
- New `risk_posture_params` versions are minted on demand in the experiment DB only (κ = 0.06).
- A live switch of the anchor changes only the version that `run_ingestion.py` passes. A live
  switch of a planner flag sets that version in the model team's `active_versions`.
- Rollback: revert that one commit.

## Interfaces, Integrations & Credentials
GitHub Actions workflow_dispatch of `branch_walkforward.yml` and `season_sim_arms.yml`. No new
secrets. Output formats are unchanged.

## Edge Cases & Failure Handling
- **A run stops before 75/75 steps:** it is not judged. Re-dispatch once on the same ref.
- **R6(ii)'s MAE differs within batch K:** stop, and find what else the flag changed.
- **control ≠ control_b in P1:** the batch isn't judged; open a determinism item.

## Risks, Landmines & Adaptations
- **The anchor was the owner's deliberate choice** → it is re-tested only because he asked (Q2),
  and judged against what is live.
- **The anchor no longer touches the model team's future squads** (Current State) → batch K is
  ranked below P1 for the live score. It still decides the published optimal squad and next
  season's seed.
- **Old planner results were inconclusive under noise** → re-run with `same_runner`, one chunk
  and a control_b, rather than reading the old numbers.
- **The Wildcard window closes at GW19** → P1 is dispatched in the same session as K, so a passing
  chip rule can go live before the window ends.

## Assumptions Ledger
| ID | Assumption | Basis | Blast radius if wrong | Check |
|----|------------|-------|-----------------------|-------|
| A1 | Five GW2–38 season-sim runs fit one job | GW2–19 chunks took 17–20 min | P1 times out | Phase 3: the job's timing |
| A2 | The cached DB holds settled 2026-27 GWs up to the last finished one | the nightly pipeline ingests live data | Phase 5 falls back to the leaderboard comparison | Phase 5, step 1 |
| A3 | Batch K's three runs land on the same cached DB | dispatched minutes apart | runner and DB noise in the pairs | Phase 2: the logged cache key |

## Open Items (none blocking)
- A projection track, decided by Phase 5's result. Proceed with no projection changes until then.

## Verification
- `PYTHONPATH=src python -m pytest -q tests/test_run_walkforward.py`
- `python scripts/compare_arms.py --rule r6 <k012 summary> <arm summary>` for K.
- `scripts/aggregate_season_sim_arms.py` paired output for P1.
- The owner reads the "Live decision layer" section of `docs/reports/2026-10_beat_the_template.md`.

## Build Phases
- [x] Phase 1: Build the anchor-weight flag and declare batch K
      Done when: R1's tests pass and batch K is declared in the report before dispatch.
      Steps: add `--eo-anchor-kappa` and `squad_optimizer.template_anchor_version()`, with tests;
      write the declaration; commit; dispatch the three runs.
      Covers: R1, R2
- [x] Phase 2: Judge batch K and switch the winner live
      Done when: the report records both verdicts, and either the live switch commit or "κ = 0.12
      stays".
      Steps: download the summaries; check 75/75 steps and matching MAE; run compare_arms; on a
      pass, change `run_ingestion.py`'s versions.
      Covers: R3; checks: A3
- [x] Phase 3: Declare and dispatch batch P1
      Done when: P1's declaration is committed and the run is dispatched with `chunks: 2-38`,
      `same_runner: true`.
      Covers: R2; checks: A1
- [x] Phase 4: Judge P1 and promote
      Done when: the report records control vs control_b, each arm's m ± SE, and the live switch
      or rejection.
      Steps: download `season-sim-summary`; pair; decide; on a pass, set the version in the model
      team's active versions with a test that the live walk passes it.
      Covers: R3
      Critique: important phase (it changes the live team). After building, hand the live-switch
      diff, the P1 table and this phase's Done-when to a blind critic (a fresh sub-agent if
      available, else a fresh-eyes reread). The critic defaults to FAIL and returns PASS/FAIL,
      strengths, weaknesses and fixes. Apply the blocker fixes and repeat, at most 3 rounds.
      Done only on PASS.
- [x] Phase 5: Run the walk-forward out of sample on 2026-27
      Done when: R4's test passes and a control walk-forward over 2026-27's finished GWs reports
      model vs gated template and vs FPL's average.
      Steps: extend `ALL_SEASON_GAMEWEEKS` to the last finished 2026-27 GW (A2); test; dispatch
      `--seasons 2026-2027 --template-lag 2`; record the step against the 2025-26 figure.
      Covers: R4; checks: A2
- [ ] Phase 6: Declare P2 from P1's traces (optional, at most 3 arms)
      Done when: P2 is declared and judged by P1's rule, or the report says why there is no P2.
- [ ] Phase 7: Review at GW13 and GW19
      Done when: the report holds the live table and each promoted change's off-switch
      comparison, with a keep or roll-back decision (roll back if m + SE < 0).

## Status (2026-10-21)
- **Batch K:** no anchor won the backtest (+3.66 ± 1.60) but lost 2026-27 GW2–7 (−14.5 ± 8.5).
  The live switch was rolled back before merge, and κ = 0.12 stays. It is re-judged at GW13.
- **P1:** no planner arm beat control (multi −0.14, chipov −0.57, both −1.86). Nothing is
  promoted.
- **Phase 5:** the oracle edge over the gated template collapses in 2026-27: −12.4 ± 12.0, with
  EP MAE 1.36 against 1.05 in the backtest.
- **So Phase 6 (P2) is replaced by a projection track.** Its first step is to find why 2026-27's
  EP error and its picks away from the template are so much worse than the backtest seasons:
  - compare EP MAE and bias by GW for 2025-26 GW2–7 against 2026-27 GW2–7;
  - split the gap by component and by price band;
  - check the 2026-27 data feed (the match-feed lag of PR #261).

  A new plan is written for that track before any lever is built.
