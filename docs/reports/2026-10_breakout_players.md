# Breakout players: why the live model under-ranks them, and what was tested

Oct 8, 2026 · draft, filled in as the plan's phases finish (docs/plans/2026-10_breakout_players.md)

## Diagnosis (phase 1)

`diagnose_breakouts.yml` run 37822409858 read the live GW6 EP (ep_model_version 2, data cutoff
2026-10-08 12:53 UTC). Every number matched `projections_latest.json`.

| player | p_start | P(60+) | starts this season | model goals/assists per 90 | xG/xA per 90 | main gap (pts/GW) | in breakout group |
|---|---|---|---|---|---|---|---|
| Groß (MID, £5.9m) | 0.60 | 0.56 | 5 of 5 | 0.16 / 0.21 | 0.35 / 0.17 | minutes +1.67, rates +0.53 | no (real starting record last season) |
| Kinsky (GK, £4.5m) | 0.31 | 0.23 | 5 of 5 | — | — | minutes +1.78 | yes |
| Kostoulas (FWD, £5.6m) | 0.62 | 0.16 | 4 of 5 | 0.32 / 0.06 | 0.34 / 0.02 | minutes +0.45 | yes |
| De Cuyper (DEF, £5.0m) | 0.77 | 0.41 | 5 of 5 | 0.11 / 0.16 | 0.33 / 0.14 | rates +0.72, minutes +0.68 | no (real starting record last season) |
| Bruno Fernandes (reference) | 0.95 | 0.90 | 5 of 5 | 0.46 / 0.42 | 0.51 / 0.26 | small | no |
| Mbeumo (reference) | 0.93 | 0.92 | 5 of 5 | 0.43 / 0.24 | 0.61 / 0.16 | rates +0.61 | no |

"Main gap" is the rough points per gameweek each cause would add if the model matched this season.

- **Minutes is the main cause.** Players who start every match this season get a start
  probability of 0.31-0.77, against 0.93-0.95 for established starters. The minutes model
  weighs this season's five matches against two earlier seasons in a different role.
- **The 60-minute rate lags too.** Kostoulas starts, but P(60+) is 0.16: the model's
  P(60+ | started) comes from earlier seasons only. The current-season role blend
  (`current_season_role_params`) moves the start probability and leaves this rate alone.
- **Scoring rates are a smaller, separate gap.** The rate gap is about the same for the
  reference starter Mbeumo (+0.61), so it is not specific to breakout players.
- **Club strength is not a cause.** Brighton's attack is +0.20 above the league median.
- **The backtest gate measures a proxy for two of the four.** Groß and De Cuyper are outside
  the breakout group, because they had a real starting record last season. The group
  definition (rule R2) was fixed before any data and stays as it is. The report says how far
  a passing arm moves these two.

## Arms (declared before any arm result was read)

- **A1** `--role-matches-threshold 4`: the current-season role blend on the start probability.
- **A2** `--rate-current-season-weight 2 --rate-season-decay 0.5`: this season's scoring
  rates count double.
- **A3** both together.
- **A4 (diagnosis-led, phase 4):** A1 plus the same current-season blend on
  P(60+ | started). This season's own rate is blended in at weight min(1, this season's starts
  / 4), on top of the multi-season rate. It is a new opt-in flag (`--role-minutes-blend`, used
  with `--role-matches-threshold 4`). It runs in its own batch with its own control.
- **A5:** none. The scoring-rate gap is not breakout-specific (above), so it is left to A2/A3.

Declared at 18:23 UTC on 2026-10-08, after reading the phase 1 diagnosis and before any of the
18:11 UTC batch (control, A1-A3) was read.

## Arms A1-A3 (phase 3)

The 18:11 UTC batch on master (merge commit 9bb8ba4) ran on one cached DB (`duckdb-37798981174`),
and every run completed. Run ids: control 37822413654, A1 37822417305, A2 37822421212,
A3 37822425829.

**Stop conditions: passed.** Control's 2025-26 breakout group has 3,493 player-steps, and the
model under-predicts it by +0.272 points per player-week (MAE 1.84). Players at promoted clubs,
reported apart, are under-predicted by +0.186 (1,235 player-steps).

**Sanity read.** Control's sample mixes real breakouts (Diego Gómez, McAtee, Barry) with a
few players who were regulars the season before (Kilman). Those most likely reach the group
through gaps in the 2024-25 match data. That dilutes the measure but doesn't bias it, because
every arm is scored on the same group.

| arm | breakout mean resid | 2025-26 EP MAE | 2025-26 squad pts vs control | 2024-25 squad pts | R4 |
|---|---|---|---|---|---|
| control | +0.272 | 1.0825 | — | 63.00 | — |
| A1 role blend (`--role-matches-threshold 4`) | **−0.004** ✔ | 1.1044 ✘ (+0.022) | −1.35 ± 1.41 ✘ (16 better, 18 worse) | 63.59 | FAIL |
| A2 recency (`--rate-current-season-weight 2 --rate-season-decay 0.5`) | +0.275 ✘ | 1.0816 ✔ | −0.24 ± 1.23 ✔ | 63.54 | FAIL |
| A3 both | **−0.002** ✔ | 1.1034 ✘ | −1.87 ± 1.56 ✘ | 64.84 | FAIL |

- **The role blend fixes the breakout group and overshoots everyone else.** The overall mean
  residual goes from −0.103 to −0.177 in 2025-26, so the model now over-predicts. Any player
  who started 3-4 of the last matches is treated as nailed on, and many of them are then
  rotated or injured. Squad points fall, because the optimizer buys those players.
- **Recency on scoring rates doesn't touch the group.** It changes nothing for breakout players
  and stays level elsewhere, which fits the diagnosis: their gap is minutes, not rates.
- **2024-25 is not comparable.** That season has no earlier one, so its breakout block is null.
  A1's and A3's 2024-25 squad points rise, but that season doesn't decide.

## Arm A4 (phase 4)

The 18:29 UTC batch on the A4 commit (f377533) ran on the same cached DB (`duckdb-37798981174`),
and both runs completed. Run ids: control 37824630684, A4 37824634995.

**This batch's control matches the 18:11 control.** Breakout residual +0.272 in both, 2025-26
EP MAE 1.0827 against 1.0825, 2024-25 squad points 62.86 against 63.00. The code between the
two commits differs only behind the opt-in flag, so the gap is the walk-forward's run-to-run
variation. That is why each arm is judged against the control in its own batch.

| arm | breakout mean resid | 2025-26 EP MAE | 2025-26 squad pts vs control | 2024-25 squad pts | R4 |
|---|---|---|---|---|---|
| control (A4 batch) | +0.272 | 1.0827 | — | 62.86 | — |
| A4 role blend + P(60+) blend (`--role-matches-threshold 4 --role-minutes-blend`) | **+0.108** ✔ (limit +0.182) | **1.0665** ✔ (−0.016) | +0.00 ± 1.67 ✔ (18 better, 17 worse, 2 level) | 63.54 | **PASS** |

Squad points differ in 35 of 37 gameweeks; the season totals happen to tie (60.16 points a
gameweek each).

- **The P(60+) blend removes the role blend's overshoot.** A1 alone moved the 2025-26 overall
  mean residual from −0.103 to −0.177. A4 leaves it at −0.099. A starter who keeps being taken
  off early this season now loses P(60+) as fast as a new starter gains P(start).
- **It doesn't fix the whole gap.** The breakout group is still under-predicted by +0.108
  points a player-week, and its MAE moves from 1.84 to 1.86: the bias is smaller, the spread
  is not.
- **Promoted clubs' breakout players gain too** (+0.186 → +0.103, reported apart, not a gate).
  The widened group goes from +0.219 to +0.080.

Not gates, recorded so the next change starts from them:

| 2025-26 unless noted | control | A4 |
|---|---|---|
| minutes Brier | 0.2974 | 0.2940 |
| minutes log score | −0.5308 | −0.5347 |
| 9.0m+ price band mean resid | +0.069 | +0.175 |
| 7.0-9.0m price band mean resid | +0.252 | +0.272 |
| captain points a gameweek (both seasons) | 6.28 | 6.42 |
| 2024-25 EP MAE | 1.0463 | 1.0544 |

- **The minutes log score is a little worse while the Brier improves.** After 4 starts a
  player's P(60+ | started) is this season's raw rate, which can be 0 or 1, so a wrong call is
  penalised hard (the 0.005 floor still applies).
- **Premiums are under-predicted a little more** (+0.07 → +0.17 at 9.0m+). The price anchor
  (#244) had brought this band from +0.54 to +0.07, so this gives back part of that gain.
- **2024-25 calibrates worse** (MAE +0.008). That season has no earlier season loaded, so the
  blend only replaces the shrunk rate with the raw one there. The rule judges 2025-26, which
  has an earlier season, as the live season does.

## Decision (phase 5)

**A4 goes live.** It is the only arm that passed. Its paired m is +0.00, not below zero, so
under the plan it needed no extra sign-off. `current_season_role_params` v1 (threshold 4) and
`current_season_minutes_params` v1 (threshold 4) join `active_recalibratable_versions()`, so
every minutes model run gets them. That covers live ingestion, the walk-forward and the season
simulations with their baselines, and the forward season sim. `run_walkforward.py
--no-current-season-blend` runs the model as it was before.

The minutes log-score refit (`_minutes_log_score_for_step`) keeps scoring without the blends,
like the start prior and the floor, so it stays comparable with earlier recalibrations.

## Parked follow-ups

- **The remaining +0.11 breakout gap and the 1.86 MAE.** The next arm should shrink this
  season's P(60+ | started) toward the multi-season rate by pseudo-counts, instead of switching
  to the raw rate at 4 starts. That should also win back the log score. It needs a new declared
  arm, judged by R4 against the live model.
- **The 9.0m+ band.** Find out why premiums' P(60+) falls under the blend before tuning
  anything there.
- **Scoring rates.** The rate gap is the same for established starters (Mbeumo +0.61), so it is
  a separate problem from breakouts. A2's recency weights didn't move it.
- **Groß and De Cuyper are outside the group** (real starting records last season), so R4
  never measured them. Their live move is in the live check below.
- **January movers in 2024-25.** A move between two Premier League clubs that season loses the
  first club's starts (`breakout.py`'s known simplification).
- **No standard error on the breakout cut.** R4(iii) compares two group means. Both arms score
  the same 3,493 player-steps, so a paired per-step comparison could put an SE on the cut.

## Live check (phase 5)

This compares the same gameweeks before and after the first pipeline run with the blends. Each
named player's change is taken net of the two reference starters' average change, so new
results and news over the same hours don't count as the fix. This is a check, not a target.

Before: `projections_latest.json` generated 2026-10-08 13:01 UTC (commit 061c81f), the
price-anchored model without the blends.

| player | GW6 EP (rank) before | GW6-13 EP before |
|---|---|---|
| Groß | 2.51 (#127) | 19.72 |
| Kinsky | 0.79 (#394) | 7.21 |
| Kostoulas | 1.60 (#265) | 12.58 |
| De Cuyper | 2.35 (#151) | 17.80 |
| Bruno Fernandes (reference) | 6.16 (#1) | 44.97 |
| Mbeumo (reference) | 5.34 (#2) | 37.80 |

After: filled in once the pipeline has run on the merged change.
