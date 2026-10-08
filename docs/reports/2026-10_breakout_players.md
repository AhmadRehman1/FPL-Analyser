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
