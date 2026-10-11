# Premium and in-form attacking returns: where the error is, and what was tested

Oct 11, 2026 · in progress: Phase 1 done; B2 declared (Phase 2 of docs/plans/2026-10_premium_and_form_returns.md)

## Rule R5 (agreed 2026-10-11, before any B2 result)

An arm is compared with its batch's control on 2025-26 (`compare_arms.py --rule r5 --target ...`).
It passes only if all four hold:
- (i) paired squad points m ≥ −0.25 and m + SE ≥ 0;
- (ii) EP MAE ≤ control + 0.002;
- (iii) its declared target group's component sum (|appearance| + |goals| + |assists| +
  |clean sheet| + |other|) is at most ⅔ of control's;
- (iv) the breakout group's residual stays ≤ +0.182 (⅔ of the before-fix model's +0.272).

If the winner's m is below 0, the owner signs off before it merges.

## Batch B1: the two bonus arms (run 2026-10-11)

The bonus arms from `docs/reports/2026-10_open_issues.md` (issue 6) were never judged.
- **The batch:** all three runs used cached DB `duckdb-38085065975` and commit e452e49, with
  `--seasons 2025-2026`.
- **Run ids:** control 38099909460, `--bps-calibration-k 450` 38099910627, and `--bps-tau 7`
  38099912176. Each was mapped to its arm by the `ARGS:` log line.

| 2025-26 | control | `--bps-calibration-k 450` | `--bps-tau 7` |
|---|---|---|---|
| squad pts vs control (m ± SE a GW) | — | −0.14 ± 0.87 (12 better, 12 worse) | **−1.03 ± 0.76** (8 / 8) |
| EP MAE | 1.0663 | 1.0658 | 1.0655 |
| 9.0m+ appearance / goals / assists / clean sheet / other | −0.004 / −0.267 / +0.227 / −0.007 / +0.158 | other +0.166 | other **+0.030** |
| 9.0m+ component sum (R5 limit 0.441) | 0.662 | 0.670 ✘ | 0.534 ✘ |
| breakout residual (limit +0.182) | +0.108 | +0.111 | +0.107 |
| captain pts/GW (share of forwards) | 5.89 (54%) | 5.84 (43%) | 5.57 (68%) |

- **Both fail.** Neither cuts the premium component sum by a third. `--bps-tau 7` also fails (i).
- **The premium error is three errors that partly cancel.** Goals are over-predicted (−0.27), while
  assists (+0.23) and bonus/other (+0.16) are under-predicted. The band total of +0.107 hides
  this.
- **Fixing the bonus alone made picks worse.** `--bps-tau 7` gives the top BPS in a match more of
  the bonus, which removes the premium bonus error. But it lifted premium forwards, whose goals are
  already over-predicted, and captain points fell from 5.89 to 5.57 a gameweek.
- **That is why R5 judges the component sum.** A target on the band total could be met by moving
  one component while the ranking got worse.

## Phase 1: what is measured, and how each answer is used (declared before the results)

`diagnose_premium_returns.yml` runs `scripts/diagnose_premium_returns.py` on the cached DB.

**H2: does FPL award premiums more assists than xA implies?**
- Measured as FPL assists / xA at 9.0m+ in 2025-26, over players with ≥ 900 league minutes.
- H2 is killed if that ratio is ≤ 1.05.

**H3: is part of the 9.0m+ "other" residual FPL-only assists?**
- Background: the backtest counts realized assists from the match feed. Anything FPL awards
  beyond that lands in "other".
- H3 is killed if FPL and the feed agree within 3% at 9.0m+ in 2025-26.
- If H3 survives *and* the alignment check passes, Phase 2 counts realized assists from FPL's
  own per-gameweek count (plan RQ2b), and B2's control uses it.
- **The alignment check:**
  - FPL's per-gameweek assists (the differences of the cumulative count) must sum to each
    player's total;
  - they must match the feed on ≥ 95% of the 9.0m+ player-gameweeks he played in.
- If the check fails, the match feed stays the measure for this round.

**H1: does the price anchor overshoot premium goals?**
- Phase 1 reports goals / xG by band. The verdict waits for B2's `--no-rate-price-anchor` arm.

**The in-form group (`src/fpl_quant/form_groups.py`)**
- **Who is in it:** players who started ≥ 3 of their club's last 4 league matches, have ≥ 360
  league minutes this season, and whose (xG + xA) per 90 this season is ≥ 1.30 × their own
  earlier seasons' (≥ 900 earlier minutes).
- **Who is excluded:** goalkeepers and breakout players.
- **Cooling** is the same test with ≤ 1/1.30.
- **Which version is judged:** the declared group, if it has ≥ 150 2025-26 player-steps.
  Otherwise the widened one (ratio 1.20), if that reaches 150. Otherwise the in-form half
  stops.

## Phase 1 results

`diagnose_premium_returns.yml`, on the cached DB, ran twice:
- **run 38128184580** (master), which had defenders in the in-form group;
- **run 38128511323** (this branch), with midfielders and forwards only. See the group change below.

**H3 is killed, so RQ2b is not applied.** In 2025-26 the match feed's assists equal FPL's exactly.
- Every band matches, with no player differing; at 9.0m+ both count 57.
- FPL's per-gameweek counts match the feed on 100% of the 9.0m+ player-gameweeks played (182 of
  182). Shifting by a week either way drops the match to 66-68%.
- So the 9.0m+ "other" residual is not FPL-only assists, and the realized split stays on the
  match feed.
- **The live 2026-27 season does differ.** FPL counts 1.5x the feed's assists, and per-gameweek
  agreement is 98%. GW6 is part-played at the data cutoff (6 of 10 matches). The judging season
  isn't affected, and the live projections read FPL's own season totals.

**H2 holds.** In 2025-26, among players with ≥ 900 minutes, FPL assists per xA were:

| price band | <£5.0m | £5.0-7.0m | £7.0-9.0m | £9.0m+ |
|---|---|---|---|---|
| 2025-26 assists / xA | 1.31 | 1.36 | 1.41 | **1.63** (56 / 34.4) |
| 2025-26 goals / xG | 0.83 | 0.96 | 1.10 | **1.00** (85 / 85.0) |
| 2024-25 assists / xA (match feed) | 1.05 | 1.15 | 1.21 | 1.38 |
| 2024-25 goals / xG (match feed) | 0.83 | 1.00 | 1.11 | 1.05 |

- FPL awards more assists than xA in every band, and most for premiums. The model's assist rate
  is built on xA, which fits the assists residual rising with price (+0.006 / +0.038 / +0.088 /
  +0.227 in B1's control).

**H1, first half: premiums convert their xG.** Their goals per xG is 1.00 (1.05 in 2024-25), so
the −0.27 goals residual at 9.0m+ is the model predicting more than xG, not premiums finishing
badly. B2's `--no-rate-price-anchor` arm decides whether the price anchor is why.

**Bonus is in the raw data.**
- 2025-26 has a per-gameweek `bonus` column (`By Gameweek/GW*/playerstats.csv` and
  `player_gameweek_stats.csv`); 2024-25 has it per season only.
- This round keeps "other" as a residual, as planned. A bonus / non-bonus split is possible later.

**The in-form group, fixed before any arm (2026-10-11).**
- **Why it changed:** the first run's live group was mostly defenders whose (xG + xA) per 90 rose
  from about 0.03 to 0.10. That is a ratio of noise on a rate that barely moves their points.
- **The change:** the group is now midfielders and forwards only (`form_groups.py`, clause c).
- **2025-26 sizes:** 434 in-form player-steps (592 widened) and 735 cooling (884 widened). The
  **declared** group is judged.
- **Live, GW7, in form:** Brobbey, Barry, Le Fée, Xhaka, Rogers, Lewis-Potter, Ødegaard, Groß,
  Gibbs-White, Sadiki, Mainoo.
- **Live, GW7, cooling:** among others Palmer, Rice, Wirtz, Semenyo, Havertz and Cunha.

## Batch B2: the existing levers (declared 2026-10-11, before any of its results)

**Batch:** six runs of `branch_walkforward.yml`, all with `--seasons 2025-2026`, on one cached DB.
Each arm is judged with `compare_arms.py control arm --rule r5 --target <its target>`. RQ2b does
not apply, so the control is the live model.

| arm | flags | target |
|---|---|---|
| control | none | — |
| finishing prior | `--finishing-prior-xg 10` | 9.0m+ |
| assist prior | `--assist-prior-xa 30` | 9.0m+ |
| no anchor | `--no-rate-price-anchor` | 9.0m+ |
| recency | `--rate-current-season-weight 2 --rate-season-decay 0.5` | in_form |
| k 450 | `--k-minutes 450` | in_form |

- Every arm also reports the other group and the cooling group.
- **H1** is decided by the no-anchor arm. **H4** is killed if control's in-form component sum is
  below 0.15.
- **If no arm passes,** Phase 4 builds the pooled band conversion (`--band-conversion 20`),
  since H2 holds.
