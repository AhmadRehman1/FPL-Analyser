# Premium and in-form attacking returns: where the error is, and what was tested

Oct 11, 2026 · final: every phase of docs/plans/2026-10_premium_and_form_returns.md is done; no arm passed, the live model is unchanged

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

### Result: no B2 arm passes R5

**The batch.** It ran at 09:57 UTC on 2026-10-11 on master 1fc1bba. All six runs used cached DB
`duckdb-38085065975` (the same as B1), all were complete (38 of 38 steps), and each was mapped to
its arm by the `ARGS:` log line. Run ids:
- control: 38130203928
- finishing prior: 38130205472
- assist prior: 38130206727
- no anchor: 38130208017
- recency: 38130209466
- k 450: 38130210836

| 2025-26 | control | finishing prior | assist prior | no anchor | recency | k 450 |
|---|---|---|---|---|---|---|
| target | — | 9.0m+ | 9.0m+ | 9.0m+ | in_form | in_form |
| (i) squad pts vs control (m ± SE a GW) | — | −0.70 ± 1.21 ✘ | −0.03 ± 0.59 ✔ | **+1.32 ± 1.01** ✔ | −0.30 ± 1.51 ✘ | −0.22 ± 0.83 ✔ |
| (ii) EP MAE (limit 1.0685) | 1.0665 | 1.0679 ✔ | 1.0748 ✘ | 1.0771 ✘ | 1.0656 ✔ | 1.0671 ✔ |
| (iii) target component sum | 9.0m+ 0.661, in_form 0.343 | 0.591 ✘ (limit 0.441) | 0.521 ✘ | 0.564 ✘ | 0.281 ✘ (limit 0.229) | 0.375 ✘ |
| (iv) breakout residual (limit +0.182) | +0.108 | +0.102 ✔ | +0.064 ✔ | +0.086 ✔ | +0.111 ✔ | +0.099 ✔ |
| overall mean residual | −0.099 | −0.104 | −0.122 | −0.114 | −0.097 | −0.101 |
| captain pts/GW | 5.89 | 5.46 | 5.97 | 5.84 | 5.87 | 5.68 |

Where each arm moved the error (residual = realized − predicted; one component per row):

| 2025-26 residual | control | finishing | assist | no anchor | recency | k 450 |
|---|---|---|---|---|---|---|
| 9.0m+ goals | −0.267 | −0.285 | −0.267 | **−0.102** | −0.260 | −0.261 |
| 9.0m+ assists | +0.227 | +0.151 | **+0.094** | +0.265 | +0.244 | +0.228 |
| 9.0m+ other (bonus etc.) | +0.157 | +0.145 | +0.150 | +0.187 | +0.150 | +0.158 |
| in-form goals | +0.073 | +0.076 | +0.073 | +0.070 | **−0.013** | +0.084 |
| in-form assists | +0.133 | +0.124 | **+0.044** | +0.131 | +0.117 | +0.134 |
| price band <5.0 total | −0.164 | −0.163 | −0.173 | −0.188 | −0.166 | −0.167 |
| price band 5.0-7.0 total | −0.013 | −0.019 | −0.054 | −0.028 | −0.007 | −0.018 |

**H1 holds: the price anchor overshoots premium goals.**
- Removing the anchor cuts the 9.0m+ goals over-prediction from −0.27 to −0.10, and squad points
  rise +1.32 ± 1.01 a gameweek.
- But premium assists and bonus get worse, and cheap players are over-predicted more. So the EP
  error rises (+0.011) and the arm fails (ii) and (iii).
- The anchor tilts goals too steeply and assists too gently.

**H4 holds: in-form starters are under-predicted.** Control's in-form residual is +0.195 (component
sum 0.343, above 0.15). Assists are +0.133, goals +0.073 and appearance +0.063.
- Recency removes the in-form goals part (+0.073 → −0.013), but not the assists, and its points
  are just below the floor.
- k 450 changes almost nothing for this group.

**The assist calibration fixes assists everywhere but lifts everyone.**
- `--assist-prior-xa 30` pools FPL assists per xA by position (1.3-1.6 in every band), not per
  player as the plan's text said.
- It cuts the assist error at 9.0m+ (+0.227 → +0.094), in form (+0.133 → +0.044) and in the
  breakout group (+0.108 → +0.064).
- But it also raises cheap players' assists, and the model already over-predicts them (<£5.0m
  −0.164, overall −0.099). EP MAE rises by 0.008.

**What this says about the next lever.**
- The model has to move *by price*: more for 7.0m+ players, less for cheap ones. A uniform lift
  (assist prior) or a uniform cut (finishing prior) loses on MAE.
- MAE also penalizes any mean-correcting lift in a skewed points distribution, which the rule
  should keep in mind.

## Phase 4, batch B3 (declared 2026-10-11, before any of its results)

No B2 arm passed and H1 and H2 hold, so the plan's Phase 4 lever is built: the pooled band
conversion (`--band-conversion 20`, `conversion_params` / `pseudo_xg`).
- **What it does:** each position and price band's goal and assist rates are scaled by that band's
  `(goals + 20) / (xG + 20)` and `(assists + 20) / (xA + 20)` in the seasons before the one
  predicted.
- **Inputs:** FPL totals from 2025-26 onward. 2024-25 uses the match feed and per-90 xG/xA ×
  match minutes, and a test checks the ratio isn't stuck at 1.
- **For 2025-26 steps** the earlier season is 2024-25, whose ratios are goals/xG 0.83 (<£5.0m) to
  1.11 (£7.0-9.0m) and assists/xA 1.05 to 1.38. So it lowers cheap players' goals and lifts
  premium assists most: the tilt B2 points to.
- **Batch:** control and `--band-conversion 20`, `--seasons 2025-2026`, one cached DB.
- **Target:** 9.0m+, judged with `compare_arms.py --rule r5 --target price_band=9.0+`.
- **In-form:** it is reported but does not decide.

### Result: B3 fails; nothing goes live this round

**The batch.** It ran at 10:40 UTC on 2026-10-11 on master cd3c624, on cached DB
`duckdb-38085065975`. Both runs completed. Run ids: control 38132767767, `--band-conversion 20`
38132769976, each mapped by the `ARGS:` log line.

| 2025-26 | control | band conversion |
|---|---|---|
| (i) squad pts vs control (m ± SE a GW) | — | **−2.27 ± 0.97** ✘ (10 better, 15 worse) |
| (ii) EP MAE | 1.0663 | 1.0670 ✔ |
| (iii) 9.0m+ component sum (limit 0.441) | 0.662 | 0.629 ✘ |
| (iv) breakout residual (limit +0.182) | +0.108 | +0.107 ✔ |
| 9.0m+ goals / assists / other | −0.267 / +0.227 / +0.158 | **−0.322** / +0.156 / +0.141 |
| £7.0-9.0m goals / assists | +0.011 / +0.088 | −0.061 / +0.053 |
| in-form residual (component sum) | +0.195 (0.343) | +0.154 (0.309) |
| captain pts/GW | 5.89 | 5.49 |

- **Why it lost.** 2024-25's conversion is above 1 for goals at £7.0m+ (1.05 at 9.0m+, 1.11 at
  £7.0-9.0m). So the lever raised premium goals, which the model already over-predicts. Captaincy
  followed those goals, and points fell by about two a gameweek.
- **What it did fix.** It lifted assists where they were short (9.0m+ +0.227 → +0.156) and cut
  cheap players' goal over-prediction a little. That wasn't enough.

## Conclusion (Phase 5)

No arm passed rule R5, so **the live model is unchanged**.

Every arm, and why it failed:

| batch | arm | target | failed on |
|---|---|---|---|
| B1 | `--bps-calibration-k 450` | 9.0m+ | (iii) |
| B1 | `--bps-tau 7` | 9.0m+ | (i) −1.03, (iii) |
| B2 | `--finishing-prior-xg 10` | 9.0m+ | (i) −0.70, (iii) |
| B2 | `--assist-prior-xa 30` | 9.0m+ | (ii) +0.008, (iii) |
| B2 | `--no-rate-price-anchor` | 9.0m+ | (ii) +0.011, (iii) — points **+1.32 ± 1.01** |
| B2 | recency (weight 2, decay 0.5) | in_form | (i) −0.30, (iii) |
| B2 | `--k-minutes 450` | in_form | (iii) |
| B3 | `--band-conversion 20` | 9.0m+ | (i) −2.27, (iii) |

**What the round established:**
- **The premium error is three model errors that partly cancel:**
  - goals over-predicted by −0.27, mostly the price anchor (−0.10 without it);
  - assists under-predicted by +0.23, because FPL awards 1.3-1.6 assists per xA and the model is
    built on xA;
  - bonus under-predicted by +0.16.
- **A lever that fixes one of them alone moves the ranking the wrong way.**
  - More bonus or more goals for premiums over-ranks premium forwards, and captaincy loses
    (bps-tau 7, band conversion).
  - Fewer premium goals (no anchor) gains points, but leaves assists and bonus short.
- **The in-form gap is real.** The residual is +0.195 over 434 player-steps: assists +0.13,
  goals +0.07. Recency closes the goals part.
- **The match feed's assists equal FPL's in 2025-26**, so the realized split is sound.

**The next round's best candidate** (not run; it needs its own declaration and the owner's go-ahead,
because this round's batch budget is spent):
- Turn the price anchor off for goals only, keeping it for assists.
- Pair that with the position-pooled assist calibration.
- Each half fixed its own component in B2: goals −0.27 → −0.10 without the anchor, assists
  +0.23 → +0.09 with the calibration.
- The open question is whether the two together keep the cheap bands and EP MAE in check.

**A note on the rule.** The model over-predicts cheap players (−0.16 below £5.0m, −0.10 overall).
So any lever that lifts predictions broadly raises MAE even when it fixes an under-predicted group.
A future rule might pair MAE with a mean-bias check by band.

## Round 2: goals-only anchor off, with the assist calibration (declared 2026-10-11, before any of its results)

**Go-ahead.** The owner started this round on 2026-10-11. It runs under the same rule R5, and the
target is the 9.0m+ component sum.

**Why this pair.** Each half fixed its own component in B2. The pairing tests whether together they
fix premiums without lifting cheap players.
- Switching the price anchor off took premium goals from −0.27 to −0.10, but it also moved assists.
- The position-pooled assist calibration took premium assists from +0.23 to +0.09.

**The new switch.** `anchor_goals` is an optional key in the `rate_prior_params` bundle.
- At 0, goals shrink toward the position average, while assists keep the price anchor.
- A bundle without the key (live v1) reads 1, so nothing live changes.
- The flag is `run_walkforward.py --no-goal-anchor`.

**Batch C1.** Three runs on one cached DB, all with `--seasons 2025-2026`:

| arm | flags | target |
|---|---|---|
| control | none | — |
| goal anchor off | `--no-goal-anchor` | 9.0m+ |
| goal anchor off + assist calibration | `--no-goal-anchor --assist-prior-xa 30` | 9.0m+ |

- Each arm is judged with `compare_arms.py control arm --rule r5 --target price_band=9.0+`. The
  in-form group is reported but does not decide.
- **If both arms pass,** the one with the larger m is promoted.
- **Promotion paths:**
  - The goals-only arm promotes as a new `rate_prior_params` bundle, which is already on the live
    path.
  - The combined arm also needs `assist_calibration_params` wired into the live call sites, as
    Phase 5 of the plan lists.
- **If m < 0,** the owner signs off first.

### Result: neither C1 arm passes; the live model is unchanged

**The batch.** It ran at 17:39 UTC on 2026-10-11 on master 4c275eb, on cached DB
`duckdb-38148073054`. All three runs completed. Run ids: control 38160537244, `--no-goal-anchor`
38160539088, `--no-goal-anchor --assist-prior-xa 30` 38160540950, each mapped by the `ARGS:` log
line.

| 2025-26 | control | goal anchor off | goal anchor off + assist calibration |
|---|---|---|---|
| (i) squad pts vs control (m ± SE a GW) | — | +1.00 ± 0.83 ✔ (9 better, 7 worse) | **−1.03 ± 0.77** ✘ (7 better, 11 worse) |
| (ii) EP MAE (limit 1.0683) | 1.0663 | 1.0745 ✘ | 1.0831 ✘ |
| (iii) 9.0m+ component sum (limit 0.441) | 0.662 | 0.523 ✘ | 0.382 ✔ |
| (iv) breakout residual (limit +0.182) | +0.108 | +0.090 ✔ | +0.046 ✔ |
| 9.0m+ goals / assists / other | −0.267 / +0.227 / +0.158 | −0.102 / +0.227 / +0.184 | −0.102 / +0.094 / +0.176 |
| £7.0-9.0m goals / assists | +0.011 / +0.088 | +0.058 / +0.088 | +0.058 / +0.004 |
| <£5.0m total (goals) | −0.164 (−0.022) | −0.182 (−0.040) | −0.192 (−0.040) |
| overall mean residual | −0.098 | −0.110 | −0.134 |
| in-form residual (component sum) | +0.195 (0.343) | +0.197 (0.337) | +0.105 (0.251) |
| captain pts/GW | 5.89 | 5.62 | 6.05 |

**Why each arm failed.**
- **Goal anchor off.** It fixed premium goals (−0.27 → −0.10), as in B2. But the anchor was
  right below £9.0m:
  - with it off, £7.0-9.0m goals went from +0.01 to +0.06 short;
  - cheap players' goals went from −0.02 to −0.04 over.
  
  That cost MAE, and premium assists and bonus stayed short, so (iii) fell short too.
- **Combined.** It passed (iii), the first arm in either round to do so, and nearly halved the
  in-form residual. But the position-pooled assist lift raised every band, including the cheap
  players the model already over-predicts:
  - the overall mean residual went from −0.10 to −0.13;
  - MAE rose by 0.017;
  - squad points fell by about one a gameweek.

**What round 2 adds.**
- The price anchor for goals is a linear fit that is right below £9.0m and overshoots above it.
  Switching it off wholesale trades a premium error for a mid-price and cheap-player one.
- The two halves fix the premium component split together, but not without a broad lift.
- The largest systematic error in the table is not at the top: cheap players (<£5.0m) are
  over-predicted by 0.16 points a player-gameweek, and they are most of the player-steps (about
  19,000 of the summary's minutes table).
  - Most of it, −0.11, is appearance points.
  - Their start probability is calibrated (predicted 0.26 against 0.27 started, from the
    summary's `minutes_prior_by_price_band`). So the excess is in the appearance points
    themselves. The hypothesis, not yet measured, is the chance of a substitute appearance for
    players who don't start.

**Next: measure the cheap-player appearance error before building anything.**
- `walkforward_summary.py` gains `minutes_states_by_price_band`. For each predicted
  player-fixture it sets the minutes model's states against what happened: start, substitute
  appearance among non-starts, 1-59 and 60+ minutes, and the appearance-point residual.
- The block splits by price band and by start probability (<0.1, 0.1-0.5, 0.5+).
- A control run on master reports it. Nothing live changes.
- An arm comes only after it shows which state carries the −0.11. That arm will need its own
  declaration.
