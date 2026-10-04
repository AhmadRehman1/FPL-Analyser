# Open issues: hand-off for diagnosis (2026-10-01)

These are the clear problems left after the review-fix work (#215–#228). Each one has where to look, what the evidence is, and what is still unknown. They are ordered roughly by how much they affect the results. Nothing here has been fixed yet.

Walk-forward numbers are 74 gameweeks (2024-25 + 2025-26) unless noted. "Real-GW paired" means the paired per-gameweek difference against FPL's real average, which only exists for 2025-26 (n ≈ 37).

## 0. The evolving manager can make only one transfer a week, and never a hit (season sim / live model team)

This is the most obvious structural gap the season-sim arms turned up. Every arm took **0 hits over 37 gameweeks**, with 25–28 transfers in total.
- **Single transfers only:** `transfer_planner.apply_recommendation()` applies one row from `transfer_recommendations`.
  - `run()` computes two-transfer moves when 2+ free transfers are banked (`evaluate_multi_transfers` → `multi_transfer_recommendations`).
  - But no path applies them: `backtest._decide_gameweek_action` only ever returns a single `accept_transfer_rank`.
- **Hits are impossible:** after a free transfer, free transfers become `FT − 1 + 1 = FT`, so the count never drops to 0.
  - A single transfer is therefore always free.
  - The two-transfer move is only evaluated when FT ≥ 2, so `n_hits = max(0, 2 − FT)` is always 0.
  - Free transfers bank up to 5 but are never spent more than one at a time.
- **Consequence:** the live model team can't react to a double gameweek, a run of injuries, or a price-rise wave beyond one move a week. The 2.0 vs 0.5 transfer-threshold arms are nearly identical (thr2 − thr05 = +0.14 ± 0.09/GW over 37 GWs) because the threshold only ever gates that one move.

## 0b. Chips are burned at the first chance in each half

Control 2025-26 played Bench Boost in GW3, Free Hit in GW5 and Triple Captain in GW6. Then, from a fresh squad at GW19: Free Hit GW20, Bench Boost GW21, Triple Captain GW22, Wildcard GW23. The first-half Wildcard was never played.
- **The `--chip-timing` arm played exactly the same chips in the same gameweeks as control, in both halves.** The season-horizon timing gate never moved a single chip.
  - Its paired difference from control (−0.22 ± 0.64, n = 37) comes from transfer choices, not chip timing.
- **To check:** whether the TC/BB thresholds (`triple_captain_threshold_params`, `bench_boost_threshold_params`) are low enough that almost any week clears them; and why the timing gate's "a later week in the window is better" check never fires in GW3–6.

## 1. 2024-25 is scored under 2025-26 rules in more places than DefCon

#227 fixed DefCon: no defensive contribution points before 2025-26. Other rule differences between the seasons are still not modelled:

- **Chips are not season-aware.** The planner (`transfer_planner`) and `backtest._decide_gameweek_action` always allow two full chip sets, split at `GW19_DEADLINE_GAMEWEEK = 19`.
  - 2024-25 had two Wildcards (one per half), but only one Triple Captain, one Bench Boost and one Free Hit for the whole season.
  - 2024-25 also had the Assistant Manager chip, which isn't modelled at all.
  - The season-sim arms run 2025-26 only, so they are unaffected. Anything that runs `run_season_simulation` on 2024-25 plays too many chips.
- **Other scoring differences** (bonus/BPS formula changes, assist rules, etc.) haven't been checked season by season.
  - `scoring_params` has one version for all seasons, and `expected_points` resolves it the same way for every target season.
  - Worth a single "rules by season" table, with every scorer reading from it. DefCon now has its own helper (`expected_points.defcon_in_force`), but that is one special case rather than a general mechanism.

## 2. The season simulation is not reproducible run to run

Two local runs of the same code on the same DB produced different results: control, 2025-26, GW2–3.

| run | weekly points | chips | transfers |
|---|---|---|---|
| A (pre-#228) | 43, 67 | none | 1 |
| B (pre-#228, profiled) | 42, 65 | Bench Boost GW3 | 0 |
| C (post-#228) | 42, 65 | Bench Boost GW3 | 0 |

B and C match exactly, so #228 did not change behaviour, but A differs from both.
- **Likely sources:**
  - an unseeded random draw somewhere in the planner or Monte Carlo path;
  - or row order from a DuckDB query without an `ORDER BY`, feeding a tie-break.
- **Consequence:** the season-sim arm comparisons (control vs thresholds/chips) carry run-to-run noise on top of the real differences. A chip decision flipping on noise is a big swing.
- **How to find it:** run the same GW2–3 arm 3–4 times and diff `transfer_recommendations` / `chip_evaluations` between runs.

## 3. Season-sim speed: what is still slow after #228

After #228, local GW2–3 dropped from 1549 s to 602 s. That is still about 8 minutes per planner gameweek. From the cProfile of the pre-#228 run (`transfer_planner.run` → `compute_horizon_ep`):
- **`expected_points.run`:** 348 s across 6 calls, rebuilding EP for every horizon gameweek every week.
  - `player_rates_shrunk` runs 10k times (166 s); `_position_average_rates` and `_player_rate_pool` are re-queried per player.
  - These are per-asof constants and could be computed once per call.
- **`params.resolve_param`:** called 183,734 times (131 s). It is a pure lookup and could be cached per (table, key, version).
- **`snapshot.get_claims_asof` → pandas `to_dict`:** 16,820 calls (61 s).
- **`minutes_model.compute_logit_adjustment`:** 40 s.

## 4. The minutes price prior: helps calibration, slightly hurts points

`--minutes-price-prior 50`, before the DefCon fix:
- **Calibration:** MAE 1.242 → 1.202 and minutes Brier 0.362 → 0.342. That is a real improvement.
- **Points:** 60.09 vs 61.64 pts/GW, with real-GW paired −0.27 ± 0.32.
- **Where the losses came from:** a local diagnosis found the 2025-26 XIs were almost unchanged (8 of 9 steps identical). The losses were in 2024-25 cold-start steps, for example Salah with p_start 0.57 and EP 3.5 at 2024-25 GW6. The price-band prior pulled premium starters toward the band average when they had little 2024-25 history.
- **Open question:** whether the band prior should weight the player's own previous-season minutes (or a club/role prior) ahead of the price band.
- Re-run on the DefCon-corrected baseline: see the results section below.

## 5. Finishing-skill prior changes sign depending on the data window

`--finishing-prior-xg 10`:
- **v1:** 62.35 pts/GW; real-GW paired +1.43 ± 1.38.
- **v2:** rebuilds the snapshot seasons from match-level data (#224), adding 2024-25. That gave 61.35 pts/GW; real-GW paired −1.62 ± 1.27.
- **Calibration:** 9.0+ bias improved (0.76 → 0.64) but points did not.
- **Reading:** a sign flip of that size means the effect is noise at this sample size, or the ratio is unstable once 2024-25 (where xG and goal coverage differ) is included. `MAX_FINISHING_RATIO = 2.0` is a hard clamp; check how many players hit it.

## 6. Premium players' "other" points are under-predicted

From the walk-forward summary's price-band residuals, the 9.0+ band's `other` component (bonus and the like) is under-predicted by about +0.27 per player-gameweek. This is consistent with the bonus model (BPS → Plackett-Luce rank) under-rating premiums, who get more bonus than their BPS rate alone implies. See `expected_points.plackett_luce_rank_distribution` and the `tau` (BPS dispersion) parameter.

## 7. Captaincy: a large gap no simple rule closes

From run 33's captain counterfactuals (same XI):
- **Ceiling:** the hindsight best captain scores 13.0 points per gameweek, against 5.95 for top-EP.
- **Simple rules don't help:**
  - top-P95: +0.28 ± 0.37
  - EP + half SD: −0.01 ± 0.16
- **Reading:** the captain gap is about the per-player EP ranking itself, not the rule applied to it. That points back at issues 5 and 6 (premium attacking returns).

## 8. Data: rosters rewritten after transfers

Each ingestion run logs `reconcile.suspect_transfer_player_seasons`: 17 player-seasons in 2026-2027 are excluded from the minutes fit, because the source roster was rewritten after a transfer. Examples: Grealish, Marmoush, N.Jackson, Delap and Enzo. The exclusion is safe, but these are high-profile players for the live season. Their 2026-27 minutes history is being dropped rather than repaired.

## 9. Measurement caveats to keep in mind

- **`beats_crowd` pairing** isn't clean when an arm changes EP, because the crowd benchmark moves with it. Use headline points plus the real-GW paired figure.
- **Season-sim chunks:**
  - Each chunk (GW2–18, GW19–38) starts from a fresh squad with its half's chips, so the second half doesn't inherit the first half's squad or bank.
  - A chunk cut off by the step time limit is flagged "incomplete" in `SUMMARY.md`.
- **mypy baseline:** 75 errors in 19 files, pre-existing. CI only checks that the count doesn't grow.

## Results on the DefCon-corrected baseline

_From the runs dispatched after #227/#228 merged._

### Walk-forward (74 GWs, master after #227)

| arm | pts/GW | real-GW paired vs control (2025-26, n=37) | MAE | 9.0+ total resid |
|---|---|---|---|---|
| control (corrected 2024-25 scoring) | 61.92 | - | 1.226 | +0.78 |
| finishing prior xG 10 | 62.07 | −1.62 ± 1.27 | 1.227 | +0.66 |
| minutes price prior 50 | _timed out_ | - | - | - |

- **DefCon fix:** control moved from 61.54 (run 33, before the fix) to 61.92. Per-season beats-crowd:
  - 2024-25: +4.80 → +5.53
  - 2025-26: +2.29 → +2.48
  - The 2025-26 move is unexpected, because DefCon was already correct for that season. That points at issue 2 (run-to-run nondeterminism).
- **Finishing:** gives the same real-GW result as before (−1.62 ± 1.27). Not worth promoting.
- **Price prior:** this run hit the 330-minute job limit and left no summary file. The walk-forward has no checkpointing (unlike the season-sim chunks).
  - It needs a re-dispatch on a quiet runner, or the season-sim-style checkpoint/chunking.
  - Its earlier (pre-DefCon) numbers are in issue 4.
- **Captain counterfactuals (control):** top-P95 +0.43 ± 0.40, EP + half SD −0.16 ± 0.15. Same picture as issue 7.

### Season simulation (2025-26, chunks GW2–18 + GW19–38, run 36916902463)


| arm | changed | net pts/GW | real avg/GW | vs real avg | vs control (paired) | hits | transfers | chips |
|---|---|---|---|---|---|---|---|---|
| control | live settings | 60.30 | 49.76 | +10.54 | - | 0 | 28 | GW3 bench_boost, GW5 free_hit, GW6 triple_captain, GW20 free_hit, GW21 bench_boost, GW22 triple_captain, GW23 wildcard |
| chips | triple_captain_timing_params_version=1, bench_boost_timing_params_version=1 | 60.08 | 49.76 | +10.32 | -0.22 ± 0.64 (n=37) | 0 | 28 | GW3 bench_boost, GW5 free_hit, GW6 triple_captain, GW20 free_hit, GW21 bench_boost, GW22 triple_captain, GW23 wildcard |
| thr05 | accept_transfer_if_net_value_above=0.5 | 57.59 | 49.76 | +7.84 | -2.70 ± 1.57 (n=37) | 0 | 26 | GW3 bench_boost, GW5 wildcard, GW6 triple_captain, GW11 free_hit, GW20 free_hit, GW21 bench_boost, GW22 triple_captain, GW23 wildcard |
| thr2 | accept_transfer_if_net_value_above=2.0 | 57.73 | 49.76 | +7.97 | -2.57 ± 1.57 (n=37) | 0 | 26 | GW3 bench_boost, GW5 wildcard, GW6 triple_captain, GW11 free_hit, GW20 free_hit, GW21 bench_boost, GW22 triple_captain, GW23 wildcard |

- **Control** beats FPL's real average by +10.5 net points per gameweek. Part of that is the fresh-squad restart at GW19, which acts as a free Wildcard.
- **Transfer threshold:** raising it from 0 to 0.5 or 2.0 costs about 2.6 points per gameweek, and the two arms barely differ from each other (issue 0).
- **Chip-wait arm:** its chunks were still running when this was written. Results will be in the run's `season-sim-summary` artifact.

### Season-sim arms (2025-26, run 37201797073, master after #231/#232)

| arm | changed | net pts/GW | vs real avg | vs control (paired) | hits | transfers | chips |
|---|---|---|---|---|---|---|---|
| control | live settings | 60.24 | +10.49 | - | 0 | 28 | GW3 BB, GW5 FH, GW6 TC, GW21 FH, GW22 WC, GW23 BB, GW24 TC |
| multi | `--multi-transfers` | 59.89 | +10.14 | −0.35 ± 1.44 | 36 | 44 | as control |
| chipov | `--chip-option-value` | 59.51 | +9.76 | −0.73 ± 2.01 | 0 | 27 | GW3 BB, GW5 WC, GW8 TC, GW11 FH, GW21 BB, GW22 WC, GW26 TC, GW31 FH |
| both | both flags | 60.92 | +11.16 | +0.68 ± 1.36 | 40 | 45 | GW3 BB, GW5 FH, GW6 TC, GW21 BB, GW22 WC, GW26 TC, GW34 FH |

Paired over the same 37 gameweeks (n = 37); FPL's real average is 49.76 a gameweek.

- **No arm separates from control.** Every difference is under one standard error, so both settings stay
  opt-in.
- **Multi-transfers** takes hits now (9 in the season: 5 in GW2–19, 4 in GW20–38). Set 1
  +0.78 ± 1.47, set 2 −1.42 ± 2.46.
- **Chip option value** spreads the chips out instead of playing them at the first chance. Set 1
  −4.22 ± 3.38 (Triple Captain GW6 → GW8, Free Hit GW5 → GW11), set 2 +2.58 ± 2.08.
- **Both together:** set 1 is identical to multi-transfers alone. On that squad the option-value rule
  agreed with the old one (BB GW3, FH GW5, TC GW6), so only set 2 differs (+0.58 ± 2.28).


## Status (2026-10-03)

| # | status | what changed / what's left |
|---|---|---|
| 0 | built, opt-in; arm level with control | `multi_transfer_params` v1: a week with no chip makes the best two-transfer combination instead of the best single when it's worth more net of its hit. `apply_recommendation()` applies combinations and spends free transfers first. Season-sim arm `--multi-transfers` (2026-10-04): −0.35 ± 1.44 pts/GW against control, 9 hits and 44 transfers to control's 0 and 28 (results below). Stays opt-in. |
| 0b | diagnosed, fixed, opt-in rule | The magnitude floors never bind (TC 0.1 vs ~6, BB 0.5 vs ~8, FH 1.5). The `--chip-timing` arm could not differ from control: the simulations showed only the planning horizon's 5 weeks of fixtures, so the 10-week window was the 5-week one (fixed). The gate plays at the first local maximum of a sliding window, and set 2 has none. `chip_wait_params` v2 (`--chip-option-value`) holds a chip unless it beats the value of every week left in the half (backward induction, unseen weeks as draws from the projected ones), for FH too. Season-sim arm (2026-10-04): −0.73 ± 2.01 pts/GW against control, so it stays opt-in (results below). |
| 1 | fixed | `season_rules.py`: one table of DefCon, chip allowances and BPS weights by season, read by every scorer and planner. 2024-25 gets one FH/BB/TC for the season and two Wildcards. GW19 is the first half's last week (it was treated as a second-half week, in the app's planner too). BPS: +1 per 2 CBI until 2025-26 (per 3 since), keeper saves 2 BPS in 2024-25. Not modelled: 2024-25's Assistant Manager chip, 2025-26's relaxed assist definition. |
| 2 | fixed | DuckDB pinned to one thread (multi-threaded DISTINCT/GROUP BY order and float sums changed every run), ordered query inputs, Monte Carlo seeded on what it simulates instead of DB sequence ids, tie-breaks by player_uid. The synthetic season simulation now repeats bit for bit across processes and DB histories. |
| 3 | fixed | One memo per asof view shared across the horizon's EP/uncertainty/MC runs, `resolve_param()` cache, one claims query per run, the timing window reuses the horizon. Synthetic 4-GW season sim 28.4s -> 10.6s; the test suite 12m47s -> ~6m. Not yet timed on the real DB. |
| 4 | promoted (live 2026-10-04) | Price is the last fallback (2026-10-04 decision and result below). A thin history shrinks toward the player's own record this season, else his earlier seasons, else a start rate that rises with price, each level counting as 5 matches. 2025-26: points level with control against the real average, minutes MAE 1.27 -> 1.10, Brier 0.36 -> 0.30. `--no-minutes-start-prior` runs the old position-average prior. |
| 5 | diagnostic | `scripts/diagnose_finishing_ratios.py` prints clamp counts and ratio quantiles per window; needs a run on the ingested DB. |
| 6 | two opt-in arms | `--bps-calibration-k 450` adds each player's BPS the estimate misses (real season BPS minus the estimate's own terms on his matches, per 90, shrunk to his position). `--bps-tau 7` sharpens the Plackett-Luce bonus split (live 10). |
| 7 | follows 5 and 6 | Also: Monte Carlo and uncertainty added ball recoveries to a defender's DefCon (the EP engine doesn't), inflating defenders' simulated points, which Triple Captain picks from. Fixed. |
| 8 | checked; repair in its own PR | No opponent or home/away field exists, but the weekly roster snapshots are point-in-time and every mover's match rows agree with them, so nothing was rewritten (2026-10-04 decision and check below). The minutes fit can measure each match against that week's club instead of dropping movers. |
| 9 | partly addressed | Walk-forward: `--max-minutes` (branch_walkforward.yml passes 300), `--resume RUN_ID`, `--seasons 2025-2026`, and a `progress` block in the summary. mypy stays at its 75-error baseline. |

**Baselines move.** Re-run control before comparing any arm: outputs are now deterministic but not
bit-identical to earlier runs (summation order, Monte Carlo seeds), Monte Carlo no longer inflates
defenders, 2024-25 and 2025-26 use their own BPS weights, 2024-25 plays fewer chips, and GW19 is a
set-1 week (season-sim chunks default to GW2-19 and GW20-38).

**Runs to dispatch:**
- Walk-forward control, one job per season (`--seasons 2025-2026`, `--seasons 2024-2025`):
  dispatched 2026-10-04, runs 37163852286 and 37163854338. Then `--bps-calibration-k 450` and
  `--bps-tau 7`. Issue 4's start prior is done (promoted, above); a control dispatched after
  2026-10-04 includes it.
- Season-sim control (2025-26): dispatched 2026-10-04, run 37163855869. Then
  `--multi-transfers`, `--chip-option-value`, and both together: done, run 37201797073 (below).
- `scripts/diagnose_finishing_ratios.py` against the ingested DB.

## Decisions (2026-10-04)

### Issue 4: price is the last fallback; judge on 2025-26

- **Order of evidence** for a player's start prior: his own record this season once he has
  one, then his own record in earlier seasons, and a price-based prior only when neither exists.
  - "Record" means the team matches he was available for, so zero minutes in matches he could
    have played counts as evidence of not being picked.
  - Each level counts the next one down as a few extra matches (`pseudo_matches`), so a thin
    record is steadied, not replaced.
- **The price prior is a curve, not a band.** A start rate rising steadily with price
  (`minutes_model.fit_price_start_curve()`), fitted to every player's starts out of the matches
  he was available for. The 9.0+ band average pooled rotated and injured weeks and held Salah
  at 0.57; premiums should come out around 0.85 or higher.
- **Judge on 2025-26.** The live season always has two earlier seasons, so 2024-25's cold
  start only happens in the backtest. The earlier arm's losses were almost all 2024-25
  cold-start steps, and its 2025-26 picks were nearly unchanged.
  - Decide on the 2025-26 real-average comparison plus calibration.
  - Report 2024-25 separately as a cold-start stress test.
- **Promote if** it keeps the calibration gain (MAE about 1.24 → 1.20, minutes Brier
  0.36 → 0.34) without losing points against the real average in 2025-26.
- **Scoreboard.** `walkforward_summary.py` now reports `headline_by_season`, and
  `minutes_prior_by_price_band` (mean start prior vs the share that started, by season, tier
  and price band) for the premium check.

**Result: promoted.** Walk-forward on master after #231 and the issue 8 repair (#232), one job
per season and arm, runs 37200700553–37200708321 (#233's branch: master plus the variant's
option, off unless asked). The variant counted a record only once a player had appeared
(`min_appearances` 1), so a player who never played fell back to the price curve instead of a
record of zeros.

| 2025-26 (decides) | control | start prior | variant: appearances only |
|---|---|---|---|
| squad points vs control, paired per GW | | **+0.03 ± 0.27** | −0.51 ± 0.35 |
| EP calibration MAE | 1.273 | **1.095** | 1.173 |
| minutes Brier | 0.364 | **0.297** | 0.326 |
| minutes log score | −0.627 | **−0.531** | −0.574 |

The real average is the same for every arm, so points against it differ exactly as the squad
points do.

- **2024-25 cold start** (no real average in the data, so squad points): the start prior
  +2.54 a gameweek over control, MAE 1.185 -> 1.068, Brier 0.362 -> 0.298. The variant
  +2.27, MAE 1.139, Brier 0.333.
- **Premiums:** the 9.0+ start prior is 0.90 at the 2024-25 cold start (76% started) and
  0.87–0.89 in 2025-26 (64–80% started), against Salah's 0.57 under the band average.
- **The variant is rejected:** it lost half a point a gameweek and kept less of the
  calibration gain. Zero minutes in matches a player was available for are evidence.
- **The crowd benchmark is not a comparison here.** `beats_crowd` fell (2025-26 −0.90 ± 0.34,
  2024-25 −2.54) because the synthetic average manager discounts each owned player by the
  model's own chance of that player playing, so a better minutes model raises the benchmark
  itself. FPL's real average doesn't move with the model.
- **Season sim** (the evolving manager with transfers and chips; run 37203343610 against
  37201797073's control): −1.84 ± 2.02 pts/GW, n = 37, within noise. One week carries it. At GW6
  both played Triple Captain on squads that differed after GW5's chip (a Wildcard instead of
  the Free Hit): 31 points against 78. Without GW6 the gap is −0.58 ± 1.62.
- **Not yet re-tuned:** the M7 recalibration still scores the minutes model with the old
  prior, like the floor, so a refit of `competitive_matches_threshold` is tuned for the old
  prior.

### Issue 8: keep the exclusion

- **Impact is small.** The minutes fit is mostly about the player, not the club, so dropping
  17 player-seasons costs little. Club matters mainly for team-strength attribution and
  teammate covariance; look there if anything breaks.
- **Cheap repair, if the data allows.** If the per-gameweek rows carry the opponent and a
  home/away flag (FPL's own gameweek data does), the player's club for that match is the other
  side of that fixture: a lookup, not an inference.
- **Check first.** Confirm those columns exist and weren't rewritten too: spot-check Grealish
  and Marmoush against the provider's files. If they're clean, the repair is small; if not,
  the exclusion stays.

**What the check found (FPL-Core-Insights, 2026-10-04):**
- The per-gameweek player rows carry no opponent, home/away or team field, and
  `playermatchstats.csv` has `match_id` but no team.
- The weekly roster snapshots (`By Gameweek/GW{n}/players.csv`) are point-in-time.
  - 2026-27: Grealish is at Man City for GW1–2 and Everton from GW3; Marmoush is at Man City
    for GW1 and Spurs from GW2.
  - 2025-26: Isak is at Newcastle to GW3 and Liverpool from GW4. Marmoush, Enzo, Delap and
    N.Jackson keep their 2025-26 clubs all season.
- Every mover's Premier League match row sits in a fixture of the club his snapshot names
  that week (604 rows in 2025-26, 67 in 2026-27, no mismatches).
- So nothing was rewritten. The season-root roster lists each player's latest club, and the
  flagged players are within-season movers: 26 in 2025-26, the 17 in 2026-27.
- The repair is a lookup on the weekly snapshot rather than on the fixture's other side. It
  is in its own PR so it doesn't confound the issue 4 comparison.
