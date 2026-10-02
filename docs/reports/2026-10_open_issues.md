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

