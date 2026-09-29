# 2026-10 live-path diagnosis: why the fixes that work in the walk-forward don't reach the team

Follow-up to `2026-10_model_status_refresh.md`. That report fixed the model **as the walk-forward
sees it** (captain variance weight, lambda, minutes floor, gate). The nightly walk-forward re-solves
the squad from scratch every gameweek, so it picked those fixes up at once. It now reads +2 to +5
pts/GW against the average manager.

The live outputs don't: the model's own team is **-69 vs the field after GW5**, and the real-squad
planner is telling the tracked account to hold with 4 free transfers banked. This report traces
why. Nearly all of it is scoring and plumbing between the parts, not the EP model itself.

State checked on `origin/master` at `653430c` (2026-09-29). Live data from the committed
`data/model_team/state.json`, `data/dashboard/*.json`, the FPL API, and the FPL-Core-Insights
`playerstats.csv` files the pipeline ingests.

## Headline

Scored by real FPL rules, the model team's **own picks** made 248 points in GW2-5. The ledger
says 188. The field average over the same weeks is 249. So about 60 of the gap is our scorer, not
our picks. Re-picking the captain each week (finding 2) would have added another 27.

| GW | ledger | real FPL rules, same picks | + weekly top-EP captain and vice |
|---|---|---|---|
| 2 | 51 | 62 | 85 |
| 3 | 60 | 67 | 74 |
| 4 | 39 | 73 | 59 |
| 5 | 38 | 46 | 57 |
| **GW2-5** | **188** | **248** | **275** |

Field average GW2-5 (`average_entry_score`): 249.

GW4 drops in the last column because the frozen-captain vice rule happened to hand the Triple
Captain to Leno's 9. Over the four weeks the weekly re-pick still wins by 27.

## Summary

| # | finding | new? | measured cost |
|---|---|---|---|
| 1 | Vice-captain rule and auto-subs never fire, because `minutes` is a season total | **new** | **60 pts in GW2-5** on the model team. Every walk-forward and backtest number is affected too |
| 2 | The forward path never re-picks the XI or captain after the GW1 solve | **new** | 27 captain-bonus pts in GW2-5, plus the frozen XI |
| 3 | Triple Captain triples that frozen captain, not the chip's own candidate | **new** | GW4 TC on a defender who didn't play |
| 4 | The captain fix and minutes floor reach 2 of ~12 optimizer callers | **new** | every other screen still uses the 3x captain penalty |
| 5 | Hold-vs-transfer only lets the "hold" side make a double move | **new** | systematic bias to "hold"; the tracked account has 4 FTs banked |
| 6 | The "average manager" benchmark is built from the model's own EP | known, re-scoped | blocks judging the role-blend and assists fixes |
| 7 | lambda 0.10 (measured +3.3 pts/GW) was never switched on | known | still v1 = 0.15 live |
| 8 | New signings / new roles under-projected (Groß, João Pedro, Kinsky...) | known | planner is recommending selling Groß |
| 9 | Triple Captain fires on weak weeks (floor too low, no "wait") | known, re-confirmed | GW4 TC with a best XI option of 4.8 EP |
| 10 | Premiums under-predicted (+0.93 at £9m+) | known | unchanged |

## 1. The vice rule and auto-subs never fire (new)

`backtest._realized_xi_points()` is the one scorer used by the model-team ledger, the forward
sim, the season sim and the walk-forward. It passes the armband to the vice, and (when given
`squad_uids`) runs `simulate_auto_substitutions()`, only for a player with **`minutes == 0`
this gameweek**. It reads `fact_player_season_stats.minutes`.

That column isn't per-gameweek. `reconcile._COLUMN_SEMANTICS` tags it `cumulative_to_date`,
and the source CSV confirms it:

| `playerstats.csv` | GW1 | GW2 | GW3 |
|---|---|---|---|
| Senesi `minutes` | 90 | 90 | 90 |
| Senesi `event_points` | 3 | 0 | 0 |

Senesi didn't play in GW2 or GW3, but his row says 90, so the check sees "played". The scorer
only treats a player as absent if they haven't played **all season**. For 2024-25 the column
doesn't exist at all (NULL, treated as "unknown"), so the rule never fires there either.

Result: the captain's zero is doubled instead of the vice's points, and a non-playing starter
scores 0 instead of being replaced from the bench. The model team had 1-3 non-playing starters
every week (Senesi each week; Watkins, Struijk and Collins in some), with bench points sitting
unused.

Two smaller gaps sit on top of this: `model_team.realize()` and `forward_season_sim` never pass
`squad_uids`/`bench_order`, so auto-subs are off there even when minutes are right. And the GW1-3
ledger rows have no `vice_captain_uid`.

**Scope.** This deflates **every** points figure the pipeline reports: the model team, the
walk-forward headline and each arm in it, and the season sims. The synthetic benchmark (finding 6)
doesn't use this scorer, so the headline gap is understated as well as noisy. Some past arm
comparisons may change, because a riskier squad loses more to missing auto-subs.

**Fix.** Per-gameweek minutes as `minutes - lag(minutes)` over the player's rows, ordered by `gw`,
or taken directly from `event/{gw}/live` for the live season. For 2024-25, which has no minutes
column, use a per-GW source (FPL API history / vaastav `merged_gw`). Pass `squad_uids` and
`bench_order` from every scorer caller. Re-score the unlocked and locked ledger rows once, with a
note in the ledger.

## 2. The forward path freezes the XI and captain (new)

`transfer_planner.apply_recommendation()` rolls holdings forward a week by copying `in_xi`,
`is_captain` and `is_vice` unchanged. A transferred-in player inherits the outgoing player's XI
slot and is never captain (`transfer_planner.py:1893-1897`). Nothing re-selects the XI or captain
for the new gameweek. Both scoring paths then read the frozen flags:

- `forward_season_sim.run_forward_season_sim()` (model team, forward plan): `forward_season_sim.py:464-477`
- `backtest.run_season_simulation()`: `backtest.py:1396-1413`

So the model team has captained **Marcos Senesi**, the pick from the GW1 solve, in every
non-Free-Hit week. That GW1 solve still carried the old 3x captain variance weight, which picks the
flattest player available. Bruno Fernandes and Watkins sit in the XI uncaptained, with Gakpo,
Rogers and Igor Jesus on the bench (`app_model_team.json`, `current_squad`).

**Cost, point-in-time, with the vice rule applied.** Each week, the alternative captain and vice
are the two XI players with the highest projected EP in the last `projections_<date>.json`
snapshot dated **before** that gameweek's deadline (no hindsight). Captain bonus from the picks
used: 23. From the weekly top-EP picks: 50. That's **+27 over four weeks**, and it doesn't count the
frozen XI keeping attackers on the bench.

The walk-forward headline doesn't have this problem: `backtest.run()` re-solves every week. That's
one reason the headline and the live team disagree.

## 3. Triple Captain triples the frozen captain (new)

`evaluate_triple_captain()` picks its own candidate (`captain_candidate`), but scoring multiplies
whichever holding has `is_captain` (`backtest.py:1409-1413`, `forward_season_sim.py:474-477`).
The chip decision and the chip's effect are about different players. The GW4 chip projected 62.15
and realised 39 (73 under real rules, with the armband going to Leno).

## 4. The captain fix and minutes floor reach 2 callers out of about 12 (new)

#198 passes `captain_risk_params_version` (multiplier 0) and `minutes_bounds_params_version`
(floor 0.005) in exactly two places: `scripts/run_ingestion.py` and `scripts/run_walkforward.py`.
Every other `squad_optimizer.run()` caller leaves them `None`, which means the old 3x captain
penalty:

- `grade_squad.py`: "Rate my team" optimal squad and upgrades
- `run_scenarios.py`, `explain_my_move.py`, `export_leaderboard.py`, `track_elite.py`
- `forward_season_sim`'s kwargs (no captain key at all), so Wildcard and Free Hit rebuilds in the
  model team still use it. That's why the GW3 Free Hit captained Ampadu over Haaland.

**Root cause:** these two families aren't in `backtest.active_recalibratable_versions()`, the one
function every script reads its active versions from. So each script has to remember to pass them
itself, and most don't.

## 5. Hold-vs-transfer compares one transfer against two (new)

`evaluate_hold_recommendation()` (`transfer_planner.py:819`) compares:

- **transfer now:** the best *single* transfer this week;
- **hold:** the better of next week's single transfer or next week's best *2-for-2 combo*.

The combo is only ever evaluated on the hold side, even when enough free transfers exist to make it
now. Two transfers nearly always beat one, so "hold" wins most weeks. It also never charges for a
free transfer wasted at the 5-FT cap (`held_free_transfers = min(5, ...)`). The tracked account
shows the result: **4 free transfers banked**, and "hold" again for GW6 (23.3 vs 16.6).

## 6. The headline benchmark moves with the model (re-scoped)

`backtest._avg_manager_benchmark_points()` builds the "average manager" from real ownership, but
weights each player by the **model's own** P(plays). Any minutes or EP change moves the benchmark:
the role blend moved it from 49.86 to 54.41. That's why the last report couldn't judge the role
blend or the assists calibration on the headline, only on raw model points.

FPL publishes the real per-GW average (`bootstrap-static events[].average_entry_score`), and
`scripts/run_model_team.py` already reads it for the current season. The walk-forward needs the
2024-25 and 2025-26 values as a small committed dataset.

## 7-10. Known items, status

- **lambda (7):** 0.15 -> 0.10 measured +3.3 pts/GW on 70 GWs (best of four values, so part of it is
  selection). Never activated: `risk_aversion_params` has only v1. It needs a v2 and the gate, and
  it should be re-measured after fix 1, since auto-subs change what a risky squad costs.
- **New signings / role changes (8):** still the biggest EP-side gap. Live example: this week's
  real-squad plan wants to sell Groß (47 pts, form 10.7, 68% elite-owned) for Ampadu. The role-blend
  fix (`bt/role-on`) was +1.07 model pts/GW on top of the captain fix, but it's parked until (6)
  gives a clean benchmark.
- **Triple Captain timing (9):** the magnitude floor (`min_tc_score` 0.1) let TC fire with the
  best XI option at 4.8 EP. The "wait for a better week" plan in the previous report still applies,
  and it only helps once (2) and (3) are fixed.
- **Premium under-prediction (10):** unchanged (+0.93 at £9m+); the assists calibration is +0.59/GW.

## Plan

Ordered by live points per unit of work. Each fix ships with a test, and each model change goes
through the gate with a walk-forward before/after.

| step | fix | size | how it's validated |
|---|---|---|---|
| 1 | **Score by real FPL rules.** Per-GW minutes (diff of the cumulative column, or `event/{gw}/live`), a per-GW source for 2024-25, and `squad_uids`/`bench_order` passed by every scorer caller. Re-score the ledger once. Covers finding 1. | small-medium | Unit test: cumulative 90 -> 90 gives 0 per-GW minutes and triggers vice + auto-sub. Replay GW2-5: ledger should read 248. Re-run the walk-forward baseline and current control; record how much each moves. |
| 2 | **Weekly XI + captain re-selection in the forward path.** After `apply_recommendation()`, re-pick a formation-legal best-EP XI, captain and vice with `reporting.rank_captain()` for that gameweek. TC scores the re-picked captain. Covers findings 2 and 3. | medium | Replay GW2-5 (expect about +27 captain pts plus bench gains). `run_season_simulation()` over 2024-26 with and without the fix. Unit tests: transferred-in players can captain; a sold captain gets replaced; TC multiplies the chip's candidate. |
| 3 | **One source of active versions.** Add `captain_risk_params` and `minutes_bounds_params` to `active_recalibratable_versions()` and remove the hand-passed copies. Add a test that fails if any `squad_optimizer.run()` caller omits them. Covers finding 4. | small | Test plus a re-run of `grade_squad` / scenarios on today's data. |
| 4 | **Symmetric hold check.** Evaluate the 2-for-2 combo on the transfer-now side whenever FT >= 2, and charge holding for a free transfer lost at the cap. Covers finding 5. | small | Unit tests for FT 1/2/4/5. Re-run the tracked account's GW6 plan. |
| 5 | **Real average-manager benchmark.** Commit per-GW `average_entry_score` for 2024-25 and 2025-26, and score the headline against it (keep the synthetic one as a secondary column). Covers finding 6. | small-medium | Headline recomputed on the cached DB for baseline and the current control. |
| 6 | **lambda v2 = 0.10** through the gate, re-measured on step 1's scorer. | small | Gate plus the existing 70-GW arms. |
| 7 | **Role blend, then assists calibration**, each judged on the step-5 benchmark. | small (already built) | Walk-forward arms on the step-5 headline. |
| 8 | **TC/BB "wait" logic** (option value against the remaining weeks, per the previous report). | medium | `run_season_simulation()` chip points per season, then the gate. |
| 9 | **Premiums (Fix C)**: penalties from match data, then bonus, then DefCon split. | medium each | Walk-forward per suspect. |

Steps 1-4 are bugs rather than model changes. Step 1 comes first because every later measurement
is taken with that scorer. Then 5, which unblocks 6 and 7. Each is its own small PR.

### Status (2026-09-29)

| step | status |
|---|---|
| 1 | Done, #202. The model team's ledger re-scores itself once on the next pipeline run (`SCORING_VERSION = 2`). |
| 2 | Done, #203. |
| 3 | Done, #204. `tests/test_live_switch_wiring.py` fails if a call site omits the switches. |
| 4 | Done with this status note: transfer-now can make the 2-for-2 when 2+ free transfers are banked, and holding at the 5-transfer cap is charged one hit. |
| 5-9 | Open. Each is judged on walk-forward runs against the pipeline's DB. |

Walk-forward numbers from before #202 were scored without the vice rule or auto-subs, and before
#204 most paths ran without the live captain and minutes settings. Re-run the baseline and the
control before comparing any step 5-9 arm against them.

## How the numbers were computed

All figures are for GW2-5 of the model-team ledger (`data/model_team/state.json`).

- **Real points:** `stats.total_points` and `stats.minutes` from
  `https://fantasy.premierleague.com/api/event/{gw}/live/`, which are per-gameweek.
- **Captain multiplier:** 3 on the triple-captain week, 2 otherwise.
- **"Real FPL rules, same picks":** the ledger's own XI, bench, captain and vice.
  - A starter with 0 minutes is replaced by the first bench player (bench ordered by projected EP)
    who played, as long as the formation stays legal (1 GK, at least 3 DEF, 2 MID and 1 FWD).
  - If the captain played 0 minutes, the vice gets the multiplier. The GW2-3 rows have no vice, so
    none is used there.
- **"Weekly top-EP captain and vice":** the same XI, with the captain and vice set to the two
  highest `ep_per_gw` players for that gameweek. EP comes from the latest
  `data/dashboard/projections_<date>.json` dated before the deadline in `app_fixtures.json`.
- **Field average:** `events[].average_entry_score` from `bootstrap-static`.
- **Cumulative minutes:** from olbauday/FPL-Core-Insights
  `data/2026-2027/By Gameweek/GW{n}/playerstats.csv`, the files `reconcile.py` loads into
  `fact_player_season_stats`.
