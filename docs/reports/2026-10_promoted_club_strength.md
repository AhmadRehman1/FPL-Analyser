# FPL model gap: Coventry and Hull at league-average strength

Oct 5, 2026 · @Ahmad Rehman

## Summary

Your hypothesis holds on all three links, and it is not a name mismatch. The obvious fix is unsafe on its own, though: the same call that would bring 2026-27 results into the live fit gave Coventry an attack rating of -13 at the GW5 deadline.

- **Cause.** The live fit stops at 2025-26, neither club has a match in it, and the only Elo the code looks at is blank or does not list them. Ipswich is a third casualty: live uses their raw 2024-25 fit, unshrunk, with no warning.
- **Size.** Live gives both clubs 1.25 goals for and 1.46 against per match versus the rest of the league. A shrunk refit puts Coventry at 0.68 and 1.78, and Hull at 1.02 and 1.43. Coventry is badly over-rated; Hull only in attack.
- **Who it hits.** Coventry's defenders and keeper are about 0.25 points a gameweek too high. Defenders and keepers facing Coventry are 0.4 to 0.9 points too low in that gameweek. Coventry's own attackers barely move, because a club's attack rating does not feed its own players.
- **Recommended fix.** Pass `fit_seasons_for(TARGET_SEASON)` in live, and in the same change make `calibrate()` shrink thin fits: a promoted-club prior when there is no Elo, weight by matches played, a clamp, and no change to the established clubs' 2/3 weight.
- **How sure.** On 2025-26 match scores the recommended version is level with live today, not better. The case is that it stops fits diverging, stops stale fits going in unshrunk, and makes live and backtest the same model.
- **Side question.** Yes, backtests look ahead on Elo. Sunderland's 2025-26 prior was 189 points above what was known at GW1.

## Causal chain

All three links hold. A local ingestion at `6d23661` logged the same two warnings and wrote the snapshot below.

| Link | Verdict | Evidence |
| --- | --- | --- |
| 1. Live fit ignores 2026-27 | Confirmed | `scripts/run_ingestion.py:193-196` calls `calibrate()` with no `fit_seasons`; the default is at `team_strength.py:205`. The live `team_strength_model_versions` row has `seasons_fit = ["2024-2025", "2025-2026"]`. `fact_match` holds 50 finished 2026-27 league matches that the fit never reads. It is the only caller on the default: `backtest.py:409, 1736, 1807, 2162` and `forward_season_sim.py:373, 406` all pass `fit_seasons_for(season)`. |
| 2. No MLE fit for Coventry and Hull | Confirmed | FPL codes 9 (Coventry) and 88 (Hull) are absent from the 2024-25 and 2025-26 `teams.csv`. Each has 38 league rows in `fact_match`, all 2026-27. Snapshot: `attack_mle` and `defence_mle` NULL, `seasons_of_topflight_data = 0`. |
| 2b. Ipswich gets the stale fit | Confirmed, and worse than expected | `team_ipswich` has 38 league matches in 2024-25 and none in 2025-26. Its live strength is the raw 2024-25 fit: attack -0.348, defence -0.643, both last of 20. It has no Elo prior either, so `team_strength.py:309-310` uses the fit unshrunk, and no warning is logged. |
| 3. No Elo prior | Confirmed | `data/2026-2027/teams.csv` has `elo` blank on all 20 rows, and so does every `By Gameweek/GW1` to `GW38` copy. `fetch_current_elo()` (`team_strength.py:164-178`) returns `{}`. The fallback at `team_strength.py:254-259` reads 2025-26's file, which lists Burnley, West Ham and Wolves, not Coventry, Hull or Ipswich. Both clubs fall to `team_strength.py:311-318`. |

Live values for both clubs: `final_attack = 0.000`, `final_defence = -0.1745`, with `attack_elo_prior`, `defence_elo_prior` and `elo_at_calibration` all NULL.

The name-mismatch theory is refuted, so the private workbook was not needed for this.

- `team_alias` has `Coventry City` and `Hull City` rows for 2026-27, sourced from `2026-2027:teams.csv`.
- `apply_club_name_map()` only adds aliases to clubs already in `dim_team`. It cannot create matches or Elo values.
- The warning text at `team_strength.py:313-317` is therefore wrong for these clubs. The comment above it (`team_strength.py:279-286`) describes the old Ipswich spelling split, which `reconcile.py` now handles by FPL code.
- The comment at `reconcile.py:84-85` (a promoted club "still gets a ... real Elo prior") is false whenever the target season's `elo` column is blank. That has been the case for all of 2026-27 so far.

Two corrections to the brief:

- The dataset does hold an Elo for both clubs, just not where the code looks. 17 of the 50 finished 2026-27 league rows in `matches.csv` carry `home_team_elo` and `away_team_elo`: Coventry 1661.48, Hull 1532.88, Ipswich 1640.00. These already land in `fact_match`.
- `fit_seasons_for()` has returned three seasons for 2026-27 since `6b234f5` (#86, 2026-09-01), not #203.

## Impact

The fallback flatters Coventry a lot and Hull a little. The points cost falls mainly on defenders and keepers who face Coventry, at 0.4 to 0.9 points in that gameweek.

Three versions are compared below.

- **Live**: today's snapshot.
- **Refit as coded**: `fit_seasons_for("2026-2027")` passed to the current `calibrate()`, which is fix option 1 on its own.
- **Recommended**: the same refit with the shrinkage described under Fix options.

### Results so far

Per match, after 5 league matches each (50 finished in total).

|  | Goals for | Goals against | xG for | xG against |
| --- | --- | --- | --- | --- |
| Coventry | 0.20 | 2.00 | 0.97 | 1.73 |
| Hull | 1.20 | 0.80 | 1.05 | 1.48 |
| League average | 1.41 | 1.41 | 1.52 | 1.52 |

Coventry have met Arsenal, Man City and Brighton already. Hull have 8 points despite losing the xG count.

### Team strength

Attack and defence are the stored log-scale values. They are not comparable across fits, because each fit re-centres on its own mean. The goals columns are comparable: expected goals per match against the other 19 clubs at a neutral venue.

| Club | Version | Attack | Defence | Goals for | Goals against |
| --- | --- | --- | --- | --- | --- |
| Coventry | Live | 0.000 | -0.175 | 1.25 | 1.46 |
| Coventry | Refit as coded | -1.476 | -0.261 | 0.27 | 1.69 |
| Coventry | Recommended | -0.548 | -0.330 | 0.68 | 1.78 |
| Hull | Live | 0.000 | -0.175 | 1.25 | 1.46 |
| Hull | Refit as coded | -0.060 | 0.351 | 1.13 | 0.89 |
| Hull | Recommended | -0.163 | -0.126 | 1.02 | 1.43 |

"League average" here is the mean of the 23 clubs in the two-season fit, relegated clubs included. Among the current 20 it ranks joint 14th for attack and joint 15th for defence. It sits 0.13 below the 17 established clubs' mean attack and 0.10 below their mean defence. The six promoted clubs in the dataset averaged 0.39 and 0.32 below in their first season.

### Players, GW6 to GW10

These tables isolate the gap: the live snapshot with only Coventry and Hull moved to the recommended values. The local baseline matches the published `projections_latest.json` to within 0.05 points on 99.6% of player-gameweeks.

A club's own attack rating never reaches its own attackers. `_fixture_attack_multiplier()` in `expected_points.py` cancels `own_attack`, so Coventry's midfielders and forwards barely move. The rating matters through clean sheets, goals conceded and saves.

Coventry and Hull players, points per gameweek averaged over the five:

| Player | Club | Pos | Live | Corrected | Change |
| --- | --- | --- | --- | --- | --- |
| Bobby Thomas | Coventry | DEF | 2.85 | 2.58 | -0.28 |
| Jay Dasilva | Coventry | DEF | 2.66 | 2.39 | -0.27 |
| Carl Rushworth | Coventry | GK | 2.19 | 1.97 | -0.22 |
| Milan van Ewijk | Coventry | DEF | 1.55 | 1.39 | -0.16 |
| Ethan Pinnock | Coventry | DEF | 1.72 | 1.57 | -0.15 |
| Matt Grimes | Coventry | MID | 2.62 | 2.57 | -0.05 |
| Konstantinos Tzolakis | Hull | GK | 2.97 | 3.00 | +0.03 |
| John Egan | Hull | DEF | 2.67 | 2.70 | +0.02 |
| Oli McBurnie | Hull | FWD | 3.22 | 3.21 | -0.01 |

Opponents, points in the gameweek they face Coventry or Hull:

| Player | Club | Pos | Fixture | Live | Corrected | Change |
| --- | --- | --- | --- | --- | --- | --- |
| Calvin Bassey | Fulham | DEF | GW8 Coventry (A) | 3.00 | 3.92 | +0.92 |
| Micky van de Ven | Spurs | DEF | GW7 Coventry (H) | 2.90 | 3.81 | +0.91 |
| Thomas Meunier | Sunderland | DEF | GW9 Coventry (A) | 2.52 | 3.40 | +0.88 |
| Lewis Hall | Newcastle | DEF | GW6 Coventry (A) | 2.85 | 3.73 | +0.88 |
| Malick Thiaw | Newcastle | DEF | GW6 Coventry (A) | 3.57 | 4.36 | +0.79 |
| Jake O'Brien | Everton | DEF | GW10 Coventry (H) | 2.65 | 3.37 | +0.73 |
| Jordan Pickford | Everton | GK | GW10 Coventry (H) | 4.11 | 4.76 | +0.64 |
| Bernd Leno | Fulham | GK | GW8 Coventry (A) | 3.95 | 4.56 | +0.62 |
| Harvey Barnes | Newcastle | MID | GW6 Coventry (A) | 2.84 | 3.18 | +0.34 |
| Brennan Johnson | Everton | MID | GW10 Coventry (H) | 2.83 | 3.16 | +0.33 |
| Gonzalo Garcia | Fulham | FWD | GW8 Coventry (A) | 3.34 | 3.65 | +0.31 |
| Calvin Bassey | Fulham | DEF | GW7 Hull (H) | 3.31 | 3.61 | +0.31 |
| Jordan Pickford | Everton | GK | GW6 Hull (A) | 3.88 | 4.10 | +0.21 |

- Against Coventry, likely starters gain on average 0.64 to 0.71 as defenders, 0.38 to 0.64 as keepers, 0.17 to 0.31 as forwards and 0.10 to 0.23 as midfielders.
- Against Hull, defenders gain 0.18 to 0.24 and keepers 0.13 to 0.23. Attackers move by -0.09 to +0.01.
- Fulham and Everton meet both clubs in the window. Bassey gains 1.23 points over the five gameweeks, or 0.25 per gameweek; Pickford gains 0.86, or 0.17.
- Refit as coded would roughly double the gains against Coventry (Bassey +1.94 in GW8) and add 0.57 to 0.65 per gameweek to Hull's four regular defenders and 0.73 to Tzolakis.

### All 20 clubs

Expected goals per match against the rest of the league, live against recommended, largest move first. The mean absolute move is 0.12 goals for and 0.09 against. Refit as coded moves clubs further: 0.19 and 0.13.

| Club | For, live | For, rec. | Change | Against, live | Against, rec. | Change |
| --- | --- | --- | --- | --- | --- | --- |
| Coventry | 1.25 | 0.68 | -0.57 | 1.46 | 1.78 | +0.32 |
| Ipswich | 0.85 | 1.03 | +0.18 | 2.37 | 1.94 | -0.43 |
| Brighton | 1.38 | 1.69 | +0.31 | 1.30 | 1.24 | -0.06 |
| Hull | 1.25 | 1.02 | -0.23 | 1.46 | 1.43 | -0.03 |
| Chelsea | 1.38 | 1.53 | +0.16 | 1.42 | 1.53 | +0.11 |
| Brentford | 1.44 | 1.52 | +0.08 | 1.38 | 1.26 | -0.12 |
| Everton | 1.26 | 1.22 | -0.05 | 1.41 | 1.25 | -0.15 |
| Liverpool | 1.66 | 1.60 | -0.07 | 1.31 | 1.18 | -0.13 |
| Nott'm Forest | 1.40 | 1.28 | -0.13 | 1.32 | 1.29 | -0.03 |
| Spurs | 1.21 | 1.08 | -0.12 | 1.58 | 1.56 | -0.02 |
| Leeds | 1.29 | 1.30 | +0.01 | 1.45 | 1.33 | -0.12 |
| Newcastle | 1.43 | 1.51 | +0.09 | 1.43 | 1.48 | +0.05 |
| Aston Villa | 1.56 | 1.49 | -0.07 | 1.30 | 1.35 | +0.04 |
| Crystal Palace | 1.18 | 1.21 | +0.03 | 1.48 | 1.57 | +0.09 |
| Man City | 1.90 | 1.99 | +0.09 | 0.98 | 1.00 | +0.01 |
| Fulham | 1.22 | 1.18 | -0.04 | 1.35 | 1.37 | +0.02 |
| Man Utd | 1.69 | 1.68 | -0.01 | 1.22 | 1.25 | +0.03 |
| Arsenal | 1.93 | 1.89 | -0.04 | 0.85 | 0.86 | +0.01 |
| Bournemouth | 1.51 | 1.47 | -0.04 | 1.29 | 1.28 | 0.00 |
| Sunderland | 1.14 | 1.16 | +0.02 | 1.56 | 1.57 | +0.02 |

Ipswich is the second-largest error and runs the other way. Live rates them far the worst side in the league on a 2024-25 record. Under the full recommended fix their defenders gain 0.37 to 0.50 points per gameweek (Dara O'Shea +0.50, Leif Davis +0.46).

## Fix options

Recommended: make the live call use `fit_seasons_for(TARGET_SEASON)`, but only together with a guard that stops `calibrate()` using a thin fit unshrunk. Option 1 alone would make live worse than it is today.

### Why option 1 alone is unsafe

The current `calibrate()` has no defence against a club that has not yet scored, or not yet conceded. The fit for that club runs off to minus or plus infinity. Replaying the real code at each deadline with `asof_scope()`:

| Step | Club | Stored value | Cause |
| --- | --- | --- | --- |
| 2026-27, GW2 to GW5 | Coventry | attack -11.7 to -13.9 | no goals in their first four matches |
| 2026-27, GW2 to GW4 | Hull | defence +11.9 to +13.1 | no goals conceded in their first three |
| 2026-27, GW6 (today) | Coventry | attack -1.48 | one goal in five |
| 2025-26, GW2 | Burnley | attack -5.31 | blanked in GW1; one third of a diverged fit, even with an Elo prior |
| 2025-26, GW2 | Leeds, Sunderland | defence +4.93, +4.73 | clean sheets in GW1 |

- This is the path the 2026-27 walk-forward and forward season sim already take, so those steps are affected today.
- The damage spreads. `_league_defence_and_home_adv()` averages the stored defences, so Hull at +12 lifted every attacker's fixture multiplier at 2026-27 GW3 to about 1.8 to 2.0, against 1.0 to 1.1 normally.
- The `elif a_mle is not None` branch (`team_strength.py:309-310`) is the direct cause for clubs with no Elo. The blend at `team_strength.py:304-306` has the same problem at one third weight.

### What option 1 does to the Elo regression

- Eligibility is unchanged. Today it is the 15 clubs in both prior seasons; with three seasons it is the same 15.
- Coefficients barely move: attack slope 0.001818 to 0.001824, defence slope 0.001814 to 0.001725.
- The weight changes, though. Counting 2026-27 as a season lifts those 15 clubs from `weight_own_data` 2/3 to 1.0, so their Elo prior drops out entirely. In a 2025-26 replay, pure fit scored 0.0195 log-likelihood per match worse than the 2/3 blend over 272 matches (t = 3.2).

### The options

| Option | Verdict | Reason |
| --- | --- | --- |
| 1. `fit_seasons_for(TARGET_SEASON)` in live | Do it, with the guard | Lines live up with every other caller and lets 2026-27 results in. Unsafe alone, as above. |
| 2. Promoted-club prior | Adopt as the fallback when there is no Elo | Six promoted clubs in the data averaged attack -0.39 and defence -0.32 against the rest: 33% fewer goals scored, 37% more conceded. The spread is wide, from Southampton (-0.78, -0.53) to Leeds (-0.06, -0.11). |
| 3. Another Elo source | Worth doing, second | `fact_match` already holds a per-match Elo that is point-in-time. This season it covers GW1 and GW2 only and the values are frozen, so the upstream feed looks stalled. ClubElo's own API is the obvious independent source; I could not reach it from this session to test it. |
| 4. Correct warning | Do it regardless | Suggested text: "no top-flight match in fit\_seasons and no Elo in teams.csv for the target or fallback season (promoted club, or upstream elo column blank)". Fix the comments at `team_strength.py:279-286` and `reconcile.py:84-85` too. |

### Recommended change

1. `scripts/run_ingestion.py:193`: pass `fit_seasons=backtest.fit_seasons_for(TARGET_SEASON)`.
2. `calibrate()`: when a club has no Elo prior, use a promoted-club prior instead of the raw fit or league average. Compute it from the fit: the mean first-season offset of earlier promoted clubs from the established clubs' mean.
3. For a club in its first season in the fit, weight its own fit by matches played, n / (n + 10), and clamp the fit to within 1.0 of the prior before blending. At five matches this gives 1/3, the same as today's `seasons / 3`.
4. Do not count the in-progress season in `seasons_of_topflight_data`, so the 15 ever-present clubs keep their 2/3 weight.
5. Replace the warning and the two comments.

A quick check of the alternatives, scoring 2025-26 match scores walk-forward (mean log-likelihood per match, higher is better):

| Version | All 380 | 108 with a promoted club |
| --- | --- | --- |
| Live today: prior seasons only, promoted clubs at league average | -2.916 | -2.943 |
| Option 1 as coded: raw fit for promoted clubs | -3.187 | -3.876 |
| Backtest today: 1/3 fit plus 2/3 end-of-season Elo | -2.996 | -3.273 |
| Recommended | -2.914 | -2.963 |

### Trade-offs

- The recommended version matches live today on this check; it does not beat it. Any sensible prior for promoted clubs is within one standard error of any other on 108 matches. The case for the fix is removing the failure modes and making live and backtest the same model, not a proven accuracy gain.
- League average was not a bad guess in 2025-26, because Leeds and Sunderland were close to average. It is a worse guess for a Coventry.
- The constants (10 matches, clamp of 1.0) are mine and untuned. The prior rests on six clubs.
- The fit uses goals, not xG. Coventry's one goal from 4.9 xG may make the recommended attack value too harsh.
- Baselines move: the 2026-27 walk-forward and forward sim change, which is intended.
- `fit_seasons_for()` raises for any season it does not list. Once live depends on it, next summer's rollover needs a 2027-28 entry or ingestion fails.

## Validation

The 2025-26 walk-forward cannot test this fix as it stands, for the reason in the side question: promoted clubs always have an Elo prior there, and it is the end-of-season one. Fix that first, then compare.

1. **Make Elo point-in-time.** Read each club's latest `fact_match` Elo from rows before the deadline instead of the root `teams.csv`. Re-run the control so the baseline is honest.
2. **Branch walk-forward** (`branch_walkforward.yml` on a `bt/` branch, `--seasons 2025-2026`). `run_walkforward.py` has no team-strength flag yet, so it needs one. Four arms:
   - control: current `calibrate()`.
   - live-like: prior seasons only, promoted clubs at league average. This is the arm that reproduces today's live gap.
   - fix: the recommended blend.
   - fix with Elo withheld from clubs that have no prior-season fit. This mimics 2026-27 and is the only arm that exercises the promoted-club prior.
3. **What to read.** The `walkforward_summary.py` scoreboard cut to two groups: promoted clubs' players, and players in fixtures against them. Add match-score log-likelihood per gameweek from `team_strength_snapshots`; it is the direct measure and costs nothing.
4. **Regression test.** The 2026-27 GW2 to GW5 steps and 2025-26 GW2 must give no stored attack or defence beyond about 1.5 in size. Add a unit test for a club with no goals scored.
5. **Season sim** (`season_sim_arms.yml`, 2025-2026, chunks 2-19 and 20-38): `control` against a fix arm, as a guard that net points do not fall. It will not be sensitive enough to pick between priors.

Set the pass mark accordingly: failure cases gone, and no worse than control within noise. Only 108 matches a season involve a promoted club, so do not expect a points gain to show.

## Side question: Elo look-ahead in backtests

Yes. Every backtest step reads the season's root `teams.csv` Elo, and `asof_scope()` does not shadow it.

- `asof_scope()` (`backtest.py:168-272`) shadows `fact_match`, `fact_player_match_stats`, `fact_player_season_stats`, `player_alias` and, optionally, `evidence_claims`. `fetch_current_elo()` reads the raw `teams.csv` table through `reconcile._season_root_table()`, which is untouched.
- Replaying 2025-26 GW1 with the real code gives `elo_at_calibration` of 1797 for Leeds, 1666 for Burnley and 1736 for Sunderland. Those are the root-file values.
- The root file is end-of-season. It matches each club's last per-match Elo in `matches.csv`, not its first:

| Club | Per-match Elo, GW1 | Per-match Elo, last | Root `teams.csv` | Look-ahead |
| --- | --- | --- | --- | --- |
| Burnley | 1729.60 | 1666.34 | 1666 | -64 |
| Leeds | 1722.37 | 1797.39 | 1797 | +75 |
| Sunderland | 1547.12 | 1735.93 | 1736 | +189 |

- At the GW1 step the regression slopes were 0.00072 per point for attack and 0.00152 for defence. Sunderland's 189 points were worth +0.14 attack and +0.29 defence: about 15% more goals scored and 25% fewer conceded than the information available then supported. At GW1 a promoted club's rating is its Elo prior alone, so that is the full error.
- Promoted clubs therefore always had a prior, so the league-average fallback never ran in a 2025-26 backtest. The backtest cannot reproduce the live gap.
- `By Gameweek/GWn/teams.csv` is not a clean substitute. GW1 to GW25 all carry one backfilled value (Burnley 1685).
- The per-match Elo in `fact_match` is the point-in-time source: 380 of 380 rows in 2025-26 and 371 of 380 in 2024-25. The existing `fact_match` shadow already cuts it at the deadline.
- Measured size: in my match-score replay, end-of-season Elo beat point-in-time Elo by 0.015 log-likelihood per match over 380 matches (t = 2.5), and by 0.030 on the 108 with a promoted club. Small, but it flatters every 2025-26 backtest.

## Method and caveats

Analysis only. Nothing in the repository was changed, pushed or opened as a PR.

- **Code and data.** FPL-Analyser `master` at `6d23661` (2026-10-05, contains `9a78c65`). FPL-Core-Insights at `69cd7ef` (2026-10-05), read as CSV only.
- **Run.** `scripts/run_ingestion.py` unmodified, on Python 3.13 rather than the workflow's 3.11. Variants were written to a copy of `db/fpl_quant_v2.duckdb`.
- **Evidence workbook not fetched.** This session's network policy blocks `drive.google.com`. I ran with an empty stand-in workbook that has the right tabs; the committed research pull was ingested as normal. Team strength does not read the workbook.
- **Baseline check.** Local xP for GW6 to GW10 matches the published `data/dashboard/projections_latest.json` within 0.05 points on 99.6% of 3,335 player-gameweeks. Alex Scott, Ezri Konsa and Max Dowman differ, presumably on workbook evidence.
- **Actions log not read.** GitHub API access is not enabled for this session, so I could not open run 37201797073. The local run printed the same two warnings.
- **Recommended numbers are a hand-built snapshot.** I took the three-season fit and applied the blend outside the code. The constants are untuned.
- **The replay is a scratch harness.** It scores match scores, not FPL points, over 2025-26 and five gameweeks of 2026-27. Its "live today" arm has one prior season, not two, and its "recommended" arm gives promoted clubs no Elo, as in live.
- **Not checked.** ClubElo's API (unreachable from here). Anything about the minutes model: it does not read team strength, contrary to the brief.

## Implementation (2026-10-05; live 2026-10-06)

Built behind `team_strength_guard_params` (#235). Live since 2026-10-06:
- `run_ingestion.py` fits `fit_seasons_for(TARGET_SEASON)`, so 2026-27's own matches count.
- v1, the `fix-withheld` arm, joins `RECALIBRATABLE_VERSION_ARGS`. It is the default for live
  ingestion, the forward season sim, the walk-forward and the season simulations.
- `--team-strength off` runs the model from before 2026-10-06 for comparisons.
- Backtest baselines are point-in-time now, so they sit about 1.8 points a gameweek below
  earlier runs.

- **Point-in-time Elo** (`fetch_point_in_time_elo`): each club's Elo from its latest finished
  match before the deadline, from `fact_match`. The target gameweek's own fixtures stay
  unfinished under `asof_scope()`, so their Elo is not read.
- **The guard, v1 = the recommended change with a newcomer's Elo withheld**
  (`team_strength.GUARD_RECOMMENDED`, the `fix-withheld` arm; chosen after the results below):
  - a club with no match in an earlier fit season weights its own fit n / (n + 10), held
    within 1.0 of its prior first;
  - a newcomer with no Elo starts from the established clubs' mean −0.39 attack, −0.32 defence;
  - every club has a prior, so the league-average fallback no longer runs.
- **Season counting, one deviation.** "Don't count the in-progress season" would also drop the
  2025-26 backtest's established clubs from 2/3 to 1/3: they count 2025-26 from its first fixture.
  The guard instead caps the count at two seasons (`max_own_seasons`). Live keeps 2/3 with 2026-27
  in the fit, and backtests are unchanged for established clubs.
- **The clamp covers every first-season club.** That includes every club in 2024-25's
  one-season fit, the cold-start stress test.
- **Cold-start rule (added after the first arms).** A first-season club with fewer than 10
  matches stays out of the Elo regression and the centre. At 2024-25 GW2–10, when every club's
  fit is that thin, there is no regression: every club starts from the median club. The first
  run fitted the regression on runaway values, and GW3 still scored a near-impossible match.
  Live and 2025-26 are unaffected, because their regression clubs are all established.
- **Scoreboard cap.** `match_score_log_lik_mean` caps each rate at 15 goals
  (`MAX_PHYSICAL_LAMBDA`). A runaway 2024-25 fit reached 1e6, and that one match pulled its
  gameweek's mean to −100,000.
- **Stored values move together.** The fit centres attack on all clubs, so a runaway raw fit
  shifts every club's stored values alike (about +0.57 for one club at −13 in a 23-club fit).
  Lambdas depend only on differences, and priors are fitted in the same frame.
- **`fit_seasons_for()` is generic.** Any season fits its two predecessors and itself, so next
  summer's rollover needs no code change.
- **Scoreboard.** `match_score_log_lik_mean` per gameweek, plus `:promoted_match`; a
  `vs_promoted_team` segment next to `promoted_team`. `walkforward_summary.py` reports both under
  `promoted_clubs`.

**Arms** (`run_walkforward.py --team-strength ARM`, also `run_season_sim_arm.py`):
`honest` (the blend from before the guard, on point-in-time Elo: the baseline), `live-like`,
`fix`, `fix-withheld`. Since 2026-10-06 a run with no flag is `fix-withheld` (v1), and `off` is
the model from before, end-of-season Elo included.

### Results (2026-10-06)

Walk-forward from the #235 branch. Paired per gameweek against `honest`, the blend from before
the guard on point-in-time Elo, the new baseline. 2025-26 decides; 2024-25 is the cold-start
stress test.

| 2025-26 (n = 37) | squad pts | vs `honest` | match log-lik | vs `honest` | promoted matches |
|---|---|---|---|---|---|
| `off` (end-of-season Elo) | 61.59 | +1.78 ± 0.81 | −2.992 | +0.006 ± 0.013 | −3.46 |
| `honest` | 59.81 | | −2.998 | | −3.43 |
| `live-like` | 61.30 | +1.49 ± 1.55 | −2.895 | +0.107 ± 0.101 | −2.95 |
| `fix` | 60.14 | +0.32 ± 0.83 | −2.904 | +0.098 ± 0.100 | −2.96 |
| `fix-withheld` | 60.16 | +0.35 ± 0.67 | −2.904 | +0.098 ± 0.100 | −2.96 |

- **The pass mark holds.** Both fix arms remove the failure (GW2's match log-likelihood −3.16
  against `honest`'s −6.88, when the promoted clubs' one-match fits run off) and are no worse
  than `honest` on points.
- **The fix arms can't be told apart in 2025-26.** Promoted clubs' own-match Elo was live all
  season then, unlike 2026-27's feed, which stopped after GW2.
- **The Elo look-ahead flattered the backtests.** Point-in-time Elo cost 1.78 ± 0.81 points a
  gameweek in 2025-26 and 2.2 in 2024-25 (squad points 65.05 → 62.81). Every reported 2025-26
  real-average margin carried it.
- **2024-25** (match log-likelihood −4.05 for `honest`):
  - `fix` −3.20, +0.85 ± 0.68, with squad points level (62.78 against 62.81);
  - GW3's near-impossible match is gone (−3.5 against −27.9);
  - GW2 is still poor (−9.0 against −15.2): that step fits on 10 matches, and the unguarded home
    advantage is unstable there;
  - `fix-withheld` equals `fix` there, since no club in 2024-25 is promoted against an earlier season.
- **Season sim (2025-26, against `honest`):** `fix` −0.57 ± 1.52, `fix-withheld` −0.73 ± 1.43,
  within noise. The `honest` control is +8.1 against the real average, against +10.5 under
  end-of-season Elo.
- **The season sim does not repeat across CI jobs (checked after the switch).** In run
  37397424563, `control` and `--team-strength fix-withheld` had the same database and param versions.
  They still split at GW3, where one played Bench Boost and the other held it to GW6. That gave 1,059
  and 1,064 points over GW2–19, +0.28 ± 1.81 a gameweek, with weeks up to 17 points apart. Four
  such runs, on #235's first commit, #236's branch and master, gave only those two paths. The arm
  differences above are no larger than this noise floor. The cause is not confirmed. Wall time does
  not track the path, so it is not the solver's time limit; floating point that differs between
  runner CPUs is suspected.
