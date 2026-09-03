# EP model: the attacker/defender imbalance

**Status:** diagnosis + first fix (PR: `claude/ep-attacker-defender-imbalance`). The rest is
gated on the walk-forward backtest carrying PR #124's position/price calibration segments.

## The problem, and why it matters

The two strongest "does the model work" signals both say it currently loses to the average
human manager:

| Signal | Result |
|---|---|
| Retrospective validation (`docs/reports/2025-26_retrospective_validation.md`, blind v1 params, GW2-38) | **6.9th percentile** vs 2,000 random real managers; -371 pts vs their mean |
| Live model-managed team (`data/dashboard/app_model_team.json`, 2026-27) | **-38 pts vs field average after 2 GWs**; no premium attacker; captained a defender (Senesi) in GW2 |

The visible mechanism: the EP model gives cheap defenders/GKs a high floor (clean sheet +
DefCon ≈ 2 "free" points) while compressing the premium-attacker ceiling, so the MIQP builds
defensively-tilted squads with weak captaincy. In the GW3 production captain ranking, **Van
Dijk is #3** (4.67), ahead of every forward but Haaland.

`scripts/diagnose_ep_calibration.py --live` on the pre-fix DB:

```
segment                n   mean ep_total   top-10 mean
position=Forward       76           1.59          3.47
position=Defender     205           1.68          3.97
position=Midfielder   267           1.57          4.07
position=Goalkeeper    68           1.48          3.96
price_band=9.0+         8           3.13          3.13   <- the premiums, LOWER than...
price_band=<5.0       360           1.43          3.99   <- ...the best cheap players
```

The best forwards' predicted ceiling sat *below* the best cheap defenders'.

## Ruled out

- **Elite-finisher term (goals minus xG).** Checked in the real data: 2025-26 goals track xG
  closely for every over-performer (Haaland +1.5 over a season, Mbeumo -1.0, Watkins +0.6,
  Salah -1.2). There is no large, systematic finishing-skill effect to model here.

## Lead A — SHIPPED in this PR: 2024-25 attacking rates were silently dropped

`_player_rate_pool()` / `_position_average_rates()` read `fact_player_season_stats.minutes`
and `.expected_goals` (season totals). **2024-2025's `playerstats.csv` snapshot has neither
column** (`reconcile.build_fact_player_season_stats` documents this: "2024-2025's
playerstats.csv genuinely predates several columns 2025-2026+ has"). It *does* publish
`expected_goals_per_90` directly, and the real minutes exist at match grain in
`fact_player_match_stats` (11,567 rows).

So the old code required `minutes` and **silently skipped all of 2024-2025** for attacking
rates — while `_defensive_action_rates_per_90()` (reads `fact_player_match_stats`, which has
2024-25) kept using both seasons. **Attacking rates fit on one season, defensive rates on
two.** Fewer sample minutes → harder shrinkage toward the (low) position average → premiums
(high own rate, small sample) lose the most. A direct structural tilt toward defenders.

**Fix:** both functions now recover a snapshot-schema season via
`expected_goals_per_90 x (match-grain minutes)`. Effect on real rates:

| player | own xG/90 before (2025-26 only) | after (2-season) | why it moved |
|---|---|---|---|
| Isak | 0.336 (694 injury-truncated min) | 0.608 (3553 min) | 2024-25 Newcastle season restored |
| Salah | 0.345 (down year) | 0.538 | 2024-25 (24.7 xG) restored |
| Palmer | ~0.47 | 0.473 → less shrinkage (5231 vs ~1954 min) | sample size |
| Haaland | 0.777 | 0.750 | stable both seasons — barely moves |
| Van Dijk | 0.082 | 0.080 | defenders already had 2 seasons — unchanged |

Recomputed `ep_total` (GW3, local): Haaland 4.85→**5.35**, Isak 1.7→**4.44**, Palmer 3.8→**4.34**,
Watkins 3.84→**4.04**; Van Dijk 4.67→**4.48**. Premiums up, over-rated defenders slightly down.

This is **one contributing factor**, not the whole fix — measure it via the walk-forward
before merge (does `ep_total_calibration_mean_resid:position=Forward` move toward 0?).

## Lead B — needs #124's backtest data: defensive-points magnitude

`ep_clean_sheet` and `ep_defcon` are principled calcs (`exp(-lambda_against) * p_60plus * 4`
and `P(CBIT >= threshold) * 2`), but they rest on invented v1 params (`defcon_threshold` per
position; the CBIT rate shrinkage `RATE_SHRINKAGE_K_MINUTES = 450`) and on `team_strength`'s
`lambda_against`. Once the nightly walk-forward carries PR #124's segments, check:

- `log_score_clean_sheet_mean:position=Defender` — is CS probability over-confident?
- `ep_total_calibration_mean_resid:position=Defender` vs `:position=Forward` — the headline
  imbalance number.
- `poisson_calibration_mean_resid` by team tier — does `team_strength` over-rate mid-table
  defenses (which would inflate every CS)?

Candidate levers, each a versioned param, each backtest-gated:
1. `RATE_SHRINKAGE_K_MINUTES` — recalibrate (already flagged for M7). A lower `k` trusts a
   player's own rate sooner; helps de-compress the premium ceiling.
2. `defcon_threshold` — currently the real FPL rule values (10 DEF / 12 MID-FWD); leave unless
   the backtest shows the *rate model* feeding it is biased.
3. Clean-sheet: no free param — the fix would be in `team_strength` calibration, not here.

## Lead C — the DefCon/attacking asymmetry is now smaller but not zero

Even after Lead A, the diagnostic still shows FWD top-10 ceiling (~3.8) below MID (~4.2). Some
of that is real (forward is a shallow position). Confirm with the walk-forward whether the
residual is model bias or genuine.

## Measurement plan (once #124 + a nightly land)

```
PYTHONPATH=src python scripts/diagnose_ep_calibration.py           # reads max(backtest_run_id)
```
Then dispatch a walk-forward on this branch and diff `ep_total_calibration_mean_resid:*`
against master's nightly.
