# EP model: the attacker/defender imbalance

**Status:** diagnosis + first fix shipped (Lead A). Lead B's measurement plan is unblocked
(PR #124's segments landed in `backtest_run_id=1`) and its recalibration wiring is shipped
(`claude/nightly-progress-model-points-cx2yds`). Update 2026-09-08: a `recalibrate.yml`
dispatch (run 34220621168) + `review_recalibration.yml` confirmations have now actually moved
`fact_type_multiplier_params.multiplier` (1.2 -> 1.0, v8) and `minutes_adjustment_params`'s
`magnitude`/`cap` pair (-4.0 -> -3.0 / 6.0 -> 6.0, both v18, the first time this pair was ever
confirmed at a SHARED version -- see `resolve_active_version()`'s own docstring on why the
earlier v8/v16 vs v9/v17 mismatch left it inert) live in `data/recalibration/seeds_1.json`,
alongside the already-confirmed `kappa_tc` (v3), `rho_residual` (v4), and
`minutes_model_shrinkage_params` (v11). Update 2026-09-09: `rate_shrinkage_params` -- the
constant this section's own Lead B write-up was actually about -- is now confirmed and live too
(v8, `k_minutes` 450.0 -> 900.0). Getting there required a real, separate fix first:
`nightly_backtest.yml` had been failing for 3+ consecutive days on a `squad_optimizer` SCIP
timeout, root-caused to a non-PSD cross-player Sigma bug in `uncertainty.py` (PR #163) --
without a fresh walk-forward DB, `rate_shrinkage_params` (added after the last successful
build) had no seeded v1 row to recalibrate against at all. Once PR #163 unblocked the walk-
forward, the grid search picked `k_minutes=900.0` (the top of the tested grid `{150, 250, 350,
450, 600, 900}`), a real but modest improvement in `ep_total_calibration_mae` (1.1457 ->
1.1454, ~0.03% relative) -- confirmed via `review_recalibration.yml`. Landing on the grid's own
upper boundary is worth flagging for a future round: it means the search never bracketed an
interior optimum, so a wider grid (testing values above 900) is worth trying to check whether
an even larger k does better still.

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

## Lead B — SHIPPED (recalibration wiring only, not a confirmed value yet)

The measurement plan below is now unblocked: `data/dashboard/app_track_record.json`
(`backtest_run_id=1`, generated 2026-09-05) carries #124's segment_calibration, and it confirms
the imbalance survives Lead A: `ep_total_calibration_mean_resid` is **-0.1033 for Defender,
+0.1512 for Forward, -0.2793 for Goalkeeper**, and by price band it's monotonic and much
starker -- **-0.24 at <£5.0m growing to +0.84 at £9.0m+** (as measured against `backtest_run_id=1`
before any M7 confirmation existed). Update 2026-09-08: five parameters are now confirmed and
live (`fact_type_multiplier_params`, `minutes_adjustment_params.magnitude`/`.cap`, `kappa_tc`,
`rho_residual`, `minutes_model_shrinkage_params` -- see Status above) -- `rate_shrinkage_params`,
the one this section's segment-calibration evidence actually motivated, is still unconfirmed.
The mean_resid numbers above have not been re-measured since; a fresh walk-forward is needed to
see whether the now-live confirmations moved them.

`RATE_SHRINKAGE_K_MINUTES` was flagged for M7 recalibration since its own introduction but was
never actually wired into any refit technique -- `recalibrate()`'s `MINUTES_PARAM_GRIDS` covers
`fact_type_multiplier_params`/`minutes_model_shrinkage_params`/`minutes_adjustment_params`, none
of which is this constant. Closed by:

- `rate_shrinkage_params`/`k_minutes` is now a real versioned param (`expected_points.seed_v1_params()`,
  v1 = 450.0, byte-identical to the old hardcoded constant) with a new optional
  `rate_shrinkage_params_version` argument threaded through `player_rates_shrunk()`,
  `_defensive_action_rates_per_90()`, `compute_player_fixture_components()`, and
  `expected_points.run()` -- `None` (every existing caller) preserves the exact old behavior.
- `backtest.refit_rate_shrinkage()`: a grid search over candidate k values, minimizing mean
  `ep_total_calibration_mae` (the same metric segment_calibration already tracks) across the
  walk-forward's eval_steps -- re-runs `expected_points.run()` per candidate per gameweek (no
  SCIP, but a real per-fixture loop, so opt-in via `refit_rate_shrinkage_flag`/
  `current_rate_shrinkage_version`, same shape as `refit_kappa_tc_flag`). Wired into
  `recalibrate()`, `RECALIBRATABLE_VERSION_ARGS`, `run_backtest.py`, and a new
  `--stage rate_shrinkage` in `run_recalibrate.py` / `recalibrate.yml`.
- The one live (non-backtest) production call site, `scripts/run_ingestion.py`'s
  `expected_points.run()` call, now passes `ACTIVE["rate_shrinkage_params_version"]` -- so a
  future confirmed recalibration actually takes effect live, closing the exact drift
  `resolve_active_version()`'s own docstring warns about. `transfer_planner.compute_horizon_ep()`
  also accepts the new argument (opt-in, default `None`).
- **[CLOSED]** The scoped follow-up above -- `compute_horizon_ep()`'s other real callers not yet
  passing `rate_shrinkage_params_version` -- is done: `compute_shared_horizon.py`, `grade_squad.py`,
  `print_chip_timing_roadmap.py`, `run_transfer_planner_for_real_squad.py`'s ML-lane helper, and
  `export_projections.py` (via `projections.build_projections()`, which needed the same optional
  kwarg threaded one level deeper) all now pass it through `active["rate_shrinkage_params_version"]`.
  `run_scenarios.py`/`explain_my_move.py` needed care since their broader `_param_versions()` dict
  is also unpacked into `decision_engine.recommend_best_move()`, which has no such argument --
  passed directly to their own `compute_horizon_ep()` call instead. Still a pure no-op today (see
  below), but the threading gap that made this session's live `kappa_tc`/`minutes_model_shrinkage`
  recalibration bugs possible (PRs #154-157) can no longer repeat itself for `rate_shrinkage`.

**[CLOSED] 2026-09-09:** `rate_shrinkage_params.k_minutes` is confirmed and live (v8, 450.0 ->
900.0) -- see the Status section above for the full story (it took fixing a separate,
independently-discovered 3-day `nightly_backtest.yml` outage first, PR #163). Next actual step:
re-measure `ep_total_calibration_mean_resid` by price band against a fresh walk-forward to see
whether this -- combined with the other confirmations landed the same week -- actually moved
the premium/cheap imbalance the original diagnosis found; and consider widening the
`k_minutes` grid above 900 (the search hit that value at the edge of the tested range, so a
still-larger k was never ruled out).

## Lead B (original) — needs #124's backtest data: defensive-points magnitude

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
