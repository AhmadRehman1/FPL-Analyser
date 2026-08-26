# Existing Model Audit — FPL-Analyser Quant System

This audit was produced by reading the repository end-to-end before writing any ML code,
exactly as Phase 0 section 3 requires. Its purpose is to identify what already exists so the
ML research layer can reuse it rather than duplicate it, and to pin down the exact seam where
residual modelling will plug in.

The system is a DuckDB-backed, schema-migrated, walk-forward backtested quantitative FPL
prediction + optimisation stack. Modules are labelled M0–M8 in the source. Every module's
parameters resolve through a single `param_versions` mechanism (`params.py`), and every
prediction is pinned to a `calibration_asof_date` so a backtest can never see data that was
not yet knowable at the point it simulates a decision.

---

## 1. What the existing model predicts

`expected_points.run()` (M3) produces one row per **player × fixture** in `ep_outputs`, with a
total expected-points figure `ep_total` plus its component sub-expectations:

- `ep_appearance`, `ep_goals`, `ep_assists`, `ep_clean_sheet`, `ep_goals_conceded`,
  `ep_defcon`, `ep_bonus`, `ep_saves`, `ep_penalty_save`, `ep_cards`, `ep_own_goal`,
  `ep_total`, `expected_bps`.

`ep_total` is the Quant point prediction **Q(x)** for a player in a given fixture. Every
category is its own sub-model, all conditioned on the M2 minutes distribution; category
expectations are summed for total EP (linearity of expectation holds regardless of
correlation between categories — correlation is M4's job, handled separately).

The prediction is per-fixture, not per-gameweek. A player can have 0 fixtures in a blank
gameweek or 1 in a normal one; DGW/multi-fixture handling is explicitly out of scope for v1
(see `expected_points.py` module docstring and `backtest.has_double_gameweek()`).

## 2. What inputs it uses

- **M1 Team Strength** (`team_strength.py`): Dixon-Coles bivariate-Poisson attack/defence
  model with Elo regression priors. Produces per-fixture `lambda_home` / `lambda_away`
  (expected goals for each side).
- **M2 Minutes Model** (`minutes_model.py`): three-state distribution
  `p_0min` / `p_1_59min` / `p_60plus_min` per player, with shrinkage toward position averages
  for small samples (the "2-minute cameo extrapolated to xG/90=3.6" bug is explicitly guarded
  against via `_shrink_rate()`).
- **M3 Expected Points** (`expected_points.py`): per-90 rates pooled across a lookback
  window and shrunk toward position averages; Plackett-Luce bonus model over `expected_bps`;
  set-piece uplift multipliers for confirmed penalty/free-kick takers.
- **M4 Uncertainty** (`uncertainty.py`): Cornish-Fisher quantile bands (`quantile_05` /
  `quantile_95`) per player per fixture.
- **M5 Squad Optimiser** (`squad_optimizer.py`): SCIP MIQP selection of a 15-man squad.
- **M6 Monte Carlo** (`monte_carlo.py`): antithetic fixture simulation for an already-chosen
  squad's 15 players.
- **M7 Backtest** (`backtest.py`): walk-forward over 2024-25 + 2025-26 (~76 gameweeks),
  tiered cold/warm/mature, scoring against realised outcomes.
- **M8 Transfer Planner** (`transfer_planner.py`): multi-gameweek-horizon planning.
- **Evidence blend** (`evidence_blend.py` / `snapshot.py`): `data_asof` snapshot discipline
  — `snapshot.get_claims_asof()` is the look-ahead-safe evidence query.

Raw inputs live in three reconciled fact tables (see `schema/0001_core_schema.sql`):

- `fact_match` — fixtures (match_id, season, gameweek, kickoff_time, teams, scores, finished).
- `fact_player_match_stats` — per-match player stats (minutes_played, goals, assists, saves,
  defensive actions). PK `(player_uid, match_id)`.
- `fact_player_season_stats` — per-gameweek cumulative snapshot. PK `(player_uid, season, gw)`.
  Carries `now_cost`, `selected_by_percent`, `ep_next`, `chance_of_playing_next_round`,
  `status`, `minutes`, `goals_scored`, `assists`, `bps`, `expected_goals`,
  `expected_assists`, `expected_goals_per_90`, `expected_assists_per_90`,
  `defensive_contribution`, `defensive_contribution_per_90`, `saves_per_90`, `total_points`,
  `event_points`.

## 3. Which outputs can be recorded historically

- `ep_outputs` (Q(x)) — durable, written by every backtest step, keyed by `model_version`.
- `minutes_model_outputs`, `team_strength_snapshots`, `uncertainty_outputs` — durable.
- `backtest_gameweek_steps` — links each `(season, gameweek)` walk-forward step to the exact
  `ts/mm/ep/un/so/mc` model versions it produced, plus `data_asof` and tier.
- `backtest_metrics` — per-step realised-vs-predicted metric rows.
- `fact_player_match_stats` / `fact_player_season_stats` — the realised outcomes.

## 4. Which outputs are currently unavailable historically

- No `db/fpl_quant_v2.duckdb` file ships with the repository (gitignored), and no raw CSV/XLSX
  source data is committed. The clone is empty of ingested data — the engine is only runnable
  after `scripts/run_ingestion.py` has populated the DB. This Phase-0 engine is therefore
  designed to be **provable via synthetic test fixtures** (see `tests/test_ml_*.py`) and to run
  for real once ingestion has been performed locally.
- `fact_player_season_stats` carries xG/xA columns but the M3 BPS formula deliberately omits
  passing/crossing/key-pass/foul granularity (never reconciled into `fact_reconciled`) — noted
  as a scope limitation, not silently approximated.
- No per-player ICT (influence/creativity/threat) is stored; `expected_bps` is the only
  bonus-relevant signal available.

## 5. What constitutes one prediction observation

For Phase 0 the fundamental observation is **player × gameweek** (not player × fixture):

- `player_uid`, `season`, `gameweek`, `team_uid`, `position`
- `quant_prediction` = `ep_total` (aggregated to player×gw where a player has one fixture;
  DGW players are skipped — see leakage protocol §3)
- `actual_points` = `fact_player_season_stats.event_points` for that `(player, season, gw)`
- `residual` = `actual_points − quant_prediction`
- `prediction_timestamp` = the step's `data_asof` (= `gameweek_deadline`)
- `prediction_horizon` = 1 gameweek

The player×fixture granularity of `ep_outputs` is reduced to player×gameweek by the dataset
builder (§6 of the spec requires player×gw as the preferred initial observation).

## 6. What the existing model's natural prediction horizon is

One gameweek. `expected_points.run()` is called once per `(target_season, target_gameweek)`
inside `backtest.run_gameweek_step()`. M8's `transfer_planner.compute_horizon_ep()` extends
forward over a multi-gameweek horizon but reuses the same per-gameweek EP machinery. Phase 0
targets the single-gameweek horizon.

## 7. Which components are deterministic

- M1 Dixon-Coles MLE fit, M3 category expectations, M5 SCIP MIQP solve, the Plackett-Luce
  bonus expectation, the Cornish-Fisher quantile bands — all deterministic given the same
  inputs, param versions, and data_asof. SCIP tie-breaking is pinned via `requirements.lock`
  (see its own comment on solver nondeterminism).
- `param_versions` are explicit-version-only (`resolve_param` never activates a version), so
  a recomputed prediction is byte-identical to the original as long as the same version
  numbers are passed.

## 8. Which components are stochastic

- M6 Monte Carlo (`monte_carlo.run()`) is the only genuinely stochastic component — antithetic
  sampling with a deterministic seed derived from `(model_version, calibration_asof_date,
  query_id)` (`monte_carlo.deterministic_seed()`). Phase 0 does **not** use Monte Carlo; it
  works with the analytic `ep_total` point prediction only.

## 9. Which components could be used as ML features

Approved pre-prediction feature sources (all constructed strictly before the target GW's
deadline — see leakage protocol):

- Player rolling stats from `fact_player_match_stats` (goals, assists, minutes, starts,
  defensive actions) over 3/5/10-match windows.
- Player rolling per-90 rates from `fact_player_season_stats` at `gw < target`
  (`expected_goals_per_90`, `expected_assists_per_90`, `defensive_contribution_per_90`,
  `saves_per_90`, `bps`).
- `now_cost`, `selected_by_percent`, `chance_of_playing_next_round`, `status`, `position` —
  snapshotted at the most recent `gw < target` (price/ownership/news known before deadline).
- Team rolling xG/xGA proxies from `fact_match` scores and `fact_player_match_stats`
  aggregated by team over a preceding window.
- Fixture context: opponent, home/away, fixture difficulty (from M1 lambdas — but see §10),
  fixture congestion (matches in preceding 7/14 days).
- M2 minutes probabilities (`p_start_final`, `p_60plus_min`) for the target GW — these are
  themselves a Quant model output made asof the deadline, usable as a feature because they
  are a prediction, not a realised outcome.

## 10. Which components must NOT be used because of leakage risk

- The target gameweek's own `event_points`, `total_points`, `minutes`, `goals_scored`,
  `assists`, `bps`, `expected_goals`, `expected_assists` in `fact_player_season_stats` at
  `gw == target` — these encode the realised outcome being predicted. Only `event_points` is
  used, and only as the **label**, never as a feature.
- The target gameweek's `fact_player_match_stats` rows (the match being predicted).
- End-of-season aggregates, future fixtures beyond the schedule-only horizon, future prices,
  future ownership, future injury/transfer/team-news, future `status`.
- `fact_player_season_stats.total_points` is **cumulative-to-date**: at `gw == target` it
  includes the target GW's points — never use it at `gw == target`. At `gw < target` it is a
  valid running total, but per-gameweek deltas must be derived as differences of consecutive
  snapshots, never by reading `event_points` at `gw == target`.
- M1 lambdas for the target fixture are a Quant prediction, not a feature — using them as an
  ML feature would make the residual model a trivial identity of the Quant model. M2 minutes
  probabilities are permitted because they are an *input* the residual model can correct, not
  the output it is correcting; M1 lambdas are excluded to avoid this circularity.

---

## Audit conclusion

The repository already contains: a versioned, asof-safe prediction engine producing Q(x)
(`ep_total`), a walk-forward backtest with `data_asof` discipline (`asof_scope`), realised
outcomes (`event_points`), and a strong pytest culture with synthetic in-memory DuckDB
fixtures. **None of this needs to be rebuilt.** The Phase-0 ML layer's only job is to:

1. Extract `(Q(x), y)` pairs from the existing backtest steps asof-safely.
2. Build leakage-free rolling features from the same shadowed fact tables.
3. Learn a residual correction `ML(x) ≈ r = y − Q(x)` under chronological walk-forward
   validation, and measure whether `Q(x) + ML(x)` beats `Q(x)`.

The seam is `backtest_gameweek_steps.ep_model_version → ep_outputs.ep_total →
fact_player_season_stats.event_points`, scoped inside `backtest.asof_scope()`.
