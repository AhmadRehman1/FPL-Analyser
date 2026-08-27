# Leakage Protocol — Phase 0 Residual ML Research

> The single most important requirement of this phase. The model must NEVER use information
> that would not have been available at the time the prediction was supposedly made.

This protocol is **enforced programmatically**, not by convention. `leakage_checks.py` runs
assertions against the built dataset and is part of the experiment pipeline and the test
suite. A dataset that fails these checks cannot be used for training or evaluation.

---

## 1. The asof boundary

For a prediction of gameweek `G` in season `S`:

- **Knowable before the deadline** = any row whose effective timestamp is strictly before
  `gameweek_deadline(S, G)` (the earliest kickoff of GW G's fixtures, the repo's documented
  approximation of the FPL transfer deadline — see `backtest.gameweek_deadline()`).
- **The label** = `fact_player_season_stats.event_points` at `(player, S, G)`. This is the
  realised outcome. It is used ONLY as the prediction target, never as a feature, and is only
  read **after** the asof window closes (from `main.*`, not the shadowed temp tables).

The dataset builder reuses the existing `backtest.asof_scope(con, S, G)` context manager,
which shadows the three fact tables to pre-deadline rows. This is the same machinery the
Quant backtest itself relies on — the ML layer does not invent a parallel asof path.

## 2. Per-table rules

### `fact_match`
- Schedule (match_id, teams, kickoff_time) for the target GW and the schedule-only horizon
  `[G, G+horizon)` is knowable in advance and may be read for fixture context.
- Scores (`home_score`, `away_score`, `finished`) for the target GW are NOT knowable and must
  never enter features. `asof_scope` nulls them inside the window.
- A target fixture's `kickoff_time` is exactly the deadline; a strict `< deadline` cutoff
  would hide the fixture being predicted. The schedule exception is correct and is not a leak.

### `fact_player_match_stats`
- Rows for matches with `kickoff_time < deadline` are knowable → may be used for rolling
  player stats.
- Rows for the target GW's matches are NOT knowable → excluded by `asof_scope`'s
  `kickoff_time < deadline` filter.

### `fact_player_season_stats`
- PK is `(player_uid, season, gw)` — a per-gameweek cumulative snapshot.
- Rows at `gw < G` for the in-progress season are knowable asof the deadline → may be used.
- The row at `gw == G` is the realised snapshot → **only `event_points` may be read, and only
  as the label**, fetched from `main.*` outside `asof_scope`.
- `total_points` is cumulative-to-date: at `gw == G` it contains the target GW's points —
  forbidden. Per-gameweek deltas are derived as differences of consecutive `gw < G` snapshots.

## 3. Gameweek cardinality (DGW handling)

`ep_outputs` is player × fixture. The Phase-0 observation is player × gameweek. A player with
exactly one fixture in GW G maps 1:1 to one observation. The builder asserts one fixture per
player per GW; where a double gameweek is detected (`backtest.has_double_gameweek`), the GW is
**skipped** with a recorded reason — consistent with the existing M3/M7 v1 scope boundary, not
a new aggregation invented here.

## 4. Accepted feature sources

| Feature | Source | As-of rule |
|---|---|---|
| rolling goals/assists/minutes/starts/def-actions | `fact_player_match_stats` | `kickoff_time < deadline` |
| rolling per-90 xG/xA/defcon/saves/bps | `fact_player_season_stats` | `gw < G` |
| price, ownership, status, injury news | `fact_player_season_stats` | most-recent `gw < G` snapshot |
| position, team | `dim_player` / `dim_team` / aliases | static identity, knowable |
| opponent, home/away, fixture difficulty, congestion | `fact_match` schedule | schedule-only, `kickoff_time` unrestricted |
| minutes probabilities (M2) | `minutes_model_outputs` for the target GW's `mm_model_version` | a Quant prediction made asof the deadline — permitted |

## 5. Rejected feature sources (leakage)

| Feature | Why rejected |
|---|---|
| target GW `event_points` / `total_points` / `minutes` / `goals_scored` / `assists` / `bps` / xG / xA | the realised outcome being predicted |
| target GW `fact_player_match_stats` row | the match being predicted |
| end-of-season aggregates | future information |
| future fixtures beyond schedule horizon | future information |
| future prices / ownership / status / transfers / injuries | future information |
| M1 lambdas for the target fixture | a Quant output — using it makes the residual model a trivial identity of Q(x) |
| any future-derived aggregate | by definition |

## 6. Train/test discipline

- **No `train_test_split`.** FPL is a time-series. Chronological walk-forward validation only
  (`walk_forward.py`), splits aligned to existing `backtest_gameweek_steps`.
- Train = all steps whose season precedes the test season; test = one held-out season at a
  time. The latest season is always genuinely out-of-sample.
- Hyperparameters (incl. ensemble weights) are tuned using training data / time-aware
  validation only — never against the final test season.
- Target `event_points` is read for test rows only to compute metrics, never to fit anything.

## 7. Provenance

Every feature column documents its source table, the asof filter applied, and whether it is a
raw value, a delta, or a rolling aggregate. The experiment manifest
(`results/experiment_manifest.json`) records git commit, dataset version, train/test seasons,
feature list, model type, hyperparameters, and random seed — so any result is reproducible.

## 8. Programmatic checks (`leakage_checks.py`)

1. Each dataset row's `ep_model_version` resolves to a `backtest_gameweek_steps` row for the
   same `(season, gameweek)` — i.e. Q(x) was genuinely produced asof that step.
2. `prediction_timestamp` (= the step's `data_asof`) is strictly before the target GW's first
   kickoff — the prediction precedes the outcome.
3. No feature column is computed from rows at `gw >= target` (for season stats) or
   `kickoff_time >= deadline` (for match stats).
4. `event_points` appears in the dataset exactly once, as the label column, and is absent
   from the feature matrix.
5. No DGW step is present without an explicit skip record.
6. Chronological ordering: every training step's deadline precedes every test step's
   deadline within a walk-forward fold.

A dataset that fails any check raises `LeakageError` and the experiment aborts before any
model is trained.
