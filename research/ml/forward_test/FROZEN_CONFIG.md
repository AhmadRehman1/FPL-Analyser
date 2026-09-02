# Frozen forward test — ML shadow config

Per `research/ml/REPORT.md` §10b. The 70-fold walk-forward is now a **development set**. This
file pins the exact configuration of the forward (live) ML shadow. **Do not revise any value
here in response to a shadow-period outcome.** Iteration resumes only after the pre-declared
window closes and a promotion decision is made.

- **Frozen at:** 2026-09-02 (first commit of `research/ml/forward.py` + `scripts/compute_ml_shadow.py`)
- **Shadow window:** from the first `data/dashboard/ml_shadow.json` with `status: "ok"` through
  at least GW19 of 2026-27 (a meaningful fraction of the first half). Extend, never shorten.
- **What the shadow does:** writes `ep_ml = ep_quant + predicted_residual` per player for the
  upcoming gameweek to `ml_shadow.json`, for a side-by-side "does the ML view agree?" panel.
  **It feeds no recommendation.** Promotion to a decision input is a separate, human-gated step.

## Pinned configuration

| item | frozen value |
|---|---|
| Quant baseline `Q(x)` | `expected_points.run()` `ep_total`, whatever `ep_model_version` the live pipeline produced for the target gameweek (the shadow reads it, never re-runs it) |
| Residual target | `y − Q(x)` where `y` = `fact_player_season_stats.event_points` for the realised gameweek |
| Model | `LightGBMResidualModel`, `objective="huber"`, `alpha=4.0` (δ=4), `random_state=42` — REPORT.md §8c shipped arm |
| Tree params | `residual_model.LightGBMResidualModel` defaults as of the freeze commit (n_estimators / learning_rate / num_leaves / subsample / reg_alpha) — not overridden here |
| Ensemble | none — single Huber δ=4 arm. No q90 ceiling, no L2 blend in the shadow feed |
| Feature list | `feature_engineering.feature_columns()` as of the freeze commit — the exact ordered list, `position` categorical included |
| Preprocessing | `residual_model.Preprocessor` — train-median numeric impute, train-fixed one-hot categories, unseen category → all-zero row |
| Training set | **every** `backtest_gameweek_steps` row with an `ep_model_version` in the DB at shadow time (the nightly walk-forward's full history), DGWs excluded, leakage-checked |
| As-of protocol | training rows built inside `backtest.asof_scope` per step; the forward inference row built inside `backtest.asof_scope(con, target_season, target_gameweek)` — rolling features see only played gameweeks. `leakage_checks` aborts on any violation |
| Calibration | none applied to the shadow number (REPORT.md notes a −0.17 residual bias on Huber δ=4; left visible, not corrected, during the shadow) |
| Refit cadence | full refit on every nightly run (`nightly_backtest.yml`), on the then-current walk-forward history |
| Min training rows | `forward.MIN_TRAIN_ROWS` (200) — below it the shadow reports a placeholder, never a number |

## Shadow ledger

`data/dashboard/ml_shadow.json` is overwritten each nightly run (latest view). The immutable
per-deadline record is the git history of that file plus the committed
`data/dashboard/app_track_record.json` / `report_history/` snapshots that pin the Quant side.
A dedicated append-only `weekly_ml_shadow_history.csv` and a stateful manager sim are **not**
part of this first freeze — they are prerequisites for *promotion*, tracked in REPORT.md §10b,
not for the shadow display.

## Promotion gate (unchanged from REPORT.md §10b / §10a)

Before `ep_total_ml` may influence any recommendation:
- aggregate calibration + within-gameweek ranking + a **stateful** manager-sim policy backtest
  as primary metrics;
- safety-critical slices (nailed starters, premiums, optimiser picks) as blockers;
- forward-test points ≥ Quant with a paired per-gameweek block-bootstrap CI excluding zero.
