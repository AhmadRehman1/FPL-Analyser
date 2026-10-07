# Scoring-rate prior: the price anchor goes live

Oct 7, 2026

## Summary

Four walk-forward arms tested how goal and assist rates are shrunk (#243). The price anchor passed the promotion rule in both seasons and is live from today as `rate_prior_params` v1. The other knobs stay opt-in.

- **Problem.** Early in 2026-27 the model ranked the season's template picks 100th to 500th. A player new to the league has a few hundred minutes of his own rate, which is shrunk toward the flat position average. Every premium is shrunk toward that same average, and the walk-forward under-predicted £9.0m+ players by 0.54 points a gameweek.
- **Fix.** With the anchor on, goal and assist rates shrink toward a per-position rate that rises with price (`expected_points.price_anchor_fits()`), not toward the position average. Saves keep the average.
- **Result.** In 2025-26 the anchor cut the EP error (MAE 1.093 → 1.083) and nearly removed the premium bias (+0.54 → +0.07). Squad points against FPL's real average were level or better: +0.59 ± 0.71 a gameweek against control. Captain points rose from 5.84 to 6.28 a gameweek.
- **Not promoted.**
  - The recency weights gave no calibration gain, and their points were noisy.
  - k_minutes 450 had the steadiest points gain (+0.89 ± 0.49), but no MAE gain. It would also reverse the 2026-09-09 recalibration (450 → 900).
  - "All three" scored no better than the anchor alone on calibration, and its points were twice as noisy.

## Arms

Each run walks 2024-25 GW1 to 2025-26 GW38 from the same cached DB, with only the flag changed. The runs are `branch_walkforward.yml` 37695517276 (control), 37695520412 (`--k-minutes 450`), 37695524298 (`--rate-price-anchor`), 37695527687 (`--rate-current-season-weight 2 --rate-season-decay 0.5`) and 37695531269 (all three).

### 2025-26: points against FPL's real average, and EP calibration

"Paired" is the per-gameweek difference in squad points against control, with its standard error. W/L counts the gameweeks the arm scored more or fewer points than control; the rest were ties.

| arm | squad pts/GW | vs real avg | paired vs control | W/L | EP MAE | mean resid |
| --- | --- | --- | --- | --- | --- | --- |
| control | 60.03 | +10.27 | — | — | 1.0934 | −0.118 |
| k 450 | 60.92 | +11.16 | +0.89 ± 0.49 | 8/5 | 1.0934 | −0.120 |
| **price anchor** | 60.62 | +10.86 | +0.59 ± 0.71 | 11/6 | **1.0825** | −0.103 |
| recency | 60.65 | +10.89 | +0.62 ± 1.23 | 18/12 | 1.0922 | −0.115 |
| all three | 61.00 | +11.24 | +0.97 ± 1.25 | 15/13 | 1.0824 | −0.104 |

### 2024-25: the cold start, reported separately

This season has no real FPL average, so only squad points and calibration are shown.

| arm | squad pts/GW | vs control | EP MAE |
| --- | --- | --- | --- |
| control | 62.78 | — | 1.0561 |
| k 450 | 63.92 | +1.14 | 1.0548 |
| **price anchor** | 63.00 | +0.22 | **1.0463** |
| recency | 63.30 | +0.51 | 1.0547 |
| all three | 62.95 | +0.16 | 1.0461 |

### Where the error moved (both seasons, realized − predicted)

| arm | <£5.0m | £5.0–7.0m | £7.0–9.0m | £9.0m+ | captain pts/GW |
| --- | --- | --- | --- | --- | --- |
| control | −0.18 | +0.05 | +0.37 | +0.54 | 5.84 |
| k 450 | −0.18 | +0.05 | +0.33 | +0.40 | 5.99 |
| **price anchor** | −0.16 | +0.06 | +0.25 | **+0.07** | **6.28** |
| recency | −0.18 | +0.06 | +0.35 | +0.48 | 6.08 |
| all three | −0.16 | +0.06 | +0.25 | +0.11 | 6.28 |

The minutes metrics (log score, Brier) and the match-score likelihood are identical across arms. That is expected, since the rate prior touches only goal and assist rates.

The walk-forward's "beats crowd" number is left out of the comparison on purpose. Its synthetic crowd is built from each arm's own EP (`_avg_manager_benchmark_points()` takes the arm's `ep_model_version`), so it moves with the arm. FPL's real average does not.

## What changed

- `rate_prior_params` v1 = `{price_anchor: 1, current_season_weight: 1, season_decay: 1}` (`expected_points.LIVE_RATE_PRIOR`).
  - `run_ingestion.py` and `materialize_confirmed_seeds()` seed it, so a cached DB from before today has it too.
  - It joins `RECALIBRATABLE_VERSION_ARGS`, so `active_recalibratable_versions()` hands it to every caller that builds EP: live ingestion, the shared horizon, the planners and real-squad scripts, the scenario and explain-my-move tools, projections, the forward season sim, the walk-forward, the season simulations and their baselines.
  - `tests/test_live_switch_wiring.py` now requires it at every such call.
- The k_minutes refit (`_ep_calibration_mae_for_step()`) scores each candidate k with the rate prior its step ran with, because k shrinks toward that anchor.
- `research/ml/forward.py` copies the whole EP recipe into an ML horizon version, so a replay describes the same model. The ML shadow's frozen config is unaffected: it reads whatever `ep_model_version` live produced.
- `run_walkforward.py`:
  - `--no-rate-price-anchor` runs the old model.
  - `--rate-current-season-weight` and `--rate-season-decay` now change the live bundle.

## Next

- **Anchor + k 450.** k 450 had the steadiest points gain of the four arms (positive in both seasons). On top of the anchor it may add points without giving up calibration, so it is worth one more arm against the new live control.
- **Role changes.** Evidence that a player has won a role (a `RoleChange` claim) is logged but has no effect on his minutes. Several of this season's template picks are new starters, so that gap still under-ranks them after this change.
