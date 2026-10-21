# Plan: Projection track (why 2026-27's projections miss)

One-line goal: the projections' 2026-27 error shrinks toward the backtest's, and the model's picks
away from the template stop losing, judged on both backtest seasons and on 2026-27 itself.

Opened by `docs/plans/2026-10_live_decision_layer.md` Phase 5. Results go in
`docs/reports/2026-10_beat_the_template.md`.

## Classification
Track: Performance, on the projections. Every decision here was taken by its recommended option
(owner's standing instruction, 2026-10-21).

## Current State
- **2026-27 GW2–7 walk-forward, control** (run 39626660473): EP MAE 1.363, against 1.045 and
  1.051 in the backtest seasons. Bias −0.23 per player against −0.07.
- **2026-27 realized − projected per player-GW, by price band** (same run, verified:
  `walkforward_summary.json` `price_band`):

  | band | total | goals | assists | appearance | other |
  |---|---|---|---|---|---|
  | <£5.0m | −0.29 | −0.01 | −0.01 | −0.14 | −0.11 |
  | £5.0–7.0m | −0.25 | −0.08 | −0.01 | −0.04 | −0.11 |
  | £7.0–9.0m | +0.48 | +0.02 | +0.23 | +0.02 | +0.17 |
  | **£9.0m+** | **+1.38** | **+1.00** | −0.17 | +0.25 | +0.34 |

  The backtest control has £9.0m+ at +0.55, with goals +0.14 (run 39595787814).
- **By position, 2026-27:** GK −0.53, DEF −0.21, MID −0.22, FWD −0.08.
- **Reading:** this season the model under-projects premium goals by a full point per
  player-GW, and over-projects keepers and cheap players. The template owns the premiums, so it
  wins, and the anchor helps (live decision layer, Phase 5).
- **Earlier evidence:** `--k-minutes 450` gained +0.89 ± 0.49 pts/GW, with no MAE gain, under the
  older rule (verified: `docs/reports/2026-10_rate_prior.md`). Recency with decay
  (`--rate-current-season-weight 2 --rate-season-decay 0.5`) failed (verified:
  `docs/reports/2026-10_breakout_players.md`).

## Rule (fixed before any run)
An arm promotes only if it passes all three:
1. R6 on both backtest seasons against a fresh control: m − SE ≥ 0, and EP MAE within +0.01 per
   season.
2. On 2026-27's settled GWs against a fresh 2026-27 control, both `--template-lag 2`: squad
   points m ≥ 0 and EP MAE no worse.
3. The live switch goes through `active_recalibratable_versions()`, with a wiring test.

If several arms pass, the highest backtest m wins.

## Batch R1 (declared 2026-10-21)
- **Two harnesses, three runs each:**
  - backtest: `--template-lag 2` plus each arm's args;
  - 2026-27: `--seasons 2026-2027 --template-lag 2` plus each arm's args.
- **Arms:**
  - control: no extra flag;
  - k450: `--k-minutes 450`, which shrinks own rates less, so premiums stay closer to their own
    record;
  - csw2: `--rate-current-season-weight 2`, which makes this season's minutes and returns count
    double, with no decay change.
- **Why these:** both move premium goal rates toward the player's own recent record, which is
  where the 2026-27 miss sits. Both are existing flags, so there is no new code.

## Build Phases
- [ ] Phase 1: Run and judge R1 (above).
- [ ] Phase 2: Keeper over-projection. Split GK −0.53 into saves, clean sheets and appearance on
      2026-27, then declare an arm or record why there is none.
- [ ] Phase 3: Data-feed check for 2026-27 (the match-feed lag found in PR #261). Compare the
      cached DB's 2026-27 player-match minutes and goals with FPL's event-live totals for GW2–7.
- [ ] Phase 4: Re-judge the template anchor at GW13 (live decision layer plan).
