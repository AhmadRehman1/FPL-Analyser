# Beat the template, then push toward a top-100k season

Oct 12, 2026 · in progress: Phase 0 done; Phase 1 (the oracle yardstick, rule R6, the residual
export) being built. Plan: `docs/plans/2026-10_beat_the_template.md`.

Work runs on branch `claude/new-session-11j6no`. The plan names `claude/beat-template`, but this
session is bound to its own branch. Dispatches use that branch's ref.

## Rule R6 (agreed with the owner, before any result)

An arm is judged against its batch's control with
`compare_arms.py control.json arm.json --rule r6`. Both runs walk both seasons (no `--seasons`).
- **(i) Points.** Paired per-gameweek `squad_points` over every GW of every season: mean m and
  SE = sd / √n. Passes when m − SE ≥ 0.
- **(ii) EP MAE.** Each season's EP MAE ≤ control's + 0.01.
- **(iii) Season sim.** Before going live, `season_sim_arms.yml` on 2025-26 (GW2–38, control,
  control_b and the arm on one runner) must give m ≥ 0 on paired net points (Req 7).

The tool refuses runs on different cached DBs, over different seasons, an incomplete run, or any
season with fewer than 2 paired GWs.

Plan additions, which the owner may drop:
- **Req 9, the template-collapse guard.** Promotion is blocked if mean XI overlap with the gated
  template rises by ≥ 1 player/GW while the paired change in `non_template_points` is ≤ 0.
  Mean overlap ≥ 9/11 pauses the programme for owner review.
- **Phase 10's planner rule.** m − SE ≥ 0 on season-sim net points.

**Replicate rule.** When an arm's |m − SE| ≤ 0.1, the control and the arm are each re-dispatched
once on the same commit. The verdict is R6 on the per-GW mean of the two runs' differences, and
the MAE check uses the mean of the two runs' MAEs (`--replicate control_b.json arm_b.json`). The
replicate counts as one arm.

## Goals

- **G1 (fair yardsticks).** Every walk-forward summary reports, per GW, the model squad's
  realized points and a legal availability-gated ownership-template squad's.
  - The gate drops players whose latest row before G (gw ≤ G−1) has `status` 'i', 's', 'u' or
    'n'. A missing status never drops a player.
  - The template is a fresh hit-free squad each week, under the model's budget, formation, club
    cap, captain/vice/bench and auto-sub rules. The ungated template is reported alongside.
  - Live: the leaderboard gains a "model (oracle)" row and an "ownership template (available)"
    row (Req 3).
- **G2 (beat the template by enough).** On the two-season oracle walk-forward, the pooled paired
  gap m = model − gated template satisfies m ≥ M̄ and m − SE > 0. All terms in pts/GW.
  - Per season s: M_s = max(2.0, C_s + E + O⁺ − T_s).
    - C_s: season s's real top-100k cutoff pace (rank-100k total ÷ 38).
    - T_s: the gated template's mean squad points over season s's walk-forward GWs.
    - E: the evolving loss, oracle model − season-sim control, both 2025-26.
    - O⁺: the live optimism O (2026-27 leaderboard sim pace − live team pace, same GWs) when
      O > 2·SE(O), else 0.
  - M̄ is the GW-weighted mean of M_2024-25 and M_2025-26. Phase 3 freezes M and its inputs.
- **G3 (top-100k pace, backtest).** The evolving manager, one continuous GW2–38 squad on the
  2025-26 season sim, scores at least that season's top-100k cutoff pace (per GW over GW2–38).
- **G4 (live).** The live model team's cumulative net points reach the 2026-27 top-100k
  cutoff's cumulative points. Reviewed at GW19, judged at GW38.

## Declared hyperparameters (fixed before any run)

| lever | flag | values |
|---|---|---|
| Z | `--no-goal-anchor`, `--no-rate-price-anchor` | as on master |
| L1 crowd prior | `--crowd-prior` | ridge α 10 on standardized features; `min_rows` 2,000 per fold; clip ±1.5 pts/GW; features `own_pct`, `log1p(selected_by_percent)`, `xfer_pct` from row G−1; no band terms, no intercept |
| L2 band bias | `--band-bias` | additive mean residual per (position, price band), shrunk n/(n + 500) |
| L3 player residual | `--player-residual` | last 6 GWs, shrunk n/(n + 6), players with ≥ 900 league minutes this and last season |
| L4 optimizer's curse | `--selection-shrink` | κ ∈ {0.25, 0.5} (two arms) |
| L5 crowd captaincy | `--crowd-captain` | δ = 1.0 pt per 100% EO |

These are assumed defaults with no fitted basis. Fixing them first prevents tuning on results.

Coefficient folds go by the season scored or replayed: 2025-26 uses the 2024-25 fit, 2024-25 uses
the 2025-26 fit, and only 2026-27 uses the both-season fit.

## Caps and stop rule

- A lever family (Z, L1, L2, L3, L4 or L5, with its variants) is dropped after two of its arms
  in a row fail R6.
- Lever arms in Phases 4–9 count towards a cap of 10. Planner arms (Phase 10) have their own cap
  of 6. A replicate counts as one arm. Controls don't count.
- Batches have at most 3 arms plus control.
- The programme pauses for owner review when:
  - either cap is reached;
  - mean template overlap reaches ≥ 9/11;
  - the backlog is empty with G2 unmet;
  - the GW19 review shows G4 more than 100 points behind the cutoff.
- It ends at GW38 of 2026-27 with a final report, or earlier if G2, G3 and G4's GW19 review all
  hold and the owner closes it.

## Arm count

| | lever arms (cap 10) | planner arms (cap 6) |
|---|---|---|
| used | 0 | 0 |
