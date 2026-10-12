# Beat the template, then push toward a top-100k season

Oct 12, 2026 · in progress. Phases 0–2 are built and merged (PR #262, 2dff696). Phase 1's
control has run. Phase 3 is waiting on the A2/A3 diagnostic, the real cutoffs and a fresh
leaderboard. Plan: `docs/plans/2026-10_beat_the_template.md`.

**The headline so far.** Against a legal, availability-gated ownership template, the oracle
model is ahead by **+7.30 ± 1.93 pts/GW** pooled over 74 GWs (2024-25 +9.11 ± 2.65, 2025-26
+5.49 ± 2.81). The plan's starting premise, that the model "only roughly matches the template",
compared it with the synthetic EO crowd. That crowd is not a legal squad, and it sits far above
FPL's real average.

Work runs on branch `claude/new-session-11j6no`. The plan names `claude/beat-template`, but this
session is bound to its own branch. Dispatches use that branch's ref, or master after a merge.

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

## Phase 1: the oracle yardstick, rule R6 and the residual export (built)

- **The template (Req 1).** `backtest.template_squads()` builds `beats_baseline()`'s ownership
  squad inside each walk-forward step's own as-of scope, twice: with and without the
  availability gate. `score_gameweek()` scores both like the model squad (auto-subs, the solve's
  vice). `run_walkforward.py` always turns it on; the model's own numbers are untouched (a test
  runs the walk-forward with and without it and compares every model metric).
- **Diagnostics (Req 2).** Per GW, in `backtest_metrics` and the summary's `per_gameweek`:
  `template_points` (gated) and `template_points_ungated`, `template_xi_overlap`,
  `non_template_points`, the model's and the template's captain points, the `ep_plus_eo`
  captain counterfactual, £ per price band for both XIs, and realized − projected for both XIs
  under the model's EP. The summary's `template` block pools them, overall and by season.
  - `template_xi_overlap` compares the two XIs as picked.
  - `non_template_points` sums the model's scoring XI after auto-subs, with the armband, over
    players outside the template's picked XI.
  - Captain points are before doubling. EO for `ep_plus_eo` is the latest ownership before the
    deadline, spread as captaincy over the model's EP (`ownership.compute_eo_for_pool`).
- **Rule R6 (Req 5, Req 9).** `compare_arms.py --rule r6` returns one of:
  - REPLICATE (exit 3) whenever |m − SE| ≤ 0.1, before any other verdict;
  - FAIL on points or MAE;
  - BLOCKED when R6 passes but Req 9 blocks;
  - PAUSE when mean overlap is ≥ 9/11 in either run;
  - PASS.
  It refuses summaries without the per-GW template rows Req 9 reads. `--replicate c_b a_b`
  judges the two runs' mean differences and mean MAEs, and reports any squad-point difference
  between the two controls as measured nondeterminism.
- **Escape hatches.** `--no-template` skips the template (R6 then refuses the run), and
  `--template-lag 2` makes it read rows up to G−2, the fallback if A2 is inconclusive.
  - The summary records the lag (`template.lag`), and R6 refuses runs built with different lags.
  - `--export-residuals` reads its features from the same row as the template.
- **Runtime.** `template_solve_seconds` times building the pool and the two template solves; it
  doesn't include the scoring diagnostics, which are database reads.
- **Residual export (Req 12).** `run_walkforward.py --export-residuals` writes
  `residual_features.csv`; `branch_walkforward.yml` uploads it as `residual-features`. The
  features come from `residual_levers.feature_rows()`, the one reader every harness will share.
  - `price_band` uses the latest price before the deadline, so 2025-26 GW1 rows have none.
- **2025-26 GW1** has no row before its deadline, so no template (and no model squad).

## Phase 2: the live yardsticks (built)

- **Req 3.** `export_leaderboard.py` appends "model (oracle)" (`kind: model_oracle`, the
  walk-forward step on 2026-27 with the export's own live versions, no EO term) and "ownership
  template (available)" (`kind: template_gated`). Both are `is_baseline` rows after the
  existing three, so the evolving model stays the first non-baseline row. Every oracle row
  gains `oracle_model_beats_baseline_by`. The crowd row is renamed "synthetic EO crowd (not
  FPL's average)". The weekly job's limit goes from 120 to 150 minutes.
- **Req 4.** `rank_tracking.yml` runs `scripts/derive_rank_cutoff.py` after the autopsy. It
  fetches `entry/{id}/history` for the sampled entries and interpolates total points against
  overall rank (linear in log rank) at rank 100,000. It writes `top_100k_cutoff` into
  `rank_autopsy.json`: the 2024-25 and 2025-26 totals and paces, and the 2026-27 cumulative
  cutoff by GW, each with its bracket and the number of entries within 2× of the target.
- **Req 7.** `season_sim_arms.yml` gains `same_runner`: every arm runs one after another in one
  job (`scripts/run_season_sim_batch.py`), at most 5 runs.

## Runs dispatched (2026-10-12)

| run | workflow | ref / commit | args | purpose |
|---|---|---|---|---|
| 38228939128 | `branch_walkforward.yml` | branch, 9261170 | `--export-residuals` | Phase 1 control: template gaps, diagnostics, residual export |
| 38229024307 | `season_sim_arms.yml` | branch, 9261170 | `control: ; control_b:`, 2025-26, chunk `2-38`, `same_runner` | Phase 2: one-runner GW2–38 control and noise floor, chips used (A7) |

- **A4 holds.** Run 38228939128's `head_sha` is the branch head, 9261170: a dispatch on the
  branch ref runs the branch's code.
- **A5 holds.** The unit test `test_template_captain_is_the_most_owned_outfield_xi_player`
  checks that the template captain is the most-owned outfield XI player, never a goalkeeper,
  even one owned by more managers than anyone else.
- **Dispatched after the merge (master, 2dff696):**

| run | workflow | args | purpose |
|---|---|---|---|
| 38267689849 | `diagnose_crowd_columns.yml` | — | A2, A3 |
| 38267694488 | `rank_tracking.yml` | — | Req 4: `top_100k_cutoff` (A1) |
| 38267806962 | `weekly_backtest.yml` | — | Req 3's new rows, per-GW leaderboard rows (O) |
| 38267875032 | `branch_walkforward.yml` | `--export-residuals` | the current control |
| 38267878143 | `branch_walkforward.yml` | (none) | control_b, same DB: A9 |

## Phase 1 results: control run 38228939128 (cached DB duckdb-38189062194, 75/75 steps)

| per GW | pooled (74 GWs) | 2024-25 (37) | 2025-26 (37) |
|---|---|---|---|
| oracle model squad points | 61.64 | 63.41 | 59.87 |
| gated template | 54.34 | 54.30 | 54.38 |
| ungated template | 54.82 | 55.03 | 54.62 |
| **model − gated template (m ± SE)** | **+7.30 ± 1.93** | +9.11 ± 2.65 | +5.49 ± 2.81 |
| model − ungated template | +6.81 ± 1.98 | +8.38 ± 2.83 | +5.24 ± 2.79 |
| XI overlap with the gated template | 3.53 / 11 | 3.68 | 3.38 |
| points from outside the template XI | 36.2 | 36.7 | 35.7 |
| model captain pts (before doubling) | 6.42 | 6.95 | 5.89 |
| template captain pts (most-owned) | 5.89 | 6.87 | 4.92 |
| `ep_plus_eo` captain − model captain | +0.08 ± 0.20 | +0.11 ± 0.36 | +0.05 ± 0.16 |
| model XI realized − projected (per player) | −0.149 | −0.100 | −0.198 |
| template XI realized − projected (model EP) | +0.040 | −0.135 | +0.215 |

- **The gate made the template slightly worse, not better:** gated 54.34 against ungated
  54.82, −0.49 a GW. The gate reads the last row before the deadline, so some players flagged
  there were back by kickoff, and the next most-owned players who replaced them scored less.
  The difference is within noise. The gated template stays the yardstick, as declared: it is the
  squad a manager following the crowd could actually field.
- **The model differs from the template far more than "matches the template" suggests:**
  only 3.5 of its 11 starters are in the template XI, and 36 of its 62 points come from the other 7.5.
- **Spend by band (pooled XIs, £m):** model 5.6 / 27.6 / 21.0 / 28.5 and template
  3.8 / 28.6 / 22.6 / 27.7 (<5.0 / 5.0–7.0 / 7.0–9.0 / 9.0+). The two spread their budget
  much the same way.
- **The optimizer's curse is visible but small:** the model's XI scores 0.15 pts per player
  below its projection, while the template XI scored under the same EP is +0.04.
- **The template costs 58.5 s** of SCIP time over the run (two solves a step).
- **A3 (provisional, from the residual export):** ownership and transfer features cover 99.4%
  (2024-25) and 97.1% (2025-26) of projected player-GWs, so the transfer columns are populated
  in both seasons. The diagnostic run gives the formal count.

## Phase 2 results

**The one-runner season sim, run 38229024307** (2025-26, GW2–38, `control: ; control_b:`):

| arm | net pts/GW | FPL real avg | vs real avg | hits | transfers |
|---|---|---|---|---|---|
| control | 57.81 | 49.76 | +8.05 | 0 | 29 |
| control_b | 57.81 | 49.76 | +8.05 | 0 | 29 |

- **The noise floor on one runner is zero:** control_b − control is +0.00 ± 0.00 over 37 GWs,
  with identical chips. The ±1.81/GW split seen earlier was between runners. Req 7's
  `same_runner` removes it.
- **A7 holds.** One continuous GW2–38 squad played BB GW3, WC GW5, TC GW6 and FH GW8 in the
  first chip half, then FH GW20, BB GW21 and TC GW22 in the second. Each half's chips were played
  inside that half. The second wildcard went unused: the planner's threshold was never met.
  That is a planner choice for Phase 10, not a chip-accounting fault.
- **The evolving loss is small.** Paired over the same 37 GWs, the oracle model (59.87) beats
  the season sim (57.81) by E = +2.05 ± 2.07 pts/GW. That is below the plan's 3.0 trigger,
  so Phase 10 does not move earlier.

## A2 and A3: diagnostic run 38267689849 (cached DB duckdb-38263007316)

- **A3 holds.** `status`, `selected_by_percent`, `transfers_in_event`, `transfers_out_event` and
  `now_cost` are non-null on 100% of rows in every gameweek of 2024-25 (27,657 rows) and 2025-26
  (29,978 rows).
- **A2 is inconclusive under the declared rule, so the crowd features and the template gate move
  to row G−2.** In 2026-27, 4 finished GWs had a commit before the deadline:
  - GW2 and GW4: row G−1 matched that snapshot on all four judged columns;
  - GW3 and GW5: status matched for 98%, but `selected_by_percent` for only 76–78% and the
    transfer counts for 0.3–16%.
  - The same split shows in history: most weeks match 100%, while 2025-26 GW2–4, 9, 11, 24, 27
    and 30 and 2024-25 GW34 and 37 don't. 2024-25 has no pre-deadline commit before GW20.
- **What the mismatches can and can't tell us.** Row G−1's transfer counts accumulate until G's
  deadline. A snapshot taken hours before the deadline differs from the final row without any
  leak. But in those weeks the source also committed row G−1 again after the deadline, so the
  final row may carry some post-deadline information. The git history can't separate the two.
  As declared, the rule treats this as unknowable and uses G−2.
- **The consequence for the existing model.** The minutes model's FPL availability flag
  (`minutes_model.live_availability_by_player`) also reads the latest row before the deadline,
  i.e. row G−1. Status matched 97.5–100% in every judged week, so at most a few percent of flags
  could have moved after a deadline. Phase 3 measures what the flag is worth: its squad points
  at row G−1 against row G−2.

## Phase 3 measurement batch (declared 2026-10-12, before dispatch)

These runs measure named causes in the oracle → live chain. They are not levers and don't count
towards the caps. They run on this branch at the commit that adds the flags, back-to-back on one
cached DB, with the G−2 template:

| run | args | measures |
|---|---|---|
| control | `--template-lag 2 --export-residuals` | the G−2 template's T_s; residuals for L1/L2 (features from G−2) |
| availability | `--template-lag 2 --availability-lag 2` | what row G−1's availability flag is worth (an upper bound on any leak through it) |
| live EO anchor | `--template-lag 2 --live-eo-anchor` | the cost of the live-only EO template anchor (ownership v1, risk posture v2) |

How each result is used:
- **Availability.** If removing row G−1's flag costs more than 1.0 pt/GW with m + SE < 0, G3's
  backtest pace is haircut by the measured m before any lever runs. The cause can't be fixed
  live, where the flag is genuinely current.
- **EO anchor.** The paired m enters the chain table as the live anchor's contribution.
- **A9.** The master pair 38267875032 / 38267878143 (control and control_b, G−1 template) gives
  the across-runner control noise.

## Phase 3: reconcile, set M, decompose, declare the lever order (committed before any lever run)

**Inputs.**
- Control **38269659997** (`--template-lag 2 --export-residuals`, commit bbf68e9, cached DB
  duckdb-rank-tracking-38267694488, 75/75 steps). It runs the G−2 template that A2 requires.
  2024-25 GW2 and 2025-26 GW2 have no row G−2, so they have no template: 72 template GWs.
- Season sim **38229024307** (2025-26, GW2–38, one runner).
- Cutoffs from rank tracking **38267694488**: 487 entry histories.
- Leaderboard from **38267806962** (2026-27 GW2–5) and `app_model_team.json` (GW1–5 settled).
- Computed by `scripts/beat_template_margin.py`.

### The real top-100k cutoffs (Req 4)

| season | rank-100k total | pace / GW | bracket (ranks) | entries within 2× (better + worse) |
|---|---|---|---|---|
| 2024-25 | 2511.9 | 66.10 | 98,417 – 101,280 | 10 + 22 of 380 |
| 2025-26 | 2326.5 | 61.22 | 98,762 – 101,290 | 6 + 26 of 433 |
| 2026-27 after GW5 | 377.4 cumulative | 75.48 | 99,310 – 100,399 | 58 + 79 of 487 |

### The chain table (net pts/GW)

| row | season | GWs | pts/GW | vs FPL average | vs cutoff pace |
|---|---|---|---|---|---|
| oracle model | 2024-25 | 37 | 63.43 | — (no published average) | −2.67 |
| gated template (G−2) | 2024-25 | 36 | 53.06 | — | −13.05 |
| oracle model | 2025-26 | 37 | 60.11 | +10.35 | −1.12 |
| gated template (G−2) | 2025-26 | 36 | 52.53 | +2.81 | −8.70 |
| season sim control | 2025-26 | 37 | 57.81 | +8.05 | −3.41 |
| season sim control_b | 2025-26 | 37 | 57.81 | +8.05 | −3.41 |
| model (oracle), leaderboard | 2026-27 | 4 (GW2–5) | 63.00 | +0.75 | −12.48 |
| leaderboard sim (evolving) | 2026-27 | 4 (GW2–5) | 60.25 | −2.00 | −15.23 |
| live team | 2026-27 | 5 (GW1–5) | 60.40 | +0.60 | −15.08 |

2026-27's cutoff pace is high (75.5/GW) because early-season averages are high and the top
100k have played their chips well. It can't be compared with a full season's pace yet.

### Each named cause, measured (pts/GW)

| cause | where in the chain | measured | source |
|---|---|---|---|
| evolving loss: carrying a squad, hits, chip timing | oracle model → season sim (2025-26) | **E = +2.30 ± 2.12** | paired over 37 GWs, control vs season sim |
| availability flag read from row G−1 (possible post-deadline information) | inside the oracle backtest | **−0.47 ± 0.79** when read from G−2 instead (MAE +0.04 worse) | run 38269664839 vs control |
| the live-only EO template anchor (risk posture v2, κ 0.12) | live from-scratch solves only | **−2.96 ± 1.49** (overlap 3.25 → 7.40; non-template points −20.4) | run 38269669897 vs control |
| backtest sim vs live sim (live optimism O) | 2026-27 leaderboard sim → live team | **O = −1.75 ± 12.66** (4 GWs), so O⁺ = 0 | per-GW pairs |
| control noise across runners (A9) | any walk-forward comparison | **+0.135 ± 0.125**; 3 of 74 GWs differ | runs 38267875032 / 38267878143, same DB |
| season-sim noise on one runner | season-sim comparisons | **0.00 ± 0.00** | control vs control_b, run 38229024307 |
| the synthetic EO crowd is not FPL's average | the old yardstick | 325.4 vs FPL's 249 over GW2–5: **+19.1/GW above the real field** | leaderboard and `app_model_team.json` |

- **SE of the live sample:** the live team's GW1–5 gaps to FPL's average are +4, −19, +16, +4
  and −2: mean +0.6, **SE 5.7/GW**. After 5 GWs the live record says almost nothing about pace.
- **The EO anchor touches only from-scratch solves:** the GW1 seed and wildcard or free-hit
  weeks. Transfers come from the planner, which has no EO term. Its live cost is therefore at
  most its walk-forward cost in those weeks. Per the plan's Current State decision it stays as
  is. This is the largest measured cost in the chain, and the owner may want to revisit it.
- **No cause of backtest optimism was found:** the availability flag is worth less than the
  declared 1.0 pts/GW trigger, so **G3 is not haircut**.

### G2's margin M, by the declared formula

M_s = max(2.0, C_s + E + O⁺ − T_s), with E = +2.297 (both seasons) and O⁺ = 0:

| season | C_s | T_s | E | O⁺ | M_s | template GWs |
|---|---|---|---|---|---|---|
| 2024-25 | 66.103 | 53.056 | 2.297 | 0 | **15.344** | 36 |
| 2025-26 | 61.224 | 52.528 | 2.297 | 0 | **10.993** | 36 |

**M̄ = 13.169 pts/GW** (GW-weighted). M and its inputs are frozen here, and later controls are
judged against this M̄.

**G2 today:** the control's pooled gap to the gated template is m = **+9.19 ± 1.93** (72 GWs),
so G2 is **not met**. The model needs about +4.0 pts/GW more over the template. Equivalently,
the oracle model must reach C_s + E: 68.4 (2024-25) and 63.5 (2025-26) against today's 63.4
and 60.1.

**G3 today:** season sim 57.81 against 2025-26's cutoff pace of 61.22: **−3.41/GW, not met**.

**G4 today:** the live team has 302 points after GW5 against the cutoff's 377.4: **75.4 behind**.
The GW19 review pauses the programme if this exceeds 100.

### The decomposition (control, both seasons pooled, realized − projected per player-GW)

| band | total | appearance | goals | assists | clean sheet | other (bonus …) |
|---|---|---|---|---|---|---|
| <£5.0m | −0.131 | −0.056 | −0.020 | +0.003 | −0.037 | −0.021 |
| £5.0–7.0m | +0.025 | +0.026 | −0.011 | +0.028 | −0.006 | −0.013 |
| £7.0–9.0m | +0.244 | +0.009 | +0.033 | +0.083 | +0.006 | +0.113 |
| £9.0m+ | +0.170 | +0.008 | −0.177 | +0.168 | −0.011 | +0.181 |

By position: GK −0.076, DEF −0.115, MID −0.051, FWD +0.031. The model's own XI scores −0.11 per
player below its projection (2024-25 −0.06, 2025-26 −0.16). The template XI, scored on the same
EP, is +0.01 (2024-25 −0.15, 2025-26 +0.18).

### The branches (declared in the plan)

- **Backtest optimism:** none found (see above). No fix and no haircut.
- **Evolving loss ≥ 3.0?** E = 2.30, so no. Phase 10 does not move earlier.
- **Control already meets G2?** m = 9.19 < M̄ = 13.17, so no. The lever phases run.

### The lever queue (declared before any lever run)

1. **Z** (Phase 4): `--no-goal-anchor` and `--no-rate-price-anchor`, against a fresh G−2 control.
2. **L1**, the crowd prior (Phase 5). Features from row **G−2** (A2), fitted on this control's
   residual export.
3. **L2**, the band bias (Phase 6), fitted after L1 if L1 is promoted.
4. **L3** only if £9.0m+ realized − projected is still ≥ +0.1 after the levers that passed.
   Today +0.170, so provisionally queued; re-checked after L2.
5. **L4, not queued:** the model XI's residual is −0.11, short of the −0.3 trigger.
6. **L5, not queued:** `ep_plus_eo` − model captain is −0.07 ± 0.13 (trigger ≥ +0.3 with
   m − SE > 0).

Every lever run from here uses `--template-lag 2`, and R6 refuses mismatched lags.
