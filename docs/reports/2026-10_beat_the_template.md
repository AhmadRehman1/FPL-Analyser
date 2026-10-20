# Beat the template, then push toward a top-100k season

Oct 12, 2026 · in progress. Phases 0–2 are built and merged (PR #262, 2dff696). Phase 3 is done:
M̄ = 13.17 pts/GW and the lever queue, with the blind critique passed in round 2. Phase 4 (lever Z)
is declared and running.
Plan: `docs/plans/2026-10_beat_the_template.md`.

**The headline so far.** Against a legal, availability-gated ownership template, the oracle
model is ahead by **+9.19 ± 1.93 pts/GW** pooled over 72 GWs (the G−2 template A2 requires;
+7.30 ± 1.93 over 74 GWs with the G−1 template). The plan's starting premise, that the model
"only roughly matches the template", compared it with the synthetic EO crowd. That crowd is not
a legal squad, and it sits far above FPL's real average.
- That margin is still short of M̄ = 13.17, the gap that reaching top-100k pace in the backtest
  needs. In 2026-27's first 4 GWs the oracle model hasn't beaten the template at all
  (−3.5 ± 8.6; see Phase 3).

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
| used | 2 (Z: 2) | 0 |

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
- **The evolving loss is small.** Paired over the same 37 GWs, the Phase 1 control (59.87)
  beats the season sim (57.81) by E = +2.05 ± 2.07 pts/GW. That is below the plan's 3.0
  trigger, so Phase 10 does not move earlier. Phase 3 freezes E = +2.30 ± 2.12 from its own
  control; the two controls landed on different branches of the across-runner noise (A9, see
  Phase 3).

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
  i.e. row G−1. In 2026-27's judged weeks, status matched for 97.5–100% of players. History has
  worse weeks: 2025-26 GW2 matched only 64%, and GW3–4 93.8–94.7%. Phase 3 measures what the
  flag is worth: its squad points at row G−1 against row G−2.

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
- Season sim **38229024307** (2025-26, GW2–38, one runner). Its prepare job built its own
  freshly ingested DB at commit 9261170. The model code at 9261170 and bbf68e9 is the same: the
  commits between them add the template lag, report rows and measurement flags, which default
  off. So the control and the season sim share a model but not a DB. The A9 evidence below shows
  DB differences didn't move the control at all: the same per-GW squad points came back on three
  different cached DBs.
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
| oracle model | 2024-25 | 37 | 63.43 | — (FPL published one, but the data source has no 2024-25 `gameweek_summaries.csv`) | −2.67 |
| gated template (G−2) | 2024-25 | 36 | 53.06 | — | −13.05 |
| oracle model | 2025-26 | 37 | 60.11 | +10.35 | −1.12 |
| gated template (G−2) | 2025-26 | 36 | 52.53 | +2.81 | −8.70 |
| season sim control | 2025-26 | 37 | 57.81 | +8.05 | −3.41 |
| season sim control_b | 2025-26 | 37 | 57.81 | +8.05 | −3.41 |
| model (oracle), leaderboard | 2026-27 | 4 (GW2–5) | 63.00 | +0.75 | −11.35 (GW2–5 pace 74.35) |
| leaderboard sim (evolving) | 2026-27 | 4 (GW2–5) | 60.25 | −2.00 | −14.10 (GW2–5 pace 74.35) |
| live team | 2026-27 | 5 (GW1–5) | 60.40 | +0.60 | −15.08 (GW1–5 pace 75.48) |

Each 2026-27 row is set against the cutoff's pace over its own gameweeks. Five weeks of a
cutoff pace can't be compared with a full season's.

### Each named cause, measured (pts/GW)

| cause | where in the chain | measured | source |
|---|---|---|---|
| evolving loss: carrying a squad, hits, chip timing | oracle model → season sim (2025-26) | **E = +2.30 ± 2.12** | paired over 37 GWs, control vs season sim |
| availability flag read from row G−1 (possible post-deadline information) | inside the oracle backtest | **−0.47 ± 0.79** when read from G−2 instead (MAE +0.04 worse) | run 38269664839 vs control |
| the live-only EO template anchor (risk posture v2, κ 0.12) | live from-scratch solves, plus the GW1 seed carried forward | **−2.96 ± 1.49** (overlap 3.25 → 7.40; non-template points −20.4) | run 38269669897 vs control |
| backtest sim vs live sim (live optimism O) | 2026-27 leaderboard sim → live team | **O = −1.75 ± 12.66** (4 GWs), so O⁺ = 0 | per-GW pairs |
| control noise across runners (A9) | any walk-forward comparison | **+0.135 ± 0.125**; 3 of 74 GWs differ | runs 38267875032 / 38267878143, same DB |
| season-sim noise on one runner | season-sim comparisons | **0.00 ± 0.00** | control vs control_b, run 38229024307 |
| the synthetic EO crowd is not FPL's average | the old yardstick | 325.4 vs FPL's 249 over GW2–5: **+19.1/GW above the real field** | leaderboard and `app_model_team.json` |

- **SE of the live sample:** the live team's GW1–5 gaps to FPL's average are +4, −19, +16, +4
  and −2: mean +0.6, **SE 5.7/GW**. After 5 GWs the live record says almost nothing about pace.
- **The backtest → 2026-27 step, the largest in the chain, has no found cause:**

  | step (vs FPL's average, pts/GW) | backtest | 2026-27 (GW2–5) | change |
  |---|---|---|---|
  | oracle model | 2025-26 +10.35 ± 2.29 (37 GWs) | +0.75 ± 4.87 | **−9.60 ± 5.38** (1.8 SE) |
  | oracle model, same weeks of the season | 2025-26 GW2–5 +8.75 ± 5.72 | +0.75 ± 4.87 | −8.00 ± 7.51 (1.1 SE) |
  | evolving manager | season sim 2025-26 +8.05 ± 2.01 | leaderboard sim −2.00 ± 5.55 | −10.05 ± 5.90 |
  | oracle model − gated template | +9.19 ± 1.93 (72 GWs, G−2 template) | −3.50 ± 8.59 (leaderboard, G−1 template) | −12.69 ± 8.80 (−10.80 against the G−1 backtest figure, +7.30) |

  None of these changes is beyond 2 SE on 4 live GWs. The one candidate tested inside the backtest,
  the availability flag, is worth −0.47 ± 0.79. An early-season effect is unlikely: the
  backtest's own GW2–5 of 2025-26 ran at +8.75. The remaining hypotheses are unmeasured: the
  2026-27 data feed (the match-feed lag found in PR #261), and plain luck. **Decision under the declared branch:** no cause is found, so there
  is no fix and no haircut. The step is an open risk. Phase 11's out-of-sample check (model
  (oracle) − gated template over GW2–N on the weekly leaderboard) re-tests it at the first
  review. If m + SE < 0 by GW19, the programme pauses.
- **The EO anchor shapes from-scratch solves:** the GW1 seed and wildcard or free-hit weeks
  (the live team played Free Hit in GW3). The GW1 seed carries forward every week until it is
  traded away, so the anchor's live cost isn't limited to those weeks. Transfers come from the
  planner, which has no EO term.
  - The leaderboard sim's own bootstrap has no anchor (`export_leaderboard.py` passes no
    ownership or risk-posture version), so O already contains part of the anchor's effect.
  - Per the plan's Current State decision the anchor stays as is. At −2.96 it is the largest
    measured cost in the chain, and the owner may want to revisit it.
- **Backtest optimism:** of the candidates, only the availability flag was testable inside the
  backtest. It is worth less than the declared 1.0 pts/GW trigger, so **G3 is not haircut**.
  The larger backtest → 2026-27 step is unexplained and stays an open risk (above).

### G2's margin M, by the declared formula

M_s = max(2.0, C_s + E + O⁺ − T_s), with E = +2.297 (both seasons) and O⁺ = 0.
- **Which E is frozen, and why:** E comes from this phase's own control, 38269659997, paired with
  the season sim over GW2–38 of 2025-26.
- **The two values of E are a runner-noise effect:** the walk-forward isn't deterministic across
  runners (A9). This control's per-GW squad points are identical to A9's control_b (38267878143).
  The Phase 1 control's are identical to A9's control (38267875032). That holds across three
  different cached DBs, so the DB doesn't matter but the runner does.
- **The other branch:** it gives E = +2.054 ± 2.067, and M̄ = 12.926 instead of 13.169 (±0.24).
- **The higher, stricter bar is the one frozen.**

| season | C_s | T_s | E | O⁺ | M_s | template GWs |
|---|---|---|---|---|---|---|
| 2024-25 | 66.103 | 53.056 | 2.297 | 0 | **15.344** | 36 |
| 2025-26 | 61.224 | 52.528 | 2.297 | 0 | **10.993** | 36 |

**M̄ = 13.169 pts/GW** (GW-weighted). M and its inputs are frozen here, and later controls are
judged against this M̄.
- **C_s carries no stated error.** Each cutoff is interpolated between sampled entries 1.2–1.6k
  ranks either side of 100,000. Within 2× of the target, only 10 (2024-25) and 6 (2025-26)
  sampled entries rank better and 22 and 26 rank worse. Near rank 100k one rank is worth a small
  fraction of a point, so the interpolation error is small, but it is not measured.
- **The G−2 lag weakens the template, and that moves m and M̄ equally.** On common GWs the lag
  costs the template 0.78 pts/GW (2024-25) and 1.33 (2025-26). That raises m and M̄ by the same
  amount, because T_s cancels out of m − M̄.

**G2 today:** the control's pooled gap to the gated template is m = **+9.19 ± 1.93** (72 GWs),
so G2 is **not met**. The model needs about +4.0 pts/GW more over the template. Equivalently,
the oracle model must reach C_s + E: 68.4 (2024-25) and 63.5 (2025-26). On the template's
gameweeks it is at 63.42 and 60.56 today.

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

- **Backtest optimism:** none found among the tested candidates, so no fix and no haircut. The
  backtest → 2026-27 step is an open risk for Phase 11.
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

**Phase 3 critique:** FAIL in round 1, with two blockers: the backtest → 2026-27 step was
unexplained, and E had two values. Both were fixed in 11fc397. PASS in round 2; its five wording
fixes were applied in the next commit.

## Phase 4: re-judge the near misses (lever Z), batch Z1 (declared 2026-10-12, before dispatch)

- **Arms:** `--no-goal-anchor` and `--no-rate-price-anchor`, each against one control. All three
  run with `--template-lag 2`, over both seasons (no `--seasons`). They are dispatched
  back-to-back on this branch at the commit that adds this declaration, through
  `branch_walkforward.yml`, so they share one cached DB.

  | run | args |
  |---|---|
  | control | `--template-lag 2` |
  | Z-goal | `--template-lag 2 --no-goal-anchor` |
  | Z-price | `--template-lag 2 --no-rate-price-anchor` |

- **Rule:** R6, via `compare_arms.py control.json arm.json --rule r6`:
  - (i) paired squad points over 74 GWs with m − SE ≥ 0;
  - (ii) each season's MAE ≤ control + 0.01;
  - Req 9's collapse guard;
  - the replicate rule when |m − SE| ≤ 0.1.
- **If both pass:** the higher m wins, and the other is re-judged against the new control.
- **Refusals:** a run on a different DB re-dispatches both.
- **Background:** in 2025-26 alone the arms were +1.00 ± 0.83 (MAE +0.008) and +1.32 ± 1.01
  (MAE +0.011) (`docs/reports/2026-10_premium_and_form_returns.md`).
- **The season-sim check, on a pass (Req 7):** `season_sim_arms.yml`, 2025-26, chunk `2-38`,
  `same_runner`, with arms `control: ; control_b: ; arm: <flag>`. It passes on paired net points
  m ≥ 0. Then comes the live switch (Req 8), unless m < 0, which goes to the owner.
- **Arms used after this batch:** 2 of 10 (family Z).

### Batch Z1 results (cached DB duckdb-rank-tracking-38267694488, commit 3fcb3c5, 75/75 steps each)

| run | args | vs control (pts/GW, 74 GWs) | m − SE | EP MAE 2024-25 / 2025-26 | Req 9 | R6 |
|---|---|---|---|---|---|---|
| 38275606306 | control, `--template-lag 2` | — | — | 1.0385 / 1.0450 | — | — |
| 38275613712 | `--no-goal-anchor` | **+0.824 ± 0.655** (28 better, 16 worse) | **+0.169** | 1.0453 (+0.0068) / 1.0513 (+0.0063) | ok: overlap 3.25 → 3.24, non-template +1.44 ± 0.64 | **PASS** |
| 38275619225 | `--no-rate-price-anchor` | +0.635 ± 0.736 (28 better, 21 worse) | −0.101 | 1.0481 (+0.0096) / 1.0533 (+0.0083) | ok: overlap 3.25 → 3.26, non-template +1.17 ± 0.68 | FAIL (i) |

- **Verdicts:** `--no-goal-anchor` passes R6. Its m − SE (+0.169) is outside the replicate band
  (≤ 0.1), so no replicate. `--no-rate-price-anchor` fails (i). Its |m − SE| = 0.101 is just
  outside the band too, so it gets no replicate either. The plan's band is fixed and this is not
  re-litigated.
- **Template gap:** the control beats the gated template by +9.06 ± 1.93 pts/GW and Z-goal by
  +9.92 ± 1.96, against M-bar 13.17. G2 is not met.
- **Measured nondeterminism:** this control and Phase 3's control 38269659997 have the same args
  and the same DB, at commits that differ only in reporting code. They differ in 3 of 74 GWs,
  −0.135 ± 0.125 pts/GW, in line with A9.
- **Family Z:** one pass and one fail, so the family is not dropped. Only one arm passed, so the
  tie-break doesn't apply.
- **PROMOTE, step 1:** the season-sim check declared above, for `--no-goal-anchor`, at the commit
  that records these results. Alongside it, a walk-forward with
  `--template-lag 2 --no-goal-anchor --export-residuals` is dispatched. It is not an arm and does
  not count. If Z-goal is promoted, it is the new control's residual export that L1 is refitted
  from. If not, L1 is fitted from 38269659997's export.

### PROMOTE: the season-sim check and the live switch for `--no-goal-anchor`

- **Season-sim check 38312626413** (commit cb4b651, 2025-26 GW2–38, one runner, arms
  `control: ; control_b: ; z_goal: --no-goal-anchor`). Net is gross XI points minus hits.

  | arm | net pts/GW | vs FPL average | vs control (paired, 37 GWs) | hits | transfers | chips |
  |---|---|---|---|---|---|---|
  | control | 59.49 | +9.73 | — | 0 | 29 | GW5 WC, GW6 TC, GW8 FH, GW9 BB, GW20 FH, GW21 BB, GW22 TC |
  | control_b | 59.49 | +9.73 | +0.00 ± 0.00 | 0 | 29 | the same |
  | z_goal | 61.00 | +11.24 | **+1.51 ± 1.13** (21 better, 10 worse) | 0 | 29 | GW4 TC, GW5 WC, GW8 FH, GW12 BB, GW20 FH, GW21 BB, GW22 TC |

  The declared rule is paired net points m ≥ 0: **+1.51, PASS**. The one-runner floor is exactly
  0. The arm moves two chips (Triple Captain GW6 → GW4, Bench Boost GW9 → GW12).
- **Live switch (Req 8), commit after this section:**
  - `rate_prior_params` v2 = v1 with `anchor_goals` 0 (`expected_points.LIVE_RATE_PRIOR`), seeded
    by `seed_rate_prior_params()`. That is called from `materialize_confirmed_seeds()` and
    `run_ingestion.py`, as v1 was.
  - The family was already in `RECALIBRATABLE_VERSION_ARGS`, and every call site already passes it
    (the wiring test's `REQUIRED` lists include it). The active version moves to v2 through the
    committed seed `data/recalibration/seeds_live_switches_2026-10-13.json`, which confirms every
    bundle key at v2.
  - The off switch is `--goal-anchor` in both scripts, which maps to v1 (the rate prior live until
    2026-10-13). `--no-goal-anchor` alone is now the control.
  - The wiring test, the flag tests and the full suite pass.
- **What follows:** PR #263 merges once CI is green, then `scheduled_pipeline.yml` runs on master
  with a projections before/after diff. The kill-switch clock (Phase 11) starts with that
  pipeline's first live gameweek.
- **Live (2026-10-19):** PR #263 merged as `3b16a20`. `scheduled_pipeline.yml` run 39340718301 on
  master rebuilt the projections with rate prior v2, generated 23:35 UTC.
  - **Before/after diff**, against master's 21:58 UTC projections on the same day's data (v1):
    - GW8 EP moved by a mean |change| of 0.018 pts a player, net +0.009.
    - By position and band in GW8: midfielders 9.0+ −0.22, midfielders 7.0–9.0 −0.08,
      forwards 9.0+ −0.10, midfielders <5.0 +0.04, midfielders 5.0–7.0 +0.01.
    - Over the 8-GW horizon (GW8–15), the biggest fallers are Barcola −2.63, Bruno Fernandes −1.93,
      Saka −1.56, Palmer −1.36, Haaland −1.18 and Cherki −0.82.
    - The biggest risers are cheap midfielders: Slater +1.85, Crooks +1.52, Grimes +1.45 and
      Rudoni +1.32.
    - The captain ranking's top 10 is unchanged (Palmer, Bruno Fernandes, Saka, Mbeumo, …).
  - This is the intended effect: premium goal rates shrink toward the position average and assists
    keep their price anchor.
  - The kill-switch clock (Phase 11) starts at GW8, the first live gameweek on v2. Its comparison
    is against `--goal-anchor` from that gameweek on.
- **The new control** is `--template-lag 2` on this commit or later (Z-goal live). Run 38312629527
  (`--no-goal-anchor --export-residuals`, cached DB duckdb-38283940509) is its residual export,
  and L1 is fitted from it.

## Phase 5: lever L1, the crowd prior, batch L1 (declared 2026-10-13, before dispatch)

- **Code:** `daedeee` (the build) and `f93743e` (critique round 1 fixes). The blind critic returned
  PASS with no blockers. Its minor findings are fixed in `f93743e` except three, which are deferred:
  - the live script is wired to the flag at PROMOTE (Req 8);
  - the own-season adjustment column is added before Phase 6 needs it (stacked folds);
  - a golden pre-change EP comparison is replaced by the zero-coefficient identity test plus the
    existing EP suites.
- **Coefficient file:** `data/recalibration/crowd_prior_coeffs_20261013.json`, committed with this
  declaration.
  - It was fitted by `scripts/fit_residual_lever.py` on the residual export of run 38312629527,
    which is the new control (`--template-lag 2 --no-goal-anchor`, now live; cached DB
    duckdb-38283940509). The export's sha256 is
    `d714b8132d57786dc38ff6873d519f4bec1491e294b3dd201896b117b446a915`.
  - Declared values: `ridge_alpha` 10 on standardized features, no intercept; `min_rows` 2,000;
    clip ±1.5 pts/GW; features `own_pct`, `log_sbp`, `xfer_pct` from row G−2 (A2 failed).
  - Folds: 2024-25 is scored with the 2025-26 fit, 2025-26 with the 2024-25 fit, and 2026-27 with
    both.

  | fold | n | β own_pct | β log_sbp | β xfer_pct |
  |---|---|---|---|---|
  | 2024-25 (scores 2025-26) | 25,675 | +0.075 | +0.038 | +0.067 |
  | 2025-26 (scores 2024-25) | 27,882 | +0.070 | +0.022 | +0.126 |
  | both (2026-27) | 53,557 | +0.072 | +0.031 | +0.098 |

  β is in pts/GW per standard deviation of the feature.
- **Out-of-fold check** (in the file): each season's player-gameweek residual MAE against the other
  season's fit is 1.0394 → 1.0363 for 2024-25 and 1.0496 → 1.0423 for 2025-26. RMSE is
  1.9563 → 1.9542 and 1.9436 → 1.9398.
- **Deviation from the plan's text, disclosed: the zero floor.** An adjusted fixture EP never goes
  below zero, or below itself when it is already negative.
  - It was decided after the cross-fold check on the previous control's export (38269659997). There,
    the unfloored adjustment left 21–24% of player-gameweeks with negative EP. Unfloored MAE was
    1.0406 / 1.0528 and floored 1.0301 / 1.0395, against the control's 1.0328 / 1.0435. No other
    variant was tried.
  - On this export it binds on 21.9% / 26.3% of adjusted player-gameweeks. Unfloored MAE would be
    1.0470 / 1.0577.
  - The floor keeps an invariant the control already holds (EP ≥ 0). It binds only where a fixture's
    EP is below the adjustment, which in practice means non-playing players.
  - It gives the adjustment a small positive mean (+0.015 / +0.012 pts per player-gameweek).
- **The Monte Carlo summary** (Triple Captain score, vice fallback) is shifted by the same
  adjustment. Variance is unchanged.
- **Arms** (one arm; both runs with `--template-lag 2`, both seasons), dispatched back-to-back on
  this branch at the commit that adds this declaration, through `branch_walkforward.yml`:

  | run | args |
  |---|---|
  | control | `--template-lag 2` |
  | L1 | `--template-lag 2 --crowd-prior 20261013` |

- **Rule:** R6 via `compare_arms.py control.json arm.json --rule r6`, with Req 9 and the replicate
  rule. A run on a different DB re-dispatches both. The arm's log line `[crowd_prior]` must show
  adjusted steps in both seasons, or the arm is not judged.
- **On a PASS:** the season-sim check `control: ; control_b: ; l1: --crowd-prior 20261013`
  (2025-26, chunk `2-38`, `same_runner`), then the live switch (Req 8), unless m < 0, which goes to
  the owner.
- **Arms used after this batch:** 3 of 10 (Z: 2, L1: 1).

### Batch L1 results (cached DB duckdb-38324360004, commit 22112a1, 75/75 steps each)

| run | args | vs control (pts/GW, 74 GWs) | m − SE | EP MAE 2024-25 / 2025-26 | Req 9 | R6 |
|---|---|---|---|---|---|---|
| 38327384582 | control, `--template-lag 2` (Z-goal live) | — | — | 1.0453 / 1.0515 | — | — |
| 38327387171 | `--template-lag 2 --crowd-prior 20261013` | **−0.770 ± 0.816** (21 better, 32 worse, 53 changed) | **−1.587** | 1.0424 (−0.0029) / 1.0439 (−0.0076) | ok: overlap 3.24 → 3.43 (+0.19), non-template −1.93 ± 0.84 | **FAIL (i)** |

- **The arm ran as declared:** `[crowd_prior] v1: 72/75 steps adjusted, 54048 player-fixtures,
  mean |adjustment| 0.0768, 13057 floored at zero EP`. The three unadjusted steps are each
  season's opening gameweeks, which have no row G−2.
- **Verdict: FAIL.** m − SE is −1.587, far outside the replicate band. No season-sim check is run.
- **Why it failed:**
  - The adjustment did what the fit promised for EP error: MAE fell in both seasons, matching the
    out-of-fold check.
  - It cost squad points. It raised the EP of owned, transferred-in players, which pulled the XI
    toward the template (overlap +0.19 a GW). The points the model scores away from the template
    fell by −1.93 ± 0.84 a GW, and the arm's gap to the gated template dropped from +9.92 to +9.08.
  - The crowd's information is already largely in the template's picks, and the model's edge is
    in the players the crowd doesn't hold. A better calibrated mean on average did not help the
    picks at the top of the EP distribution. That is the decision-weighted error, which ridge on
    every player-gameweek does not target.
- **The new control** reproduces the Z1 `--no-goal-anchor` arm (38275613712) on a newer DB: 1 of
  74 GWs differs, −0.041 ± 0.041. That confirms the live switch (v2) is the arm that was tested.
- **Family L1:** one fail. It is dropped after a second consecutive fail, and no L1 variant is
  queued (the plan declares one arm).
- **Arms used:** 3 of 10 (Z: 2, L1: 1).
- **Next:** Phase 6 (L2, the position × price-band bias). Its residuals come from the current
  control, which has Z-goal live and no L1, so export 38312629527 stands in. Its fit adds no crowd
  term, so the own-season L1 column the stacked-folds rule needs is not required.

## Phase 6: lever L2, the position × price-band bias, batch L2 (declared 2026-10-19, before dispatch)

- **Code** (this commit):
  - `residual_levers.fit_band_cells` / `fit_band_bias`: each (position, price band) cell's mean
    residual, realized − the projection before any lever, shrunk by n/(n + `pseudo_rows`). Folds
    follow `fold_for`, as L1's do.
  - `band_adjustments` gives each player his cell at the deadline. The price is his latest before
    the deadline: the one the optimizer budgets with and the residual export bands by. Price is not
    one of A2's crowd columns.
  - `expected_points.run(band_bias_params_version=…)` applies it after L1 through the shared
    `_apply_lever_adjustments` (split by EP across a double gameweek, floored at zero EP) into
    `ep_band_adj` (schema 0030).
  - It is threaded through the walk-forward step, the season sim and the planner horizon (the
    deadline argument is now `lever_deadline_gameweek`, shared with L1). The Monte Carlo summary
    shifts by both levers.
  - `--band-bias COEFF_DATE` is in both scripts. `fit_residual_lever.py --lever band_bias` writes
    the file.
  - Tests in `tests/test_band_bias.py` cover shrinkage, folds, the leak (row G's price is never
    read), missing data, the floor, stacking on L1, the export, the same adjustment in the
    walk-forward and the planner horizon, the season-sim threading, the Monte Carlo shift, both
    flags, control identity and the file hash.
- **Coefficient file:** `data/recalibration/band_bias_coeffs_20261019.json`, committed with this
  declaration.
  - Fitted on residual export 38312629527 of the current control (`--template-lag 2`, Z-goal live,
    L1 off). The export's sha256 is
    `d714b8132d57786dc38ff6873d519f4bec1491e294b3dd201896b117b446a915`.
  - `pseudo_rows` 500, as declared.
  - No lever is live in those residuals, so the stacked-folds rule has nothing to substitute.

  | cell | 2024-25 fit (scores 2025-26): n, mean, adj | 2025-26 fit (scores 2024-25): n, mean, adj |
  |---|---|---|
  | Midfielder 9.0+ | 221, +0.815, **+0.250** | 142, +0.467, **+0.103** |
  | Midfielder 7.0–9.0 | 312, +0.491, +0.189 | 627, +0.235, +0.131 |
  | Forward 7.0–9.0 | 390, +0.298, +0.131 | 523, +0.180, +0.092 |
  | Forward 9.0+ | 73, +0.423, +0.054 | 98, −0.174, −0.029 |
  | Midfielder 5.0–7.0 | 5,276, +0.066, +0.061 | 5,708, −0.080, −0.074 |
  | Midfielder <5.0 | 6,072, −0.164, −0.152 | 6,324, −0.115, −0.107 |
  | Defender <5.0 | 7,905, −0.150, −0.141 | 7,698, −0.085, −0.080 |
  | Goalkeeper <5.0 | 2,462, +0.008, +0.007 | 2,924, −0.206, −0.176 |

  The full 13 cells per fold are in the file.
- **Out-of-fold check** (in the file, floored as applied):

  | season | MAE | RMSE |
  |---|---|---|
  | 2024-25 | 1.0394 → 1.0042 | 1.9563 → 1.9541 |
  | 2025-26 | 1.0496 → 1.0306 | 1.9436 → 1.9425 |

  - The floor binds on 25.7% / 21.9% of player-gameweeks. These are mostly the cheap cells' cuts on
    near-zero EP, which is where most of the MAE gain comes from.
  - The premium cells' boosts carry the expected squad effect.
- **Stated risks:**
  - B3's band *conversion* (scaling goal and assist rates by realized/xG per band) failed:
    −2.27 ± 0.97.
  - L1 just failed by pulling the XI toward the template. A premium boost could do the same,
    because premiums are the most-owned players. Req 9 guards this.
- **Arms** (one arm; both runs with `--template-lag 2`, both seasons), dispatched back-to-back on
  this branch at the commit that adds this declaration, through `branch_walkforward.yml`:

  | run | args |
  |---|---|
  | control | `--template-lag 2` |
  | L2 | `--template-lag 2 --band-bias 20261019` |

- **Rule:** R6 via `compare_arms.py control.json arm.json --rule r6`, with Req 9 and the replicate
  rule. A run on a different DB re-dispatches both. The arm's `[band_bias]` log line must show
  adjusted steps, or the arm is not judged.
- **On a PASS:** the season-sim check `control: ; control_b: ; l2: --band-bias 20261019` (2025-26,
  chunk `2-38`, `same_runner`), then the live switch (Req 8), unless m < 0, which goes to the owner.
- **Arms used after this batch:** 4 of 10 (Z: 2, L1: 1, L2: 1).

### Batch L2, first run (cached DB duckdb-39340718301, commit 2b0086a, 75/75 steps each)

| run | args | vs control (pts/GW, 74 GWs) | m − SE | EP MAE 2024-25 / 2025-26 | Req 9 | R6 |
|---|---|---|---|---|---|---|
| 39342719420 | control, `--template-lag 2` | — | — | 1.0453 / 1.0515 | — | — |
| 39342721804 | `--template-lag 2 --band-bias 20261019` | +0.486 ± 0.560 (19 better, 27 worse, 46 changed) | −0.073 | 1.0100 (−0.0353) / 1.0325 (−0.0190) | ok: overlap 3.24 → 3.42 (+0.18), non-template −0.25 ± 0.63 | **REPLICATE** |

- **The arm ran as declared:** `[band_bias] v1: 74/75 steps adjusted, 55748 player-fixtures, mean
  |adjustment| 0.0781, 13206 floored at zero EP`.
- **EP error** fell in both seasons, by about what the out-of-fold check predicted. The arm beats
  the gated template by +10.42 ± 1.94 a GW, against the control's +9.92.
- **Squad points:** |m − SE| = 0.073 ≤ 0.1, so the replicate rule applies. Control and arm are
  re-dispatched once on the same code. This session can push only the branch, so they run at the
  branch head that adds this note. Its only difference from `2b0086a` is this report: no code, data
  or workflow change, checked with
  `git diff 2b0086a HEAD -- src scripts schema tests .github data/recalibration`. The verdict
  is R6 on the per-GW mean of the two runs' (arm − control) differences, with MAE the mean of the
  two runs' MAEs (`compare_arms.py --replicate`).
- **Arm count:** the replicate counts as one arm, so 5 of 10 after it (Z: 2, L1: 1, L2: 2).
- **Drift:** this control is identical to the L1 control (38327384582) on a newer DB, with 0 of 74
  GWs changed.

### Batch L2 replicate (cached DB duckdb-39340718301, branch head 75dd2d8, 75/75 steps each)

| run | args | vs control (pts/GW, 74 GWs) | m − SE | EP MAE 2024-25 / 2025-26 | Req 9 | R6 |
|---|---|---|---|---|---|---|
| 39345466273 | control, `--template-lag 2` | — | — | 1.0453 / 1.0513 | — | — |
| 39345468971 | `--template-lag 2 --band-bias 20261019` | +0.446 ± 0.562 (19 better, 28 worse, 47 changed) | −0.116 | 1.0100 (−0.0353) / 1.0325 (−0.0188) | ok: overlap 3.24 → 3.42 (+0.18), non-template −0.25 ± 0.63 | — |
| verdict | means of both runs (`--replicate`) | **+0.466 ± 0.560** (19 better, 28 worse, 47 changed) | **−0.094** | 1.0453 → 1.0100 / 1.0514 → 1.0325 | ok | **FAIL** |

- **The arm ran as declared:** the replicate's log line is identical to the first run's,
  `[band_bias] v1: 74/75 steps adjusted, 55748 player-fixtures, mean |adjustment| 0.0781, 13206
  floored at zero EP`. All four runs used the same DB and scored 75/75 steps.
- **Measured nondeterminism:** 1 of 74 GWs differs between the two controls, by up to 3.0 pts.
- **Why it failed:** R6(i). The paired gain is positive but within one SE of zero (m − SE −0.094).
  R6(ii) passes comfortably: EP error falls in both seasons.
  - The squad changes are lopsided: 19 GWs better against 28 worse. A few large wins carry the
    mean.
  - The XI moves 0.18 players toward the template. That is within Req 9 but it is the direction the
    stated risk named: the premium boost buys the most-owned premiums.
  - The arm's model-XI spend on £9.0m+ fell (26.3 against the control's 27.4 a GW). So the gain does
    not come from buying premiums; it comes from the cheap cells' cuts moving money elsewhere.
- **Family count:** L2 1 fail (L1 1 fail; Z promoted). **Arms used:** 5 of 10 (Z: 2, L1: 1,
  L2: 2).
- **Combine candidate (Phase 9):** L2 has m > 0 and failed only m − SE ≥ 0, so it is listed with
  `--no-rate-price-anchor`.

### The L3 gate, re-checked after L2

The queue's rule: L3 runs only if £9.0m+ realized − projected is still ≥ +0.1 after the levers that
passed. Z is the only lever that passed. The current control (39345466273; Z-goal live, no other
lever) gives **+0.547** per player-GW for £9.0m+. Its components:

| total | appearance | goals | assists | clean sheet | other (bonus …) |
|---|---|---|---|---|---|
| +0.547 | +0.008 | +0.144 | +0.168 | −0.011 | +0.237 |

- The total is higher than Phase 3's +0.170. Unanchoring goals moved the goal component from
  −0.177 to +0.144, while assists and bonus are unchanged.
- **L3 is queued.** Phase 7 next names the point-in-time EP-history source in each harness, and
  drops L3 if any harness lacks one.
