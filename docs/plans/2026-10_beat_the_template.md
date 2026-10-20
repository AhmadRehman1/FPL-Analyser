# Plan: Beat the template, then push toward a top-100k season

One-line goal: the model's picks beat a legal, availability-aware ownership-template squad by a wide
margin on a fair backtest, and the live model team scores at the pace of FPL's actual top-100k
cutoff. Today the oracle model only roughly matches the template, and the live team sits about
level with FPL's average.

## Classification
Track: Performance. The end state is a number: squad points against the template, then season
points against the real top-100k cutoff. Unlike a classic performance job, the projections are
meant to change, so "frozen surfaces" means output formats and the live pipeline, not model
behaviour.

Parked secondary asks:
- Planner and chip re-tuning: Phase 10. It runs early if Phase 3 finds the evolving-vs-oracle loss
  is large.
- Elite (top-10k) ownership as an input: Out of Scope, with triggers.

## Interview Ledger
- **Q1 crowd input:** ownership and transfer trends MAY feed the projections as a backtested prior
  (user, accepted recommendation).
- **Q2 promotion rule:** "rule R6", points-first (user, accepted recommendation):
  - paired squad points over both seasons with m − SE ≥ 0;
  - EP MAE ≤ control + 0.01 in each season;
  - then a season-simulation check with m ≥ 0 before going live.
- **Q3 finish line:** "beat template by a lot; we need to push to like top 100k overall" (user, free
  form). It is translated into G2–G4 below [A10].

Questions used: 3 of 14, plus the approval question.

## Goal & Success Criteria
- **G1 (fair yardsticks exist).**
  - **Oracle.** Every walk-forward summary reports, per gameweek, the model squad's realized points
    and a legal *availability-gated* ownership-template squad's realized points.
    - The gate excludes players whose row G−1 `status` is 'i', 's', 'u' or 'n'.
    - The template is built under the same budget, formation, club cap, captain/vice/bench and
      auto-sub rules, as a fresh hit-free squad each week, like the model's.
    - The ungated template is reported alongside.
    - `compare_arms.py --rule r6` judges arms on paired squad points.
  - **Live.** The live leaderboard gains a "model (oracle)" row (Req 3), so its oracle template
    baselines are compared like for like.
- **G2 (beat the template by enough to reach the cutoff).** On the two-season oracle walk-forward,
  the pooled paired gap m = model − gated template satisfies m ≥ M̄ and m − SE > 0. Phase 3
  computes M by this declared formula [A10]. All terms are in pts/GW.
  - Per season s ∈ {2024-25, 2025-26}: M_s = max(2.0, C_s + E + O⁺ − T_s).
    - **C_s:** season s's real top-100k cutoff pace (rank-100k total ÷ 38, Req 4).
    - **T_s:** the gated template's mean squad points over season s's walk-forward GWs (Phase 1
      control).
    - **E:** the evolving loss, oracle model − season-sim control, both 2025-26. The season sim runs
      for 2025-26 only, so E is used for both seasons.
    - **O⁺:** the live optimism O = 2026-27 leaderboard sim pace − live team pace, both over the
      same GWs. O⁺ = O when O > 2·SE(O), else 0. A noisy or negative O never lowers M.
  - M̄ is the GW-weighted mean of M_2024-25 and M_2025-26, matching how m pools the two seasons.
  - M and its inputs are frozen in the Phase 3 commit. Later controls are judged against that M̄.
  - Today the oracle model trails the *synthetic* crowd by 1.15 pts/GW pooled (verified:
    `data/dashboard/app_track_record.json` headline).
  - The oracle model is +10.4 pts/GW over FPL's real 2025-26 average (verified: walk-forward
    control run 38167650902, `beats_real_avg_points_delta`).
- **G3 (top-100k pace, backtest).** The evolving manager (transfers, hits, chips;
  `season_sim_arms.yml`) is run as one continuous GW2–38 squad, with crowd-prior coefficients from
  the *other* season's fold. Its net pts/GW is at least the real top-100k cutoff pace of the season
  replayed.
  - The cutoff pace is total points at overall rank 100,000 ÷ 38, from sampled managers' FPL
    `entry/{id}/history` `past` arrays (Req 4) [A1].
  - The comparison is per GW over GW2–38, which approximates the full season. That is stated in the
    report.
- **G4 (live).** The live model team's cumulative net points (`data/dashboard/app_model_team.json`
  `total_realized_points`, net of hits) reach the 2026-27 top-100k cutoff's cumulative points.
  - The cutoff is estimated per GW from sampled managers' `entry/{id}/history` `current` arrays
    near rank 100k (Req 4).
  - Reviewed at GW19 and judged at GW38 of 2026-27.
  - Today: 302 pts over GW1–5, +3 against FPL's average, with Free Hit and Triple Captain used
    (verified: `app_model_team.json`).
- **Stop rule.**
  - A lever family (one of Z, L1, L2, L3, L4, L5, together with its variants) is dropped after two
    of its arms in a row fail rule R6.
  - Lever arms in Phases 4–9 count towards a cap of 10. Planner arms (Phase 10) have their own cap
    of 6. A replicate (Key Decisions) counts as one arm. Controls do not count.
  - The programme pauses for owner review at any of these:
    - either cap is reached;
    - mean template overlap reaches ≥ 9/11 (Req 9);
    - the backlog is empty with G2 unmet;
    - the GW19 review shows G4 is more than 100 points behind the cutoff.
  - The programme ends at GW38 of 2026-27 with a final report, or earlier if G2, G3 and G4's GW19
    review all hold and the owner closes it.

## Current State
- **The walk-forward squad is an oracle.** It is a fresh, hit-free squad every gameweek (verified:
  `src/fpl_quant/backtest.py` `real_average_benchmark_provenance()` "oracle": True). Against the
  *synthetic* crowd (real points weighted by ownership; `_avg_manager_benchmark_points`):
  - 2024-25: model 63.54 pts/GW, crowd 61.85;
  - 2025-26: model 60.16, crowd 64.15.
  
  (verified: control run 38167650902 `headline_by_season`)
- **Pace by harness:**
  - oracle model +10.4 pts/GW over FPL's real 2025-26 average (same run);
  - honest 2025-26 season sim +8.1 (verified: `docs/reports/2026-10_promoted_club_strength.md`,
    "Season sim" bullet);
  - live model team +3 over 5 GWs, about +0.6/GW (verified: `app_model_team.json`).
- **A legal template squad already exists in code.** `backtest._naive_candidate_pool(signal=
  "ownership")` builds it (mu = the latest `selected_by_percent`, read inside `asof_scope()`). It is
  solved by `squad_optimizer.solve(pool, {}, 0.0, guardrail_cap, captain_variance_multiplier=
  LIVE_CAPTAIN_VARIANCE_MULTIPLIER)` and scored by `_realized_solve_points()` with the solve's own
  vice and bench order. It has no availability filter. Only `backtest.beats_baseline()` uses it
  (verified: backtest.py `_naive_candidate_pool`, `beats_baseline`, `_realized_solve_points`).
- **The live leaderboard is not like for like.** `scripts/export_leaderboard.py` scores the
  *evolving* `run_season_simulation()` model against *oracle* baselines (verified: that script,
  lines 4–24 and 148). Over 2026-27 GW2–5:
  - evolving model 241;
  - oracle ownership template 266;
  - "crowd-average baseline" 323.1, the synthetic EO crowd, far above FPL's real average for those
    weeks (81 + 51 + 69 + 48 = 249);
  
  (verified: `leaderboard_latest.json`, `app_model_team.json` `field_average`).
  `index.html:6810` takes the first row with `is_baseline` false as the model (verified).
- **The season sim does not repeat across CI jobs.** The same DB and params split at GW3 (Bench
  Boost played vs held): ±1.81/GW. Floating point differing between runner CPUs is suspected
  (verified: `promoted_club_strength.md`).
- **Season-sim runtime.** In run 37397424563 each GW2–19 chunk took 17–20 min (verified:
  `actions_list list_workflow_jobs`). So a GW2–38 job fits well inside the 345-min step limit (job
  timeout 358, verified: `season_sim_arms.yml`).
- **Walk-forward runtime.** A two-season walk-forward took 21 min (control) and 39 min (arm)
  (verified: runs 38167650902 and 38167653259). `branch_walkforward.yml` uploads only
  `walkforward_summary.json`; its DB is never saved (verified: that workflow). Its concurrency group
  is `branch-walkforward-${ref}-${args}` with `cancel-in-progress: true`.
- **Ownership and transfers never reach the projections** (verified: no `selected_by_percent` or
  `transfers_in_event` in `expected_points.py` or `minutes_model.py`).
  - The columns exist in `fact_player_season_stats` (verified: `schema/0001_core_schema.sql`,
    `schema/0014_priority4_transfer_momentum.sql`).
  - `fact_player_season_stats.status` holds FPL's availability code per gameweek (verified:
    `minutes_model.compute_player_historical_components` docstring, statuses 'i', 's', 'u', 'n').
  - Historical coverage of the transfer and status columns is unverified [A3].
- **Projection bias by price band, pre-promotion** (both seasons pooled, realized − projected per
  player-GW, control run 38167650902 on 60d3e0e, before the own substitute rate went live, verified):

  | band | total | goals | assists | bonus/other |
  |---|---|---|---|---|
  | <£5.0m | −0.158 | | | |
  | £5.0–7.0m | +0.062 | | | |
  | £7.0–9.0m | +0.272 | | | |
  | £9.0m+ | +0.175 | −0.177 | +0.169 | +0.182 |

- **Captaincy, pre-promotion** (same run): 6.42 pts/GW before doubling; the P95 rule +0.39 ± 0.30;
  the best pick in hindsight 12.87.
- **Optimizer settings.** Risk aversion λ is 0 live (verified:
  `data/recalibration/seeds_lambda_2026-10-01.json`). Guardrails: captain not a goalkeeper
  (`squad_optimizer.py:395`), ≤3 XI players per club (`:405`), a bench-quality floor (verified).
- **Season-sim flags.** `scripts/run_season_sim_arm.py` has its own flag parser and passes only
  `lambda_value`, `role_matches_threshold`, `assist_prior_xa` and `team_strength` into
  `_experiment_versions` (verified: lines 70–77). Every other walk-forward flag must be added there
  explicitly.
- **Rank data.**
  - The rank goal "overall rank top 100k" is declared in `data/dashboard/rank_autopsy.json`.
  - Its `model_optimal` is the from-scratch oracle pick, and `estimated_rank` is a single-GW
    projection.
  - `fact_rival_squad_sample` stores per sampled entry: `entry_id`, `league_rank`, picks, captain
    and multiplier. It has no season totals or history.
  - `rank_tracking.yml` runs `scripts/run_rival_sample_ingestion.py` with no arguments.
    `rival_squad_sample.yml` passes a legacy argument.
  
  (verified: those files and `schema/0017_priority10_phaseA_rival_squad_sample.sql`)
- **Prerequisites on master.**
  - PR #260 put per-gameweek `squad_points` in the summary and `--rule confirm` squad-point pairing
    on master (verified: merged 2026-10-11). The own substitute rate is live (PR #259).
  - PR #260 also carried a *pre-revision draft* of this plan. This revision (critique rounds 2 and 3)
    lands with PR #261. Phase 0 checks which version the worktree has.
- **The live from-scratch solve carries an EO template anchor.**
  - `scripts/run_ingestion.py:300–309` passes `risk_posture_params` v2 (`eo_weight_kappa` 0.12)
    and `ownership_params` v1 into `squad_optimizer.run`. The live model team's GW1 squad came from
    that solve (verified: `run_model_team.py` docstring).
  - The walk-forward's `solve_risk_posture_params_version` defaults to None, so backtests run
    without it (verified: `backtest.py:1254`).
  - **Decision:** it stays as is. Phase 3 lists it as a named cause in the chain, since live has it
    and the harnesses don't. Levers and Req 9 are measured without it, in the walk-forward.
- **The walk-forward cannot run 2026-27 today.** `ALL_SEASON_GAMEWEEKS` hard-codes 2024-25 and
  2025-26 (verified: `backtest.py:1228`), so `--seasons 2026-2027` walks zero steps.
- **mypy is advisory in CI**, with pre-existing errors (verified: `.github/workflows/tests.yml`
  `continue-on-error: true`).

## Scope (v1)
The projections and selection layer, judged on fair yardsticks:
1. Template and live yardsticks, and the top-100k cutoffs.
2. A reconciliation of backtest pace against live pace that sets M (Phase 3).
3. A zero-build batch: two prior near-miss arms, plus their season-sim flags.
4. A crowd prior from ownership and transfer momentum.
5. A position × price-band bias correction.
6. Conditional levers: per-player residuals, the optimizer's curse, crowd captaincy.
7. Live promotion of each passing lever.
8. Planner and chip re-tuning: after G2 holds, or earlier if Phase 3 finds a large evolving loss.

## Out of Scope & Parked Items
- **Elite (top-10k) ownership as a feature.** The rival sample exists only for 2026-27, so it
  cannot be backtested. Trigger: a full season of rival picks is stored.
- **New data sources: expert picks, articles, predicted line-ups.** They must pass the backtest.
  Trigger: a lever round shows the gap is team news the crowd has and FPL data lacks.
- **A rewrite of the EP model's structure.** Trigger: every lever here fails rule R6.
- **The app UI**, beyond keeping `index.html`'s leaderboard reading the right row (Req 3).

## Approach
**Measure fairly first.**
- **Oracle against oracle.** The availability-gated template squad joins the walk-forward, paired
  per GW over both seasons. Its points don't depend on the model, so an arm's template gap moves by
  its gain over control.
- **Evolving against reality.** The season sim runs as one continuous GW2–38 squad, and the live
  team is judged on cumulative net points against the real cutoff.
- **Reconcile.** Phase 3 measures each step of the chain (oracle model → season sim → live) and
  sets M.

**Then pull levers, one per arm, each declared before it runs.** Each is opt-in, behind its own
param family and flags in both `run_walkforward.py` and `run_season_sim_arm.py`.

- **Z (near-miss re-judge).** `--no-goal-anchor` and `--no-rate-price-anchor` under rule R6. In
  2025-26 alone they were +1.00 ± 0.83 (MAE +0.008) and +1.32 ± 1.01 (MAE +0.011, above the 0.01
  tolerance) (verified: `docs/reports/2026-10_premium_and_form_returns.md`).
  `run_season_sim_arm.py` needs both flags added.
- **Where L1–L3 apply.** Inside `expected_points.run()`, after each fixture's components are
  summed, the adjustment is added to `ep_total` as its own column `ep_crowd_adj` (L1), `ep_band_adj`
  (L2) or `ep_player_adj` (L3). It is split across a double gameweek's fixtures in proportion to
  their EP. The goal, assist and appearance components stay untouched. So `ep_outputs.ep_total`,
  rule R6(ii)'s MAE and the band residuals all see it.
- **L1, crowd prior.** EP gets a per-gameweek additive adjustment from a ridge regression of
  residuals (realized − pre-adjustment EP) on crowd features.
  - **Features, all from row gw = G−1:**
    - `own_pct`: ownership percentile within position;
    - `log1p(selected_by_percent)`;
    - `xfer_pct`: the percentile of `transfers_in_event − transfers_out_event` among all players in
      that row.
  - **No band terms and no intercept.** A mean shift per position and band is L2's job. L1 only
    adds what the crowd knows beyond price. If A3 finds either season lacks transfer columns,
    `xfer_pct` is dropped from every fold.
  - **Coefficient folds by the season scored or replayed:** 2025-26 steps and 2025-26 replays use
    the 2024-25 fit; 2024-25 steps and replays use the 2025-26 fit. Only 2026-27 (live and the
    kill-switch) uses the both-season fit.
  - **Live** reads the same row-G−1 semantics: the latest completed GW's row, not the current
    bootstrap.
- **L2, band bias.** An additive mean residual per (position, price band), with the same folds,
  shrunk by n/(n + 500). It is fitted on residuals from the *current* control. If L1 is live, L2 is
  fitted after L1, so nothing is double-corrected.
- **Stacked folds.** When a later lever is fitted on season s's residuals, every earlier promoted
  lever's adjustment in those residuals uses season s's *own* fit, not the fit that scores s. This
  stops a fit on 2025-26 leaking into 2025-26's scoring through a later lever's training data. The
  `--export-residuals` artifact writes the pre-adjustment EP and each promoted lever's own-season
  adjustment separately. It differs from B3's failed band *conversion*
  (−2.27 ± 0.97, scaled goal and assist rates by realized/xG). B3's failure is a stated risk.
- **L3, per-player residual (conditional).** A player's own mean residual over his last 6 GWs
  before G, shrunk n/(n+6). It needs point-in-time EP history in each harness (Phase 7 defines the
  source or drops L3).
- **L4, optimizer's curse (conditional, flag `--selection-shrink κ`).** Before the solve,
  μ′ = μ − κ·sd·(1 − s). Here sd is the player's GW points SD from the uncertainty model, and s is
  his `selected_by_percent` divided by the highest `selected_by_percent` in his position (row G−1),
  so s ∈ [0, 1].
- **L5, crowd captaincy (conditional, flag `--crowd-captain δ`).** After the solve, re-pick the
  captain among the XI's outfield players by argmax(EP + δ·EO/100). It is pre-screened as a free
  summary counterfactual (Req 2).
- **Flag names:** `--crowd-prior` (L1), `--band-bias` (L2), `--player-residual` (L3),
  `--selection-shrink` (L4), `--crowd-captain` (L5), each with `--no-<flag>` once live.

L4 and L5 pull toward the template, so they are risk controls, not the route to the margin. Req 9
guards every arm against collapsing into the template.

**Residual pipeline (for L1–L3).**
- When `--export-residuals` is passed, the walk-forward job uploads a `residual-features` artifact
  (Req 12): per player-GW, realized points, pre-adjustment EP, features and season.
- Before each residual-fitted lever, re-export from the *current* control, which includes every
  promoted lever.
- `scripts/fit_residual_lever.py` fits each fold from that artifact in the sandbox. The
  coefficients are committed as `data/recalibration/<lever>_coeffs_<date>.json` and named in the
  batch declaration.
- One shared apply function reads that file in every harness.

**PROMOTE procedure.** It runs as soon as an arm passes rule R6 and Req 9. If two arms in a batch
pass, the higher m wins and the other is re-judged against the new control.
1. Season-sim check (Req 7): control, control_b and the arm in one job on one runner.
2. If m ≥ 0, apply the live-switch pattern (Req 8). Full suite green, then the PR, merge,
   `scheduled_pipeline.yml`, and a projections before/after diff.
3. If m < 0, the arm is not promoted. Record it, then ask the owner whether to promote anyway
   (the owner's standing instruction: ask before merging a winner with m < 0; A8). Meanwhile the
   programme continues with the next lever.
4. Start the kill-switch clock (Phase 11).

Plan additions on top of the owner's R6, tagged so the owner can drop them at approval
[assumed: they protect R6's intent]:
- the Req 9 template-collapse block;
- Phase 10's planner rule m − SE ≥ 0 on season-sim net points (R6's season-sim check alone is
  m ≥ 0, but planner arms have no walk-forward stage before it).

A promoted lever becomes part of every later control.

Executor's choice:
- internal function layout and new module names (follow the `fpl_quant/<topic>.py` pattern);
- the ridge implementation (closed-form numpy is enough);
- artifact formats (CSV or Parquet).

## Requirements
- **Req 1.** WHEN a walk-forward step is scored THE SYSTEM SHALL record
  `template_squad_realized_points` (availability-gated) and
  `template_squad_realized_points_ungated`, plus `beats_template_points_delta` against the gated
  one.
  - The pool is built inside the same `asof_scope()` as the model's candidate pool. It excludes
    players whose latest row with gw ≤ G−1 has `status` in ('i', 's', 'u', 'n').
  - It is solved and scored as `beats_baseline` does.
  - Acceptance:
    - unit tests: a known template squad; captain = the most-owned outfield XI player [A5]; a
      suspended player excluded by the gate;
    - both seasons' summaries carry the fields every GW;
    - the added runtime per run is recorded.
- **Req 2.** WHEN a summary is written THE SYSTEM SHALL include, per gameweek, `squad_points`,
  `template_points` and diagnostics:
  - XI overlap count with the gated template XI;
  - `non_template_points`: the model XI's realized points, with multipliers, from players not in
    the template XI;
  - captain realized points against the most-owned captain's;
  - a captain counterfactual `ep_plus_eo` (argmax EP + 1.0·EO/100 among outfield XI);
  - £ spent per price band for both XIs;
  - realized − projected for the model XI and for the template XI under the model's EP.
  
  Acceptance: unit tests on a seeded run.
- **Req 3.** WHEN `export_leaderboard.py` runs THE SYSTEM SHALL append two rows after the existing
  ones, both with `is_baseline: true`. That keeps the evolving model the first non-baseline row for
  `index.html:6810`.
  - **"model (oracle)"**, `kind: "model_oracle"`. For each GW in the window, the model's fresh,
    hit-free squad: `backtest.run_gameweek_step` with the export's own `param_versions` (the live
    switches, λ, captain rule and bench floor the season sim gets), scored by
    `_realized_solve_points`. So it is the walk-forward's model squad, on 2026-27.
  - **"ownership template (available)"**, `kind: "template_gated"`. Req 1's gated template for the
    same GWs.
  - Every `is_baseline` row, including these two, carries numeric `total_points`, `n_gameweeks`,
    `n_gameweeks_skipped` and `model_beats_baseline_by`. That key keeps its meaning: evolving model
    total − this row's total. `index.html:6823` calls `.toFixed(1)` on it for every baseline row.
  - Each oracle baseline row also gains `oracle_model_beats_baseline_by` (model (oracle) total −
    this row's total).
  - The crowd row's display name becomes "synthetic EO crowd (not FPL's average)".
  - Acceptance:
    - a unit test on row order and keys;
    - a unit test that every `is_baseline` row has numeric `total_points` and
      `model_beats_baseline_by`;
    - the next weekly `leaderboard_latest.json`, rendered by `npm test`'s leaderboard check if one
      exists, else by loading the page.
- **Req 4.** WHEN `rank_tracking.yml` runs THE SYSTEM SHALL fetch each sampled entry's FPL
  `entry/{id}/history` and derive, in the job:
  - the 2024-25 and 2025-26 rank-100k season totals, from `past`;
  - the 2026-27 per-GW cumulative cutoff, from `current`;
  
  by interpolating total points against overall rank across the sampled entries.
  - Only the derived cutoffs are persisted, as an additive key `top_100k_cutoff` in
    `data/dashboard/rank_autopsy.json`. No per-entry history is retained beyond the run (the
    rival-sample retention rule).
  - Acceptance: a test on canned JSON; the report records the cutoffs and the number of entries
    bracketing rank 100k.
- **Req 5.** WHEN `compare_arms.py control arm --rule r6` runs THE SYSTEM SHALL return PASS only if:
  - (i) paired `squad_points` over every GW of every season gives m − SE ≥ 0;
  - (ii) each season's EP MAE ≤ control + 0.01.
  
  It also:
  - refuses on different cached DBs, different seasons, an incomplete run, or any season with
    fewer than 2 paired GWs;
  - prints both runs' gated and ungated template gaps, mean overlap, the change in
    `non_template_points`, the number of changed GWs and the SE.
  
  Acceptance: unit tests for PASS, each FAIL and each refusal.
- **Req 6.** WHEN a lever flag is given THE SYSTEM SHALL mint or reuse an immutable param version.
  Without the flag, the EP output is byte-identical to the control. Acceptance: a flag test per
  lever in both scripts, plus a control-identity test.
- **Req 7.** WHEN an arm passes rule R6 THE SYSTEM SHALL run `season_sim_arms.yml` for 2025-26 with
  chunks `2-38` and arms `control:`, `control_b:` and `arm: <flags>`.
  - All arms run in one job on one runner, via a new workflow input `same_runner: true`.
  - **Runtime:** about 40 min per arm for GW2–38 (two 17–20 min halves). So control, control_b and
    up to 3 arms take about 200 min, inside the 345-min step limit. A batch never exceeds 5 arms
    in one job.
  - The check passes on paired net points, arm − control, m ≥ 0, over complete runs only.
  - control_b − control is recorded as the noise floor.
  
  Acceptance: the result and floor are in the report before the promotion PR.
- **Req 8.** WHEN a lever is promoted THE SYSTEM SHALL follow the live-switch pattern:
  - a v1 seed called from `backtest.materialize_confirmed_seeds()` and `scripts/run_ingestion.py`;
  - a `RECALIBRATABLE_VERSION_ARGS` entry;
  - every call site of the affected function passing it;
  - the `REQUIRED` lists in `tests/test_live_switch_wiring.py`;
  - a `--no-<lever>` flag in both scripts.
  
  Acceptance: the wiring test and the full suite pass.
- **Req 9 (template-collapse guard).** WHEN an arm is judged THE SYSTEM SHALL block promotion if
  mean XI overlap with the gated template rises by ≥ 1 player/GW and the paired change in
  `non_template_points` is ≤ 0.
  - Mean overlap ≥ 9/11 stops the programme for owner review.
  - Acceptance: printed by `--rule r6`; a unit test for each branch.
- **Req 10.** Every batch is declared in `docs/reports/2026-10_beat_the_template.md` (arms, flags,
  coefficient files, hyperparameter values, rule, harness) and committed BEFORE dispatch. Results are
  appended after. Acceptance: the git history shows the declaration commit precedes the run ids.
- **Req 11.** WHEN a residual-fitted lever is on THE SYSTEM SHALL:
  - leave a player's EP unchanged when he has no feature row;
  - read features from rows gw ≤ G−1 (gw ≤ G−2 if A2 fails);
  - pick the coefficient fold from the season being scored or replayed (2024-25 ↔ the 2025-26 fit;
    2025-26 ↔ the 2024-25 fit; 2026-27 → the both-season fit);
  - apply it through one function in the walk-forward, the season sim and live, to every
    planning-horizon GW.
  
  Acceptance:
  - a leak test (changing row G's ownership doesn't change step G's EP);
  - a missing-data test;
  - a fold-selection test per season;
  - an identical-adjustment test across the three call paths.
- **Req 12.** WHEN `run_walkforward.py --export-residuals` runs THE SYSTEM SHALL write
  `residual_features.csv` (per player-GW: season, gw, player_uid, position, price band, realized
  points, pre-adjustment EP, each promoted lever's own-season adjustment, each feature), and `branch_walkforward.yml` SHALL upload it as
  `residual-features`. Acceptance: a unit test on a seeded run, and the artifact present on a real
  dispatch.

## Key Decisions
- **Crowd information may feed projections** (user).
- **Rule R6** (user):
  - m − SE ≥ 0 on paired squad points over both seasons;
  - EP MAE ≤ control + 0.01 per season;
  - a season-sim check with m ≥ 0 before live.
  
  R5's target and breakout checks are reported only. A single passing arm goes straight to PROMOTE
  (the owner's standing instruction; A8).
- **The finish line is top-100k** (user), translated as G2 (M from Phase 3's formula) [A10], G3
  (season sim at the cutoff pace) and G4 (live cumulative at the cutoff, judged at GW38).
- **The yardstick is the availability-gated template** (model-free, row G−1 `status`). The ungated
  template is reported alongside so the gate's effect is visible.
- **The noise floor.** Per arm, SE = sd(paired diff)/√74. It ranges from about 0.4 (an arm
  changing few squads, D2) to about 1.2 (λ arms changing most squads; seeds_lambda note) [A6].
  - Each arm records its changed-GW count and SE.
  - A null arm passes m − SE ≥ 0 with probability ≈ 16%, hence the caps.
  - **Replicate rule.** When an arm's |m − SE| ≤ 0.1, the control and the arm are each
    re-dispatched once on the same commit before any verdict.
    - The verdict is rule R6 on the per-GW mean of the two runs' (arm − control) differences
      (n = 74 GWs, SE from those means). The MAE check uses the mean of the two runs' MAEs.
    - The replicate counts as one arm toward the cap; its control does not.
    - If the two controls' squad points differ, that is recorded as measured nondeterminism.
  - Why: walk-forwards are not deterministic across runners. Two equivalent controls differed in
    3 of 74 GWs (paired SD 1.07, SE ≈ 0.12) through small team-strength fit differences (A9).
- **Declared hyperparameters** (single values, fixed before any run):
  - **L1:** `ridge_alpha` 10 on standardized features; `min_rows` 2,000 player-GWs per fold; the
    adjustment is clipped to ±1.5 pts/GW; features as in the Approach.
  - **L2:** `pseudo_rows` 500.
  - **L3:** 6 GWs, shrink n/(n + 6), players with ≥ 900 league minutes in the current plus previous
    season.
  - **L4:** κ ∈ {0.25, 0.5} (two arms).
  - **L5:** δ = 1.0 pt per 100% EO.
  
  These values are [assumed] defaults with no fitted basis. Fixing them in advance prevents tuning
  after the results.
- **The walk-forward harness for every lever arm** is `branch_walkforward.yml` with no `--seasons`,
  dispatched with `actions_run_trigger` (method `run_workflow`) on the pushed worktree branch's ref
  [A4].
  - Control and arm use different args, so they don't cancel each other. They are dispatched
    back-to-back.
  - Both are re-dispatched if their `db_cache_key` values differ.
- **Batches have at most 3 arms plus control.**

## Data & State Changes
- **New param families**, numeric and versioned: `crowd_prior_params` (which coefficient file, plus
  `ridge_alpha`, `min_rows`, `clip`), `band_bias_params`, `player_residual_params`,
  `selection_shrink_params`, `captaincy_crowd_params`.
- **Coefficient files** go under `data/recalibration/` and are committed per declared fit.
- **No DB schema change for v1.** Cutoffs are derived in-job and published in `rank_autopsy.json`.
  Template metrics go into the existing `backtest_metrics` (additive names).
- **Rollback:** every lever has a `--no-<lever>` flag and a live switch version. Removing its seed
  entry reverts live behaviour.

## Interfaces, Integrations & Credentials
- **GitHub Actions** via the GitHub MCP tools: `actions_run_trigger` (method `run_workflow`) for
  `branch_walkforward.yml`, `season_sim_arms.yml`, `scheduled_pipeline.yml` and `rank_tracking.yml`;
  `actions_list` and `actions_get` for runs, logs (`ARGS:`, `DB_CACHE_KEY:`) and artifacts.
- **FPL's public API** from Actions only: `entry/{id}/history`. The sandbox cannot reach
  fantasy.premierleague.com (verified: `run_rival_sample_ingestion.py` docstring). Use the
  existing request pacing. No credentials.
- **Frozen surfaces:**
  - `data/dashboard/*.json` read by the app: only add keys or rows, and keep row 1 of the
    leaderboard the evolving model;
  - the walk-forward summary keys existing tools read: only add keys.

## Edge Cases & Failure Handling
- No feature row for a player-GW → no adjustment. A fold with fewer than `min_rows` rows → the
  lever is off for that fold, logged.
- `status` missing for a player-GW → the gate does not exclude him (no data is never "unavailable").
  Coverage per season is reported (A3).
- Double gameweeks: the per-GW adjustment is split across fixtures in proportion to each fixture's
  EP. Blank GWs: no fixture, no adjustment.
- A template solve status outside (`optimal`, `timelimit`) → None for that GW. Rule R6 pairs only
  GWs present in both runs.
- Incomplete runs are refused or not judged. Re-run; never judge a partial run.
- An Actions job that dies before any step ran → re-run once. A second failure is real.
- No sampled entries bracket rank 100k → use the nearest bracket and say so.

## Risks, Landmines & Adaptations
- **Point-in-time leakage** through ownership, transfers or status (#214's leak, see the
  `seeds_lambda_2026-10-01.json` note) → Phase 1 verifies [A2]. Req 11 enforces gw ≤ G−1, with a
  fallback to G−2.
- **Leakage through fitted coefficients** → folds by the season scored or replayed (Req 11). The
  both-season fit is used only for 2026-27.
- **A soft yardstick** (a template that starts injured or suspended players) → the availability
  gate, with the ungated template reported alongside.
- **An unfair live yardstick** → the Req 3 oracle row; G4 uses real cumulative points against the
  real cutoff.
- **An inflated or unrepresentative backtest** (+10.4 oracle, +8.1 sim, about +0.6 live) → Phase 3
  measures each step and sets M from the chain. If a cause of backtest optimism is found, it is
  fixed or G3 is haircut before any lever.
- **Season-sim nondeterminism** (±1.81/GW) → all arms on one runner (Req 7), the control_b floor,
  and the live kill-switch.
- **Overfitting by reusing 74 GWs.** Only 2026-27 is out of sample → declaration first, caps, folds,
  replicates near the threshold, and Phase 11's 2026-27 confirmation of every promoted lever and of
  G2.
- **Template collapse** → Req 9. L4 and L5 are classed as risk controls.
- **Concurrent edits** to shared scripts → merge `origin/master` into the worktree branch before
  every PR. Never force-push over others' commits.

## Assumptions Ledger
| ID | Assumption | Basis | Blast radius if wrong | Check |
|----|------------|-------|-----------------------|-------|
| A1 | Sampled entries' `entry/{id}/history` `past` arrays give per-season totals and ranks enough to interpolate rank 100k | [assumed: FPL API shape, unverified] | G3 has no target | Phase 2, Req 4 test on a real response |
| A2 | Row gw = G−1 values (`selected_by_percent`, transfers, `status`) are knowable before G's deadline | [assumed] | L1 and the gate leak; fallback G−2 | Phase 1. Historical rows carry no snapshot time, so test on 2026-27, which has the same source semantics. A2 holds if, for each finished 2026-27 GW G with a live pipeline run before G's deadline (`_ingested_at` < deadline), row G−1's values equal that run's bootstrap values for ≥ 95% of players. Fewer than 2 such GWs, or below 95%, means inconclusive: use G−2 everywhere. Plus the Req 11 leak test |
| A3 | Transfer and `status` columns are populated for 2024-25 and 2025-26 | [assumed] | L1 loses features; the gate is weaker | Phase 1: non-null counts per season |
| A4 | Dispatching on the pushed worktree branch's ref runs that branch's code | [assumed: standard GitHub behaviour] | Arms run master's code | Phase 1: run `head_sha` equals the branch head |
| A5 | The template captain is the most-owned outfield XI player | [assumed: `solve` maximises mu with the captain doubled; captain-not-GK verified] | The template baseline is slightly off | Phase 1 unit test |
| A6 | The per-arm paired SE over 74 GWs is 0.4–1.2 | verified: D2 ±0.39–0.74; λ arms ±1.21 | Rule R6's power | Recorded per arm |
| A7 | One GW2–38 season-sim job uses both chip halves correctly | [assumed: the sim tracks chip sets by GW] | G3 is mis-measured | Phase 2: inspect chips used in a GW2–38 control |
| A8 | The executor has the owner's standing authority: merge green PRs; promote an arm that passes rule R6 and the season-sim check; ask the owner before merging any winner with m < 0 | (user) standing instructions | It would need to stop for approval | none |
| A9 | Walk-forward controls on the same commit, DB and args are deterministic | **False across runners** (verified 2026-10-12: runs 38185477854 and 38185480048, same DB, code differing only in a minutes change that left every minutes metric identical). The team-strength fit differed slightly (`match_log_lik`), and 3 of 74 GWs' squad points changed: paired mean +0.14, SD 1.07, so control-vs-control SE ≈ 0.12. D1 and D2's controls happened to match | Control noise adds about 0.12 to each arm's SE | Replicates near the threshold (Key Decisions); Phase 1 records one control_b walk-forward |
| A10 | "By a lot" means ≥ +2.0 pts/GW over the template, raised to M if the cutoff chain needs more | [assumed: the owner said "by a lot"] | G2's bar | The owner may restate it at approval |
| A11 | The walk-forward can run 2026-27 for the kill-switch | **False today** (verified: `ALL_SEASON_GAMEWEEKS`, `backtest.py:1228`, lists only 2024-25 and 2025-26). [assumed: the cached DB holds settled 2026-27 data, so extending that list is enough] | The kill-switch falls back to the one-runner season sim with control_b | Phase 11, step 1 |
| A12 | `backtest.run_gameweek_step` can score a 2026-27 GW (for Req 3's oracle row) | [assumed: it takes season and GW as arguments] | The oracle row needs its own step loop | Phase 2, Req 3 test |

## Open Items (none blocking)
- **Elite ownership for 2026-27.** Proceed without it.
- **Showing the crowd adjustment in the app.** The app's EP field *schema* stays unchanged; its
  values change with each promotion (hence the projections diff). Proceed without a UI change.

## Verification
- `PYTHONPATH=src python -m pytest tests/ -q` is green, `ruff check .` is clean, `npm test` passes
  (CI runs it), and `mypy src/` shows no new errors versus master, before every PR.
- `PYTHONPATH=src python scripts/compare_arms.py control.json arm.json --rule r6` prints PASS/FAIL
  plus the gaps, overlap, `non_template_points` change, changed GWs and SE.
- Batch sections list the declaration commit, run ids, the `ARGS:` mapping, `db_cache_key`, the
  coefficient file and the verdict.
- Season-sim batches record control, control_b and the arm, plus the floor.
- Live: a `projections_latest.json` diff per promotion. `app_model_team.json` cumulative points
  against `rank_autopsy.json` `top_100k_cutoff` per GW.
- The owner confirms done from the report's headline table:
  - model − gated template against M̄ (G2);
  - season-sim pace against the cutoff pace (G3);
  - live cumulative points against the cutoff (G4).

## Build Phases
- [ ] Phase 0: Set up the worktree
      Done when: in `../FPL-Analyser-btt`, branch `claude/beat-template` exists on origin;
      `grep -n squad_points scripts/walkforward_summary.py` matches; and
      `grep -n "Replicate rule" docs/plans/2026-10_beat_the_template.md` matches (this revision,
      not PR #260's draft).
      Steps:
      - `git fetch origin master && git worktree add ../FPL-Analyser-btt -b claude/beat-template origin/master`
        (or the host's EnterWorktree).
      - If the plan grep fails, PR #261 has not merged yet. Merge it if green (A8), then
        `git merge origin/master`. Otherwise run
        `git checkout origin/claude/sharp-noether-3w5b3t -- docs/plans/2026-10_beat_the_template.md`
        and commit.
      - `git push -u origin claude/beat-template` (a dispatch needs the ref on origin).
      - Create `docs/reports/2026-10_beat_the_template.md` with rule R6, G1–G4, the declared
        hyperparameters, the caps and the stop rule, then commit and push.
      Covers: Req 10

- [ ] Phase 1: Build the oracle yardstick, the residual export and `--rule r6`
      Done when:
      - Req 1, 2, 5, 9 and 12 tests pass and the PR is merged;
      - a both-season control walk-forward on the branch (with `--export-residuals`) records the
        gated and ungated template gaps with SE, the overlap and the diagnostics;
      - A2, A3, A4 and A5 are resolved in the report.
      Steps:
      - Write failing tests first, then implement. Reuse `_naive_candidate_pool`,
        `squad_optimizer.solve` and `_realized_solve_points` as `beats_baseline` does, with the
        status gate. The control's EP must not change.
      - Check A2 and A3 with a diagnostic script in Actions on the cached DB (pattern:
        `.github/workflows/diagnose_premium_returns.yml`).
      - Dispatch the control (args `"--export-residuals"`) on the branch and confirm A4 from its
        `head_sha`. Once it finishes, dispatch the same control again as control_b (a concurrent
        duplicate would cancel the first) and record their paired squad-point SD (A9).
      Covers: Req 1, Req 2, Req 5, Req 9, Req 12; checks: A2, A3, A4, A5
      Critique: important phase. After building, hand the result plus this phase's Done-when and
      Covers lines to a blind critic (a fresh sub-agent if available, else a fresh-eyes reread). The
      critic defaults to FAIL and returns PASS/FAIL, strengths, weaknesses and fixes. Apply blocker
      fixes and repeat, max 3 rounds. Done only on PASS.

- [ ] Phase 2: Build the live yardsticks, the cutoffs and the one-runner season sim
      Done when:
      - Req 3, Req 4 and the Req 7 `same_runner` input are merged with tests;
      - `rank_autopsy.json` carries `top_100k_cutoff` for 2024-25, 2025-26 and 2026-27 to date;
      - a GW2–38 one-runner season-sim run (`control: ; control_b:`) completes, and its chips used
        are recorded (A7).
      Steps:
      - Implement Req 3 (row order and new keys), Req 4 (history fetch and cutoff derivation inside
        `rank_tracking.yml`'s job), and the `same_runner` input (arms sequential in one job).
      - Dispatch `rank_tracking.yml` and `season_sim_arms.yml` (2025-26, chunks `2-38`,
        `same_runner: true`).
      Covers: Req 3, Req 4, Req 7; checks: A1, A7, A12

- [ ] Phase 3: Reconcile, set M, decompose, and declare the lever order
      Done when: the report has the following, committed before any lever run:
      - the chain table, each row as net pts/GW against FPL's average and against that season's
        cutoff pace:
        - oracle model and gated template, for 2024-25 and 2025-26;
        - season sim control and control_b, for 2025-26 only;
        - for 2026-27: the "model (oracle)" row, the leaderboard sim and the live team;
      - a measured contribution in pts/GW for each named cause of the differences, including the
        live-only EO anchor (Current State);
      - the SE of the live 5+ GW sample and of O;
      - C_s, T_s, E, O, O⁺, M_2024-25, M_2025-26 and M̄ by the G2 formula;
      - the decomposition table;
      - the ordered lever queue.
      Steps:
      - Compute E, O and O⁺ and each M_s exactly as G2 defines them, then M̄.
      - **Branches, declared:**
        - If a cause of backtest optimism is found (e.g. a leak or a free restart), fix it or haircut
          G3 before any lever.
        - If the evolving loss is ≥ 3.0 pts/GW, run Phase 10 now, alongside the lever queue.
        - If the Phase 1 control already has m ≥ M̄ with m − SE > 0, G2 is met: skip Phases 4–9 and
          go to Phase 10.
      - **Lever order: Z, then L1, then L2.**
        - L3 only if £9.0m+ realized − projected is still ≥ +0.1 after the levers that passed.
        - L4 only if the model XI's realized − projected is ≤ −0.3 pts/player-GW and the template
          XI's is ≥ 0.
        - L5 only if `ep_plus_eo` beats the model's captain by ≥ +0.3 pts/GW with m − SE > 0.
      Covers: Req 7, Req 10
      Critique: important phase (it sets M and what "fair" means). Run the blind-critic loop as in
      Phase 1 on the write-up.

- [ ] Phase 4: Re-judge the near misses (Z)
      Done when: `--no-goal-anchor` and `--no-rate-price-anchor` exist in `run_season_sim_arm.py`
      (with tests), and the batch (control and both arms, both seasons) is declared, run, judged
      with `--rule r6` and recorded. Any PASS has completed PROMOTE (the tie-break rule applies).
      Steps:
      - Add both flags to `run_season_sim_arm.py`'s parser and Namespace, with tests.
      - Declare, commit, dispatch the three runs back-to-back, judge, then PROMOTE on PASS.
      Covers: Req 5, Req 6, Req 7, Req 8, Req 9, Req 10

- [ ] Phase 5: L1, the crowd prior
      Done when: declared, run, judged with `--rule r6` and recorded. A PASS has completed PROMOTE;
      a FAIL records the reason and the family count.
      Steps:
      - Write failing tests first: the four Req 11 tests, the `min_rows` gate, and the flags in both
        scripts (Req 6).
      - Dispatch the current control with `--export-residuals`. Fit the 2024-25, 2025-26 and
        both-season folds with `scripts/fit_residual_lever.py` using the declared values, and commit
        `data/recalibration/crowd_prior_coeffs_<date>.json`.
      - Implement the shared apply function and `--crowd-prior` in both scripts. Declare, commit,
        dispatch the control and arm, judge, and PROMOTE on PASS.
      Covers: Req 6, Req 9, Req 10, Req 11, Req 12
      Critique: important phase (leakage). Run the blind-critic loop on the fit, the fold selection
      and the apply code.

- [ ] Phase 6: L2, the position × price-band bias correction
      Done when: declared, run, judged with `--rule r6` and recorded. A PASS has completed PROMOTE.
      Steps:
      - Re-export residuals from the current control, which includes L1 if promoted. Fit folds with
        n/(n+500) shrinkage and commit the coefficient file.
      - Tests (folds, shrinkage, paths, flag), then declare, run `--band-bias` against control,
        judge, and PROMOTE on PASS.
      Covers: Req 6, Req 9, Req 10, Req 11, Req 12

- [ ] Phase 7: L3, the per-player residual (only if Phase 3's rule queued it)
      Done when: declared, run, judged and recorded, or the report says why it was skipped,
      including when any harness lacks point-in-time EP history.
      Steps:
      - EP-history sources, named 2026-10-20 (report, "Phase 7 prep"):
        - walk-forward = earlier steps of the same `backtest_run_id`
          (`backtest_gameweek_steps.ep_model_version`);
        - season sim = each GW's `transfer_plan_runs.ep_model_versions[gw]` within its single
          GW2–38 job (never the default 2-19/20-38 chunks), the bootstrap GW's from its solve;
        - live = a replay of each needed 2026-27 GW's EP inside `asof_scope()`, because the live
          DB is rebuilt every run and keeps no deadline-time versions. The committed
          `projections_<date>.json` snapshots are a cross-check only. Drop L3 if A11's per-GW
          2026-27 data can't support the replay.
        - One shared function serves all three. Every harness computes residuals from
          pre-adjustment EP (`ep_total − ep_player_adj`) and counts only the current season's
          GWs in the 6-GW window.
      - Tests (leak, shrinkage, paths, flag), then declare, run, judge, PROMOTE on PASS.
      Covers: Req 6, Req 9, Req 10, Req 11

- [ ] Phase 8: L4 and L5, the risk controls (only those Phase 3's rule queued)
      Done when: each queued lever is declared, run, judged and recorded, or skipped with the
      reason.
      Steps:
      - L4: `selection_shrink_params`, μ′ = μ − κ·sd·(1 − s) before the solve. Tests: no effect at
        κ = 0, monotonic in sd, the flag. Two arms (κ 0.25, 0.5); the tie-break applies.
      - L5: `captaincy_crowd_params`, a post-solve re-pick by argmax(EP + 1.0·EO/100) over outfield
        XI. Tests: XI only, the vice unchanged unless it becomes the captain, the flag.
      - Declare, run, judge (Req 9 strictly), PROMOTE on PASS.
      Covers: Req 6, Req 9, Req 10

- [ ] Phase 9: Combine plausible levers, then re-check G2
      Done when: the report records (a) either "no combine candidates" or the combined arm's
      verdict (PROMOTE on PASS; on FAIL nothing changes, since every lever that passed alone is
      already live), and (b) the G2 decision for the current control: model − gated template m ≥ M̄
      with m − SE > 0 (G2 met), or G2 unmet with the queue exhausted or capped, which triggers owner
      review.
      Steps:
      - The candidates are arms with m > 0 that failed only m − SE ≥ 0. Combine them, declare, run
        against a fresh control, judge, then PROMOTE on PASS. The arm counts towards the lever cap.
      - Run `--rule r6` on the latest control summary to print the gaps against M̄, and record the
        decision.
      Covers: Req 5, Req 7, Req 10

- [ ] Phase 10: Re-tune the planner and chips (after G2 holds, or early per Phase 3's branch)
      Done when: each declared season-sim batch (`control: ; control_b: ; arm:`, 2025-26, GW2–38,
      one runner) has a recorded decision: promote if arm − control m − SE ≥ 0 on net pts/GW, else
      reject. At most 6 arms.
      Steps:
      - One existing flag per arm, at most 3 per batch: `--transfer-threshold`, `--multi-transfers`,
        `--chip-timing`, `--chip-wait`, `--chip-option-value`.
      - Declare, run, judge, and promote by Req 8.
      Covers: Req 7, Req 8, Req 10

- [ ] Phase 11: Track live, run the kill-switch, and review at GW19 and GW38
      Done when, for each review gameweek N (every 6 live GWs after a promotion, plus GW19 and GW38),
      the report records:
      - the headline table (G2 gap vs M̄, G3 pace vs cutoff, G4 cumulative vs cutoff with the live
        sample's SE);
      - for each promoted lever, its 2026-27 comparison against `--no-<lever>` (paired m ± SE over
        GW2–N), and the decision to keep it or roll it back (roll back if m + SE < 0);
      - G2's 2026-27 check: "model (oracle)" − "ownership template (available)" per GW from the
        leaderboard (Req 3). Over GW2–N, m − SE > 0 confirms G2 out of sample. m + SE < 0 at GW19
        or later pauses the programme for owner review. Anything between is recorded as
        inconclusive.
      - the programme decision for GW N: continue, pause for owner review, or close (GW38 = final
        report).
      Steps:
      - Step 1 (first use, A11): add the finished 2026-27 GWs to `ALL_SEASON_GAMEWEEKS` (up to the
        last GW FPL marks finished), with a test that `--seasons 2026-2027` walks ≥ 1 step. If the
        cached DB can't support it (A11 fails), the kill-switch is the one-runner season sim on
        2026-27 (`control: ; control_b: ; arm: --no-<lever>`), judged the same way. Record which.
      - After each weekly backtest and rank-tracking run, update the table from
        `app_model_team.json`, `leaderboard_latest.json` and `rank_autopsy.json`
        `top_100k_cutoff`.
      - Dispatch the kill-switch comparison for each promoted lever.
      Covers: Req 3, Req 4, Req 8, Req 10; checks: A11
