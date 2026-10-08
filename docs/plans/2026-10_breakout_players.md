# Plan: Stop under-projecting this season's breakout players

One-line goal: players who have clearly won a starting role or are new to the Premier League this season are projected in line with what they are actually doing, without the model losing squad points or accuracy anywhere else.

## Classification
Track: bug fix. The defect is that the live model under-projects established breakout players. The cause is not yet confirmed, so the plan diagnoses before it changes anything.
Parked secondary asks:
- role-change evidence from articles moving live minutes;
- a club-strength fix, if diagnosis points there;
- the earlier backlog: the planner always holding, stale chip timing, and `TARGET_GAMEWEEK = 1` in ingestion.

## Interview Ledger
- Q1 evidence source → only FPL's own minutes and starts. Article role-change claims stay a flag (accepted, user).
- Q2 promotion rule → the new segment-aware rule in R4 (accepted, user).
- Questions spent: 2, plus the approval question.

## Goal & Success Criteria
- **Measured, not felt.** The walk-forward scoreboard reports a "breakout group" (definition in R2) for 2025-26. For each season it gives the group's size, mean residual (realized − predicted points) and MAE.
- **Promotion only through the rule.** A fix goes live only if `scripts/compare_arms.py` prints PASS for it under R4, against a control run of the current live model on the same cached DB.
- **Live sanity check after a promotion.** Comparing the same future gameweeks before and after the promotion's pipeline run, the named players that the promoted change affects rise. Each rise is measured net of the two reference starters' change over the same gameweeks, so new results and news don't count as the fix. Today their GW6 predictions are Groß 2.51 (#127), Kinsky 0.79, Kostoulas 1.60 (#265) and De Cuyper 2.35 (#151) (verified: `data/dashboard/projections_latest.json` at commit 283f6d8). This is a check, not a target: no fix is tuned to these four players.
- **If nothing passes,** the report says so with numbers and names the next step. Shipping nothing is an acceptable outcome; shipping an untested change is not.

## Current State
- **Season stats** (verified: `data/dashboard/app_players.json`):
  - Groß (Brighton MID, £5.9m): 450 minutes, 3 goals, 4 assists, 47 points.
  - Kinsky (Spurs GK, £4.5m): 450 minutes, 18 points.
  - Kostoulas (Brighton FWD, £5.6m): 353 minutes, 2 goals, 3 assists, 28 points.
  - De Cuyper (Brighton DEF, £5.0m): 423 minutes, 1 goal, 3 assists, 38 points.
  - Three of the four are Brighton (team code 36 in the projections file, BHA in app_players).
- **Live model and settings:**
  - The price anchor on goal and assist rates has been live since #244 (verified: merge commit 8332a40).
  - Rate shrinkage `k_minutes` is 900 (verified: `scripts/run_walkforward.py` docstring "live: 900 since 2026-09-09").
  - The evidence-order minutes start prior has been live since #234 (verified: `src/fpl_quant/minutes_model.py`, `seed_start_prior_params`).
- **Already built but opt-in:**
  - Current-season role blend, via `minutes_model.run(current_season_role_params_version=...)` and the walk-forward flag `--role-matches-threshold N` (verified: `src/fpl_quant/minutes_model.py:1081`, `scripts/run_walkforward.py`).
  - Season-recency weighting on scoring rates: `--rate-current-season-weight`, `--rate-season-decay` (verified: `scripts/run_walkforward.py`).
- **Past results under the old rule:**
  - Role blend: −0.62 ± 1.56 points a gameweek, EP MAE 1.250 vs 1.238 (verified: `docs/reports/2026-10_live_path_diagnosis.md:204`). That test predates the start prior (#234), the club-strength guard (#236) and the anchor (#244).
  - Recency (current season ×2, decay 0.5): +0.62 ± 1.23 and MAE 1.0922 vs 1.0934 (verified: `docs/reports/2026-10_rate_prior.md`). Its effect on breakout players was never measured.
- **Article role-change claims** are logged as a flag and change no number (verified: `src/fpl_quant/minutes_model.py`, `role_change_evidence_flag` at line 732).
- **Data the new metric can use:**
  - `scripts/walkforward_summary.py` builds its groups after the run from `backtest_gameweek_steps`, `ep_gameweek_outputs` (predicted) and `fact_player_season_stats` (realized `event_points` per gameweek; also `expected_goals_per_90`, `expected_assists_per_90`) (verified: `scripts/walkforward_summary.py:54-130`, `schema/0001_core_schema.sql:115-138`).
  - Its `per_gameweek` list holds only `beats_crowd`, `beats_real` and `match_log_lik` per step (verified: `scripts/walkforward_summary.py:236-247`).
  - Per-match starts are in `fact_player_match_stats`: `start_min = 0` means started. That table has no club column (verified: `src/fpl_quant/minutes_model.py:199`, `schema/0001_core_schema.sql:94-113`).
  - A player's club per season comes from the temp table `_player_season_team`, built by `minutes_model._build_player_season_team_map(con, seasons)` (verified: `src/fpl_quant/minutes_model.py:99-105`).
  - Newly promoted clubs are identified by `backtest._is_newly_promoted_team(con, team_uid, season)` (verified: `src/fpl_quant/backtest.py:574`).
  - Step deadlines come from `backtest.gameweek_deadline(con, season, gw)` (verified: `src/fpl_quant/backtest.py:126`).
- **The backtest's data starts in 2024-25** (`FIRST_LOADED_SEASON = "2024-2025"`, verified: `src/fpl_quant/backtest.py:90`). So 2024-25 steps have no earlier season to compare against.
- **Where the live DB lives:** only in the GitHub Actions `duckdb-` cache. Workflows restore the newest one with `actions/cache/restore`, step `id: cache` (verified: `.github/workflows/branch_walkforward.yml:43-50`). This environment cannot rebuild it locally [assumed: the dataset download was refused here earlier - if wrong: run the diagnosis locally].
- **Walk-forward arms:**
  - They run from `branch_walkforward.yml` with `workflow_dispatch` input `args` on any ref (verified: that file).
  - Its concurrency group is `branch-walkforward-<ref>-<args>` with cancel-in-progress (verified: lines 25-27), so a second dispatch with the same ref and args cancels the first.
  - It never saves the DB, so `--resume` cannot work there: `backtest.run` raises without the earlier run's row (verified: `src/fpl_quant/backtest.py:1235-1238`, the workflow has no cache save).
  - Each run uploads a `walkforward-summary` artifact (`walkforward_summary.json`) and takes about 30 minutes (verified: runs 37703804006 and 37703806488, 23:42 → 00:10-00:13 UTC).
- **Deadlines:** GW6 2026-10-10 10:00 UTC, GW7 2026-10-17 10:00 UTC (verified: `data/dashboard/app_fixtures.json`).
- **Merging:** the user lets Claude merge its own PRs once CI is green (user: "please just keep merging new prs").

## Scope (v1)
1. A diagnosis of where the gap comes from for the four named players.
2. A breakout-group metric in the walk-forward scoreboard.
3. A mechanical arm-comparison script that applies the new rule.
4. Up to five predeclared walk-forward arms.
5. Promotion of the single best passing arm through the existing live-switch pattern.
6. A report.

## Out of Scope & Parked Items
- **Article role-change claims moving live minutes** (user, Q1). They stay a flag until `--backtest-evidence` (verified: `scripts/run_walkforward.py:111`) has enough dated claims to test them.
- **A Brighton club-strength fix.** If the diagnosis attributes most of the gap to the club rating, record it in the report as a separate follow-up plan. Club strength has its own guard and its own validation (`docs/reports/2026-10_promoted_club_strength.md`).
- **k_minutes changes:** anchor + k 450 was tested on 2026-10-08 and failed the rule (verified: runs 37703804006/37703806488).
- **The ML shadow:** frozen through GW19 (verified: `research/ml/forward_test/FROZEN_CONFIG.md`). Quant EP changes are allowed because its baseline is "whatever ep_model_version the live pipeline produced".
- **Matching expert projections directly:** that would mean tuning to four players.
- **The planner, chip timing, `TARGET_GAMEWEEK = 1`:** separate earlier backlog.

## Approach
Diagnose first, then measure, then test only predeclared arms, then promote at most one.

1. **Diagnose on the live DB, in CI.** For each named player, split the next gameweek's prediction into minutes and point components, then compare each against what the player is doing this season. Scoring rates are compared to his expected goals and assists per 90 (xG/xA), not his actual goals, so five matches of finishing luck don't pass for a rate problem. Attribute the gap to minutes, scoring rates, club strength, or bonus and defensive points.
2. **Measure the problem in the backtest** with a breakout group that excludes promoted clubs. Without a measure, no fix can be judged.
3. **Arms:**
   - Three existing-flag arms on the current live model (A1-A3 in Key Decisions).
   - Up to two new arms, declared from the diagnosis before any arm results are read.
   - Every batch runs its own control on the same ref at the same time, and every comparison checks both runs used the same cached DB.
4. **Promote** the best passing arm the same way #244 promoted the anchor.

Executor's choice: script internals, output formatting, function names.

## Requirements
- **R1 (diagnosis).** WHEN `diagnose_breakouts.yml` runs on the cached live DB, THE SYSTEM SHALL output a JSON file with:
  - the target gameweek: the first one whose deadline is after the DB's data cutoff, via `compute_ml_shadow._data_cutoff` and `_target_gameweek_from_db`;
  - the ep_model_version used: the newest covering every fixture of that gameweek, via `_ep_versions_for_gameweek` (all three in `scripts/compute_ml_shadow.py`);
  - for Groß, Kinsky, Kostoulas, De Cuyper, and two established starters as reference (Bruno Fernandes, Mbeumo):
    - `p_start_final`, `p_60plus_min` and `weight_own` from that version's minutes model;
    - each `ep_*` component and `ep_total` (from `ep_gameweek_outputs`);
    - the club's attack and defence ratings from the team-strength version used;
    - this season's minutes, starts, goals, assists, `expected_goals_per_90`, `expected_assists_per_90` and points per 90;
    - `in_breakout_group`: R2's rules (a)-(c) applied at the live step. A named player outside the group is reported plainly, because then the backtest gate only tests a proxy for him.
  - Acceptance:
    - every field is present for all six players;
    - each player's `ep_total` matches `data/dashboard/projections_latest.json` for the same gameweek within 0.05, or the JSON explains the mismatch (for example, a different ep version);
    - there is one gap-attribution line per named player.
- **R2 (breakout metric).** WHEN `scripts/walkforward_summary.py` runs, THE SYSTEM SHALL add a `breakout` block for 2025-26 with `n_player_steps`, `mean_resid` (realized − predicted `ep_total`) and `mae`, plus a separate `breakout_promoted` block that does not decide. For 2024-25 it emits `null`: there is no earlier season to judge "new" against.
  - A player-step (season S, gameweek G) is in the group when all of these hold, using only matches before `backtest.gameweek_deadline(con, S, G)`:
    - (a) his club in S had played at least 4 league matches (`fact_match`, `competition = 'Premier League'`), and he started at least 3 of the last 4 (`fact_player_match_stats.start_min = 0`);
    - (b) across seasons before S, he either played fewer than 900 league minutes or started fewer than 40% of the league matches of the clubs he was at (clubs from `_player_season_team`) [A1];
    - (c) his club in S is not newly promoted. Players at promoted clubs go into `breakout_promoted` instead.
  - The summary builds `_player_season_team` first, with `minutes_model._build_player_season_team_map`. Predicted points come from an inner join to `ep_gameweek_outputs`, so blank gameweeks drop out and double gameweeks are summed, matching realized `event_points` per gameweek.
  - Acceptance: a unit test on a synthetic DB classifies exactly the first two of these as breakout:
    - a new-to-league starter;
    - a backup turned starter;
    - an established starter;
    - a starter at a promoted club, who lands in `breakout_promoted`;
    - a player with only 3 club matches played.
- **R3 (comparison script).** WHEN `python scripts/compare_arms.py control.json arm.json` runs, THE SYSTEM SHALL print PASS or FAIL with each R4 component and its numbers.
  - It refuses with a clear error when either file has `progress.complete` false, or when either `db_cache_key` is missing or empty, or the two differ.
  - Acceptance: unit tests for a pass, each single failing reason, and both refusals.
- **R3b (cache key).** WHEN `branch_walkforward.yml` writes the scoreboard, THE SYSTEM SHALL include `db_cache_key` (the `steps.cache.outputs.cache-matched-key` value, passed as an environment variable) in `walkforward_summary.json`.
- **R4 (promotion rule, user Q2).** An arm passes only when all three hold against a control run on the same `db_cache_key`:
  - (i) **Points level or better.** Pair `per_gameweek[].beats_real` for season `2025-2026` on the gameweeks that are non-null in both files. FPL's real average is identical in both runs, so the difference is the squad-point difference. Let the mean be m and its standard error SE = sample sd / √n. It passes when m ≥ −0.25 and m + SE ≥ 0 [A2].
  - (ii) **Error no worse.** `headline_by_season["2025-2026"]["ep_total_calibration_mae"]` is at most control's + 0.002.
  - (iii) **Breakout under-prediction cut by a third.** Control's `breakout.mean_resid` for 2025-26 is > 0, and the arm's is at most two thirds of it.
  - 2024-25 numbers are reported but do not decide.
- **R5 (promotion).** WHEN an arm passes, THE SYSTEM SHALL make its setting live for every caller:
  - add a seed and a `RECALIBRATABLE_VERSION_ARGS` entry (the pattern of #234/#236/#244). For A2 the family is already live: seed `rate_prior_params` v2 with the arm's values and update `expected_points.LIVE_RATE_PRIOR`. The existing `--rate-current-season-weight 1 --rate-season-decay 1` flags are then its off switch;
  - extend `tests/test_live_switch_wiring.py` for any new version argument;
  - add a `run_walkforward.py` flag that turns it off.
  - Acceptance: the full test suite and `ruff check .` pass, and in the next pipeline run the named players' predictions move as the success criteria describe.
- **R6 (report).** `docs/reports/2026-10_breakout_players.md` records:
  - the diagnosis;
  - control's breakout metric;
  - every arm's R4 numbers, with the raw paired m ± SE;
  - the decision.

## Key Decisions
- **Signals:** FPL minutes and starts only (user).
- **Gate:** R4 (user, Q2), with the numeric reading of "level or better" in [A2].
- **Arms, at most five in total.** A1-A3 are declared now; any A4-A5 are declared in the Phase 2 PR description before Phase 3 is dispatched.
  - A1 `--role-matches-threshold 4`. This re-tests the existing current-season role blend: since #234 its minutes model already starts from this season's record, so the earlier failure may not carry over. It must still pass R4(ii).
  - A2 `--rate-current-season-weight 2 --rate-season-decay 0.5`
  - A3 both A1 and A2 together
- **Live-DB work runs in CI, not locally** [assumed: see Current State - if wrong: run it locally].

## Data & State Changes
- No schema change: the breakout metric is computed from existing tables (verified: `scripts/walkforward_summary.py`).
- A promoted arm adds a parameter version (seed / `params.write_param`), as #244 did.
- Rollback: revert the promotion PR. The switch is a version argument, and `None` restores the old behaviour.

## Interfaces, Integrations & Credentials
- New workflow `.github/workflows/diagnose_breakouts.yml`:
  - `workflow_dispatch`;
  - restores the newest `duckdb-` cache like `branch_walkforward.yml`, and fails when `cache-matched-key == ''`;
  - runs `PYTHONPATH=src python scripts/diagnose_breakouts.py`;
  - uploads `diagnose_breakouts.json` as an artifact named `diagnose-breakouts`;
  - commits nothing.
- A changed `branch_walkforward.yml` Scoreboard step passes `DB_CACHE_KEY: ${{ steps.cache.outputs.cache-matched-key }}`.
- No new secrets. FPL data comes from the cached DB only.

## Edge Cases & Failure Handling
- **No cache restored** → the diagnosis workflow fails loudly (the `cache-matched-key == ''` pattern from #242).
- **No EP version for the target gameweek** → the diagnosis writes `status: no_ep_outputs_for_target_gameweek`. The executor dispatches `scheduled_pipeline.yml` and retries after it finishes.
- **Breakout group too small** (2025-26 `n_player_steps` below 200 [A3]) → widen once, as declared in advance: rule (a) becomes "his club had played at least 3 league matches, and he started at least 2 of the last 3". If it is still below 200, stop and report.
- **Control's 2025-26 breakout residual ≤ 0** (the backtest doesn't show the live problem) → R4(iii) can't pass. Stop after Phase 2, report, and name a forward-test ledger as the next step.
- **An arm run fails or is incomplete** (`progress.complete` false) → re-dispatch it fresh with the same args; `--resume` can't work in this workflow. If it is incomplete again, re-run the control and every arm of that batch with `--seasons 2025-2026` added.
- **Two batches on the same ref** → never dispatch the same args on the same ref while that run is still going, because the concurrency group cancels the first.
- **Different caches** → `compare_arms.py` refuses, and the executor re-runs the control and the arm together.

## Risks, Landmines & Adaptations
- **Promoted-club players would swamp the 2025-26 group:** every regular at a promoted club has under 900 earlier league minutes. → Rule (c) moves them to a separate, non-deciding block, so the club-strength bias stays out of the gate.
- **2024-25 has no earlier season in the data,** so every starter would count as "new". → That season's block is `null`; 2025-26 decides, as it already does for points.
- **The backtest may not reproduce the live problem.** → Phase 2 measures control's breakout residual before any arm runs, and the plan stops if it is ≤ 0.
- **The cause may be Brighton's club rating, not player-level.** → Phase 1 attributes the gap; a mostly-club cause is parked as its own plan.
- **Fishing across many arms.** → Arms are declared before results, five at most; R4 is applied mechanically by `compare_arms.py`; raw m ± SE is always reported.
- **Runs on different cached DBs aren't comparable.** → Each batch has its own control, and `db_cache_key` is checked.
- **Luck mistaken for skill in the diagnosis.** → Rates are compared to xG/xA per 90, not actual goals and assists.
- **The "beats crowd" number is built from each arm's own EP** (verified: `docs/reports/2026-10_rate_prior.md`). → It is used nowhere in the gate.
- **Deadline pressure (GW6, 10 Oct 10:00 UTC).** → Phases 1-3 use existing flags and fit before it [A4]. Diagnosis-led arms can land for GW7, and no phase is rushed past the gate to make GW6.

## Assumptions Ledger
| ID | Assumption | Basis | Blast radius if wrong | Check |
|----|------------|-------|-----------------------|-------|
| A1 | Breakout = ≥3 of the club's last 4 started (with ≥4 played), AND (<900 earlier league minutes or <40% earlier starts), AND not at a promoted club | captures new-to-league and backup-to-starter; decided before any data is seen | the metric measures the wrong group; R4(iii) misleads | Phase 2 unit test; the report lists 10 sample player-steps for a sanity read |
| A2 | "Level or better" = m ≥ −0.25 and m + SE ≥ 0 | fits past decisions: the anchor (+0.59 ± 0.71) and the start prior (+0.03 ± 0.27) passed; #233 (−0.51 ± 0.35) and the role blend (−0.62 ± 1.56) did not (verified: commit 96a7f51 message, `docs/reports/2026-10_live_path_diagnosis.md:204`). That commit also called season-sim arms at −0.35 ± 1.44 "level", but that is a different metric (season simulation), not walk-forward squad points | the gate is too loose or too tight | the report shows raw m ± SE so the user can overrule |
| A3 | 2025-26 has at least 200 breakout player-steps after excluding promoted clubs | many new signings and role changes in a season | R4(iii) is noise | Phase 2 prints `n_player_steps`, with the declared widening fallback |
| A4 | An arm takes about 30 min, and the diagnosis plus metric fit in under a day | verified run times above | GW6 is missed; GW7 still works | Phase 1 and 3 timestamps |
| A5 | The newest `duckdb-` cache holds live EP for the next gameweek | the nightly ML shadow found `ep_model_version` 2 for GW6 on a restored cache (verified: `data/dashboard/ml_shadow.json`) | the diagnosis has nothing to read | Phase 1 first step; on a miss, dispatch `scheduled_pipeline.yml` and retry |
| A6 | `gh workflow run <file> --ref <ref> -f args="..."` and `gh run download <id> -n walkforward-summary -D <dir>` are the right CLI forms | [assumed: GitHub CLI syntax - if wrong: use the GitHub API or MCP equivalents (`actions_run_trigger` run_workflow, `actions_get` download_workflow_run_artifact)] | only the commands change | first use in Phase 1 |

## Open Items (none blocking)
- Thresholds for diagnosis-led arms - proceed with the values the diagnosis suggests, written into the Phase 2 PR description before Phase 3 is dispatched.

## Verification
- On every PR: `PYTHONPATH=src python -m pytest tests/ -q` and `ruff check .`
- Diagnosis: `gh workflow run diagnose_breakouts.yml --ref master`, then `gh run download <run-id> -n diagnose-breakouts -D diag/` [A6], and read `diag/diagnose_breakouts.json`.
- Arms, one batch:
  - `gh workflow run branch_walkforward.yml --ref master -f args=""` (control);
  - `gh workflow run branch_walkforward.yml --ref master -f args="--role-matches-threshold 4"` (A1), and likewise for A2 and A3;
  - after each run, `gh run download <run-id> -n walkforward-summary -D <arm>/`;
  - then `python scripts/compare_arms.py control/walkforward_summary.json a1/walkforward_summary.json` → PASS/FAIL [A6].
- After a promotion: note the commit of `data/dashboard/projections_latest.json` before the pipeline run, dispatch `scheduled_pipeline.yml`, then compare `ep_per_gw` for the gameweeks present in both versions for the four named players.
- The user confirms done by reading `docs/reports/2026-10_breakout_players.md` and the four players' new projections.

## Build Phases
- [ ] Phase 1: Diagnose the gap on the live DB
      Done when: the `diagnose_breakouts.json` artifact meets R1's acceptance for all six players.
      Steps:
        - First step: check A5. Confirm the restored cache has EP for the next gameweek; if not, dispatch `scheduled_pipeline.yml` and retry.
        - Write a failing test with a synthetic DB, then `scripts/diagnose_breakouts.py`, reusing `compute_ml_shadow`'s three helpers named in R1.
        - Add `.github/workflows/diagnose_breakouts.yml`: restore the cache, fail on a miss, run the script, upload the artifact.
        - Open a PR, merge when green, dispatch on master, and read the artifact.
      Covers: R1; checks: A5, A6.
- [ ] Phase 2: Add the breakout metric, the cache key and the comparison script
      Done when:
        - the R2, R3 and R3b unit tests pass;
        - a control `branch_walkforward.yml` run on this branch writes `breakout`, `breakout_promoted` and `db_cache_key`;
        - the stop conditions are checked (2025-26 n ≥ 200 after at most one widening, and control's `mean_resid` > 0);
        - the PR description lists any A4-A5 arms (mechanism, flag, value) chosen from the Phase 1 diagnosis.
      Steps:
        - Write failing tests first for the classification (all five synthetic cases in R2) and for `compare_arms.py`.
        - Implement R2 in `scripts/walkforward_summary.py`, using only matches before each step's deadline.
        - Implement `scripts/compare_arms.py` (R3, R4) and the `DB_CACHE_KEY` pass-through (R3b).
        - Dispatch a control run on the branch, record the numbers, then merge when CI is green.
      Covers: R2, R3, R3b, R4; checks: A1, A2, A3.
      Critique: important phase - after building, hand the result plus this phase's Done-when and Covers lines to a blind critic (fresh sub-agent if available, else a fresh-eyes reread). The critic defaults to FAIL and returns PASS/FAIL, strengths, weaknesses and fixes. It checks especially that no data from after a step's deadline leaks into the classification, and that the paired comparison uses only gameweeks non-null in both files. Apply blocker fixes and repeat, max 3 rounds. Done only on PASS.
- [ ] Phase 3: Run the predeclared existing-flag arms
      Done when: a control and A1-A3, dispatched together on master, are complete (`progress.complete` true, same `db_cache_key`), and `compare_arms.py` output is recorded for each.
      Steps:
        - Dispatch the control and A1-A3 at the same time on master; they run in parallel (distinct args, so the concurrency group doesn't cancel any).
        - Download the artifacts and run `compare_arms.py` for each arm.
        - Record every number in the report draft.
      Covers: R4; checks: A4.
- [ ] Phase 4: Run the diagnosis-led arms (only those declared in the Phase 2 PR)
      Done when: each declared arm (at most two) has an opt-in flag, tests, a complete run with its own control from the same batch, and a `compare_arms.py` result. If none were declared, the report says why.
      Steps:
        - Build each arm opt-in, default off, behind a `run_walkforward.py` flag, following the rate prior pattern in #243. Write tests first.
        - Merge when green, then dispatch a fresh control and the arm(s) together on master.
        - Run `compare_arms.py` and record the results.
      Covers: R4.
- [ ] Phase 5: Promote the best passing arm, if any
      Done when:
        - the promotion PR is merged with green CI;
        - the next `scheduled_pipeline.yml` run succeeded;
        - for gameweeks present before and after, the affected named players' predicted points moved the way the arm implies, net of the two reference starters.
        - If no arm passed, this phase is skipped and the report records that.
      Steps:
        - Pick the passing arm with the largest 2025-26 breakout-residual cut. On a tie, pick the one with fewer changed settings.
        - Make it live via a seed and a version argument for every caller, extend `tests/test_live_switch_wiring.py`, and add an off flag to `run_walkforward.py`.
        - Run the full test suite and open a PR with the arm's R4 numbers in it.
        - If the winning arm's paired m < 0 (it passed only through the −0.25 tolerance), ask the user before merging. Otherwise merge when green.
        - Note the projections commit, dispatch the pipeline, and compare.
      Covers: R5.
      Critique: important phase - after building, hand the diff plus this phase's Done-when and Covers lines to a blind critic (fresh sub-agent if available, else a fresh-eyes reread). The critic defaults to FAIL and checks that every caller that builds minutes or EP receives the new setting. Apply blocker fixes and repeat, max 3 rounds. Done only on PASS.
- [ ] Phase 6: Write the report
      Done when: `docs/reports/2026-10_breakout_players.md` is merged and contains the diagnosis table, control's breakout metric, every arm's R4 numbers with raw m ± SE, the decision, and the parked follow-ups.
      Steps:
        - Write it in the house report style (see `docs/reports/2026-10_rate_prior.md`).
        - Open a PR and merge when green.
      Covers: R6.
