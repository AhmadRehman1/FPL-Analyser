# 2026-10 model status refresh

Re-verification of every open item from `2026-09_model_failure_diagnosis.md`,
`2026-09_chip_policy_and_scoring_diagnosis.md` and `docs/plans/2026-09_ep_attacker_defender_imbalance.md`
against `origin/master` as of 2026-09-29 (data commit `4956fe1`, track record generated
2026-09-28T23:37). Code line numbers are from that commit.

Nothing below was re-diagnosed from scratch. Each item is marked **still open / partly fixed /
fixed / was wrong**, with the current number next to it.

## Headline (start of the 2026-10 work)

| metric | value | source |
|---|---|---|
| beats avg manager, pts/GW (walk-forward, 70 GWs, 2024-25 + 2025-26) | **-0.71** (model 49.16 vs avg 49.86) | `app_track_record.json` headline |
| minutes log score | **-1.256** (uniform 3-state baseline is -1.099) | same |
| minutes Brier | 0.371 | same |
| global EP bias per player | -0.014 | same |
| price-band mean residual `<5.0 / 5.0-7.0 / 7.0-9.0 / 9.0+` | **-0.24 / +0.31 / +0.78 / +0.93** | same, `segment_calibration` |
| model's own 2026-27 team after GW5 | 230 pts, **-69 vs field average**, FH + TC used | `app_model_team.json` |
| planner "if you had followed it" (6 real decisions) | -0.17 pts per decision (preliminary) | `planner_decision_accuracy` |
| ML shadow vs quant (latest weekly log) | ML 4,222 vs quant 4,134 manager points, ML ahead on every logged run | `research/ml/results_history/weekly_quality_history.csv` |

Positive residual = the model under-predicts that group.

## Status of each open item

### 1. Finding 5 - the recalibration gate - **partly fixed**

- Fixed since 2026-09-13: #185 set `backtest._DEFAULT_MIN_RELATIVE_IMPROVEMENT = 0.01`, so the
  automated gate now needs a 1% gain (`backtest.py:1961`). The k_minutes 450->900 (0.029%) change
  would not pass it today.
- Still open:
  - the 1% is a module constant, not a versioned parameter;
  - `scripts/review_recalibration.py --confirm` still does no check at all (`set_status()`, line 89);
  - no held-out evaluation - the gate compares the same eval-set score the grid search optimised;
  - no collision check before confirming;
  - no grid-boundary check.

### 2. Finding 1 - recalibration optimises the wrong target - **still open, and worse than described**

- `_ep_calibration_mae_for_step()` (`backtest.py:2230`) is still an unweighted MAE over every
  player with a points row.
- The `k_minutes` grid still tops out at 900 (`backtest.py:2279`).
- **New: the walk-forward never passes `rate_shrinkage_params_version` to `ep.run()`**
  (`run_gameweek_step()`, `backtest.py:325`), so every walk-forward, including the nightly track
  record, runs the hardcoded default `k_minutes = 450`. Live runs use the confirmed 900. The
  450->900 change has never been measured by the walk-forward at all. So the earlier "the 9.0+
  residual got worse after the k_minutes confirmation" reading was new data drifting in, not the
  parameter. Fix: branch `claude/walkforward-rate-shrinkage`.
- Direction matters: a higher k shrinks own rates harder toward the position average, so 900
  compresses premiums more than 450 did. Experiments `bt/k900` and `bt/k150` measure both ways.

### 3. Finding 2 - premiums under-predicted - **still open**

Price-band residuals now: `<5.0` -0.237, `5.0-7.0` +0.314, `7.0-9.0` +0.779, `9.0+` **+0.932**
(0.945 on 2026-09-13, 0.84 before the Sept work). The global bias is -0.014, which still hides it.

The component split (already in the track record) says where it comes from:

| band | appearance | goals | assists | clean sheet | other (bonus, DefCon, saves, cards) | total |
|---|---|---|---|---|---|---|
| `<5.0` | -0.05 | -0.05 | -0.01 | -0.05 | -0.08 | -0.24 |
| `5.0-7.0` | **+0.23** | +0.04 | +0.05 | +0.02 | -0.03 | +0.31 |
| `7.0-9.0` | **+0.24** | +0.21 | +0.13 | +0.02 | +0.18 | +0.78 |
| `9.0+` | **+0.21** | +0.20 | **+0.28** | 0.00 | +0.24 | +0.93 |

What this says:

- **Appearance points are under-predicted by about +0.22 for every band above £5m, and over-predicted
  below £5m.** That's a minutes-model problem (regular starters get too little P(60+), fringe players
  too much), shared with Finding 4. It is roughly a quarter of the premium gap and most of the
  £5-7m gap.
- Goals are only +0.20 at 9.0+. **Penalties are less "missing" than the brief assumed:** FPL's xG
  already includes penalty xG, and a 1.15x penalty-taker multiplier already exists
  (`set_piece_evidence_params`, `expected_points.py:94`). It only fires when a set-piece evidence
  claim exists at the asof date, which never happens in the 2024-26 walk-forward. A penalty fix is
  worth checking, but it isn't the main lever.
- Assists (+0.28) are the biggest single component for premiums. A likely mechanism is that FPL
  awards assists more generously than Opta counts them (won penalties, deflected passes, rebounds),
  so an xA-based rate under-counts FPL assists most for the players who create the most. This is
  a hypothesis still to be tested.
- "Other" (+0.24 at 9.0+, -0.08 below £5m) is mostly bonus for attackers, and DefCon/saves for cheap
  defenders and keepers.
- By position: GK -0.25, DEF -0.09, MID +0.05, FWD +0.17.

### 4. Finding 3 - invented risk aversion, captain hit hardest - **still open**

- `risk_aversion_params.lambda_value = 0.15` is still the v1 spec value, never fitted
  (`squad_optimizer.py:86`).
- The captain's variance is still weighted 3x (`squad_optimizer.py:434`). The headline
  beats-avg-manager metric scores exactly this solver captain (`score_gameweek()`,
  `backtest.py:884`), so this feeds the headline directly.
- `risk_posture.py`'s docstring (lines 20-22) still quotes "lambda 0.15->0.05 was 3.52->4.27;
  kappa_tc 0.15->0.5 was 1.02->1.06". Neither traces to a committed artifact. The only committed
  kappa_tc confirmation is 0.15->0.2 (Sharpe 1.243->1.279). Needs correcting in Fix D/E.

### 5. Finding 4 - minutes model - **still open**

- log score -1.256 and Brier 0.371, both unchanged.
- `avail == 0.0 => p_0min == 1` is still there (`minutes_model.py:879`). The log score still floors
  at `_EPS = 1e-9` (`backtest.py:374`), so one wrong "ruled out" flag costs -20.7 on that row.
- See item 3: the appearance residual shows the calibration problem isn't only the hard zeros.
  Regular starters are under-rated across the board.

### 6. Finding 7 - several captain code paths - **still open**

`reporting.build_report()` still takes the headline captain straight from
`squad_optimizer_selections.is_captain` (`reporting.py:246`). That feeds `report_history/` and the
public Track Record page. Only the real-account panel goes through `build_captain_recommendation()`.

### 7. Workstream B - chip timing - **partly fixed**

- #173 added magnitude floors for triple captain and bench boost.
- There is still no "is this the best week or should I wait" logic (`transfer_planner.py:1031`
  and `:1157` say so in their own comments).
- Live cost so far: the GW4 triple captain projected 62.15 and realized 39 (the model team is -30
  vs field that week). The GW3 free hit was +39 vs holding.

### 8. Workstream C - the backtest's scoring - **partly fixed**

- #177 added a real auto-sub simulation (bench order plus formation legality), but only as an
  opt-in on `run_season_simulation()` and the retrospective script.
- The headline walk-forward metric (`score_gameweek()` -> `_realized_xi_points()`, no `squad_uids`)
  still scores the XI only, with no auto-subs. The headline slightly understates what a real
  manager with a bench would score, in both arms, so it's a measurement caveat rather than a
  ranking problem.

### 9. Finding 9 - evidence layer blind - **still open**

`consensus_check.py` still uses `aggregate_evidence_weight()` as a volume proxy (module docstring,
lines 3-12). `analyst_debate` / `community_sentiment` / `youtube_evidence` still come in with
`claim_value_numeric=None`. Fix G is the hook.

### Already fixed (unchanged since 2026-09-14)

Finding 6 (`parameters_backtested`, now 8 of 62) - #171. §9 (attack-posture version collision) -
#170. Finding 8 (ML lane is frozen and correctly parked) - nothing to fix.

## New diagnostic: why the eyeball beats the model (GW2-5, 2026-27)

Method: the top 1,000 managers in the overall league (by rank on 2026-09-29) and their real picks
for GW2-5 from the FPL API. The model's projection for each gameweek is from the last
`projections_<date>.json` snapshot dated before that gameweek's deadline (point-in-time; GW1 has no
pre-deadline snapshot, so it's excluded). Caveat: picking today's top 1,000 has survivorship bias,
since they're top partly because these picks worked.

Top-20 overlap per GW (how many players two lists share):

| GW | model top-20 vs actual top-20 | elite-owned top-20 vs actual top-20 | model vs elite |
|---|---|---|---|
| 2 | 5 | 8 | 4 |
| 3 | 2 | 2 | 8 |
| 4 | 3 | 7 | 9 |
| 5 | 2 | 5 | 8 |

Players in the elite's top-20 most-owned list, sorted by how much the model under-rated them
(totals over the GWs they were in that list):

| player | pos | price | avg elite own | model projected | actual | model rank each GW |
|---|---|---|---|---|---|---|
| Groß | MID | 5.8 | 68% | 3.7 | 45 | 573, 432, 447, 319 |
| Schade | MID | 6.2 | 28% | 14.6 | 36 | 41, 25, 51, 24 |
| Tarkowski | DEF | 6.1 | 31% | 12.8 | 34 | 19, 15, 3 |
| Semenyo | MID | 8.4 | 24% | 15.1 | 31 | 133, 8, 32, 7 |
| Haaland | FWD | 15.6 | 82% | 21.4 | 37 | 6, 1, 3, 1 |
| Gvardiol | DEF | 5.7 | 43% | 13.2 | 28 | 141, 35, 79, 21 |
| Kinsky | GKP | 4.5 | 34% | 1.9 | 16 | 605, 496, 488, 454 |
| João Pedro | FWD | 7.7 | 84% | 9.7 | 22 | 155, 165, 71, 158 |
| De Cuyper | DEF | 5.0 | 46% | 9.1 | 21 | 70, 160, 175, 190 |
| Calafiori | DEF | 5.8 | 77% | 9.6 | 20 | 161, 139, 145, 126 |

The answer in numbers: **the elite's core template players that the model ranked 100-600th are
almost all new signings or players in a new role** (Groß, João Pedro, Kinsky, De Cuyper,
Calafiori, Gvardiol). Those six scored 152 points in the weeks the elite held them, against 47
projected.

The model's minutes model builds P(start) from two prior Premier League seasons. A player new to the
league or to the role has almost no weight on his own record, so he gets shrunk to a low position
average. The fix for this already exists in code (`current_season_role_params`, `minutes_model.py:767`)
but has never been switched on anywhere. Its one committed study only ran the "on" arm, so it's
inconclusive. Experiment `bt/role-on` runs the proper comparison.

The same table also shows premiums under-projected (Haaland 21 vs 37, Isak 12 vs 23, Bruno 19 vs
29), consistent with item 3.

## Ranked by expected points per gameweek for a real manager

These are estimates from the evidence above, not measured gains. Each gets replaced by a
walk-forward before/after number as the fixes land.

| rank | item | why this rank | rough gain |
|---|---|---|---|
| 1 | **New-signing / role-change blindness in the minutes model** (new, plus Finding 4) | Keeps the model off the season's template picks entirely; 6 core elite picks at rank 70-600 | +1.5 to +3 pts/GW live, early season most |
| 2 | **Captaincy** (Findings 3 + 7) | The armband is the single biggest weekly lever, and the solver actively avoids high-ceiling captains | +0.5 to +1.5 |
| 3 | **Premium under-prediction** (Finding 2) | +0.93/player/GW at £9m+, and it tilts the whole squad toward cheap defenders | +0.5 to +1.5 |
| 4 | **Minutes calibration** (Finding 4: probability floors, stale evidence) | Overlaps with 1 and 3 (appearance residual); also the log score | +0.3 to +1 |
| 5 | **Chip timing** (Workstream B) | One mistimed TC/BB is a 10-30 pt swing, about 1-2 per season | +0.3 to +0.8 averaged |
| 6 | **Calibration target + k grid** (Finding 1) | Only matters through 3; the walk-forward doesn't even use k yet | via 3 |
| 7 | **Lambda** (Finding 3) | Unknown until Fix D removes the captain part | 0 to +0.5 |
| 8 | **Recalibration gate** (Finding 5) | No direct points, but every other parameter change goes through it | 0 direct |
| 9 | **Auto-subs in the headline** (Workstream C) | Measurement accuracy, same in both arms | ~0 |
| 10 | **Expert consensus** (Finding 9) | Check first, input later | unknown |

## Proposed fix order (what I'm doing)

1. **Fix A (gate)**, as asked: cheap, and nothing else should go live without it. No walk-forward
   needed; it's a code gate with tests.
2. **Fix D (captaincy)**: the experiment is already running (`bt/fix-d-cap0`, captain variance
   multiplier 0 vs 1).
3. **Minutes: role-change blend** (`bt/role-on`), then **Fix F** (probability floors, stale-evidence
   decay). Promoted above Fix C because the elite table says this is the biggest live gap.
4. **Fix C** suspects, one at a time: k_minutes (`bt/k900`, `bt/k150` running), then assists,
   penalties, bonus, DefCon.
5. **Fix B** (weighted calibration target) through the new gate.
6. **Fix E** (lambda sweep) if time allows.
7. **Fix G** (expert consensus ingester, check-only first).

Walk-forwards run on GitHub Actions via `.github/workflows/branch_walkforward.yml` (experiment
branches `bt/**`, restore the cached ingested DB, commit nothing). The overnight container could
not build its own DB, because cloning the third-party FPL-Core-Insights dataset was blocked.

## Plan for items not started tonight

**Chip timing (Workstream B, "should I wait?").** Everything it needs is already computed and thrown
away each week (`evaluate_triple_captain()` / `evaluate_bench_boost()` read existing MC/EP data; no
solves).

1. Carry `tc_score` and `bench_ep_sum` per visible gameweek on `forward_season_sim.GameweekResult`.
2. Option value: `value_now - E[max(value over the remaining eligible GWs before the chip-set
   deadline)]`, where later weeks are shrunk toward the season mean by how far out they are
   (versioned placeholder).
3. Fire only when `value_now >= E[max remaining] - margin` (versioned margin), the same shape as
   the wildcard sweep's gate.
4. Validate with `run_season_simulation()` over 2024-26 (TC/BB points per season with vs without
   the wait rule), then through the Fix A gate.

This means overriding the "per-week call, not swept" TC exemption in
`reconcile_chips_with_timing_sweep()`. That's a deliberate reversal, so it should be named in the PR.

**Bonus (Fix C suspect 4).** The BPS model leaves out passing/key-pass and winning-goal BPS
(attackers) and the clean-sheet BPS (keepers and defenders). The net direction isn't clear without
per-match key-pass data, which isn't reconciled. The assists calibration already raises attackers'
BPS through `e_assists`. Next: reconcile key passes from the match data, then test.

**Penalties (Fix C suspect 1).** FPL xG already includes penalty xG, and a 1.15x penalty-taker
multiplier exists but never fires in the 2024-26 walk-forward (no dated set-piece claims). The
expected gain is small. Do it after assists, by deriving historical takers from match data.

**DefCon (Fix C suspect 5).** "Other" is -0.08 for `<5.0` and for defenders overall, a slight
over-prediction of cheap defenders. It needs a position x band split of the DefCon component; the
walk-forward currently folds it into "other".

## Progress log

_(Plans for items not started tonight are in the section above this one.)_

_Updated after each fix. Scoreboard numbers come from `scripts/walkforward_summary.py` on the same
cached DB for every arm._

- 2026-09-29: report written; experiments `bt/baseline`, `bt/fix-d-cap0`, `bt/k900`, `bt/k150`,
  `bt/role-on` started.
