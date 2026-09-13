# 2026-09 model failure diagnosis — Phase 1 verification

Verification pass against Findings 1-8 from the "why the model picks badly" prompt, run
2026-09-13 against `origin/master` (production) plus a fresh local ingestion where the
sandbox allowed it. Every claim below is either (a) read directly from committed
`data/dashboard/*.json` / `data/recalibration/*.json` on `origin/master`, (b) read from
`src/`/`scripts/` source at the same commit, or (c) a live DB query — labelled per-item.
`db/fpl_quant_v2.duckdb` is gitignored and was stale (2026-09-09); a fresh `run_ingestion.py`
against the real FPL API mostly succeeded but the sandbox's destructive-action guard blocked
the rebuild-from-scratch step needed to reproduce Finding 3's collision end-to-end, so that
item is confirmed via code + timeline evidence, not a live crash trace. Noted inline.

**Bottom line: the core hypothesis holds, but the two most damaging live bugs found this pass
are not in the original 8 — a broken transparency counter (Finding 6) and a currently-silent
production crash in the Attack-rank posture (new, see §9), both downstream of the same root
cause as Finding 5: recalibration proposals are confirmed with no effect-size floor and no
collision check.**

## Methodology note: is `app_track_record.json` stale?

No. `backtest_run_id` has been `1` throughout, but `nightly_backtest.yml` re-runs
`backtest.run()` (not `backtest.recalibrate()`) against that same run id every day and
overwrites `data/dashboard/app_track_record.json`. Diffing the committed file at `fa5176c`
(2026-09-09, the commit the project memory calls the "−0.88/GW" baseline) against today's
`0e6adb7` (2026-09-13, generated_at `2026-09-13T20:57:18`) shows the headline moved from
**−0.8762 → −0.9009 pts/GW** and every segment residual drifted by 0.0003–0.005 (e.g.
`price_band=9.0+` 0.9414 → 0.9447) — consistent with new real 2026-27 gameweeks entering the
scored window, not a parameter change (no confirmed recalibration has actually been
activated since 2026-09-09; see §5). **All of the user's quoted numbers are current, not
stale** — they match the live file exactly as of this run.

---

## Finding 1 — recalibration optimises a metric dominated by irrelevant players

**Confirmed**, including every number.

- `backtest.py:2014` `_ep_calibration_mae_for_step()` returns an unweighted mean over every
  player with a realized `event_points` row — no ownership/price/selection weighting.
- `backtest.py:2060` `refit_rate_shrinkage()` grid-searches
  `k_minutes_grid: tuple = (150.0, 250.0, 350.0, 450.0, 600.0, 900.0)` — **900 is the literal
  top of the tested range**, confirming the "never bracketed an optimum" claim.
- Its own docstring (lines 2071-2076) admits the eval-set-tuning risk verbatim: "picking the
  candidate that minimizes eval_steps and reporting that same eval_steps score is optimistic
  by construction."
- Live price-band split, computed directly from today's `data/dashboard/projections_latest.json`
  (657 players): `<5.0` 304 (46.3%), `5.0-7.0` 324 (49.3%), `7.0-9.0` 24 (3.7%), `9.0+` 5 (0.8%)
  — exact match. The 5 players are exactly Haaland (£15.5m), Bruno (£12.0m), Palmer (£9.7m),
  Saka (£9.5m), Isak (£9.1m).
- `data/recalibration/seeds_1.json`: `rate_shrinkage_params.k_minutes` v1→v8, 450.0→900.0,
  `reviewed_by: "claude-code"`, `reviewed_at: 2026-09-09T07:52:34`, metric
  `ep_total_calibration_mae` 1.1456995269035073 → 1.1453659985877347 (**0.029% gain**) — exact
  match to the user's numbers, and confirms this went through the *automated* gate (see
  Finding 5), not a human review.
- Git-history reproduction (ran a variant of the user's script directly against
  `data/dashboard/app_track_record.json`): `price_band=9.0+` mean residual **0.8414 (pre-#131/
  #134 fixture scaling) → 0.9414 (2026-09-09, post k_minutes confirm) → 0.9447 (today)**. Same
  direction, same order of magnitude as claimed.

## Finding 2 — monotonic price bias, hidden by a near-zero global mean

**Confirmed**, current as of today (not stale — see Methodology note above). Today's
`segment_calibration` in `app_track_record.json`:

| segment | mean residual | MAE |
|---|---|---|
| `price_band=<5.0` | −0.2358 | 0.9175 |
| `price_band=5.0-7.0` | +0.3149 | 1.4479 |
| `price_band=7.0-9.0` | +0.7753 | 2.3921 |
| `price_band=9.0+` | +0.9447 | 3.3318 |

Monotonically increasing with price, matching the user's table to 4 decimal places.
`headline.ep_points_bias_per_player` = **−0.0126** (global), confirming the "cancels out"
framing. `docs/plans/2026-09_ep_attacker_defender_imbalance.md:99` records "+0.84 at £9.0m+" —
current value 0.9447 confirms **it has gotten worse (+12.6%), not better**, exactly as the
user suspected.

## Finding 3 — risk aversion is invented, uncalibrated, and hits the captain hardest

**Confirmed**, and it's worse than described — it's live in three separate places, not one,
and only one of the three has ever been patched.

- `squad_optimizer.py:86`: `risk_aversion_params` v1 `lambda_value=0.15`, unchanged since
  2026-08-10 seeding, current active version (per `data/recalibration/seeds_1.json`, no
  `risk_aversion_params`/`lambda_value` proposal exists there at all — confirmed, see below).
- `squad_optimizer.py:434`: `(xi[uid] + 3 * captain[uid]) * c["var"]` — the captain's variance
  is weighted at 3× exactly as described (comment at line 430 derives this algebraically from
  `w_i = xi_i + captain_i`, so it's intentional, not a bug in isolation).
- The `captain_ep_gap` diagnostic and its comment (`squad_optimizer.py:777-807`) are **quoted
  verbatim in the current code**, including the exact GW4/Senesi/Bruno/Watkins sentence.
- `data/lambda_study/SUMMARY.md`: **confirmed as described** — one λ arm (0.15, the live pin),
  one 5-gameweek 2025-2026 season simulation. Sharpe 9.544/16.734/3.334 are the cap sweep, not
  a lambda sweep, and are 5-GW noise.
- **`risk_posture.py`'s docstring discrepancy — confirmed, and worse.** The module docstring
  (lines 18-23) claims: "lambda 0.15→0.05 was 3.52→4.27; kappa_tc 0.15→0.5 was 1.02→1.06."
  Today's `seeds_1.json` has **no `risk_aversion_params` proposal at all** (nothing to
  corroborate 3.52→4.27), and its only `tc_risk_aversion_params` confirmations show Sharpe
  **1.2431108370340345 → 1.2791735227188274** (≈1.243→1.279, matching the user's "doesn't
  match either" complaint) for value 0.15→**0.2** — not 0.5. **None of the docstring's four
  numbers trace to any committed artifact.** This is either a stale draft from an earlier,
  since-reverted proposal generation, or a fabricated-sounding placeholder that was never
  cross-checked before being shipped as "pre-validated, not a guess" (the module's own words).
  It needs correcting regardless of which.
- **New, live consequence of this (see §9 below): the "attack" posture's hardcoded
  `tc_risk_aversion_params` v2=0.5 now collides with a *different*, since-confirmed v2=0.2 from
  the recalibration pipeline — same (family, version) key, two different values. This is not
  hypothetical; the evidence below shows it is actively crashing in production.**

## Finding 4 — the minutes model is worse than random guessing on log score

**Mechanism confirmed in code; magnitude plausible but not independently re-measured** (would
need a DB query counting how many gameweek-steps hit the `avail=0.0` floor, which needs the
real backtest DB — see Methodology).

- `backtest.py:341` `_EPS = 1e-9`; `log_score_categorical` (line 353-355) floors any observed
  probability at `_EPS` → `log(1e-9) = -20.72`, exact.
- `backtest.py:366-369` `_minutes_state()`: exactly 3 states (`"0"`/`"1_59"`/`"60plus"`) →
  uniform baseline `ln(1/3) = -1.0986`, matching the "worse than a coin toss over three sides"
  framing.
- `minutes_model.py:790` comment, verbatim: `"avail == 0.0 (ruled out) => p_0min == 1"` —
  confirmed exactly as quoted.
- Today's headline: `minutes_log_score = -1.256`, `minutes_brier = 0.3709` — both match. The
  mechanism (a hard zero from a stale/wrong availability flag, then a −20.7 floor penalty on
  the rare miss) is real and present in the current code; whether it's *the dominant* driver of
  the −0.16 gap to uniform (vs. e.g. genuinely-hard rotation calls that are merely well-ranked
  but miscalibrated in scale) isn't separable from committed artifacts alone.

## Finding 5 — the recalibration gate accepts changes that are literally nothing

**Confirmed, and the code-level root cause is now precise** (this was the most productive
thread to pull on this pass).

Two independent confirmation paths exist, and **neither enforces a minimum effect size**:

1. **`backtest.evaluate_and_promote_proposal()`** (the automated gate `scripts/run_backtest.py`
   calls right after `recalibrate()`): takes `min_relative_improvement: float = 0.0` —
   **the default is literally no floor.** It does special-case `rho_residual`
   (`_NOT_A_SCORE_METRICS`) to reject a `metric_before == metric_after` no-op, but that check
   is scoped to that one metric name, not universal.
2. **`scripts/review_recalibration.py`'s `--confirm` (the human/manual path)**: `set_status()`
   (lines 89-121) does **zero validation** — no sign check, no effect-size check, not even a
   "did the value change" check. It will happily flip any `proposal_id` to `confirmed`.

Today's `data/recalibration/seeds_1.json` confirmed rows, cross-checked:

- `correlation_params.rho_residual` v2→v3 **and** v2→v4, both 0.0→0.0, confirmed
  `2026-09-08T01:16:33`/`01:16:59`, `reviewed_by: "ahmadrehman1-via-claude"` (i.e. went through
  the zero-check manual path, not the automated one — the automated gate's rho_residual
  special-case would have rejected these). **Confirmed exactly as described, twice.**
- `minutes_adjustment_params.cap`, v1→v18, 6.0→6.0, confirmed `2026-09-08T17:15:06` by
  `"claude-code"` (the automated path) — its own metric (`log_score_minutes_mean_holdout`) did
  move (−1.8806→−1.8788), but that's from a *sibling* key in the same batched proposal
  (`magnitude`, which did change, −4.0→−3.0), not from `cap` itself. **Confirmed exactly as
  described.**
- The pending (not yet confirmed) `minutes_model_shrinkage_params.competitive_matches_threshold`
  v11→v14, 15.0→15.0, metric −1.8787601645729581 → −1.8787601645729584 — **16th-decimal
  floating-point noise, currently sitting as a live pending proposal today**, exactly as
  described (the user's Sept-5 numbers for this same family/value are stale by proposal id but
  the phenomenon is reproduced fresh in the current pending queue).

## Finding 6 — nothing is actually calibrated

**Refuted as written — the counter is broken, not honest, and I can show exactly how.**

`reporting.py:632`: `n_backtested = sum(1 for row in transparency if row["backtested_via_m7"])`.
`params.py:117` (`transparency_panel()`): a family counts as `backtested_via_m7` **iff it has
at least one row in the `recalibration_proposals` DB table**, queried fresh
(`params.py:101-103`) against whatever `con` the caller passed.

The catch: `export_track_record.py`/`run_report.py` (which write `parameters_total` /
`parameters_backtested`) run against `nightly_backtest.yml`'s DB, built by
`scripts/run_walkforward.py` — whose own docstring says outright: *"Recalibration is a
separate concern and stays in `scripts/run_backtest.py`... run that locally, or on a runner
without the 6h limit, when you actually want new `recalibration_proposals`."*
`run_walkforward.py` never calls `recalibrate()`. So the DB session that computes
`parameters_backtested` **has an empty `recalibration_proposals` table by construction, every
single night, forever** — confirmed on my own fresh local DB (built today):
`SELECT count(*) FROM recalibration_proposals` → **0**, even though the committed
`seeds_1.json` has 12 real proposal records including several genuinely-confirmed value
changes (k_minutes 450→900, `fact_type_multiplier` 1.2→1.0, `minutes_adjustment.magnitude`
−4.0→−3.0, `minutes_model_shrinkage.competitive_matches_threshold` 10→15, `kappa_tc` 0.15→0.2).

So `parameters_total=62, parameters_backtested=0` is not "62 invented defaults, 0 recalibrated"
— it's "this specific counter can structurally never see recalibration work, because the two
workflows that produce its two halves (the nightly walk-forward, and the weekly/manual
recalibration) never share a database." At least 5 of the 62 families have in fact been
through a real M7 refit and had a value change confirmed. **Both are serious, and they need
different fixes, as the user anticipated** — but the fix here is "read `touched_families`
from the committed seed file (which persists across runs) instead of the DB table (which
doesn't)," not "start actually calibrating things" — that part is already happening.

## Finding 7 — two live feeds disagree about who the captain is

**Confirmed as a live discrepancy, but the framing is wrong — these are not two views of the
same decision, they're two entirely different squads, and the API gives no field to tell them
apart.**

Today's data: `app_model_team.json` GW4 `current_squad`: **`is_captain: true` on "Marcos
Senesi Barón"**. `app_track_record.json` → `transparency_log.snapshots` GW4:
**`captain_name: "Ethan Ampadu"`**. Generated 2026-09-13T20:57:18 and 20:57:22 — 4 seconds
apart, same nightly run, exactly as described.

Traced both to source:

- **`app_model_team.json` is the model's own stateful, chip-using managed team** (`model_team.py`'s
  ledger, `data/model_team/state.json`) — a real squad with real transfer/chip history since
  GW1. Its GW4 captain is whoever the actual chip-week decision picked; GW4's action was
  `triple_captain`, decided by `transfer_planner.evaluate_triple_captain()` — see Finding 3:
  `tc_score = mean_total - kappa_tc * sqrt(var_total)` (`transfer_planner.py:978`), still fully
  risk-adjusted. **This is the team that actually played and scored 30 points off a 62.15
  projection — this feed is authoritative for "what did the model do."**
- **`app_track_record.json`'s `transparency_log.snapshots` is a snapshot of a completely
  different subject: the from-scratch "model_optimal" squad**, re-solved daily with no chip
  state, no transfer history, no relation to `app_model_team.json` at all.
  `scripts/run_report.py:_resolve_report_run_id()` explicitly selects the
  `is_manager_snapshot=FALSE` `squad_optimizer_runs` row (a fresh £100m build), and
  `reporting.build_report()` (line 236) sets `captain = next(p for p in squad if
  p["is_captain"])` — **read straight off `squad_optimizer_selections.is_captain`, i.e. the
  raw in-solve MIQP captain pick**, never passed through `build_captain_recommendation()`
  (the analytic, PR #138-fixed ranking used for the two real tracked accounts' actual
  `captain_recommendation` field — confirmed today: `real_squad_7139944.json` correctly shows
  `recommended_name: "Erling Haaland"`, `recommended_expected_points: 6.45`). **So PR #138's
  captain-ranking fix was applied to exactly one of at least three captain-selection code
  paths in this codebase** (the real-account recommendation panel), and never to
  `build_report()`'s own `headline["captain"]` — which is what feeds every `report_history/`
  snapshot and therefore the public Track Record page. Ethan Ampadu being GW4's model-optimal
  captain is very likely the same 3×-variance-penalized-captain mechanism as Finding 3, in a
  location #132/#138 never touched.

**This is a real UX/architecture bug, not a data race**: the Track Record page's "this is what
the model said" timeline is narrating a squad the app never otherwise shows the user, under
the same field name (`captain_name`) the model-team card uses for a completely different
squad. Recommend either labeling the snapshot's subject explicitly, or routing it through
`build_captain_recommendation()` like every other user-facing captain field, or replacing it
with `model_team`'s own real captain.

## Finding 8 — the ML lane beats the quant model on every logged run, and is correctly parked

**Confirmed exactly.** `research/ml/results_history/weekly_quality_history.csv`: 7 rows, all
`ml_beats_quant=True`; `quant_manager_points` 3745-3750, `ml_manager_points` 4086-4356 (same
direction/magnitude as the user's 3,763/4,265, different specific run). `research/ml/
forward_test/FROZEN_CONFIG.md`: frozen at 2026-09-02, runs "at least GW19 of 2026-27." Today's
`current_gameweek` = 4. **Shadow discipline is being honored — nothing to fix here.**

---

## §9 — New finding: the Attack-rank posture has likely been silently crashing since 2026-09-08

Not one of the original 8, found while tracing Finding 3/5's version-collision thread to
ground. Confidence: **high (code + timeline evidence), not reproduced end-to-end locally**
(the sandbox's destructive-action guard blocked rebuilding a clean local DB to catch the
exception live — noted as a limitation, not asserted as directly observed).

The chain:

1. `risk_posture.py`'s `_POSTURES["attack"]` hardcodes `"kappa_tc": (2, 0.5)` —
   `resolve_versions()` (line 117) calls `params_mod.write_param(con, "tc_risk_aversion_params",
   2, ..., "kappa_tc", value_numeric=0.5)` every time the attack posture is resolved.
2. `data/recalibration/seeds_1.json` has **`tc_risk_aversion_params.kappa_tc` v1→v2, 0.15→0.2,
   confirmed 2026-09-08T01:16:40** — a *different* value at the *same* version number.
3. `scripts/run_ingestion.py:142` loads every confirmed seed via
   `backtest.load_confirmed_recalibration_seeds()` and writes it into the DB — **before** any
   planner step runs, in the same job, same DB file
   (`.github/workflows/scheduled_pipeline.yml`'s own comment confirms the attack-posture step
   "uses the SAME freshly-ingested database this run just built").
4. `params.write_param()` (params.py:39-53) is idempotent for an identical value but **raises
   `ValueError` for a value mismatch at the same (family, version, key)** — exactly this case:
   existing (0.2) vs attempted (0.5).
5. `.github/workflows/scheduled_pipeline.yml` runs the Attack-rank step under
   **`continue-on-error: true`** (confirmed at the step definition, line 261) — a crash here is
   swallowed silently, the job stays green, and nothing alerts anyone.
6. **Corroborating evidence**: `data/dashboard/real_squad_7139944_attack.json` and
   `real_squad_1305242_attack.json` were last committed **2026-09-07** — the day before the
   colliding confirmation — and have not updated since, while every other scheduled feed
   (`app_players.json`, `real_squad_*.json`, `app_track_record.json`, etc.) has updated
   continuously through today. That's a 6-day-stale feed sitting exactly at the fault line.

If this reading is right, the "Attack rank" toggle in the app has been silently serving
6-day-old data since 2026-09-08, and this is precisely the failure mode Finding 5's missing
effect-size/collision check exists to prevent — this is what "the gate needs a minimum effect
size and a sign check" costs in practice when it's absent. The fix isn't just a floor on
effect size, though; it also needs a same-version-different-value collision check before
confirming a proposal (or `risk_posture.py`'s attack posture needs to stop hand-picking a
version number that a live recalibration lineage can walk into).

---

## What needs a DB and couldn't be fully closed out this pass

- Finding 4's exact count of `avail=0.0`-then-started mispredictions (would need a query
  against a real walk-forward DB with `backtest_gameweek_steps` populated).
- §9's crash reproduced live (blocked by the sandbox's destructive-action guard on rebuilding
  `db/fpl_quant_v2.duckdb` from scratch; a `.venv` ingestion run against the real FPL API did
  succeed for everything up to the point where a stale local DB file — unrelated pre-existing
  local state, not a production issue — hit its own, different immutability conflict).
- Whether `parameters_backtested`'s fix (reading from the seed file) would change the final
  count meaningfully vs. some families being double-counted across shared param families
  (`model_decay_params` xi/rho, noted in `run_report.py`'s own comment as already a known sharp
  edge for this exact panel).

## Recommended reading order for Phase 2

The user's original priority order (fix the objective → per-segment gate → effect-size gate →
real λ study → minutes floor) still makes sense, but two items should move up given what this
pass found:

1. **The recalibration confirmation gate (effect-size + sign + collision check, Finding 5 +
   §9)** — this is actively breaking a shipped feature right now, not a theoretical risk.
2. **The parameters_backtested counter (Finding 6)** — cheap, mechanical, and currently telling
   the user "nothing is calibrated" when that's false; actively undermines trust in a
   correctly-*more*-calibrated model.
3. Then the calibration-objective work (Findings 1-2) as originally proposed.
4. The captain-path unification (Finding 3 + Finding 7 combined) — one fix
   (`build_captain_recommendation()` used everywhere `is_captain`/`captain_name` is surfaced,
   including `build_report()`'s headline) closes two findings at once.
5. Real λ study, minutes floor — unchanged from the original plan.

Full ranked fix proposals with measurement plans belong in Phase 2 proper; this document is
Phase 1 only, per the brief.
