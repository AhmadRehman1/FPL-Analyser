# 2026-09 model failure diagnosis — Workstreams B & C (chip policy, scoring rules)

Investigation-only pass (two parallel read-only subagents), companion to
`2026-09_model_failure_diagnosis.md` (calibration, Findings 1-8 + §9). No production code
changed in this pass.

---

**2026-09-13/14 update — the magnitude-floor half of Workstream B and the cheap-wiring half
of Workstream C are now fixed, merged to master:**
- Workstream B's missing gain thresholds for `triple_captain`/`bench_boost`: fixed in
  [#173](https://github.com/AhmadRehman1/FPL-Analyser/pull/173). **Only the magnitude floor**
  ("is this worth firing at all") — the fuller ask (a season-horizon option-value estimate,
  "is this the best week to wait for") is **not built**, flagged as a follow-up task (larger,
  needs its own walk-forward validation).
- Workstream C's recalibration-wiring gap (vice-captain fallback missing from
  `refit_lambda()`/`report_concentration_sensitivity()`): fixed in
  [#172](https://github.com/AhmadRehman1/FPL-Analyser/pull/172).
- **Not fixed, deliberately deferred** (both need their own walk-forward validation before
  going live, unlike the pure wiring/accuracy fixes above): wiring
  `bench_quality_params_version`/`risk_posture_params_version`/etc. into the walk-forward's own
  per-step `squad_optimizer.run()` calls; the full real-auto-substitution build (bench order +
  formation-legality simulation); the stale vice-captain caveat in
  `docs/reports/2025-26_retrospective_validation.md` (still says "no vice-captain logic
  anywhere," which stopped being true 2026-09-07) — that report needs regenerating, not just a
  text correction, so it's left alone rather than hand-edited.

## Workstream B — the chip policy has no concept of patience

All three of the user's numbered claims **confirmed**, no corrections needed — if anything,
the situation is worse than framed on two of the three.

1. **`evaluate_triple_captain()` (`transfer_planner.py:951-989`) is unconditional — confirmed
   exactly.** The only guard is an empty-candidate-pool check; past that, `recommended: True`
   fires regardless of how small (or negative) the winning `tc_score` is. **Worse than
   framed: `evaluate_bench_boost()` has the identical shape** — same unconditional
   `recommended: True`, no magnitude floor. So this isn't a Triple-Captain-specific gap, it's
   structural across both non-swept chips.

   A second, previously-unknown decision surface makes this concrete: `run_transfer_planner_
   for_real_squad.py::reconcile_chips_with_timing_sweep()` applies a real chip-timing-sweep
   veto to wildcard/free_hit/bench_boost, but **explicitly hardcodes an exemption for
   triple_captain** — `"per-week call, not swept"`, with a code comment calling it
   *"a genuine week-by-week call... not a play-once-optimally chip."* That's a considered
   design decision somewhere in this codebase, not an oversight. A fix has to consciously
   override that rationale, not just patch a gap nobody thought about.

2. **No threshold params for TC/BB — confirmed exactly.** `wildcard_gain_threshold_params.
   min_horizon_gain=8.0` and `free_hit_gain_threshold_params.min_horizon_gain=1.5` both exist
   and gate real `recommended = gain > threshold` checks. No equivalent family exists for
   `triple_captain` or `bench_boost` anywhere in `src/fpl_quant/`.

3. **No option-value/optimal-stopping model — confirmed exactly.** The suggested grep returns
   zero matches. `gw19_urgent_flag` is confirmed to be a pure calendar countdown to the GW19
   chip-set-1 deadline, with no value-of-waiting logic. **Worse than framed: the asymmetry is
   three-tiered, not two.** Wildcard gets a real, independent, full-season-horizon sweep
   (`chip_timing_analysis.py`, live in `data/chip_timing/chip_timing_latest.json`). Bench
   Boost gets `bench_boost_window()` — a narrow ~3-4 gameweek window anchored to *wherever
   Wildcard was forced*, not an independent sweep. Triple Captain gets nothing at all.

**Cost of the fix is smaller than it looks.** TC/BB evaluation never calls the MIQP solver
(unlike Wildcard's sweep, which re-solves at every candidate week) — it's a pure read over
already-computed `monte_carlo_player_summary`/EP data for the existing squad. That number is
already computed every single gameweek inside `transfer_planner.run()` and thrown away
(`forward_season_sim.GameweekResult` doesn't even carry a `tc_score`/`bench_ep_sum` field
today). Building "expected max over remaining eligible gameweeks" is an aggregation over data
that already exists, structured like the Wildcard sweep already shipped — genuinely a smaller
build than what's already in production for Wildcard.

**The live cost of not having this**: GW3 Free Hit (a *swept* chip, functioning correctly)
returned +39 vs. a hold counterfactual. GW4 Triple Captain (unswept, ungated) projected 62.15
and realized 30.0 — a −32-point swing in one gameweek from firing a once-per-season chip with
no season-horizon comparison. Chips are the highest-variance, highest-magnitude decisions this
model makes.

## Workstream C — the backtest is scoring a game that isn't FPL

**Correction to lead with, per the "if any claim is wrong" instruction — this one is on the
user's premise, not on Workstream B.** The retrospective report's caveat that "there is no
vice-captain logic anywhere in `backtest.py`" is **stale, not currently true**. Commit
`171dc5c` (2026-09-07) added real vice-captain armband-transfer logic to
`_realized_xi_points()` and threaded it through most (not all — see below) call sites. The
committed report (`docs/reports/2025-26_retrospective_validation.md`, last touched 2026-08-28)
predates that fix by 9 days and has never been regenerated. Its 1,575-point / 6.9th-percentile
headline reflects the pre-fix, no-vice-fallback behavior regardless of what the caveat text
currently says. **The bench/auto-sub half of the original claim is fully confirmed** — that
part of the report is accurate and still true today.

- `_realized_xi_points()` (`backtest.py:2213-2260`, read in full): iterates only the `xi_uids`
  frozenset passed in — bench players' `event_points` are never read under any circumstances.
  No auto-sub logic exists. Vice-captain: armband moves to `vice_captain_uid` only when the
  captain's own `minutes == 0` exactly, otherwise the captain keeps the multiplier no matter
  how few points they score — this part matches the real FPL rule.
- **Real FPL auto-sub rules** (confirmed via web search): trigger is exactly 0 minutes played;
  substitution follows the manager's bench order (1st/2nd/3rd sub) but skips an eligible sub if
  it would break formation legality (1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD); the bench GK only ever
  replaces the starting GK. Vice-captain: armband transfers only on the captain's exact 0
  minutes.
- **This is a real data-model gap, not a small scoring-function patch.**
  `squad_optimizer_selections` has no bench-order column at all — `solve()` currently produces
  undifferentiated squad/xi/captain/vice binaries with no ordering among the 4 bench slots.
  Real auto-sub needs that new ordering decision inside the MIQP *and* a formation-legality
  simulation function. Two explicit disclosures already exist in the codebase admitting this
  exact gap (`reporting.py:1035-1042`, `field_rank.py:275-278`) — this has been a known,
  named limitation, not a hidden one.
- **Blast radius**: 11 call sites of `_realized_xi_points()`, split by whether they got the
  2026-09-07 vice fix — the live model team, the headline beats-crowd metric, and
  `run_season_simulation()` did; **`refit_lambda()` and `report_concentration_sensitivity()`
  (the M7 recalibration search loops for `lambda_value` and the concentration cap) did not.**
  Affected committed artifacts on any future fix: the retrospective report (needs
  regeneration + explicit pre/post labeling regardless of the bench question),
  `app_track_record.json`, `app_model_team.json`, `data/report_history/*.json`.

### The crux finding: the solve-time incentive question

Not purely a measurement problem, and not purely a solve-time incentive problem — **both, and
the solve-time half is a wiring gap, not a missing mechanism.**

- `squad_optimizer.solve()`'s linear-EP objective is built only from `xi[uid]`/`captain[uid]`
  — a benched player contributes **zero** to the objective regardless of their own EP. The one
  bench-relevant mechanism, `min_bench_p_start_probability`, is a one-sided exclusion floor
  (stops an obviously-risky player from being benched instead of dropped), never a reward for
  bench upside.
- **That guardrail — along with `risk_posture_params_version`, `field_covariance_params_version`,
  `concentration_risk_params_version` — is never even passed at the two backtest/walk-forward
  call sites of `squad_optimizer.run()`** (`backtest.py:308-311`, `backtest.py:1130-1133`); a
  repo-wide grep across the whole backtest stack for those four param names returns zero
  matches. They're wired only in live production (`run_ingestion.py`) and one reporting path.
  `run_season_simulation()`'s own signature doesn't expose a way to pass them through at all.
- **Compounding effect on Workstream A**: `refit_lambda()`/`report_concentration_sensitivity()`
  score every lambda/concentration-cap candidate via `realized_sharpe` built on
  `_realized_xi_points()` — which structurally cannot see a bench player's points. So even if
  someone wired the bench guardrail into the walk-forward solve tomorrow, **recalibration could
  not detect whether it helped**: a strong bench and a garbage bench produce byte-identical
  `realized_sharpe` for every candidate in the grid, as long as XI/captain are unchanged. The
  feedback loop is blind on both ends right now, independently of each other.

**Estimate**: the full real-auto-sub build (bench order + formation-legality simulation) is a
genuinely larger data-model change, not a small patch — contrast with the vice-captain fix,
which was small specifically because `is_vice` already existed unused in the schema. But the
**recalibration-wiring gaps found alongside it** (thread `vice` into `refit_lambda()`/
`report_concentration_sensitivity()`; pass `bench_quality_params_version` and friends into the
two backtest solve call sites) are genuinely small, mechanical, no-schema-change fixes — and
they directly de-bias Workstream A's own recalibration work.

---

## Recommendation: which workstream is worth the most points

**Workstream B (chip policy), by a clear margin — do this one first.**

Reasoning:

- **Magnitude and directness.** A chip mistiming is a single-gameweek, tens-of-points swing we
  already watched happen (GW4 TC: −32 vs. projection). Calibration fixes (Workstream A) nudge
  every player's EP by fractions of a point, spread across 657 players and diluted by the
  optimizer's own diversification — the aggregate walk-forward gap is already down to
  **−0.90 pts/GW** after the Sept prior-work stack (#124-135), meaning most of the large,
  cheap calibration wins are already banked; what's left is real but smaller-marginal. A bench/
  auto-sub fix (Workstream C) only pays out on the subset of weeks a starter blanks with a
  usable bench alternative sitting behind them — real, but rarer and smaller per event than a
  mistimed chip.
- **Structural, not statistical.** TC and BB aren't miscalibrated, they're **ungated** — every
  single eligible week, for two of the four chips, "fire" is the only path the code can reach.
  That's a certainty of eventually firing on a bad week, not a probabilistic bias to correct at
  the margin.
- **Cost-to-fix is low relative to payoff.** Per the brief, the expensive infrastructure
  (asof walking, per-GW EP computation) already exists for both chips; what's missing is two
  threshold param families plus an aggregation over numbers already being computed and
  discarded. This is cheaper than Workstream C's full auto-sub build and arguably cheaper than
  the remaining calibration work in Workstream A.

**Second priority: the cheap half of Workstream C — not the whole thing.** Threading `vice`
into `refit_lambda()`/`report_concentration_sensitivity()` and wiring
`bench_quality_params_version`/`risk_posture_params_version` into the two backtest solve call
sites are small, mechanical, no-schema-change fixes with outsized value: they directly fix
what Workstream A's own recalibration is being validated against. Treat this as close to a
*prerequisite* for trusting any future lambda/concentration recalibration, not as competing
with Workstream A. Regenerating the retrospective report (correcting the now-stale
vice-captain caveat, even before any bench fix) is a same-tier, cheap, honesty fix.

**Lower priority for now: the full real-auto-sub build** (bench order + formation-legality
simulation) and **the remaining Workstream A calibration-objective work** (Findings 1-2's
decision-weighted metric). Both are real and worth doing, but genuinely larger builds for a
smaller marginal payoff than B, given how much of Workstream A's low-hanging fruit is already
picked and how rare/small a single auto-sub event is compared to a mistimed chip. Two items
from Workstream A should still happen immediately regardless of this ranking, because they're
free and already broken: **Finding 6's counter fix** (cheap, restores trust in reporting) and
**§9's Attack-posture collision fix** (a currently-live production bug with users depending on
that feature).
