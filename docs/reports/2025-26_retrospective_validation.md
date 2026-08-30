# Retrospective Validation: 2025-26 Season, Engine vs. Real Managers

**Status: unreviewed draft.** Per this plan's own interview record (Q6), this report is not
linked from or wired into any public page. It exists for human review before any public or
marketing claim is made from it — see `docs/plans/2026-08_retrospective_validation_and_ml_decision_plan.md`.

## Headline number

The engine's own simulated 2025-26 season strategy (see Methodology below for exactly what this
does and does not mean) scored **1,575.0 points**, placing at the
**6.9th percentile** against a sample of 2,000
real FPL managers' actual 2025-26 season totals.

| Metric | Value |
|---|---|
| Engine's simulated total | 1,575.0 |
| Real sample size | 2,000 |
| Percentile rank | 6.9 |
| Point differential vs. real mean (1,946.4) | -371.4 |
| Point differential vs. real median (1,995.0) | -420.0 |
| Real sample range | 82.0 – 2,457.0 |

## Methodology — read this before citing the number above anywhere

This report answers a narrower, more honest question than "would this engine have beaten real
managers' actual decisions" — that literal question cannot be answered with real data (see (a)
below). What it does answer:

**(a) This is NOT a replay of any real manager's actual decisions.** The comparison is the
engine's own strategy, played out by itself from its own bootstrap squad, against real managers'
actual final outcomes. It is not, and cannot be, "we took over real manager X's real squad and
made different choices for them" — the FPL public API does not serve per-gameweek historical
picks for a completed past season (verified live this session: a real request for a past
season's gameweek returns HTTP 404; only season-level *totals*, which the real comparison sample
above uses, survive for past seasons). Any framing of this number as "we replayed real managers'
decisions and beat them" is not an honest reading of what was measured.

**(b) This is a genuinely blind simulation, not a hindsight-tuned one.** All 18 parameter-version
arguments `run_season_simulation()` requires were hardcoded to version 1 — including the 8
families (`xi`, `rho`, `rho_residual`, `adjustment`, `shrinkage`, `fact_multiplier`, `lambda`,
`kappa_tc`) that are normally resolved from whatever is *currently* confirmed-active via
automated recalibration. Those currently-active versions were fit in part against 2025-26's own
real outcomes, so using them here would have quietly given the engine hindsight it never
genuinely had. Every version actually used:

```
{
  "xi_params_version": 1,
  "rho_params_version": 1,
  "decay_params_version": 1,
  "adjustment_params_version": 1,
  "shrinkage_params_version": 1,
  "fact_multiplier_params_version": 1,
  "scoring_params_version": 1,
  "bps_params_version": 1,
  "tau_params_version": 1,
  "rho_residual_params_version": 1,
  "corr_params_version": 1,
  "lambda_params_version": 1,
  "guardrail_params_version": 1,
  "horizon_params_version": 1,
  "transfer_cost_params_version": 1,
  "wildcard_threshold_params_version": 1,
  "free_hit_threshold_params_version": 1,
  "kappa_tc_params_version": 1
}
```

**(c) This starts at GW2, not a true GW1.** Building this phase for real found that no FPL season
in this database has a pre-GW1 price snapshot — `squad_optimizer.fetch_candidate_pool()` needs a
real `now_cost` per player, and `asof_scope()` correctly shows zero rows before a season's own
GW1. A true GW1 bootstrap is therefore not possible for *any* season via this mechanism, which is
why the codebase's own existing `scripts/run_season_simulation.py` already defaults to
`START_GAMEWEEK = 2`. This report's number reflects the engine's strategy from GW2 onward, not a
full 38-gameweek season from a literal opening gameweek.

**(d) The real-manager comparison sample is a genuinely random, anonymized population sample —
not the world's elite.** 2,000 real entries were drawn via
uniform-random `entry_id` sampling over the live FPL API's real ID space, not via any
leaderboard/standings-ranked source — see `src/fpl_quant/retrospective_validation.py` for why
that distinction matters. No entry ID, name, or team name was retained anywhere in this pipeline,
including in logs. This method has one disclosed, not fully resolved, limitation: uniform-random
sampling over the ID space skews toward accounts by signup recency rather than by engagement, so
it may still somewhat over-represent dormant, rarely-managed accounts relative to actively-played
ones — the sample's mean (1,946.4) and median
(1,995.0) were sanity-checked against plausible general-population FPL
score ranges before this report was generated, which is a backstop, not a guarantee this skew is
fully absent.

**(e) The engine's weekly score does not model FPL's automatic substitution rule.** Real FPL
promotes a bench player into the starting XI when a picked starter scores 0 minutes (injury, late
withdrawal, rotation), so a real manager's actual score is often higher than their pre-deadline
XI alone would suggest. `backtest._realized_xi_points()` — the function this simulation's every
weekly score comes from — sums only the picked starting XI's real `event_points` (captain
doubled), with no substitution logic: a starter who blanks with 0 minutes counts as 0 here, the
same as it would count in the picked XI's raw total, but a real manager holding an identical squad
could have scored more via a bench player their engine never gets credit for. This makes the
engine's simulated total above conservative *in expectation* relative to what an identical real
squad's real FPL score would have been — not a hard per-gameweek guarantee, since a player who
did play can still score negative points (red card, own goals, heavy defensive concession), a
case where "count the blank as 0" would instead be a slight overestimate for that week. In
aggregate over a full season the effect is conservative, but "systematically" would overstate
the certainty this mechanism actually provides.

**(f) The engine's captaincy has no vice-captain fallback.** Real FPL automatically promotes the
vice-captain's points if the captain doesn't play; `_realized_xi_points()` takes a single fixed
`captain_uid` and simply doubles whatever that player scored, including a blank — there is no
vice-captain logic anywhere in `backtest.py` for this scoring path. (This is a distinct, verified
gap from (e), not a restatement of it: this project already discloses the same class of issue for
a *different* function, `reporting.py`'s counterfactual transfer-decision scoring, which does
apply a baseline-vice-captain fallback in that one specific case — this simulation's own weekly
scoring has no equivalent, in either direction.) The net effect skews the same direction as (e)
(conservative, in expectation) but is a separate, independently real simplification, not covered
by fixing (e) alone.

## What this number can and cannot honestly support

**Can support:** "the engine's own season-long strategy, played blind from GW2 with no
hindsight, would have placed at roughly the 7th percentile
among a large, genuinely random sample of real FPL managers' actual final outcomes."

**Cannot support:** "the engine beat real managers' actual decisions" (it never saw their
decisions); "the engine played a full season from GW1" (it started at GW2); "the engine's advice
would have improved a specific real person's actual season" (that is a different, complementary
question — Track C's decision-log mechanism for the 2 already-tracked real managers, not this
retrospective comparison).

## Data provenance

- Real sample: data/retrospective/2025-2026_real_season_totals.json (sampled at 2026-08-27T21:26:17Z, n_attempted=2894, rejection_rate=0.308914996544575)
- Engine simulation: data/retrospective/2025-2026_engine_simulation.json (run at 2026-08-28T14:34:10Z, GW2-38, wall_clock_seconds=23882)
- Skipped double-gameweeks during the engine's simulation: [26, 33, 36]
