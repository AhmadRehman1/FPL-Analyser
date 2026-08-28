"""Track E, Phase E-3: compute the comparison and write the report. See
docs/plans/2026-08_retrospective_validation_and_ml_decision_plan.md for the full design.

Reads Phase E-1's cached real sample (data/retrospective/2025-2026_real_season_totals.json) and
Phase E-2's cached engine simulation (data/retrospective/2025-2026_engine_simulation.json),
computes the comparison via fpl_quant.retrospective_validation.compute_retrospective_comparison,
and writes docs/reports/2025-26_retrospective_validation.md.

R5 (this plan's own hard requirement): the report's methodology section must state, explicitly
and plainly:
  (a) this compares the engine's own from-GW2 strategy against real managers' actual outcomes --
      it is NOT a replay of any real manager's actual decisions (the FPL API cannot supply that
      for a past season -- verified live this session, see the plan's Current State);
  (b) the simulation used default, pre-recalibration parameters (all 18 required version kwargs
      hardcoded to 1), not the currently-active, hindsight-tuned ones;
  (c) the simulation starts at GW2, not a true GW1 -- no season has a pre-GW1 price snapshot in
      this schema (verified live), so a true GW1 bootstrap is not possible for any season.
This script does not touch track-record.html or index.html -- Q6 of the plan's interview
requires human review before any public claim, so this report is written but never published.

Usage (from repo root):
    .venv/Scripts/python scripts/run_retrospective_report.py
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import retrospective_validation as rv  # noqa: E402

SAMPLE_FILE = REPO_ROOT / "data" / "retrospective" / "2025-2026_real_season_totals.json"
SIMULATION_FILE = REPO_ROOT / "data" / "retrospective" / "2025-2026_engine_simulation.json"
REPORT_FILE = REPO_ROOT / "docs" / "reports" / "2025-26_retrospective_validation.md"


def _fmt(x: float) -> str:
    return f"{x:,.1f}"


def main() -> None:
    if not SAMPLE_FILE.exists():
        raise SystemExit(f"Phase E-1's cached sample not found at {SAMPLE_FILE} -- run scripts/run_retrospective_sample.py first.")
    if not SIMULATION_FILE.exists():
        raise SystemExit(f"Phase E-2's cached simulation not found at {SIMULATION_FILE} -- run scripts/run_retrospective_engine_simulation.py first.")

    sample = json.loads(SAMPLE_FILE.read_text(encoding="utf-8"))
    simulation = json.loads(SIMULATION_FILE.read_text(encoding="utf-8"))

    real_totals = sample["totals"]
    engine_total = simulation["total_points"]
    comparison = rv.compute_retrospective_comparison(engine_total, real_totals)

    param_versions = simulation["param_versions_used"]
    assert len(param_versions) == 18, f"expected 18 param versions in the simulation cache, got {len(param_versions)}"
    assert all(v == 1 for v in param_versions.values()), "every param version must be 1 for a genuinely blind simulation -- refusing to write a report otherwise"
    assert simulation["start_gameweek"] == 2, f"expected the simulation to start at GW2 (see plan's Current State), got GW{simulation['start_gameweek']}"

    report = f"""# Retrospective Validation: 2025-26 Season, Engine vs. Real Managers

**Status: unreviewed draft.** Per this plan's own interview record (Q6), this report is not
linked from or wired into any public page. It exists for human review before any public or
marketing claim is made from it — see `docs/plans/2026-08_retrospective_validation_and_ml_decision_plan.md`.

## Headline number

The engine's own simulated 2025-26 season strategy (see Methodology below for exactly what this
does and does not mean) scored **{_fmt(engine_total)} points**, placing at the
**{comparison['percentile_rank']:.1f}th percentile** against a sample of {comparison['n_real_sample']:,}
real FPL managers' actual 2025-26 season totals.

| Metric | Value |
|---|---|
| Engine's simulated total | {_fmt(engine_total)} |
| Real sample size | {comparison['n_real_sample']:,} |
| Percentile rank | {comparison['percentile_rank']:.1f} |
| Point differential vs. real mean ({_fmt(comparison['real_mean'])}) | {comparison['point_differential_vs_mean']:+.1f} |
| Point differential vs. real median ({_fmt(comparison['real_median'])}) | {comparison['point_differential_vs_median']:+.1f} |
| Real sample range | {_fmt(comparison['real_min'])} – {_fmt(comparison['real_max'])} |

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
{json.dumps(param_versions, indent=2)}
```

**(c) This starts at GW2, not a true GW1.** Building this phase for real found that no FPL season
in this database has a pre-GW1 price snapshot — `squad_optimizer.fetch_candidate_pool()` needs a
real `now_cost` per player, and `asof_scope()` correctly shows zero rows before a season's own
GW1. A true GW1 bootstrap is therefore not possible for *any* season via this mechanism, which is
why the codebase's own existing `scripts/run_season_simulation.py` already defaults to
`START_GAMEWEEK = 2`. This report's number reflects the engine's strategy from GW2 onward, not a
full 38-gameweek season from a literal opening gameweek.

**(d) The real-manager comparison sample is a genuinely random, anonymized population sample —
not the world's elite.** {comparison['n_real_sample']:,} real entries were drawn via
uniform-random `entry_id` sampling over the live FPL API's real ID space, not via any
leaderboard/standings-ranked source — see `src/fpl_quant/retrospective_validation.py` for why
that distinction matters. No entry ID, name, or team name was retained anywhere in this pipeline,
including in logs. This method has one disclosed, not fully resolved, limitation: uniform-random
sampling over the ID space skews toward accounts by signup recency rather than by engagement, so
it may still somewhat over-represent dormant, rarely-managed accounts relative to actively-played
ones — the sample's mean ({_fmt(comparison['real_mean'])}) and median
({_fmt(comparison['real_median'])}) were sanity-checked against plausible general-population FPL
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
hindsight, would have placed at roughly the {comparison['percentile_rank']:.0f}th percentile
among a large, genuinely random sample of real FPL managers' actual final outcomes."

**Cannot support:** "the engine beat real managers' actual decisions" (it never saw their
decisions); "the engine played a full season from GW1" (it started at GW2); "the engine's advice
would have improved a specific real person's actual season" (that is a different, complementary
question — Track C's decision-log mechanism for the 2 already-tracked real managers, not this
retrospective comparison).

## Data provenance

- Real sample: {SAMPLE_FILE.relative_to(REPO_ROOT).as_posix()} (sampled at {sample.get('sampled_at_utc', 'unknown')}, n_attempted={sample.get('n_attempted')}, rejection_rate={sample.get('rejection_rate')})
- Engine simulation: {SIMULATION_FILE.relative_to(REPO_ROOT).as_posix()} (run at {simulation.get('run_at_utc', 'unknown')}, GW{simulation['start_gameweek']}-{simulation['end_gameweek']}, wall_clock_seconds={simulation.get('wall_clock_seconds'):.0f})
- Skipped double-gameweeks during the engine's simulation: {simulation.get('skipped_dgw_gameweeks')}
"""

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    # encoding="utf-8" explicitly -- Path.write_text()'s default encoding is the platform's own
    # (the system codepage on Windows, not UTF-8), which would mangle every em-dash in this
    # report's own prose (found dry-running this script: "—" rendered as "�" on read-back).
    REPORT_FILE.write_text(report, encoding="utf-8")
    print(f"[run_retrospective_report] wrote {REPORT_FILE}")
    print(f"[run_retrospective_report] engine_total={engine_total} percentile_rank={comparison['percentile_rank']:.1f} "
          f"diff_vs_mean={comparison['point_differential_vs_mean']:+.1f}")


if __name__ == "__main__":
    main()
