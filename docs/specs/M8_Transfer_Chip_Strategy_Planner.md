# M8 — Transfer & Chip Strategy Planner

**Status: FROZEN**

Depends on: M5 (Squad Optimizer, as subroutine), M6 (Monte Carlo Simulation Engine, for chip-value estimation).

---

## Research findings

- Confirmed 2026-27 rules from `13_Rule Changes Database`/`13b`: 2 sets of 4 chips (Wildcard/Free Hit/Triple Captain/Bench Boost) = 8 total across the season; **set 1 must be used by the GW19 deadline (2 Jan 13:30 GMT), with no rollover into the second half**; up to 5 free transfers can be banked.
- The standard −4-points-per-extra-transfer hit is **not explicitly documented** anywhere in the workbook's rule tabs — only deltas from 25/26 are covered there, the same gap pattern found in M3's base scoring matrix.
- **Tabs `31_Captaincy-Transfer Plan` through `36_Top-N Final Lists` are all downstream products of `28_Master Rating Engine v2`**, which is on the kickoff prompt's explicit deprecated list. These six tabs weren't individually named as deprecated but are transitively built on a deprecated source, and should be excluded from M8's evidence base too — extending the stated deprecation principle rather than deviating from it. Recommended for addition to M0's exclusion allowlist.
- **Tab `33_10 Squad Variants` contains a genuinely valuable cautionary finding even while being excluded from live ingestion**: its own audit note documents that the old v1 approach — "maximize sum of composite rating under a £100m budget" — systematically excluded premium players like Haaland, because an uncapped 0–100 composite rating doesn't scale with price. Several cheap high-rated players always beat one expensive elite one on that objective. Direct historical evidence validating why M5 was built around an EP/price mean-variance formulation rather than a composite-rating approach.
- **DGW/BGW status is directly computable from the FPL-Core-Insights fixture data itself**, without relying on the workbook's Fixture Database (which tab 32 itself flags as not containing DGW/BGW information for this season). Checked programmatically: scanning all 38 gameweeks of the confirmed 2026-27 fixture list shows every gameweek has exactly 20 teams playing exactly once — **no doubles or blanks currently scheduled**. A snapshot-in-time finding, not permanent — cup rounds and European clashes routinely introduce DGW/BGW later in a real season.

---

## Locked specification

### Scope

M8 operates on an **existing squad** — distinct from M5's from-scratch problem — deciding transfers and chip timing over a multi-gameweek planning horizon. Calls M5 as a subroutine for "ideal squad" comparisons (Wildcard evaluation) and M6 for chip-value estimation via simulation (Triple Captain, Bench Boost).

### Free-transfer state

Modeled as a state variable evolving over the horizon: `free_transfers_available_t`, capped at 5, incrementing by 1 each gameweek not fully used, decrementing (and incurring the points-hit cost) for each transfer made beyond the free allocation.

### Transfer valuation over the horizon

```
transfer_value = Σ_{gw=t}^{t+H} (EP_new_player,gw − EP_old_player,gw) − transfer_cost
```

using M3's EP forecasts extended across the horizon, with M4's variance naturally widening for further-out gameweeks.

**Planning horizon**: pinned at **H = 5 gameweeks**, stored in a versioned `planning_horizon_params` table (`horizon_gameweeks`, `params_version`, `effective_date`).

### Chip timing

- **Wildcard**: call M5 fresh (frozen `λ`) for a full rebuild, compare its projected value over the remaining horizon against the current squad's projected value, recommend when the gain clears a versioned threshold.
- **Free Hit**: one-gameweek-only rebuild. Given the confirmed absence of DGW/BGW in the current fixture list, FH's primary use case for this season as currently scheduled is a single gameweek with an unusually poor fixture swing for the current squad — not DGW exploitation, the more commonly assumed use case. Re-evaluated against live fixture data at decision time, not assumed fixed from this finding.
- **Triple Captain**: risk-adjusted objective using M6's simulated outcome distributions (with vs. without doubling captain `i`'s points):

  ```
  TC_score_i = E[marginal_value_i] − κ_TC · StdDev[marginal_value_i]
  ```

  favoring nailed-on, high-floor picks over high-variance ones with equal or higher expected value. `κ_TC` is stored as its own versioned parameter — **separate from M5's `λ`**, since captaincy risk preference isn't necessarily identical to squad-construction risk preference — in a `tc_risk_aversion_params` table (`kappa_tc`, `params_version`, `effective_date`).
- **Bench Boost**: compare projected bench EP sum (from M3) across candidate gameweeks, recommend the gameweek maximizing it.
- **GW19 hard deadline constraint**: since chip set 1 is forfeited entirely if unused by the GW19 deadline, this is modeled as an explicit use-it-or-lose-it constraint in M8's horizon logic — flagging urgently as the deadline approaches if a chip from set 1 remains unused, not treated as a soft preference that can silently lapse.

### Points-hit value

**Under the same implementation-time verification gate as M3's base scoring matrix**: M8 is not implementation-ready until the actual points-hit-per-extra-transfer value is confirmed against current official FPL rules, not assumed as −4 from convention.

---

## Self-critique / known limitations

- The multi-period free-transfer banking problem is genuinely a sequential decision problem (a dynamic program, or at minimum a rolling-horizon lookahead) — this spec defines the state variables and objective but doesn't commit to a specific solution algorithm, reasonably left as an implementation-time choice.
- `κ_TC` is invented, not derived — same status as `λ` and the project's other risk-preference parameters, flagged for M7 recalibration.
- DGW/BGW absence is a snapshot finding from the current data pull, not a permanent property of the season — M8's chip logic must re-check fixture data at decision time rather than bake in "no DGWs this season" as a fixed assumption.

---

## Design review

**Alternative considered:** treating each gameweek's transfer decision myopically — best single transfer this week, ignoring banking dynamics entirely.

**Rejected because:** the confirmed 5-transfer banking rule makes saving transfers a genuinely valuable strategic lever a real FPL manager routinely uses. A myopic optimizer would systematically discard that lever, failing the project's own core benchmark of never producing an output a human could trivially beat by hand.
