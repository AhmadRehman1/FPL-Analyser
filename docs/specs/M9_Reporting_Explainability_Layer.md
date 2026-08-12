# M9 — Reporting / Explainability Layer

**Status: FROZEN**

Cross-cutting — depends on M0 through M8. Specified last, but `LESSONS_LEARNED.md` explicitly recommends pulling this scaffolding forward to build *alongside* M3 rather than deferring implementation to the end; specification order does not have to match implementation order (consistent with M0's own design review).

---

## Research findings

- Predicted XI's free-text `Reasoning` field and Injury DB's structured fields are exactly the kind of human-readable context that should flow through to a squad report, not just inform the numbers silently — the workbook already models a distinction between structured claim fields and narrative justification that M9 should preserve rather than collapse into a single opaque score.
- The project has accumulated a substantial, explicitly tracked list of invented-not-derived parameters across every module (`ξ`, `ρ`, tier weights, `minutes_adjustment_params`, `τ`, `ρ_residual`/`Z_fixture`, `λ`, `N=3` cap, `κ_TC`, `H=5`) — unusually good raw material for a transparency feature most FPL tools don't have. Burying it defeats the purpose of having tracked it so carefully through nine modules of freezes.
- M5's guardrails and the mandatory `λ=0`-vs-`λ=0.15` sanity check are internal safety mechanisms with no human-facing surface yet — a human reviewing a squad currently has no way to see that the concentration-avoidance machinery actually fired for their specific run.

---

## Locked specification

### Disclosure pattern

**Minimal headline view by default** — squad, captain, total projected EP, one-line rationale. Every other section (category breakdown, risk quantiles, guardrail audit trail, evidence provenance, parameter transparency panel, backtest summary, chip rationale) is **expandable on demand**, not shown by default.

### Core report sections (all behind the expand)

1. **Category-level EP breakdown** per player (appearance, goals/assists, clean sheet, DefCon, bonus), from M3's sub-models directly — not a single blended number, so a human can see e.g. "this defender's value is mostly DefCon-driven."
2. **Risk display**: M4's Cornish-Fisher floor/ceiling quantiles, carried through with their "unvalidated pending per-gameweek panel reconciliation" caveat attached at the display layer, not buried only in internal docs.
3. **Guardrail/audit trail**: which M5 guardrails bound for this specific run (captain-not-GK, per-club XI cap) and the pass/fail result of the `λ=0`-vs-`λ=0.15` divergence check for this run — structural evidence the concentration-avoidance mechanism fired.
4. **Evidence provenance panel**: for any player, the underlying `evidence_claims` behind their minutes/injury adjustment — source, confidence, reliability tier, `information_type` (FACT/OPINION). Community/analyst/YouTube evidence (never auto-consumed, per M1b) surfaced as qualitative context only, clearly labeled as not affecting the model's numbers.
5. **Assumptions/parameter transparency panel**: every versioned parameter active in this run, flagged as either **backtested/recalibrated via M7** or **still literature/invented default**.
6. **Backtest performance summary**: M7's tiered (cold/warm/mature) metrics, both log score and Brier score, so a human sees the system's actual track record, not just this week's output in isolation.
7. **Transfer/chip rationale**: from M8, including GW19 deadline urgency flagging when a chip from set 1 remains unused.

### Sanity-check surface — two distinct components

- **Automated flags**: the system runs its own pattern-detection heuristics matching the specific historical failure signatures (team/position concentration, captained GK, guardrail-binding status, `λ=0` divergence result) and surfaces pass/fail per heuristic.
- **Explicit human prompt**: a separate, clearly-labeled section inviting the person's own judgment — framed as a question ("does this squad look defensible to you?"), not a system self-certification. The automated flags inform that judgment; they do not replace it. This distinction is deliberate: the original `λ=0` bug passed whatever internal checks existed at the time, so the human prompt exists precisely because automated self-certification alone already failed once in this project's history.

### Integration pattern

**Adapter functions per module**: each of M0–M8 exposes its own `explain()`-style interface returning its relevant data in a display-ready shape. M9 does not reach into other modules' internals directly — loose coupling, so a schema change inside any module only requires updating that module's own adapter, not M9's central logic.

---

## Self-critique / known limitations

- Tiered disclosure is a real design tension: showing every caveat and parameter flag risks overwhelming a casual user and undermining trust even in well-calibrated parts of the system. The minimal-headline-plus-expand structure addresses this in principle, but the actual interaction design (exact UI, information hierarchy within the expanded view) is out of scope for a backend-module spec and left to implementation.
- M9's cross-cutting nature means it depends on adapters from every other module — even with loose coupling, a sufficiently large schema change upstream could still require adapter updates across several modules simultaneously. Worth flagging as an ongoing maintenance cost, not a one-time integration cost.
- This spec was frozen last in sequence despite `LESSONS_LEARNED.md`'s explicit recommendation to build its scaffolding early, alongside M3 — a deliberate deviation in specification order, named plainly here rather than left unremarked, though implementation order remains free to differ from specification order.

---

## Design review

**Alternative considered:** a thin pass-through report — just M5's chosen squad and M3's total EP numbers, no category breakdown or provenance detail.

**Rejected because:** it would not give a human reviewer enough surface area to actually apply the "could I beat this by eye" test, which is the entire reason this module exists given the project's history. A bare numbers dump would quietly recreate the same blind-trust failure that produced the original `λ=0` bug, just relocated from the optimizer to the reporting boundary instead of fixed.
