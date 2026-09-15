"""Collects the two scripts/run_squad_optimizer_wiring_sensitivity_arm.py outputs ("off" vs
"on") into one comparison -- the walk-forward evidence the deferred task requires before
squad_optimizer.run()'s solve_* params (see backtest.run_gameweek_step()'s/
run_season_simulation()'s own docstrings) go from opt-in to any default.

Reads data/squad_optimizer_wiring_study/arms/*.json and writes:

  data/squad_optimizer_wiring_study/squad_optimizer_wiring_study_latest.json -- machine-readable
  data/squad_optimizer_wiring_study/SUMMARY.md                               -- the human table
                                                                                 + evidence-only
                                                                                 reading

This does NOT flip any default. It states, plainly, whether "on" regresses the headline
beats_crowd_points_delta metric relative to "off" -- the specific bar the deferred task set --
and lets a human decide whether wiring these four families in is worth it. No recommendation
beyond that regression check is baked in.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARMS_DIR = REPO_ROOT / "data" / "squad_optimizer_wiring_study" / "arms"
OUT_DIR = REPO_ROOT / "data" / "squad_optimizer_wiring_study"


def _load_arms() -> dict[str, dict]:
    arms = {}
    for path in sorted(ARMS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            arms[data["arm"]] = data
        except (OSError, ValueError, KeyError) as exc:
            print(f"::warning::could not read {path.name}: {exc}")
    return arms


def _fmt(v, spec="{:+.3f}"):
    return spec.format(v) if isinstance(v, (int, float)) else "-"


def main() -> None:
    arms = _load_arms()
    if not arms:
        raise SystemExit(f"no arm files in {ARMS_DIR} -- nothing to aggregate")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "squad_optimizer_wiring_study_latest.json").write_text(json.dumps({"arms": arms}, indent=2))

    md = ["# squad_optimizer solve-time wiring sensitivity study", ""]
    md += [
        "Full M7 walk-forward (`backtest.run()`), both historical seasons. \"off\" = current "
        "production behavior (the four families never activated at either squad_optimizer.run() "
        "call site in backtest.py); \"on\" = adds ownership/risk_posture/field_covariance/"
        "bench_quality/concentration_risk at their un-recalibrated v1 defaults.", "",
    ]

    off, on = arms.get("off"), arms.get("on")
    md += ["## beats_crowd_points_delta by tier", ""]
    md += ["| tier | off | on | delta | verdict |", "|---|---|---|---|---|"]
    if off and on:
        tiers = sorted(set(off["beats_crowd_points_delta_by_tier"]) | set(on["beats_crowd_points_delta_by_tier"]))
        for tier in tiers:
            off_m = off["beats_crowd_points_delta_by_tier"].get(tier, {}).get("mean")
            on_m = on["beats_crowd_points_delta_by_tier"].get(tier, {}).get("mean")
            if off_m is None or on_m is None:
                md.append(f"| {tier} | {_fmt(off_m)} | {_fmt(on_m)} | - | incomplete |")
                continue
            delta = on_m - off_m
            verdict = "REGRESSION" if delta < 0 else "improvement" if delta > 0 else "tie"
            md.append(f"| {tier} | {_fmt(off_m)} | {_fmt(on_m)} | {_fmt(delta)} | {verdict} |")
    else:
        md.append("| - | - | - | - | one or both arms missing |")

    md += ["", "## segment_calibration (mature tier, signed resid = realized - predicted)", ""]
    md += ["| segment | off resid | on resid | delta (closer to 0 is better) |", "|---|---|---|---|"]
    if off and on:
        segs = sorted(set(off["segment_calibration_mature"]) | set(on["segment_calibration_mature"]))
        for seg in segs:
            off_r = off["segment_calibration_mature"].get(seg, {}).get("mean_resid")
            on_r = on["segment_calibration_mature"].get(seg, {}).get("mean_resid")
            if off_r is None or on_r is None:
                md.append(f"| {seg} | {_fmt(off_r)} | {_fmt(on_r)} | - |")
                continue
            md.append(f"| {seg} | {_fmt(off_r)} | {_fmt(on_r)} | {_fmt(abs(on_r) - abs(off_r))} |")
    else:
        md.append("| - | - | - | - |")

    md += ["", "## Reading the evidence", ""]
    if off and on:
        deltas = [
            on["beats_crowd_points_delta_by_tier"][t]["mean"] - off["beats_crowd_points_delta_by_tier"][t]["mean"]
            for t in off["beats_crowd_points_delta_by_tier"]
            if t in on["beats_crowd_points_delta_by_tier"]
        ]
        any_regression = any(d < -1e-9 for d in deltas)
        if any_regression:
            md.append(
                "- **\"on\" regresses beats_crowd_points_delta on at least one tier.** Per the "
                "deferred task's own bar (\"must not be merged without a real walk-forward "
                "comparison showing it doesn't regress the headline beats_crowd_points_delta "
                "metric\"), this wiring should stay off by default until the regression is "
                "understood (a different param version, not just v1, may be needed -- v1's "
                "eo_weight_kappa=0.02/concentration kappa=0.0/field_covariance kappa=0.001 are "
                "the ORIGINAL invented, un-recalibrated defaults, not tuned values)."
            )
        else:
            md.append(
                "- **\"on\" does not regress beats_crowd_points_delta on any tier** -- the "
                "specific bar the deferred task set. That clears the way to consider defaulting "
                "this wiring on, but is not by itself a recommendation to do so (segment_"
                "calibration and divergence-check pass rates above are also worth a human look "
                "before deciding)."
            )
    else:
        md.append("_One or both arms missing -- no comparison possible._")
    md += [
        "", "_This is evidence, not a decision. squad_optimizer.run()'s five Priority 1/2 terms "
        "stay unwired-by-default at both backtest.py call sites until the project owner reviews "
        "this and explicitly opts a caller in._",
    ]
    (OUT_DIR / "SUMMARY.md").write_text("\n".join(md))

    print(f"[aggregate] {len(arms)} arm(s) -> {OUT_DIR/'SUMMARY.md'}, {OUT_DIR/'squad_optimizer_wiring_study_latest.json'}")
    print("\n".join(md))


if __name__ == "__main__":
    main()
