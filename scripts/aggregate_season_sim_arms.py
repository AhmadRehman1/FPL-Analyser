"""Pairs every season-sim arm (scripts/run_season_sim_arm.py) with the 'control' arm of the same
season, gameweek by gameweek, and writes data/season_sim_arms/SUMMARY.md.

Per arm: net points per GW (hits deducted), FPL's real average over the same gameweeks, the
margin over it, and the paired per-GW difference from control with its standard error. Arms are
compared on net points because FPL's real average already includes managers' hits.

Usage (from repo root):
    python scripts/aggregate_season_sim_arms.py [arms_dir]
"""

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = REPO_ROOT / "data" / "season_sim_arms"


def _net_by_gw(arm: dict) -> dict[int, float]:
    return {gw: p - h for gw, p, h in zip(arm["gameweeks"], arm["weekly_points"], arm["weekly_hits"])}


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.2f}"


def paired_difference(arm: dict, control: dict) -> dict:
    """Mean and standard error of (arm net - control net) over the gameweeks both scored."""
    a, c = _net_by_gw(arm), _net_by_gw(control)
    diffs = [a[gw] - c[gw] for gw in sorted(set(a) & set(c))]
    if not diffs:
        return {"n": 0, "mean": None, "se": None}
    mean = sum(diffs) / len(diffs)
    if len(diffs) < 2:
        return {"n": len(diffs), "mean": mean, "se": None}
    var = sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)
    return {"n": len(diffs), "mean": mean, "se": math.sqrt(var / len(diffs))}


def summarize(arms: list[dict]) -> str:
    lines = ["# Season-simulation arms", "",
             "Evolving manager (transfers, hits, chips) over a real season with the live model team's",
             "settings; each arm changes one. Net = gross XI points minus transfer hits.", ""]
    for season in sorted({a["season"] for a in arms}):
        rows = [a for a in arms if a["season"] == season]
        control = next((a for a in rows if a["label"] == "control"), None)
        lines += [f"## {season}", "",
                  "| arm | changed | net pts/GW | real avg/GW | vs real avg | vs control (paired) | hits | transfers | chips |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for arm in sorted(rows, key=lambda a: (a["label"] != "control", a["label"])):
            rb = arm["real_benchmark"]
            net = rb["net_points_per_gw"]
            real = rb["real_avg_per_gw"]
            beats = rb["beats_real_avg_per_gw"]
            if control is None or arm is control:
                paired = "-"
            else:
                d = paired_difference(arm, control)
                paired = "n/a" if d["mean"] is None else (
                    f"{d['mean']:+.2f}" + (f" ± {d['se']:.2f}" if d["se"] is not None else "") + f" (n={d['n']})"
                )
            changed = ", ".join(f"{k}={v}" for k, v in arm["changed"].items()) or "live settings"
            chips = ", ".join(f"GW{gw} {chip}" for gw, chip in arm["chips_played"]) or "none"
            lines.append(
                f"| {arm['label']} | {changed} | {_fmt(net)} | {_fmt(real)} | "
                f"{'n/a' if beats is None else f'{beats:+.2f}'} | {paired} | {rb['total_hits']:.0f} | "
                f"{arm['n_transfers']} | {chips} |"
            )
        if control is None:
            lines.append("")
            lines.append("_No 'control' arm for this season, so no paired comparison._")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    arms_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    arms = [json.loads(p.read_text()) for p in sorted(arms_dir.glob("*.json"))]
    if not arms:
        raise SystemExit(f"no arm files in {arms_dir}")
    summary = summarize(arms)
    (arms_dir / "SUMMARY.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
