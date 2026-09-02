"""Diff two dump_backtest_metrics.py JSON files (an A/B pair) into a Markdown table.

Used by ab_evidence_strength.yml's compare job. Prints to stdout and, if $GITHUB_STEP_SUMMARY
is set, appends there too. Exits 0 always -- this reports a comparison, it does not pass/fail
a build (a human reads the table and decides).

Usage:
    python scripts/compare_backtest_metrics.py baseline.json variant.json
"""

import json
import os
import sys
from pathlib import Path

# Metrics where LOWER is better (calibration error / probabilistic loss). Everything else
# (points deltas) is "higher is better". "abs(mean_resid) lower" is handled specially.
_LOWER_IS_BETTER = {
    "brier_minutes_mean", "brier_clean_sheet_mean", "log_score_minutes_mean",
    "poisson_calibration_degenerate_count",
}
_KEY_ORDER = [
    "beats_crowd_points_delta", "model_squad_realized_points", "avg_manager_benchmark_points",
    "brier_minutes_mean", "log_score_minutes_mean", "poisson_calibration_mean_resid",
    "brier_clean_sheet_mean", "poisson_calibration_degenerate_count",
]


def _direction(name: str, b: float, v: float) -> str:
    if abs(v - b) < 1e-6:
        return "="
    if name == "poisson_calibration_mean_resid":  # signed bias -- closer to 0 is better
        if abs(abs(v) - abs(b)) < 1e-6:
            return "="
        return "better" if abs(v) < abs(b) else "worse"
    better = (v < b) if name in _LOWER_IS_BETTER else (v > b)
    return "better" if better else "worse"


def main() -> None:
    base = json.loads(Path(sys.argv[1]).read_text())
    var = json.loads(Path(sys.argv[2]).read_text())

    lines = [
        f"## Evidence-strength A/B  ·  {base['label']} vs {var['label']}",
        "",
        f"- baseline: predicted_xi pull `{base['config'].get('predicted_xi_pull_strength')}`, "
        f"official tier weight `{base['config'].get('official_tier_weight')}`, "
        f"{base['n_gameweek_steps']} steps ({', '.join(base['seasons_covered'])})",
        f"- variant:  predicted_xi pull `{var['config'].get('predicted_xi_pull_strength')}`, "
        f"official tier weight `{var['config'].get('official_tier_weight')}`, "
        f"{var['n_gameweek_steps']} steps",
        "",
        "| metric | baseline | variant | delta | |",
        "|---|--:|--:|--:|---|",
    ]
    names = _KEY_ORDER + sorted(set(base["metrics"]) | set(var["metrics"]) - set(_KEY_ORDER))
    seen = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        b = base["metrics"].get(name, {}).get("mean")
        v = var["metrics"].get(name, {}).get("mean")
        if b is None and v is None:
            continue
        if b is None or v is None:
            lines.append(f"| {name} | {b if b is not None else '—'} | {v if v is not None else '—'} | — | |")
            continue
        d = v - b
        lines.append(f"| {name} | {b:+.4f} | {v:+.4f} | {d:+.4f} | {_direction(name, b, v)} |")

    lines += [
        "",
        "**How to read it:** `beats_crowd_points_delta` (model squad realised points minus the "
        "average manager, per gameweek) is the headline -- higher is better. `brier_*` / "
        "`log_score_*` are probabilistic calibration -- lower is better. "
        "`poisson_calibration_mean_resid` is a signed bias -- closer to 0 is better. A variant "
        "that lifts the crowd delta without hurting calibration is a real improvement; one that "
        "only helps in-sample calibration while the crowd delta drops is overfitting to noisy "
        "evidence claims.",
    ]
    out = "\n".join(lines)
    print(out)
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write(out + "\n")


if __name__ == "__main__":
    main()
