"""PASS or FAIL for one walk-forward arm against its control, under the breakout promotion rule
(docs/plans/2026-10_breakout_players.md, R4 -- agreed with the project owner 2026-10-08):

  (i)   2025-26 squad points level or better: per_gameweek[].beats_real paired on the gameweeks
        non-null in both runs (FPL's real average is the same in both, so the difference is the
        squad-point difference); mean m, SE = sample sd / sqrt(n); passes when m >= -0.25 and
        m + SE >= 0.
  (ii)  2025-26 EP MAE no more than 0.002 above control's.
  (iii) the 2025-26 breakout group's under-prediction cut by at least a third: control's mean
        residual > 0 and the arm's at most two thirds of it. The group is the declared rule, or
        its one declared widening when control has fewer than breakout.MIN_GROUP_SIZE
        player-steps; with neither big enough, the comparison stops.

With --baseline (the follow-up rule, docs/reports/2026-10_breakout_players.md, declared
2026-10-09 before any of its results): control is the live model and baseline the model before
the breakout fix, on the same cached DB. (i) and (ii) as above against control; (iii) the arm
keeps the fix: its breakout residual at most two thirds of the baseline's; (iv) its 2025-26
minutes log score is better than control's.

Refuses (exit 2) when any run is incomplete or they weren't made on the same cached DB.

Usage:
    python scripts/compare_arms.py control/walkforward_summary.json arm/walkforward_summary.json [--json]
    python scripts/compare_arms.py control.json arm.json --baseline before_fix.json [--json]
"""

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import breakout  # noqa: E402

DECIDING_SEASON = "2025-2026"
POINTS_FLOOR = -0.25
MAE_TOLERANCE = 0.002
BREAKOUT_CUT = 2.0 / 3.0


class Refused(ValueError):
    """The two runs can't be compared at all."""


def _check_comparable(control: dict, arm: dict, baseline: dict | None = None) -> None:
    runs = [("control", control), ("arm", arm)] + ([("baseline", baseline)] if baseline is not None else [])
    for name, run in runs:
        if not (run.get("progress") or {}).get("complete"):
            raise Refused(f"{name} run is incomplete (progress.complete is not true)")
        if not run.get("db_cache_key"):
            raise Refused(f"{name} run has no db_cache_key")
    keys = {run["db_cache_key"] for _, run in runs}
    if len(keys) > 1:
        raise Refused(f"different cached DBs: {' vs '.join(run['db_cache_key'] for _, run in runs)}")


def paired_points(control: dict, arm: dict, season: str = DECIDING_SEASON) -> dict:
    c = {r["gw"]: r["beats_real"] for r in control["per_gameweek"] if r["season"] == season and r["beats_real"] is not None}
    a = {r["gw"]: r["beats_real"] for r in arm["per_gameweek"] if r["season"] == season and r["beats_real"] is not None}
    diffs = [a[gw] - c[gw] for gw in sorted(set(c) & set(a))]
    n = len(diffs)
    if n < 2:
        return {"n": n, "mean": None, "se": None, "wins": None, "losses": None}
    m = sum(diffs) / n
    sd = math.sqrt(sum((d - m) ** 2 for d in diffs) / (n - 1))
    return {"n": n, "mean": m, "se": sd / math.sqrt(n),
            "wins": sum(d > 0 for d in diffs), "losses": sum(d < 0 for d in diffs)}


def _group(control: dict) -> tuple[str | None, str]:
    """The breakout group to judge on (its size is the same in every run: the group is defined by
    the data, not the model)."""
    block = (control.get("breakout") or {}).get(DECIDING_SEASON) or {}
    for label in (breakout.BREAKOUT, breakout.BREAKOUT + "_widened"):
        n = (block.get(label) or {}).get("n_player_steps") or 0
        if n >= breakout.MIN_GROUP_SIZE:
            return label, f"{label} (n={n})"
    return None, f"breakout group too small in control (< {breakout.MIN_GROUP_SIZE} even widened): stop and report"


def _headline(run: dict, metric: str):
    return run["headline_by_season"][DECIDING_SEASON][metric]


def _breakout_check(reference: dict, arm: dict) -> dict:
    """R4(iii): the arm's breakout residual at most two thirds of the reference run's."""
    label, why = _group(reference)
    if label is None:
        return {"pass": False, "reason": why}
    r_c = reference["breakout"][DECIDING_SEASON][label]["mean_resid"]
    r_a = ((arm.get("breakout") or {}).get(DECIDING_SEASON) or {}).get(label, {}).get("mean_resid")
    if r_c is None or r_c <= 0:
        return {"pass": False, "group": why, "control": r_c,
                "reason": "control does not under-predict the group: stop and report"}
    return {"pass": r_a is not None and r_a <= BREAKOUT_CUT * r_c, "group": why,
            "control": r_c, "arm": r_a, "limit": BREAKOUT_CUT * r_c}


def compare(control: dict, arm: dict, baseline: dict | None = None) -> dict:
    _check_comparable(control, arm, baseline)
    checks = {}

    pts = paired_points(control, arm)
    ok = pts["mean"] is not None and pts["mean"] >= POINTS_FLOOR and pts["mean"] + pts["se"] >= 0
    checks["points"] = {"pass": ok, **pts}

    mae_c, mae_a = _headline(control, "ep_total_calibration_mae"), _headline(arm, "ep_total_calibration_mae")
    checks["mae"] = {"pass": mae_a <= mae_c + MAE_TOLERANCE + 1e-12, "control": mae_c, "arm": mae_a,
                     "delta": mae_a - mae_c}

    # R4 judges the cut against control; the follow-up rule against the model before the fix
    checks["breakout"] = _breakout_check(control if baseline is None else baseline, arm)
    checks["breakout"]["reference"] = "control" if baseline is None else "before fix"

    if baseline is not None:
        log_c, log_a = _headline(control, "log_score_minutes_mean"), _headline(arm, "log_score_minutes_mean")
        checks["log_score"] = {"pass": log_a > log_c, "control": log_c, "arm": log_a, "delta": log_a - log_c}
    return {"verdict": "PASS" if all(c["pass"] for c in checks.values()) else "FAIL", "checks": checks}


def _fmt(result: dict) -> str:
    c = result["checks"]
    lines = [result["verdict"]]
    p = c["points"]
    lines.append(
        f"  (i)   points  {'pass' if p['pass'] else 'FAIL'}: "
        + ("no paired gameweeks" if p["mean"] is None else
           f"{p['mean']:+.3f} ± {p['se']:.3f} a gameweek over {p['n']} ({p['wins']} better, {p['losses']} worse)")
    )
    m = c["mae"]
    lines.append(f"  (ii)  EP MAE  {'pass' if m['pass'] else 'FAIL'}: {m['control']:.4f} -> {m['arm']:.4f} ({m['delta']:+.4f})")
    b = c["breakout"]
    if "limit" in b:
        arm_r = "missing" if b["arm"] is None else f"{b['arm']:+.4f}"
        lines.append(f"  (iii) breakout {'pass' if b['pass'] else 'FAIL'}: {b['group']}: {b['reference']} {b['control']:+.4f} -> arm {arm_r} (limit {b['limit']:+.4f})")
    else:
        lines.append(f"  (iii) breakout FAIL: {b['reason']}")
    if "log_score" in c:
        g = c["log_score"]
        lines.append(f"  (iv)  minutes log score {'pass' if g['pass'] else 'FAIL'}: "
                     f"{g['control']:.4f} -> {g['arm']:.4f} ({g['delta']:+.4f})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in args
    baseline_path = None
    if "--baseline" in args:
        i = args.index("--baseline")
        if i + 1 >= len(args):
            print(__doc__)
            return 2
        baseline_path = args[i + 1]
        del args[i:i + 2]
    paths = [a for a in args if a != "--json"]
    if len(paths) != 2:
        print(__doc__)
        return 2
    try:
        control, arm = (json.loads(Path(p).read_text()) for p in paths)
        baseline = json.loads(Path(baseline_path).read_text()) if baseline_path else None
        result = compare(control, arm, baseline)
    except Refused as e:
        print(f"REFUSED: {e}")
        return 2
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as e:  # unreadable or malformed: not a FAIL
        print(f"REFUSED: unreadable summary ({type(e).__name__}: {e})")
        return 2
    print(json.dumps(result, indent=2) if as_json else _fmt(result))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
