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

With --rule r5 --target <price_band=9.0+ | price_band=<5.0 | in_form> (rule R5, docs/plans/
2026-10_premium_and_form_returns.md, agreed with the project owner 2026-10-11): (i) and (ii) as
above against control; (iii) the target group's component sum (|appearance| + |goals| +
|assists| + |clean sheet| + |other|, so errors that cancel still count) at most two thirds of
control's -- a price band (9.0m+, or below 5.0m for the round-3 minutes arm), or the in-form group (fpl_quant.form_groups: declared, or its
one widening when control has fewer than form_groups.MIN_GROUP_SIZE player-steps); (iv) the
breakout group's residual stays at or below BREAKOUT_LIMIT, so the breakout fix isn't undone.

With --rule confirm (round 3's confirmation, docs/reports/2026-10_premium_and_form_returns.md,
declared 2026-10-11 before any of its results): two runs over every season. (i) paired squad
points over every gameweek of every season (per_gameweek squad_points: beats_real needs the real
average manager, which 2024-25 lacks), the same floor as above; refuses a season it can't pair; (ii) EP MAE no worse than
control's in each season on its own, with no tolerance; (iii) the declared breakout group's
2025-26 residual at most BREAKOUT_LIMIT, as R5 (iv).

With --rule r6 (docs/plans/2026-10_beat_the_template.md, agreed with the project owner 2026-10-12):
two runs over every season. (i) paired squad points over every gameweek of every season,
m - SE >= 0; (ii) each season's EP MAE at most control's + 0.01. Refuses runs on different cached
DBs or seasons, an incomplete run, or a season with fewer than 2 paired gameweeks. It also prints
both runs' gaps to the availability-gated and ungated ownership templates, XI overlap with the
gated template, the change in points from outside it, the gameweeks the arm changed and the SE,
and applies the plan's Req 9 guard: promotion is BLOCKED when mean overlap rises by >= 1 player a
gameweek while the paired change in non-template points is <= 0, and mean overlap >= 9/11 in
either run is PAUSE, for the owner. It refuses summaries without the template rows Req 9 reads.
When |m - SE| <= 0.1 the verdict is REPLICATE, before anything else: re-dispatch control and arm
once on the same commit and pass them with --replicate, which judges R6 on the per-gameweek mean
of the two runs' differences and the mean of their MAEs, and reports any difference between the
two controls as measured nondeterminism.

Refuses (exit 2) when any run is incomplete or they weren't made on the same cached DB.

Usage:
    python scripts/compare_arms.py control/walkforward_summary.json arm/walkforward_summary.json [--json]
    python scripts/compare_arms.py control.json arm.json --baseline before_fix.json [--json]
    python scripts/compare_arms.py control.json arm.json --rule r5 --target in_form [--json]
    python scripts/compare_arms.py control.json arm.json --rule confirm [--json]
    python scripts/compare_arms.py control.json arm.json --rule r6 [--replicate control_b.json arm_b.json] [--json]

Exit codes: 0 PASS, 1 FAIL, BLOCKED or PAUSE, 2 refused, 3 REPLICATE (rule r6 only).
"""

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import breakout, form_groups  # noqa: E402

DECIDING_SEASON = "2025-2026"
POINTS_FLOOR = -0.25
MAE_TOLERANCE = 0.002
BREAKOUT_CUT = 2.0 / 3.0
# R5 (iv): two thirds of the before-fix model's breakout residual, +0.272
# (docs/reports/2026-10_breakout_players.md, follow-up 2).
BREAKOUT_LIMIT = 0.182
TARGET_CUT = 2.0 / 3.0
TARGETS = ("price_band=9.0+", "price_band=<5.0", "in_form")
COMPONENTS = ("appearance", "goals", "assists", "cleansheet", "other")


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


def paired_points(control: dict, arm: dict, season: str | None = DECIDING_SEASON, key: str = "beats_real") -> dict:
    """Paired per-gameweek differences of `key` over one season's gameweeks, or every season's
    (None). beats_real and squad_points give the same difference wherever both exist."""
    def by_step(run):
        return {(r["season"], r["gw"]): r.get(key) for r in run["per_gameweek"]
                if (season is None or r["season"] == season) and r.get(key) is not None}
    c, a = by_step(control), by_step(arm)
    diffs = [a[k] - c[k] for k in sorted(set(c) & set(a))]
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


def component_sum(block: dict | None) -> float | None:
    """A split block's component_abs_sum, or the same sum from its components (summaries written
    before that key existed carry the components alone)."""
    if not block:
        return None
    if block.get("component_abs_sum") is not None:
        return block["component_abs_sum"]
    parts = block.get("components") or block
    values = [parts.get(c) for c in COMPONENTS]
    return None if any(v is None for v in values) else sum(abs(v) for v in values)


def _target_block(run: dict, target: str, label: str | None = None) -> dict | None:
    if target.startswith("price_band="):
        return (run.get("price_band") or {}).get(target.split("=", 1)[1])
    return ((run.get("form_groups") or {}).get(DECIDING_SEASON) or {}).get(label)


def _form_label(control: dict) -> tuple[str | None, str]:
    """The in-form group to judge on: declared, or its one widening (the group is defined by the
    data, so its size is the same in every run)."""
    block = (control.get("form_groups") or {}).get(DECIDING_SEASON) or {}
    for label in (form_groups.IN_FORM, form_groups.IN_FORM + "_widened"):
        n = (block.get(label) or {}).get("n_player_steps") or 0
        if n >= form_groups.MIN_GROUP_SIZE:
            return label, f"{label} (n={n})"
    return None, f"in-form group missing or too small in control (< {form_groups.MIN_GROUP_SIZE} even widened)"


def _target_check(control: dict, arm: dict, target: str) -> dict:
    """R5(iii): the target group's component sum at most two thirds of control's."""
    label, group = None, target
    if target == "in_form":
        label, group = _form_label(control)
        if label is None:
            return {"pass": False, "group": group, "reason": group}
    s_c, s_a = component_sum(_target_block(control, target, label)), component_sum(_target_block(arm, target, label))
    if s_c is None or s_c <= 0:
        return {"pass": False, "group": group, "reason": f"no component split for {group} in control"}
    return {"pass": s_a is not None and s_a <= TARGET_CUT * s_c, "group": group,
            "control": s_c, "arm": s_a, "limit": TARGET_CUT * s_c}


def _breakout_limit_check(control: dict, arm: dict) -> dict:
    """R5(iv): the arm's breakout residual at most BREAKOUT_LIMIT, on the declared breakout group
    (the limit was set from that group's before-fix residual, so no widening here)."""
    label = breakout.BREAKOUT
    n = ((((control.get("breakout") or {}).get(DECIDING_SEASON) or {}).get(label)) or {}).get("n_player_steps") or 0
    if n < breakout.MIN_GROUP_SIZE:
        return {"pass": False, "reason": f"declared breakout group too small in control (n={n} < {breakout.MIN_GROUP_SIZE})"}
    r_a = ((arm.get("breakout") or {}).get(DECIDING_SEASON) or {}).get(label, {}).get("mean_resid")
    return {"pass": r_a is not None and r_a <= BREAKOUT_LIMIT, "group": f"{label} (n={n})", "arm": r_a, "limit": BREAKOUT_LIMIT}


def _confirm(control: dict, arm: dict, baseline: dict | None) -> dict:
    """--rule confirm: both seasons, pooled points, each season's MAE, the breakout limit."""
    if baseline is not None:
        raise Refused("rule confirm takes no --baseline")
    _check_comparable(control, arm)
    seasons = sorted(control.get("headline_by_season") or {})
    if len(seasons) < 2 or seasons != sorted(arm.get("headline_by_season") or {}):
        raise Refused("rule confirm needs two runs over the same seasons, more than one (run without --seasons)")
    # every season must pair: beats_real is missing where the data has no real average manager
    # (2024-25), so the rule pairs the squad's own points, and refuses a summary without them
    for season in seasons:
        n = paired_points(control, arm, season=season, key="squad_points")["n"]
        if n < 2:
            raise Refused(f"{season}: {n} gameweeks with squad_points in both runs; rule confirm pairs every "
                          "season (summaries written before per-gameweek squad_points can't be judged by it)")
    checks = {}
    pts = paired_points(control, arm, season=None, key="squad_points")
    checks["points"] = {"pass": pts["mean"] is not None and pts["mean"] >= POINTS_FLOOR and pts["mean"] + pts["se"] >= 0,
                        **pts}
    by_season = {}
    for season in seasons:
        mae_c = control["headline_by_season"][season]["ep_total_calibration_mae"]
        mae_a = arm["headline_by_season"][season]["ep_total_calibration_mae"]
        by_season[season] = {"control": mae_c, "arm": mae_a, "delta": None if None in (mae_c, mae_a) else mae_a - mae_c}
    checks["mae_by_season"] = {"pass": all(v["delta"] is not None and v["delta"] <= 1e-12 for v in by_season.values()),
                               "seasons": by_season}
    checks["breakout_limit"] = _breakout_limit_check(control, arm)
    return {"verdict": "PASS" if all(c["pass"] for c in checks.values()) else "FAIL", "checks": checks}


# Rule R6 (docs/plans/2026-10_beat_the_template.md)
R6_MAE_TOLERANCE = 0.01
REPLICATE_BAND = 0.1
# Req 9, the template-collapse guard
COLLAPSE_OVERLAP_RISE = 1.0
PAUSE_OVERLAP = 9.0


def _squad_diffs(control: dict, arm: dict) -> dict:
    """{(season, gw): arm - control squad_points} over the gameweeks both runs scored."""
    def by_step(run):
        return {(r["season"], r["gw"]): r["squad_points"] for r in run["per_gameweek"] if r.get("squad_points") is not None}
    c, a = by_step(control), by_step(arm)
    return {k: a[k] - c[k] for k in sorted(set(c) & set(a))}


def _stats(diffs: list[float]) -> dict:
    n = len(diffs)
    if n < 2:
        return {"n": n, "mean": None, "se": None}
    m = sum(diffs) / n
    sd = math.sqrt(sum((d - m) ** 2 for d in diffs) / (n - 1))
    return {"n": n, "mean": m, "se": sd / math.sqrt(n)}


def _template_view(run: dict) -> dict:
    """The run's own template numbers (walkforward_summary's template block), or {} for a
    summary written before it existed."""
    pooled = (run.get("template") or {}).get("pooled") or {}
    if not pooled.get("n"):
        return {}
    return {"gap_vs_gated": pooled.get("gap_vs_gated"), "gap_vs_ungated": pooled.get("gap_vs_ungated"),
            "xi_overlap_mean": pooled.get("xi_overlap_mean"), "non_template_points_per_gw": pooled.get("non_template_points_per_gw")}


def _paired_key(control: dict, arm: dict, key: str) -> dict:
    """Paired arm - control of a per_gameweek key, every season."""
    def by_step(run):
        return {(r["season"], r["gw"]): r[key] for r in run["per_gameweek"] if r.get(key) is not None}
    c, a = by_step(control), by_step(arm)
    out = _stats([a[k] - c[k] for k in sorted(set(c) & set(a))])
    if out["mean"] is None:
        return out
    common = sorted(set(c) & set(a))
    return {**out, "control": sum(c[k] for k in common) / len(common), "arm": sum(a[k] for k in common) / len(common)}


def _req9(pairs: list[tuple[dict, dict]]) -> dict:
    """Req 9 on the first pair (and the replicate's, averaged, when given): mean XI overlap with
    the gated template, and the paired change in non-template points."""
    overlaps = [_paired_key(c, a, "template_xi_overlap") for c, a in pairs]
    non_template = [_paired_key(c, a, "non_template_points") for c, a in pairs]
    if any(o["mean"] is None for o in overlaps) or any(n["mean"] is None for n in non_template):
        # a promotion needs R6 and Req 9 both, so an unguarded comparison is no comparison
        raise Refused("rule r6 needs per-gameweek template_xi_overlap and non_template_points in every run "
                      "(Req 9): a summary from before the template yardstick can't be judged by it")
    rise = sum(o["mean"] for o in overlaps) / len(overlaps)
    control_overlap = sum(o["control"] for o in overlaps) / len(overlaps)
    arm_overlap = sum(o["arm"] for o in overlaps) / len(overlaps)
    change = sum(n["mean"] for n in non_template) / len(non_template)
    return {
        "control_overlap": control_overlap, "arm_overlap": arm_overlap,
        "overlap_rise": rise, "non_template_change": change, "non_template_change_se": non_template[0]["se"],
        "blocks_promotion": rise >= COLLAPSE_OVERLAP_RISE and change <= 0,
        "pause_for_owner": max(control_overlap, arm_overlap) >= PAUSE_OVERLAP,
    }


def _control_drift(control: dict, control_b: dict) -> dict:
    """The replicate's two controls, same commit and DB: any squad-point difference between them
    is measured nondeterminism (the plan's replicate rule records it)."""
    diffs = list(_squad_diffs(control, control_b).values())
    changed = [d for d in diffs if abs(d) > 1e-9]
    return {"n": len(diffs), "changed": len(changed), "mean": sum(diffs) / len(diffs) if diffs else None,
            "max_abs": max((abs(d) for d in changed), default=0.0)}


def _r6(control: dict, arm: dict, baseline: dict | None, replicate: tuple[dict, dict] | None) -> dict:
    """--rule r6: paired squad points over every season (m - SE >= 0), each season's MAE within
    control + 0.01, Req 9's guard, and the replicate rule."""
    if baseline is not None:
        raise Refused("rule r6 takes no --baseline")
    pairs = [(control, arm)] + ([replicate] if replicate is not None else [])
    names = [("control", control), ("arm", arm)]
    if replicate is not None:
        names += [("control_b", replicate[0]), ("arm_b", replicate[1])]
    for name, run in names:
        if not (run.get("progress") or {}).get("complete"):
            raise Refused(f"{name} run is incomplete (progress.complete is not true)")
        if not run.get("db_cache_key"):
            raise Refused(f"{name} run has no db_cache_key")
    keys = {run["db_cache_key"] for _, run in names}
    if len(keys) > 1:
        found = " vs ".join(f"{name} {run['db_cache_key']}" for name, run in names)
        raise Refused(f"different cached DBs: {found}")
    lags = {name: (run.get("template") or {}).get("lag") or 1 for name, run in names}  # runs before the key: lag 1
    if len(set(map(str, lags.values()))) > 1:
        raise Refused("runs built their templates from different rows (template lag "
                      + ", ".join(f"{name} {lag}" for name, lag in lags.items()) + ")")
    seasons = sorted(control.get("headline_by_season") or {})
    if len(seasons) < 2:
        raise Refused("rule r6 judges every season: run without --seasons (both seasons)")
    for name, run in names[1:]:
        if sorted(run.get("headline_by_season") or {}) != seasons:
            raise Refused(f"{name} run scored different seasons from control")
    per_pair = [_squad_diffs(c, a) for c, a in pairs]
    for i, diffs in enumerate(per_pair):
        for season in seasons:
            n = sum(1 for (s_, _gw) in diffs if s_ == season)
            if n < 2:
                raise Refused(f"{season}: {n} gameweeks with squad_points in both runs"
                              + (" of the replicate" if i else "") + "; rule r6 pairs every season")
    common = sorted(set.intersection(*(set(d) for d in per_pair)))
    diffs = [sum(d[k] for d in per_pair) / len(per_pair) for k in common]
    pts = _stats(diffs)
    pts.update(wins=sum(d > 0 for d in diffs), losses=sum(d < 0 for d in diffs),
               changed=sum(abs(d) > 1e-9 for d in diffs), m_minus_se=pts["mean"] - pts["se"])
    checks = {"points": {"pass": pts["m_minus_se"] >= -1e-12, **pts}}

    by_season = {}
    for season in seasons:
        mae_c = [c["headline_by_season"][season]["ep_total_calibration_mae"] for c, _ in pairs]
        mae_a = [a["headline_by_season"][season]["ep_total_calibration_mae"] for _, a in pairs]
        if None in mae_c or None in mae_a:
            by_season[season] = {"control": None, "arm": None, "delta": None}
            continue
        c_mean, a_mean = sum(mae_c) / len(mae_c), sum(mae_a) / len(mae_a)
        by_season[season] = {"control": c_mean, "arm": a_mean, "delta": a_mean - c_mean}
    checks["mae_by_season"] = {
        "pass": all(v["delta"] is not None and v["delta"] <= R6_MAE_TOLERANCE + 1e-12 for v in by_season.values()),
        "seasons": by_season, "tolerance": R6_MAE_TOLERANCE,
    }

    req9 = _req9(pairs)
    # near the threshold, control and arm are re-run before ANY verdict: the replicate's mean
    # MAE can land on the other side of the tolerance too
    replicate_required = replicate is None and abs(pts["m_minus_se"]) <= REPLICATE_BAND
    if replicate_required:
        verdict = "REPLICATE"
    elif not (checks["mae_by_season"]["pass"] and checks["points"]["pass"]):
        verdict = "FAIL"
    elif req9["blocks_promotion"]:
        verdict = "BLOCKED"
    elif req9["pause_for_owner"]:
        verdict = "PAUSE"
    else:
        verdict = "PASS"
    return {
        "verdict": verdict, "rule": "r6", "checks": checks, "req9": req9,
        "replicate_required": replicate_required, "replicated": replicate is not None,
        "control_drift": _control_drift(control, replicate[0]) if replicate is not None else None,
        "template": {"control": _template_view(control), "arm": _template_view(arm)},
    }


def compare(control: dict, arm: dict, baseline: dict | None = None, *, rule: str = "r4", target: str | None = None,
            replicate: tuple[dict, dict] | None = None) -> dict:
    if rule == "r5" and (baseline is not None or target not in TARGETS):
        raise Refused(f"rule r5 takes --target {' or '.join(TARGETS)} and no --baseline")
    if rule != "r5" and target is not None:
        raise Refused("--target is rule r5's: pass --rule r5")
    if replicate is not None and rule != "r6":
        raise Refused("--replicate is rule r6's: pass --rule r6")
    if rule == "r6":
        return _r6(control, arm, baseline, replicate)
    if rule == "confirm":
        return _confirm(control, arm, baseline)
    _check_comparable(control, arm, baseline)
    if target is not None and target.startswith("price_band="):
        # price_band pools every season the run scored: it is 2025-26's only on a 2025-26 run
        for name, run in (("control", control), ("arm", arm)):
            other = sorted(set(run.get("headline_by_season") or {}) - {DECIDING_SEASON})
            if other:
                raise Refused(f"{name} run also scored {', '.join(other)}: price_band pools seasons; run with --seasons 2025-2026")
    checks = {}

    pts = paired_points(control, arm)
    ok = pts["mean"] is not None and pts["mean"] >= POINTS_FLOOR and pts["mean"] + pts["se"] >= 0
    checks["points"] = {"pass": ok, **pts}

    mae_c, mae_a = _headline(control, "ep_total_calibration_mae"), _headline(arm, "ep_total_calibration_mae")
    checks["mae"] = {"pass": mae_a <= mae_c + MAE_TOLERANCE + 1e-12, "control": mae_c, "arm": mae_a,
                     "delta": mae_a - mae_c}

    if rule == "r5":
        checks["target"] = _target_check(control, arm, target)
        checks["breakout_limit"] = _breakout_limit_check(control, arm)
        return {"verdict": "PASS" if all(c["pass"] for c in checks.values()) else "FAIL", "checks": checks}

    # R4 judges the cut against control; the follow-up rule against the model before the fix
    checks["breakout"] = _breakout_check(control if baseline is None else baseline, arm)
    checks["breakout"]["reference"] = "control" if baseline is None else "before fix"

    if baseline is not None:
        log_c, log_a = _headline(control, "log_score_minutes_mean"), _headline(arm, "log_score_minutes_mean")
        checks["log_score"] = {"pass": log_a > log_c, "control": log_c, "arm": log_a, "delta": log_a - log_c}
    return {"verdict": "PASS" if all(c["pass"] for c in checks.values()) else "FAIL", "checks": checks}


def _gap(g: dict | None) -> str:
    if not g or g.get("mean") is None:
        return "n/a"
    return f"{g['mean']:+.2f}" + ("" if g.get("se") is None else f" ± {g['se']:.2f}")


def _fmt_r6(result: dict) -> str:
    c = result["checks"]
    p = c["points"]
    lines = [result["verdict"] + (" (replicated: means of two runs)" if result["replicated"] else "")]
    lines.append(
        f"  (i)   squad points {'pass' if p['pass'] else 'FAIL'}: {p['mean']:+.3f} ± {p['se']:.3f} a gameweek over "
        f"{p['n']} ({p['wins']} better, {p['losses']} worse, {p['changed']} changed); m - SE {p['m_minus_se']:+.3f}"
    )
    m = c["mae_by_season"]
    parts = [f"{season} {v['control']:.4f} -> {v['arm']:.4f} ({v['delta']:+.4f})" if v["delta"] is not None
             else f"{season} missing" for season, v in m["seasons"].items()]
    lines.append(f"  (ii)  EP MAE by season {'pass' if m['pass'] else 'FAIL'} (tolerance +{m['tolerance']}): " + "; ".join(parts))
    for name in ("control", "arm"):
        t = result["template"][name]
        if t:
            lines.append(f"  {name:<7} vs gated template {_gap(t['gap_vs_gated'])}, ungated {_gap(t['gap_vs_ungated'])} a gameweek; "
                         f"XI overlap {t['xi_overlap_mean']}, non-template points {t['non_template_points_per_gw']}")
        else:
            lines.append(f"  {name:<7} template: not in this summary")
    r = result["req9"]
    state = "BLOCKS PROMOTION" if r["blocks_promotion"] else "ok"
    se = "" if r["non_template_change_se"] is None else f" ± {r['non_template_change_se']:.2f}"
    lines.append(f"  Req 9 {state}: overlap {r['control_overlap']:.2f} -> {r['arm_overlap']:.2f} ({r['overlap_rise']:+.2f}), "
                 f"non-template points {r['non_template_change']:+.2f}{se} a gameweek")
    if r["pause_for_owner"]:
        lines.append(f"  PAUSE FOR OWNER REVIEW: mean overlap {max(r['control_overlap'], r['arm_overlap']):.2f} "
                     f">= {PAUSE_OVERLAP:.0f}/11")
    drift = result.get("control_drift")
    if drift is not None:
        lines.append(f"  controls: {drift['changed']} of {drift['n']} gameweeks differ between the two control runs"
                     + (f" (measured nondeterminism, up to {drift['max_abs']:.1f} pts)" if drift["changed"] else ""))
    if result["replicate_required"]:
        lines.append(f"  replicate required: |m - SE| = {abs(p['m_minus_se']):.3f} <= {REPLICATE_BAND}; re-dispatch control "
                     "and arm on the same commit, then pass them with --replicate")
    return "\n".join(lines)


def _fmt(result: dict) -> str:
    if result.get("rule") == "r6":
        return _fmt_r6(result)
    c = result["checks"]
    lines = [result["verdict"]]
    p = c["points"]
    lines.append(
        f"  (i)   points  {'pass' if p['pass'] else 'FAIL'}: "
        + ("no paired gameweeks" if p["mean"] is None else
           f"{p['mean']:+.3f} ± {p['se']:.3f} a gameweek over {p['n']} ({p['wins']} better, {p['losses']} worse)")
    )
    if "mae_by_season" in c:
        m = c["mae_by_season"]
        parts = [f"{season} {v['control']:.4f} -> {v['arm']:.4f} ({v['delta']:+.4f})" if v["delta"] is not None
                 else f"{season} missing" for season, v in m["seasons"].items()]
        lines.append(f"  (ii)  EP MAE by season {'pass' if m['pass'] else 'FAIL'}: " + "; ".join(parts))
        b = c["breakout_limit"]
        if "limit" in b:
            arm_r = "missing" if b["arm"] is None else f"{b['arm']:+.4f}"
            lines.append(f"  (iii) breakout {'pass' if b['pass'] else 'FAIL'}: {b['group']}: arm {arm_r} (limit {b['limit']:+.4f})")
        else:
            lines.append(f"  (iii) breakout FAIL: {b['reason']}")
        return "\n".join(lines)
    m = c["mae"]
    lines.append(f"  (ii)  EP MAE  {'pass' if m['pass'] else 'FAIL'}: {m['control']:.4f} -> {m['arm']:.4f} ({m['delta']:+.4f})")
    if "target" in c:
        t = c["target"]
        if "limit" in t:
            arm_s = "missing" if t["arm"] is None else f"{t['arm']:.4f}"
            lines.append(f"  (iii) target  {'pass' if t['pass'] else 'FAIL'}: {t['group']} component sum "
                         f"{t['control']:.4f} -> {arm_s} (limit {t['limit']:.4f})")
        else:
            lines.append(f"  (iii) target  FAIL: {t['reason']}")
        b = c["breakout_limit"]
        if "limit" in b:
            arm_r = "missing" if b["arm"] is None else f"{b['arm']:+.4f}"
            lines.append(f"  (iv)  breakout {'pass' if b['pass'] else 'FAIL'}: {b['group']}: arm {arm_r} (limit {b['limit']:+.4f})")
        else:
            lines.append(f"  (iv)  breakout FAIL: {b['reason']}")
        return "\n".join(lines)
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
    replicate_paths = None
    if "--replicate" in args:
        i = args.index("--replicate")
        if i + 2 >= len(args):
            print(__doc__)
            return 2
        replicate_paths = args[i + 1:i + 3]
        del args[i:i + 3]
    opts = {}
    for flag in ("--rule", "--target"):
        if flag in args:
            i = args.index(flag)
            if i + 1 >= len(args):
                print(__doc__)
                return 2
            opts[flag[2:]] = args[i + 1]
            del args[i:i + 2]
    paths = [a for a in args if a != "--json"]
    if len(paths) != 2 or opts.get("rule", "r4") not in ("r4", "r5", "confirm", "r6"):
        print(__doc__)
        return 2
    try:
        control, arm = (json.loads(Path(p).read_text()) for p in paths)
        baseline = json.loads(Path(baseline_path).read_text()) if baseline_path else None
        replicate = tuple(json.loads(Path(p).read_text()) for p in replicate_paths) if replicate_paths else None
        result = compare(control, arm, baseline, rule=opts.get("rule", "r4"), target=opts.get("target"), replicate=replicate)
    except Refused as e:
        print(f"REFUSED: {e}")
        return 2
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as e:  # unreadable or malformed: not a FAIL
        print(f"REFUSED: unreadable summary ({type(e).__name__}: {e})")
        return 2
    print(json.dumps(result, indent=2) if as_json else _fmt(result))
    return {"PASS": 0, "REPLICATE": 3}.get(result["verdict"], 1)


if __name__ == "__main__":
    sys.exit(main())
