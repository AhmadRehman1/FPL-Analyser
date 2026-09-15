"""Collects the two scripts/run_minutes_model_current_season_sensitivity_arm.py outputs
("off" vs "on") into one comparison -- the walk-forward evidence required before
minutes_model.run()'s current_season_role_params_version (see its own docstring for the real
incident it fixes) goes from opt-in to any default.

Reads data/minutes_model_current_season_study/arms/*.json and writes:

  data/minutes_model_current_season_study/minutes_model_current_season_study_latest.json
  data/minutes_model_current_season_study/SUMMARY.md

States plainly whether "on" regresses the headline beats_crowd_points_delta metric relative
to "off" -- the specific bar this study exists to check -- and reports the minutes-specific
calibration metrics (brier_minutes_mean, log_score_minutes_mean) alongside it, since those are
the more DIRECT read on whether the current-season blend is actually a better minutes model,
not just a downstream points effect. No recommendation beyond that check is baked in.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARMS_DIR = REPO_ROOT / "data" / "minutes_model_current_season_study" / "arms"
OUT_DIR = REPO_ROOT / "data" / "minutes_model_current_season_study"


def _load_arms() -> dict[str, dict]:
    arms = {}
    for path in sorted(ARMS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            arms[data["arm"]] = data
        except (OSError, ValueError, KeyError) as exc:
            print(f"::warning::could not read {path.name}: {exc}")
    return arms


def _fmt(v, spec="{:+.4f}"):
    return spec.format(v) if isinstance(v, (int, float)) else "-"


def main() -> None:
    arms = _load_arms()
    if not arms:
        raise SystemExit(f"no arm files in {ARMS_DIR} -- nothing to aggregate")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "minutes_model_current_season_study_latest.json").write_text(json.dumps({"arms": arms}, indent=2))

    md = ["# Minutes-model current-season-role sensitivity study", ""]
    md += [
        "Full M7 walk-forward (`backtest.run()`), both historical seasons. \"off\" = current "
        "production behavior (current_season_role_params_version never activated); \"on\" = "
        "adds the fast-reacting current-season-own-rate blend at its v1 default "
        "(current_season_matches_threshold=4). Both arms already include minutes_model.run()'s "
        "own unconditional lookback_seasons fix (provably backtest-neutral, not what this "
        "study is testing).", "",
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

    md += ["", "## Minutes-model calibration (direct read, not just downstream points)", ""]
    md += ["| metric | off | on | delta |", "|---|---|---|---|"]
    if off and on:
        for metric in sorted(set(off.get("minutes_calibration", {})) | set(on.get("minutes_calibration", {}))):
            off_v = off.get("minutes_calibration", {}).get(metric, {}).get("mean")
            on_v = on.get("minutes_calibration", {}).get(metric, {}).get("mean")
            delta = (on_v - off_v) if (off_v is not None and on_v is not None) else None
            md.append(f"| {metric} | {_fmt(off_v)} | {_fmt(on_v)} | {_fmt(delta)} |")
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
                "deferred task's own bar, current_season_role_params_version should stay off by "
                "default until the regression is understood -- current_season_matches_threshold=4 "
                "is the ORIGINAL invented v1 default (see minutes_model.seed_current_season_role_"
                "params()'s own comment), not a tuned value; a different threshold may be needed "
                "before this helps in aggregate, even if it correctly fixes the specific live "
                "incident that motivated it."
            )
        else:
            md.append(
                "- **\"on\" does not regress beats_crowd_points_delta on any tier.** That clears "
                "the specific bar this study exists to check, but is not by itself a recommendation "
                "to default it on -- the minutes-calibration metrics above are the more direct "
                "signal for whether this genuinely improves minutes projections in aggregate, "
                "worth a human look before deciding."
            )
    else:
        md.append("_One or both arms missing -- no comparison possible._")
    md += [
        "", "_This is evidence, not a decision. current_season_role_params_version stays "
        "unwired-by-default at both backtest.py call sites until the project owner reviews "
        "this and explicitly opts a caller in._",
    ]
    (OUT_DIR / "SUMMARY.md").write_text("\n".join(md))

    print(f"[aggregate] {len(arms)} arm(s) -> {OUT_DIR/'SUMMARY.md'}, {OUT_DIR/'minutes_model_current_season_study_latest.json'}")
    print("\n".join(md))


if __name__ == "__main__":
    main()
