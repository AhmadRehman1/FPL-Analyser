"""One arm of the squad_optimizer solve-time wiring sensitivity study (docs/reports/2026-09_
model_failure_diagnosis.md's Workstream C recalibration-wiring gap -- see backtest.
run_gameweek_step()'s/run_season_simulation()'s own docstrings on
ownership_params_version/risk_posture_params_version/field_covariance_params_version/
bench_quality_params_version/concentration_risk_params_version for what this actually wires up
and why it's opt-in).

Runs ONE full M7 walk-forward pass (backtest.run(), the same ~1-2h mechanism scripts/
run_walkforward.py already uses to provision the ML experiment) over both historical seasons,
at one of two fixed arms:

  "off" -- the exact current production behavior: squad_optimizer.run()'s own Priority 1/2
           terms (EO-weighted posture, field-covariance, bench-quality floor, concentration
           risk) are never activated at this call site (a repo-wide grep confirmed zero
           production caller ever did). compute_segments=True, ownership_params_version=1 are
           still passed (these feed score_gameweek()'s POST-HOC reporting, a separate,
           already-wired concern -- see run()'s own docstring on the naming collision this
           avoids).
  "on"  -- same, plus solve_ownership_params_version=1, solve_risk_posture_params_version=1,
           solve_field_covariance_params_version=1, solve_bench_quality_params_version=1,
           solve_concentration_risk_params_version=1 -- the un-recalibrated v1 defaults for
           all four families (NOT PR #168's still-unmerged, more aggressive risk_posture v2
           eo_weight_kappa=0.12 -- that is a separate, pending decision on a different call
           site entirely; this study answers a narrower question first).

Both arms hold every OTHER param version fixed (the same active_recalibratable_versions()
base every other walk-forward caller uses), so the only difference is whether the walk-forward's
own per-step squad_optimizer.run() solve sees the four new terms. This is the walk-forward
comparison the deferred task explicitly requires before merging this wiring: "must not be
merged without a real walk-forward comparison showing it doesn't regress the headline
beats_crowd_points_delta metric."

This writes NOTHING to any committed param file and activates NO version -- same "immutable
versioning, promotion stays a human gate" property as every other sensitivity script here.

Env:
  SOWS_ARM     "off" | "on"   (required)
  SOWS_OUT     output JSON path (default data/squad_optimizer_wiring_study/arms/<arm>.json)
"""

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import backtest, db  # noqa: E402
from run_backtest import _param_versions, RECALIBRATION_SEED_DIR  # noqa: E402


def main() -> None:
    arm = os.environ["SOWS_ARM"].strip().lower()
    if arm not in ("off", "on"):
        raise SystemExit(f"SOWS_ARM must be 'off' or 'on', got {arm!r}")

    default_out = REPO_ROOT / "data" / "squad_optimizer_wiring_study" / "arms" / f"{arm}.json"
    out_path = Path(os.environ.get("SOWS_OUT") or default_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    con = db.connect()
    active = backtest.active_recalibratable_versions(RECALIBRATION_SEED_DIR)
    base_versions = _param_versions(active)

    kwargs = dict(
        compute_segments=True,
        ownership_params_version=1,
        notes=f"squad-optimizer-wiring-sensitivity arm={arm}",
    )
    if arm == "on":
        kwargs.update(
            solve_ownership_params_version=1,
            solve_risk_posture_params_version=1,
            solve_field_covariance_params_version=1,
            solve_bench_quality_params_version=1,
            solve_concentration_risk_params_version=1,
        )

    t0 = time.time()
    backtest_run_id = backtest.run(con, **base_versions, n_antithetic_pairs=5000, run_monte_carlo=True, **kwargs)
    wall = time.time() - t0

    # beats_crowd_points_delta -- the headline metric the deferred task names explicitly --
    # plus segment_calibration (ep_total_calibration_mean_resid by position/price_band), both
    # per tier (warm vs mature -- see tier_for()'s own docstring).
    headline = con.execute(
        "SELECT tier, avg(metric_value), count(*) FROM backtest_metrics "
        "WHERE backtest_run_id = ? AND metric_name = 'beats_crowd_points_delta' GROUP BY tier",
        [backtest_run_id],
    ).fetchall()
    beats_crowd_by_tier = {tier: {"mean": mean, "n": n} for tier, mean, n in headline}

    segments = con.execute(
        "SELECT metric_name, avg(metric_value), count(*) FROM backtest_metrics "
        "WHERE backtest_run_id = ? AND tier = 'mature' "
        "AND (metric_name LIKE 'ep_total_calibration_mean_resid:position=%' "
        "     OR metric_name LIKE 'ep_total_calibration_mean_resid:price_band=%') "
        "GROUP BY metric_name ORDER BY metric_name",
        [backtest_run_id],
    ).fetchall()
    segment_calibration = {name.split(":", 1)[1]: {"mean_resid": mean, "n": n} for name, mean, n in segments}

    divergence = con.execute(
        "SELECT tier, count(*), sum(CASE WHEN divergence_check_passed THEN 1 ELSE 0 END) "
        "FROM backtest_gameweek_steps WHERE backtest_run_id = ? GROUP BY tier",
        [backtest_run_id],
    ).fetchall()
    divergence_by_tier = {tier: {"n_steps": n, "n_passed": n_passed} for tier, n, n_passed in divergence}

    payload = {
        "arm": arm,
        "backtest_run_id": backtest_run_id,
        "base_param_versions": base_versions,
        "solve_kwargs": {k: v for k, v in kwargs.items() if k.startswith("solve_")},
        "beats_crowd_points_delta_by_tier": beats_crowd_by_tier,
        "segment_calibration_mature": segment_calibration,
        "divergence_by_tier": divergence_by_tier,
        "wall_seconds": round(wall, 1),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(
        f"[squad-optimizer-wiring-arm] {arm}: beats_crowd={beats_crowd_by_tier} "
        f"({wall:.0f}s) -> {out_path}"
    )
    con.close()


if __name__ == "__main__":
    main()
