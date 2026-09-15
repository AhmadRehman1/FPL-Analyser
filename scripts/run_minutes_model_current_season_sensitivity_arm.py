"""One arm of the minutes-model current-season-role sensitivity study -- the real incident:
a live goalkeeper who started every match this season projected p_start_final=0.13, because
target_season's own already-played matches never entered p_start_historical_own's computation
at all (only two complete PRIOR seasons did -- see minutes_model.run()'s own docstring on
lookback_seasons/current_season_role_params_version for the full account and the fix).

lookback_seasons' own new default (unconditionally includes target_season) is provably
backtest-neutral -- see minutes_model.run()'s docstring -- and ships without a study. This
script is for the SEPARATE, opt-in current_season_role_params_version blend, which IS a real
behavior change for any target_season with its own in-progress role changes, including the
two historical seasons this walk-forward actually exercises.

Runs ONE full M7 walk-forward pass (backtest.run(), the same mechanism scripts/
run_squad_optimizer_wiring_sensitivity_arm.py already uses) over both historical seasons, at
one of two fixed arms:

  "off" -- current_season_role_params_version=None. Current production behavior (the
           lookback_seasons default fix is unconditional and applies to both arms equally --
           this study isolates ONLY the opt-in fast-reacting blend's own marginal effect).
  "on"  -- current_season_role_params_version=1 (current_season_matches_threshold=4).

Both arms hold every OTHER param version fixed (the same active_recalibratable_versions()
base every other walk-forward caller uses). This is the walk-forward comparison required
before current_season_role_params_version could ever default on anywhere: "must not regress
the headline beats_crowd_points_delta metric" -- same bar item 2's own study used.

This writes NOTHING to any committed param file and activates NO version -- same "immutable
versioning, promotion stays a human gate" property as every other sensitivity script here.

Env:
  MMCS_ARM     "off" | "on"   (required)
  MMCS_OUT     output JSON path (default data/minutes_model_current_season_study/arms/<arm>.json)
"""

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import backtest, db, minutes_model as mm  # noqa: E402
from run_backtest import _param_versions, RECALIBRATION_SEED_DIR  # noqa: E402


def main() -> None:
    arm = os.environ["MMCS_ARM"].strip().lower()
    if arm not in ("off", "on"):
        raise SystemExit(f"MMCS_ARM must be 'off' or 'on', got {arm!r}")

    default_out = REPO_ROOT / "data" / "minutes_model_current_season_study" / "arms" / f"{arm}.json"
    out_path = Path(os.environ.get("MMCS_OUT") or default_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    con = db.connect()
    active = backtest.active_recalibratable_versions(RECALIBRATION_SEED_DIR)
    base_versions = _param_versions(active)

    kwargs = dict(
        compute_segments=True,
        ownership_params_version=1,
        notes=f"minutes-model-current-season-sensitivity arm={arm}",
    )
    if arm == "on":
        mm.seed_current_season_role_params(con)
        kwargs["current_season_role_params_version"] = 1

    t0 = time.time()
    backtest_run_id = backtest.run(con, **base_versions, n_antithetic_pairs=5000, run_monte_carlo=True, **kwargs)
    wall = time.time() - t0

    headline = con.execute(
        "SELECT tier, avg(metric_value), count(*) FROM backtest_metrics "
        "WHERE backtest_run_id = ? AND metric_name = 'beats_crowd_points_delta' GROUP BY tier",
        [backtest_run_id],
    ).fetchall()
    beats_crowd_by_tier = {tier: {"mean": mean, "n": n} for tier, mean, n in headline}

    minutes_calib = con.execute(
        "SELECT metric_name, avg(metric_value), count(*) FROM backtest_metrics "
        "WHERE backtest_run_id = ? AND metric_name IN ('brier_minutes_mean', 'log_score_minutes_mean') "
        "GROUP BY metric_name",
        [backtest_run_id],
    ).fetchall()
    minutes_calibration = {name: {"mean": mean, "n": n} for name, mean, n in minutes_calib}

    payload = {
        "arm": arm,
        "backtest_run_id": backtest_run_id,
        "base_param_versions": base_versions,
        "current_season_role_params_version": kwargs.get("current_season_role_params_version"),
        "beats_crowd_points_delta_by_tier": beats_crowd_by_tier,
        "minutes_calibration": minutes_calibration,
        "wall_seconds": round(wall, 1),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(
        f"[minutes-model-current-season-arm] {arm}: beats_crowd={beats_crowd_by_tier} "
        f"minutes_calibration={minutes_calibration} ({wall:.0f}s) -> {out_path}"
    )
    con.close()


if __name__ == "__main__":
    main()
