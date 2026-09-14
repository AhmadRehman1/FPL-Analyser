"""One arm of the season-horizon chip-timing sensitivity study (docs/reports/2026-09_chip_
policy_and_scoring_diagnosis.md, Workstream B's "fuller ask" -- see
transfer_planner.seed_v1_params()'s own comment on triple_captain_timing_params/
bench_boost_timing_params for what this actually computes and why it's opt-in).

Runs ONE full-season evolving-manager simulation (backtest.run_season_simulation(), same
mechanism scripts/run_lambda_sensitivity_arm.py already uses for the lambda study) over a real
historical season, scored on REALIZED FPL points, at one of two fixed arms:

  "off" -- triple_captain_timing_params_version=None, bench_boost_timing_params_version=None.
           The CURRENT greedy approach: PR #173's magnitude floor is on (both threshold
           versions pinned to 1, matching forward_season_sim.py's real live default), but no
           season-horizon timing check -- CHIP_TIMING_FIELD_SEASON never fires.
  "on"  -- same, plus triple_captain_timing_params_version=1, bench_boost_timing_params_version=1.
           Adds the wider-window timing check on top.

Both arms hold every OTHER param version fixed (the same active_recalibratable_versions() +
threshold=1 base every other season-sim caller uses), so the only thing that differs between
"off" and "on" is whether the new season-horizon timing gate can hold a chip that the narrow
visible-horizon check alone would have played. This is the comparison
docs/reports/2026-09_chip_policy_and_scoring_diagnosis.md's Workstream B explicitly asks for
before any default is flipped on.

This writes NOTHING to any committed param file and activates NO version -- same "immutable
versioning, promotion stays a human gate" property as every other sensitivity script here.

Env:
  CTS_ARM          "off" | "on"               (required)
  CTS_SEASON       "2024-2025" | "2025-2026"  (default "2025-2026")
  CTS_START_GW     first gameweek to walk     (default 2 -- GW1 is not bootstrappable)
  CTS_END_GW       last gameweek to walk      (default 38)
  CTS_OUT          output JSON path           (default data/chip_timing_study/arms/<arm>_<season>.json)
"""

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import backtest, db  # noqa: E402
from run_season_simulation import _param_versions  # noqa: E402  -- same version resolution every season-sim caller uses

RECALIBRATION_SEED_DIR = REPO_ROOT / "data" / "recalibration"


def main() -> None:
    arm = os.environ["CTS_ARM"].strip().lower()
    if arm not in ("off", "on"):
        raise SystemExit(f"CTS_ARM must be 'off' or 'on', got {arm!r}")
    season = os.environ.get("CTS_SEASON", "2025-2026").strip()
    start_gw = int(os.environ.get("CTS_START_GW", "2"))
    end_gw = int(os.environ.get("CTS_END_GW", "38"))

    default_out = REPO_ROOT / "data" / "chip_timing_study" / "arms" / f"{arm}_{season}.json"
    out_path = Path(os.environ.get("CTS_OUT") or default_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    con = db.connect()
    active = backtest.active_recalibratable_versions(RECALIBRATION_SEED_DIR)
    base_versions = _param_versions(active)
    base_versions["triple_captain_threshold_params_version"] = 1
    base_versions["bench_boost_threshold_params_version"] = 1
    if arm == "on":
        base_versions["triple_captain_timing_params_version"] = 1
        base_versions["bench_boost_timing_params_version"] = 1

    t0 = time.time()
    result = backtest.run_season_simulation(con, season, start_gw, end_gw, **base_versions)
    wall = time.time() - t0

    metrics = backtest.season_cumulative_metrics(result["weekly_points"])
    actions = result["actions"]
    n_transfers = sum(1 for a in actions if a.get("accepted_transfer_rank") is not None)
    chips_played = [(a["gameweek"], a["accepted_chip"]) for a in actions if a.get("accepted_chip")]

    payload = {
        "arm": arm,
        "season": season,
        "start_gameweek": start_gw,
        "end_gameweek": end_gw,
        "base_param_versions": base_versions,
        "metrics": metrics,
        "n_gameweeks_scored": metrics.get("n_gameweeks"),
        "action_counts": {"transfers": n_transfers, "chips_played": len(chips_played)},
        "chips_played": chips_played,
        "skipped_dgw_gameweeks": result["skipped_dgw_gameweeks"],
        "wall_seconds": round(wall, 1),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(
        f"[chip-timing-arm] {arm} {season} GW{start_gw}-{end_gw}: "
        f"total={metrics.get('total_points')} sharpe={metrics.get('realized_sharpe')} "
        f"chips={chips_played} ({wall:.0f}s) -> {out_path}"
    )
    con.close()


if __name__ == "__main__":
    main()
