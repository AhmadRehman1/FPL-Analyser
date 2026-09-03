"""One arm of the risk-aversion (lambda) / concentration-cap sensitivity study.

Runs ONE full-season evolving-manager simulation (backtest.run_season_simulation via
report_season_simulation_sensitivity) at ONE trial value of either `lambda_value` or
`xi_club_concentration_cap`, over a real historical season, scored on REALIZED FPL points --
then writes that arm's season_cumulative_metrics (total / mean / realized_sharpe /
max_drawdown) plus its per-gameweek action log to a JSON file.

Why this exists: the lambda in {0.05..0.30} study the README / project backlog has flagged as
"still unrun" -- it gates data/recalibration/seeds_1.json promotion, the "attack" risk-posture
default (risk_posture.py), and the deferred "protect rank" third toggle. The one existing
caller, scripts/run_season_simulation.py, runs a fixed 5-gameweek 2026-2027 window (projected
EP, not realized) -- too short and not realized-scored to settle the question. A full 2024-25 /
2025-26 season with an evolving manager, scored on real event_points, is the right evidence.

One arm is a real MIQP solve + Monte Carlo simulation + transfer-planner call per gameweek
across the whole season window (~2-3h on a GitHub runner), so the study fans out one arm per
(axis, value, season) via .github/workflows/lambda_sensitivity.yml rather than one serial job.

This writes NOTHING to any committed param file and activates NO version -- every trial value
goes through params_mod.write_param()'s immutable-versioning mechanism (writing a version never
activates it), exactly as report_season_simulation_sensitivity() already does. Promotion stays
scripts/review_recalibration.py's human gate.

Env:
  LSA_AXIS        "lambda" | "cap"          (required)
  LSA_VALUE       trial value, float         (required; e.g. 0.05 or 3)
  LSA_SEASON      "2024-2025" | "2025-2026"  (default "2025-2026")
  LSA_START_GW    first gameweek to walk     (default 2 -- GW1 is not bootstrappable, see
                                              run_season_simulation()'s own guard)
  LSA_END_GW      last gameweek to walk      (default 38)
  LSA_OUT         output JSON path           (default data/lambda_study/arms/<axis>_<value>_<season>.json)
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
    axis = os.environ["LSA_AXIS"].strip().lower()
    if axis not in ("lambda", "cap"):
        raise SystemExit(f"LSA_AXIS must be 'lambda' or 'cap', got {axis!r}")
    raw_value = os.environ["LSA_VALUE"].strip()
    value: float = float(raw_value)
    season = os.environ.get("LSA_SEASON", "2025-2026").strip()
    start_gw = int(os.environ.get("LSA_START_GW", "2"))
    end_gw = int(os.environ.get("LSA_END_GW", "38"))

    value_slug = raw_value.replace(".", "p")
    default_out = REPO_ROOT / "data" / "lambda_study" / "arms" / f"{axis}_{value_slug}_{season}.json"
    out_path = Path(os.environ.get("LSA_OUT") or default_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    con = db.connect()
    active = backtest.active_recalibratable_versions(RECALIBRATION_SEED_DIR)
    base_versions = _param_versions(active)

    grid_kwarg = "lambda_grid" if axis == "lambda" else "guardrail_cap_grid"
    grid_value: tuple = (value,)

    t0 = time.time()
    sensitivity = backtest.report_season_simulation_sensitivity(
        con, season, start_gw, end_gw, base_versions, **{grid_kwarg: grid_value},
    )
    wall = time.time() - t0

    bucket = sensitivity["lambda"] if axis == "lambda" else sensitivity["guardrail_cap"]
    # report_season_simulation_sensitivity keys the bucket by the raw grid value it was handed.
    (key, arm), = bucket.items()
    actions = arm.pop("actions", [])

    n_transfers = sum(1 for a in actions if a.get("action") == "transfer")
    n_hits = sum(1 for a in actions if a.get("action") == "transfer" and (a.get("detail") or "").find("-4") != -1)
    n_chips = sum(1 for a in actions if a.get("action") == "chip")

    payload = {
        "axis": axis,
        "value": value,
        "season": season,
        "start_gameweek": start_gw,
        "end_gameweek": end_gw,
        "base_param_versions": base_versions,
        "metrics": arm,  # season_cumulative_metrics: total_points / mean_points / realized_sharpe / max_drawdown
        "n_gameweeks_scored": arm.get("n_gameweeks"),
        "action_counts": {"transfers": n_transfers, "hits_taken": n_hits, "chips_played": n_chips},
        "actions": actions,
        "wall_seconds": round(wall, 1),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(
        f"[lambda-arm] {axis}={value} {season} GW{start_gw}-{end_gw}: "
        f"total={arm.get('total_points')} sharpe={arm.get('realized_sharpe')} "
        f"drawdown={arm.get('max_drawdown')} ({wall:.0f}s) -> {out_path}"
    )
    con.close()


if __name__ == "__main__":
    main()
