"""One arm of a season-simulation experiment, scored against FPL's real average.

The walk-forward (scripts/run_walkforward.py) scores a fresh, hit-free squad every gameweek. The
live model team is different: it carries a squad, makes transfers, takes hits and plays chips
through transfer_planner.run(). This runs that evolving manager over a real historical season
(backtest.run_season_simulation()) with the live model team's own settings
(forward_season_sim._resolve_versions()), then scores its net points (hits deducted) against
FPL's real average_entry_score week by week.

With no flags this is the live control. Each flag changes one setting, as an immutable param
version (nothing is activated):

    --lambda 0.05            risk_aversion_params lambda_value
    --assist-prior-xa 30     FPL/xA assist calibration prior (--no-assists turns it off)
    --chip-timing            season-horizon TC/BB timing gate
    --chip-wait              ... plus the "is a later week better" wait rule
    --chip-option-value      ... the wait rule counting every week left in the half (v2), Free Hit too
    --transfer-threshold 1.0 accept the top transfer only above this net value
    --multi-transfers        two-transfer moves, taking a hit when only one is free
    --team-strength off      the team-strength model from before 2026-10-06 (or another arm of
                             run_walkforward.py --team-strength)

Usage (from repo root):
    PYTHONPATH=src python scripts/run_season_sim_arm.py --label control --season 2025-2026
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from fpl_quant import backtest, db  # noqa: E402
from fpl_quant import forward_season_sim as fss  # noqa: E402
from run_walkforward import _experiment_versions  # noqa: E402

RECALIBRATION_SEED_DIR = REPO_ROOT / "data" / "recalibration"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True, help="arm name; 'control' is the baseline the aggregator pairs against")
    parser.add_argument("--season", default="2025-2026")
    parser.add_argument("--start-gw", type=int, default=2, help="GW1 is not bootstrappable")
    parser.add_argument("--end-gw", type=int, default=38)
    parser.add_argument("--out", default=None)
    parser.add_argument("--lambda", dest="lambda_value", type=float, default=None)
    parser.add_argument("--assist-prior-xa", type=float, default=None)
    parser.add_argument("--no-assists", action="store_true", help="assist calibration off")
    parser.add_argument("--chip-timing", action="store_true")
    parser.add_argument("--chip-wait", action="store_true", help="implies --chip-timing")
    parser.add_argument("--chip-option-value", action="store_true",
                        help="chip_wait_params v2: every week left in the half counts; implies --chip-timing")
    parser.add_argument("--transfer-threshold", type=float, default=None)
    parser.add_argument("--multi-transfers", action="store_true",
                        help="multi_transfer_params v1: two-transfer moves and hits")
    parser.add_argument("--team-strength", default=None,
                        help="a team-strength arm, as run_walkforward.py --team-strength "
                             "(team_strength.GUARD_ARMS, or off)")
    return parser.parse_args(argv)


def arm_versions(con, args: argparse.Namespace) -> tuple[dict, dict]:
    """(run_season_simulation kwargs, the settings this arm changed)."""
    active = backtest.active_recalibratable_versions(RECALIBRATION_SEED_DIR)
    versions = fss._resolve_versions(con, active)
    changed: dict = {}
    # run_walkforward's flags map to the same param versions; role_matches_threshold is a
    # walk-forward-only lever, so it is not offered here.
    experiment = _experiment_versions(con, argparse.Namespace(
        lambda_value=args.lambda_value, role_matches_threshold=None, assist_prior_xa=args.assist_prior_xa,
        team_strength=args.team_strength,
    ))
    versions.update(experiment)
    changed.update(experiment)
    if args.no_assists:
        versions["assist_calibration_params_version"] = None
        changed["assist_calibration_params_version"] = None
    if args.chip_timing or args.chip_wait or args.chip_option_value:
        for key in ("triple_captain_timing_params_version", "bench_boost_timing_params_version"):
            versions[key] = changed[key] = 1
    if args.chip_wait:
        versions["chip_wait_params_version"] = changed["chip_wait_params_version"] = 1
    if args.chip_option_value:
        versions["chip_wait_params_version"] = changed["chip_wait_params_version"] = 2
    if args.transfer_threshold is not None:
        versions["accept_transfer_if_net_value_above"] = changed["accept_transfer_if_net_value_above"] = args.transfer_threshold
    if args.multi_transfers:
        versions["multi_transfer_params_version"] = changed["multi_transfer_params_version"] = 1
    return versions, changed


def arm_payload(args: argparse.Namespace, changed: dict, versions: dict, result: dict,
                complete: bool, wall_seconds: float) -> dict:
    actions = result["actions"]
    return {
        "label": args.label,
        "season": args.season,
        "start_gameweek": args.start_gw,
        "end_gameweek": args.end_gw,
        "complete": complete,
        "changed": changed,
        "versions": versions,
        "gross_metrics": backtest.season_cumulative_metrics(result["weekly_points"]),
        "real_benchmark": result["real_benchmark"],
        "gameweeks": result["gameweeks"],
        "weekly_points": result["weekly_points"],
        "weekly_hits": result["weekly_hits"],
        "weekly_real_avg": result["weekly_real_avg"],
        # a two-transfer move counts as two
        "n_transfers": sum(
            a.get("n_transfers", 1 if a.get("accepted_transfer_rank") is not None else 0) for a in actions
        ),
        "chips_played": [(a["gameweek"], a["accepted_chip"]) for a in actions if a.get("accepted_chip")],
        "skipped_dgw_gameweeks": result["skipped_dgw_gameweeks"],
        "wall_seconds": round(wall_seconds, 1),
    }


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    out_path = Path(args.out or REPO_ROOT / "data" / "season_sim_arms" / f"{args.label}_{args.season}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    con = db.connect()
    backtest.materialize_confirmed_seeds(con, RECALIBRATION_SEED_DIR)
    versions, changed = arm_versions(con, args)

    t0 = time.time()

    def write(result: dict, complete: bool) -> dict:
        payload = arm_payload(args, changed, versions, result, complete, time.time() - t0)
        out_path.write_text(json.dumps(payload, indent=2, default=str))
        if not complete:
            print(f"[season-sim-arm] {args.label} GW{payload['gameweeks'][-1]} scored "
                  f"({payload['wall_seconds']:.0f}s)", flush=True)
        return payload

    # checkpoint after every gameweek: a chunk cut off by the job time limit keeps what it scored
    result = backtest.run_season_simulation(
        con, args.season, args.start_gw, args.end_gw, **versions,
        on_gameweek=lambda partial: write(partial, complete=False),
    )
    payload = write(result, complete=True)
    wall = payload["wall_seconds"]
    rb = result["real_benchmark"]
    print(
        f"[season-sim-arm] {args.label} {args.season} GW{args.start_gw}-{args.end_gw}: "
        f"net {rb['net_points_per_gw']} vs real {rb['real_avg_per_gw']} "
        f"(beats by {rb['beats_real_avg_per_gw']}/GW over {rb['n_gameweeks']} GWs), hits {rb['total_hits']}, "
        f"chips {payload['chips_played']} ({wall:.0f}s) -> {out_path}"
    )
    con.close()


if __name__ == "__main__":
    main()
