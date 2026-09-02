"""One recalibration STAGE against an already-computed walk-forward, so
weekly_backtest.yml's monolithic run_backtest.py -- walk-forward + every refit technique in a
single job that has been cancelled at the 6h Actions cap on every real dispatch, producing
zero proposals ever -- can be split across separate, individually-budgeted jobs.

`backtest.recalibrate()` already has a per-technique flag for exactly this ("their real costs
differ by roughly two orders of magnitude ... a caller may reasonably want to run them
separately"). This script runs ONE technique per invocation against `max(backtest_run_id)`
(whatever the restored walk-forward DB holds), writes/refreshes a human-readable
`data/recalibration/proposals_<date>.json`, and -- unless --no-commit -- commits it plus the
`seeds_<run_id>.json` file recalibrate() itself writes. recalibrate.yml runs the cheap stages
first, so a timeout in `minutes` or `lambda` still lands the xi_rho / rho_residual / kappa_tc
proposals.

This NEVER auto-promotes. Activation stays the human gate (scripts/review_recalibration.py) --
consistent with data/recalibration/seeds_1.json being parked pending the owner's review.

Usage (from repo root, walk-forward already in the DB):
    PYTHONPATH=src python scripts/run_recalibrate.py --stage {xi_rho|rho_residual|kappa_tc|minutes|lambda}
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import backtest, db  # noqa: E402
from run_backtest import MINUTES_PARAM_GRIDS, RECALIBRATION_SEED_DIR  # noqa: E402

STAGE_FLAGS = {
    "xi_rho": "refit_xi_rho_flag",
    "rho_residual": "refit_rho_residual_flag",
    "minutes": "refit_minutes_flag",
    "lambda": "refit_lambda_flag",
    "kappa_tc": "refit_kappa_tc_flag",
}
PROPOSALS_JSON = REPO_ROOT / "data" / "recalibration" / f"proposals_{date.today().isoformat()}.json"


def _dump_proposals(con, backtest_run_id: int) -> int:
    rows = con.execute(
        """
        SELECT proposal_id, param_family, param_key, dimensions, old_params_version,
               new_params_version, old_value, new_value, metric_name, metric_before,
               metric_after, status
        FROM recalibration_proposals WHERE backtest_run_id = ? ORDER BY proposal_id
        """,
        [backtest_run_id],
    ).fetchdf()
    def _opt(v):
        return None if v is None or (isinstance(v, float) and pd.isna(v)) else v

    proposals = [
        {
            "proposal_id": int(r.proposal_id),
            "param_family": r.param_family, "param_key": r.param_key,
            "dimensions": json.loads(r.dimensions) if _opt(r.dimensions) else None,
            "old_params_version": None if _opt(r.old_params_version) is None else int(r.old_params_version),
            "new_params_version": int(r.new_params_version),
            "old_value": None if _opt(r.old_value) is None else float(r.old_value),
            "new_value": None if _opt(r.new_value) is None else float(r.new_value),
            "metric": r.metric_name,
            "metric_before": float(r.metric_before), "metric_after": float(r.metric_after),
            "metric_delta": round(float(r.metric_after) - float(r.metric_before), 5),
            "status": r.status,
        }
        for r in rows.itertuples(index=False)
    ]
    payload = {
        "backtest_run_id": backtest_run_id,
        "generated_at": date.today().isoformat(),
        "n_proposals": len(proposals),
        "note": "Proposals only -- nothing is activated. Review and confirm with "
                "scripts/review_recalibration.py. Every proposal must improve its own "
                "out-of-sample metric to be listed here (recalibrate() writes a row only on "
                "a strict improvement).",
        "proposals": proposals,
    }
    PROPOSALS_JSON.parent.mkdir(parents=True, exist_ok=True)
    PROPOSALS_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return len(proposals)


def _commit(stage: str) -> None:
    script = REPO_ROOT / "scripts" / "ci_commit_generated.sh"
    subprocess.run(
        ["bash", str(script), f"chore: recalibration proposals ({stage}) [skip ci]",
         str(PROPOSALS_JSON.relative_to(REPO_ROOT)), "data/recalibration/"],
        cwd=str(REPO_ROOT), check=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=sorted(STAGE_FLAGS))
    ap.add_argument("--lambda-grid", default=None, help="comma-separated subset, e.g. '0.0,0.05,0.15'")
    ap.add_argument("--no-commit", action="store_true")
    args = ap.parse_args()

    con = db.connect()
    row = con.execute("SELECT max(backtest_run_id) FROM backtest_runs").fetchone()
    if row is None or row[0] is None:
        raise SystemExit("no backtest_runs row -- run scripts/run_walkforward.py first "
                         "(recalibrate.yml restores a cached walk-forward DB before this step)")
    backtest_run_id = row[0]
    active = backtest.active_recalibratable_versions(RECALIBRATION_SEED_DIR)

    flags = {v: False for v in STAGE_FLAGS.values()}
    flags[STAGE_FLAGS[args.stage]] = True
    kwargs: dict = {}
    if args.stage == "lambda" and args.lambda_grid:
        kwargs["lambda_grid"] = tuple(float(x) for x in args.lambda_grid.split(","))

    print(f"[run_recalibrate] stage={args.stage} against backtest_run_id={backtest_run_id}")
    t0 = time.time()
    proposal_ids = backtest.recalibrate(
        con, backtest_run_id,
        current_xi_version=active["xi_params_version"], current_rho_version=active["rho_params_version"],
        current_rho_residual_version=active["rho_residual_params_version"],
        current_minutes_versions={
            "decay_params_version": 1, "adjustment_params_version": active["adjustment_params_version"],
            "shrinkage_params_version": active["shrinkage_params_version"],
            "fact_multiplier_params_version": active["fact_multiplier_params_version"],
        },
        current_lambda_version=active["lambda_params_version"],
        guardrail_cap=3,
        minutes_param_grids=MINUTES_PARAM_GRIDS,
        current_kappa_tc_version=active["kappa_tc_params_version"],
        seed_dir=RECALIBRATION_SEED_DIR,
        **flags, **kwargs,
    )
    n_total = _dump_proposals(con, backtest_run_id)
    con.close()
    print(f"[run_recalibrate] {time.time() - t0:.1f}s -> stage '{args.stage}' added "
          f"{len(proposal_ids)} proposal(s); {n_total} total for run {backtest_run_id} -> {PROPOSALS_JSON.name}")

    if not args.no_commit:
        _commit(args.stage)


if __name__ == "__main__":
    main()
