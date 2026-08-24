"""Human review gate for M7's recalibration_proposals.

recalibrate() writes each proposal's candidate value as a normal new param_versions row
(write_param() is unchanged -- writing a version never activates it, resolve_param() is
explicit-version-only per params.py's own docstring) plus a recalibration_proposals row
recording the metric delta that justifies it, status='pending'. This script is the review
step: it never edits scripts/run_ingestion.py's version-number arguments itself -- confirming
a proposal here only marks it reviewed; a human still has to go update the explicit version
number the live pipeline passes for that param family (or, for a family run_ingestion.py loads
via load_confirmed_recalibration_seeds(), the confirmed value is picked up automatically on the
next run -- see SEED_DIR below).

Every status change here also re-writes the JSON seed file for that proposal's backtest run
(backtest.write_recalibration_seed_file()), so the committed file and the DB's own
recalibration_proposals.status never drift apart -- confirming/rejecting only in the DB and
forgetting to touch the file would silently leave a stale status sitting in git.

Usage (from repo root):
    .venv/Scripts/python scripts/review_recalibration.py                       # list pending
    .venv/Scripts/python scripts/review_recalibration.py --confirm 3 --reviewed-by alex
    .venv/Scripts/python scripts/review_recalibration.py --reject 7 --reviewed-by alex
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import backtest  # noqa: E402
from fpl_quant import db  # noqa: E402

SEED_DIR = REPO_ROOT / "data" / "recalibration"


def list_pending(con) -> None:
    rows = con.execute(
        "SELECT proposal_id, param_family, param_key, dimensions, old_value, new_value, "
        "metric_name, metric_before, metric_after, old_params_version, new_params_version "
        "FROM recalibration_proposals WHERE status = 'pending' ORDER BY proposal_id"
    ).fetchall()
    if not rows:
        print("No pending recalibration proposals.")
        return
    for (proposal_id, family, key, dims, old_value, new_value, metric_name,
         metric_before, metric_after, old_version, new_version) in rows:
        dims_str = f" {dims}" if dims else ""
        print(
            f"#{proposal_id}  {family}.{key}{dims_str}: "
            f"{old_value} (v{old_version}) -> {new_value} (v{new_version})  "
            f"[{metric_name}: {metric_before:.4f} -> {metric_after:.4f}]"
        )


def set_status(con, proposal_id: int, status: str, reviewed_by: str | None) -> None:
    row = con.execute(
        "SELECT status, backtest_run_id FROM recalibration_proposals WHERE proposal_id = ?", [proposal_id]
    ).fetchone()
    if row is None:
        print(f"No proposal #{proposal_id} found.")
        return
    old_status, backtest_run_id = row
    con.execute(
        "UPDATE recalibration_proposals SET status = ?, reviewed_by = ?, reviewed_at = ? WHERE proposal_id = ?",
        [status, reviewed_by, datetime.now(timezone.utc), proposal_id],
    )
    seed_path = backtest.write_recalibration_seed_file(con, backtest_run_id, SEED_DIR)
    print(f"#{proposal_id} -> {status} (was {old_status}). Seed file updated: {seed_path}")
    if status == "confirmed":
        print(
            "Reminder: this does not activate the new version by itself. If run_ingestion.py loads "
            "this param family via load_confirmed_recalibration_seeds(), the confirmed value is picked "
            "up automatically on the next run -- otherwise, update the explicit version-number argument "
            "scripts/run_ingestion.py passes for this param family to actually use it live."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--confirm", type=int, metavar="PROPOSAL_ID")
    parser.add_argument("--reject", type=int, metavar="PROPOSAL_ID")
    parser.add_argument("--reviewed-by", type=str, default=None)
    args = parser.parse_args()

    con = db.connect()
    if args.confirm is not None:
        set_status(con, args.confirm, "confirmed", args.reviewed_by)
    elif args.reject is not None:
        set_status(con, args.reject, "rejected", args.reviewed_by)
    else:
        list_pending(con)
    con.close()


if __name__ == "__main__":
    main()
