"""End-to-end M0 pipeline: fact_raw -> fact_reconciled -> evidence_claims.

Usage (from repo root):
    .venv/Scripts/python scripts/run_ingestion.py
"""

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import db, ingest_csv, ingest_workbook, params, reconcile  # noqa: E402

DATA_ROOT = REPO_ROOT / "data" / "external" / "FPL-Core-Insights-main" / "data"
XLSX_PATH = REPO_ROOT / "data" / "external" / "FPL_202627_Master_Evidence_Database.xlsx"

SOURCE_TIER_WEIGHTS_V1 = [
    ("official", 1.0),
    ("journalist", 0.8),
    ("specialist", 0.6),
    ("community", 0.4),
]


def main() -> None:
    con = db.connect()

    t0 = time.time()
    csv_results = ingest_csv.ingest_all(con, DATA_ROOT)
    statuses = {}
    for r in csv_results:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    print(f"[fact_raw] {len(csv_results)} files in {time.time() - t0:.1f}s -> {statuses}")

    for source_type, weight in SOURCE_TIER_WEIGHTS_V1:
        params.write_param(
            con, "source_tier_weights", 1, "2026-08-10", "tier_weight",
            value_numeric=weight, dimensions={"source_type": source_type},
        )
    print("[params] source_tier_weights v1 seeded")

    t0 = time.time()
    reconcile_results = reconcile.reconcile_all(con, str(XLSX_PATH))
    print(f"[fact_reconciled] {time.time() - t0:.1f}s -> {json.dumps(reconcile_results)}")

    t0 = time.time()
    workbook_results = ingest_workbook.ingest_all(con, str(XLSX_PATH), source_tier_params_version=1)
    print(f"[evidence_claims] {time.time() - t0:.1f}s -> {json.dumps(workbook_results)}")

    con.close()


if __name__ == "__main__":
    main()
