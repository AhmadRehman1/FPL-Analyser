"""Priority 10 Phase A: samples real rival squads from FPL's own public API for the current
gameweek and stores them in fact_rival_squad_sample.

Deliberately a SEPARATE script from scripts/run_ingestion.py, not wired into its default
flow -- sampling n_entries means that many real HTTP requests to FPL's own API per run, which
needs real rate-limiting/caching discipline a bare "run on every ingestion" wiring wouldn't
respect. Run this on its own, lower cadence (once per gameweek is enough -- real historical
picks for an already-sampled gameweek never change, and ingest_rival_squad_sample() is
idempotent per (season, event) regardless).

Usage (from repo root):
    PYTHONPATH=src python scripts/run_rival_sample_ingestion.py [n_entries]
"""

import sys
from datetime import date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import db, ingest_fpl_entry_picks as ifp  # noqa: E402

TARGET_SEASON = "2026-2027"
DEFAULT_N_ENTRIES = 200


def main() -> None:
    n_entries = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_N_ENTRIES
    con = db.connect()

    real_run = con.execute(
        "SELECT target_gameweek FROM squad_optimizer_runs WHERE target_season = ? AND is_manager_snapshot = FALSE "
        "ORDER BY target_gameweek DESC LIMIT 1",
        [TARGET_SEASON],
    ).fetchone()
    if not real_run:
        raise SystemExit(f"no real squad_optimizer_runs row for {TARGET_SEASON} -- run scripts/run_ingestion.py first")
    event = real_run[0]

    result = ifp.ingest_rival_squad_sample(
        con, TARGET_SEASON, event, datetime.now(), n_entries=n_entries,
    )
    print(f"[rival_squad_sample] GW{event}, n_entries={n_entries} -> {result}")

    if result["status"] == "ingested" and result["picks_inserted"]:
        print("\n--- most-owned players in the sampled field ---")
        for row in ifp.most_owned_players(con, TARGET_SEASON, event):
            print(f"  {row['name']:30s} owned by {row['n_owners']} ({row['n_captains']} captains)")

    con.close()


if __name__ == "__main__":
    main()
