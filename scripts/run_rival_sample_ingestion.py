"""Priority 10 Phase A: samples real rival squads from FPL's own public API for the current
gameweek and stores them in fact_rival_squad_sample.

Deliberately a SEPARATE script from scripts/run_ingestion.py, not wired into its default
flow -- sampling n_entries means that many real HTTP requests to FPL's own API per run, which
needs real rate-limiting/caching discipline a bare "run on every ingestion" wiring wouldn't
respect. Run this on its own, lower cadence (once per gameweek is enough -- real historical
picks for an already-sampled gameweek never change, and ingest_rival_squad_sample() is
idempotent per (season, event) regardless).

Phase A-1 (docs/plans/2026-08_roadmap_plan.md, Track A): default sample size scaled 200 -> 2,000
([A1]); every run now logs wall-clock time and the real error rate (entries that 404'd / total
entries sampled) so that measurement is visible for [A1]'s own "revise the number if wrong"
escape hatch; and every run purges any prior-season rows ([A9]) so retention doesn't grow
unbounded as the sample scales up -- see purge_prior_season_rival_squad_sample()'s own
docstring for why this is safe to run unconditionally rather than only at a season rollover.

Usage (from repo root):
    PYTHONPATH=src python scripts/run_rival_sample_ingestion.py [n_entries]
"""

import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import db, ingest_fpl_entry_picks as ifp  # noqa: E402

TARGET_SEASON = "2026-2027"
DEFAULT_N_ENTRIES = 2000


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

    t0 = time.time()
    result = ifp.ingest_rival_squad_sample(
        con, TARGET_SEASON, event, datetime.now(), n_entries=n_entries,
    )
    elapsed = time.time() - t0
    print(f"[rival_squad_sample] GW{event}, n_entries={n_entries} -> {result}")
    if result["status"] == "ingested":
        print(f"[rival_squad_sample] wall_clock={elapsed:.1f}s error_rate={result['error_rate']:.1%}")
        if result["error_rate"] > 0.05:
            print(f"::warning::rival_squad_sample error rate {result['error_rate']:.1%} exceeds the 5% budget (Phase A-1 [R1])")

    if result["status"] == "ingested" and result["picks_inserted"]:
        print("\n--- most-owned players in the sampled field ---")
        for row in ifp.most_owned_players(con, TARGET_SEASON, event):
            print(f"  {row['name']:30s} owned by {row['n_owners']} ({row['n_captains']} captains)")

    deleted = ifp.purge_prior_season_rival_squad_sample(con, TARGET_SEASON)
    if deleted:
        print(f"[rival_squad_sample] purged {deleted} prior-season row(s), retaining only {TARGET_SEASON}")

    con.close()


if __name__ == "__main__":
    main()
