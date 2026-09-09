"""Priority 10 Phase B: samples real rival squads from FPL's own public API for every
COMPLETED 2026-27 gameweek not yet sampled, into fact_rival_squad_sample. Feeds
scripts/rank_autopsy.py.

Deliberately a SEPARATE script from scripts/run_ingestion.py, not wired into its default
flow -- sampling means ~250 real HTTP requests to FPL's own API per gameweek, which needs
real rate-limiting/caching discipline a bare "run on every ingestion" wiring wouldn't
respect. .github/workflows/rank_tracking.yml runs this once a week; ingest_rival_squad_sample
is idempotent per (season, event), so a re-run is a no-op and a first run backfills the
whole season's completed gameweeks in one pass (bounded by MAX_GAMEWEEKS_PER_RUN).

Sampling shape: ingest_fpl_entry_picks.DEFAULT_RANK_BANDS -- a STRATIFIED sample around the
model-managed team's top-100k rank target (plus elite, mid-field and 600k-2M tail slices),
not the old top-N-by-rank. The top-200 the Phase A scaffold sampled measured the model
against the 200 best managers alive, which is the wrong bar for a top-100k goal. Whether FPL's
leagues-classic standings paginate deep enough to reach the 60k-140k band is not verifiable
from this sandbox (fantasy.premierleague.com is network-blocked here); fetch_entries_in_rank_bands
reports the bands it ACTUALLY reached, logged below, so a truncated sample is visible rather
than assumed.

[A9] retention: every run purges any non-2026-27 rows so the volume of real people's picks
retained stays bounded -- see purge_prior_season_rival_squad_sample's own docstring.

Usage (from repo root):
    PYTHONPATH=src python scripts/run_rival_sample_ingestion.py [gameweek]

    With no argument: samples every completed gameweek not yet in the table (oldest first,
    up to MAX_GAMEWEEKS_PER_RUN). With a gameweek number: samples just that one.

    RIVAL_SAMPLE_REPLACE_EVENTS="1,2" forces those gameweeks to be re-sampled even though
    they are already in the table -- for a stored sample built with an older/wrong strategy
    (the retired top-2000 cohort, or a sample taken before a DEFAULT_RANK_BANDS change). The
    old rows are only dropped once the fresh sample is in hand. Deliberately an explicit
    opt-in, not an auto-detect: re-sampling costs ~250 real FPL API requests per gameweek.
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import db, field_rank, ingest_fpl_entry_picks as ifp  # noqa: E402

TARGET_SEASON = "2026-2027"
# One run never samples more than this many gameweeks -- a first run mid-season (or after a
# season-rollover purge) would otherwise fire thousands of requests at once. The weekly
# cadence catches up over subsequent runs; a manual `... <gameweek>` call bypasses this.
MAX_GAMEWEEKS_PER_RUN = 6


def _completed_unsampled_gameweeks(con) -> list[int]:
    settled = [
        r[0] for r in con.execute(
            "SELECT gameweek FROM fact_match WHERE season = ? GROUP BY gameweek "
            "HAVING count(*) > 0 AND count(*) = count(*) FILTER (WHERE finished) ORDER BY gameweek",
            [TARGET_SEASON],
        ).fetchall()
    ]
    sampled = {
        r[0] for r in con.execute(
            "SELECT DISTINCT event FROM fact_rival_squad_sample WHERE season = ?", [TARGET_SEASON],
        ).fetchall()
    }
    return [gw for gw in settled if gw not in sampled]


def _replace_events() -> set[int]:
    raw = os.environ.get("RIVAL_SAMPLE_REPLACE_EVENTS", "").strip()
    return {int(tok) for tok in raw.replace(" ", "").split(",") if tok.isdigit()}


def _sample_one(con, event: int, *, replace: bool = False) -> None:
    t0 = time.time()
    result = ifp.ingest_rival_squad_sample(
        con, TARGET_SEASON, event, datetime.now(), bands=ifp.DEFAULT_RANK_BANDS, replace=replace,
    )
    elapsed = time.time() - t0
    print(f"[rival_squad_sample] GW{event} -> {result['status']}: "
          f"{result.get('picks_inserted', 0)} picks from {result.get('entries_sampled', 0)} entries")
    if result["status"] != "ingested":
        return
    print(f"[rival_squad_sample] GW{event} wall_clock={elapsed:.1f}s error_rate={result['error_rate']:.1%} "
          f"achieved_bands={result.get('achieved_bands')}")
    if result["error_rate"] > 0.05:
        print(f"::warning::rival_squad_sample GW{event} error rate {result['error_rate']:.1%} exceeds the 5% budget")
    for want, got in zip(ifp.DEFAULT_RANK_BANDS, result.get("achieved_bands") or []):
        if got[2] < want[2] * 0.75:
            print(f"::warning::rival_squad_sample GW{event} band {want[0]}-{want[1]} wanted {want[2]}, "
                  f"reached only rank {got[1]} with {got[2]} entries (standings pagination frontier?)")
    if result["picks_inserted"]:
        print(f"  most-owned in the GW{event} sample:")
        for row in ifp.most_owned_players(con, TARGET_SEASON, event):
            print(f"    {row['name']:28s} {row['n_owners']:4d} owners  {row['n_captains']:3d} captains")


def main() -> None:
    con = db.connect()

    replace = _replace_events()
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        targets = [int(sys.argv[1])]
    else:
        targets = _completed_unsampled_gameweeks(con)
        if len(targets) > MAX_GAMEWEEKS_PER_RUN:
            print(f"[rival_squad_sample] {len(targets)} gameweeks to backfill; taking the oldest "
                  f"{MAX_GAMEWEEKS_PER_RUN} this run, the rest next run")
            targets = targets[:MAX_GAMEWEEKS_PER_RUN]
        # explicit re-sample requests jump the MAX_GAMEWEEKS_PER_RUN queue
        targets = sorted(set(targets) | replace)

    if replace:
        print(f"[rival_squad_sample] RIVAL_SAMPLE_REPLACE_EVENTS -> forcing re-sample of GW {sorted(replace)}")
    if not targets:
        print(f"[rival_squad_sample] nothing to do -- every completed {TARGET_SEASON} gameweek is already sampled")
    for event in targets:
        if not field_rank.gameweek_is_settled(con, TARGET_SEASON, event):
            print(f"[rival_squad_sample] GW{event} not settled yet -- skipping")
            continue
        _sample_one(con, event, replace=event in replace)

    deleted = ifp.purge_prior_season_rival_squad_sample(con, TARGET_SEASON)
    if deleted:
        print(f"[rival_squad_sample] purged {deleted} prior-season row(s), retaining only {TARGET_SEASON}")

    con.close()


if __name__ == "__main__":
    main()
