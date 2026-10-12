"""Derives the real top-100k cutoffs (fpl_quant.rank_cutoff) from the sampled rival entries' FPL
history and adds them to data/dashboard/rank_autopsy.json as `top_100k_cutoff`
(docs/plans/2026-10_beat_the_template.md, Req 4): the 2024-25 and 2025-26 rank-100k season
totals and paces, and the 2026-27 cumulative cutoff per gameweek. Runs in rank_tracking.yml
after the rank autopsy (which rewrites the file), so the key is added back every week. Only the
derived cutoffs are written; no per-entry history is kept.

~1 FPL API request per sampled entry (~250), paced. fantasy.premierleague.com is reachable from
Actions, not from the Claude Code sandbox.

Usage (from repo root):
    PYTHONPATH=src python scripts/derive_rank_cutoff.py
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import db, rank_cutoff  # noqa: E402

CURRENT_SEASON = "2026-2027"
PAST_SEASONS = ["2024-2025", "2025-2026"]
AUTOPSY_PATH = REPO_ROOT / "data" / "dashboard" / "rank_autopsy.json"


def sampled_entry_ids(con, season: str = CURRENT_SEASON) -> list[int]:
    return [r[0] for r in con.execute(
        "SELECT DISTINCT entry_id FROM fact_rival_squad_sample WHERE season = ? ORDER BY entry_id", [season],
    ).fetchall()]


def merge_into_autopsy(cutoffs: dict, path: Path = AUTOPSY_PATH) -> None:
    payload = json.loads(path.read_text()) if path.exists() else {}
    payload["top_100k_cutoff"] = cutoffs
    path.write_text(json.dumps(payload, indent=2))


def main(fetch=rank_cutoff.fetch_entry_history, path: Path = AUTOPSY_PATH, entry_ids: list[int] | None = None) -> int:
    if entry_ids is None:
        con = db.connect(read_only=True)
        entry_ids = sampled_entry_ids(con)
        con.close()
    if not entry_ids:
        print("::warning::derive_rank_cutoff: no sampled rival entries -- top_100k_cutoff not written")
        return 0
    histories = rank_cutoff.fetch_histories(entry_ids, fetch=fetch)
    cutoffs = rank_cutoff.derive_cutoffs(histories, PAST_SEASONS, CURRENT_SEASON)
    merge_into_autopsy(cutoffs, path)
    print(f"[derive_rank_cutoff] {len(histories)}/{len(entry_ids)} entry histories")
    for season, cut in cutoffs["seasons"].items():
        if cut is None:
            print(f"  {season}: no entry has this season in its history")
            continue
        print(f"  {season}: rank-100k total {cut['total_points']} ({cut['pace_per_gw']}/GW), {cut['method']}, "
              f"bracket {cut['bracket_ranks']}, {cut['n_near_better']} + {cut['n_near_worse']} entries within 2x")
    by_gw = cutoffs["current_season"]["by_gameweek"]
    if by_gw:
        last = by_gw[-1]
        print(f"  {CURRENT_SEASON} GW{last['gameweek']}: cumulative cutoff {last['total_points']}, {last['method']}, "
              f"bracket {last['bracket_ranks']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
