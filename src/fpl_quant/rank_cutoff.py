"""The real top-100k cutoff: the total points at overall rank 100,000, per finished season and per
gameweek of the current one (docs/plans/2026-10_beat_the_template.md, Req 4; the yardstick for
its G2-G4).

Source: FPL's public entry/{id}/history for the entries scripts/run_rival_sample_ingestion.py
already sampled (fact_rival_squad_sample). Each history carries the entry's earlier seasons
(`past`: season_name, total_points, rank) and this season's gameweeks (`current`: event,
total_points, overall_rank). total_points is FPL's own, net of transfer hits. Across the sampled
entries, total points are interpolated against overall rank at the target rank, linearly in
log(rank) between the nearest entry ranked at or better than the target and the nearest ranked
worse. When the entries don't bracket the target, the nearest entry's total stands in and the
result says so (method "nearest").

Only the derived cutoffs are kept (rank_autopsy.json's top_100k_cutoff key); no per-entry
history outlives the run, the rival sample's own retention rule.
"""

import math
import time
from datetime import datetime, timezone

from . import ingest_fpl_entry_picks as ifp

TARGET_RANK = 100_000
GAMEWEEKS_PER_SEASON = 38
# entries counted as "near" the target when reporting how well it is bracketed
NEAR_FACTOR = 2.0
REQUEST_PAUSE_SECONDS = 0.2


def season_name(season: str) -> str:
    """'2024-2025' -> FPL's '2024/25'."""
    start, end = season.split("-")
    return f"{start}/{end[2:]}"


def fetch_entry_history(entry_id: int) -> dict | None:
    """entry/{id}/history, or None for an entry FPL no longer serves (404)."""
    import requests

    try:
        return ifp._fetch_json(f"{ifp.FPL_API_BASE}/entry/{entry_id}/history/")
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return None
        raise


def interpolate_at_rank(points: list[tuple[int, float]], target_rank: int = TARGET_RANK) -> dict | None:
    """points: (overall rank, total points) per entry. None with no usable point."""
    usable = sorted((int(r), float(t)) for r, t in points if r and r > 0 and t is not None)
    if not usable:
        return None
    better = [p for p in usable if p[0] <= target_rank]
    worse = [p for p in usable if p[0] > target_rank]
    near = [p for p in usable if target_rank / NEAR_FACTOR <= p[0] <= target_rank * NEAR_FACTOR]
    out = {
        "n_entries": len(usable),
        "n_near_better": sum(1 for p in near if p[0] <= target_rank),
        "n_near_worse": sum(1 for p in near if p[0] > target_rank),
    }
    if better and worse:
        (r1, t1), (r2, t2) = better[-1], worse[0]
        if r1 == target_rank or r1 == r2:
            total = t1
        else:
            w = (math.log(target_rank) - math.log(r1)) / (math.log(r2) - math.log(r1))
            total = t1 + w * (t2 - t1)
        return {**out, "total_points": round(total, 1), "bracket_ranks": [r1, r2], "method": "interpolated"}
    nearest = min(usable, key=lambda p: abs(math.log(p[0]) - math.log(target_rank)))
    return {**out, "total_points": round(nearest[1], 1), "bracket_ranks": [nearest[0]], "method": "nearest"}


def derive_cutoffs(histories: list[dict], past_seasons: list[str], current_season: str,
                   target_rank: int = TARGET_RANK) -> dict:
    """The cutoff for each past season (total and pace per gameweek) and, per gameweek of the
    current season, the cumulative total at the target rank."""
    seasons = {}
    for season in past_seasons:
        name = season_name(season)
        pts = [(p.get("rank"), p.get("total_points")) for h in histories for p in (h.get("past") or [])
               if p.get("season_name") == name]
        cut = interpolate_at_rank(pts, target_rank)
        if cut is not None:
            cut["pace_per_gw"] = round(cut["total_points"] / GAMEWEEKS_PER_SEASON, 3)
        seasons[season] = cut
    by_event: dict[int, list[tuple[int, float]]] = {}
    for h in histories:
        for row in h.get("current") or []:
            if row.get("event") is not None:
                by_event.setdefault(int(row["event"]), []).append((row.get("overall_rank"), row.get("total_points")))
    current = []
    for event in sorted(by_event):
        cut = interpolate_at_rank(by_event[event], target_rank)
        if cut is not None:
            current.append({"gameweek": event, **cut})
    return {
        "target_rank": target_rank,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_histories": len(histories),
        "seasons": seasons,
        "current_season": {"season": current_season, "by_gameweek": current},
        "method": "total points interpolated against overall rank (linear in log rank) across the sampled "
                  "entries' FPL entry/{id}/history; net of hits",
    }


def fetch_histories(entry_ids: list[int], *, fetch=fetch_entry_history, pause: float = REQUEST_PAUSE_SECONDS) -> list[dict]:
    """Each entry's history, skipping ones FPL no longer serves. Paced between requests."""
    out = []
    for i, entry_id in enumerate(entry_ids):
        if i and pause:
            time.sleep(pause)
        history = fetch(entry_id)
        if history is not None:
            out.append(history)
    return out
