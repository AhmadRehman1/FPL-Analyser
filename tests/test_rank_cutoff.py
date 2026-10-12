"""fpl_quant.rank_cutoff and scripts/derive_rank_cutoff.py: the real top-100k cutoff from the
sampled entries' FPL history (docs/plans/2026-10_beat_the_template.md, Req 4)."""

import json
import math
import sys
from pathlib import Path

import pytest

from fpl_quant import rank_cutoff as rc

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import derive_rank_cutoff as drc  # noqa: E402


def _history(past, current):
    """Canned entry/{id}/history: past = [(season_name, total, rank)], current = [(event, total, overall_rank)]."""
    return {
        "past": [{"season_name": s, "total_points": t, "rank": r} for s, t, r in past],
        "current": [{"event": e, "points": 0, "total_points": t, "rank": 0, "overall_rank": r} for e, t, r in current],
        "chips": [],
    }


HISTORIES = [
    _history([("2024/25", 2500, 50_000), ("2025/26", 2440, 90_000)], [(1, 80, 120_000), (2, 150, 95_000)]),
    _history([("2024/25", 2400, 200_000), ("2025/26", 2400, 160_000)], [(1, 90, 60_000), (2, 140, 130_000)]),
    _history([("2025/26", 2600, 4_000)], [(1, 50, 2_000_000), (2, 110, 1_500_000)]),  # no 2024-25 season
]


def test_season_name():
    assert rc.season_name("2024-2025") == "2024/25"
    assert rc.season_name("2026-2027") == "2026/27"


def test_interpolates_in_log_rank_between_the_bracketing_entries():
    out = rc.interpolate_at_rank([(50_000, 2500), (200_000, 2400), (5_000, 2700)])
    w = (math.log(100_000) - math.log(50_000)) / (math.log(200_000) - math.log(50_000))  # 0.5
    assert out["total_points"] == pytest.approx(round(2500 - 100 * w, 1))
    assert out["bracket_ranks"] == [50_000, 200_000] and out["method"] == "interpolated"
    assert (out["n_entries"], out["n_near_better"], out["n_near_worse"]) == (3, 1, 1)


def test_nearest_entry_stands_in_when_nothing_brackets_the_target():
    out = rc.interpolate_at_rank([(150_000, 2380), (400_000, 2300)])
    assert out["method"] == "nearest" and out["total_points"] == 2380 and out["bracket_ranks"] == [150_000]
    assert rc.interpolate_at_rank([]) is None
    assert rc.interpolate_at_rank([(None, 10), (0, 20)]) is None


def test_derive_cutoffs_by_season_and_by_gameweek():
    out = rc.derive_cutoffs(HISTORIES, ["2024-2025", "2025-2026"], "2026-2027")
    s24 = out["seasons"]["2024-2025"]
    assert s24["n_entries"] == 2 and s24["method"] == "interpolated"
    assert s24["pace_per_gw"] == pytest.approx(s24["total_points"] / 38, abs=1e-3)
    s25 = out["seasons"]["2025-2026"]
    assert s25["bracket_ranks"] == [90_000, 160_000]
    gws = out["current_season"]["by_gameweek"]
    assert [g["gameweek"] for g in gws] == [1, 2]
    assert gws[1]["bracket_ranks"] == [95_000, 130_000]
    assert 140 <= gws[1]["total_points"] <= 150
    assert out["n_histories"] == 3 and out["target_rank"] == 100_000


def test_fetch_histories_skips_entries_fpl_no_longer_serves():
    served = {1: HISTORIES[0], 3: HISTORIES[2]}
    assert rc.fetch_histories([1, 2, 3], fetch=served.get, pause=0) == [HISTORIES[0], HISTORIES[2]]


def test_script_adds_only_the_cutoff_key_to_the_autopsy(con, tmp_path):
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p', 'P', 'Forward')")
    for entry_id in (11, 12, 13):
        con.execute(
            "INSERT INTO fact_rival_squad_sample (entry_id, season, event, player_uid, is_captain, multiplier, "
            "league_rank, _ingested_at) VALUES (?, '2026-2027', 1, 'p', FALSE, 1, 1, current_timestamp)", [entry_id],
        )
    path = tmp_path / "rank_autopsy.json"
    path.write_text(json.dumps({"season": "2026-2027", "subjects": {"model": 1}}))
    entry_ids = drc.sampled_entry_ids(con)
    assert entry_ids == [11, 12, 13]
    canned = dict(zip(entry_ids, HISTORIES))
    assert drc.main(fetch=canned.get, path=path, entry_ids=entry_ids) == 0
    payload = json.loads(path.read_text())
    assert payload["season"] == "2026-2027" and payload["subjects"] == {"model": 1}
    cut = payload["top_100k_cutoff"]
    assert set(cut["seasons"]) == {"2024-2025", "2025-2026"}
    assert cut["current_season"]["season"] == "2026-2027"
    # no per-entry data is persisted
    assert "past" not in json.dumps(cut) and "entry" not in json.dumps(cut["seasons"])
