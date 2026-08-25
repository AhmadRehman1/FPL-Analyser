import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_transfer_planner_for_real_squad import _build_chip_preview_squad, _order_chip_evaluations  # noqa: E402


def test_order_chip_evaluations_matches_chip_priority_even_when_db_order_disagrees():
    # This is the exact real-world shape that motivated the fix: chip_evaluations rows come
    # back wildcard/free_hit/triple_captain/bench_boost (insertion order in transfer_planner.py),
    # the reverse of backtest.py's own CHIP_PRIORITY (bench_boost before triple_captain).
    chips_out = [
        {"chip_type": "wildcard", "recommended": False, "score": 1.0},
        {"chip_type": "free_hit", "recommended": False, "score": 2.0},
        {"chip_type": "triple_captain", "recommended": True, "score": 3.7},
        {"chip_type": "bench_boost", "recommended": True, "score": 9.0},
    ]
    ordered = _order_chip_evaluations(chips_out)
    assert [c["chip_type"] for c in ordered] == ["wildcard", "free_hit", "bench_boost", "triple_captain"]
    # the dashboard's own chips.find(c => c.recommended) now correctly lands on bench_boost,
    # not triple_captain, when both clear their threshold in the same week.
    assert next(c for c in ordered if c["recommended"])["chip_type"] == "bench_boost"


def test_order_chip_evaluations_preserves_all_rows():
    chips_out = [
        {"chip_type": "bench_boost", "recommended": False, "score": None},
        {"chip_type": "wildcard", "recommended": False, "score": None},
    ]
    ordered = _order_chip_evaluations(chips_out)
    assert {c["chip_type"] for c in ordered} == {"bench_boost", "wildcard"}
    assert len(ordered) == 2


def test_order_chip_evaluations_unknown_chip_type_sorts_last():
    chips_out = [
        {"chip_type": "some_future_chip", "recommended": False, "score": None},
        {"chip_type": "wildcard", "recommended": False, "score": None},
    ]
    ordered = _order_chip_evaluations(chips_out)
    assert [c["chip_type"] for c in ordered] == ["wildcard", "some_future_chip"]


def test_build_chip_preview_squad_resolves_names_and_clubs():
    # Real shape: transfer_planner.read_fresh_chip_squad()'s own return rows.
    preview_rows = [
        {"player_uid": "uid-1", "in_xi": True, "is_captain": True, "is_vice": False},
        {"player_uid": "uid-2", "in_xi": False, "is_captain": False, "is_vice": False},
    ]
    name_by_uid = {"uid-1": "Erling Haaland", "uid-2": "Bernd Leno"}
    team_by_player = {"uid-1": "team-mci", "uid-2": "team-ful"}
    team_names = {"team-mci": "Man City", "team-ful": "Fulham"}

    out = _build_chip_preview_squad(preview_rows, name_by_uid, team_by_player, team_names)

    assert out == [
        {"player_name": "Erling Haaland", "club": "Man City", "in_xi": True, "is_captain": True, "is_vice": False},
        {"player_name": "Bernd Leno", "club": "Fulham", "in_xi": False, "is_captain": False, "is_vice": False},
    ]


def test_build_chip_preview_squad_falls_back_to_uid_when_name_unresolved():
    # A player_uid missing from dim_player (shouldn't happen against real data, but the
    # dashboard should still show all 15/11 names rather than silently dropping one) falls
    # back to the raw uid rather than crashing the whole export.
    preview_rows = [{"player_uid": "uid-missing", "in_xi": True, "is_captain": False, "is_vice": True}]
    out = _build_chip_preview_squad(preview_rows, {}, {}, {})
    assert out == [{"player_name": "uid-missing", "club": None, "in_xi": True, "is_captain": False, "is_vice": True}]


def test_build_chip_preview_squad_empty_input():
    assert _build_chip_preview_squad([], {}, {}, {}) == []
