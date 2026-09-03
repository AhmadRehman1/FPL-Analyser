"""forward_plan -- reshapes a model_choice forward_season_sim walk into the app's display
payload (per-week team / transfers / captain / projected points + the Wildcard squad).

The pure reshaping helpers are unit-tested without a DB; one integration test runs a short
real walk over test_backtest's synthetic league (same fixture forward_season_sim itself uses)
to prove the end-to-end shape.
"""

from __future__ import annotations

import json

import pytest

from fpl_quant import forward_plan as fp
from tests.test_backtest import _seed_season_simulation_league
from tests.test_forward_season_sim import _bootstrap_squad


# ------------------------------------------------------------------ pure helpers

def test_week_summary_covers_every_action():
    assert fp._week_summary("wildcard", []) == "Wildcard — full rebuild"
    assert fp._week_summary("free_hit", []) == "Free Hit — one-week squad"
    assert fp._week_summary("bench_boost", []) == "Bench Boost"
    assert fp._week_summary("triple_captain", []) == "Triple Captain"
    assert fp._week_summary("hold", []) == "Hold — no transfer"
    assert fp._week_summary("transfer", [{"out": "Salah", "in": "Saka", "net": 2.1}]) == "Salah → Saka"
    two = [{"out": "A", "in": "B", "net": 1.0}, {"out": "C", "in": "D", "net": 0.5}]
    assert "A → B" in fp._week_summary("transfer", two) and "C → D" in fp._week_summary("transfer", two)
    # a transfer action with no parsed rows still reads sensibly
    assert fp._week_summary("transfer", []) == "Hold — no transfer"


def test_squad_rows_are_preview_squad_shaped():
    name_by_uid = {"u1": "Alice", "u2": "Bob", "u3": "Cara"}
    club_by_uid = {"u1": "Arsenal", "u2": None, "u3": "Chelsea"}
    rows = fp._squad_rows(["u1", "u2", "u3"], ["u1", "u2"], "u1", name_by_uid, club_by_uid)
    assert rows[0] == {"player_name": "Alice", "club": "Arsenal", "in_xi": True, "is_captain": True, "is_vice": False}
    # vice = first XI non-captain
    assert rows[1] == {"player_name": "Bob", "club": None, "in_xi": True, "is_captain": False, "is_vice": True}
    assert rows[2]["in_xi"] is False
    assert set(rows[0]) == {"player_name", "club", "in_xi", "is_captain", "is_vice"}


def test_transfer_rows_resolve_names_and_keep_net():
    rows = fp._transfer_rows([{"out_uid": "u1", "in_uid": "u9", "net": 3.14}], {"u1": "Alice"})
    assert rows == [{"out": "Alice", "in": "u9", "net": 3.14}]  # unresolved uid falls through


# ------------------------------------------------------------------ integration

@pytest.fixture
def league(con):
    _seed_season_simulation_league(con)
    return con


def test_build_forward_plan_end_to_end_shape(league):
    con = league
    plan = fp.build_forward_plan(
        con,
        entity_key="model_team",
        entry_label="Test Squad",
        entry_id=None,
        target_season="2025-2026",
        bootstrap_squad=_bootstrap_squad(con),
        active_versions={},
        start_gameweek=2,
        end_gameweek=4,
    )

    assert plan["entity_key"] == "model_team"
    assert plan["label"] == "Test Squad"
    assert plan["base_gameweek"] == 1
    assert [w["gameweek"] for w in plan["weeks"]] == [2, 3, 4]
    assert plan["total_projected_points"] == pytest.approx(
        sum(w["projected_points"] for w in plan["weeks"]), abs=0.3
    )

    for w in plan["weeks"]:
        assert w["action"] in ("hold", "transfer", "wildcard", "free_hit", "bench_boost", "triple_captain")
        assert isinstance(w["summary"], str) and w["summary"]
        assert len(w["squad"]) == 15
        assert all(set(p) == {"player_name", "club", "in_xi", "is_captain", "is_vice"} for p in w["squad"])
        assert sum(1 for p in w["squad"] if p["in_xi"]) == 11
        assert sum(1 for p in w["squad"] if p["is_captain"]) == 1
        # names resolved to canonical_name, never a raw uid (synthetic league: name == uid, so
        # just assert every squad member appears in dim_player)
    all_names = {p["player_name"] for w in plan["weeks"] for p in w["squad"]}
    known = {r[0] for r in con.execute("SELECT canonical_name FROM dim_player").fetchall()}
    assert all_names <= known

    # wildcard block present iff the model played it in-window; if not, held-until may be set
    assert "wildcard" in plan and "wildcard_held_until" in plan
    if plan["wildcard"] is not None:
        assert len(plan["wildcard"]["squad"]) == 15
        assert plan["wildcard"]["gameweek"] in (2, 3, 4)

    json.dumps(plan)  # serialisable


def test_forward_plan_horizon_default_is_gw18():
    assert fp.HORIZON_END_GAMEWEEK == 18
