"""forward_season_sim -- a forward-looking, real-squad-seeded, projected-EP season walk.

Reuses test_backtest's synthetic league fixture (the same one run_season_simulation is tested
against) so this exercises the real transfer_planner.run() / apply_recommendation() path every
gameweek, not a stubbed one.
"""

from __future__ import annotations

import pytest

from fpl_quant import forward_season_sim as fss
from tests.test_backtest import _seed_season_simulation_league


def _bootstrap_squad(con) -> list[dict]:
    """A legal 15 from the synthetic league's 18 players (2 GK / 6 DEF / 6 MID / 4 FWD ->
    take 2/5/5/3), XI = first 11, captain = a nailed forward."""
    rows = con.execute(
        "SELECT player_uid, position FROM dim_player ORDER BY position, player_uid"
    ).fetchall()
    by_pos: dict[str, list[str]] = {}
    for uid, pos in rows:
        by_pos.setdefault(pos, []).append(uid)
    picks = (
        by_pos["Goalkeeper"][:2] + by_pos["Defender"][:5]
        + by_pos["Midfielder"][:5] + by_pos["Forward"][:3]
    )
    squad = []
    for i, uid in enumerate(picks):
        squad.append({
            "player_name": uid,  # synthetic canonical_name == uid
            "in_xi": i < 11,
            "is_captain": uid == by_pos["Forward"][0],
            "is_vice": uid == by_pos["Midfielder"][0],
        })
    return squad


@pytest.fixture
def league(con):
    _seed_season_simulation_league(con)
    return con


def test_forward_sim_walks_forward_and_scores_on_projected_ep(league):
    con = league
    result = fss.run_forward_season_sim(
        con, entry_label="test", target_season="2025-2026",
        start_gameweek=2, end_gameweek=4, bootstrap_squad=_bootstrap_squad(con),
        active_versions={},
    )

    assert result.mode == "model_choice"
    assert [r.gameweek for r in result.rows] == [2, 3, 4]
    assert all(r.projected_points >= 0.0 for r in result.rows)
    assert all(r.band_low <= r.projected_points <= r.band_high for r in result.rows)
    assert result.total_projected_points == pytest.approx(sum(r.projected_points for r in result.rows))
    # evaluate_wildcard ran every gameweek
    assert all(r.wildcard_gain is not None for r in result.rows)
    # squad genuinely evolved -- final holdings are a legal 15
    # (run_forward_season_sim leaves the last state_version reachable via the planner tables)


def test_force_wildcard_at_plays_the_wildcard_exactly_there(league):
    con = league
    result = fss.run_forward_season_sim(
        con, entry_label="test", target_season="2025-2026",
        start_gameweek=2, end_gameweek=4, bootstrap_squad=_bootstrap_squad(con),
        active_versions={}, force_wildcard_at=3,
    )
    assert result.mode == "force_wildcard_gw3"
    gw3 = next(r for r in result.rows if r.gameweek == 3)
    assert gw3.action == "wildcard"
    assert "wildcard" in gw3.chips_used
    # not played on any other gameweek
    assert all(r.action != "wildcard" for r in result.rows if r.gameweek != 3)


def test_hold_wildcard_never_plays_it(league):
    con = league
    result = fss.run_forward_season_sim(
        con, entry_label="test", target_season="2025-2026",
        start_gameweek=2, end_gameweek=4, bootstrap_squad=_bootstrap_squad(con),
        active_versions={}, hold_wildcard=True,
    )
    assert result.mode == "hold_wildcard"
    assert all(r.action != "wildcard" for r in result.rows)
    # the gain trajectory is still recorded for the timing read
    assert all(r.wildcard_gain is not None for r in result.rows)


def test_wildcard_recommendation_picks_the_max_gain_recommended_gameweek(league):
    con = league
    result = fss.run_forward_season_sim(
        con, entry_label="test", target_season="2025-2026",
        start_gameweek=2, end_gameweek=4, bootstrap_squad=_bootstrap_squad(con),
        active_versions={}, hold_wildcard=True,
    )
    reco = result.wildcard_recommendation
    recommended_rows = [r for r in result.rows if r.wildcard_recommended]
    if recommended_rows:
        assert reco is not None
        assert reco["gameweek"] == max(recommended_rows, key=lambda r: r.wildcard_gain).gameweek
    else:
        assert reco is None


def test_to_dict_is_json_shaped(league):
    con = league
    result = fss.run_forward_season_sim(
        con, entry_label="acct", target_season="2025-2026",
        start_gameweek=2, end_gameweek=3, bootstrap_squad=_bootstrap_squad(con),
        active_versions={},
    )
    d = result.to_dict()
    assert d["entry_label"] == "acct"
    assert d["gameweeks"] and set(d["gameweeks"][0]) >= {
        "gameweek", "projected_points", "band_low", "band_high", "action", "wildcard_gain",
    }
    import json
    json.dumps(d)  # must be serialisable
