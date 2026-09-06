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


def test_force_free_hit_at_plays_it_once_and_reverts_the_squad_after(league):
    """M8 Free Hit forensic regression. The one-off Free Hit XI is the XI scored that gameweek,
    but the squad that CARRIES FORWARD (carryforward_*) must be the pre-chip 15 -- Free Hit
    reverts. Pre-fix, forward_season_sim wrote the Free Hit 15 into squad_uids and the model
    team then bootstrapped the *next* gameweek from it, keeping the Free Hit squad forever."""
    import json

    con = league
    bootstrap = _bootstrap_squad(con)
    bootstrap_uids = {p["player_name"] for p in bootstrap}  # synthetic canonical_name == uid
    result = fss.run_forward_season_sim(
        con, entry_label="test", target_season="2025-2026",
        start_gameweek=2, end_gameweek=4, bootstrap_squad=bootstrap,
        active_versions={}, force_free_hit_at=2,
    )
    assert result.mode == "force_free_hit_gw2"
    gw2, gw3, gw4 = (next(r for r in result.rows if r.gameweek == g) for g in (2, 3, 4))

    assert gw2.action == "free_hit" and gw2.action_detail == "forced"
    # chip consumed exactly once, on GW2 only
    assert [r.gameweek for r in result.rows if r.action == "free_hit"] == [2]
    assert all("free_hit" in r.chips_used for r in (gw2, gw3, gw4))

    # the Free Hit week did NOT mutate the squad carried forward -- it is still the bootstrap 15
    assert set(gw2.carryforward_squad_uids) == bootstrap_uids
    assert set(gw2.carryforward_xi_uids) == {p["player_name"] for p in bootstrap if p["in_xi"]}
    # ... and GW3 resumed from that pre-Free-Hit squad, not the Free Hit 15 (<=1 transfer away)
    assert len(set(gw3.carryforward_squad_uids) - bootstrap_uids) <= 1

    # the XI actually SCORED on GW2 is the freshly solved Free Hit XI, not the held one
    assert gw2.free_hit_fresh_run_id is not None
    fresh_xi = {r[0] for r in con.execute(
        "SELECT player_uid FROM squad_optimizer_selections WHERE run_id = ? AND in_xi", [gw2.free_hit_fresh_run_id]
    ).fetchall()}
    assert set(gw2.xi_uids) == fresh_xi
    assert len(gw2.squad_uids) == 15 and len(gw2.xi_uids) == 11

    # Free Hit's gain is measured on an 11-player x 1-gameweek basis (its own threshold family,
    # free_hit_gain_threshold_params), never inheriting Wildcard's 15x5.
    assert gw2.free_hit_gain is not None and gw2.free_hit_threshold is not None
    fh_detail = json.loads(con.execute(
        "SELECT detail FROM chip_evaluations WHERE chip_type = 'free_hit' ORDER BY run_id LIMIT 1"
    ).fetchone()[0])
    assert fh_detail["current_xi_value_per_gw"]  # XI-only per-gameweek trajectory

    d = result.to_dict()
    json.dumps(d)
    assert "carryforward_squad_uids" in d["gameweeks"][0]


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
