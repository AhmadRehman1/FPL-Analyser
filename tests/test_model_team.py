"""model_team -- the model-managed team track record. The weekly `advance()` walk is a thin
wrapper over `forward_season_sim.run_forward_season_sim()` (integration-tested there); these
tests cover the state ledger, realised re-scoring, and the public summary maths.
"""

from __future__ import annotations

import json

from fpl_quant import model_team


def _seed_players(con, uids):
    for u in uids:
        con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, 'Midfielder')",
                    [u, u.replace("player_", "").title()])


def _seed_points(con, season, gw, points_by_uid):
    for u, p in points_by_uid.items():
        con.execute("INSERT INTO fact_player_season_stats (player_uid, season, gw, event_points, _ingested_at) "
                    "VALUES (?, ?, ?, ?, current_timestamp)", [u, season, gw, p])


def _ledger_row(gw, xi, cap, *, realized=None, action="hold", simulated=False, projected=50.0):
    squad = xi + [f"player_bench{i}" for i in range(4)]
    return {
        "gameweek": gw, "entry_label": "FPL Quant Model Team", "simulated": simulated,
        "action": action, "action_detail": "", "projected_points": projected,
        "realized_points": realized, "squad_uids": sorted(squad), "xi_uids": sorted(xi), "captain_uid": cap,
        "chips_used": [], "wildcard_gain": None, "wildcard_recommended": False, "free_hit_gain": None,
        "free_hit_recommended": False, "current_squad_horizon_value": None, "band_low": 40.0, "band_high": 60.0,
    }


def _write_state(state_dir, ledger, gw, **extra):
    (state_dir / "state.json").write_text(json.dumps({
        "season": "2026-2027", "current_gameweek": gw, "ledger": ledger,
        "chips_used_set1": [], "chips_used_set2": [], **extra,
    }))


# ------------------------------------------------------------------ state I/O

def test_load_state_none_when_missing(tmp_path):
    assert model_team.load_state(tmp_path) is None


def test_save_then_load_roundtrips(tmp_path):
    state = {"season": "2026-2027", "current_gameweek": 3, "ledger": [], "chips_used_set1": [], "chips_used_set2": []}
    model_team.save_state(tmp_path, state)
    assert model_team.load_state(tmp_path) == state


# ------------------------------------------------------------------ realize()

def test_realize_scores_a_now_ingested_gameweek(con, tmp_path):
    xi = [f"player_x{i}" for i in range(11)]
    _seed_players(con, xi + [f"player_bench{i}" for i in range(4)])
    _write_state(tmp_path, [_ledger_row(2, xi, "player_x0", realized=None)], gw=2)
    # GW2 now has ingested points: 3 each, captain doubled -> 10*3 + 1*(3*2) = 36
    _seed_points(con, "2026-2027", 2, {u: 3 for u in xi})

    out = model_team.realize(con, tmp_path)
    assert out == {"realized": 1}
    state = model_team.load_state(tmp_path)
    assert state["ledger"][0]["realized_points"] == 36.0

    # idempotent -- a second call re-scores nothing
    assert model_team.realize(con, tmp_path) == {"realized": 0}


def test_realize_skips_a_gameweek_with_no_ingested_points(con, tmp_path):
    xi = [f"player_y{i}" for i in range(11)]
    _seed_players(con, xi)
    _write_state(tmp_path, [_ledger_row(5, xi, "player_y0", realized=None)], gw=5)
    assert model_team.realize(con, tmp_path) == {"realized": 0}


def test_realize_triple_captain_uses_a_3x_multiplier(con, tmp_path):
    xi = [f"player_z{i}" for i in range(11)]
    _seed_players(con, xi + [f"player_bench{i}" for i in range(4)])
    _write_state(tmp_path, [_ledger_row(4, xi, "player_z0", realized=None, action="triple_captain")], gw=4)
    _seed_points(con, "2026-2027", 4, {u: 2 for u in xi})  # 10*2 + 1*(2*3) = 26
    model_team.realize(con, tmp_path)
    assert model_team.load_state(tmp_path)["ledger"][0]["realized_points"] == 26.0


# ------------------------------------------------------------------ build_summary()

def test_build_summary_not_ready_before_seeding(con, tmp_path):
    assert model_team.build_summary(con, tmp_path, {})["ready"] is False


def test_build_summary_cumulative_and_vs_field(con, tmp_path):
    xi = [f"player_s{i}" for i in range(11)]
    _seed_players(con, xi + [f"player_bench{i}" for i in range(4)])
    ledger = [
        _ledger_row(1, xi, "player_s0", realized=55.0, simulated=True),
        _ledger_row(2, xi, "player_s0", realized=48.0, simulated=True),
        _ledger_row(3, xi, "player_s0", realized=None, action="hold"),  # not played yet
    ]
    _write_state(tmp_path, ledger, gw=3)

    summary = model_team.build_summary(con, tmp_path, {1: 50.0, 2: 52.0})
    assert summary["ready"] is True
    assert summary["n_gameweeks_scored"] == 2
    assert summary["n_gameweeks_simulated"] == 2
    assert summary["total_realized_points"] == 103.0
    assert summary["total_vs_field"] == 103.0 - 102.0  # +1.0
    gw2 = next(w for w in summary["weeks"] if w["gameweek"] == 2)
    assert gw2["delta_vs_field"] == -4.0
    assert gw2["cumulative_points"] == 103.0
    gw3 = next(w for w in summary["weeks"] if w["gameweek"] == 3)
    assert gw3["realized_points"] is None and gw3["cumulative_points"] is None
    assert summary["next_decision"]["gameweek"] == 3
    assert len(summary["current_squad"]) == 15
