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


def _seed_points_and_minutes(con, season, gw, rows):
    """rows: {player_uid: (points, minutes)} -- needed to exercise the vice-captain
    armband-transfer fallback, which keys off minutes == 0, not points."""
    for u, (p, mins) in rows.items():
        con.execute(
            "INSERT INTO fact_player_season_stats (player_uid, season, gw, event_points, minutes, _ingested_at) "
            "VALUES (?, ?, ?, ?, ?, current_timestamp)", [u, season, gw, p, mins],
        )


def _ledger_row(gw, xi, cap, *, realized=None, action="hold", simulated=False, projected=50.0, vice=None):
    squad = xi + [f"player_bench{i}" for i in range(4)]
    return {
        "gameweek": gw, "entry_label": "FPL Quant Model Team", "simulated": simulated,
        "action": action, "action_detail": "", "projected_points": projected,
        "realized_points": realized, "squad_uids": sorted(squad), "xi_uids": sorted(xi), "captain_uid": cap,
        "vice_captain_uid": vice,
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


def test_realize_transfers_the_armband_to_the_vice_when_captain_blanks(con, tmp_path):
    """Real FPL rule: captain records 0 minutes -> the multiplier moves to the vice-captain.
    Closes the gap model_team._squad_from_ledger_row()'s own comment used to disclose
    ("never silently wrong: _realized_xi_points just doubles the captain's real points")."""
    xi = [f"player_v{i}" for i in range(11)]
    _seed_players(con, xi + [f"player_bench{i}" for i in range(4)])
    _write_state(tmp_path, [_ledger_row(6, xi, "player_v0", realized=None, vice="player_v1")], gw=6)
    points = {u: 2 for u in xi}
    del points["player_v0"]
    _seed_points_and_minutes(con, "2026-2027", 6, {
        **{u: (p, 90) for u, p in points.items()},
        "player_v0": (0, 0),  # captain blanked -- didn't play at all
    })
    # without the fallback: 0*2 (captain) + 10*2 (rest, incl. vice at its plain 2) = 20;
    # with it: 0 (captain, un-doubled) + 9*2 (rest) + 2*2 (vice doubled) = 22
    model_team.realize(con, tmp_path)
    assert model_team.load_state(tmp_path)["ledger"][0]["realized_points"] == 22.0


def test_realize_keeps_old_behavior_when_ledger_row_has_no_vice_captain_uid(con, tmp_path):
    """A legacy ledger row written before this field existed (row.get() returns None): the
    captain's own (zero) points still get doubled, exactly the old behavior."""
    xi = [f"player_w{i}" for i in range(11)]
    _seed_players(con, xi + [f"player_bench{i}" for i in range(4)])
    row = _ledger_row(7, xi, "player_w0", realized=None)
    del row["vice_captain_uid"]
    _write_state(tmp_path, [row], gw=7)
    points = {u: 2 for u in xi}
    del points["player_w0"]
    _seed_points_and_minutes(con, "2026-2027", 7, {
        **{u: (p, 90) for u, p in points.items()},
        "player_w0": (0, 0),
    })
    model_team.realize(con, tmp_path)
    assert model_team.load_state(tmp_path)["ledger"][0]["realized_points"] == 20.0


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


# ------------------------------------------------------------------ Free Hit semantics
#
# Regression coverage for the GW3 2026-27 model-team Free Hit forensic (M8): the one-off Free
# Hit XI is the XI scored that gameweek, but the squad that carries forward -- and that the
# public panel shows as "current" -- must be the pre-chip long-term 15, never the Free Hit 15.
# forward_season_sim now records the real persisted holdings every gameweek in `carryforward_*`;
# model_team must consume those, and self-heal a legacy Free Hit row (written before the fix)
# that stored the Free Hit 15 in `squad_uids`.

_LONG_TERM_XI = [f"player_lt{i}" for i in range(11)]
_LONG_TERM_SQUAD = sorted(_LONG_TERM_XI + [f"player_bench{i}" for i in range(4)])  # matches _ledger_row()
_FH_XI = [f"player_fh{i}" for i in range(11)]
_FH_SQUAD = sorted(_FH_XI + [f"player_fhbench{i}" for i in range(4)])


def _name(uid):
    return uid.replace("player_", "").title()


def _fh_ledger_row(gw, *, with_carryforward, realized=None, gain=22.12, threshold=1.5):
    row = {
        "gameweek": gw, "entry_label": "FPL Quant Model Team", "simulated": False,
        "action": "free_hit", "action_detail": "model chose free hit", "projected_points": 78.23,
        "realized_points": realized,
        "squad_uids": _FH_SQUAD, "xi_uids": sorted(_FH_XI), "captain_uid": "player_fh0",
        "formation_xi_uids": sorted(_FH_XI),
        "chips_used": ["free_hit"], "wildcard_gain": None, "wildcard_recommended": False,
        "free_hit_gain": gain, "free_hit_recommended": True, "free_hit_threshold": threshold,
        "current_squad_horizon_value": None, "band_low": 61.0, "band_high": 94.0,
    }
    if with_carryforward:
        row["carryforward_squad_uids"] = _LONG_TERM_SQUAD
        row["carryforward_xi_uids"] = sorted(_LONG_TERM_XI)
        row["carryforward_captain_uid"] = "player_lt0"
    return row


def test_free_hit_week_current_squad_is_the_reverted_long_term_squad(con, tmp_path):
    _seed_players(con, _LONG_TERM_SQUAD + _FH_SQUAD)
    gw2 = _ledger_row(2, _LONG_TERM_XI, "player_lt0", realized=50.0)
    gw3 = _fh_ledger_row(3, with_carryforward=True, realized=44.0)
    _write_state(tmp_path, [gw2, gw3], gw=3, chips_used_set1=["free_hit"])

    summary = model_team.build_summary(con, tmp_path, {2: 48.0, 3: 40.0})
    current = {p["name"] for p in summary["current_squad"]}
    # the panel's "current squad" after a Free Hit week is the reverted long-term 15,
    # NOT the one-week Free Hit XI
    assert current == {_name(u) for u in _LONG_TERM_SQUAD}
    assert not (current & {_name(u) for u in _FH_SQUAD})
    captain = next(p["name"] for p in summary["current_squad"] if p["is_captain"])
    assert captain == _name("player_lt0")


def test_legacy_free_hit_row_without_carryforward_reverts_via_prior_week(con, tmp_path):
    _seed_players(con, _LONG_TERM_SQUAD + _FH_SQUAD)
    gw2 = _ledger_row(2, _LONG_TERM_XI, "player_lt0", realized=50.0)
    gw3 = _fh_ledger_row(3, with_carryforward=False, realized=13.0)  # the real committed shape
    _write_state(tmp_path, [gw2, gw3], gw=3, chips_used_set1=["free_hit"])

    summary = model_team.build_summary(con, tmp_path, {2: 48.0})
    current = {p["name"] for p in summary["current_squad"]}
    assert current == {_name(u) for u in _LONG_TERM_SQUAD}


def test_carryforward_row_skips_a_legacy_free_hit_row():
    gw2 = _ledger_row(2, _LONG_TERM_XI, "player_lt0", realized=50.0)
    gw3 = _fh_ledger_row(3, with_carryforward=False)
    assert model_team._carryforward_row([gw2, gw3]) is gw2


def test_carryforward_row_uses_a_free_hit_row_that_has_carryforward_fields():
    gw2 = _ledger_row(2, _LONG_TERM_XI, "player_lt0", realized=50.0)
    gw3 = _fh_ledger_row(3, with_carryforward=True)
    assert model_team._carryforward_row([gw2, gw3]) is gw3


def test_squad_from_ledger_row_prefers_carryforward_fields(con):
    _seed_players(con, _LONG_TERM_SQUAD + _FH_SQUAD)
    squad = model_team._squad_from_ledger_row(con, _fh_ledger_row(3, with_carryforward=True))
    assert {p["player_name"] for p in squad} == {_name(u) for u in _LONG_TERM_SQUAD}
    assert next(p["player_name"] for p in squad if p["is_captain"]) == _name("player_lt0")
    assert {p["player_name"] for p in squad if p["in_xi"]} == {_name(u) for u in _LONG_TERM_XI}


def test_free_hit_audit_reports_gain_threshold_and_counterfactual_hold(con, tmp_path):
    _seed_players(con, _LONG_TERM_SQUAD + _FH_SQUAD)
    # GW3 points: every Free Hit XI player scores 4, every long-term XI player scores 2.
    _seed_points(con, "2026-2027", 3, {**{u: 4 for u in _FH_XI}, **{u: 2 for u in _LONG_TERM_XI}})
    gw2 = _ledger_row(2, _LONG_TERM_XI, "player_lt0", realized=50.0)
    gw3 = _fh_ledger_row(3, with_carryforward=True, realized=48.0)
    _write_state(tmp_path, [gw2, gw3], gw=3, chips_used_set1=["free_hit"])

    audit = model_team.build_summary(con, tmp_path, {2: 48.0})["free_hit_audit"]
    assert audit["gameweek"] == 3
    assert audit["projected_gain_vs_current_xi"] == 22.12
    assert audit["gain_threshold"] == 1.5
    assert audit["projected_points"] == 78.23
    assert audit["realized_points"] == 48.0
    # counterfactual: hold the long-term XI -> 10*2 + captain lt0 doubled (2*2) = 24
    assert audit["counterfactual_hold_realized_points"] == 24.0
    assert audit["realized_minus_hold"] == 24.0
    assert audit["gameweek_final"] is False  # GW3 has no field average in the map


def test_build_summary_keeps_an_unfinished_gameweek_out_of_the_headline(con, tmp_path):
    xi = [f"player_p{i}" for i in range(11)]
    _seed_players(con, xi + [f"player_bench{i}" for i in range(4)])
    ledger = [
        _ledger_row(1, xi, "player_p0", realized=50.0, simulated=True),
        _ledger_row(2, xi, "player_p0", realized=40.0, simulated=True),
        _ledger_row(3, xi, "player_p0", realized=13.0),  # scored off a still-in-progress GW3
    ]
    _write_state(tmp_path, ledger, gw=3)
    summary = model_team.build_summary(con, tmp_path, {1: 48.0, 2: 45.0})  # no GW3 field average

    gw3 = next(w for w in summary["weeks"] if w["gameweek"] == 3)
    assert gw3["provisional"] is True
    assert gw3["realized_points"] == 13.0
    assert gw3["cumulative_points"] is None and gw3["cumulative_vs_field"] is None
    # the half-played gameweek does not move the headline
    assert summary["total_realized_points"] == 90.0
    assert summary["total_vs_field"] == 90.0 - 93.0
    assert summary["n_gameweeks_scored"] == 2


def test_realize_does_not_freeze_a_partial_score_for_an_unfinished_gameweek(con, tmp_path):
    xi = [f"player_q{i}" for i in range(11)]
    _seed_players(con, xi + [f"player_bench{i}" for i in range(4)])
    _write_state(tmp_path, [_ledger_row(3, xi, "player_q0", realized=None)], gw=3)

    # only 3 of 11 have played so far -> provisional
    _seed_points(con, "2026-2027", 3, {xi[0]: 5, xi[1]: 2, xi[2]: 6})
    out = model_team.realize(con, tmp_path, finished_gameweeks=set())
    assert out == {"realized": 1}
    row = model_team.load_state(tmp_path)["ledger"][0]
    assert row["realized_points"] == 13.0 + 5.0  # q0 captain doubled: (5*2)+2+6 = 18
    assert row["realized_final"] is False

    # the rest come in, gameweek now finished -> re-scored and locked
    _seed_points(con, "2026-2027", 3, {u: 3 for u in xi[3:]})
    out = model_team.realize(con, tmp_path, finished_gameweeks={3})
    assert out == {"realized": 1}
    row = model_team.load_state(tmp_path)["ledger"][0]
    assert row["realized_points"] == (5 * 2) + 2 + 6 + 3 * 8  # 42.0
    assert row["realized_final"] is True

    # locked: a later run does not touch it
    assert model_team.realize(con, tmp_path, finished_gameweeks={3}) == {"realized": 0}
