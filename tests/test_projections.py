from datetime import date

import pytest

from fpl_quant import projections as proj
from fpl_quant.errors import MissingModelVersionError

PARAM_VERSIONS = dict(
    scoring_params_version=1, bps_params_version=1, tau_params_version=1,
    rho_residual_params_version=1, corr_params_version=1,
)


def _seed_projection_scenario(con, gw_players, target_season="2026-2027"):
    """gw_players: {gw: {uid: (position, club, price, ep_total, q05, q95)}}. Builds real
    ep_model_version/uncertainty_model_version rows + ep_outputs/uncertainty_outputs directly
    (NOT via ep.run()/uncertainty.run(), which need this project's full M3/M4 dependency
    chain seeded -- build_projections()'s own job is reading through an ALREADY-computed
    horizon, not re-deriving one, so tests monkeypatch transfer_planner.compute_horizon_ep()
    to return this fixture's {gw: (ep_mv, un_mv)} exactly the way test_transfer_planner.py's
    own tests already do for code downstream of it).

    Returns {gw: (ep_model_version, uncertainty_model_version)}.
    """
    # fetch_candidate_pool() reports "club" as player_alias.team_code verbatim (not
    # dim_team.team_uid) -- use the same club label as the team_code directly, so this
    # fixture's own club names are exactly what a real caller sees back.
    clubs = sorted({row[1] for rows in gw_players.values() for row in rows.values()})
    for club in clubs:
        con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES (?, ?) ON CONFLICT DO NOTHING", [club, club])

    uid_club = {uid: club for rows in gw_players.values() for uid, (_pos, club, *_rest) in rows.items()}
    for uid, club in uid_club.items():
        con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, 'Forward') ON CONFLICT DO NOTHING", [uid, uid])
        con.execute(
            "INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) "
            "VALUES (?, ?, ?, ?, ?) ON CONFLICT DO NOTHING", [uid, uid.lower(), club, target_season, uid],
        )

    con.execute(
        "INSERT INTO team_strength_model_versions (calibration_asof_date, home_advantage, xi_params_version, "
        "rho_params_version, reference_team_uid) VALUES ('2026-08-10', 0.2, 1, 1, 'team_a')"
    )
    ts_mv = con.execute("SELECT max(model_version) FROM team_strength_model_versions").fetchone()[0]
    con.execute(
        "INSERT INTO minutes_model_versions (calibration_asof_date, target_season, decay_params_version, "
        "adjustment_params_version, shrinkage_params_version, fact_multiplier_params_version, lookback_seasons) "
        "VALUES ('2026-08-10', ?, 1, 1, 1, 1, '[]')", [target_season],
    )
    mm_mv = con.execute("SELECT max(model_version) FROM minutes_model_versions").fetchone()[0]

    club_list = sorted(clubs)
    horizon_versions = {}
    for gw, players in gw_players.items():
        con.execute(
            "INSERT INTO fact_match (match_id, season, gameweek, home_team_uid, away_team_uid, finished, "
            "competition, kickoff_time, _ingested_at) VALUES (?, ?, ?, ?, ?, FALSE, 'Premier League', "
            "'2026-08-24', current_timestamp)",
            [f"m{gw}", target_season, gw, club_list[0], club_list[-1] if len(club_list) > 1 else club_list[0]],
        )
        for uid, (_pos, _club, price, *_rest) in players.items():
            con.execute(
                "INSERT INTO fact_player_season_stats (player_uid, season, gw, now_cost, _ingested_at) "
                "VALUES (?, ?, 1, ?, current_timestamp) ON CONFLICT DO NOTHING", [uid, target_season, price],
            )
        ep_mv = con.execute(
            "INSERT INTO ep_model_versions (calibration_asof_date, target_season, team_strength_model_version, "
            "minutes_model_version, scoring_matrix_params_version, bps_params_version, bps_tau_params_version) "
            "VALUES ('2026-08-10', ?, ?, ?, 1, 1, 1) RETURNING model_version", [target_season, ts_mv, mm_mv],
        ).fetchone()[0]
        un_mv = con.execute(
            "INSERT INTO uncertainty_model_versions (calibration_asof_date, ep_model_version, minutes_model_version, "
            "team_strength_model_version, rho_residual_params_version) VALUES ('2026-08-10', ?, ?, ?, 1) "
            "RETURNING model_version", [ep_mv, mm_mv, ts_mv],
        ).fetchone()[0]
        for uid, (position, club, price, ep_total, q05, q95) in players.items():
            con.execute(
                "INSERT INTO ep_outputs (model_version, player_uid, fixture_match_id, ep_appearance, ep_goals, "
                "ep_assists, ep_clean_sheet, ep_goals_conceded, ep_defcon, ep_bonus, ep_saves, ep_penalty_save, "
                "ep_cards, ep_own_goal, ep_total, expected_bps) VALUES (?, ?, ?, 0,0,0,0,0,0,0,0,0,0,0, ?, 5.0)",
                [ep_mv, uid, f"m{gw}", ep_total],
            )
            con.execute(
                "INSERT INTO uncertainty_outputs (model_version, player_uid, fixture_match_id, var_appearance, "
                "var_goals, var_assists, var_clean_sheet, var_goals_conceded, var_defcon, var_bonus, var_saves, "
                "var_total, skew, excess_kurtosis, quantile_05, quantile_25, quantile_75, quantile_95) "
                "VALUES (?, ?, ?, 0,0,0,0,0,0,0,0, 1.0, 0,0,?,0,0,?)",
                [un_mv, uid, f"m{gw}", q05, q95],
            )
        horizon_versions[gw] = (ep_mv, un_mv)
    return horizon_versions


def _build(con, monkeypatch, horizon_versions, gameweeks, **kwargs):
    monkeypatch.setattr(proj.tp, "compute_horizon_ep", lambda *a, **k: horizon_versions)
    return proj.build_projections(
        con, calibration_asof_date=date(2026, 8, 24), target_season="2026-2027", gameweeks=gameweeks,
        ts_model_version=1, mm_model_version=1, **PARAM_VERSIONS, **kwargs,
    )


def test_build_projections_returns_one_row_per_player_with_a_fixture(con, monkeypatch):
    gw_players = {2: {
        "haaland": ("Forward", "clubA", 12.5, 6.3, 5.9, 6.7),
        "salah": ("Forward", "clubB", 13.0, 6.1, 4.0, 8.2),
    }}
    horizon_versions = _seed_projection_scenario(con, gw_players)

    rows = _build(con, monkeypatch, horizon_versions, gameweeks=[2])
    assert {r.player_uid for r in rows} == {"haaland", "salah"}
    haaland = next(r for r in rows if r.player_uid == "haaland")
    assert haaland.name == "haaland"
    assert haaland.team == "clubA"
    assert haaland.pos == "FWD"
    assert haaland.now_cost == pytest.approx(12.5)
    assert len(haaland.ep_per_gw) == 1
    band = haaland.ep_per_gw[0]
    assert band.gw == 2 and band.ep == pytest.approx(6.3) and band.ci_low == pytest.approx(5.9) and band.ci_high == pytest.approx(6.7)
    assert haaland.provenance.data_asof == "2026-08-24"
    assert "ep_v" in haaland.provenance.model_version


def test_build_projections_spans_multiple_gameweeks(con, monkeypatch):
    gw_players = {
        2: {"haaland": ("Forward", "clubA", 12.5, 6.3, 5.9, 6.7)},
        3: {"haaland": ("Forward", "clubA", 12.5, 7.0, 6.0, 8.0)},
    }
    horizon_versions = _seed_projection_scenario(con, gw_players)

    rows = _build(con, monkeypatch, horizon_versions, gameweeks=[2, 3])
    assert len(rows) == 1
    gws = [b.gw for b in rows[0].ep_per_gw]
    assert gws == [2, 3]
    assert rows[0].ep_per_gw[1].ep == pytest.approx(7.0)


def test_build_projections_raises_when_a_requested_gameweek_has_no_fixtures(con, monkeypatch):
    gw_players = {2: {"haaland": ("Forward", "clubA", 12.5, 6.3, 5.9, 6.7)}}
    horizon_versions = _seed_projection_scenario(con, gw_players)

    with pytest.raises(MissingModelVersionError):
        _build(con, monkeypatch, horizon_versions, gameweeks=[2, 99])


def test_build_projections_is_deterministic_across_repeated_calls(con, monkeypatch):
    gw_players = {2: {
        "haaland": ("Forward", "clubA", 12.5, 6.3, 5.9, 6.7),
        "salah": ("Forward", "clubB", 13.0, 6.1, 4.0, 8.2),
        "kane": ("Forward", "clubA", 11.0, 3.0, 1.0, 5.0),
    }}
    horizon_versions = _seed_projection_scenario(con, gw_players)

    def _run():
        rows = _build(con, monkeypatch, horizon_versions, gameweeks=[2])
        return [(r.player_uid, r.ep_per_gw[0].ep, r.ep_per_gw[0].ci_low, r.ep_per_gw[0].ci_high) for r in rows]

    first, second = _run(), _run()
    assert first == second


def test_captain_ranking_higher_ep_wins_when_not_tied(con, monkeypatch):
    gw_players = {2: {
        "haaland": ("Forward", "clubA", 12.5, 6.3, 5.9, 6.7),
        "kane": ("Forward", "clubA", 11.0, 3.0, 1.0, 5.0),
    }}
    horizon_versions = _seed_projection_scenario(con, gw_players)
    rows = _build(con, monkeypatch, horizon_versions, gameweeks=[2])

    ranking = proj.build_captain_ranking(rows, gw=2)
    assert [r["player_uid"] for r in ranking] == ["haaland", "kane"]
    assert ranking[0]["rank"] == 1 and ranking[0]["vice_captain_reason"] is None
    assert ranking[1]["rank"] == 2 and ranking[1]["vice_captain_reason"] is not None


def test_captain_ranking_narrower_band_wins_a_near_tie():
    """This feature's own explicit spec example: a 6.3+-0.4 captain ranks above a 6.5+-2.1
    captain (here 6.3 vs 6.25, within tie_epsilon) -- lower variance wins ties."""
    haaland = proj.ProjectionRow(
        player_uid="haaland", name="Haaland", team="clubA", pos="FWD", now_cost=12.5,
        ep_per_gw=[proj.GWBand(gw=1, ep=6.3, ci_low=5.9, ci_high=6.7)],
        provenance=proj.Provenance(model_version="v1", data_asof="2026-08-24", calibrated_params_fraction=None),
    )
    salah = proj.ProjectionRow(
        player_uid="salah", name="Salah", team="clubB", pos="FWD", now_cost=13.0,
        ep_per_gw=[proj.GWBand(gw=1, ep=6.25, ci_low=4.0, ci_high=8.2)],  # ep within 0.15 of haaland, far wider band
        provenance=proj.Provenance(model_version="v1", data_asof="2026-08-24", calibrated_params_fraction=None),
    )
    ranking = proj.build_captain_ranking([haaland, salah], gw=1)
    assert ranking[0]["player_uid"] == "haaland"  # narrower band wins the near-tie
    assert ranking[1]["player_uid"] == "salah"
    assert "wider confidence band" in ranking[1]["vice_captain_reason"]


def test_captain_ranking_empty_when_no_fixture_at_that_gameweek():
    row = proj.ProjectionRow(
        player_uid="p1", name="P1", team="clubA", pos="FWD", now_cost=5.0,
        ep_per_gw=[proj.GWBand(gw=1, ep=3.0, ci_low=1.0, ci_high=5.0)],
        provenance=proj.Provenance(model_version="v1", data_asof="2026-08-24", calibrated_params_fraction=None),
    )
    assert proj.build_captain_ranking([row], gw=99) == []


def test_build_projections_rejects_empty_gameweek_list(con):
    with pytest.raises(ValueError):
        proj.build_projections(
            con, calibration_asof_date=date(2026, 8, 24), target_season="2026-2027", gameweeks=[],
            ts_model_version=1, mm_model_version=1, **PARAM_VERSIONS,
        )


def test_projection_rows_and_captain_ranking_serialize_to_the_documented_json_shape(con, monkeypatch):
    """Equivalent of a schema-invariant check for this feature's JSON output contract
    (data/dashboard/projections_<asof>.json) -- test_schema_invariants.py itself only covers
    DuckDB CHECK constraints, not dashboard JSON shapes, so this lives alongside the rest of
    this module's own tests instead."""
    import dataclasses
    import json

    gw_players = {2: {
        "haaland": ("Forward", "clubA", 12.5, 6.3, 5.9, 6.7),
        "salah": ("Forward", "clubB", 13.0, 6.1, 4.0, 8.2),
    }}
    horizon_versions = _seed_projection_scenario(con, gw_players)
    rows = _build(con, monkeypatch, horizon_versions, gameweeks=[2])
    captain_ranking = proj.build_captain_ranking(rows, gw=2)

    payload = {
        "data_asof": "2026-08-24", "model_version": rows[0].provenance.model_version,
        "generated_at": "2026-08-24T15:40:00+00:00",
        "captain_ranking": captain_ranking,
        "players": [dataclasses.asdict(r) for r in rows],
    }
    text = json.dumps(payload)  # must not raise -- every field is JSON-serializable
    reloaded = json.loads(text)
    assert set(reloaded) == {"data_asof", "model_version", "generated_at", "captain_ranking", "players"}
    assert len(reloaded["players"]) == 2
    player = reloaded["players"][0]
    assert set(player) == {"player_uid", "name", "team", "pos", "now_cost", "ep_per_gw", "provenance"}
    assert set(player["ep_per_gw"][0]) == {"gw", "ep", "ci_low", "ci_high"}
    assert set(player["provenance"]) == {"model_version", "data_asof", "calibrated_params_fraction"}
    assert reloaded["captain_ranking"][0]["rank"] == 1


# ============================================================
# resolve_element_ids -- player_uid -> FPL bootstrap-static element id, for the PWA planner's
# own squad-join (see scripts/export_projections.py's own docstring on why this exists).
# ============================================================

def _seed_resolvable_player(con, name, normalized, player_uid, season="2026-2027"):
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, 'Midfielder')", [player_uid, name])
    con.execute(
        "INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) "
        "VALUES (?, ?, '1', ?, ?)", [name, normalized, season, player_uid],
    )


def test_resolve_element_ids_maps_uid_to_a_real_element_id(con):
    _seed_resolvable_player(con, "Bruno Fernandes", "bruno fernandes", "p_bruno")
    _seed_resolvable_player(con, "Erling Haaland", "erling haaland", "p_haaland")
    element_names = {8: "Bruno Fernandes", 9: "Erling Haaland"}
    out = proj.resolve_element_ids(con, "2026-2027", element_names)
    assert out == {"p_bruno": 8, "p_haaland": 9}


def test_resolve_element_ids_omits_unresolvable_names_never_guesses(con):
    element_names = {8: "Not A Real Player"}
    assert proj.resolve_element_ids(con, "2026-2027", element_names) == {}


def test_resolve_element_ids_empty_input_returns_empty(con):
    assert proj.resolve_element_ids(con, "2026-2027", {}) == {}
