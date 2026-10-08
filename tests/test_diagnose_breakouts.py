"""scripts/diagnose_breakouts.py: splits the next gameweek's live EP of a named player into
minutes, scoring rates and club strength, against what he is doing this season."""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import diagnose_breakouts as diag  # noqa: E402

LIVE, PRIOR = "2026-2027", "2025-2026"
INGESTED_AT = datetime(2026, 10, 6, 18, 5)
GW1 = datetime(2026, 8, 22, 15, 0)


def _seed(con):
    """team_a v team_b every week. p1 joined team_a this season (no earlier league minutes) and
    started GW2-5; the minutes model gives him p_start 0.4. GW6 is the next gameweek."""
    for team in ("team_a", "team_b"):
        con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES (?, ?)", [team, team])
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p1', 'Player One', 'Midfielder')")
    con.execute(
        "INSERT INTO fact_match (match_id, season, gameweek, kickoff_time, home_team_uid, away_team_uid, home_score, "
        "away_score, finished, competition, _ingested_at) VALUES ('prior', ?, 38, ?, 'team_a', 'team_b', 1, 0, TRUE, "
        "'Premier League', ?)", [PRIOR, datetime(2026, 5, 24, 15, 0), INGESTED_AT],
    )
    for gw in range(1, 7):
        kickoff = GW1 + timedelta(days=7 * (gw - 1)) if gw < 6 else datetime(2026, 10, 10, 14, 0)
        played = gw < 6
        con.execute(
            "INSERT INTO fact_match (match_id, season, gameweek, kickoff_time, home_team_uid, away_team_uid, home_score, "
            "away_score, finished, competition, _ingested_at) VALUES (?, ?, ?, ?, 'team_a', 'team_b', ?, ?, ?, "
            "'Premier League', ?)",
            [f"m{gw}", LIVE, gw, kickoff, 1 if played else None, 0 if played else None, played, INGESTED_AT],
        )
        if 2 <= gw <= 5:
            con.execute(
                "INSERT INTO fact_player_match_stats (player_uid, match_id, season, start_min, minutes_played, _ingested_at) "
                "VALUES ('p1', ?, ?, 0, 90, ?)", [f"m{gw}", LIVE, INGESTED_AT],
            )
    con.execute(
        "INSERT INTO fact_player_season_stats (player_uid, season, gw, minutes, goals_scored, assists, expected_goals, "
        "expected_assists, expected_goals_per_90, expected_assists_per_90, total_points, now_cost, event_points, _ingested_at) "
        "VALUES ('p1', ?, 5, 360, 2, 2, 1.2, 1.0, 0.30, 0.25, 30, 5.9, 8, ?)", [LIVE, INGESTED_AT],
    )
    ts_mv = con.execute(
        "INSERT INTO team_strength_model_versions (calibration_asof_date, home_advantage, xi_params_version, "
        "rho_params_version, reference_team_uid) VALUES ('2026-10-06', 0.2, 1, 1, 'team_a') RETURNING model_version"
    ).fetchone()[0]
    for team, attack, defence in (("team_a", 0.1, -0.1), ("team_b", -0.2, 0.2)):
        con.execute(
            "INSERT INTO team_strength_snapshots (model_version, team_uid, final_attack, final_defence, "
            "seasons_of_topflight_data, weight_own_data) VALUES (?, ?, ?, ?, 2, 1.0)",
            [ts_mv, team, attack, defence],
        )
    mm_mv = con.execute(
        "INSERT INTO minutes_model_versions (calibration_asof_date, target_season, decay_params_version, "
        "adjustment_params_version, shrinkage_params_version, fact_multiplier_params_version, lookback_seasons) "
        "VALUES ('2026-10-06', ?, 1, 1, 1, 1, '[]') RETURNING model_version", [LIVE],
    ).fetchone()[0]
    con.execute(
        "INSERT INTO minutes_model_outputs (model_version, player_uid, position, p_start_historical_final, "
        "p_start_historical_position_avg, weight_own, logit_adjustment_total, p_start_final, "
        "p_used_as_sub_given_not_started, p_0min, p_1_59min, p_60plus_min, competitive_matches_last_2_seasons) "
        "VALUES (?, 'p1', 'Midfielder', 0.4, 0.5, 0.2, 0.0, 0.4, 0.3, 0.42, 0.2, 0.38, 4)", [mm_mv],
    )
    ep_mv = con.execute(
        "INSERT INTO ep_model_versions (calibration_asof_date, target_season, team_strength_model_version, "
        "minutes_model_version, scoring_matrix_params_version, bps_params_version, bps_tau_params_version) "
        "VALUES ('2026-10-06', ?, ?, ?, 1, 1, 1) RETURNING model_version", [LIVE, ts_mv, mm_mv],
    ).fetchone()[0]
    con.execute(
        "INSERT INTO ep_outputs (model_version, player_uid, fixture_match_id, ep_appearance, ep_goals, ep_assists, "
        "ep_clean_sheet, ep_goals_conceded, ep_defcon, ep_bonus, ep_saves, ep_penalty_save, ep_cards, ep_own_goal, "
        "ep_total, expected_bps) VALUES (?, 'p1', 'm6', 1.2, 0.5, 0.3, 0.1, 0, 0, 0.2, 0, 0, 0, 0, 2.3, 15.0)", [ep_mv],
    )
    return ep_mv


def _club_spells(con, seasons):
    con.execute(
        "CREATE OR REPLACE TEMP TABLE _player_season_team "
        "(player_uid VARCHAR, season VARCHAR, team_uid VARCHAR, first_gw INTEGER, last_gw INTEGER)"
    )
    con.execute("INSERT INTO _player_season_team VALUES ('p1', ?, 'team_a', 0, 99)", [LIVE])


def test_diagnosis_splits_the_next_gameweek_and_names_the_main_cause(con, monkeypatch, tmp_path):
    ep_mv = _seed(con)
    monkeypatch.setattr(diag.breakout, "build_club_spells", _club_spells)
    projections = tmp_path / "projections_latest.json"
    projections.write_text(json.dumps({"players": [{"player_uid": "p1", "ep_per_gw": [{"gw": 6, "ep": 2.31}]}]}))
    monkeypatch.setattr(diag, "PROJECTIONS_PATH", projections)

    out = diag.diagnose(con, ("p1",))

    assert (out["status"], out["target_gameweek"], out["ep_model_version"]) == ("ok", 6, ep_mv)
    (p1,) = out["players"]
    assert p1["minutes_model"]["p_start_final"] == 0.4
    assert p1["ep"]["ep_total"] == 2.3
    assert p1["club"]["team_uid"] == "team_a" and p1["opponent"]["team_uid"] == "team_b"
    assert p1["this_season"]["starts"] == 4 and p1["this_season"]["club_matches_played"] == 5
    assert p1["this_season"]["xg_per_90"] == 0.30
    assert p1["in_breakout_group"] == "breakout"  # 4 of the last 4, no earlier league minutes
    assert p1["matches_projections_within_0_05"] is True
    # starts 4 of 5 (0.8) against p_start 0.4: minutes would add 2.3 * (0.8/0.4 - 1) = 2.3
    assert abs(p1["gap"]["causes_points_per_gw"]["minutes"] - 2.3) < 1e-9
    assert p1["gap"]["main_cause"] == "minutes"


def test_placeholder_when_the_next_gameweek_has_no_ep(con):
    _seed(con)
    con.execute("DELETE FROM ep_outputs")
    assert diag.diagnose(con, ("p1",))["status"] == "no_ep_outputs_for_target_gameweek"
