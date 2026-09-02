"""research.ml.forward.predict_forward -- the one live/forward path: fit Huber δ=4 on all
walk-forward history, predict ep_total_ml for the upcoming gameweek."""

from __future__ import annotations

from datetime import datetime

import pytest

from research.ml import forward
from research.ml.tests.conftest import CLUBS, ROUNDS, _player_pool

pytest.importorskip("lightgbm")

LIVE_SEASON = "2026-2027"
NOW = datetime(2026, 8, 1)


def _seed_live_upcoming_gameweek(con, season=LIVE_SEASON, gw=1):
    """A live, not-yet-played gameweek: player_alias + raw teams + a fixture card with no
    result + model-version rows + ep_outputs Q(x). No fact_player_season_stats at `gw` (that's
    the realised outcome, which doesn't exist yet) and no backtest_gameweek_steps row."""
    club_code = {name: str(i + 1) for i, name in enumerate(CLUBS)}
    table = f"raw_{season.replace('-', '_')}_teams"
    con.execute(f'CREATE TABLE "{table}" (code VARCHAR, id VARCHAR, name VARCHAR, short_name VARCHAR)')
    for name in CLUBS:
        code = club_code[name]
        con.execute(f'INSERT INTO "{table}" VALUES (?, ?, ?, ?)', [code, code, name, name[:3]])
        con.execute("INSERT INTO team_alias (alias_name, season, team_uid, alias_source) VALUES (?, ?, ?, 't')",
                    [name, season, f"team_{name.lower()}"])
    con.execute("INSERT INTO fact_raw_ingestion_log (raw_table_name, season, source_relpath, source_file_hash, row_count) "
                "VALUES (?, ?, 'teams.csv', ?, ?)", [table, season, f"hash_{season}", len(CLUBS)])
    for pid, _pos, club in _player_pool():
        con.execute("INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) "
                    "VALUES (?, ?, ?, ?, ?)", [pid, pid, club_code[club], season, pid])

    kickoff = datetime(2026, 8, 15, 15, 0)
    match_by_club = {}
    for i, (h, a) in enumerate(ROUNDS[(gw - 1) % len(ROUNDS)]):
        mid = f"m_{season}_{gw}_{i}"
        con.execute(
            "INSERT INTO fact_match (match_id, season, gameweek, kickoff_time, home_team_uid, away_team_uid, "
            "home_score, away_score, finished, competition, _ingested_at) "
            "VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, FALSE, 'Premier League', ?)",
            [mid, season, gw, kickoff, f"team_{h.lower()}", f"team_{a.lower()}", NOW],
        )
        match_by_club[h] = mid
        match_by_club[a] = mid

    con.execute("INSERT INTO team_strength_model_versions (calibration_asof_date, home_advantage, "
                "xi_params_version, rho_params_version, reference_team_uid) VALUES ('26-08-01', 0.2, 1, 1, 'team_a')")
    ts_mv = con.execute("SELECT max(model_version) FROM team_strength_model_versions").fetchone()[0]
    con.execute("INSERT INTO minutes_model_versions (calibration_asof_date, target_season, decay_params_version, "
                "adjustment_params_version, shrinkage_params_version, fact_multiplier_params_version, lookback_seasons) "
                "VALUES ('26-08-01', ?, 1, 1, 1, 1, '[]')", [season])
    mm_mv = con.execute("SELECT max(model_version) FROM minutes_model_versions").fetchone()[0]
    con.execute("INSERT INTO ep_model_versions (calibration_asof_date, target_season, team_strength_model_version, "
                "minutes_model_version, scoring_matrix_params_version, bps_params_version, bps_tau_params_version) "
                "VALUES ('26-08-01', ?, ?, ?, 1, 1, 1)", [season, ts_mv, mm_mv])
    ep_mv = con.execute("SELECT max(model_version) FROM ep_model_versions").fetchone()[0]

    for pid, pos, club in _player_pool():
        con.execute(
            "INSERT INTO minutes_model_outputs (model_version, player_uid, position, p_start_historical_final, "
            "p_start_historical_position_avg, weight_own, logit_adjustment_total, p_start_final, "
            "p_used_as_sub_given_not_started, p_0min, p_1_59min, p_60plus_min, competitive_matches_last_2_seasons) "
            "VALUES (?, ?, ?, 0.9, 0.9, 1.0, 0.0, 0.9, 0.0, 0.05, 0.05, 0.9, 20)",
            [mm_mv, pid, pos],
        )
        mid = match_by_club.get(club)
        if mid is None:
            continue
        q_pred = {"Goalkeeper": 2.0, "Defender": 3.0, "Midfielder": 4.0, "Forward": 5.0}.get(pos, 2.0)
        con.execute(
            "INSERT INTO ep_outputs (model_version, player_uid, fixture_match_id, ep_appearance, ep_goals, "
            "ep_assists, ep_clean_sheet, ep_goals_conceded, ep_defcon, ep_bonus, ep_saves, ep_penalty_save, "
            "ep_cards, ep_own_goal, ep_total, expected_bps) VALUES (?, ?, ?, 1.0, 0.3, 0.1, 0, 0, 0, 0.2, 0, 0, 0, 0, ?, 20.0)",
            [ep_mv, pid, mid, q_pred],
        )
    return ep_mv, mm_mv


def test_predict_forward_returns_ep_total_ml_for_the_upcoming_gameweek(seeded_db):
    ep_mv, mm_mv = _seed_live_upcoming_gameweek(seeded_db)

    out = forward.predict_forward(seeded_db, LIVE_SEASON, 1, ep_mv, mm_mv, min_train_rows=1)

    assert out is not None
    assert set(out.columns) == {"player_uid", "ep_quant", "predicted_residual", "ep_total_ml"}
    assert len(out) == 18  # every player with a fixture
    assert (out["ep_total_ml"] == out["ep_quant"] + out["predicted_residual"]).all()
    # the model was trained on the two synthetic backtested seasons
    assert set(out.attrs["train_seasons"]) == {"2024-2025", "2025-2026"}
    # predictions are finite, and near the quant baseline (a residual correction, not a rewrite)
    assert out["ep_total_ml"].notna().all()
    assert (out["predicted_residual"].abs() < 15).all()


def test_predict_forward_none_without_walk_forward_history(con):
    # `con` is the empty-schema fixture: no backtest_gameweek_steps at all.
    assert forward.predict_forward(con, LIVE_SEASON, 1, 1, 1) is None


def test_predict_forward_none_when_no_live_ep_outputs(seeded_db):
    # walk-forward history exists (seeded_db), but nothing seeded for the target gameweek.
    assert forward.predict_forward(seeded_db, LIVE_SEASON, 1, 999, 999, min_train_rows=1) is None


def test_build_ml_shadow_payload_ok_and_placeholder(seeded_db, monkeypatch):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    import compute_ml_shadow as cms

    monkeypatch.setattr("research.ml.forward.MIN_TRAIN_ROWS", 1)

    # no live gameweek seeded yet -> honest placeholder, never fabricated numbers
    before = cms.build_ml_shadow_payload(seeded_db, target_gameweek=1)
    assert before["status"] != "ok" and before["players"] == []

    _seed_live_upcoming_gameweek(seeded_db)
    payload = cms.build_ml_shadow_payload(seeded_db, target_gameweek=1, element_names={})
    assert payload["status"] == "ok"
    assert len(payload["players"]) == 18
    p0 = payload["players"][0]
    assert set(p0) == {"player_uid", "name", "fpl_element_id", "ep_quant", "ep_ml", "ml_residual"}
    assert payload["players"] == sorted(payload["players"], key=lambda x: x["ep_ml"], reverse=True)
    assert len(payload["ml_boosts"]) <= 5 and len(payload["ml_fades"]) <= 5
