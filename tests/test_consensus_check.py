import json
from datetime import datetime

import pytest

from fpl_quant import consensus_check as cc


def _seed_pool_and_squad(con, target_season="2026-2027", target_gameweek=2):
    """A minimal but real (DB-backed) candidate pool: two defenders at the SAME price
    (same-price-band by construction) plus enough other players to fill a legal 15-man
    squad, with the squad already stored directly in squad_optimizer_selections (bypassing a
    real MIQP solve -- this test is about consensus_check's own evidence-comparison logic,
    not re-proving solve() works)."""
    clubs = ["clubA", "clubB", "clubC", "clubD", "clubE", "clubF"]
    for club in clubs:
        con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES (?, ?)", [club, club])
    con.execute(
        "INSERT INTO fact_match (match_id, season, gameweek, home_team_uid, away_team_uid, finished, "
        "competition, kickoff_time, _ingested_at) VALUES ('m1', ?, ?, 'clubA', 'clubB', FALSE, "
        "'Premier League', '2026-08-24', current_timestamp)", [target_season, target_gameweek],
    )
    con.execute(
        "INSERT INTO team_strength_model_versions (calibration_asof_date, home_advantage, xi_params_version, "
        "rho_params_version, reference_team_uid) VALUES ('2026-08-10', 0.2, 1, 1, 'clubA')"
    )
    ts_mv = con.execute("SELECT max(model_version) FROM team_strength_model_versions").fetchone()[0]
    con.execute(
        "INSERT INTO minutes_model_versions (calibration_asof_date, target_season, decay_params_version, "
        "adjustment_params_version, shrinkage_params_version, fact_multiplier_params_version, lookback_seasons) "
        "VALUES ('2026-08-10', ?, 1, 1, 1, 1, '[]')", [target_season],
    )
    mm_mv = con.execute("SELECT max(model_version) FROM minutes_model_versions").fetchone()[0]
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

    # 2 GK, 5 DEF (def0/def1 at the SAME price -- the price-band comparison pair), 5 MID, 3 FWD
    players = []
    for i in range(2):
        players.append((f"gk{i}", "Goalkeeper", 3.0 + i * 0.5, 4.5 + i, clubs[i % 6]))
    for i in range(5):
        players.append((f"def{i}", "Defender", 2.5 + i * 0.3, 5.0, clubs[i % 6]))  # flat price 5.0 for all 5
    for i in range(5):
        players.append((f"mid{i}", "Midfielder", 3.0 + i * 0.4, 6.0 + i * 0.5, clubs[i % 6]))
    for i in range(3):
        players.append((f"fwd{i}", "Forward", 3.5 + i * 0.5, 7.0 + i * 0.5, clubs[i % 6]))

    for uid, position, mu, price, club in players:
        con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, ?)", [uid, uid, position])
        con.execute(
            "INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) "
            "VALUES (?, ?, ?, ?, ?)", [uid, uid.lower(), club, target_season, uid],
        )
        con.execute(
            "INSERT INTO fact_player_season_stats (player_uid, season, gw, now_cost, _ingested_at) "
            "VALUES (?, ?, 1, ?, current_timestamp)", [uid, target_season, price],
        )
        con.execute(
            "INSERT INTO ep_outputs (model_version, player_uid, fixture_match_id, ep_appearance, ep_goals, "
            "ep_assists, ep_clean_sheet, ep_goals_conceded, ep_defcon, ep_bonus, ep_saves, ep_penalty_save, "
            "ep_cards, ep_own_goal, ep_total, expected_bps) VALUES (?, ?, 'm1', 0,0,0,0,0,0,0,0,0,0,0, ?, 5.0)",
            [ep_mv, uid, mu],
        )
        con.execute(
            "INSERT INTO uncertainty_outputs (model_version, player_uid, fixture_match_id, var_appearance, "
            "var_goals, var_assists, var_clean_sheet, var_goals_conceded, var_defcon, var_bonus, var_saves, "
            "var_total, skew, excess_kurtosis, quantile_05, quantile_25, quantile_75, quantile_95) "
            "VALUES (?, ?, 'm1', 0,0,0,0,0,0,0,0, 1.0, 0,0,0,0,0,0)", [un_mv, uid],
        )

    run_id = con.execute(
        "INSERT INTO squad_optimizer_runs (calibration_asof_date, target_season, target_gameweek, "
        "ep_model_version, uncertainty_model_version, lambda_params_version, lambda_value, "
        "guardrail_params_version, divergence_check_passed, solver_status, objective_value) "
        "VALUES ('2026-08-10', ?, ?, ?, ?, 1, 0.15, 1, TRUE, 'optimal', 9.0) RETURNING run_id",
        [target_season, target_gameweek, ep_mv, un_mv],
    ).fetchone()[0]
    for uid, position, *_rest in players:
        con.execute(
            "INSERT INTO squad_optimizer_selections (run_id, player_uid, in_squad, in_xi, is_captain, is_vice) "
            "VALUES (?, ?, TRUE, FALSE, FALSE, FALSE)", [run_id, uid],
        )
    return run_id


def _seed_source(con, source_id, reliability):
    con.execute(
        "INSERT INTO sources (source_id, source_name, source_type, base_reliability_score) "
        "VALUES (?, ?, 'community', ?)", [source_id, source_id, reliability],
    )


def _seed_evidence_claim(con, player_uid, claim_type, source_id, reliability, confidence):
    import uuid
    con.execute(
        "INSERT INTO evidence_claims (claim_id, subject_entity_type, subject_entity_id, claim_type, "
        "claim_value, claim_value_numeric, information_type, source_id, source_reliability_score, "
        "confidence, observed_date, ingested_date) "
        "VALUES (?, 'player', ?, ?, ?, NULL, 'OPINION', ?, ?, ?, '2026-08-01', ?)",
        [str(uuid.uuid4()), player_uid, claim_type, json.dumps({"claim": "some free text"}), source_id,
         reliability, confidence, datetime(2026, 8, 1)],
    )


def test_flags_a_same_price_position_alternative_with_meaningfully_higher_evidence(con):
    """Real motivating case: def0 (selected) has NO evidence at all; def1 (same price, same
    position) is well-regarded (real community evidence). def0 must be flagged with def1
    named as the alternative."""
    run_id = _seed_pool_and_squad(con)
    _seed_source(con, "s1", reliability=1.0)
    _seed_evidence_claim(con, "def1", "community_sentiment", "s1", reliability=1.0, confidence=1.0)

    flags = cc.flag_consensus_divergent_picks(
        con, run_id, datetime(2026, 8, 10), decay_params_version=1, fact_multiplier_params_version=1,
        price_band=0.5, divergence_ratio_threshold=0.2,
    )
    by_selected = {f["selected_player_uid"]: f for f in flags}
    assert "def0" in by_selected
    assert by_selected["def0"]["alternative_player_uid"] == "def1"
    assert by_selected["def0"]["selected_evidence_weight"] == 0.0
    assert by_selected["def0"]["alternative_evidence_weight"] > 0.0


def test_no_flag_when_selected_player_has_equal_or_better_evidence(con):
    run_id = _seed_pool_and_squad(con)
    _seed_source(con, "s1", reliability=1.0)
    _seed_evidence_claim(con, "def0", "community_sentiment", "s1", reliability=1.0, confidence=1.0)  # SELECTED player has the evidence

    flags = cc.flag_consensus_divergent_picks(
        con, run_id, datetime(2026, 8, 10), decay_params_version=1, fact_multiplier_params_version=1,
        price_band=0.5, divergence_ratio_threshold=0.2,
    )
    by_selected = {f["selected_player_uid"]: f for f in flags}
    assert "def0" not in by_selected


def test_no_flag_for_a_different_position_even_with_higher_evidence(con):
    """A same-price MIDFIELDER with tons of evidence must never be proposed as an alternative
    to a DEFENDER -- position match is a hard requirement, not a soft preference."""
    run_id = _seed_pool_and_squad(con)
    _seed_source(con, "s1", reliability=1.0)
    _seed_evidence_claim(con, "fwd0", "community_sentiment", "s1", reliability=1.0, confidence=1.0)

    flags = cc.flag_consensus_divergent_picks(
        con, run_id, datetime(2026, 8, 10), decay_params_version=1, fact_multiplier_params_version=1,
        price_band=10.0, divergence_ratio_threshold=0.2,  # deliberately huge price band
    )
    by_selected = {f["selected_player_uid"]: f for f in flags}
    assert "def0" not in by_selected  # fwd0 is a Forward, never eligible as a Defender's alternative


def test_no_flag_when_alternative_is_outside_price_band(con):
    run_id = _seed_pool_and_squad(con, target_gameweek=3)
    _seed_source(con, "s1", reliability=1.0)
    # fwd2 (position Forward, priced far above any defender) has huge evidence but is
    # irrelevant to def0 both by position and price -- reuse the position-mismatch fixture's
    # own price gap isn't enough on its own; assert directly with a same-position but
    # out-of-band defender instead by widening def4's price artificially.
    con.execute("UPDATE fact_player_season_stats SET now_cost = 20.0 WHERE player_uid = 'def4'")
    _seed_evidence_claim(con, "def4", "community_sentiment", "s1", reliability=1.0, confidence=1.0)

    flags = cc.flag_consensus_divergent_picks(
        con, run_id, datetime(2026, 8, 10), decay_params_version=1, fact_multiplier_params_version=1,
        price_band=0.5, divergence_ratio_threshold=0.2,
    )
    by_selected = {f["selected_player_uid"]: f for f in flags}
    assert "def0" not in by_selected  # def4 is now GBP20.0 vs def0's GBP5.0 -- far outside the +/-0.5 band


def test_raises_on_unknown_run(con):
    with pytest.raises(ValueError):
        cc.flag_consensus_divergent_picks(
            con, 999, datetime(2026, 8, 10), decay_params_version=1, fact_multiplier_params_version=1,
            price_band=0.5, divergence_ratio_threshold=0.2,
        )
