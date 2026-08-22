import json
from datetime import date, datetime, timezone

import pytest

from fpl_quant import minutes_model as mm
from fpl_quant import params


def test_logit_sigmoid_round_trip():
    for p in [0.01, 0.1, 0.5, 0.835, 0.99]:
        assert abs(mm.sigmoid(mm.logit(p)) - p) < 1e-9


def test_logit_clips_extremes_without_erroring():
    assert mm.logit(0.0) < 0
    assert mm.logit(1.0) > 0


# ---------------------------------------------------------------- fixtures ----

def _seed_raw_teams_csv(con, season, rows):
    """minutes_model resolves player->team via reconcile._season_root_table, which reads
    fact_raw's teams.csv table -- a real dependency, not an artifact to route around in
    tests. rows: list of (code, name)."""
    table = f"raw_{season.replace('-', '_')}_teams"
    con.execute(f'CREATE TABLE "{table}" (code VARCHAR, name VARCHAR)')
    for code, name in rows:
        con.execute(f'INSERT INTO "{table}" VALUES (?, ?)', [code, name])
    con.execute(
        "INSERT INTO fact_raw_ingestion_log (raw_table_name, season, source_relpath, source_file_hash, row_count) "
        "VALUES (?, ?, 'teams.csv', ?, ?)",
        [table, season, f"fakehash_{table}", len(rows)],
    )


def _seed_league(con, seasons=("2024-2025", "2025-2026")):
    """Two teams, A and B, each playing the other repeatedly across two seasons, with
    player P1 (MID) a nailed-on starter for A and player P2 (MID) a fringe/sub player."""
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('team_a', 'A')")
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('team_b', 'B')")
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p1', 'Player One', 'Midfielder')")
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p2', 'Player Two', 'Midfielder')")

    now = datetime.now(timezone.utc)
    match_i = 0
    for season in seasons:
        _seed_raw_teams_csv(con, season, [("1", "A"), ("2", "B")])
        con.execute(
            "INSERT INTO team_alias (alias_name, season, team_uid, alias_source) VALUES ('A', ?, 'team_a', 't')",
            [season],
        )
        con.execute(
            "INSERT INTO team_alias (alias_name, season, team_uid, alias_source) VALUES ('B', ?, 'team_b', 't')",
            [season],
        )
        con.execute(
            "INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) "
            "VALUES ('Player One', 'player one', '1', ?, 'p1')",
            [season],
        )
        con.execute(
            "INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) "
            "VALUES ('Player Two', 'player two', '1', ?, 'p2')",
            [season],
        )
        for _ in range(10):
            match_id = f"m{match_i}"
            match_i += 1
            con.execute(
                "INSERT INTO fact_match (match_id, season, home_team_uid, away_team_uid, finished, "
                "competition, kickoff_time, _ingested_at) VALUES (?, ?, 'team_a', 'team_b', TRUE, "
                "'Premier League', ?, ?)",
                [match_id, season, datetime(2025, 1, 1) if season == "2024-2025" else datetime(2026, 1, 1), now],
            )
            # P1: always starts, always plays 90.
            con.execute(
                "INSERT INTO fact_player_match_stats (player_uid, match_id, season, start_min, "
                "finish_min, minutes_played, _ingested_at) VALUES ('p1', ?, ?, 0, 90, 90, ?)",
                [match_id, season, now],
            )
            # P2: never features at all (an unused/absent squad player throughout).


def test_required_invariant_probabilities_sum_to_one(con):
    _seed_league(con)
    params.write_param(con, "minutes_model_decay_params", 1, "2026-08-10", "xi", value_numeric=0.0018)
    params.write_param(con, "minutes_adjustment_params", 1, "2026-08-10", "cap", value_numeric=6.0, dimensions={"scope": "global"})
    params.write_param(con, "minutes_model_shrinkage_params", 1, "2026-08-10", "competitive_matches_threshold", value_numeric=10)

    model_version = mm.run(
        con, date(2026, 8, 10), "2025-2026",
        decay_params_version=1, adjustment_params_version=1,
        shrinkage_params_version=1, fact_multiplier_params_version=1,
        lookback_seasons=("2024-2025", "2025-2026"),
    )
    df = con.execute(
        "SELECT player_uid, p_0min, p_1_59min, p_60plus_min FROM minutes_model_outputs WHERE model_version = ?",
        [model_version],
    ).fetchdf()
    assert len(df) == 2
    totals = df.p_0min + df.p_1_59min + df.p_60plus_min
    assert (totals.sub(1.0).abs() < 1e-9).all()
    assert (df.p_0min >= 0).all() and (df.p_1_59min >= 0).all() and (df.p_60plus_min >= 0).all()


def test_nailed_on_starter_has_low_p0_fringe_player_has_high_p0(con):
    _seed_league(con)
    params.write_param(con, "minutes_model_decay_params", 1, "2026-08-10", "xi", value_numeric=0.0018)
    params.write_param(con, "minutes_adjustment_params", 1, "2026-08-10", "cap", value_numeric=6.0, dimensions={"scope": "global"})
    params.write_param(con, "minutes_model_shrinkage_params", 1, "2026-08-10", "competitive_matches_threshold", value_numeric=10)

    model_version = mm.run(
        con, date(2026, 8, 10), "2025-2026",
        decay_params_version=1, adjustment_params_version=1,
        shrinkage_params_version=1, fact_multiplier_params_version=1,
        lookback_seasons=("2024-2025", "2025-2026"),
    )
    rows = con.execute(
        "SELECT player_uid, p_0min, weight_own, competitive_matches_last_2_seasons FROM minutes_model_outputs "
        "WHERE model_version = ?", [model_version],
    ).fetchdf().set_index("player_uid")

    assert rows.loc["p1", "p_0min"] < rows.loc["p2", "p_0min"]
    assert rows.loc["p1", "weight_own"] == 1.0  # 20 competitive matches >> threshold of 10
    assert rows.loc["p1", "competitive_matches_last_2_seasons"] == 20

    # Priority 6's bulk accessor must agree with a direct column read, and must not include
    # uids that weren't asked for.
    weight_own_by_uid = mm.weight_own_by_player(con, model_version, ["p1", "p2"])
    assert weight_own_by_uid["p1"] == pytest.approx(rows.loc["p1", "weight_own"])
    assert weight_own_by_uid["p2"] == pytest.approx(rows.loc["p2", "weight_own"])
    assert mm.weight_own_by_player(con, model_version, []) == {}
    assert mm.weight_own_by_player(con, model_version, ["p1"]) == {"p1": weight_own_by_uid["p1"]}


def test_zero_history_player_gets_pure_position_average(con):
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('team_a', 'A')")
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('team_b', 'B')")
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p_new', 'New Signing', 'Forward')")
    con.execute(
        "INSERT INTO team_alias (alias_name, season, team_uid, alias_source) VALUES ('A', '2026-2027', 'team_a', 't')"
    )
    con.execute(
        "INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) "
        "VALUES ('New Signing', 'new signing', '1', '2026-2027', 'p_new')"
    )
    params.write_param(con, "minutes_model_decay_params", 1, "2026-08-10", "xi", value_numeric=0.0018)
    params.write_param(con, "minutes_adjustment_params", 1, "2026-08-10", "cap", value_numeric=6.0, dimensions={"scope": "global"})
    params.write_param(con, "minutes_model_shrinkage_params", 1, "2026-08-10", "competitive_matches_threshold", value_numeric=10)

    model_version = mm.run(
        con, date(2026, 8, 10), "2026-2027",
        decay_params_version=1, adjustment_params_version=1,
        shrinkage_params_version=1, fact_multiplier_params_version=1,
        lookback_seasons=("2024-2025", "2025-2026"),
    )
    row = con.execute(
        "SELECT weight_own, p_start_historical_own, p_start_historical_final, p_start_historical_position_avg "
        "FROM minutes_model_outputs WHERE model_version = ? AND player_uid = 'p_new'", [model_version],
    ).fetchone()
    weight_own, p_own, p_final, p_pos_avg = row
    assert weight_own == 0.0
    assert p_own is None
    assert p_final == pytest.approx(p_pos_avg)


def test_log_preseason_involvement_claims_are_low_weight_and_not_double_logged(con):
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('team_a', 'A')")
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('team_b', 'B')")
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p1', 'Player One', 'MID')")
    con.execute("INSERT INTO sources (source_id, source_name, source_type, base_reliability_score) "
                 "VALUES ('src_system-derived', 'system-derived', 'system-derived', NULL)")
    now = datetime.now(timezone.utc)
    con.execute(
        "INSERT INTO fact_match (match_id, season, gameweek, home_team_uid, away_team_uid, finished, "
        "competition, _ingested_at) VALUES ('gw0m1', '2026-2027', 0, 'team_a', 'team_b', TRUE, 'Friendlies', ?)",
        [now],
    )
    con.execute(
        "INSERT INTO fact_player_match_stats (player_uid, match_id, season, start_min, finish_min, "
        "minutes_played, _ingested_at) VALUES ('p1', 'gw0m1', '2026-2027', 0, 60, 60, ?)", [now],
    )
    n = mm.log_preseason_involvement_claims(con, "2026-2027")
    assert n == 1
    claim = con.execute(
        "SELECT claim_type, confidence, claim_value, subject_entity_id FROM evidence_claims "
        "WHERE claim_type = 'preseason_involvement'"
    ).fetchone()
    claim_type, confidence, claim_value, subject_entity_id = claim
    assert subject_entity_id == "p1"
    assert confidence < 0.5  # deliberately low-weight, per spec
    assert json.loads(claim_value)["total_preseason_minutes"] == 60


# ============================================================
# M9 adapter -- explain_player_adjustment() must agree with compute_logit_adjustment()
# ============================================================

def _seed_evidence_and_params_for_adjustment(con):
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p1', 'Player One', 'Forward')")
    con.execute("INSERT INTO sources (source_id, source_name, source_type, base_reliability_score) "
                 "VALUES ('s_official', 'Official', 'official', 1.0)")
    con.execute(
        "INSERT INTO evidence_claims (claim_id, subject_entity_type, subject_entity_id, claim_type, "
        "claim_value, information_type, source_id, source_reliability_score, confidence, observed_date, ingested_date, raw_text) "
        "VALUES ('c1', 'player', 'p1', 'injury_status', ?, 'FACT', 's_official', 1.0, 0.9, '2026-08-01', ?, 'ruled out')",
        [json.dumps({"category": "Out"}), datetime(2026, 8, 1)],
    )
    con.execute(
        "INSERT INTO evidence_claims (claim_id, subject_entity_type, subject_entity_id, claim_type, "
        "claim_value, claim_value_numeric, information_type, source_id, source_reliability_score, confidence, "
        "observed_date, ingested_date) "
        "VALUES ('c2', 'player', 'p1', 'predicted_xi', ?, 0.9, 'OPINION', 's_official', 1.0, 0.8, '2026-08-01', ?)",
        [json.dumps({"reasoning": "Named in the provisional XI by the beat reporter"}), datetime(2026, 8, 1)],
    )
    params.write_param(con, "minutes_adjustment_params", 1, "2026-08-10", "magnitude",
                        value_numeric=-4.0, dimensions={"claim_type": "injury_status", "category": "Out"})
    params.write_param(con, "minutes_adjustment_params", 1, "2026-08-10", "magnitude",
                        value_numeric=0.8, dimensions={"claim_type": "predicted_xi"})
    params.write_param(con, "minutes_adjustment_params", 1, "2026-08-10", "cap",
                        value_numeric=6.0, dimensions={"scope": "global"})


def test_explain_player_adjustment_contributions_sum_to_compute_logit_adjustment(con):
    _seed_evidence_and_params_for_adjustment(con)
    asof = datetime(2026, 8, 10, 23, 59, 59, tzinfo=timezone.utc)
    p_start_historical_final = 0.7

    expected_total = mm.compute_logit_adjustment(
        con, "p1", p_start_historical_final, asof,
        adjustment_params_version=1, decay_params_version=1, fact_multiplier_params_version=1,
    )

    # explain_player_adjustment() reads its inputs from a real minutes_model_outputs row, not
    # bare arguments -- seed the minimal versions/output row it depends on.
    con.execute(
        "INSERT INTO minutes_model_versions (calibration_asof_date, target_season, decay_params_version, "
        "adjustment_params_version, shrinkage_params_version, fact_multiplier_params_version, lookback_seasons) "
        "VALUES ('2026-08-10', '2026-2027', 1, 1, 1, 1, '[]')"
    )
    model_version = con.execute("SELECT max(model_version) FROM minutes_model_versions").fetchone()[0]
    con.execute(
        "INSERT INTO minutes_model_outputs (model_version, player_uid, position, p_start_historical_final, "
        "p_start_historical_position_avg, weight_own, logit_adjustment_total, p_start_final, "
        "p_used_as_sub_given_not_started, p_0min, p_1_59min, p_60plus_min, competitive_matches_last_2_seasons) "
        "VALUES (?, 'p1', 'Forward', ?, ?, 0.0, ?, 0.5, 0.0, 0.5, 0.2, 0.3, 0)",
        [model_version, p_start_historical_final, p_start_historical_final, expected_total],
    )

    detail = mm.explain_player_adjustment(con, model_version, "p1")
    assert len(detail) == 2  # both claims considered
    included = [d for d in detail if d["included"]]
    assert len(included) == 2  # both actually contributed (real magnitude params, real numeric value)

    reconstructed_total = sum(d["contribution"] for d in included)
    reconstructed_capped = max(-6.0, min(6.0, reconstructed_total))
    assert reconstructed_capped == pytest.approx(expected_total)

    # provenance detail is real, not placeholder
    injury = next(d for d in detail if d["claim_type"] == "injury_status")
    assert injury["source_name"] == "Official"
    assert injury["source_type"] == "official"
    predicted_xi = next(d for d in detail if d["claim_type"] == "predicted_xi")
    assert predicted_xi["reasoning"] == "Named in the provisional XI by the beat reporter"


def test_explain_player_adjustment_flags_excluded_claims_with_a_reason(con):
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p1', 'Player One', 'Forward')")
    con.execute("INSERT INTO sources (source_id, source_name, source_type, base_reliability_score) "
                 "VALUES ('s_official', 'Official', 'official', 1.0)")
    con.execute(
        "INSERT INTO evidence_claims (claim_id, subject_entity_type, subject_entity_id, claim_type, "
        "claim_value, information_type, source_id, source_reliability_score, confidence, observed_date, ingested_date) "
        "VALUES ('c1', 'player', 'p1', 'transfer_likelihood', ?, 'FACT', 's_official', 1.0, 0.9, '2026-08-01', ?)",
        [json.dumps({"status": "Complete"}), datetime(2026, 8, 1)],
    )
    params.write_param(con, "minutes_adjustment_params", 1, "2026-08-10", "magnitude",
                        value_numeric=-2.0, dimensions={"claim_type": "transfer_likelihood"})
    params.write_param(con, "minutes_adjustment_params", 1, "2026-08-10", "cap", value_numeric=6.0, dimensions={"scope": "global"})
    con.execute(
        "INSERT INTO minutes_model_versions (calibration_asof_date, target_season, decay_params_version, "
        "adjustment_params_version, shrinkage_params_version, fact_multiplier_params_version, lookback_seasons) "
        "VALUES ('2026-08-10', '2026-2027', 1, 1, 1, 1, '[]')"
    )
    model_version = con.execute("SELECT max(model_version) FROM minutes_model_versions").fetchone()[0]
    con.execute(
        "INSERT INTO minutes_model_outputs (model_version, player_uid, position, p_start_historical_final, "
        "p_start_historical_position_avg, weight_own, logit_adjustment_total, p_start_final, "
        "p_used_as_sub_given_not_started, p_0min, p_1_59min, p_60plus_min, competitive_matches_last_2_seasons) "
        "VALUES (?, 'p1', 'Forward', 0.7, 0.7, 0.0, 0.0, 0.5, 0.0, 0.5, 0.2, 0.3, 0)",
        [model_version],
    )

    detail = mm.explain_player_adjustment(con, model_version, "p1")
    assert len(detail) == 1
    assert detail[0]["included"] is False
    assert detail[0]["exclusion_reason"] == "transfer already completed"
