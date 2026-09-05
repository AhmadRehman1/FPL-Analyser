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


def _seed_two_same_position_starters(con, seasons=("2024-2025", "2025-2026")):
    """Team A with two Midfielders who BOTH start every game, but p_full always plays 90 and
    p_early is always hooked at 45. The position-wide P(60+ | started) is ~0.5; a per-player
    conditional rate must keep p_full near 1.0 and p_early near 0.0."""
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('team_a', 'A'), ('team_b', 'B')")
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p_full', 'Full Ninety', 'Midfielder')")
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p_early', 'Early Hook', 'Midfielder')")
    now = datetime.now(timezone.utc)
    match_i = 0
    for season in seasons:
        _seed_raw_teams_csv(con, season, [("1", "A"), ("2", "B")])
        for name, uid in (("A", "team_a"), ("B", "team_b")):
            con.execute(
                "INSERT INTO team_alias (alias_name, season, team_uid, alias_source) VALUES (?, ?, ?, 't')",
                [name, season, uid],
            )
        for name, norm, uid in (("Full Ninety", "full ninety", "p_full"), ("Early Hook", "early hook", "p_early")):
            con.execute(
                "INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) "
                "VALUES (?, ?, '1', ?, ?)", [name, norm, season, uid],
            )
        for _ in range(12):
            match_id = f"m{match_i}"
            match_i += 1
            con.execute(
                "INSERT INTO fact_match (match_id, season, home_team_uid, away_team_uid, finished, "
                "competition, kickoff_time, _ingested_at) VALUES (?, ?, 'team_a', 'team_b', TRUE, "
                "'Premier League', ?, ?)",
                [match_id, season, datetime(2025, 1, 1) if season == "2024-2025" else datetime(2026, 1, 1), now],
            )
            con.execute(
                "INSERT INTO fact_player_match_stats (player_uid, match_id, season, start_min, "
                "finish_min, minutes_played, _ingested_at) VALUES ('p_full', ?, ?, 0, 90, 90, ?)",
                [match_id, season, now],
            )
            con.execute(
                "INSERT INTO fact_player_match_stats (player_uid, match_id, season, start_min, "
                "finish_min, minutes_played, _ingested_at) VALUES ('p_early', ?, ?, 0, 45, 45, ?)",
                [match_id, season, now],
            )


def test_nailed_full_90_starter_keeps_high_p60_despite_low_position_average(con):
    """Regression: P(60+ | started) was a single position-wide average (~0.71 for forwards,
    ~0.75 for midfielders) applied to every starter, so a nailed 90-minute player inherited a
    rotation-dragged rate and got an inflated p_1_59 / deflated p_60plus. It must now track the
    player's own history."""
    _seed_two_same_position_starters(con)
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
        "SELECT player_uid, p_0min, p_1_59min, p_60plus_min FROM minutes_model_outputs WHERE model_version = ?",
        [model_version],
    ).fetchdf().set_index("player_uid")

    # position average P(60+|started) here is ~0.5 (one always-90, one always-45 starter)
    assert rows.loc["p_full", "p_60plus_min"] > 0.85
    assert rows.loc["p_full", "p_1_59min"] < 0.15
    assert rows.loc["p_early", "p_60plus_min"] < 0.15
    # both are near-certain to feature -- the split is 1-59 vs 60+, not 0
    assert rows.loc["p_early", "p_1_59min"] > 0.7
    assert (rows.p_0min + rows.p_1_59min + rows.p_60plus_min).sub(1.0).abs().max() < 1e-9


def test_player_conditional_minutes_rates_shrinks_small_samples_to_position_average(con):
    _seed_two_same_position_starters(con)
    pos = mm.compute_conditional_minutes_rates(con)
    per_player = mm.compute_player_conditional_minutes_rates(con)
    pos_avg = float(pos.loc["Midfielder", "p_60plus_given_started"])
    # large own sample (24 starts) with threshold 10 -> essentially the own rate
    big = mm._shrunk_conditional_rate(per_player, "p_full", "n_started", "n_started_60plus", pos_avg, 10.0)
    assert big == pytest.approx(1.0, abs=1e-9)
    # a player with no history at all -> pure position average
    fallback = mm._shrunk_conditional_rate(per_player, "unknown_uid", "n_started", "n_started_60plus", pos_avg, 10.0)
    assert fallback == pytest.approx(pos_avg)


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


# ============================================================
# M2 role/club-change data-quality flag -- role_change_evidence_flag()
# ============================================================

def _seed_role_change_scenario(
    con, *, p_start_final, weight_own, claims,
    player_uid="p1", canonical_name="Player One", competitive_matches=75, seed_flag_params=True,
):
    """One modelled player + a minutes_model_outputs row + a list of evidence claims.
    `claims`: list of dicts with keys claim_type, claim_value (dict), and optionally
    claim_value_numeric / observed_date / raw_text / information_type. Returns model_version."""
    con.execute(
        "INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, 'Midfielder')",
        [player_uid, canonical_name],
    )
    con.execute(
        "INSERT INTO sources (source_id, source_name, source_type, base_reliability_score) "
        "VALUES ('s_journo', 'Beat Reporter', 'journalist', 0.8) ON CONFLICT DO NOTHING"
    )
    for i, c in enumerate(claims):
        con.execute(
            "INSERT INTO evidence_claims (claim_id, subject_entity_type, subject_entity_id, claim_type, "
            "claim_value, claim_value_numeric, information_type, source_id, source_reliability_score, "
            "confidence, observed_date, ingested_date, raw_text) "
            "VALUES (?, 'player', ?, ?, ?, ?, ?, 's_journo', 0.8, 0.7, ?, ?, ?)",
            [
                f"{player_uid}_c{i}", player_uid, c["claim_type"],
                json.dumps(c["claim_value"]), c.get("claim_value_numeric"),
                c.get("information_type", "OPINION"),
                c.get("observed_date", "2026-08-01"), datetime(2026, 8, 1), c.get("raw_text"),
            ],
        )
    con.execute(
        "INSERT INTO minutes_model_versions (calibration_asof_date, target_season, decay_params_version, "
        "adjustment_params_version, shrinkage_params_version, fact_multiplier_params_version, lookback_seasons) "
        "VALUES ('2026-08-10', '2026-2027', 1, 1, 1, 1, '[]')"
    )
    model_version = con.execute("SELECT max(model_version) FROM minutes_model_versions").fetchone()[0]
    con.execute(
        "INSERT INTO minutes_model_outputs (model_version, player_uid, position, p_start_historical_own, "
        "p_start_historical_final, p_start_historical_position_avg, weight_own, logit_adjustment_total, "
        "p_start_final, p_used_as_sub_given_not_started, p_0min, p_1_59min, p_60plus_min, "
        "competitive_matches_last_2_seasons) "
        "VALUES (?, ?, 'Midfielder', ?, ?, 0.32, ?, 0.0, ?, 0.1, 0.05, 0.05, 0.9, ?)",
        [model_version, player_uid, p_start_final, p_start_final, weight_own, p_start_final, competitive_matches],
    )
    if seed_flag_params:
        mm.seed_role_change_flag_params(con)
    return model_version


_ASOF = datetime(2026, 9, 2, 23, 59, 59, tzinfo=timezone.utc)


def _flag(con, model_version, player_uid="p1"):
    return mm.role_change_evidence_flag(
        con, model_version, player_uid, _ASOF,
        decay_params_version=1, fact_multiplier_params_version=1, flag_params_version=1,
    )


def test_seed_role_change_flag_params_is_versioned_and_explicit_version_only(con):
    mm.seed_role_change_flag_params(con)
    assert params.resolve_param(con, "role_change_flag_params", "min_p_start_final", 1)[0] == 0.75
    assert params.resolve_param(con, "role_change_flag_params", "min_weight_own", 1)[0] == 0.60
    assert params.resolve_param(con, "role_change_flag_params", "recent_transfer_lookback_days", 1)[0] == 140.0
    # resolve is explicit-version-only -- an unseeded version must hard-error, never default
    with pytest.raises(params.ParamNotFoundError):
        params.resolve_param(con, "role_change_flag_params", "min_p_start_final", 2)


def test_role_change_flag_fires_for_confident_mover_with_completed_transfer_claim(con):
    mv = _seed_role_change_scenario(
        con, p_start_final=0.95, weight_own=1.0,
        claims=[{"claim_type": "transfer_likelihood", "claim_value_numeric": 1.0,
                 "claim_value": {"status": "Complete", "old_club": "Forest", "new_club": "Man City"}}],
    )
    flag = _flag(con, mv)
    assert flag is not None
    assert flag["flag"] == "role_change_evidence_unvalidated"
    assert flag["passed"] is False
    assert "transfer_likelihood" in flag["signal_claim_types"]
    assert "optimistic" in flag["message"] and "un-validated" in flag["message"]
    # provenance is real: the reported effective_weight matches an independent recompute
    from fpl_quant import evidence_blend as eb
    from fpl_quant import snapshot as snap
    c = snap.get_claims_asof(con, _ASOF, subject_entity_type="player", subject_entity_id="p1",
                             claim_type="transfer_likelihood").to_dict("records")[0]
    assert flag["signals"][0]["effective_weight"] == pytest.approx(
        eb.effective_weight(con, c, _ASOF, 1, 1)
    )


def test_role_change_flag_fires_on_predicted_xi_club_correction_text(con):
    mv = _seed_role_change_scenario(
        con, p_start_final=0.94, weight_own=1.0,
        claims=[{"claim_type": "predicted_xi", "claim_value_numeric": 0.55,
                 "claim_value": {"predicted_starter": "Yes",
                                 "notes": "CLUB CORRECTION: transferred from Forest to Man City; role still bedding in."}}],
    )
    flag = _flag(con, mv)
    assert flag is not None
    assert flag["signal_claim_types"] == ["predicted_xi"]
    assert flag["signals"][0]["detection"] == "text"


def test_role_change_flag_fires_on_new_position_manager_tendency_with_zero_minutes_effect(con):
    # a `new_position` RoleChange row routes to manager_tendency valence "note" -> sign 0 ->
    # zero effect in compute_logit_adjustment; the flag must still surface it (structural).
    mv = _seed_role_change_scenario(
        con, p_start_final=0.83, weight_own=1.0,
        claims=[{"claim_type": "manager_tendency",
                 "claim_value": {"change": "new_position", "valence": "note",
                                 "cause": "confirmed as first-choice creative hub"}}],
    )
    flag = _flag(con, mv)
    assert flag is not None
    assert flag["signals"][0]["detection"] == "structural"


def test_role_change_flag_silent_when_projection_not_confident(con):
    mv = _seed_role_change_scenario(
        con, p_start_final=0.50, weight_own=1.0,
        claims=[{"claim_type": "transfer_likelihood", "claim_value_numeric": 1.0,
                 "claim_value": {"status": "Complete", "old_club": "Forest", "new_club": "Man City"}}],
    )
    assert _flag(con, mv) is None


def test_role_change_flag_silent_when_history_not_dominant(con):
    mv = _seed_role_change_scenario(
        con, p_start_final=0.90, weight_own=0.20,
        claims=[{"claim_type": "transfer_likelihood", "claim_value_numeric": 1.0,
                 "claim_value": {"status": "Complete", "old_club": "Forest", "new_club": "Man City"}}],
    )
    assert _flag(con, mv) is None


def test_role_change_flag_silent_without_a_role_change_signal(con):
    # confident + history-dominant, but the only evidence is a same-club injury update
    mv = _seed_role_change_scenario(
        con, p_start_final=0.95, weight_own=1.0,
        claims=[{"claim_type": "predicted_xi", "claim_value_numeric": 0.9,
                 "claim_value": {"predicted_starter": "Yes", "notes": "nailed, played 90 last week"}}],
    )
    assert _flag(con, mv) is None


def test_role_change_flag_has_no_player_specific_branching(con):
    # identical setup under two different player identities -> identical flag (modulo uid)
    mv_a = _seed_role_change_scenario(
        con, p_start_final=0.95, weight_own=1.0, player_uid="elliot_anderson", canonical_name="Elliot Anderson",
        claims=[{"claim_type": "transfer_likelihood", "claim_value_numeric": 1.0,
                 "claim_value": {"status": "Complete", "old_club": "Forest", "new_club": "Man City"}}],
    )
    mv_b = _seed_role_change_scenario(
        con, p_start_final=0.95, weight_own=1.0, player_uid="generic_player", canonical_name="Generic Player",
        claims=[{"claim_type": "transfer_likelihood", "claim_value_numeric": 1.0,
                 "claim_value": {"status": "Complete", "old_club": "Forest", "new_club": "Man City"}}],
        seed_flag_params=False,
    )
    fa = _flag(con, mv_a, "elliot_anderson")
    fb = _flag(con, mv_b, "generic_player")
    assert fa is not None and fb is not None
    fa_norm = {k: v for k, v in fa.items() if k not in ("player_uid", "signals")}
    fb_norm = {k: v for k, v in fb.items() if k not in ("player_uid", "signals")}
    assert fa_norm == fb_norm
    # and no player identity is hard-coded in the detector (a `== "<name>"` style branch)
    import inspect
    src = (
        inspect.getsource(mm.role_change_evidence_flag)
        + inspect.getsource(mm._claim_text_blob)
        + inspect.getsource(mm.seed_role_change_flag_params)
        + mm._ROLE_CHANGE_TEXT_RE.pattern
    ).lower()
    for banned in ("anderson", "elliot", "tzolis", "haaland", "== \"player", "== 'player"):
        assert banned not in src


def test_role_change_flag_does_not_alter_the_logit_adjustment_or_its_clip(con):
    # The flag is pure advisory: a role-change predicted_xi claim still gets the SAME pull
    # and the SAME global +/-6.0 clip as any other predicted_xi claim -- assert the clip math
    # directly and that it is untouched by this feature.
    _seed_role_change_scenario(
        con, p_start_final=0.95, weight_own=1.0,
        claims=[{"claim_type": "predicted_xi", "claim_value_numeric": 0.99,
                 "claim_value": {"notes": "CLUB CORRECTION: new signing"}}],
    )
    params.write_param(con, "minutes_adjustment_params", 1, "2026-08-10", "magnitude",
                       value_numeric=0.8, dimensions={"claim_type": "predicted_xi"})
    params.write_param(con, "minutes_adjustment_params", 1, "2026-08-10", "cap",
                       value_numeric=6.0, dimensions={"scope": "global"})
    # a huge pull target vs a tiny base -> unclipped total would exceed the cap; assert it clips
    adj_low_base = mm.compute_logit_adjustment(
        con, "p1", 0.001, _ASOF, adjustment_params_version=1, decay_params_version=1, fact_multiplier_params_version=1,
    )
    assert -6.0 <= adj_low_base <= 6.0
    # symmetric clip
    params.write_param(con, "minutes_adjustment_params", 2, "2026-08-10", "magnitude",
                       value_numeric=50.0, dimensions={"claim_type": "predicted_xi"})
    params.write_param(con, "minutes_adjustment_params", 2, "2026-08-10", "cap",
                       value_numeric=6.0, dimensions={"scope": "global"})
    adj_capped = mm.compute_logit_adjustment(
        con, "p1", 0.5, _ASOF, adjustment_params_version=2, decay_params_version=1, fact_multiplier_params_version=1,
    )
    assert adj_capped == pytest.approx(6.0)
