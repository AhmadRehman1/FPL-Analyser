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


# ============================================================
# 2026-09-15 fix: a real, live goalkeeper who started every match this season (4/4) projected
# p_start_final=0.13 because target_season's own already-played matches never entered
# p_start_historical_own's computation at all -- only complete PRIOR seasons did. Two pieces:
# lookback_seasons' new default (provably backtest-neutral, see run()'s own docstring) and the
# opt-in current_season_role_params_version fast-reacting blend (a real behavior change,
# requires its own walk-forward evidence before defaulting on).
# ============================================================

def _seed_league_with_current_season(con):
    """Extends _seed_league(): the same 2-season A vs B league (p1 nailed, p2 never features),
    PLUS a third, "live" season (2026-2027) with one more player, p3 -- a goalkeeper who was a
    bench/fringe player across the two historical seasons (started only 2 of 20 team matches,
    same shape a real backup keeper's history would have) but has started every one of his
    team's 4 already-played 2026-2027 matches -- the real Antonín Kinský shape, generalized."""
    _seed_league(con)
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p3', 'Player Three', 'Goalkeeper')")
    now = datetime.now(timezone.utc)
    match_i = 1000
    for season in ("2024-2025", "2025-2026"):
        con.execute(
            "INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) "
            "VALUES ('Player Three', 'player three', '1', ?, 'p3')", [season],
        )
        for i in range(20):
            match_id = f"m{match_i}"
            match_i += 1
            base = datetime(2025, 1, 1) if season == "2024-2025" else datetime(2026, 1, 1)
            kickoff = base + i * (datetime(2025, 1, 8) - datetime(2025, 1, 1))  # one week apart, real gameweeks
            con.execute(
                "INSERT INTO fact_match (match_id, season, gameweek, home_team_uid, away_team_uid, finished, "
                "competition, kickoff_time, _ingested_at) VALUES (?, ?, ?, 'team_a', 'team_b', TRUE, "
                "'Premier League', ?, ?)",
                [match_id, season, i + 1, kickoff, now],
            )
            started = i < 2  # bench/fringe: only 2 of 20 historical team matches started
            con.execute(
                "INSERT INTO fact_player_match_stats (player_uid, match_id, season, start_min, "
                "finish_min, minutes_played, _ingested_at) VALUES ('p3', ?, ?, ?, ?, ?, ?)",
                [match_id, season, 0 if started else 200, 90 if started else 0, 90 if started else 0, now],
            )
    # live 2026-2027 season: p3 has started every one of the 4 matches played so far.
    _seed_raw_teams_csv(con, "2026-2027", [("1", "A"), ("2", "B")])
    con.execute(
        "INSERT INTO team_alias (alias_name, season, team_uid, alias_source) VALUES ('A', '2026-2027', 'team_a', 't')"
    )
    con.execute(
        "INSERT INTO team_alias (alias_name, season, team_uid, alias_source) VALUES ('B', '2026-2027', 'team_b', 't')"
    )
    con.execute(
        "INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) "
        "VALUES ('Player Three', 'player three', '1', '2026-2027', 'p3')",
    )
    con.execute(
        "INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) "
        "VALUES ('Player One', 'player one', '1', '2026-2027', 'p1')",
    )
    for i in range(4):
        match_id = f"m2027_{i}"
        con.execute(
            "INSERT INTO fact_match (match_id, season, gameweek, home_team_uid, away_team_uid, finished, "
            "competition, kickoff_time, _ingested_at) VALUES (?, '2026-2027', ?, 'team_a', 'team_b', TRUE, "
            "'Premier League', ?, ?)",
            [match_id, i + 1, datetime(2026, 8, 16) + i * (datetime(2026, 8, 23) - datetime(2026, 8, 16)), now],
        )
        con.execute(
            "INSERT INTO fact_player_match_stats (player_uid, match_id, season, start_min, "
            "finish_min, minutes_played, _ingested_at) VALUES ('p3', ?, '2026-2027', 0, 90, 90, ?)",
            [match_id, now],
        )


def _write_base_params(con):
    params.write_param(con, "minutes_model_decay_params", 1, "2026-08-10", "xi", value_numeric=0.0018)
    params.write_param(con, "minutes_adjustment_params", 1, "2026-08-10", "cap", value_numeric=6.0, dimensions={"scope": "global"})
    params.write_param(con, "minutes_model_shrinkage_params", 1, "2026-08-10", "competitive_matches_threshold", value_numeric=10)


def test_run_default_lookback_seasons_now_includes_target_season(con):
    _seed_league_with_current_season(con)
    _write_base_params(con)
    # No lookback_seasons kwarg -- exercises the new default directly.
    model_version = mm.run(
        con, date(2026, 9, 15), "2026-2027",
        decay_params_version=1, adjustment_params_version=1,
        shrinkage_params_version=1, fact_multiplier_params_version=1,
    )
    row = con.execute(
        "SELECT p_start_historical_own, competitive_matches_last_2_seasons, weight_own, p_start_final "
        "FROM minutes_model_outputs WHERE model_version = ? AND player_uid = 'p3'", [model_version],
    ).fetchone()
    p_start_own, competitive_matches, weight_own, p_start_final = row
    # competitive_matches_last_2_seasons counts p3's OWN appearances (2 + 2 + 4 = 8), not team
    # matches -- was 4 (2+2, entirely missing the 4 real current-season starts) before this fix.
    assert competitive_matches == 8
    # Recency-weighted own start rate must have moved UP from the pre-fix (2/40 = 0.05) shape --
    # loose bound, not pinned to an exact float (real xi decay math), but must be a real,
    # material shift, not noise.
    assert p_start_own > 0.10


def test_run_explicit_old_two_season_lookback_reproduces_prior_behavior(con):
    """Backward-compat: a caller that still explicitly passes the OLD 2-season tuple (exactly
    what every real caller in this codebase used to rely on via the default) must see p3's
    4 real current-season starts contribute NOTHING -- the exact prior bug, still reproducible
    on demand, proving this is the default that changed, not compute_player_historical_
    components() itself."""
    _seed_league_with_current_season(con)
    _write_base_params(con)
    model_version = mm.run(
        con, date(2026, 9, 15), "2026-2027",
        decay_params_version=1, adjustment_params_version=1,
        shrinkage_params_version=1, fact_multiplier_params_version=1,
        lookback_seasons=("2024-2025", "2025-2026"),
    )
    competitive_matches = con.execute(
        "SELECT competitive_matches_last_2_seasons FROM minutes_model_outputs "
        "WHERE model_version = ? AND player_uid = 'p3'", [model_version],
    ).fetchone()[0]
    assert competitive_matches == 4  # only the 2+2 historical appearances -- the 4 live 2026-2027 starts are invisible


def test_run_default_lookback_seasons_is_backtest_neutral_for_a_completed_season(con):
    """The real safety claim: for a target_season the walk-forward actually exercises
    (2025-2026), asof_scope()'s own fact_match shadow makes every 2026-2027 row structurally
    invisible regardless of what's in lookback_seasons -- so this fix must produce BYTE-
    IDENTICAL minutes_model_outputs whether or not "2026-2027" is in the tuple, wrapped in the
    same asof_scope() every real backtest caller already uses."""
    from fpl_quant import backtest as bt

    _seed_league_with_current_season(con)
    _write_base_params(con)
    with bt.asof_scope(con, "2025-2026", 20):
        mv_new_default = mm.run(
            con, date(2026, 6, 1), "2025-2026",
            decay_params_version=1, adjustment_params_version=1,
            shrinkage_params_version=1, fact_multiplier_params_version=1,
            lookback_seasons=("2024-2025", "2025-2026", "2026-2027"),
        )
        mv_old_default = mm.run(
            con, date(2026, 6, 1), "2025-2026",
            decay_params_version=1, adjustment_params_version=1,
            shrinkage_params_version=1, fact_multiplier_params_version=1,
            lookback_seasons=("2024-2025", "2025-2026"),
        )
        rows_new = con.execute(
            "SELECT player_uid, p_start_final, competitive_matches_last_2_seasons FROM minutes_model_outputs "
            "WHERE model_version = ? ORDER BY player_uid", [mv_new_default],
        ).fetchall()
        rows_old = con.execute(
            "SELECT player_uid, p_start_final, competitive_matches_last_2_seasons FROM minutes_model_outputs "
            "WHERE model_version = ? ORDER BY player_uid", [mv_old_default],
        ).fetchall()
    # approx, not exact equality: DuckDB's SUM() can accumulate in a different internal order
    # when the season IN (...) list has an extra, zero-row-contributing entry, producing a
    # last-bit-level float difference -- not a real behavioral difference (competitive_matches,
    # an exact integer count, IS asserted for byte-identical equality below).
    assert [r[0] for r in rows_new] == [r[0] for r in rows_old]
    for (uid_n, p_n, cm_n), (uid_o, p_o, cm_o) in zip(rows_new, rows_old):
        assert uid_n == uid_o
        assert p_n == pytest.approx(p_o, abs=1e-9)
        assert cm_n == cm_o


def test_current_season_role_params_off_by_default(con):
    _seed_league_with_current_season(con)
    _write_base_params(con)
    model_version = mm.run(
        con, date(2026, 9, 15), "2026-2027",
        decay_params_version=1, adjustment_params_version=1,
        shrinkage_params_version=1, fact_multiplier_params_version=1,
    )
    p_start_final = con.execute(
        "SELECT p_start_final FROM minutes_model_outputs WHERE model_version = ? AND player_uid = 'p3'",
        [model_version],
    ).fetchone()[0]
    # Fix A alone (no current_season_role_params_version): p3's 4 real current-season starts
    # are visible to the multi-season blend but still heavily diluted by 40 historical
    # matches at only ~10% own rate -- must NOT already look like a nailed starter.
    assert p_start_final < 0.5


def test_current_season_role_params_fixes_the_real_incident(con):
    """The actual fix: with current_season_role_params_version opted in, a player who has
    started every one of his team's matches so far this season (p3's real, generalized
    Kinský shape) must project as a genuinely nailed starter, not a 13%-start-probability
    bench option."""
    _seed_league_with_current_season(con)
    _write_base_params(con)
    mm.seed_current_season_role_params(con)  # v1: current_season_matches_threshold=4
    model_version = mm.run(
        con, date(2026, 9, 15), "2026-2027",
        decay_params_version=1, adjustment_params_version=1,
        shrinkage_params_version=1, fact_multiplier_params_version=1,
        current_season_role_params_version=1,
    )
    p_start_final = con.execute(
        "SELECT p_start_final FROM minutes_model_outputs WHERE model_version = ? AND player_uid = 'p3'",
        [model_version],
    ).fetchone()[0]
    assert p_start_final > 0.85  # started 4/4 of a threshold=4 window -> current season fully dominates


def test_current_season_role_params_does_not_affect_players_without_current_season_matches(con):
    """p1 (the historical nailed starter) never got a 2026-2027 player_alias row seeded for
    this specific check -- current_season_role_params_version must fall back to the exact
    unmodified multi-season blend for anyone with no current-season data at all, same "absence
    of coverage isn't evidence of anything" convention this module already uses elsewhere."""
    _seed_league(con)
    _write_base_params(con)
    mm.seed_current_season_role_params(con)
    mv_off = mm.run(
        con, date(2026, 6, 1), "2025-2026",
        decay_params_version=1, adjustment_params_version=1,
        shrinkage_params_version=1, fact_multiplier_params_version=1,
        lookback_seasons=("2024-2025", "2025-2026"),
    )
    mv_on = mm.run(
        con, date(2026, 6, 1), "2025-2026",
        decay_params_version=1, adjustment_params_version=1,
        shrinkage_params_version=1, fact_multiplier_params_version=1,
        lookback_seasons=("2024-2025", "2025-2026"), current_season_role_params_version=1,
    )
    p1_off = con.execute("SELECT p_start_final FROM minutes_model_outputs WHERE model_version = ? AND player_uid = 'p1'", [mv_off]).fetchone()[0]
    p1_on = con.execute("SELECT p_start_final FROM minutes_model_outputs WHERE model_version = ? AND player_uid = 'p1'", [mv_on]).fetchone()[0]
    assert p1_off == pytest.approx(p1_on)


def test_seed_current_season_role_params_is_idempotent(con):
    mm.seed_current_season_role_params(con)
    mm.seed_current_season_role_params(con)  # must not raise on a byte-identical re-write
    value, _ = params.resolve_param(con, "current_season_role_params", "current_season_matches_threshold", 1)
    assert value == 4


def _seed_prices(con, prices, season="2025-2026"):
    now = datetime.now(timezone.utc)
    for uid, cost in prices.items():
        con.execute(
            "INSERT INTO fact_player_season_stats (player_uid, season, gw, now_cost, _ingested_at) VALUES (?, ?, 30, ?, ?)",
            [uid, season, cost, now],
        )


def test_price_band_matches_the_walk_forward_bands():
    assert [mm.price_band(c) for c in (4.5, 5.0, 6.9, 7.0, 9.0, None)] == [
        "<5.0", "5.0-7.0", "5.0-7.0", "7.0-9.0", "9.0+", None,
    ]


def test_latest_price_by_player_takes_the_newest_row(con):
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p1', 'P1', 'Midfielder')")
    _seed_prices(con, {"p1": 9.0}, season="2024-2025")
    _seed_prices(con, {"p1": 9.5})
    assert mm.latest_price_by_player(con) == {"p1": 9.5}


def test_price_prior_pulls_a_new_cheap_player_toward_his_band_not_the_position(con):
    """p1 (9.5m) starts every match, p2 (4.5m) never features. A brand-new 4.5m midfielder
    with no history used to inherit the position average (p1's and p2's pooled,
    recency-weighted start rate); with the price prior he inherits the <5.0 band's rate instead."""
    _seed_league(con)
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p3', 'New Kid', 'Midfielder')")
    con.execute(
        "INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) "
        "VALUES ('New Kid', 'new kid', '1', '2025-2026', 'p3')"
    )
    _seed_prices(con, {"p1": 9.5, "p2": 4.5, "p3": 4.5})
    params.write_param(con, "minutes_model_decay_params", 1, "2026-08-10", "xi", value_numeric=0.0018)
    params.write_param(con, "minutes_adjustment_params", 1, "2026-08-10", "cap", value_numeric=6.0, dimensions={"scope": "global"})
    params.write_param(con, "minutes_model_shrinkage_params", 1, "2026-08-10", "competitive_matches_threshold", value_numeric=10)
    params.write_param(con, "minutes_price_prior_params", 1, "2026-08-10", "min_band_weight", value_numeric=0.0)

    def p_start_hist(**kw):
        mv = mm.run(
            con, date(2026, 8, 10), "2025-2026", decay_params_version=1, adjustment_params_version=1,
            shrinkage_params_version=1, fact_multiplier_params_version=1,
            lookback_seasons=("2024-2025", "2025-2026"), **kw,
        )
        return dict(con.execute(
            "SELECT player_uid, p_start_historical_final FROM minutes_model_outputs WHERE model_version = ?", [mv],
        ).fetchall())

    off = p_start_hist()
    on = p_start_hist(price_prior_params_version=1)
    assert off["p3"] > 0.3  # the recency-weighted midfielder average
    assert on["p3"] == pytest.approx(0.0, abs=1e-9)
    assert on["p1"] == pytest.approx(off["p1"])  # a full own history is unaffected


def test_price_band_priors_skip_thin_bands(con):
    _seed_league(con)
    _seed_prices(con, {"p1": 9.5, "p2": 4.5})
    per_player = mm.compute_player_historical_components(con, ("2024-2025", "2025-2026"), date(2026, 8, 10), 0.0018)
    prices = mm.latest_price_by_player(con)
    assert set(mm.compute_price_band_start_priors(con, per_player, prices, 0.0)) == {
        ("Midfielder", "9.0+"), ("Midfielder", "<5.0"),
    }
    assert mm.compute_price_band_start_priors(con, per_player, prices, 1e9) == {}


# ============================================================
# Evidence-order start prior (docs/reports/2026-10_open_issues.md, issue 4): the player's own
# record this season, then in earlier seasons, then a start rate rising with price.
# ============================================================

def _curve_inputs(rates_by_price, position="Midfielder", players_per_price=5, matches=20.0):
    import pandas as pd

    rows, prices, position_of = [], {}, {}
    for price, rate in rates_by_price.items():
        for i in range(players_per_price):
            uid = f"{position}_{price}_{i}"
            rows.append({"player_uid": uid, "weighted_starts": rate * matches, "weighted_total": matches})
            prices[uid], position_of[uid] = price, position
    return pd.DataFrame(rows), position_of, prices


def test_price_curve_rises_steadily_with_price_and_puts_premiums_high():
    """The 9.0+ prices include a rotated/injured-looking 0.55: a 9.0+ band average sits well
    under 0.85, the curve does not."""
    rates = {4.0: 0.05, 4.5: 0.1, 5.0: 0.2, 5.5: 0.35, 6.0: 0.45, 7.0: 0.6, 8.0: 0.7, 9.0: 0.55, 10.5: 0.9, 12.5: 0.95}
    per_player, position_of, prices = _curve_inputs(rates)
    curve = mm.fit_price_start_curve(per_player, position_of, prices)
    assert set(curve) == {"Midfielder"} and curve["Midfielder"][1] > 0
    grid = [mm.price_curve_start_prior(curve, "Midfielder", p) for p in (4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.5)]
    assert grid == sorted(grid)
    assert mm.price_curve_start_prior(curve, "Midfielder", 10.5) >= 0.85
    assert mm.price_curve_start_prior(curve, "Midfielder", 4.0) <= 0.15
    band_9_plus = (0.55 + 0.9 + 0.95) / 3
    assert band_9_plus < 0.85


def test_price_curve_stays_finite_when_every_premium_started():
    """A cold start's first week separates perfectly (every dear player started, no cheap one
    did); the ridge keeps the slope finite and the bounds keep the prior off 1."""
    per_player, position_of, prices = _curve_inputs({4.5: 0.0, 10.0: 1.0}, players_per_price=10, matches=1.0)
    curve = mm.fit_price_start_curve(per_player, position_of, prices)
    intercept, slope = curve["Midfielder"]
    assert abs(intercept) < 50 and 0 < slope < 50
    assert 0.85 <= mm.price_curve_start_prior(curve, "Midfielder", 10.0) <= mm.PRICE_CURVE_BOUNDS[1]
    assert mm.price_curve_start_prior(curve, "Midfielder", 4.5) <= 0.15


def test_price_curve_never_falls_with_price_and_skips_thin_positions():
    per_player, position_of, prices = _curve_inputs({4.5: 0.8, 6.0: 0.5, 9.0: 0.2})
    curve = mm.fit_price_start_curve(per_player, position_of, prices)
    assert curve["Midfielder"][1] == 0.0
    flat = {mm.price_curve_start_prior(curve, "Midfielder", p) for p in (4.5, 9.0)}
    assert len(flat) == 1 and abs(flat.pop() - 0.5) < 0.01  # the pooled rate
    thin, position_of, prices = _curve_inputs({5.0: 0.5}, position="Goalkeeper", players_per_price=1, matches=10.0)
    assert mm.fit_price_start_curve(thin, position_of, prices) == {}
    assert mm.price_curve_start_prior(curve, "Goalkeeper", 5.0) is None
    assert mm.price_curve_start_prior(curve, "Midfielder", None) is None


def test_record_start_rate_counts_the_base_as_extra_matches():
    import pandas as pd

    assert mm.record_start_rate(None, 0.4, 5.0) == 0.4
    assert mm.record_start_rate(pd.Series({"weighted_starts": 0.0, "weighted_total": 0.0}), 0.4, 5.0) == 0.4
    record = pd.Series({"weighted_starts": 2.0, "weighted_total": 10.0})
    assert mm.record_start_rate(record, 0.5, 5.0) == pytest.approx((2.0 + 2.5) / 15.0)


def _seed_league_into_2026_27(con):
    """_seed_league()'s two seasons (p1, 9.5m, starts all 20; p2, 4.5m, never features), then
    2026-27 with two matches played: p1 starts both, p2 again neither, 'rookie' (9.0m, no
    earlier record) starts both, and two arrivals with no record at all, 'star' (12.0m) and
    'kid' (4.5m)."""
    _seed_league(con)
    now = datetime.now(timezone.utc)
    _seed_raw_teams_csv(con, "2026-2027", [("1", "A"), ("2", "B")])
    for name, uid in (("A", "team_a"), ("B", "team_b")):
        con.execute("INSERT INTO team_alias (alias_name, season, team_uid, alias_source) VALUES (?, '2026-2027', ?, 't')", [name, uid])
    for uid in ("rookie", "star", "kid"):
        con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, 'Midfielder')", [uid, uid])
    for uid in ("p1", "p2", "rookie", "star", "kid"):
        con.execute(
            "INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) "
            "VALUES (?, ?, '1', '2026-2027', ?)", [uid, uid, uid],
        )
    first_gw = {"p1": 1, "p2": 1, "rookie": 1, "star": 3, "kid": 3}  # star and kid arrive after GW2
    for uid, cost in {"p1": 9.5, "p2": 4.5, "rookie": 9.0, "star": 12.0, "kid": 4.5}.items():
        con.execute(
            "INSERT INTO fact_player_season_stats (player_uid, season, gw, now_cost, _ingested_at) "
            "VALUES (?, '2026-2027', ?, ?, ?)", [uid, first_gw[uid], cost, now],
        )
    for i in range(2):
        match_id = f"m2027_{i}"
        con.execute(
            "INSERT INTO fact_match (match_id, season, gameweek, home_team_uid, away_team_uid, finished, "
            "competition, kickoff_time, _ingested_at) VALUES (?, '2026-2027', ?, 'team_a', 'team_b', TRUE, "
            "'Premier League', ?, ?)",
            [match_id, i + 1, datetime(2026, 8, 16) + i * (datetime(2026, 8, 23) - datetime(2026, 8, 16)), now],
        )
        for uid in ("p1", "rookie"):
            con.execute(
                "INSERT INTO fact_player_match_stats (player_uid, match_id, season, start_min, finish_min, "
                "minutes_played, _ingested_at) VALUES (?, ?, '2026-2027', 0, 90, 90, ?)", [uid, match_id, now],
            )


def test_start_prior_follows_the_order_of_evidence(con):
    _seed_league_into_2026_27(con)
    _write_base_params(con)
    params.write_param(con, "minutes_start_prior_params", 1, "2026-10-04", "pseudo_matches", value_numeric=5.0)

    def outputs(**kw):
        mv = mm.run(
            con, date(2026, 9, 1), "2026-2027", decay_params_version=1, adjustment_params_version=1,
            shrinkage_params_version=1, fact_multiplier_params_version=1, **kw,
        )
        return {
            uid: {"prior": prior, "hist": hist}
            for uid, prior, hist in con.execute(
                "SELECT player_uid, p_start_historical_position_avg, p_start_historical_final "
                "FROM minutes_model_outputs WHERE model_version = ?", [mv],
            ).fetchall()
        }

    off, on = outputs(), outputs(start_prior_params_version=1)
    # no record anywhere: the price curve alone, high for a premium, low for a cheap arrival
    assert on["star"]["hist"] >= 0.85 and on["kid"]["hist"] <= 0.15
    assert off["star"]["hist"] == off["kid"]["hist"]  # the position average, whatever the price
    # a current-season record (2 starts of 2) leads, the curve only steadies it
    assert on["rookie"]["hist"] >= 0.85
    # never picked across 22 available matches: his own record, not the position average
    assert on["p2"]["hist"] < 0.05 < 0.3 < off["p2"]["hist"]
    # a full history is untouched
    assert on["p1"]["hist"] == pytest.approx(off["p1"]["hist"])


def test_the_two_price_priors_are_alternatives(con):
    _seed_league(con)
    _write_base_params(con)
    params.write_param(con, "minutes_price_prior_params", 1, "2026-08-10", "min_band_weight", value_numeric=0.0)
    params.write_param(con, "minutes_start_prior_params", 1, "2026-10-04", "pseudo_matches", value_numeric=5.0)
    with pytest.raises(ValueError, match="alternative"):
        mm.run(
            con, date(2026, 8, 10), "2025-2026", decay_params_version=1, adjustment_params_version=1,
            shrinkage_params_version=1, fact_multiplier_params_version=1,
            lookback_seasons=("2024-2025", "2025-2026"), price_prior_params_version=1, start_prior_params_version=1,
        )


def test_current_season_minutes_blend_lifts_a_new_90_minute_starter(con):
    """docs/plans/2026-10_breakout_players.md, arm A4: P(60+ | started) from every season keeps
    a player taken off early last season low after he becomes a 90-minute starter."""
    _seed_league(con)
    now = datetime.now(timezone.utc)
    for i in range(20):  # _seed_league: m0-m9 are 2024-25, m10-m19 2025-26
        season, minutes = ("2024-2025", 45) if i < 10 else ("2025-2026", 90)
        con.execute(
            "INSERT INTO fact_player_match_stats (player_uid, match_id, season, start_min, finish_min, "
            "minutes_played, _ingested_at) VALUES ('p2', ?, ?, 0, ?, ?, ?)",
            [f"m{i}", season, minutes, minutes, now],
        )
    params.write_param(con, "minutes_model_decay_params", 1, "2026-08-10", "xi", value_numeric=0.0018)
    params.write_param(con, "minutes_adjustment_params", 1, "2026-08-10", "cap", value_numeric=6.0, dimensions={"scope": "global"})
    params.write_param(con, "minutes_model_shrinkage_params", 1, "2026-08-10", "competitive_matches_threshold", value_numeric=10)
    params.write_param(con, "current_season_minutes_params", 1, "2026-10-08", "starts_threshold", value_numeric=4)
    run = dict(
        decay_params_version=1, adjustment_params_version=1, shrinkage_params_version=1,
        fact_multiplier_params_version=1, lookback_seasons=("2024-2025", "2025-2026"),
    )
    base = mm.run(con, date(2026, 8, 10), "2025-2026", **run)
    blended = mm.run(con, date(2026, 8, 10), "2025-2026", current_season_minutes_params_version=1, **run)

    def p60(model_version, uid):
        return con.execute(
            "SELECT p_60plus_min FROM minutes_model_outputs WHERE model_version = ? AND player_uid = ?",
            [model_version, uid],
        ).fetchone()[0]

    # all-season P(60+ | started) is 10/20; this season's 10 starts of 90 take it to 1.0
    assert p60(blended, "p2") > p60(base, "p2") + 0.2
    # an established 90-minute starter is unchanged
    assert p60(blended, "p1") == pytest.approx(p60(base, "p1"))

    # the gentler form: this season's 10 starts against 4 pseudo-starts at the all-season rate
    params.write_param(con, "current_season_minutes_params", 2, "2026-10-09", "pseudo_starts", value_numeric=4)
    shrunk = mm.run(con, date(2026, 8, 10), "2025-2026", current_season_minutes_params_version=2, **run)
    assert p60(base, "p2") + 0.1 < p60(shrunk, "p2") < p60(blended, "p2") - 0.05
    assert p60(shrunk, "p1") == pytest.approx(p60(base, "p1"))


def test_league_only_minutes_rates_ignore_cup_matches(con):
    """The breakout report's arm A6: from 2025-26 fact_player_match_stats also holds cups; a
    regular taken off at 45 in the cup must not lower his league P(60+)."""
    _seed_league(con)
    params.write_param(con, "minutes_model_decay_params", 1, "2026-08-10", "xi", value_numeric=0.0018)
    params.write_param(con, "minutes_adjustment_params", 1, "2026-08-10", "cap", value_numeric=6.0, dimensions={"scope": "global"})
    params.write_param(con, "minutes_model_shrinkage_params", 1, "2026-08-10", "competitive_matches_threshold", value_numeric=10)
    params.write_param(con, "minutes_rates_scope_params", 1, "2026-10-09", "league_only", value_numeric=1)
    run = dict(
        decay_params_version=1, adjustment_params_version=1, shrinkage_params_version=1,
        fact_multiplier_params_version=1, lookback_seasons=("2024-2025", "2025-2026"),
    )

    def p60(model_version, uid="p1"):
        return con.execute(
            "SELECT p_60plus_min FROM minutes_model_outputs WHERE model_version = ? AND player_uid = ?",
            [model_version, uid],
        ).fetchone()[0]

    league = p60(mm.run(con, date(2026, 8, 10), "2025-2026", **run))
    now = datetime.now(timezone.utc)
    for i in range(6):  # p1 starts six cup ties and comes off at 45
        con.execute(
            "INSERT INTO fact_match (match_id, season, home_team_uid, away_team_uid, finished, competition, "
            "kickoff_time, _ingested_at) VALUES (?, '2025-2026', 'team_a', 'team_b', TRUE, 'EFL Cup', ?, ?)",
            [f"cup{i}", datetime(2026, 1, 2), now],
        )
        con.execute(
            "INSERT INTO fact_player_match_stats (player_uid, match_id, season, start_min, finish_min, "
            "minutes_played, _ingested_at) VALUES ('p1', ?, '2025-2026', 0, 45, 45, ?)",
            [f"cup{i}", now],
        )
    every_competition = p60(mm.run(con, date(2026, 8, 10), "2025-2026", **run))
    league_only = p60(mm.run(con, date(2026, 8, 10), "2025-2026", minutes_rates_scope_params_version=1, **run))
    assert every_competition < league - 0.05
    assert league_only == pytest.approx(league)
