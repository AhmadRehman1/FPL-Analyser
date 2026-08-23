import pytest
from scipy.stats import poisson

from fpl_quant import expected_points as ep


def test_seed_v1_params_resolves_expected_values(con):
    ep.seed_v1_params(con)
    assert ep._sm(con, "goal_points", 1, "Forward") == 4
    assert ep._sm(con, "goal_points", 1, "Goalkeeper") == 10
    assert ep._sm(con, "clean_sheet_points", 1, "Midfielder") == 1
    assert ep._sm(con, "defcon_threshold", 1, "Defender") == 10
    assert ep._sm(con, "defcon_threshold", 1, "Midfielder") == 12
    assert ep._bp(con, "cbi_per_point", 1) == 3.0  # 2026/27 change from 1-per-2
    assert ep._bp(con, "being_tackled", 1) == 0.0  # penalty removed for 2026/27
    assert ep._bp(con, "penalty_save", 1) == 7  # reduced from 8


def test_shrink_rate_pure_position_average_at_zero_sample():
    result = ep._shrink_rate(own_rate=5.0, sample_minutes=0, position_avg_rate=0.3)
    assert result == 0.3


def test_shrink_rate_mostly_own_rate_at_large_sample():
    result = ep._shrink_rate(own_rate=0.8, sample_minutes=5000, position_avg_rate=0.3)
    assert abs(result - 0.8) < 0.05


def test_shrink_rate_regression_two_minute_sample_stays_near_position_average():
    """The exact real bug this project hit: a 2-minute sample with one lucky xG
    contribution must not dominate the blended rate."""
    result = ep._shrink_rate(own_rate=3.6, sample_minutes=2, position_avg_rate=0.35)
    assert result < 0.4  # nowhere near the noisy 3.6 extrapolation


def test_expected_floor_half_matches_direct_enumeration():
    lam = 1.3
    expected = sum((k // 2) * poisson.pmf(k, lam) for k in range(30))
    assert abs(ep._expected_floor_half(lam) - expected) < 1e-9


def test_expected_floor_half_zero_at_zero_lambda():
    assert ep._expected_floor_half(0.0) == 0.0


def test_plackett_luce_three_players_sums_to_six():
    strengths = {"a": 3.0, "b": 2.0, "c": 1.0}
    bonus = ep.plackett_luce_bonus(strengths)
    assert abs(sum(bonus.values()) - 6.0) < 1e-9


def test_plackett_luce_many_players_still_sums_to_six():
    strengths = {f"p{i}": float(i + 1) for i in range(15)}
    bonus = ep.plackett_luce_bonus(strengths)
    assert abs(sum(bonus.values()) - 6.0) < 1e-6


def test_plackett_luce_highest_strength_gets_most_expected_bonus():
    strengths = {"star": 10.0, "average": 2.0, "weak": 0.5}
    bonus = ep.plackett_luce_bonus(strengths)
    assert bonus["star"] > bonus["average"] > bonus["weak"]


def test_plackett_luce_equal_strengths_split_evenly():
    strengths = {"a": 1.0, "b": 1.0, "c": 1.0}
    bonus = ep.plackett_luce_bonus(strengths)
    vals = list(bonus.values())
    assert max(vals) - min(vals) < 1e-9
    assert abs(sum(vals) - 6.0) < 1e-9


def test_plackett_luce_fewer_than_three_players_no_crash():
    # only 2 participants: rank1 (3pts) and rank2 (2pts) are always awarded between them,
    # rank3 (1pt) can never be awarded since no third player exists -- total 5, not 6.
    bonus = ep.plackett_luce_bonus({"a": 1.0, "b": 2.0})
    assert abs(sum(bonus.values()) - 5.0) < 1e-9


def test_plackett_luce_empty_input_no_crash():
    assert ep.plackett_luce_bonus({}) == {}


def test_non_double_counting_audit_structure():
    audit = ep.non_double_counting_audit()
    assert len(audit) > 0
    for entry in audit:
        assert {"raw_stat", "feeds", "intentional_dual_use", "note"} <= entry.keys()
        if len(entry["feeds"]) > 1:
            assert entry["intentional_dual_use"] is True, f"unreviewed dual-use: {entry['raw_stat']}"


def test_non_double_counting_audit_covers_cbi_per_spec_example():
    audit = ep.non_double_counting_audit()
    cbi_entries = [e for e in audit if "CBI" in e["raw_stat"]]
    assert len(cbi_entries) == 1
    assert "ep_defcon" in cbi_entries[0]["feeds"]
    assert any("bps" in f.lower() for f in cbi_entries[0]["feeds"])


# ============================================================
# _set_piece_goal_uplift_multiplier / e_goals uplift -- SetPieceTakers evidence, previously
# ingested (ingest_research_pull.ingest_set_piece_takers()) but confirmed unused anywhere
# (grepped the whole src/ tree before wiring this in).
# ============================================================

from datetime import date, datetime, timezone  # noqa: E402


def _seed_source_and_claim(con, player_uid, duty, order, observed_date=date(2026, 8, 1)):
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, 'Forward') ON CONFLICT DO NOTHING", [player_uid, player_uid])
    con.execute(
        "INSERT INTO sources (source_id, source_name, source_type, base_reliability_score) "
        "VALUES ('src1', 'Test Source', 'official', 0.9) ON CONFLICT DO NOTHING"
    )
    con.execute(
        "INSERT INTO evidence_claims (claim_id, subject_entity_type, subject_entity_id, claim_type, "
        "claim_value, information_type, source_id, source_reliability_score, confidence, "
        "observed_date, ingested_date, tab_origin, row_origin) "
        "VALUES (?, 'player', ?, 'set_piece_order_override', ?, 'FACT', 'src1', 0.9, 0.9, ?, ?, "
        "'research_pull:SetPieceTakers', 1)",
        [f"claim_{player_uid}", player_uid, __import__("json").dumps({"club": "A", "duty": duty, "order": order}),
         observed_date, datetime(2026, 8, 1, tzinfo=timezone.utc)],
    )


def test_set_piece_goal_uplift_applies_for_confirmed_primary_penalty_taker(con):
    ep.seed_v1_params(con)
    _seed_source_and_claim(con, "p1", duty="Penalties", order="primary")
    asof = datetime(2026, 8, 10, tzinfo=timezone.utc)
    multiplier = ep._set_piece_goal_uplift_multiplier(con, "p1", asof, set_piece_params_version=1)
    assert multiplier == pytest.approx(1.15)


def test_set_piece_goal_uplift_no_op_for_secondary_penalty_taker(con):
    ep.seed_v1_params(con)
    _seed_source_and_claim(con, "p1", duty="Penalties", order="secondary")
    asof = datetime(2026, 8, 10, tzinfo=timezone.utc)
    multiplier = ep._set_piece_goal_uplift_multiplier(con, "p1", asof, set_piece_params_version=1)
    assert multiplier == pytest.approx(1.0)


def test_set_piece_goal_uplift_no_op_for_corner_duty(con):
    """Corner duty (delivery, not a direct scoring opportunity for the taker) is out of scope
    for the GOAL uplift specifically -- see _set_piece_assist_uplift_multiplier() instead."""
    ep.seed_v1_params(con)
    _seed_source_and_claim(con, "p1", duty="Corners", order="primary")
    asof = datetime(2026, 8, 10, tzinfo=timezone.utc)
    multiplier = ep._set_piece_goal_uplift_multiplier(con, "p1", asof, set_piece_params_version=1)
    assert multiplier == pytest.approx(1.0)


def test_set_piece_goal_uplift_applies_for_confirmed_primary_free_kick_taker(con):
    """Priority 7b: a direct free-kick is a real, if far rarer, scoring opportunity for the
    taker -- a smaller uplift than penalties, same mechanism."""
    ep.seed_v1_params(con)
    _seed_source_and_claim(con, "p1", duty="Free-kicks", order="primary")
    asof = datetime(2026, 8, 10, tzinfo=timezone.utc)
    multiplier = ep._set_piece_goal_uplift_multiplier(con, "p1", asof, set_piece_params_version=1)
    assert multiplier == pytest.approx(1.05)


def test_set_piece_goal_uplift_penalty_wins_when_both_claimed(con):
    """A player confirmed as both primary penalty AND free-kick taker gets the larger, more
    established penalty multiplier, not a compounded or ambiguous result."""
    ep.seed_v1_params(con)
    _seed_source_and_claim(con, "p1", duty="Penalties", order="primary", observed_date=date(2026, 8, 1))
    con.execute(
        "INSERT INTO evidence_claims (claim_id, subject_entity_type, subject_entity_id, claim_type, "
        "claim_value, information_type, source_id, source_reliability_score, confidence, "
        "observed_date, ingested_date, tab_origin, row_origin) "
        "VALUES ('claim_p1_fk', 'player', 'p1', 'set_piece_order_override', ?, 'FACT', 'src1', 0.9, 0.9, "
        "?, ?, 'research_pull:SetPieceTakers', 2)",
        [__import__("json").dumps({"club": "A", "duty": "Free-kicks", "order": "primary"}),
         date(2026, 8, 1), datetime(2026, 8, 1, tzinfo=timezone.utc)],
    )
    asof = datetime(2026, 8, 10, tzinfo=timezone.utc)
    multiplier = ep._set_piece_goal_uplift_multiplier(con, "p1", asof, set_piece_params_version=1)
    assert multiplier == pytest.approx(1.15)


# ============================================================
# _set_piece_assist_uplift_multiplier -- Priority 7b
# ============================================================

def test_set_piece_assist_uplift_applies_for_confirmed_primary_corner_taker(con):
    ep.seed_v1_params(con)
    _seed_source_and_claim(con, "p1", duty="Corners", order="primary")
    asof = datetime(2026, 8, 10, tzinfo=timezone.utc)
    multiplier = ep._set_piece_assist_uplift_multiplier(con, "p1", asof, set_piece_params_version=1)
    assert multiplier == pytest.approx(1.20)


def test_set_piece_assist_uplift_applies_for_confirmed_primary_free_kick_taker(con):
    """A free-kick claim doesn't distinguish direct-shot duty from out-swinging delivery duty
    in the source data -- it legitimately contributes to the assist uplift too, not just the
    goal uplift (both are real possible sources of value from that role)."""
    ep.seed_v1_params(con)
    _seed_source_and_claim(con, "p1", duty="Free-kicks", order="primary")
    asof = datetime(2026, 8, 10, tzinfo=timezone.utc)
    multiplier = ep._set_piece_assist_uplift_multiplier(con, "p1", asof, set_piece_params_version=1)
    assert multiplier == pytest.approx(1.20)


def test_set_piece_assist_uplift_no_op_for_penalty_only_duty(con):
    ep.seed_v1_params(con)
    _seed_source_and_claim(con, "p1", duty="Penalties", order="primary")
    asof = datetime(2026, 8, 10, tzinfo=timezone.utc)
    multiplier = ep._set_piece_assist_uplift_multiplier(con, "p1", asof, set_piece_params_version=1)
    assert multiplier == pytest.approx(1.0)


def test_set_piece_assist_uplift_no_op_for_secondary_corner_taker(con):
    ep.seed_v1_params(con)
    _seed_source_and_claim(con, "p1", duty="Corners", order="secondary")
    asof = datetime(2026, 8, 10, tzinfo=timezone.utc)
    multiplier = ep._set_piece_assist_uplift_multiplier(con, "p1", asof, set_piece_params_version=1)
    assert multiplier == pytest.approx(1.0)


def test_set_piece_assist_uplift_no_op_with_no_claims_at_all(con):
    ep.seed_v1_params(con)
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p1', 'p1', 'Forward')")
    asof = datetime(2026, 8, 10, tzinfo=timezone.utc)
    multiplier = ep._set_piece_assist_uplift_multiplier(con, "p1", asof, set_piece_params_version=1)
    assert multiplier == pytest.approx(1.0)


def test_set_piece_goal_uplift_no_op_with_no_claims_at_all(con):
    ep.seed_v1_params(con)
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p1', 'p1', 'Forward')")
    asof = datetime(2026, 8, 10, tzinfo=timezone.utc)
    multiplier = ep._set_piece_goal_uplift_multiplier(con, "p1", asof, set_piece_params_version=1)
    assert multiplier == pytest.approx(1.0)


def test_set_piece_goal_uplift_respects_asof_look_ahead_safety(con):
    """A claim observed AFTER asof (e.g. a mid-season penalty-duty change not yet knowable at
    the asof date being evaluated) must not apply -- same look-ahead-prevention discipline
    every other evidence-claim consumer in this project already carries."""
    ep.seed_v1_params(con)
    _seed_source_and_claim(con, "p1", duty="Penalties", order="primary", observed_date=date(2026, 9, 1))
    asof_before = datetime(2026, 8, 10, tzinfo=timezone.utc)
    assert ep._set_piece_goal_uplift_multiplier(con, "p1", asof_before, set_piece_params_version=1) == pytest.approx(1.0)
    asof_after = datetime(2026, 9, 5, tzinfo=timezone.utc)
    assert ep._set_piece_goal_uplift_multiplier(con, "p1", asof_after, set_piece_params_version=1) == pytest.approx(1.15)


def test_compute_player_fixture_components_applies_uplift_when_opted_in(con):
    """Integration check: with set_piece_params_version supplied and a confirmed primary
    penalty-taker claim, ep_goals (and the e_goals-fed expected_bps term) must come out higher
    than the same call with the feature left off (set_piece_params_version=None, the default)
    -- proving the wiring actually reaches compute_player_fixture_components(), not just the
    helper in isolation."""
    ep.seed_v1_params(con)
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('team_a', 'A'), ('team_b', 'B')")
    con.execute(
        "INSERT INTO fact_match (match_id, season, gameweek, home_team_uid, away_team_uid, finished, "
        "competition, kickoff_time, _ingested_at) VALUES ('m1', '2026-2027', 2, 'team_a', 'team_b', FALSE, "
        "'Premier League', '2026-08-24', current_timestamp)"
    )
    con.execute(
        "INSERT INTO team_strength_model_versions (calibration_asof_date, home_advantage, xi_params_version, "
        "rho_params_version, reference_team_uid) VALUES ('2026-08-10', 0.2, 1, 1, 'team_a')"
    )
    ts_mv = con.execute("SELECT max(model_version) FROM team_strength_model_versions").fetchone()[0]
    for team_uid, attack, defence in (("team_a", 0.3, 0.0), ("team_b", -0.1, 0.1)):
        con.execute(
            "INSERT INTO team_strength_snapshots (model_version, team_uid, final_attack, final_defence, "
            "seasons_of_topflight_data, weight_own_data) VALUES (?, ?, ?, ?, 2, 1.0)",
            [ts_mv, team_uid, attack, defence],
        )
    _seed_source_and_claim(con, "p1", duty="Penalties", order="primary")
    con.execute(
        "INSERT INTO fact_player_season_stats (player_uid, season, gw, expected_goals, minutes, _ingested_at) "
        "VALUES ('p1', '2026-2027', 1, 3.6, 450, current_timestamp)"
    )

    mean_minutes = {"mean_1_59": 30.0, "mean_60plus": 85.0}
    asof = datetime(2026, 8, 24, tzinfo=timezone.utc)

    without_uplift = ep.compute_player_fixture_components(
        con, "p1", "Forward", "team_a", "m1", 0.05, 0.15, 0.80, ts_mv, 1, 1, ["2026-2027"], mean_minutes,
        asof=asof, set_piece_params_version=None,
    )
    with_uplift = ep.compute_player_fixture_components(
        con, "p1", "Forward", "team_a", "m1", 0.05, 0.15, 0.80, ts_mv, 1, 1, ["2026-2027"], mean_minutes,
        asof=asof, set_piece_params_version=1,
    )
    assert with_uplift["ep_goals"] == pytest.approx(without_uplift["ep_goals"] * 1.15)
    assert with_uplift["expected_bps"] > without_uplift["expected_bps"]
