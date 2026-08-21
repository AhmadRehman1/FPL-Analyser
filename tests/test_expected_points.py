import pytest
from scipy.stats import poisson

from fpl_quant import expected_points as ep
from fpl_quant import params


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
# _set_piece_goal_uplift_multiplier / _set_piece_assist_uplift_multiplier -- SetPieceTakers
# evidence, previously ingested (ingest_research_pull.ingest_set_piece_takers()) but confirmed
# unused anywhere (grepped the whole src/ tree before first wiring the primary-penalty case
# in). A3 extends both directions (secondary demotion) and duties (free-kick/corner -> assists).
# ============================================================

from datetime import date, datetime, timezone  # noqa: E402


def _seed_source_and_claim(con, player_uid, duty, order, observed_date=date(2026, 8, 1),
                            claim_id=None, source_id="src1", reliability=0.9, confidence=0.9):
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, 'Forward') ON CONFLICT DO NOTHING", [player_uid, player_uid])
    con.execute(
        "INSERT INTO sources (source_id, source_name, source_type, base_reliability_score) "
        "VALUES (?, ?, 'official', ?) ON CONFLICT DO NOTHING",
        [source_id, f"Test Source ({source_id})", reliability],
    )
    con.execute(
        "INSERT INTO evidence_claims (claim_id, subject_entity_type, subject_entity_id, claim_type, "
        "claim_value, information_type, source_id, source_reliability_score, confidence, "
        "observed_date, ingested_date, tab_origin, row_origin) "
        "VALUES (?, 'player', ?, 'set_piece_order_override', ?, 'FACT', ?, ?, ?, ?, ?, "
        "'research_pull:SetPieceTakers', 1)",
        [claim_id or f"claim_{player_uid}_{source_id}_{order}", player_uid,
         __import__("json").dumps({"club": "A", "duty": duty, "order": order}),
         source_id, reliability, confidence, observed_date, datetime(2026, 8, 1, tzinfo=timezone.utc)],
    )


def test_set_piece_goal_uplift_applies_for_confirmed_primary_penalty_taker(con):
    ep.seed_v1_params(con)
    _seed_source_and_claim(con, "p1", duty="Penalties", order="primary")
    asof = datetime(2026, 8, 10, tzinfo=timezone.utc)
    multiplier = ep._set_piece_goal_uplift_multiplier(con, "p1", asof, set_piece_params_version=1)
    assert multiplier == pytest.approx(1.15)


def test_set_piece_goal_uplift_demotes_for_confirmed_secondary_penalty_taker(con):
    """A3: previously a no-op (secondary-order claims were read then silently ignored) -- now a
    genuine, bounded demotion, since a confirmed secondary taker is real evidence of reduced
    personal conversion likelihood, not the mere absence of a boost."""
    ep.seed_v1_params(con)
    _seed_source_and_claim(con, "p1", duty="Penalties", order="secondary")
    asof = datetime(2026, 8, 10, tzinfo=timezone.utc)
    multiplier = ep._set_piece_goal_uplift_multiplier(con, "p1", asof, set_piece_params_version=1)
    assert multiplier == pytest.approx(0.90)


def test_set_piece_goal_uplift_no_op_for_non_penalty_duty(con):
    """Free-kick/corner duty affects _set_piece_assist_uplift_multiplier (see below), not this
    goal-side function -- confirmed primary FK/corner duty must stay a no-op here."""
    ep.seed_v1_params(con)
    _seed_source_and_claim(con, "p1", duty="Free-kicks", order="primary")
    asof = datetime(2026, 8, 10, tzinfo=timezone.utc)
    multiplier = ep._set_piece_goal_uplift_multiplier(con, "p1", asof, set_piece_params_version=1)
    assert multiplier == pytest.approx(1.0)


def test_set_piece_goal_uplift_strongest_evidence_wins_on_conflicting_claims(con):
    """Two sources disagree on penalty duty (primary vs secondary) -- the higher-reliability,
    higher-confidence source's effective_weight must decide the direction, not claim insertion
    order (the real bug this function's previous "first match wins" shape was exposed to)."""
    ep.seed_v1_params(con)
    _seed_source_and_claim(con, "p1", duty="Penalties", order="secondary",
                            source_id="src_weak", reliability=0.2, confidence=0.3)
    _seed_source_and_claim(con, "p1", duty="Penalties", order="primary",
                            source_id="src_strong", reliability=0.9, confidence=0.9)
    asof = datetime(2026, 8, 10, tzinfo=timezone.utc)
    multiplier = ep._set_piece_goal_uplift_multiplier(con, "p1", asof, set_piece_params_version=1)
    assert multiplier == pytest.approx(1.15)  # strong "primary" evidence outweighs weak "secondary"


def test_set_piece_goal_uplift_no_op_on_genuine_tie(con):
    """Equal-weight conflicting evidence must not guess a direction."""
    ep.seed_v1_params(con)
    _seed_source_and_claim(con, "p1", duty="Penalties", order="secondary",
                            source_id="src_a", reliability=0.5, confidence=0.5)
    _seed_source_and_claim(con, "p1", duty="Penalties", order="primary",
                            source_id="src_b", reliability=0.5, confidence=0.5)
    asof = datetime(2026, 8, 10, tzinfo=timezone.utc)
    multiplier = ep._set_piece_goal_uplift_multiplier(con, "p1", asof, set_piece_params_version=1)
    assert multiplier == pytest.approx(1.0)


def test_set_piece_assist_uplift_applies_for_confirmed_primary_corner_duty(con):
    ep.seed_v1_params(con)
    _seed_source_and_claim(con, "p1", duty="Corners", order="primary")
    asof = datetime(2026, 8, 10, tzinfo=timezone.utc)
    multiplier = ep._set_piece_assist_uplift_multiplier(con, "p1", asof, set_piece_params_version=1)
    assert multiplier == pytest.approx(1.12)


def test_set_piece_assist_uplift_applies_for_confirmed_primary_free_kick_duty(con):
    ep.seed_v1_params(con)
    _seed_source_and_claim(con, "p1", duty="Free-kicks", order="primary")
    asof = datetime(2026, 8, 10, tzinfo=timezone.utc)
    multiplier = ep._set_piece_assist_uplift_multiplier(con, "p1", asof, set_piece_params_version=1)
    assert multiplier == pytest.approx(1.12)


def test_set_piece_assist_uplift_combined_duty_string_not_double_counted(con):
    """Real workbook data contains combined duty strings like "corners/free-kicks" -- a single
    claim matching both keywords must contribute its effective_weight once, not twice."""
    ep.seed_v1_params(con)
    _seed_source_and_claim(con, "p1", duty="corners/free-kicks", order="secondary")
    asof = datetime(2026, 8, 10, tzinfo=timezone.utc)
    multiplier = ep._set_piece_assist_uplift_multiplier(con, "p1", asof, set_piece_params_version=1)
    # if double-counted, secondary_weight would still just be 2x a single claim's weight -- same
    # sign either way here, so assert the exact demotion value, not just direction, to catch it
    assert multiplier == pytest.approx(0.92)


def test_set_piece_assist_uplift_demotes_for_confirmed_secondary_duty(con):
    ep.seed_v1_params(con)
    _seed_source_and_claim(con, "p1", duty="Corners", order="secondary")
    asof = datetime(2026, 8, 10, tzinfo=timezone.utc)
    multiplier = ep._set_piece_assist_uplift_multiplier(con, "p1", asof, set_piece_params_version=1)
    assert multiplier == pytest.approx(0.92)


def test_set_piece_assist_uplift_no_op_for_penalty_only_duty(con):
    ep.seed_v1_params(con)
    _seed_source_and_claim(con, "p1", duty="Penalties", order="primary")
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


# ============================================================
# A2: _role_shift_multiplier -- exp_position evidence from 18_Predicted XI Database claims,
# consumed via evidence_blend.blend_categorical (previously dead code outside its own tests).
# ============================================================

def _seed_predicted_xi_claim(con, player_uid, exp_position, registered_position="Defender",
                              observed_date=date(2026, 8, 1), reliability=0.7, confidence=0.8, source_id="src_pxi"):
    con.execute(
        "INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, ?) ON CONFLICT DO NOTHING",
        [player_uid, player_uid, registered_position],
    )
    con.execute(
        "INSERT INTO sources (source_id, source_name, source_type, base_reliability_score) "
        "VALUES (?, 'Test XI Source', 'specialist', ?) ON CONFLICT DO NOTHING",
        [source_id, reliability],
    )
    con.execute(
        "INSERT INTO evidence_claims (claim_id, subject_entity_type, subject_entity_id, claim_type, "
        "claim_value, information_type, source_id, source_reliability_score, confidence, "
        "observed_date, ingested_date, tab_origin, row_origin) "
        "VALUES (?, 'player', ?, 'predicted_xi', ?, 'OPINION', ?, ?, ?, ?, ?, "
        "'18_Predicted XI Database', 1)",
        [f"claim_pxi_{player_uid}_{exp_position}", player_uid,
         __import__("json").dumps({"exp_position": exp_position}), source_id, reliability, confidence,
         observed_date, datetime(2026, 8, 1, tzinfo=timezone.utc)],
    )


def test_role_shift_multiplier_no_op_with_no_predicted_xi_claims(con):
    ep.seed_v1_params(con)
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p1', 'p1', 'Defender')")
    asof = datetime(2026, 8, 10, tzinfo=timezone.utc)
    multiplier = ep._role_shift_multiplier(con, "p1", "Defender", asof, 1, 1, 1)
    assert multiplier == pytest.approx(1.0)


def test_role_shift_multiplier_no_op_when_exp_position_matches_registered(con):
    ep.seed_v1_params(con)
    _seed_predicted_xi_claim(con, "p1", exp_position="Defender", registered_position="Defender")
    asof = datetime(2026, 8, 10, tzinfo=timezone.utc)
    multiplier = ep._role_shift_multiplier(con, "p1", "Defender", asof, 1, 1, 1)
    assert multiplier == pytest.approx(1.0)


def test_role_shift_multiplier_uplifts_when_predicted_more_advanced_than_registered(con):
    """A registered Defender predicted to play as a Forward (rank +2) should get a positive,
    capped attacking-output multiplier -- not the raw uncapped rank delta."""
    ep.seed_v1_params(con)
    _seed_predicted_xi_claim(con, "p1", exp_position="Forward", registered_position="Defender")
    asof = datetime(2026, 8, 10, tzinfo=timezone.utc)
    multiplier = ep._role_shift_multiplier(con, "p1", "Defender", asof, 1, 1, 1)
    # rank delta +2 * per_rank_multiplier_step (0.08) = 0.16, exactly at max_multiplier_delta (0.16)
    assert multiplier == pytest.approx(1.16)


def test_role_shift_multiplier_downshifts_when_predicted_less_advanced_than_registered(con):
    ep.seed_v1_params(con)
    _seed_predicted_xi_claim(con, "p1", exp_position="Defender", registered_position="Forward")
    asof = datetime(2026, 8, 10, tzinfo=timezone.utc)
    multiplier = ep._role_shift_multiplier(con, "p1", "Forward", asof, 1, 1, 1)
    assert multiplier == pytest.approx(0.84)


def test_role_shift_multiplier_capped_for_a_full_gk_to_forward_swing(con):
    ep.seed_v1_params(con)
    _seed_predicted_xi_claim(con, "p1", exp_position="Forward", registered_position="Goalkeeper")
    asof = datetime(2026, 8, 10, tzinfo=timezone.utc)
    multiplier = ep._role_shift_multiplier(con, "p1", "Goalkeeper", asof, 1, 1, 1)
    # raw rank delta +3 * 0.08 = 0.24, must be clamped to the 0.16 cap, not applied uncapped
    assert multiplier == pytest.approx(1.16)


def test_role_shift_multiplier_ignores_unrecognized_exp_position_strings(con):
    """A free-text exp_position outside the four canonical positions (e.g. the real workbook's
    "Right-back/Wing-back") must be excluded from the weighted signal, not guessed at."""
    ep.seed_v1_params(con)
    _seed_predicted_xi_claim(con, "p1", exp_position="Right-back/Wing-back", registered_position="Defender")
    asof = datetime(2026, 8, 10, tzinfo=timezone.utc)
    multiplier = ep._role_shift_multiplier(con, "p1", "Defender", asof, 1, 1, 1)
    assert multiplier == pytest.approx(1.0)


def test_role_shift_multiplier_respects_asof_look_ahead_safety(con):
    ep.seed_v1_params(con)
    _seed_predicted_xi_claim(con, "p1", exp_position="Forward", registered_position="Defender",
                              observed_date=date(2026, 9, 1))
    asof_before = datetime(2026, 8, 10, tzinfo=timezone.utc)
    assert ep._role_shift_multiplier(con, "p1", "Defender", asof_before, 1, 1, 1) == pytest.approx(1.0)
    asof_after = datetime(2026, 9, 5, tzinfo=timezone.utc)
    assert ep._role_shift_multiplier(con, "p1", "Defender", asof_after, 1, 1, 1) == pytest.approx(1.16)


def test_compute_player_fixture_components_applies_role_shift_when_opted_in(con):
    """Integration check mirroring the set-piece uplift test above: with the role-shift params
    supplied and a Defender predicted to play as a Forward, ep_goals/ep_assists (and the
    e_goals-fed expected_bps term) must come out higher than the same call with the feature
    left off -- proving the wiring actually reaches compute_player_fixture_components()."""
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
    _seed_predicted_xi_claim(con, "p1", exp_position="Forward", registered_position="Defender")
    con.execute(
        "INSERT INTO fact_player_season_stats (player_uid, season, gw, expected_goals, expected_assists, minutes, _ingested_at) "
        "VALUES ('p1', '2026-2027', 1, 3.6, 1.0, 450, current_timestamp)"
    )

    mean_minutes = {"mean_1_59": 30.0, "mean_60plus": 85.0}
    asof = datetime(2026, 8, 24, tzinfo=timezone.utc)

    without_role_shift = ep.compute_player_fixture_components(
        con, "p1", "Defender", "team_a", "m1", 0.05, 0.15, 0.80, ts_mv, 1, 1, ["2026-2027"], mean_minutes,
        asof=asof,
    )
    with_role_shift = ep.compute_player_fixture_components(
        con, "p1", "Defender", "team_a", "m1", 0.05, 0.15, 0.80, ts_mv, 1, 1, ["2026-2027"], mean_minutes,
        asof=asof, decay_params_version=1, fact_multiplier_params_version=1, role_shift_params_version=1,
    )
    assert with_role_shift["ep_goals"] == pytest.approx(without_role_shift["ep_goals"] * 1.16)
    assert with_role_shift["ep_assists"] == pytest.approx(without_role_shift["ep_assists"] * 1.16)
    assert with_role_shift["expected_bps"] > without_role_shift["expected_bps"]


def test_defcon_rate_excludes_recoveries_for_defenders_but_not_midfielders(con, monkeypatch):
    """Real FPL rule this caught: defenders clear a CBIT threshold (10, no recoveries);
    midfielders/forwards clear a CBIRT threshold (12, +recoveries) -- defcon_threshold already
    encoded that split (10 vs 12) but defcon_rate used to add recoveries_per_90 for every
    position unconditionally, letting recoveries alone push a defender over a threshold real
    FPL would never credit them for reaching. Real case that surfaced this: three genuine
    center-backs outscoring an elite attacking midfielder on total EP, purely on inflated
    defcon. cbi_per_90 is deliberately set BELOW the defender threshold (10) here and
    recoveries_per_90 high enough that only counting both together would clear it -- a
    defender's ep_defcon must stay ~0, a midfielder's (same raw rates) must not."""
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
    monkeypatch.setattr(
        ep, "_defensive_action_rates_per_90",
        lambda con, player_uid, position, seasons: {"cbi_per_90": 6.0, "recoveries_per_90": 8.0},
    )
    mean_minutes = {"mean_1_59": 30.0, "mean_60plus": 85.0}

    defender = ep.compute_player_fixture_components(
        con, "p1", "Defender", "team_a", "m1", 0.05, 0.15, 0.80, ts_mv, 1, 1, ["2026-2027"], mean_minutes,
    )
    midfielder = ep.compute_player_fixture_components(
        con, "p1", "Midfielder", "team_a", "m1", 0.05, 0.15, 0.80, ts_mv, 1, 1, ["2026-2027"], mean_minutes,
    )
    assert defender["ep_defcon"] < 0.1, "cbi_per_90=6 is well below the defender threshold of 10 -- recoveries must not count"
    assert midfielder["ep_defcon"] > 0.5, "cbi_per_90 + recoveries_per_90 = 14 clears the midfielder threshold of 12"


# ============================================================
# A4: explain_qualitative_adjustment() / explain_player_ep() provenance trail
# ============================================================

def _seed_ep_model_version(con, *, set_piece_params_version=None, decay_params_version=None,
                            fact_multiplier_params_version=None, role_shift_params_version=None,
                            calibration_asof_date="2026-08-10"):
    con.execute(
        "INSERT INTO team_strength_model_versions (calibration_asof_date, home_advantage, xi_params_version, "
        "rho_params_version, reference_team_uid) VALUES (?, 0.2, 1, 1, 'team_a')",
        [calibration_asof_date],
    )
    ts_mv = con.execute("SELECT max(model_version) FROM team_strength_model_versions").fetchone()[0]
    con.execute(
        "INSERT INTO minutes_model_versions (calibration_asof_date, target_season, decay_params_version, "
        "adjustment_params_version, shrinkage_params_version, fact_multiplier_params_version, lookback_seasons) "
        "VALUES (?, '2026-2027', 1, 1, 1, 1, '[]')",
        [calibration_asof_date],
    )
    mm_mv = con.execute("SELECT max(model_version) FROM minutes_model_versions").fetchone()[0]
    ep_mv = con.execute(
        "INSERT INTO ep_model_versions (calibration_asof_date, target_season, team_strength_model_version, "
        "minutes_model_version, scoring_matrix_params_version, bps_params_version, bps_tau_params_version, "
        "set_piece_params_version, decay_params_version, fact_multiplier_params_version, role_shift_params_version) "
        "VALUES (?, '2026-2027', ?, ?, 1, 1, 1, ?, ?, ?, ?) RETURNING model_version",
        [calibration_asof_date, ts_mv, mm_mv, set_piece_params_version, decay_params_version,
         fact_multiplier_params_version, role_shift_params_version],
    ).fetchone()[0]
    return ep_mv


def test_explain_qualitative_adjustment_empty_when_fully_opted_out(con):
    ep.seed_v1_params(con)
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p1', 'p1', 'Defender')")
    ep_mv = _seed_ep_model_version(con)
    assert ep.explain_qualitative_adjustment(con, ep_mv, "p1") == {}


def test_explain_qualitative_adjustment_reports_role_shift_and_set_piece_goal(con):
    ep.seed_v1_params(con)
    _seed_predicted_xi_claim(con, "p1", exp_position="Forward", registered_position="Defender")
    con.execute(
        "INSERT INTO sources (source_id, source_name, source_type, base_reliability_score) "
        "VALUES ('src_sp', 'Set Piece Source', 'official', 0.9)"
    )
    con.execute(
        "INSERT INTO evidence_claims (claim_id, subject_entity_type, subject_entity_id, claim_type, "
        "claim_value, information_type, source_id, source_reliability_score, confidence, "
        "observed_date, ingested_date, tab_origin, row_origin) "
        "VALUES ('claim_sp1', 'player', 'p1', 'set_piece_order_override', ?, 'FACT', 'src_sp', 0.9, 0.9, "
        "'2026-08-01', '2026-08-01', 'research_pull:SetPieceTakers', 1)",
        [__import__("json").dumps({"club": "A", "duty": "Penalties", "order": "primary"})],
    )
    ep_mv = _seed_ep_model_version(
        con, set_piece_params_version=1, decay_params_version=1, fact_multiplier_params_version=1, role_shift_params_version=1,
    )

    result = ep.explain_qualitative_adjustment(con, ep_mv, "p1")

    assert result["role_shift"]["applied"] is True
    assert result["role_shift"]["multiplier"] == pytest.approx(1.16)
    assert result["role_shift"]["registered_position"] == "Defender"
    role_claims = result["role_shift"]["claims"]
    assert len(role_claims) == 1
    assert role_claims[0]["included"] is True
    assert role_claims[0]["exp_position"] == "Forward"
    assert role_claims[0]["source_name"] == "Test XI Source"

    assert result["set_piece_goal"]["applied"] is True
    assert result["set_piece_goal"]["multiplier"] == pytest.approx(1.15)
    sp_claims = result["set_piece_goal"]["claims"]
    assert len(sp_claims) == 1
    assert sp_claims[0]["included"] is True
    assert sp_claims[0]["order"] == "primary"
    assert sp_claims[0]["source_name"] == "Set Piece Source"

    assert result["set_piece_assist"]["applied"] is False
    assert result["set_piece_assist"]["multiplier"] == pytest.approx(1.0)
    assert result["set_piece_assist"]["claims"] == []


def test_explain_qualitative_adjustment_marks_losing_side_excluded(con):
    """Two conflicting penalty-duty claims: the stronger evidence wins and applies, the weaker
    evidence is shown as considered-but-excluded, not silently dropped from the trail."""
    ep.seed_v1_params(con)
    _seed_source_and_claim(con, "p1", duty="Penalties", order="secondary",
                            source_id="src_weak", reliability=0.2, confidence=0.3)
    _seed_source_and_claim(con, "p1", duty="Penalties", order="primary",
                            source_id="src_strong", reliability=0.9, confidence=0.9)
    ep_mv = _seed_ep_model_version(con, set_piece_params_version=1, decay_params_version=1, fact_multiplier_params_version=1)

    result = ep.explain_qualitative_adjustment(con, ep_mv, "p1")
    assert result["set_piece_goal"]["applied"] is True
    assert result["set_piece_goal"]["multiplier"] == pytest.approx(1.15)
    by_order = {c["order"]: c for c in result["set_piece_goal"]["claims"]}
    assert by_order["primary"]["included"] is True
    assert by_order["secondary"]["included"] is False
    assert by_order["secondary"]["exclusion_reason"] == "outweighed by stronger opposing-order evidence"


def test_explain_player_ep_includes_qualitative_adjustments_key(con):
    ep.seed_v1_params(con)
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p1', 'p1', 'Defender')")
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('team_a', 'A'), ('team_b', 'B')")
    con.execute(
        "INSERT INTO fact_match (match_id, season, gameweek, home_team_uid, away_team_uid, finished, "
        "competition, kickoff_time, _ingested_at) VALUES ('m1', '2026-2027', 2, 'team_a', 'team_b', FALSE, "
        "'Premier League', '2026-08-24', current_timestamp)"
    )
    ep_mv = _seed_ep_model_version(con)
    con.execute(
        "INSERT INTO ep_outputs (model_version, player_uid, fixture_match_id, ep_appearance, ep_goals, "
        "ep_assists, ep_clean_sheet, ep_goals_conceded, ep_defcon, ep_bonus, ep_saves, ep_penalty_save, "
        "ep_cards, ep_own_goal, ep_total, expected_bps) VALUES (?, 'p1', 'm1', 1.0, 0.1, 0.1, 0.3, 0.0, 0.0, "
        "0.2, 0.0, 0.0, 0.0, 0.0, 1.7, 15.0)",
        [ep_mv],
    )
    result = ep.explain_player_ep(con, ep_mv, "p1")
    assert result is not None
    assert result["qualitative_adjustments"] == {}  # fully opted out -- consistent with the dedicated test above
