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


def test_ep_run_applies_the_set_piece_uplift_by_default():
    """The uplift was fully built (Priority 7b) but no live entrypoint ever passed
    set_piece_params_version, so it was dormant. run() / compute_horizon_ep() / backtest.run()
    now default it to 1 -- an accidental revert to None silently turns off the penalty-taker
    goal boost for ~26 real players again."""
    import inspect

    from fpl_quant import backtest as bt
    from fpl_quant import transfer_planner as tp

    assert inspect.signature(ep.run).parameters["set_piece_params_version"].default == 1
    assert inspect.signature(tp.compute_horizon_ep).parameters["set_piece_params_version"].default == 1
    assert inspect.signature(bt.run).parameters["set_piece_params_version"].default == 1
    assert inspect.signature(bt.run_gameweek_step).parameters["set_piece_params_version"].default == 1


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


# ============================================================
# Fixture-strength scaling: e_goals / e_assists were opponent-blind (only lambda_against was
# ever used -- for clean sheets). _fixture_attack_multiplier scales them by how weak THIS
# opponent's defence is vs a league-average one.
# ============================================================

def _seed_fixture_strength_scenario(con):
    ep.seed_v1_params(con)
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('strong', 'S'), ('weak', 'W'), ('mid', 'M')")
    # strong side at home vs a weak (leaky) defence, and the same strong side away at a strong defence
    for mid, h, a in (("easy", "strong", "weak"), ("hard", "mid", "strong")):
        con.execute(
            "INSERT INTO fact_match (match_id, season, gameweek, home_team_uid, away_team_uid, finished, "
            "competition, kickoff_time, _ingested_at) VALUES (?, '2026-2027', 2, ?, ?, FALSE, "
            "'Premier League', '2026-08-24', current_timestamp)", [mid, h, a],
        )
    con.execute(
        "INSERT INTO team_strength_model_versions (calibration_asof_date, home_advantage, xi_params_version, "
        "rho_params_version, reference_team_uid) VALUES ('2026-08-10', 0.25, 1, 1, 'strong')"
    )
    ts_mv = con.execute("SELECT max(model_version) FROM team_strength_model_versions").fetchone()[0]
    # final_defence: HIGHER = better defence (Arsenal ~+0.34, Coventry ~-0.44 in the real fit).
    # "weak" is the leaky side, "mid"/"strong" defend well.
    for tu, atk, dfc in (("strong", 0.4, 0.4), ("weak", -0.3, -0.4), ("mid", 0.0, 0.3)):
        con.execute(
            "INSERT INTO team_strength_snapshots (model_version, team_uid, final_attack, final_defence, "
            "seasons_of_topflight_data, weight_own_data) VALUES (?, ?, ?, ?, 2, 1.0)", [ts_mv, tu, atk, dfc],
        )
    return ts_mv


def test_fixture_attack_multiplier_direction_and_toggle(con):
    ts_mv = _seed_fixture_strength_scenario(con)
    easy = ep._fixture_attack_multiplier(con, "strong", "easy", "2026-2027", ts_mv, 1)
    hard = ep._fixture_attack_multiplier(con, "strong", "hard", "2026-2027", ts_mv, 1)
    assert easy > 1.15          # strong side, home, vs a leaky defence -> boosted
    assert hard < 0.9           # strong side, away, vs a good defence -> suppressed
    assert easy <= 2.5 and hard >= 0.4   # clipped to the sane band
    # sensitivity 0 (or an unseeded-to-0 version) is an exact no-op
    from fpl_quant import params as pmod
    pmod.write_param(con, "fixture_strength_params", 2, "2026-08-10", "attack_sensitivity", value_numeric=0.0)
    assert ep._fixture_attack_multiplier(con, "strong", "easy", "2026-2027", ts_mv, 2) == 1.0


def test_compute_player_fixture_components_scales_goals_by_fixture(con):
    ts_mv = _seed_fixture_strength_scenario(con)
    con.execute(
        "INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('fwd', 'Fwd', 'Forward')"
    )
    con.execute(
        "INSERT INTO fact_player_season_stats (player_uid, season, gw, expected_goals, expected_assists, "
        "minutes, _ingested_at) VALUES ('fwd', '2025-2026', 38, 18.0, 4.0, 3000, current_timestamp)"
    )
    mm = {"mean_1_59": 30.0, "mean_60plus": 85.0}
    easy = ep.compute_player_fixture_components(
        con, "fwd", "Forward", "strong", "easy", 0.05, 0.1, 0.85, ts_mv, 1, 1,
        ["2026-2027", "2025-2026"], mm, target_season="2026-2027",
    )
    hard = ep.compute_player_fixture_components(
        con, "fwd", "Forward", "strong", "hard", 0.05, 0.1, 0.85, ts_mv, 1, 1,
        ["2026-2027", "2025-2026"], mm, target_season="2026-2027",
    )
    off = ep.compute_player_fixture_components(
        con, "fwd", "Forward", "strong", "easy", 0.05, 0.1, 0.85, ts_mv, 1, 1,
        ["2026-2027", "2025-2026"], mm, target_season="2026-2027", fixture_params_version=None,
    )
    # same player, same minutes -- the only difference is the opponent
    assert easy["ep_goals"] > hard["ep_goals"] * 1.4
    assert easy["ep_goals"] > off["ep_goals"] and hard["ep_goals"] < off["ep_goals"]
    assert easy["ep_assists"] > off["ep_assists"]


# ============================================================
# DefCon action set is position-specific: a DEFENDER's threshold counts CBIT only (clearances,
# blocks, interceptions, tackles); a MIDFIELDER/FORWARD's counts CBIT + ball recoveries. This
# regression covers the bug where recoveries were added to every position's rate, roughly
# doubling defenders' modelled DefCon rate and making almost every nailed starter a near-certain
# +2 -- the main reason defenders outranked premium forwards for captaincy.
# ============================================================

def _seed_fixture_and_strength(con, gw=2):
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('team_a', 'A'), ('team_b', 'B')")
    con.execute(
        "INSERT INTO fact_match (match_id, season, gameweek, home_team_uid, away_team_uid, finished, "
        "competition, kickoff_time, _ingested_at) VALUES ('m1', '2026-2027', ?, 'team_a', 'team_b', FALSE, "
        "'Premier League', '2026-08-24', current_timestamp)", [gw],
    )
    con.execute(
        "INSERT INTO team_strength_model_versions (calibration_asof_date, home_advantage, xi_params_version, "
        "rho_params_version, reference_team_uid) VALUES ('2026-08-10', 0.2, 1, 1, 'team_a')"
    )
    ts_mv = con.execute("SELECT max(model_version) FROM team_strength_model_versions").fetchone()[0]
    for team_uid, attack, defence in (("team_a", 0.2, 0.1), ("team_b", 0.0, 0.0)):
        con.execute(
            "INSERT INTO team_strength_snapshots (model_version, team_uid, final_attack, final_defence, "
            "seasons_of_topflight_data, weight_own_data) VALUES (?, ?, ?, ?, 2, 1.0)",
            [ts_mv, team_uid, attack, defence],
        )
    return ts_mv


def _seed_defensive_actions(con, player_uid, position, *, cbit_per_match, recoveries_per_match, n_matches=20):
    """n_matches identical rows of 90 minutes each, so per-90 rates are exactly the per-match
    counts and the sample is large enough that shrinkage toward the position average is minimal."""
    con.execute(
        "INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, ?) ON CONFLICT DO NOTHING",
        [player_uid, player_uid, position],
    )
    tackles = cbit_per_match  # all CBIT loaded onto one column -- the model sums the four
    for i in range(n_matches):
        match_id = f"hist_{player_uid}_{i}"
        con.execute(
            "INSERT INTO fact_match (match_id, season, gameweek, home_team_uid, away_team_uid, finished, "
            "competition, kickoff_time, _ingested_at) VALUES (?, '2025-2026', ?, 'team_a', 'team_b', TRUE, "
            "'Premier League', '2026-01-01', current_timestamp)", [match_id, i + 1],
        )
        con.execute(
            "INSERT INTO fact_player_match_stats (player_uid, match_id, season, start_min, finish_min, "
            "minutes_played, tackles, clearances, interceptions, blocks, recoveries, _ingested_at) "
            "VALUES (?, ?, '2025-2026', 0, 90, 90, ?, 0, 0, 0, ?, current_timestamp)",
            [player_uid, match_id, tackles, recoveries_per_match],
        )


def test_defcon_excludes_recoveries_for_a_defender(con):
    ep.seed_v1_params(con)
    ts_mv = _seed_fixture_and_strength(con)
    # 7 CBIT/match (below the 10 threshold) but 8 recoveries/match -- old code thresholded on
    # 15, new code thresholds a defender on 7 alone.
    _seed_defensive_actions(con, "def1", "Defender", cbit_per_match=7, recoveries_per_match=8)
    con.execute(
        "INSERT INTO fact_player_season_stats (player_uid, season, gw, expected_goals, expected_assists, "
        "minutes, _ingested_at) VALUES ('def1', '2025-2026', 38, 0.0, 0.0, 1800, current_timestamp)"
    )
    mean_minutes = {"mean_1_59": 30.0, "mean_60plus": 85.0}
    comp = ep.compute_player_fixture_components(
        con, "def1", "Defender", "team_a", "m1", 0.02, 0.05, 0.93, ts_mv, 1, 1,
        ["2025-2026"], mean_minutes,
    )
    e_min = ep.expected_minutes_given_played(0.05, 0.93, mean_minutes)
    p_played = 0.98
    cbit_only_rate = 7.0 * e_min / 90.0
    cbit_plus_rec_rate = 15.0 * e_min / 90.0
    expected_new = (1.0 - poisson.cdf(9, cbit_only_rate)) * p_played * 2.0
    would_have_been_old = (1.0 - poisson.cdf(9, cbit_plus_rec_rate)) * p_played * 2.0
    assert comp["ep_defcon"] == pytest.approx(expected_new, rel=1e-3)
    assert comp["ep_defcon"] < would_have_been_old * 0.5  # the bug roughly doubled it


def test_defcon_still_includes_recoveries_for_a_midfielder(con):
    ep.seed_v1_params(con)
    ts_mv = _seed_fixture_and_strength(con)
    _seed_defensive_actions(con, "mid1", "Midfielder", cbit_per_match=7, recoveries_per_match=8)
    con.execute(
        "INSERT INTO fact_player_season_stats (player_uid, season, gw, expected_goals, expected_assists, "
        "minutes, _ingested_at) VALUES ('mid1', '2025-2026', 38, 0.0, 0.0, 1800, current_timestamp)"
    )
    mean_minutes = {"mean_1_59": 30.0, "mean_60plus": 85.0}
    comp = ep.compute_player_fixture_components(
        con, "mid1", "Midfielder", "team_a", "m1", 0.02, 0.05, 0.93, ts_mv, 1, 1,
        ["2025-2026"], mean_minutes,
    )
    e_min = ep.expected_minutes_given_played(0.05, 0.93, mean_minutes)
    p_played = 0.98
    cbit_plus_rec_rate = 15.0 * e_min / 90.0  # MID threshold is 12, over CBIT + recoveries
    expected = (1.0 - poisson.cdf(11, cbit_plus_rec_rate)) * p_played * 2.0
    assert comp["ep_defcon"] == pytest.approx(expected, rel=1e-3)


# ============================================================
# _position_average_rates() -- the shrinkage anchor for goals/assists/saves -- must be
# minutes-weighted off each player's LATEST cumulative row per season, not an unweighted
# avg() over every per-gameweek snapshot. The old version pulled the anchor toward 0 (fringe
# players and noisy early-season snapshots counted at full weight), and _shrink_rate() then
# compressed everyone toward it -- the same failure mode as the DefCon / minutes fixes.
# ============================================================

def _fps(con, uid, season, gw, xg, xa, saves_p90, minutes):
    con.execute(
        "INSERT INTO fact_player_season_stats (player_uid, season, gw, expected_goals, expected_assists, "
        "expected_goals_per_90, expected_assists_per_90, saves_per_90, minutes, _ingested_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, current_timestamp)",
        [uid, season, gw, xg, xa, (xg / minutes * 90) if minutes else 0, (xa / minutes * 90) if minutes else 0, saves_p90, minutes],
    )


def test_position_average_rates_is_minutes_weighted_not_per_player(con):
    for uid in ("reg", "fringe"):
        con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, 'Forward')", [uid, uid])
    # A 3000-minute regular at 0.60 xG/90 and a 200-minute fringe player at 0.05 xG/90.
    _fps(con, "reg", "2025-2026", 38, xg=20.0, xa=0.0, saves_p90=0, minutes=3000)
    _fps(con, "fringe", "2025-2026", 12, xg=0.111, xa=0.0, saves_p90=0, minutes=200)
    avg = ep._position_average_rates(con, "Forward", ["2025-2026"])
    # minutes-weighted: (20.0 + 0.111) / (3000 + 200) * 90 ~= 0.566, i.e. dominated by the regular.
    assert avg["expected_goals_per_90"] == pytest.approx((20.0 + 0.111) / 3200 * 90, rel=1e-6)
    # an unweighted per-player mean would be ~(0.60 + 0.05) / 2 = 0.325 -- far lower.
    assert avg["expected_goals_per_90"] > 0.5


def test_position_average_rates_uses_only_the_latest_cumulative_row_per_season(con):
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p', 'p', 'Forward')")
    # Noisy early snapshots (GW1: 1.0 xG in 90 min) then a settled GW38 cumulative row.
    _fps(con, "p", "2025-2026", 1, xg=1.0, xa=0.0, saves_p90=0, minutes=90)
    _fps(con, "p", "2025-2026", 2, xg=1.1, xa=0.0, saves_p90=0, minutes=180)
    _fps(con, "p", "2025-2026", 38, xg=15.0, xa=0.0, saves_p90=0, minutes=3200)
    avg = ep._position_average_rates(con, "Forward", ["2025-2026"])
    # only the GW38 row: 15.0 / 3200 * 90
    assert avg["expected_goals_per_90"] == pytest.approx(15.0 / 3200 * 90, rel=1e-6)


# ============================================================
# 2024-2025's playerstats snapshot has NO season-total `minutes` / `expected_goals` columns
# (reconcile.build_fact_player_season_stats), only `expected_goals_per_90`. The old code
# required `minutes`, so all of 2024-25 was silently dropped from the attacking-rate pool and
# anchor -- while _defensive_action_rates_per_90() (reads fact_player_match_stats) kept seeing
# it. Both now recover 2024-25 via the per-90 rate x match-grain minutes.
# ============================================================

def _seed_snapshot_season(con, uid, season, xg90, xa90, *, n_matches, minutes_per_match=90):
    """A 2024-25-style row: NULL minutes / NULL expected_goals, populated per-90 rates, and
    real per-match minutes in fact_player_match_stats."""
    con.execute(
        "INSERT INTO fact_player_season_stats (player_uid, season, gw, expected_goals, expected_assists, "
        "expected_goals_per_90, expected_assists_per_90, saves_per_90, minutes, _ingested_at) "
        "VALUES (?, ?, 38, NULL, NULL, ?, ?, NULL, NULL, current_timestamp)",
        [uid, season, xg90, xa90],
    )
    for tuid in ("team_a", "team_b"):
        con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES (?, ?) ON CONFLICT DO NOTHING", [tuid, tuid])
    for i in range(n_matches):
        match_id = f"{uid}_{season}_m{i}"
        con.execute(
            "INSERT INTO fact_match (match_id, season, gameweek, home_team_uid, away_team_uid, finished, "
            "competition, kickoff_time, _ingested_at) VALUES (?, ?, ?, 'team_a', 'team_b', TRUE, "
            "'Premier League', '2025-01-01', current_timestamp) ON CONFLICT DO NOTHING",
            [match_id, season, i + 1],
        )
        con.execute(
            "INSERT INTO fact_player_match_stats (player_uid, match_id, season, start_min, finish_min, "
            "minutes_played, goals, assists, _ingested_at) VALUES (?, ?, ?, 0, ?, ?, 0, 0, current_timestamp)",
            [uid, match_id, season, minutes_per_match, minutes_per_match],
        )


def test_player_rate_pool_recovers_a_snapshot_only_season(con):
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p', 'p', 'Forward')")
    # 2024-25: snapshot schema, 0.80 xG/90 over 30x90 = 2700 min. 2025-26: richer schema, an
    # injury-truncated 0.30 xG/90 over 600 min.
    _seed_snapshot_season(con, "p", "2024-2025", xg90=0.80, xa90=0.10, n_matches=30)
    _fps(con, "p", "2025-2026", 38, xg=2.0, xa=0.0, saves_p90=0, minutes=600)  # 0.30 xG/90
    pool = ep._player_rate_pool(con, "p", ["2025-2026", "2024-2025"])
    # sample_minutes = 600 + 2700; goals = 2.0 + 0.80/90*2700 = 2.0 + 24.0 = 26.0
    assert pool["sample_minutes"] == pytest.approx(3300)
    assert pool["expected_goals_per_90"] == pytest.approx(26.0 / 3300 * 90, rel=1e-6)
    # old behaviour (2025-26 only) would have been 0.30 with just 600 sample minutes -- far
    # lower rate AND far more shrinkage toward the position average.


def test_player_rate_pool_drops_a_snapshot_season_with_no_match_minutes(con):
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p', 'p', 'Forward')")
    con.execute(
        "INSERT INTO fact_player_season_stats (player_uid, season, gw, expected_goals, expected_assists, "
        "expected_goals_per_90, expected_assists_per_90, saves_per_90, minutes, _ingested_at) "
        "VALUES ('p', '2024-2025', 38, NULL, NULL, 0.9, 0.1, NULL, NULL, current_timestamp)",
    )  # per-90 rate but zero match-grain minutes -> no real sample -> excluded, not fabricated
    _fps(con, "p", "2025-2026", 38, xg=10.0, xa=0.0, saves_p90=0, minutes=2000)
    pool = ep._player_rate_pool(con, "p", ["2025-2026", "2024-2025"])
    assert pool["sample_minutes"] == pytest.approx(2000)
    assert pool["expected_goals_per_90"] == pytest.approx(10.0 / 2000 * 90, rel=1e-6)


def test_position_average_rates_includes_snapshot_only_players(con):
    for uid in ("rich", "snap"):
        con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, 'Forward')", [uid, uid])
    # richer-schema player: 0.40 xG/90 over 3000 min in 2025-26
    _fps(con, "rich", "2025-2026", 38, xg=13.333, xa=0.0, saves_p90=0, minutes=3000)
    # snapshot-only player: 0.70 xG/90 over 25x90 = 2250 min in 2024-25, nothing in 2025-26
    _seed_snapshot_season(con, "snap", "2024-2025", xg90=0.70, xa90=0.0, n_matches=25)
    avg = ep._position_average_rates(con, "Forward", ["2025-2026", "2024-2025"])
    # minutes-weighted across BOTH: (13.333 + 0.70/90*2250) / (3000 + 2250) * 90
    expected = (13.333 + 0.70 / 90 * 2250) / (3000 + 2250) * 90
    assert avg["expected_goals_per_90"] == pytest.approx(expected, rel=1e-4)
    # dropping the snapshot player (old behaviour) would give 0.40 exactly
    assert avg["expected_goals_per_90"] > 0.45
