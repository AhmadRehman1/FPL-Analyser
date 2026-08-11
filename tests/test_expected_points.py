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
