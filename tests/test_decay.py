from datetime import date

import pytest

from fpl_quant import decay, params


def test_half_life_boundary():
    d = decay.decay(date(2026, 1, 1), date(2026, 1, 11), half_life_days=10)
    assert abs(d - 0.5) < 1e-9


def test_zero_elapsed_time_is_full_weight():
    d = decay.decay(date(2026, 1, 1), date(2026, 1, 1), half_life_days=10)
    assert d == 1.0


def test_two_half_lives():
    d = decay.decay(date(2026, 1, 1), date(2026, 1, 21), half_life_days=10)
    assert abs(d - 0.25) < 1e-9


def test_nonpositive_half_life_rejected():
    with pytest.raises(ValueError):
        decay.decay(date(2026, 1, 1), date(2026, 1, 2), half_life_days=0)


def test_unset_claim_type_defaults_to_1(con):
    # M0 spec: decay = 1.0 when a claim_type has no claim_type_decay_params row yet --
    # unset never silently zeroes evidence out.
    d = decay.decay_for_claim_type(con, "injury_status", date(2026, 1, 1), date(2026, 6, 1), params_version=1)
    assert d == 1.0


def test_set_claim_type_uses_configured_half_life(con):
    params.write_param(
        con, "claim_type_decay_params", 1, "2026-08-10", "decay_half_life_days",
        value_numeric=7, dimensions={"claim_type": "injury_status"},
    )
    d = decay.decay_for_claim_type(con, "injury_status", date(2026, 1, 1), date(2026, 1, 8), params_version=1)
    assert abs(d - 0.5) < 1e-9


# ============================================================
# Priority 7c -- seed_v1_params(): claim_type_decay_params had never been populated anywhere
# in this project before this (every claim type silently decayed at 1.0/no-decay via
# decay_for_claim_type's own ParamNotFoundError fallback) -- this is the first real config.
# ============================================================

def test_seed_v1_params_covers_every_real_claim_type(con):
    decay.seed_v1_params(con)
    for claim_type in decay.CLAIM_TYPE_DECAY_HALF_LIFE_DAYS_V1:
        half_life, _ = params.resolve_param(
            con, "claim_type_decay_params", "decay_half_life_days", 1, dimensions={"claim_type": claim_type},
        )
        assert half_life > 0


def test_seed_v1_params_makes_predicted_xi_decay_faster_than_manager_tendency(con):
    """The direct Priority 7c ask: same-day team news must decay much faster than a durable
    season-long tactical pattern."""
    decay.seed_v1_params(con)
    asof = date(2026, 1, 15)
    observed = date(2026, 1, 8)  # one week old
    xi_decay = decay.decay_for_claim_type(con, "predicted_xi", observed, asof, params_version=1)
    tendency_decay = decay.decay_for_claim_type(con, "manager_tendency", observed, asof, params_version=1)
    assert xi_decay < tendency_decay


def test_seed_v1_params_predicted_xi_is_heavily_stale_after_a_week(con):
    """Same-day team news (predicted_xi) at its configured half_life of 1.5 days: a claim from
    a week ago should be down to a small fraction of its original weight, matching the
    roadmap's own 'faster-decay for late team news' ask -- a week-old lineup prediction is
    close to worthless by the next deadline."""
    decay.seed_v1_params(con)
    d = decay.decay_for_claim_type(con, "predicted_xi", date(2026, 1, 1), date(2026, 1, 8), params_version=1)
    assert d < 0.05
