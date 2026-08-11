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
