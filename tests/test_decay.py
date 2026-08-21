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
# A6: seed_v1_params() -- claim_type_decay_params was confirmed never seeded anywhere in this
# project (every real caller has silently been getting decay=1.0, no decay at all).
# ============================================================

def test_seed_v1_params_covers_every_claim_type_with_a_real_evidence_blend_consumer(con):
    decay.seed_v1_params(con)
    # every claim_type minutes_model.py's _SHIFT_CLAIM_TYPES/predicted_xi handling and
    # expected_points.py's set-piece/role-shift functions actually pass to effective_weight()
    for claim_type in ("injury_status", "manager_tendency", "transfer_likelihood", "predicted_xi", "set_piece_order_override"):
        half_life, _ = params.resolve_param(con, "claim_type_decay_params", "decay_half_life_days", 1, dimensions={"claim_type": claim_type})
        assert half_life > 0


def test_seed_v1_params_predicted_xi_decays_faster_than_manager_tendency(con):
    """Real, deliberate ordering: lineup predictions go stale far faster than a manager's
    season-long rotation pattern -- confirms the seeded values aren't just "any positive
    number" but actually encode this claim-type-specific reasoning."""
    decay.seed_v1_params(con)
    predicted_xi_hl, _ = params.resolve_param(con, "claim_type_decay_params", "decay_half_life_days", 1, dimensions={"claim_type": "predicted_xi"})
    manager_tendency_hl, _ = params.resolve_param(con, "claim_type_decay_params", "decay_half_life_days", 1, dimensions={"claim_type": "manager_tendency"})
    assert predicted_xi_hl < manager_tendency_hl


def test_seed_v1_params_makes_decay_for_claim_type_actually_decay(con):
    """Regression test for the real gap itself: before this fix, ANY asof_date in the future
    of observed_date returned exactly 1.0 for a real claim_type (no decay at all) since no row
    existed to resolve. After seeding, a month-old predicted_xi claim must be measurably
    discounted relative to a fresh one."""
    decay.seed_v1_params(con)
    fresh = decay.decay_for_claim_type(con, "predicted_xi", date(2026, 8, 1), date(2026, 8, 1), params_version=1)
    month_old = decay.decay_for_claim_type(con, "predicted_xi", date(2026, 7, 1), date(2026, 8, 1), params_version=1)
    assert fresh == pytest.approx(1.0)
    assert month_old < fresh
