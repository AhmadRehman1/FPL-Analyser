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


def test_seed_v1_params_orders_role_claims_by_how_fast_the_role_turns_over(con):
    """A starting-role read (predicted_xi) turns over faster than a set-piece rota, which turns
    over faster than a manager's season-long tendency -- so their half-lives are strictly
    increasing in that order."""
    decay.seed_v1_params(con)
    asof, observed = date(2026, 1, 15), date(2026, 1, 8)  # one week old

    def d(claim_type):
        return decay.decay_for_claim_type(con, claim_type, observed, asof, params_version=1)

    assert d("predicted_xi") < d("set_piece_order_override") < d("manager_tendency")


def test_seed_v1_params_predicted_xi_survives_a_weekly_pull_cycle(con):
    """predicted_xi claims come from the *weekly* manual evidence pull and predict a season
    starting role, not a same-day lineup. A claim must still carry most of its weight a week
    later (one pull cycle) and roughly half its weight after three weeks -- the old 1.5-day
    half-life instead zeroed the whole channel (~1e-3 weight after a week), defeating the
    evidence-pull feature minutes_model.compute_logit_adjustment() reads it for."""
    decay.seed_v1_params(con)
    after_1w = decay.decay_for_claim_type(con, "predicted_xi", date(2026, 1, 1), date(2026, 1, 8), params_version=1)
    after_3w = decay.decay_for_claim_type(con, "predicted_xi", date(2026, 1, 1), date(2026, 1, 22), params_version=1)
    assert after_1w > 0.75
    assert 0.45 < after_3w < 0.55
