"""Pinned exponential evidence-decay formula (M0 spec).

decay = 1.0 when a claim_type has no claim_type_decay_params row yet -- unset never
silently zeroes evidence out.
"""

from datetime import date

from . import params as params_mod


def decay(observed_date: date, asof_date: date, half_life_days: float) -> float:
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    days = (asof_date - observed_date).days
    return 0.5 ** (days / half_life_days)


def decay_for_claim_type(con, claim_type: str, observed_date: date, asof_date: date, params_version: int) -> float:
    try:
        value_numeric, _ = params_mod.resolve_param(
            con, "claim_type_decay_params", "decay_half_life_days", params_version,
            dimensions={"claim_type": claim_type},
        )
    except params_mod.ParamNotFoundError:
        return 1.0
    if value_numeric is None:
        return 1.0
    return decay(observed_date, asof_date, value_numeric)
