"""App gap 6: a small, user-facing risk-posture control (Plan tab) that maps to a FIXED, tested
pair of squad_optimizer / transfer_planner parameter versions -- deliberately NOT a raw slider
over uncalibrated internals (arbitrary lambda_value / kappa_tc could produce unstable or
untested optimizer behaviour -- see the app-feature-gaps prompt's own constraint).

Two postures ship now:

  balanced  Tracks the project's own current default calibration -- run_transfer_planner_for_
            real_squad.py resolves it from active_recalibratable_versions() (the same
            git-committed confirmed-seed files every other real-data script reads), NOT this
            module's own frozen v1 (0.15, 0.15) spec literals below. Those literals are the
            fallback only for a caller with no live connection/active dict (posture_meta()'s
            own docstring) -- when this WAS written, active[...] and v1 were identical, but a
            real recalibration (e.g. kappa_tc v1->v3, 0.15->0.2, confirmed 2026-09-08) can move
            them apart at any time. "balanced" is a no-op relative to whatever's currently
            active, not to these fixed numbers.

  attack    lambda_value v2 (0.05), kappa_tc v2 (0.5). A LOWER squad-concentration penalty (the
            MIQP is freer to stack correlated premium picks) plus a HIGHER captaincy-variance
            tolerance. Both are 'confirmed' recalibration proposals from backtest_run_id=1 --
            real, backtested values (realized Sharpe on the walk-forward: lambda 0.15->0.05 was
            3.52->4.27; kappa_tc 0.15->0.5 was 1.02->1.06) -- just never promoted to the active
            default. So this exposes a pre-validated alternative, not a guess.

A third 'protect' posture (a HIGHER lambda_value than 0.15, for rank protection) is deliberately
NOT shipped: no such version has been backtested yet. It is gated on the lambda-sensitivity
study (lambda in {0.05 .. 0.30}) already noted as pending elsewhere in this project.

Version numbers for these values are resolved dynamically (params.get_or_create_version()),
never hardcoded, as of the 2026-09 fix documented in
docs/reports/2026-09_model_failure_diagnosis.md section 9: a hardcoded "attack" version 2
collided in production with a later, genuinely-confirmed recalibration proposal that also
minted tc_risk_aversion_params version 2 (a different value, 0.2) -- write_param()'s
immutability check raised on every attack-posture solve from the moment that confirmation
landed, silently, because the scheduled workflow runs that step under continue-on-error. This
module now only ever asserts VALUES; whatever version number ends up holding them is an
implementation detail resolve_versions() looks up or mints fresh, and can never collide with
a version some other lineage already claimed for a different value.
"""

import duckdb

from . import params as params_mod

DEFAULT_POSTURE = "balanced"

# Values per family -- NOT paired with a hardcoded params_version (see this module's own
# docstring above for why). resolve_versions() looks up or mints the real version dynamically.
_POSTURES: dict[str, dict] = {
    "balanced": {
        "label": "Balanced",
        "blurb": "The model's default calibration.",
        "lambda_value": 0.15,
        "kappa_tc": 0.15,
    },
    "attack": {
        "label": "Attack rank",
        "blurb": "Lets the squad concentrate more (lower diversification penalty) and tolerates "
                 "a higher-variance captain -- more upside, more downside.",
        "lambda_value": 0.05,
        "kappa_tc": 0.5,
    },
}

POSTURES = tuple(_POSTURES)

# effective_date is informational only on an immutable param row; use the same date the
# backtest_run_id=1 recalibration proposals carry.
_EFFECTIVE_DATE = "2026-08-30"


def is_valid(posture: str) -> bool:
    return posture in _POSTURES


def normalize(posture: str | None) -> str:
    """A missing / unknown posture falls back to the default rather than raising -- a stale
    localStorage value or a typo in a workflow input must never break a scheduled run."""
    return posture if posture in _POSTURES else DEFAULT_POSTURE


def posture_meta(
    posture: str, *, con: duckdb.DuckDBPyConnection | None = None, active: dict | None = None,
) -> dict:
    """con/active (both optional, default None -- exact prior behavior when either is omitted):
    for the DEFAULT_POSTURE ('balanced') only, resolves lambda_value/kappa_tc from the CURRENT
    live active_recalibratable_versions() versions instead of this module's frozen v1 (0.15,
    0.15) spec literals. Needed because run_transfer_planner_for_real_squad.py's own real solve
    already does exactly this override (this module's docstring's "balanced -> {v1, v1},
    identical to active[...] today" was true when written, but stopped being true the moment
    lambda_params_version/kappa_tc_params_version were ever recalibrated away from v1 --
    data/recalibration/seeds_1.json's confirmed kappa_tc v1->v3 (0.15->0.2), 2026-09-08, is
    exactly that: without this, the exported real_squad_<id>.json's own "risk_posture" display
    block would keep showing the stale 0.15 kappa_tc as "the model's calibration" while the
    transfer/captain recommendations right next to it in the same file were actually computed
    with 0.2). 'attack' is unaffected -- it's an intentionally pinned, pre-validated snapshot
    (see this module's own docstring), not meant to track live recalibration."""
    resolved = normalize(posture)
    p = _POSTURES[resolved]
    lambda_value, kappa_tc = p["lambda_value"], p["kappa_tc"]
    if resolved == DEFAULT_POSTURE and con is not None and active is not None:
        lambda_value, _ = params_mod.resolve_param(con, "risk_aversion_params", "lambda_value", active["lambda_params_version"])
        kappa_tc, _ = params_mod.resolve_param(con, "tc_risk_aversion_params", "kappa_tc", active["kappa_tc_params_version"])
    return {
        "posture": resolved,
        "label": p["label"],
        "blurb": p["blurb"],
        "lambda_value": lambda_value,
        "kappa_tc": kappa_tc,
    }


def resolve_versions(con: duckdb.DuckDBPyConnection, posture: str) -> dict:
    """Ensures this posture's parameter values exist in param_versions under SOME version
    (looked up if a matching row already exists, minted fresh via
    params.get_or_create_version() otherwise -- never a hardcoded literal, see this module's
    own docstring) and returns the versions to hand to transfer_planner.run() /
    squad_optimizer:

        {"lambda_params_version": int, "kappa_tc_params_version": int}

    Raises on an unknown posture (a workflow / CLI wiring bug should fail loudly, unlike a
    stale client value -- see normalize())."""
    if posture not in _POSTURES:
        raise ValueError(f"unknown risk posture {posture!r} -- expected one of {POSTURES}")
    p = _POSTURES[posture]
    lam_ver = params_mod.get_or_create_version(
        con, "risk_aversion_params", "lambda_value", _EFFECTIVE_DATE, value_numeric=p["lambda_value"],
    )
    kap_ver = params_mod.get_or_create_version(
        con, "tc_risk_aversion_params", "kappa_tc", _EFFECTIVE_DATE, value_numeric=p["kappa_tc"],
    )
    return {"lambda_params_version": lam_ver, "kappa_tc_params_version": kap_ver}
