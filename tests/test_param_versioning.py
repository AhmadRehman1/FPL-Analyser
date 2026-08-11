import pytest

from fpl_quant import params


def test_write_then_resolve(con):
    params.write_param(con, "risk_aversion_params", 1, "2026-08-10", "lambda_value", value_numeric=0.15)
    value_numeric, value_text = params.resolve_param(con, "risk_aversion_params", "lambda_value", 1)
    assert value_numeric == 0.15


def test_resolve_missing_raises(con):
    with pytest.raises(params.ParamNotFoundError):
        params.resolve_param(con, "risk_aversion_params", "lambda_value", 999)


def test_resolve_never_silently_defaults_to_zero(con):
    # M5's explicit requirement: an unpopulated lookup is a hard error, not a silent 0.
    with pytest.raises(params.ParamNotFoundError):
        params.resolve_param(con, "risk_aversion_params", "lambda_value", 1)


def test_version_is_immutable_against_different_value(con):
    params.write_param(con, "risk_aversion_params", 1, "2026-08-10", "lambda_value", value_numeric=0.15)
    with pytest.raises(ValueError):
        params.write_param(con, "risk_aversion_params", 1, "2026-08-10", "lambda_value", value_numeric=0.20)


def test_identical_rewrite_is_idempotent(con):
    params.write_param(con, "risk_aversion_params", 1, "2026-08-10", "lambda_value", value_numeric=0.15)
    params.write_param(con, "risk_aversion_params", 1, "2026-08-10", "lambda_value", value_numeric=0.15)
    rows = con.execute(
        "SELECT count(*) FROM param_versions WHERE param_family = 'risk_aversion_params'"
    ).fetchone()[0]
    assert rows == 1


def test_new_tuning_is_a_new_version_not_an_edit(con):
    params.write_param(con, "risk_aversion_params", 1, "2026-08-10", "lambda_value", value_numeric=0.15)
    params.write_param(con, "risk_aversion_params", 2, "2026-09-01", "lambda_value", value_numeric=0.20)
    v1, _ = params.resolve_param(con, "risk_aversion_params", "lambda_value", 1)
    v2, _ = params.resolve_param(con, "risk_aversion_params", "lambda_value", 2)
    assert v1 == 0.15
    assert v2 == 0.20


def test_dimensions_disambiguate_within_a_family(con):
    params.write_param(
        con, "source_tier_weights", 1, "2026-08-10", "tier_weight",
        value_numeric=1.0, dimensions={"source_type": "official"},
    )
    params.write_param(
        con, "source_tier_weights", 1, "2026-08-10", "tier_weight",
        value_numeric=0.4, dimensions={"source_type": "community"},
    )
    official, _ = params.resolve_param(
        con, "source_tier_weights", "tier_weight", 1, dimensions={"source_type": "official"}
    )
    community, _ = params.resolve_param(
        con, "source_tier_weights", "tier_weight", 1, dimensions={"source_type": "community"}
    )
    assert official == 1.0
    assert community == 0.4


def test_source_tier_weights_view_matches_generic_mechanism(con):
    params.write_param(
        con, "source_tier_weights", 1, "2026-08-10", "tier_weight",
        value_numeric=0.8, dimensions={"source_type": "journalist"},
    )
    row = con.execute(
        "SELECT tier_weight FROM source_tier_weights WHERE source_type = 'journalist' AND param_version = 1"
    ).fetchone()
    assert row[0] == 0.8
