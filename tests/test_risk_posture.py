"""App gap 6: the risk-posture -> parameter-version mapping. The prompt's requirement #4 is
"confirm the correct parameter versions are passed through" -- resolve_versions() IS that
mapping (run_transfer_planner_for_real_squad.py feeds its return straight into
transfer_planner.run()'s lambda_params_version / kappa_tc_params_version)."""

import pytest

from fpl_quant import db, params, risk_posture


@pytest.fixture
def con():
    c = db.connect(":memory:")
    yield c
    c.close()


def test_balanced_resolves_to_the_v1_defaults(con):
    assert risk_posture.resolve_versions(con, "balanced") == {
        "lambda_params_version": 1, "kappa_tc_params_version": 1,
    }
    assert params.resolve_param(con, "risk_aversion_params", "lambda_value", 1)[0] == pytest.approx(0.15)
    assert params.resolve_param(con, "tc_risk_aversion_params", "kappa_tc", 1)[0] == pytest.approx(0.15)


def test_attack_resolves_to_the_confirmed_v2_values(con):
    assert risk_posture.resolve_versions(con, "attack") == {
        "lambda_params_version": 2, "kappa_tc_params_version": 2,
    }
    assert params.resolve_param(con, "risk_aversion_params", "lambda_value", 2)[0] == pytest.approx(0.05)
    assert params.resolve_param(con, "tc_risk_aversion_params", "kappa_tc", 2)[0] == pytest.approx(0.5)


def test_resolve_versions_only_touches_lambda_and_kappa_tc(con):
    risk_posture.resolve_versions(con, "attack")
    families = {r[0] for r in con.execute("SELECT DISTINCT param_family FROM param_versions").fetchall()}
    assert families == {"risk_aversion_params", "tc_risk_aversion_params"}


def test_resolve_versions_is_idempotent(con):
    a = risk_posture.resolve_versions(con, "attack")
    b = risk_posture.resolve_versions(con, "attack")
    assert a == b
    # no duplicate rows
    n = con.execute("SELECT count(*) FROM param_versions WHERE param_family = 'risk_aversion_params' AND param_version = 2").fetchone()[0]
    assert n == 1


def test_unknown_posture_raises(con):
    with pytest.raises(ValueError, match="unknown risk posture"):
        risk_posture.resolve_versions(con, "protect")


def test_normalize_falls_back_to_balanced_for_a_stale_or_missing_value():
    assert risk_posture.normalize(None) == "balanced"
    assert risk_posture.normalize("") == "balanced"
    assert risk_posture.normalize("protect") == "balanced"
    assert risk_posture.normalize("attack") == "attack"


def test_posture_meta_shape():
    m = risk_posture.posture_meta("attack")
    assert m["posture"] == "attack"
    assert m["label"] == "Attack rank"
    assert m["lambda_value"] == pytest.approx(0.05)
    assert m["kappa_tc"] == pytest.approx(0.5)
    assert isinstance(m["blurb"], str) and m["blurb"]


def test_the_planner_script_feeds_resolve_versions_into_transfer_planner_run():
    """Wiring guard: the real-squad planner script must resolve lambda/kappa_tc from the posture
    and pass those exact versions to tp.run() -- not the hardcoded active ones -- so the attack
    variant is genuinely a different solve."""
    import pathlib
    src = pathlib.Path(__file__).resolve().parents[1].joinpath("scripts", "run_transfer_planner_for_real_squad.py").read_text()
    assert "risk_posture.resolve_versions(con, posture)" in src
    assert "lambda_params_version=lambda_params_version" in src
    assert "kappa_tc_params_version=kappa_tc_params_version" in src
