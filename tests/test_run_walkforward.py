import sys
from pathlib import Path

from fpl_quant import params as params_mod

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_walkforward as rw  # noqa: E402


def test_no_flags_is_the_live_control(con):
    assert rw._experiment_versions(con, rw._parse_args([])) == {}


def test_flags_map_to_param_versions(con):
    args = rw._parse_args(["--lambda", "0.10", "--role-matches-threshold", "4", "--assist-prior-xa", "30"])
    out = rw._experiment_versions(con, args)
    assert set(out) == {"lambda_params_version", "current_season_role_params_version", "assist_calibration_params_version"}
    lam, _ = params_mod.resolve_param(con, "risk_aversion_params", "lambda_value", out["lambda_params_version"])
    prior, _ = params_mod.resolve_param(con, "fpl_assist_calibration_params", "prior_xa", out["assist_calibration_params_version"])
    assert lam == 0.10
    assert prior == 30.0
    # the same flags again reuse the same versions rather than minting new ones
    assert rw._experiment_versions(con, args) == out


def test_backtest_evidence_flag_passes_through_as_a_run_kwarg(con):
    assert rw._experiment_versions(con, rw._parse_args(["--backtest-evidence"])) == {"backtest_evidence": True}


def test_finishing_prior_flag_maps_to_a_param_version(con):
    out = rw._experiment_versions(con, rw._parse_args(["--finishing-prior-xg", "10"]))
    prior, _ = params_mod.resolve_param(con, "finishing_skill_params", "prior_xg", out["finishing_skill_params_version"])
    assert prior == 10.0


def test_minutes_price_prior_flag_maps_to_a_param_version(con):
    out = rw._experiment_versions(con, rw._parse_args(["--minutes-price-prior", "50"]))
    weight, _ = params_mod.resolve_param(con, "minutes_price_prior_params", "min_band_weight", out["minutes_price_prior_params_version"])
    assert weight == 50.0


def test_run_length_flags_are_not_experiment_settings(con):
    """--max-minutes / --resume / --seasons shape the run, they don't change the model; the
    workflow puts --max-minutes first so an arm's own value overrides it."""
    args = rw._parse_args(["--max-minutes", "300", "--resume", "5", "--seasons", "2025-2026", "--max-minutes", "120"])
    assert (args.max_minutes, args.resume, args.seasons) == (120.0, 5, "2025-2026")
    assert rw._experiment_versions(con, args) == {}


def test_bonus_model_flags_map_to_param_versions(con):
    from fpl_quant import expected_points as ep

    ep.seed_v1_params(con)  # tau v1 = 10
    out = rw._experiment_versions(con, rw._parse_args(["--bps-calibration-k", "450", "--bps-tau", "7"]))
    k, _ = params_mod.resolve_param(con, "bps_calibration_params", "k_minutes", out["bps_calibration_params_version"])
    tau, _ = params_mod.resolve_param(con, "bps_dispersion_params", "tau", out["tau_params_version"])
    assert (k, tau) == (450.0, 7.0)
    assert out["tau_params_version"] != 1
    # the live tau value reuses v1
    assert rw._experiment_versions(con, rw._parse_args(["--bps-tau", "10"])) == {"tau_params_version": 1}
