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
