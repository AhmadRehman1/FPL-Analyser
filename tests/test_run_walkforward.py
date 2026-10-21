import sys
from pathlib import Path

import pytest

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
    # replaces the live start prior: minutes_model.run() takes one or the other
    assert out["minutes_start_prior_params_version"] is None


def test_no_minutes_start_prior_flag_switches_the_live_prior_off(con):
    assert rw._experiment_versions(con, rw._parse_args(["--no-minutes-start-prior"])) == {
        "minutes_start_prior_params_version": None,
    }


def test_minutes_start_prior_flag_maps_to_a_param_version(con):
    out = rw._experiment_versions(con, rw._parse_args(["--minutes-start-prior", "5"]))
    k, _ = params_mod.resolve_param(con, "minutes_start_prior_params", "pseudo_matches", out["minutes_start_prior_params_version"])
    assert k == 5.0
    assert rw._experiment_versions(con, rw._parse_args(["--minutes-start-prior", "5"])) == out


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


def test_team_strength_flag_maps_each_arm_to_a_guard_version(con):
    from fpl_quant import team_strength as ts

    ts.seed_team_strength_guard_params(con)  # as materialize_confirmed_seeds() does first
    assert rw._experiment_versions(con, rw._parse_args(["--team-strength", "fix-withheld"])) == {
        "team_strength_guard_params_version": 1,
    }
    for arm in ("honest", "live-like", "fix"):
        version = rw._experiment_versions(con, rw._parse_args(["--team-strength", arm]))["team_strength_guard_params_version"]
        assert ts.resolve_guard_params(con, version) == ts.GUARD_ARMS[arm]
    # the model before the guard went live, for comparisons
    assert rw._experiment_versions(con, rw._parse_args(["--team-strength", "off"])) == {
        "team_strength_guard_params_version": None,
    }


def test_rate_flags_change_the_live_rate_prior_bundle(con):
    from fpl_quant import expected_points as ep

    ep.seed_rate_prior_params(con)  # main() materializes it before any arm mints a version
    out = rw._experiment_versions(con, rw._parse_args(
        ["--rate-current-season-weight", "2", "--rate-season-decay", "0.5"],
    ))
    assert set(out) == {"rate_prior_params_version"}
    assert ep.resolve_rate_prior(con, out["rate_prior_params_version"]) == {
        "price_anchor": True, "current_season_weight": 2.0, "season_decay": 0.5, "anchor_goals": False,
    }
    # the recency knobs without the anchor
    no_anchor = rw._experiment_versions(con, rw._parse_args(["--no-rate-price-anchor", "--rate-season-decay", "0.5"]))
    assert ep.resolve_rate_prior(con, no_anchor["rate_prior_params_version"]) == {
        "price_anchor": False, "current_season_weight": 1.0, "season_decay": 0.5, "anchor_goals": False,
    }
    # the live bundle's own values find v2 (live since 2026-10-13) rather than minting a copy
    same = rw._experiment_versions(con, rw._parse_args(["--rate-current-season-weight", "1"]))
    assert same == {"rate_prior_params_version": 2}
    # the model before the anchor went live, for comparisons
    assert rw._experiment_versions(con, rw._parse_args(["--no-rate-price-anchor"])) == {
        "rate_prior_params_version": None,
    }


def test_k_minutes_flag_maps_to_a_rate_shrinkage_version(con):
    out = rw._experiment_versions(con, rw._parse_args(["--k-minutes", "450"]))
    k, _ = params_mod.resolve_param(con, "rate_shrinkage_params", "k_minutes", out["rate_shrinkage_params_version"])
    assert k == 450.0


def test_role_minutes_blend_flag_maps_to_a_current_season_minutes_version(con):
    out = rw._experiment_versions(con, rw._parse_args(["--role-matches-threshold", "4", "--role-minutes-blend"]))
    assert set(out) == {"current_season_role_params_version", "current_season_minutes_params_version"}
    threshold, _ = params_mod.resolve_param(
        con, "current_season_minutes_params", "starts_threshold", out["current_season_minutes_params_version"],
    )
    assert threshold == 4.0


def test_current_season_blend_flags_find_the_live_v1_or_switch_it_off(con):
    from fpl_quant import minutes_model as mm
    mm.seed_current_season_role_params(con)
    mm.seed_current_season_minutes_params(con)
    # the live values find v1 rather than minting a copy
    assert rw._experiment_versions(con, rw._parse_args(["--role-matches-threshold", "4", "--role-minutes-blend"])) == {
        "current_season_role_params_version": 1, "current_season_minutes_params_version": 1,
    }
    # the model before the blends went live, for comparisons
    assert rw._experiment_versions(con, rw._parse_args(["--no-current-season-blend"])) == {
        "current_season_role_params_version": None, "current_season_minutes_params_version": None,
    }
    # the gentler P(60+) form is its own version, holding only pseudo_starts
    out = rw._experiment_versions(con, rw._parse_args(["--role-minutes-pseudo-starts", "4"]))
    version = out["current_season_minutes_params_version"]
    assert version not in (None, 1)
    assert params_mod.resolve_param(con, "current_season_minutes_params", "pseudo_starts", version)[0] == 4.0
    assert rw._experiment_versions(con, rw._parse_args(["--role-minutes-pseudo-starts", "4"])) == out


def test_league_only_minutes_rates_flag_maps_to_a_scope_version(con):
    out = rw._experiment_versions(con, rw._parse_args(["--league-only-minutes-rates"]))
    assert set(out) == {"minutes_rates_scope_params_version"}
    value, _ = params_mod.resolve_param(con, "minutes_rates_scope_params", "league_only", out["minutes_rates_scope_params_version"])
    assert value == 1.0


def test_band_conversion_flag_maps_to_a_param_version(con):
    out = rw._experiment_versions(con, rw._parse_args(["--band-conversion", "20"]))
    k, _ = params_mod.resolve_param(con, "conversion_params", "pseudo_xg", out["conversion_params_version"])
    assert k == 20.0


def test_no_goal_anchor_flag_mints_a_bundle_with_anchor_goals_off(con):
    out = rw._experiment_versions(con, rw._parse_args(["--no-goal-anchor"]))
    prior = rw.ep.resolve_rate_prior(con, out["rate_prior_params_version"])
    assert prior == {"price_anchor": True, "current_season_weight": 1.0, "season_decay": 1.0, "anchor_goals": False}
    # with the whole anchor off, the goals-only switch has nothing left to do
    assert rw._experiment_versions(con, rw._parse_args(["--no-goal-anchor", "--no-rate-price-anchor"])) == {
        "rate_prior_params_version": None,
    }


def test_goal_anchor_flag_is_the_kill_switch_for_the_live_goals_only_anchor(con):
    # lever Z went live on 2026-10-13 as rate_prior_params v2; --goal-anchor is its --no-<lever>
    # switch (Req 8): exactly the v1 rate prior that was live before, goals anchored to price
    rw.ep.seed_rate_prior_params(con)
    active = rw.backtest.active_recalibratable_versions(rw.RECALIBRATION_SEED_DIR)
    assert active["rate_prior_params_version"] == 2
    assert rw.ep.resolve_rate_prior(con, 2)["anchor_goals"] is False
    assert rw._experiment_versions(con, rw._parse_args(["--no-goal-anchor"])) == {"rate_prior_params_version": 2}
    assert rw._experiment_versions(con, rw._parse_args(["--goal-anchor"])) == {"rate_prior_params_version": 1}
    assert "anchor_goals" not in rw.ep.resolve_rate_prior(con, 1)  # goals anchored
    # with a recency knob, the live bundle with the goals anchored
    out = rw._experiment_versions(con, rw._parse_args(["--goal-anchor", "--rate-season-decay", "0.5"]))
    assert rw.ep.resolve_rate_prior(con, out["rate_prior_params_version"]) == {
        "price_anchor": True, "current_season_weight": 1.0, "season_decay": 0.5, "anchor_goals": True,
    }
    with pytest.raises(SystemExit):
        rw._parse_args(["--goal-anchor", "--no-goal-anchor"])


def test_own_sub_rate_flag_maps_to_a_sub_rate_version(con):
    out = rw._experiment_versions(con, rw._parse_args(["--own-sub-rate"]))
    assert set(out) == {"sub_rate_params_version"}
    value, _ = params_mod.resolve_param(con, "minutes_sub_rate_params", "own_rate", out["sub_rate_params_version"])
    assert value == 1.0


def test_no_own_sub_rate_flag_turns_the_live_switch_off(con):
    assert rw._experiment_versions(con, rw._parse_args(["--no-own-sub-rate"])) == {"sub_rate_params_version": None}


def test_template_flags_are_not_experiment_settings(con):
    args = rw._parse_args(["--no-template", "--template-lag", "2"])
    assert args.no_template and args.template_lag == 2
    defaults = rw._parse_args([])
    assert not defaults.no_template and defaults.template_lag == 1
    assert rw._experiment_versions(con, args) == {}


def test_export_residuals_is_not_an_experiment_setting(con):
    args = rw._parse_args(["--export-residuals"])
    assert args.export_residuals == "residual_features.csv"
    assert rw._parse_args(["--export-residuals", "out.csv"]).export_residuals == "out.csv"
    assert rw._parse_args([]).export_residuals is None
    assert rw._experiment_versions(con, args) == {}


def test_phase3_measurement_flags_reach_the_run(con):
    from fpl_quant import squad_optimizer as so

    so.seed_v1_params(con)
    out = rw._experiment_versions(con, rw._parse_args(["--availability-lag", "2", "--live-eo-anchor"]))
    assert out == {"availability_lag": 2, "solve_ownership_params_version": 1, "solve_risk_posture_params_version": 2}
    kappa, _ = rw.params_mod.resolve_param(con, "risk_posture_params", "eo_weight_kappa", 2)
    assert kappa == 0.12
    assert rw._experiment_versions(con, rw._parse_args([])) == {}


def test_eo_anchor_kappa_mints_a_protect_posture_version(con):
    from fpl_quant import squad_optimizer as so

    so.seed_v1_params(con)
    so.seed_template_anchor_params(con)
    out = rw._experiment_versions(con, rw._parse_args(["--eo-anchor-kappa", "0.06"]))
    assert out["solve_ownership_params_version"] == 1
    version = out["solve_risk_posture_params_version"]
    assert version not in (1, 2)
    kappa, _ = rw.params_mod.resolve_param(con, "risk_posture_params", "eo_weight_kappa", version)
    _, posture = rw.params_mod.resolve_param(con, "risk_posture_params", "posture", version)
    assert (kappa, posture) == (0.06, "protect")
    # the same kappa again reuses the version; the live 0.12 resolves to v2 itself
    assert rw._experiment_versions(con, rw._parse_args(["--eo-anchor-kappa", "0.06"])) == out
    assert so.template_anchor_version(con, 0.12, rw.EXPERIMENT_EFFECTIVE_DATE) == 2


def test_eo_anchor_kappa_and_live_eo_anchor_are_exclusive():
    with pytest.raises(SystemExit):
        rw._parse_args(["--live-eo-anchor", "--eo-anchor-kappa", "0.06"])


def test_eo_anchor_kappa_must_be_positive():
    with pytest.raises(SystemExit):
        rw._parse_args(["--eo-anchor-kappa", "0"])
