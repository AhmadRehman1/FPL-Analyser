import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_report import _resolve_report_run_id, _would_regress_track_record  # noqa: E402


def _seed_model_versions(con):
    con.execute("INSERT INTO team_strength_model_versions (calibration_asof_date, home_advantage, xi_params_version, "
                "rho_params_version, reference_team_uid) VALUES ('2026-08-01', 0.2, 1, 1, 'clubA')")
    ts = con.execute("SELECT max(model_version) FROM team_strength_model_versions").fetchone()[0]
    con.execute("INSERT INTO minutes_model_versions (calibration_asof_date, target_season, decay_params_version, "
                "adjustment_params_version, shrinkage_params_version, fact_multiplier_params_version, lookback_seasons) "
                "VALUES ('2026-08-01', '2026-2027', 1, 1, 1, 1, '[]')")
    mm = con.execute("SELECT max(model_version) FROM minutes_model_versions").fetchone()[0]
    ep = con.execute("INSERT INTO ep_model_versions (calibration_asof_date, target_season, team_strength_model_version, "
                     "minutes_model_version, scoring_matrix_params_version, bps_params_version, bps_tau_params_version) "
                     "VALUES ('2026-08-01', '2026-2027', ?, ?, 1, 1, 1) RETURNING model_version", [ts, mm]).fetchone()[0]
    un = con.execute("INSERT INTO uncertainty_model_versions (calibration_asof_date, ep_model_version, "
                     "minutes_model_version, team_strength_model_version, rho_residual_params_version) "
                     "VALUES ('2026-08-01', ?, ?, ?, 1) RETURNING model_version", [ep, mm, ts]).fetchone()[0]
    return ep, un


def _seed_run(con, run_id, gw, is_manager_snapshot, ep, un):
    con.execute(
        "INSERT INTO squad_optimizer_runs (run_id, calibration_asof_date, target_season, target_gameweek, "
        "ep_model_version, uncertainty_model_version, lambda_params_version, lambda_value, guardrail_params_version, "
        "divergence_check_passed, solver_status, is_manager_snapshot) "
        "VALUES (?, '2026-08-01', '2026-2027', ?, ?, ?, 1, 0.15, 1, TRUE, 'optimal', ?)",
        [run_id, gw, ep, un, is_manager_snapshot],
    )


def test_resolve_report_run_id_ignores_future_chip_roadmap_solves(con):
    # print_chip_timing_roadmap.py's evaluate_wildcard() creates real is_manager_snapshot=FALSE
    # rows at fixture-swing gameweeks 10-19 -- the report must NOT snapshot those as "now".
    ep, un = _seed_model_versions(con)
    _seed_run(con, 10, gw=1, is_manager_snapshot=False, ep=ep, un=un)   # from-scratch GW1 solve (ingestion)
    _seed_run(con, 20, gw=3, is_manager_snapshot=False, ep=ep, un=un)   # the real current-GW wildcard-eval solve
    _seed_run(con, 21, gw=3, is_manager_snapshot=True, ep=ep, un=un)    # the real-squad bootstrap
    _seed_run(con, 30, gw=12, is_manager_snapshot=False, ep=ep, un=un)  # chip-roadmap swing-week solve -- MUST be ignored
    _seed_run(con, 31, gw=14, is_manager_snapshot=False, ep=ep, un=un)  # ditto

    assert _resolve_report_run_id(con, current_event=3) == 20
    # bare local run (no event): ceiling from the newest manager-snapshot's gameweek (3)
    assert _resolve_report_run_id(con, current_event=None) == 20


def test_no_regression_guard_needed_when_this_run_has_a_real_backtest():
    # This run's own DB has a real backtest_run_id -- always write it, whatever was there before.
    new = {"backtest_run_id": 7, "n_gameweek_steps": 71}
    existing = {"backtest_run_id": 3, "n_gameweek_steps": 40}
    assert _would_regress_track_record(new, existing) is False


def test_blocks_overwriting_a_real_committed_backtest_with_an_empty_one():
    # The exact real scenario: a weekly scripts/run_backtest.py run committed a real track
    # record, then this script's own next twice-daily run -- against a fresh, backtest-less
    # DB -- would otherwise silently wipe it back to "no backtest yet".
    new = {"backtest_run_id": None, "n_gameweek_steps": None}
    existing = {"backtest_run_id": 3, "n_gameweek_steps": 40}
    assert _would_regress_track_record(new, existing) is True


def test_no_regression_guard_needed_when_nothing_committed_yet():
    new = {"backtest_run_id": None, "n_gameweek_steps": None}
    assert _would_regress_track_record(new, existing_track_record=None) is False


def test_no_regression_guard_needed_when_existing_file_also_has_no_backtest():
    # Nothing real to lose -- both this run and the last committed file are placeholders.
    new = {"backtest_run_id": None, "n_gameweek_steps": None}
    existing = {"backtest_run_id": None, "n_gameweek_steps": None}
    assert _would_regress_track_record(new, existing) is False
