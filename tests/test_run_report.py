import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_report import (  # noqa: E402
    _merge_track_record_onto_committed, _resolve_report_run_id, _would_regress_track_record,
)


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


# ============================================================
# _merge_track_record_onto_committed: regression tests for the 2026-09 headline-suppression bug
# (real top-level backtest_run_id/headline committed correctly, but transparency_log.backtest --
# the ONLY thing track-record.html actually reads -- silently stuck at a null headline forever,
# because export_track_record.py never wrote transparency_log at all).
# ============================================================

def _fresh_backtest_less_track_record(transparency_log_backtest=None):
    return {
        "backtest_run_id": None, "n_gameweek_steps": None, "headline": None,
        "planner_decision_accuracy": {"followed": 3, "total": 5},
        "generated_at": "2026-09-16T00:00:00",
        "transparency_log": {
            "backtest": transparency_log_backtest or {
                "n_gameweek_steps": None, "seasons_covered": [], "headline": None,
                "metrics": [], "parameters_total": None, "parameters_backtested": None,
            },
            "snapshots": [{"season": "2026-2027", "gameweek": 5}],
            "latest_diff": {"some": "diff"},
            "provenance": {"git_sha": "abc123"},
        },
    }


def test_merge_preserves_a_correctly_populated_committed_transparency_log():
    fresh = _fresh_backtest_less_track_record()
    existing = {
        "backtest_run_id": 1, "n_gameweek_steps": 71,
        "headline": {"beats_avg_manager_by_points_per_gw": -0.88},
        "transparency_log": {
            "backtest": {
                "n_gameweek_steps": 71, "seasons_covered": ["2024-2025", "2025-2026"],
                "headline": {"beats_avg_manager_by_points_per_gw": -0.88},
                "metrics": [{"metric_name": "x", "mean_value": 1.0, "n_observations": 71}],
                "parameters_total": 62, "parameters_backtested": 8,
            },
        },
    }
    merged = _merge_track_record_onto_committed(fresh, existing)
    assert merged["transparency_log"]["backtest"]["headline"] == {"beats_avg_manager_by_points_per_gw": -0.88}
    assert merged["transparency_log"]["backtest"]["n_gameweek_steps"] == 71
    # daily-cadence halves still refresh from the fresh run
    assert merged["planner_decision_accuracy"] == {"followed": 3, "total": 5}
    assert merged["transparency_log"]["snapshots"] == fresh["transparency_log"]["snapshots"]
    assert merged["backtest_run_id"] == 1  # top-level committed data preserved


def test_merge_self_heals_a_corrupted_transparency_log_with_a_null_headline():
    # The exact real bug: top-level headline/backtest_run_id are real (export_track_record.py
    # DID write those), but transparency_log.backtest was never populated by that script, so a
    # committed file could look exactly like this -- a truthy dict whose headline is None.
    fresh = _fresh_backtest_less_track_record()
    existing = {
        "backtest_run_id": 1, "n_gameweek_steps": 71,
        "headline": {"beats_avg_manager_by_points_per_gw": -0.88},
        "metrics": [{"metric_name": "x", "mean_value": 1.0, "n_observations": 71}],
        "seasons_covered": ["2024-2025", "2025-2026"],
        "parameters_total": 62, "parameters_backtested": 8,
        "transparency_log": {
            "backtest": {
                "n_gameweek_steps": None, "seasons_covered": [], "headline": None,
                "metrics": [], "parameters_total": None, "parameters_backtested": None,
            },
        },
    }
    merged = _merge_track_record_onto_committed(fresh, existing)
    assert merged["transparency_log"]["backtest"]["headline"] == {"beats_avg_manager_by_points_per_gw": -0.88}, (
        "the real committed headline must be recoverable from existing_track_record's own "
        "top-level fields even when transparency_log.backtest was never correctly populated -- "
        "this is the exact bug that silenced track-record.html's oracle headline in every "
        "commit of app_track_record.json until this fix"
    )
    assert merged["transparency_log"]["backtest"]["n_gameweek_steps"] == 71


def test_merge_stays_honestly_none_when_nothing_real_was_ever_committed():
    fresh = _fresh_backtest_less_track_record()
    existing = {
        "backtest_run_id": None, "n_gameweek_steps": None, "headline": None,
        "transparency_log": {"backtest": {"n_gameweek_steps": None, "headline": None}},
    }
    merged = _merge_track_record_onto_committed(fresh, existing)
    assert merged["transparency_log"]["backtest"]["headline"] is None
