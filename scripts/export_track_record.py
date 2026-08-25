"""Writes data/dashboard/app_track_record.json in isolation -- the one piece of run_report.py's
own output this weekly job needs, without also touching data/report_history/ snapshots or
latest_diff.json. Those are already fully owned by scheduled_pipeline.yml's own twice-daily
run_report.py call; writing them again here, against this job's own separately-restored DB,
would race that job for the same committed files (a real, avoidable conflict, not a
hypothetical one -- this job's DB snapshot and the twice-daily job's can legitimately disagree
on which gameweek is "current" by the time either one runs).

Exists because scripts/run_backtest.py is the only thing that ever populates a real
backtest_runs row, and it's a real ~1-2 hour job -- far too expensive for
scheduled_pipeline.yml's twice-daily cadence (see run_backtest.py's own module docstring), so
it runs on its own separate, weekly schedule instead. See run_report.py's own
_would_regress_track_record() for the other half of this: without that guard, the very next
twice-daily run -- starting from a fresh, backtest-less DB -- would silently wipe out what this
script just committed.

Usage (from repo root, after scripts/run_backtest.py has populated backtest_runs for real):
    PYTHONPATH=src python scripts/export_track_record.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import backtest, db, reporting  # noqa: E402

TARGET_SEASON = "2026-2027"
DASHBOARD_DIR = REPO_ROOT / "data" / "dashboard"
RECALIBRATION_SEED_DIR = REPO_ROOT / "data" / "recalibration"

# Same param-version set run_report.py's own ACTIVE_PARAM_VERSIONS uses -- duplicated rather
# than cross-imported, matching this project's own convention of small literal constants
# (TARGET_SEASON itself included) living independently in each script rather than sharing
# imports between sibling scripts/ files. See run_report.py's own comment on why the
# recalibratable families below resolve from active (Track B, docs/plans/2026-08_roadmap_plan.md)
# rather than staying hardcoded -- this transparency panel would otherwise misreport what's
# actually active. model_decay_params (xi/rho) is the one exception, left hardcoded at 1 --
# see run_report.py's own comment on why a single family-level number can't safely represent
# two independently-versioned keys here without either showing a stale value or dropping a row.
def _active_param_versions(active: dict) -> dict:
    return {
        "source_tier_weights": 1,
        "fact_type_multiplier_params": active["fact_multiplier_params_version"],
        "model_decay_params": 1,
        "minutes_adjustment_params": active["adjustment_params_version"],
        "minutes_model_decay_params": 1,
        "minutes_model_shrinkage_params": active["shrinkage_params_version"],
        "base_scoring_matrix": 1, "bps_formula_params": 1, "bps_dispersion_params": 1,
        "correlation_params": active["rho_residual_params_version"],
        "cross_player_correlation_params": 1,
        "risk_aversion_params": active["lambda_params_version"],
        "squad_optimizer_guardrail_params": 1, "planning_horizon_params": 1, "transfer_cost_params": 1,
        "tc_risk_aversion_params": active["kappa_tc_params_version"], "wildcard_gain_threshold_params": 1,
    }


def main() -> None:
    con = db.connect()
    _ACTIVE = backtest.active_recalibratable_versions(RECALIBRATION_SEED_DIR)
    ACTIVE_PARAM_VERSIONS = _active_param_versions(_ACTIVE)

    real_run_id = con.execute(
        "SELECT run_id FROM squad_optimizer_runs WHERE target_season = ? AND is_manager_snapshot = FALSE "
        "ORDER BY target_gameweek DESC LIMIT 1",
        [TARGET_SEASON],
    ).fetchone()
    if not real_run_id:
        raise SystemExit(f"no real squad_optimizer_runs row for {TARGET_SEASON} -- run scripts/run_ingestion.py first")
    real_run_id = real_run_id[0]

    backtest_run_id = con.execute("SELECT max(backtest_run_id) FROM backtest_runs").fetchone()[0]
    if backtest_run_id is None:
        raise SystemExit(
            "no backtest_runs row found -- run scripts/run_backtest.py first (this script only "
            "exports its result, it doesn't run the backtest itself)"
        )
    transfer_plan_run_id = con.execute("SELECT max(run_id) FROM transfer_plan_runs").fetchone()[0]

    # Only parameter_transparency (via build_report()) is actually needed here -- reused as-is
    # rather than duplicating params.transparency_panel()'s own logic a second time.
    report = reporting.build_report(
        con, real_run_id,
        transfer_plan_run_id=transfer_plan_run_id,
        backtest_run_id=backtest_run_id,
        active_param_versions=ACTIVE_PARAM_VERSIONS,
        ownership_params_version=1,
        sanity_check_params_version=1,
        consensus_check_params_version=1,
        evidence_decay_params_version=1,
        evidence_fact_multiplier_params_version=_ACTIVE["fact_multiplier_params_version"],
        bench_quality_params_version=1,
        confidence_score_params_version=1,
        report_asof=datetime.now(),
    )

    track_record = reporting.build_track_record_summary(con, report, backtest_run_id)
    track_record["generated_at"] = datetime.now().isoformat()
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    track_record_path = DASHBOARD_DIR / "app_track_record.json"
    track_record_path.write_text(json.dumps(track_record, indent=2))
    print(f"[dashboard] track record written to {track_record_path}")
    print(f"  backtest_run_id={backtest_run_id}, n_gameweek_steps={track_record['n_gameweek_steps']}, "
          f"parameters_backtested={track_record['parameters_backtested']}/{track_record['parameters_total']}")

    con.close()


if __name__ == "__main__":
    main()
