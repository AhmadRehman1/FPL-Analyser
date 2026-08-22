"""M9: Reporting / Explainability Layer -- builds and prints a real squad report from the
actual project database.

Usage (from repo root):
    .venv/Scripts/python scripts/run_report.py

Uses the real GW1 2026-27 squad_optimizer_runs row, plus the real M7 backtest_run_id and M8
transfer_plan_run_id already sitting in the database from earlier milestones' real runs.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import db, reporting  # noqa: E402

TARGET_SEASON = "2026-2027"
# Priority 8c: small, committed per-gameweek report snapshots -- see reporting.py's own
# module docstring for why this lives as plain JSON in the repo rather than in the (gitignored,
# never-persisted-across-runs) DuckDB file itself.
REPORT_HISTORY_DIR = REPO_ROOT / "data" / "report_history"

# Every param family currently seeded at v1 across M1-M8 (run_ingestion.py's own list).
ACTIVE_PARAM_VERSIONS = {
    "source_tier_weights": 1, "fact_type_multiplier_params": 1, "model_decay_params": 1,
    "minutes_adjustment_params": 1, "minutes_model_decay_params": 1, "minutes_model_shrinkage_params": 1,
    "base_scoring_matrix": 1, "bps_formula_params": 1, "bps_dispersion_params": 1,
    "correlation_params": 1, "cross_player_correlation_params": 1, "risk_aversion_params": 1,
    "squad_optimizer_guardrail_params": 1, "planning_horizon_params": 1, "transfer_cost_params": 1,
    "tc_risk_aversion_params": 1, "wildcard_gain_threshold_params": 1,
}


def main() -> None:
    con = db.connect()

    # Latest real (non-manager-snapshot) run for the season, not hardcoded to GW1 -- Priority
    # 8c's week-over-week diff is only useful across an advancing season, and a scheduled
    # weekly run (Priority 8a) needs this to naturally pick up each new gameweek's run.
    real_run_id = con.execute(
        "SELECT run_id FROM squad_optimizer_runs WHERE target_season = ? AND is_manager_snapshot = FALSE "
        "ORDER BY target_gameweek DESC LIMIT 1",
        [TARGET_SEASON],
    ).fetchone()
    if not real_run_id:
        raise SystemExit(f"no real squad_optimizer_runs row for {TARGET_SEASON} -- run scripts/run_ingestion.py first")
    real_run_id = real_run_id[0]

    backtest_run_id = con.execute("SELECT max(backtest_run_id) FROM backtest_runs").fetchone()[0]
    transfer_plan_run_id = con.execute("SELECT max(run_id) FROM transfer_plan_runs").fetchone()[0]

    # Every Priority 1/2/6 opt-in section wired to its real v1 params (all seeded by
    # reporting.seed_v1_params()/squad_optimizer.seed_v1_params() in run_ingestion.py) --
    # this is what makes captain_risk_eo, consensus_divergence, adversarial_review, and the
    # evidence_weight half of confidence_scores genuinely STANDING sections of every real
    # report, not one-off opt-ins only exercised in tests.
    report = reporting.build_report(
        con, real_run_id,
        transfer_plan_run_id=transfer_plan_run_id,
        backtest_run_id=backtest_run_id,
        active_param_versions=ACTIVE_PARAM_VERSIONS,
        ownership_params_version=1,
        sanity_check_params_version=1,
        consensus_check_params_version=1,
        evidence_decay_params_version=1,
        evidence_fact_multiplier_params_version=1,
        bench_quality_params_version=1,
        confidence_score_params_version=1,
        report_asof=datetime.now(),
    )
    print(reporting.render_report_text(report))

    # Priority 8c: diff against last week's saved snapshot (if any), then save this week's.
    previous_gw = report["headline"]["target_gameweek"] - 1
    previous_snapshot = reporting.load_report_snapshot(TARGET_SEASON, previous_gw, REPORT_HISTORY_DIR)
    diff = reporting.diff_reports(previous_snapshot, report)
    print("\n" + reporting.render_diff_text(diff))
    saved_path = reporting.save_report_snapshot(report, REPORT_HISTORY_DIR)
    print(f"\n[report_history] snapshot saved to {saved_path}")

    # Priority 8b: a small, always-overwritten machine-readable copy of the diff -- this is
    # what scripts/check_deadline_alerts.py (and the scheduled workflow) read to decide
    # whether a newly-doubtful-starter alert is warranted, without needing to re-derive it
    # from the DB or scrape this script's own console output.
    latest_diff_path = REPORT_HISTORY_DIR / "latest_diff.json"
    latest_diff_path.write_text(json.dumps(diff, indent=2))

    print("\n--- section sizes (sanity check) ---")
    print(f"category_breakdown: {len(report['category_breakdown'])} players")
    print(f"risk.analytic: {len(report['risk']['analytic'])} players")
    print(f"risk.empirical: {len(report['risk']['empirical'])} players")
    print(f"evidence_provenance: {len(report['evidence_provenance'])} players, "
          f"{sum(len(v) for v in report['evidence_provenance'].values())} claims considered total")
    print(f"parameter_transparency: {len(report['parameter_transparency'])} param rows")
    print(f"confidence_scores: {len(report['confidence_scores'])} players")
    n_consensus = len(report["consensus_divergence"]) if report["consensus_divergence"] is not None else 0
    print(f"consensus_divergence: {n_consensus} flagged picks")

    con.close()


if __name__ == "__main__":
    main()
