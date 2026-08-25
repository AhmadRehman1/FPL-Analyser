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
DASHBOARD_DIR = REPO_ROOT / "data" / "dashboard"

# Every param family currently seeded at v1 across M1-M8 (run_ingestion.py's own list).
ACTIVE_PARAM_VERSIONS = {
    "source_tier_weights": 1, "fact_type_multiplier_params": 1, "model_decay_params": 1,
    "minutes_adjustment_params": 1, "minutes_model_decay_params": 1, "minutes_model_shrinkage_params": 1,
    "base_scoring_matrix": 1, "bps_formula_params": 1, "bps_dispersion_params": 1,
    "correlation_params": 1, "cross_player_correlation_params": 1, "risk_aversion_params": 1,
    "squad_optimizer_guardrail_params": 1, "planning_horizon_params": 1, "transfer_cost_params": 1,
    "tc_risk_aversion_params": 1, "wildcard_gain_threshold_params": 1,
}


def _would_regress_track_record(new_track_record: dict, existing_track_record: dict | None) -> bool:
    """True when writing new_track_record would discard real backtest coverage a previous run
    already captured and committed -- see main()'s own comment on why this can happen (this
    script runs twice daily against an ephemeral, backtest-less DB; scripts/run_backtest.py's
    real ~1-2 hour job runs on its own separate, less frequent schedule). Pure and DB-free so
    this specific regression is unit-testable without a live database."""
    return (
        new_track_record.get("backtest_run_id") is None
        and existing_track_record is not None
        and existing_track_record.get("backtest_run_id") is not None
    )


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

    # App conversion, build-up phase: the model's own real backtested track record, for the
    # app's Track Record screen -- see reporting.build_track_record_summary()'s own docstring
    # for why this is two honest numbers (real backtest coverage + real parameter-transparency
    # count) rather than one invented "accuracy %".
    #
    # backtest_run_id above comes from THIS run's own DuckDB file, which -- unlike the committed
    # data/dashboard/ JSON this step writes to -- never persists across scheduled runs
    # (db/fpl_quant_v2.duckdb is gitignored and rebuilt from scratch by run_ingestion.py every
    # time; see README's own "Layout" note). scripts/run_backtest.py is a real ~1-2 hour job
    # deliberately NOT run on this script's own twice-daily schedule (see its own module
    # docstring) -- it runs on its own, less frequent schedule and commits its own fresh
    # app_track_record.json after a real backtest_runs row exists. Without the guard below,
    # THIS script's own very next twice-daily run -- starting from a fresh, backtest-less DB --
    # would silently overwrite that real track record right back to "no backtest yet", discarding
    # a ~1-2 hour job's actual output within ~12 hours. Keep the existing committed file
    # (real backtest data already captured) rather than regress it; still write fresh when this
    # run has its own real backtest_run_id, or when nothing's been committed yet.
    track_record = reporting.build_track_record_summary(con, report, backtest_run_id)
    track_record["generated_at"] = datetime.now().isoformat()
    # Priority 8d: the full public Track Record page payload -- backtest status + the dated
    # snapshot timeline + the latest week-over-week diff + data provenance, all assembled from
    # artifacts this same run already produced. The standalone track-record.html PWA reads this
    # single file (same raw.githubusercontent.com data path as every other dashboard JSON), so
    # no new hosting or endpoint is needed.
    track_record["transparency_log"] = reporting.build_transparency_log(
        track_record, REPORT_HISTORY_DIR, diff,
    )
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    track_record_path = DASHBOARD_DIR / "app_track_record.json"
    existing_track_record = None
    if track_record_path.exists():
        try:
            existing_track_record = json.loads(track_record_path.read_text())
        except (json.JSONDecodeError, OSError):
            existing_track_record = None
    if _would_regress_track_record(track_record, existing_track_record):
        print(
            f"\n[dashboard] app_track_record.json NOT overwritten -- this run has no real "
            f"backtest_run_id, but the committed file already has one "
            f"({existing_track_record['backtest_run_id']}) from a real scripts/run_backtest.py "
            f"run. Keeping it."
        )
    else:
        track_record_path.write_text(json.dumps(track_record, indent=2))
        print(f"\n[dashboard] track record written to {track_record_path}")

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
