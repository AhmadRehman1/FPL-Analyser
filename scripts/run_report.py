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

from fpl_quant import backtest, db, reporting  # noqa: E402

TARGET_SEASON = "2026-2027"
# Priority 8c: small, committed per-gameweek report snapshots -- see reporting.py's own
# module docstring for why this lives as plain JSON in the repo rather than in the (gitignored,
# never-persisted-across-runs) DuckDB file itself.
REPORT_HISTORY_DIR = REPO_ROOT / "data" / "report_history"
DASHBOARD_DIR = REPO_ROOT / "data" / "dashboard"
RECALIBRATION_SEED_DIR = REPO_ROOT / "data" / "recalibration"
DECISION_LOG_DIR = REPO_ROOT / "data" / "decision_log"
TRACKED_ENTRY_IDS = [7139944, 1305242]


def _resolve_report_run_id(con, current_event: int | None) -> int | None:
    """The from-scratch M5 solve this report should describe -- the model's optimal squad for
    the CURRENT gameweek. Not `max(target_gameweek)`: print_chip_timing_roadmap.py runs an
    evaluate_wildcard() (-> a real is_manager_snapshot=FALSE squad_optimizer_runs row) at
    fixture-swing gameweeks 10-19 BEFORE this script runs, so ordering by target_gameweek DESC
    snapshotted GW12/GW14 "reports" for a squad the model never recommended for now -- junk in
    report_history/ that then polluted the public Track Record timeline. Constrain to the real
    current gameweek instead."""
    if current_event is not None:
        row = con.execute(
            "SELECT run_id FROM squad_optimizer_runs WHERE target_season = ? AND is_manager_snapshot = FALSE "
            "AND target_gameweek <= ? ORDER BY target_gameweek DESC, run_id DESC LIMIT 1",
            [TARGET_SEASON, current_event],
        ).fetchone()
        if row:
            return row[0]
    # No event passed (a bare local run): the real-squad planner always solves at the true
    # current gameweek and its manager-snapshot runs are never future-dated by a roadmap
    # horizon solve -- use the newest one's gameweek as the ceiling.
    ceiling = con.execute(
        "SELECT max(target_gameweek) FROM squad_optimizer_runs WHERE target_season = ? AND is_manager_snapshot = TRUE",
        [TARGET_SEASON],
    ).fetchone()[0]
    q = ("SELECT run_id FROM squad_optimizer_runs WHERE target_season = ? AND is_manager_snapshot = FALSE "
         + ("AND target_gameweek <= ? " if ceiling is not None else "")
         + "ORDER BY target_gameweek DESC, run_id DESC LIMIT 1")
    row = con.execute(q, [TARGET_SEASON, ceiling] if ceiling is not None else [TARGET_SEASON]).fetchone()
    return row[0] if row else None

# Roadmap P1 item (Track B, docs/plans/2026-08_roadmap_plan.md): this feeds
# reporting.transparency_panel() (via build_report()'s active_param_versions=), whose whole
# point is disclosing what version is ACTUALLY active per family -- so these must resolve from
# the git-committed confirmed-seed files too, not stay hardcoded at 1 once a family has really
# been promoted (that would make the transparency panel itself misleading).
# Every param family currently seeded across M1-M8 (run_ingestion.py's own list). Recalibratable
# families below read from active; every other family isn't one recalibrate() can produce a new
# version for, so it stays a hardcoded 1.
# model_decay_params has two independently-recalibratable keys (xi, rho) but this transparency
# panel is family-keyed (transparency_panel() does an exact (family, version) row lookup, one
# number per family) -- with xi/rho genuinely at different real versions today (2 and 1), any
# single number here would either show a stale xi or SILENTLY DROP rho's row entirely (whichever
# key wasn't written at that version has no row to find -- found in review before this shipped).
# Left hardcoded at 1, matching this dict's pre-Track-B behavior exactly, until
# transparency_panel() itself supports per-key resolution within one family.
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
    current_event = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else None
    con = db.connect()
    _ACTIVE = backtest.active_recalibratable_versions(RECALIBRATION_SEED_DIR)
    ACTIVE_PARAM_VERSIONS = _active_param_versions(_ACTIVE)

    real_run_id = _resolve_report_run_id(con, current_event)
    if not real_run_id:
        raise SystemExit(f"no real squad_optimizer_runs row for {TARGET_SEASON} -- run scripts/run_ingestion.py first")

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
        evidence_fact_multiplier_params_version=_ACTIVE["fact_multiplier_params_version"],
        bench_quality_params_version=1,
        confidence_score_params_version=1,
        role_change_flag_params_version=1,
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
    # Phase C-3, moved onto the DAILY path (was export_track_record.py in weekly_backtest.yml,
    # which never lands -- its full run_backtest.py exceeds the 6h Actions cap). This half of the
    # track record needs no backtest: it reads the committed data/decision_log/ entries
    # realize_decision_log_outcomes.py fills in each gameweek, so it refreshes every day and
    # accumulates in the open from the first realized gameweek.
    track_record["planner_decision_accuracy"] = reporting.build_planner_decision_summary(
        DECISION_LOG_DIR, TRACKED_ENTRY_IDS, TARGET_SEASON,
    )
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
        # Keep the committed file's real backtest data -- but still refresh the daily-cadence
        # halves (planner_decision_accuracy from the decision logs, the snapshot timeline, the
        # diff, provenance) onto it, so a real backtest landing once doesn't freeze the forward
        # track record until the next one does.
        merged = dict(existing_track_record)
        merged["planner_decision_accuracy"] = track_record["planner_decision_accuracy"]
        # Refresh the daily-cadence halves of the transparency log (snapshot timeline, diff,
        # provenance) but KEEP the committed backtest sub-object -- metrics / headline / step
        # count are written only by the walk-forward job, and this daily run's DB has no
        # backtest, so transparency_log.backtest would otherwise be blanked back to []/None.
        refreshed_tlog = dict(track_record["transparency_log"])
        committed_bt = (existing_track_record.get("transparency_log") or {}).get("backtest")
        if committed_bt:
            refreshed_tlog["backtest"] = committed_bt
        merged["transparency_log"] = refreshed_tlog
        merged["generated_at"] = track_record["generated_at"]
        track_record_path.write_text(json.dumps(merged, indent=2))
        print(
            f"\n[dashboard] app_track_record.json: kept committed backtest_run_id "
            f"{existing_track_record['backtest_run_id']}, refreshed planner_decision_accuracy + "
            f"timeline + provenance."
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
