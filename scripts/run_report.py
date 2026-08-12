"""M9: Reporting / Explainability Layer -- builds and prints a real squad report from the
actual project database.

Usage (from repo root):
    .venv/Scripts/python scripts/run_report.py

Uses the real GW1 2026-27 squad_optimizer_runs row, plus the real M7 backtest_run_id and M8
transfer_plan_run_id already sitting in the database from earlier milestones' real runs.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import db, reporting  # noqa: E402

TARGET_SEASON = "2026-2027"

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

    real_run_id = con.execute(
        "SELECT run_id FROM squad_optimizer_runs WHERE target_season = ? AND target_gameweek = 1 "
        "AND is_manager_snapshot = FALSE",
        [TARGET_SEASON],
    ).fetchone()
    if not real_run_id:
        raise SystemExit(f"no real squad_optimizer_runs row for {TARGET_SEASON} GW1 -- run scripts/run_ingestion.py first")
    real_run_id = real_run_id[0]

    backtest_run_id = con.execute("SELECT max(backtest_run_id) FROM backtest_runs").fetchone()[0]
    transfer_plan_run_id = con.execute("SELECT max(run_id) FROM transfer_plan_runs").fetchone()[0]

    report = reporting.build_report(
        con, real_run_id,
        transfer_plan_run_id=transfer_plan_run_id,
        backtest_run_id=backtest_run_id,
        active_param_versions=ACTIVE_PARAM_VERSIONS,
    )
    print(reporting.render_report_text(report))

    print("\n--- section sizes (sanity check) ---")
    print(f"category_breakdown: {len(report['category_breakdown'])} players")
    print(f"risk.analytic: {len(report['risk']['analytic'])} players")
    print(f"risk.empirical: {len(report['risk']['empirical'])} players")
    print(f"evidence_provenance: {len(report['evidence_provenance'])} players, "
          f"{sum(len(v) for v in report['evidence_provenance'].values())} claims considered total")
    print(f"parameter_transparency: {len(report['parameter_transparency'])} param rows")

    con.close()


if __name__ == "__main__":
    main()
