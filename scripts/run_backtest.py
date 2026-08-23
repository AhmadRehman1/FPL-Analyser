"""M7 end-to-end walk-forward backtest: full M1->M6 pipeline against every historical
2024-25/2025-26 gameweek, tiered cold/warm/mature, scored against realized outcomes, then
recalibration proposals for every invented-not-derived parameter across M1-M6 (except
xi_club_concentration_cap, deliberately excluded -- see backtest.report_concentration_sensitivity
and the README's Design notes).

Usage (from repo root):
    .venv/Scripts/python scripts/run_backtest.py

This re-runs the full pipeline ~76 times (roughly 1-2 hours) plus recalibration (the lambda
grid search alone re-solves the SCIP MIQP per candidate per gameweek, ~19s each per README's
own live-run numbers) -- expect a long run. Every backtest_run_id/proposal is fully auditable
afterward via backtest_runs/backtest_gameweek_steps/backtest_metrics/recalibration_proposals;
nothing here activates a new parameter version -- see scripts/review_recalibration.py.
"""

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import backtest, db  # noqa: E402

# Every param family below is currently seeded at version=1 by scripts/run_ingestion.py.
PARAM_VERSIONS = dict(
    xi_params_version=1, rho_params_version=1,
    decay_params_version=1, adjustment_params_version=1, shrinkage_params_version=1, fact_multiplier_params_version=1,
    scoring_params_version=1, bps_params_version=1, tau_params_version=1,
    rho_residual_params_version=1, corr_params_version=1,
    lambda_params_version=1, guardrail_params_version=1,
)

# Modest, explicit blocks for the M2 coordinate descent -- kept small deliberately (this runs
# len(candidates) x len(warm+mature eval steps) minutes_model.run() calls per block per round,
# not free even though each individual call is cheap).
#
# source_tier_weights is deliberately NOT included here, correcting an assumption in this
# module's original design: tier weights are not resolved live per model-run at all --
# ingest_workbook.build_sources() bakes tier_weight * log-scaled(citation_count) into
# sources.base_reliability_score -> evidence_claims.source_reliability_score once, at
# ingestion time, and evidence_blend.effective_weight() never re-joins it ("snapshotted at
# ingestion -- never live-joined, so later re-scoring of a source never silently reweights old
# claims" -- the README's own stated design). Re-testing a candidate tier_weight value would
# mean re-running ingest_workbook.ingest_all() per candidate (a much more expensive,
# architecturally different operation than re-running minutes_model.run()), not a coordinate-
# descent block here. fact_type_multiplier_params IS live-resolved every minutes_model.run()
# call (via fact_multiplier_params_version -> effective_weight()) and stays in scope below.
MINUTES_PARAM_GRIDS = [
    {"param_family": "fact_type_multiplier_params", "param_key": "multiplier", "dimensions": None,
     "candidates": [1.0, 1.2, 1.5], "version_field": "fact_multiplier_params_version"},
    {"param_family": "minutes_model_shrinkage_params", "param_key": "competitive_matches_threshold",
     "dimensions": None, "candidates": [5, 10, 15, 20], "version_field": "shrinkage_params_version"},
    {"param_family": "minutes_adjustment_params", "param_key": "magnitude",
     "dimensions": {"claim_type": "injury_status", "category": "Out"},
     "candidates": [-3.0, -4.0, -5.0], "version_field": "adjustment_params_version"},
    {"param_family": "minutes_adjustment_params", "param_key": "cap", "dimensions": {"scope": "global"},
     "candidates": [4.0, 6.0, 8.0], "version_field": "adjustment_params_version"},
]


def main() -> None:
    con = db.connect()

    t0 = time.time()
    backtest_run_id = backtest.run(
        con, **PARAM_VERSIONS, n_antithetic_pairs=5000, run_monte_carlo=True,
        notes="M7 full walk-forward backtest",
        # Priority 9c opt-in: without this, score_gameweek() never records
        # model_squad_realized_points/avg_manager_benchmark_points, and beats_baseline()
        # (review B2/roadmap Feature 7) has nothing to read for this run_id.
        ownership_params_version=1,
    )
    print(f"[backtest.run] {time.time() - t0:.1f}s -> backtest_run_id={backtest_run_id}")

    steps = con.execute(
        "SELECT tier, count(*), sum(CASE WHEN divergence_check_passed THEN 1 ELSE 0 END) "
        "FROM backtest_gameweek_steps WHERE backtest_run_id = ? GROUP BY tier ORDER BY tier",
        [backtest_run_id],
    ).fetchall()
    for tier, n, n_passed in steps:
        print(f"  tier={tier}: {n} steps, {n_passed} divergence-check passes")

    metrics = con.execute(
        "SELECT tier, metric_name, count(*), avg(metric_value) FROM backtest_metrics "
        "WHERE backtest_run_id = ? AND metric_name NOT LIKE 'realized%' GROUP BY tier, metric_name ORDER BY metric_name, tier",
        [backtest_run_id],
    ).fetchall()
    for tier, name, n, avg in metrics:
        print(f"  [{tier}] {name}: n={n} mean={avg:.4f}")

    t0 = time.time()
    proposal_ids = backtest.recalibrate(
        con, backtest_run_id,
        current_xi_version=1, current_rho_version=1,
        current_rho_residual_version=1,
        current_minutes_versions={
            "decay_params_version": 1, "adjustment_params_version": 1,
            "shrinkage_params_version": 1, "fact_multiplier_params_version": 1,
        },
        current_lambda_version=1,
        guardrail_cap=3,
        minutes_param_grids=MINUTES_PARAM_GRIDS,
        current_kappa_tc_version=1,
        refit_kappa_tc_flag=True,
    )
    print(f"[recalibrate] {time.time() - t0:.1f}s -> {len(proposal_ids)} pending proposals: {proposal_ids}")

    con.close()


if __name__ == "__main__":
    main()
