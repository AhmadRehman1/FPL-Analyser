"""End-to-end M0 pipeline: fact_raw -> fact_reconciled -> evidence_claims.

Usage (from repo root):
    .venv/Scripts/python scripts/run_ingestion.py
"""

import json
import math
import sys
import time
from datetime import date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import (  # noqa: E402
    backtest, db, decay, expected_points, ingest_csv, ingest_research_pull, ingest_understat, ingest_workbook,
    minutes_model, monte_carlo, params, reconcile, reporting, squad_optimizer, team_strength, transfer_planner,
    uncertainty,
)

DATA_ROOT = REPO_ROOT / "data" / "external" / "FPL-Core-Insights-main" / "data"
XLSX_PATH = REPO_ROOT / "data" / "external" / "FPL_202627_Master_Evidence_Database.xlsx"
RESEARCH_PULL_XLSX_PATH = REPO_ROOT / "data" / "external" / "FPL_Evidence_Claims_Research_Pull.xlsx"
RECALIBRATION_SEED_DIR = REPO_ROOT / "data" / "recalibration"
# "Now", not a hardcoded date: a model run's data_asof has to be today for evidence ingested
# today to actually be visible (a fixed past date would look-ahead-safely -- but wrongly --
# exclude same-day evidence every time this script re-runs).
CALIBRATION_ASOF_DATE = date.today()
TARGET_SEASON = "2026-2027"
TARGET_GAMEWEEK = 1

SOURCE_TIER_WEIGHTS_V1 = [
    ("official", 1.0),
    ("journalist", 0.8),
    ("specialist", 0.6),
    ("community", 0.4),
]


def main() -> None:
    con = db.connect()
    # Roadmap P1 item (Track B, docs/plans/2026-08_roadmap_plan.md): the single source of truth
    # for every recalibratable parameter's active version, resolved from the git-committed
    # confirmed-seed files (not a hardcoded literal per call site -- see
    # backtest.active_recalibratable_versions()'s own docstring for why a DB table alone
    # wouldn't survive to this run anyway). Every one of this project's real-data scripts reads
    # from this same dict now, closing a real, previously-existing drift where some scripts had
    # been hand-updated to xi_params_version=2/rho_residual_params_version=2 and others hadn't.
    ACTIVE = backtest.active_recalibratable_versions(RECALIBRATION_SEED_DIR)

    t0 = time.time()
    csv_results = ingest_csv.ingest_all(con, DATA_ROOT)
    statuses = {}
    for r in csv_results:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    print(f"[fact_raw] {len(csv_results)} files in {time.time() - t0:.1f}s -> {statuses}")

    for source_type, weight in SOURCE_TIER_WEIGHTS_V1:
        params.write_param(
            con, "source_tier_weights", 1, "2026-08-10", "tier_weight",
            value_numeric=weight, dimensions={"source_type": source_type},
        )
    # M1b: invented v1 default (spec names the mechanism but not a value) -- a modest 20%
    # boost for FACT-tagged claims from official/journalist-tier sources. Flagged for M7
    # recalibration alongside every other invented constant in this project.
    params.write_param(con, "fact_type_multiplier_params", 1, "2026-08-10", "multiplier", value_numeric=1.2)
    # M1: Dixon & Coles (1997) pinned defaults.
    params.write_param(con, "model_decay_params", 1, "2026-08-10", "xi", value_numeric=0.0018)
    params.write_param(con, "model_decay_params", 1, "2026-08-10", "rho", value_numeric=-0.13)
    # M2 v1 defaults, adapted to the real injury-status vocabulary this workbook actually
    # uses (Out/Doubt/Doubt (improving)/Doubt (minutes)), not the spec's illustrative
    # Out/Doubtful/Minor-knock/Fit strings -- see README for the mapping rationale.
    injury_magnitudes = {
        "Out": -4.0, "Doubt": -1.5, "Doubtful": -1.5, "Doubt (improving)": -1.0,
        "Doubt (minutes)": -1.0, "Minor/knock": -0.5, "Fit": 0.0,
    }
    for category, magnitude in injury_magnitudes.items():
        params.write_param(
            con, "minutes_adjustment_params", 1, "2026-08-10", "magnitude", value_numeric=magnitude,
            dimensions={"claim_type": "injury_status", "category": category},
        )
    params.write_param(con, "minutes_adjustment_params", 1, "2026-08-10", "magnitude",
                        value_numeric=0.8, dimensions={"claim_type": "predicted_xi"})
    params.write_param(con, "minutes_adjustment_params", 1, "2026-08-10", "magnitude",
                        value_numeric=1.0, dimensions={"claim_type": "manager_tendency"})
    params.write_param(con, "minutes_adjustment_params", 1, "2026-08-10", "magnitude",
                        value_numeric=-2.0, dimensions={"claim_type": "transfer_likelihood"})
    params.write_param(con, "minutes_adjustment_params", 1, "2026-08-10", "cap",
                        value_numeric=6.0, dimensions={"scope": "global"})
    # M2's own decay table, kept separate from M1's model_decay_params (rotation patterns
    # track manager tenure/squad changes more than pure calendar time). Invented v1 default
    # (a shorter ~200-day half-life than M1's ~385 days), flagged for M7 recalibration.
    params.write_param(con, "minutes_model_decay_params", 1, "2026-08-10", "xi", value_numeric=math.log(2) / 200)
    params.write_param(con, "minutes_model_shrinkage_params", 1, "2026-08-10",
                        "competitive_matches_threshold", value_numeric=10)
    expected_points.seed_v1_params(con)
    uncertainty.seed_v1_params(con)
    squad_optimizer.seed_v1_params(con)
    transfer_planner.seed_v1_params(con)
    reporting.seed_v1_params(con)
    decay.seed_v1_params(con)

    # Re-materialization from a committed seed file, not hardcoded literals (Phase B1
    # hardening). model_decay_params.xi=0.005 (v2) and correlation_params.rho_residual=0.0 (v2)
    # were confirmed via a real M7 76-gameweek walk-forward backtest and activated for real in
    # commit 7bf7604 -- but the recalibration_proposals rows and param_versions rows that
    # justified them only ever existed in the local db/fpl_quant_v2.duckdb this project
    # developed against, since lost (correctly gitignored as a build artifact, never
    # committed). A fresh database (every scheduled CI run; db/*.duckdb doesn't persist between
    # them) has to re-materialize them explicitly rather than resolve a version that was never
    # written here -- previously done as two hardcoded params.write_param() literals; now
    # data/recalibration/seeds_*.json (backtest.write_recalibration_seed_file()'s own committed
    # output format -- these two were hand-recorded into that same shape rather than machine-
    # generated, since the original recalibration_proposals rows no longer exist to regenerate
    # them from) is the durable, git-committed record, and every FUTURE confirmed recalibration
    # (see scripts/review_recalibration.py) is picked up here the same generic way, without
    # needing a new hardcoded literal each time. Only 'confirmed' seeds are ever loaded (see
    # load_confirmed_recalibration_seeds()'s own docstring) -- a 'pending' proposal never
    # silently becomes a live default just by sitting in this file. write_param() is idempotent
    # for an identical value, so this is a safe no-op on every run after the first.
    for seed in backtest.load_confirmed_recalibration_seeds(RECALIBRATION_SEED_DIR):
        params.write_param(
            con, seed["param_family"], seed["new_params_version"], "2026-08-12",
            seed["param_key"], value_numeric=seed["new_value"], dimensions=seed["dimensions"],
        )
    print("[params] source_tier_weights, fact_type_multiplier_params, model_decay_params, "
          "minutes_adjustment_params, minutes_model_decay_params, minutes_model_shrinkage_params, "
          "base_scoring_matrix, bps_formula_params, correlation_params, "
          "cross_player_correlation_params, risk_aversion_params, "
          "squad_optimizer_guardrail_params, planning_horizon_params, transfer_cost_params, "
          "tc_risk_aversion_params, wildcard_gain_threshold_params, ownership_params, "
          "risk_posture_params, field_covariance_params, bench_quality_params, "
          "concentration_risk_params, sanity_check_params, consensus_check_params, "
          "confidence_score_params, claim_type_decay_params v1 seeded")

    t0 = time.time()
    reconcile_results = reconcile.reconcile_all(con, str(XLSX_PATH))
    print(f"[fact_reconciled] {time.time() - t0:.1f}s -> {json.dumps(reconcile_results)}")

    t0 = time.time()
    workbook_results = ingest_workbook.ingest_all(con, str(XLSX_PATH), source_tier_params_version=1)
    print(f"[evidence_claims] {time.time() - t0:.1f}s -> {json.dumps(workbook_results)}")

    if RESEARCH_PULL_XLSX_PATH.exists():
        # A manually-authored / Perplexity-built workbook -- never let one malformed research
        # row take down the whole M0-M6 pipeline (and with it prepare + every downstream
        # workflow). ingest_research_pull already skips bad rows internally; this is the
        # belt-and-braces for a truly corrupt file / a load error.
        t0 = time.time()
        try:
            research_pull_results = ingest_research_pull.ingest_all(
                con, str(RESEARCH_PULL_XLSX_PATH), source_tier_params_version=1
            )
            print(f"[research_pull] {time.time() - t0:.1f}s -> {json.dumps(research_pull_results)}")
        except Exception as exc:  # noqa: BLE001
            print(f"::warning::research-pull ingestion failed, continuing without it: {exc}")

    # Priority 7a: informational only (see ingest_understat.py's own module docstring) --
    # never blocks the real M1-M6 modeling pipeline below on a network hiccup. This
    # environment's own network policy blocks understat.com entirely (confirmed), so this is
    # expected to fail here and succeed wherever it actually has open internet (e.g. a CI
    # runner) -- caught broadly and reported, not silently swallowed.
    t0 = time.time()
    try:
        understat_results = ingest_understat.ingest_league_season(
            con, TARGET_SEASON, int(TARGET_SEASON.split("-")[0]), datetime.now(),
        )
        print(f"[understat] {time.time() - t0:.1f}s -> {json.dumps(understat_results)}")
    except Exception as e:
        print(f"[understat] {time.time() - t0:.1f}s -> SKIPPED (fetch/parse failed: {e})")

    t0 = time.time()
    ts_model_version = team_strength.calibrate(
        con, CALIBRATION_ASOF_DATE,
        xi_params_version=ACTIVE["xi_params_version"], rho_params_version=ACTIVE["rho_params_version"],
    )
    n_teams = con.execute(
        "SELECT count(*) FROM team_strength_snapshots WHERE model_version = ?", [ts_model_version]
    ).fetchone()[0]
    print(f"[team_strength] {time.time() - t0:.1f}s -> model_version={ts_model_version}, {n_teams} teams")

    t0 = time.time()
    n_preseason_claims = minutes_model.log_preseason_involvement_claims(con, TARGET_SEASON)
    # ACTIVE['shrinkage_params_version']/ACTIVE['fact_multiplier_params_version'] resolve to v1
    # (the honest, freshly-reproducible baseline), NOT the real v10/v8 winners commit 1f0dc7b
    # activated. Unlike xi/rho_residual above, those two versions' actual winning numeric values
    # were never recorded anywhere durable (README documents xi/rho_residual's before->after
    # numbers explicitly, but not these two) -- they only existed as intermediate
    # param_versions rows inside refit_minutes_and_evidence_params()'s multi-round block
    # coordinate descent, written directly to the same since-lost local db/fpl_quant_v2.duckdb.
    # Re-establishing the real v8/v10 values needs an actual re-run of that search (a real,
    # flagged follow-up -- see README's Design notes), not a guess at plausible-looking numbers --
    # ACTIVE will correctly pick them up automatically once a real confirmed seed exists.
    # decay_params_version stays a hardcoded literal: minutes_model_decay_params isn't one of
    # recalibrate()'s MINUTES_PARAM_GRIDS families (see run_backtest.py), so it's out of Track B's
    # scope -- there's nothing for ACTIVE to ever resolve differently here.
    mm_model_version = minutes_model.run(
        con, CALIBRATION_ASOF_DATE, TARGET_SEASON,
        decay_params_version=1, adjustment_params_version=ACTIVE["adjustment_params_version"],
        shrinkage_params_version=ACTIVE["shrinkage_params_version"],
        fact_multiplier_params_version=ACTIVE["fact_multiplier_params_version"],
    )
    n_players = con.execute(
        "SELECT count(*) FROM minutes_model_outputs WHERE model_version = ?", [mm_model_version]
    ).fetchone()[0]
    print(
        f"[minutes_model] {time.time() - t0:.1f}s -> {n_preseason_claims} preseason_involvement "
        f"claims logged, model_version={mm_model_version}, {n_players} players"
    )

    t0 = time.time()
    ep_model_version = expected_points.run(
        con, CALIBRATION_ASOF_DATE, TARGET_SEASON, TARGET_GAMEWEEK,
        ts_model_version=ts_model_version, mm_model_version=mm_model_version,
        scoring_params_version=1, bps_params_version=1, tau_params_version=1,
    )
    n_ep_rows = con.execute(
        "SELECT count(*) FROM ep_outputs WHERE model_version = ?", [ep_model_version]
    ).fetchone()[0]
    print(f"[expected_points] {time.time() - t0:.1f}s -> model_version={ep_model_version}, {n_ep_rows} player-fixture rows")

    t0 = time.time()
    un_model_version = uncertainty.run(
        con, CALIBRATION_ASOF_DATE, ep_model_version=ep_model_version, mm_model_version=mm_model_version,
        ts_model_version=ts_model_version, scoring_params_version=1, bps_params_version=1,
        tau_params_version=1, rho_residual_params_version=ACTIVE["rho_residual_params_version"], corr_params_version=1,
    )
    n_un_rows = con.execute(
        "SELECT count(*) FROM uncertainty_outputs WHERE model_version = ?", [un_model_version]
    ).fetchone()[0]
    n_cov_pairs = con.execute(
        "SELECT count(*) FROM cross_player_covariance WHERE model_version = ?", [un_model_version]
    ).fetchone()[0]
    print(
        f"[uncertainty] {time.time() - t0:.1f}s -> model_version={un_model_version}, "
        f"{n_un_rows} player-fixture rows, {n_cov_pairs} nonzero covariance pairs"
    )

    t0 = time.time()
    try:
        so_run_id = squad_optimizer.run(
            con, CALIBRATION_ASOF_DATE, TARGET_SEASON, TARGET_GAMEWEEK,
            ep_model_version=ep_model_version, uncertainty_model_version=un_model_version,
            lambda_params_version=ACTIVE["lambda_params_version"], guardrail_params_version=1,
        )
        n_squad = con.execute(
            "SELECT count(*) FROM squad_optimizer_selections WHERE run_id = ? AND in_squad", [so_run_id]
        ).fetchone()[0]
        print(
            f"[squad_optimizer] {time.time() - t0:.1f}s -> run_id={so_run_id}, "
            f"divergence check passed, {n_squad} players selected"
        )
    except squad_optimizer.DivergenceCheckFailedError as e:
        print(f"[squad_optimizer] {time.time() - t0:.1f}s -> DIVERGENCE CHECK FAILED, no squad stored: {e}")
        raise

    t0 = time.time()
    mc_model_version = monte_carlo.run(
        con, CALIBRATION_ASOF_DATE, squad_optimizer_run_id=so_run_id,
        ep_model_version=ep_model_version, mm_model_version=mm_model_version,
        ts_model_version=ts_model_version, uncertainty_model_version=un_model_version,
        scoring_params_version=1, tau_params_version=1,
        rho_residual_params_version=ACTIVE["rho_residual_params_version"],
    )
    n_mc_players = con.execute(
        "SELECT count(DISTINCT player_uid) FROM monte_carlo_player_summary WHERE model_version = ?", [mc_model_version]
    ).fetchone()[0]
    n_mc_rows = con.execute(
        "SELECT count(*) FROM monte_carlo_player_totals WHERE model_version = ?", [mc_model_version]
    ).fetchone()[0]
    print(
        f"[monte_carlo] {time.time() - t0:.1f}s -> model_version={mc_model_version}, "
        f"{n_mc_players} squad players simulated, {n_mc_rows} realization rows"
    )

    con.close()


if __name__ == "__main__":
    main()
