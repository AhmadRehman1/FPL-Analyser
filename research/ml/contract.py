"""Shared constants and the data contract for the Phase-0 ML research layer.

Single source of truth for column names, season ordering, feature lists, and on-disk
result paths. Centralising these here means the dataset builder, feature engineer, leakage
checker, and evaluator all agree on what each column means and where results land -- so a
leakage check or a metric can be written against a name, not a convention.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

# ============================================================
# Paths
# ============================================================

ML_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ML_ROOT.parents[1]
DATA_DIR = ML_ROOT / "data"
RESULTS_DIR = ML_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
# Deliberately NOT under results/ and NOT gitignored (see .gitignore's own "ML research engine
# generated artifacts" section, which stops at results/ and data/): a single real run's numbers
# are exactly the kind of "results/ is gitignored on purpose" artifact ml_experiment.yml's own
# header describes, but a *persisted, week-over-week trend* of those numbers is the only way to
# actually see whether the model is improving over time as more real gameweeks accrue -- a
# 90-day build artifact per isolated Sunday run doesn't give you that at a glance. This is a
# narrow, deliberate exception: scripts/append_ml_run_to_history.py appends only real numbers
# here (bootstrap CI point estimates/bounds/credible flags, season points) -- never REPORT.md's
# prose or its §9 decision, which stays exactly as much a human judgement call as before.
RESULTS_HISTORY_DIR = ML_ROOT / "results_history"
RESULTS_HISTORY_CSV = RESULTS_HISTORY_DIR / "weekly_quality_history.csv"

DATASET_PARQUET = DATA_DIR / "player_gw_dataset.parquet"
DATASET_CSV = DATA_DIR / "player_gw_dataset.csv"
BASELINE_METRICS_JSON = RESULTS_DIR / "baseline_metrics.json"
BASELINE_PREDICTIONS_PARQUET = RESULTS_DIR / "baseline_predictions.parquet"
MODEL_COMPARISON_CSV = RESULTS_DIR / "model_comparison.csv"
RESIDUAL_ANALYSIS_CSV = RESULTS_DIR / "residual_analysis.csv"
HIGH_DISAGREEMENT_CSV = RESULTS_DIR / "high_disagreement_cases.csv"
CALIBRATION_CSV = RESULTS_DIR / "calibration.csv"
FEATURE_IMPORTANCE_CSV = RESULTS_DIR / "feature_importance.csv"
STABILITY_CSV = RESULTS_DIR / "feature_stability.csv"
FEATURE_IMPORTANCE_LIGHTGBM_CSV = RESULTS_DIR / "feature_importance_lightgbm.csv"
STABILITY_LIGHTGBM_CSV = RESULTS_DIR / "feature_stability_lightgbm.csv"
# R13: XGBoost, the independent-implementation confirmation arm for quant_lightgbm's result --
# same per-fold importance/stability treatment, kept separate so its numbers are inspectable on
# their own footing, not merged into the LightGBM files it exists to cross-check.
FEATURE_IMPORTANCE_XGBOOST_CSV = RESULTS_DIR / "feature_importance_xgboost.csv"
STABILITY_XGBOOST_CSV = RESULTS_DIR / "feature_stability_xgboost.csv"
IMPROVEMENT_CSV = RESULTS_DIR / "improvement.csv"
ENSEMBLE_CSV = RESULTS_DIR / "ensemble.csv"
EXPERIMENT_MANIFEST_JSON = RESULTS_DIR / "experiment_manifest.json"
SEASON_POINTS_CSV = RESULTS_DIR / "season_points.csv"
# Track F (docs/plans/2026-08_retrospective_validation_and_ml_decision_plan.md), R10: per-model
# compute/runtime and bootstrap confidence intervals, not just point-estimate metrics.
COMPUTE_RUNTIME_CSV = RESULTS_DIR / "compute_runtime.csv"
BOOTSTRAP_CI_JSON = RESULTS_DIR / "bootstrap_ci.json"
# R11: per-model, per-slice metrics -- the existing sliced_metrics() machinery only ever
# persisted the Quant baseline's own slicing (baseline_metrics.json); R11's ship/no-ship
# decision needs quant_lightgbm's slicing too, across the whole walk-forward run.
SLICED_MODEL_COMPARISON_CSV = RESULTS_DIR / "sliced_model_comparison.csv"
RUN_LOG_CSV = RESULTS_DIR / "experiment_runs.csv"  # rolling log of every run (loop mode)
RUNS_DIR = RESULTS_DIR / "runs"                      # one timestamped subdir per loop run
REPORT_MD = ML_ROOT / "REPORT.md"

# Every result-artifact attribute a loop/24-7 caller must redirect into a per-run subdirectory
# before calling run_experiment(), or successive iterations silently overwrite each other's
# detailed output in place -- single source of truth so experiment.run_loop() and
# run_continuous.run_forever() can't drift out of sync with each other (or with a newly added
# result artifact) the way they previously did: run_loop() was missing BASELINE_METRICS_JSON/
# BASELINE_PREDICTIONS_PARQUET (silently overwritten every `--runs N` iteration), and
# run_continuous.py's own "one timestamped results subdir per run" docstring claim was not
# actually implemented at all (its loop never redirected anything). DATASET_PARQUET/DATASET_CSV
# and RUN_LOG_CSV/RUNS_DIR are deliberately excluded: the first two are the shared input dataset
# cache, not a per-run result, and the second two are the rolling log/container themselves.
PER_RUN_RESULT_ATTRS: tuple[str, ...] = (
    "BASELINE_METRICS_JSON", "BASELINE_PREDICTIONS_PARQUET",
    "MODEL_COMPARISON_CSV", "IMPROVEMENT_CSV", "RESIDUAL_ANALYSIS_CSV",
    "HIGH_DISAGREEMENT_CSV", "CALIBRATION_CSV", "FEATURE_IMPORTANCE_CSV",
    "STABILITY_CSV", "FEATURE_IMPORTANCE_LIGHTGBM_CSV", "STABILITY_LIGHTGBM_CSV",
    "FEATURE_IMPORTANCE_XGBOOST_CSV", "STABILITY_XGBOOST_CSV",
    "ENSEMBLE_CSV", "EXPERIMENT_MANIFEST_JSON", "SEASON_POINTS_CSV",
    "COMPUTE_RUNTIME_CSV", "BOOTSTRAP_CI_JSON", "SLICED_MODEL_COMPARISON_CSV",
)

for _d in (DATA_DIR, RESULTS_DIR, PLOTS_DIR, RUNS_DIR, RESULTS_HISTORY_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ============================================================
# Column contract
# ============================================================

# Identification
COL_PLAYER_UID = "player_uid"
COL_SEASON = "season"
COL_GAMEWEEK = "gameweek"
COL_TEAM_UID = "team_uid"
COL_POSITION = "position"
COL_OPPONENT_UID = "opponent_uid"
COL_HOME_AWAY = "home_away"

# Prediction / actual / residual
COL_QUANT_PRED = "quant_prediction"           # Q(x): ep_total aggregated to player x gw
COL_PRED_TIMESTAMP = "prediction_timestamp"   # the step's data_asof (gameweek deadline)
COL_EP_MODEL_VERSION = "ep_model_version"      # provenance: which Quant run produced Q(x)
COL_ACTUAL = "actual_points"                  # y: fact_player_season_stats.event_points
COL_RESIDUAL = "residual"                     # y - Q(x)
COL_PREDICTED_RESIDUAL = "predicted_residual"  # ML(x)
COL_ML_PRED = "ml_prediction"                  # Q(x) + ML(x)  -- L1 (median) point estimate
COL_ML_CEILING = "ml_ceiling"                   # Q(x) + q90 residual -- an upper-tail "haul" estimate, used ONLY for the season-sim captain pick

# The label. Used exactly once, as the target, never as a feature.
LABEL_COL = COL_ACTUAL

IDENTIFIER_COLS = [
    COL_PLAYER_UID, COL_SEASON, COL_GAMEWEEK, COL_TEAM_UID, COL_POSITION,
    COL_OPPONENT_UID, COL_HOME_AWAY, COL_QUANT_PRED, COL_PRED_TIMESTAMP, COL_EP_MODEL_VERSION,
]


# ============================================================
# Seasons and ordering
# ============================================================

# Canonical chronological order. `season_sort_key` converts "2024-2025" -> 2024 for comparison.
SEASON_ORDER: tuple[str, ...] = (
    "2024-2025", "2025-2026", "2026-2027",
)


def season_sort_key(season: str) -> int:
    """'2024-2025' -> 2024. Used for chronological walk-forward split ordering."""
    try:
        return int(season.split("-")[0])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"unrecognised season format {season!r}") from exc


# Premier League competition tag used throughout fpl_quant.
PL = "Premier League"

POSITIONS = ("Goalkeeper", "Defender", "Midfielder", "Forward")

# Default walk-forward experiment plan: train on everything before each test season.
DEFAULT_WALK_FORWARD_PLAN: tuple[tuple[tuple[str, ...], str], ...] = (
    (("2024-2025",), "2025-2026"),
    (("2024-2025", "2025-2026"), "2026-2027"),
)


# ============================================================
# Feature groups (names documented in LEAKAGE_PROTOCOL.md §4)
# ============================================================

# Rolling windows in matches/gameweeks. 3/5/10 mirrors the spec's "rolling X in last 3/5"
# convention and the repo's own minutes_model lookback of "competitive_matches_last_2_seasons".
ROLLING_WINDOWS = (3, 5, 10)

PLAYER_ROLLING_FEATURES: tuple[str, ...] = (
    "rolling_points", "rolling_minutes", "rolling_starts", "rolling_goals",
    "rolling_assists", "rolling_bps", "rolling_xg_per90", "rolling_xa_per90",
    "rolling_defcon_per90", "rolling_saves_per90",
)

MINUTES_FEATURES: tuple[str, ...] = (
    "p_start_final", "p_60plus_min", "starts_last_3", "starts_last_5",
    "minutes_last_5",
)

# Team strength is derived from realised goals in prior finished matches (fact_match scores).
# xG is not stored per-match in fact_player_match_stats, so team xG is NOT computed -- a
# goals-based proxy would be mislabelled, so the feature is omitted rather than faked.
TEAM_FEATURES: tuple[str, ...] = (
    "team_goals_for_last_10", "team_goals_against_last_10",
)

# fixture_difficulty is derived from opponent defensive strength (opponent goals conceded
# in prior matches), NOT from the M1 Dixon-Coles lambdas -- using M1 lambdas would make the
# residual model a trivial identity of the Quant model (LEAKAGE_PROTOCOL.md §5).
FIXTURE_FEATURES: tuple[str, ...] = (
    "is_home", "fixture_difficulty", "opponent_goals_against_last_10",
    "opponent_goals_for_last_10", "matches_last_7_days", "matches_last_14_days",
)

CONTEXT_FEATURES: tuple[str, ...] = (
    "now_cost", "selected_by_percent", "chance_of_playing_next_round", "status",
)


def connect(db_path: Path | str | None = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Thin wrapper over fpl_quant.db.connect so this layer has one import surface and never
    opens a connection that skips schema application. read_only=True is used by the dataset
    builder: the ML layer never writes to the production schema, only to its own results/."""
    from fpl_quant import db  # late import: path bootstrap in __init__ makes this resolve

    if db_path is None:
        return db.connect(read_only=read_only)
    return db.connect(Path(db_path), read_only=read_only)
