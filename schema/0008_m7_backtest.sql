-- M7 -- Walk-Forward Backtesting Framework.
-- A backtest step reuses every existing versioned-model table as-is: it just produces normal
-- team_strength_model_versions/minutes_model_versions/ep_model_versions/uncertainty_model_versions/
-- squad_optimizer_runs/monte_carlo_run_versions rows tagged with a past calibration_asof_date --
-- no schema change needed there. These three tables are the only new state M7 introduces: which
-- step produced which version rows (and its tier), the metrics scored against realized outcomes,
-- and a human-reviewed audit trail for recalibrated parameters.

CREATE SEQUENCE IF NOT EXISTS seq_backtest_run START 1;

CREATE TABLE IF NOT EXISTS backtest_runs (
    backtest_run_id     INTEGER PRIMARY KEY DEFAULT nextval('seq_backtest_run'),
    started_at          TIMESTAMP NOT NULL DEFAULT current_timestamp,
    warm_up_gameweeks   INTEGER NOT NULL,
    notes                VARCHAR
);

-- Links one walk-forward step to the model-version rows it produced across M1-M6, plus its tier.
-- Nullable version columns: a step can legitimately produce no monte_carlo run (blank fixture) or
-- fail the divergence check without a squad_optimizer_runs row existing at all (M5's own
-- DivergenceCheckFailedError refuses to store a selection on failure, per schema/0006's comment) --
-- this table still records the step happened and what it found, not just successful steps.
CREATE TABLE IF NOT EXISTS backtest_gameweek_steps (
    backtest_run_id            INTEGER NOT NULL REFERENCES backtest_runs (backtest_run_id),
    season                       VARCHAR NOT NULL,
    gameweek                      INTEGER NOT NULL,
    tier                            VARCHAR NOT NULL CHECK (tier IN ('cold', 'warm', 'mature')),
    data_asof                        TIMESTAMP NOT NULL,
    ts_model_version                   INTEGER REFERENCES team_strength_model_versions (model_version),
    mm_model_version                     INTEGER REFERENCES minutes_model_versions (model_version),
    ep_model_version                       INTEGER REFERENCES ep_model_versions (model_version),
    un_model_version                         INTEGER REFERENCES uncertainty_model_versions (model_version),
    so_run_id                                  INTEGER REFERENCES squad_optimizer_runs (run_id),
    mc_model_version                             INTEGER REFERENCES monte_carlo_run_versions (model_version),
    divergence_check_passed                        BOOLEAN,
    PRIMARY KEY (backtest_run_id, season, gameweek)
);

-- One row per (step, metric): log score / Brier per category, Poisson calibration residual,
-- realized-vs-M4-vs-M6 covariance comparison, etc. Wide-and-tall rather than one fixed-column
-- row per step because the metric set differs by module and this project's convention (params,
-- claims) is already "one immutable row per fact," not a growing fixed schema per new metric.
CREATE TABLE IF NOT EXISTS backtest_metrics (
    backtest_run_id   INTEGER NOT NULL REFERENCES backtest_runs (backtest_run_id),
    season              VARCHAR NOT NULL,
    gameweek              INTEGER NOT NULL,
    tier                    VARCHAR NOT NULL CHECK (tier IN ('cold', 'warm', 'mature')),
    metric_name               VARCHAR NOT NULL,
    metric_value                DOUBLE NOT NULL,
    PRIMARY KEY (backtest_run_id, season, gameweek, metric_name)
);

-- Recalibration audit trail. write_param() below never activates a version -- resolve_param() is
-- explicit-version-only, per params.py's own docstring -- so a proposal row is what records *why*
-- a version change is being suggested; a human still has to edit the explicit version-number
-- argument scripts/run_ingestion.py passes for that param family before it takes effect, same
-- discipline as every other version bump in this project. xi_club_concentration_cap (M5) never
-- appears here by design -- see backtest.py's report_concentration_sensitivity() instead.
CREATE SEQUENCE IF NOT EXISTS seq_recalibration_proposal START 1;

CREATE TABLE IF NOT EXISTS recalibration_proposals (
    proposal_id           BIGINT PRIMARY KEY DEFAULT nextval('seq_recalibration_proposal'),
    backtest_run_id         INTEGER NOT NULL REFERENCES backtest_runs (backtest_run_id),
    param_family              VARCHAR NOT NULL,
    param_key                   VARCHAR NOT NULL,
    dimensions                    VARCHAR,   -- JSON, matches params.py's dimensions concept
    old_params_version              INTEGER,
    new_params_version                INTEGER NOT NULL,
    old_value                           DOUBLE,
    new_value                             DOUBLE,
    metric_name                             VARCHAR NOT NULL,
    metric_before                             DOUBLE NOT NULL,
    metric_after                                DOUBLE NOT NULL,
    status                                        VARCHAR NOT NULL CHECK (status IN ('pending', 'confirmed', 'rejected')) DEFAULT 'pending',
    reviewed_by                                     VARCHAR,
    reviewed_at                                       TIMESTAMP
);
