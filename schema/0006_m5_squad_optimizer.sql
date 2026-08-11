-- M5 -- Squad Optimizer.
-- The module the original v1 rebuild exists because of (the documented lambda=0 back-five
-- failure happened here). divergence_check_passed is a hard gate, not an audit footnote --
-- squad_optimizer.run() raises and refuses to store a squad selection if it fails.

CREATE SEQUENCE IF NOT EXISTS seq_squad_optimizer_run START 1;

CREATE TABLE IF NOT EXISTS squad_optimizer_runs (
    run_id                     INTEGER PRIMARY KEY DEFAULT nextval('seq_squad_optimizer_run'),
    calibration_asof_date      DATE NOT NULL,
    target_season              VARCHAR NOT NULL,
    target_gameweek            INTEGER NOT NULL,
    ep_model_version           INTEGER NOT NULL REFERENCES ep_model_versions (model_version),
    uncertainty_model_version  INTEGER NOT NULL REFERENCES uncertainty_model_versions (model_version),
    lambda_params_version      INTEGER NOT NULL,
    lambda_value                DOUBLE NOT NULL,
    guardrail_params_version   INTEGER NOT NULL,
    divergence_check_passed    BOOLEAN NOT NULL,
    divergence_check_note      VARCHAR,
    solver_status               VARCHAR NOT NULL,
    objective_value              DOUBLE,
    created_at                    TIMESTAMP NOT NULL DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS squad_optimizer_selections (
    run_id      INTEGER NOT NULL REFERENCES squad_optimizer_runs (run_id),
    player_uid  VARCHAR NOT NULL REFERENCES dim_player (player_uid),
    in_squad    BOOLEAN NOT NULL,
    in_xi       BOOLEAN NOT NULL,
    is_captain  BOOLEAN NOT NULL,
    is_vice     BOOLEAN NOT NULL,
    PRIMARY KEY (run_id, player_uid)
);
