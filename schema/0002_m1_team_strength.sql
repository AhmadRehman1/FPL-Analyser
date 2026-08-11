-- M1 -- Team Strength Model (Dixon-Coles).
-- Global scalars (home_advantage, which xi/rho param_versions were used, Elo-regression
-- coefficients) live under one model_version; per-team attack/defence values reference it.
-- Not folded into the generic param_versions mechanism -- attack/defence are per-team
-- vectors, not scalar module parameters, a genuinely different shape.

CREATE SEQUENCE IF NOT EXISTS seq_team_strength_model_version START 1;

CREATE TABLE IF NOT EXISTS team_strength_model_versions (
    model_version          INTEGER PRIMARY KEY DEFAULT nextval('seq_team_strength_model_version'),
    calibration_asof_date  DATE NOT NULL,
    home_advantage         DOUBLE NOT NULL,
    xi_params_version      INTEGER NOT NULL,
    rho_params_version     INTEGER NOT NULL,
    reference_team_uid     VARCHAR NOT NULL,   -- fixed at (attack=0, defence=0) to resolve the model's 1 degree of freedom
    elo_regression_a0      DOUBLE,
    elo_regression_a1      DOUBLE,
    elo_regression_b0      DOUBLE,
    elo_regression_b1      DOUBLE,
    elo_regression_teams   INTEGER,            -- how many teams the regression was fit on
    seasons_fit            VARCHAR,            -- JSON list, e.g. ["2024-2025","2025-2026"]
    created_at             TIMESTAMP NOT NULL DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS team_strength_snapshots (
    model_version              INTEGER NOT NULL REFERENCES team_strength_model_versions (model_version),
    team_uid                   VARCHAR NOT NULL REFERENCES dim_team (team_uid),
    attack_mle                 DOUBLE,          -- NULL if the team wasn't in the MLE fit at all (e.g. never in our loaded PL history)
    defence_mle                DOUBLE,
    attack_elo_prior           DOUBLE,
    defence_elo_prior          DOUBLE,
    final_attack                DOUBLE NOT NULL,
    final_defence                DOUBLE NOT NULL,
    seasons_of_topflight_data     INTEGER NOT NULL,
    weight_own_data                DOUBLE NOT NULL,
    elo_at_calibration               DOUBLE,
    PRIMARY KEY (model_version, team_uid)
);
