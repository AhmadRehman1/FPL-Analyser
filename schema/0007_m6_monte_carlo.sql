-- M6 -- Monte Carlo Simulation Engine.
-- One "run" = one full-gameweek Monte Carlo simulation for one squad_optimizer_runs
-- selection (the query-scoped candidate pool per spec, not the full 577-player league).
-- Z_fixture's calibrated variance and the derived deterministic seed are stored on the run
-- row itself (like team_strength's elo_regression_a0/a1 -- a derived quantity from real
-- data, not a param_versions literal).

CREATE SEQUENCE IF NOT EXISTS seq_monte_carlo_model_version START 1;

CREATE TABLE IF NOT EXISTS monte_carlo_run_versions (
    model_version                INTEGER PRIMARY KEY,
    calibration_asof_date        DATE NOT NULL,
    squad_optimizer_run_id       INTEGER NOT NULL REFERENCES squad_optimizer_runs (run_id),
    ep_model_version              INTEGER NOT NULL REFERENCES ep_model_versions (model_version),
    minutes_model_version         INTEGER NOT NULL REFERENCES minutes_model_versions (model_version),
    team_strength_model_version   INTEGER NOT NULL REFERENCES team_strength_model_versions (model_version),
    uncertainty_model_version     INTEGER NOT NULL REFERENCES uncertainty_model_versions (model_version),
    rho_residual_params_version   INTEGER NOT NULL,
    z_fixture_lambda_representative DOUBLE NOT NULL,
    z_fixture_variance              DOUBLE NOT NULL,
    n_antithetic_pairs           INTEGER NOT NULL,
    query_id                     VARCHAR NOT NULL,
    seed                         BIGINT NOT NULL,
    created_at                   TIMESTAMP NOT NULL DEFAULT current_timestamp
);

-- One row per (squad player, simulated gameweek realization). realization_index in
-- [0, 2*n_antithetic_pairs): the first n_antithetic_pairs are the "primary" draws (u), the
-- second n_antithetic_pairs are their antithetic partners (1-u), aligned index-for-index --
-- realization_index k and k+n_antithetic_pairs are an antithetic pair for every fixture and
-- every player simultaneously (spec's "full random stream", not per-category pairing).
CREATE TABLE IF NOT EXISTS monte_carlo_player_totals (
    model_version        INTEGER NOT NULL REFERENCES monte_carlo_run_versions (model_version),
    player_uid            VARCHAR NOT NULL REFERENCES dim_player (player_uid),
    realization_index     INTEGER NOT NULL,
    minutes_state          VARCHAR NOT NULL CHECK (minutes_state IN ('0', '1_59', '60plus')),
    pts_appearance        DOUBLE NOT NULL,
    pts_goals             DOUBLE NOT NULL,
    pts_assists           DOUBLE NOT NULL,
    pts_clean_sheet       DOUBLE NOT NULL,
    pts_goals_conceded    DOUBLE NOT NULL,
    pts_defcon            DOUBLE NOT NULL,
    pts_bonus             DOUBLE NOT NULL,
    pts_saves             DOUBLE NOT NULL,
    total_points          DOUBLE NOT NULL,
    PRIMARY KEY (model_version, player_uid, realization_index)
);

-- Aggregated empirical distribution per squad player -- the direct M8 chip-value input and
-- the M9-facing summary, so downstream modules don't have to re-aggregate 10,000 rows
-- themselves every time.
CREATE TABLE IF NOT EXISTS monte_carlo_player_summary (
    model_version    INTEGER NOT NULL REFERENCES monte_carlo_run_versions (model_version),
    player_uid       VARCHAR NOT NULL REFERENCES dim_player (player_uid),
    mean_total       DOUBLE NOT NULL,
    var_total        DOUBLE NOT NULL,
    quantile_05      DOUBLE NOT NULL,
    quantile_25      DOUBLE NOT NULL,
    quantile_75      DOUBLE NOT NULL,
    quantile_95      DOUBLE NOT NULL,
    min_total        DOUBLE NOT NULL,
    max_total        DOUBLE NOT NULL,
    PRIMARY KEY (model_version, player_uid)
);

-- Empirical covariance across every squad-player pair (not sparse like M4's table -- the
-- squad-scoped pool is small, C(15,2)=105 pairs at most, so there's no storage-scale reason
-- to omit zero/near-zero pairs the way M4 omits different-fixture pairs). m4_covariance is
-- carried alongside for direct validation of M4's Sigma -- specifically its rho_residual
-- placeholder -- without a second query. relationship extends M4's teammate/opponent
-- vocabulary with 'independent' for squad-player pairs that don't share a fixture this
-- gameweek at all (their simulated covariance should empirically land near zero, which is
-- itself part of what validates the mechanism).
CREATE TABLE IF NOT EXISTS monte_carlo_empirical_covariance (
    model_version         INTEGER NOT NULL REFERENCES monte_carlo_run_versions (model_version),
    player_uid_a          VARCHAR NOT NULL REFERENCES dim_player (player_uid),
    player_uid_b          VARCHAR NOT NULL REFERENCES dim_player (player_uid),
    relationship           VARCHAR NOT NULL CHECK (relationship IN ('teammate', 'opponent', 'independent')),
    empirical_covariance  DOUBLE NOT NULL,
    m4_covariance         DOUBLE,
    PRIMARY KEY (model_version, player_uid_a, player_uid_b)
);
