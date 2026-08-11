-- M4 -- Uncertainty & Correlation Layer.
-- Per-player total variance (law of total covariance across M3's categories, gated on
-- M2's three-state minutes distribution) plus a sparse cross-player covariance matrix for
-- M5's quadratic objective. Cornish-Fisher quantiles are reporting/explainability output
-- only (M9) -- confirmed NOT wired into M5's optimization objective, per spec.

CREATE SEQUENCE IF NOT EXISTS seq_uncertainty_model_version START 1;

CREATE TABLE IF NOT EXISTS uncertainty_model_versions (
    model_version           INTEGER PRIMARY KEY DEFAULT nextval('seq_uncertainty_model_version'),
    calibration_asof_date   DATE NOT NULL,
    ep_model_version        INTEGER NOT NULL REFERENCES ep_model_versions (model_version),
    minutes_model_version   INTEGER NOT NULL REFERENCES minutes_model_versions (model_version),
    team_strength_model_version INTEGER NOT NULL REFERENCES team_strength_model_versions (model_version),
    rho_residual_params_version INTEGER NOT NULL,
    created_at              TIMESTAMP NOT NULL DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS uncertainty_outputs (
    model_version      INTEGER NOT NULL REFERENCES uncertainty_model_versions (model_version),
    player_uid         VARCHAR NOT NULL REFERENCES dim_player (player_uid),
    fixture_match_id   VARCHAR NOT NULL REFERENCES fact_match (match_id),
    var_appearance     DOUBLE NOT NULL,
    var_goals          DOUBLE NOT NULL,
    var_assists        DOUBLE NOT NULL,
    var_clean_sheet    DOUBLE NOT NULL,
    var_goals_conceded DOUBLE NOT NULL,
    var_defcon         DOUBLE NOT NULL,
    var_bonus          DOUBLE NOT NULL,
    var_saves          DOUBLE NOT NULL,
    var_total          DOUBLE NOT NULL,
    skew               DOUBLE NOT NULL,
    excess_kurtosis    DOUBLE NOT NULL,
    quantile_05        DOUBLE NOT NULL,  -- "floor" (M9 display)
    quantile_25        DOUBLE NOT NULL,
    quantile_75        DOUBLE NOT NULL,
    quantile_95        DOUBLE NOT NULL,  -- "ceiling" (M9 display)
    PRIMARY KEY (model_version, player_uid, fixture_match_id)
);

-- Sparse: only nonzero pairs are stored (teammates in the same fixture, or opponents in
-- the same fixture). Different-fixture pairs within a gameweek are a confirmed zero
-- covariance per spec -- not stored at all, not stored-as-zero.
CREATE TABLE IF NOT EXISTS cross_player_covariance (
    model_version     INTEGER NOT NULL REFERENCES uncertainty_model_versions (model_version),
    player_uid_a      VARCHAR NOT NULL REFERENCES dim_player (player_uid),
    player_uid_b      VARCHAR NOT NULL REFERENCES dim_player (player_uid),
    fixture_match_id  VARCHAR NOT NULL REFERENCES fact_match (match_id),
    relationship      VARCHAR NOT NULL CHECK (relationship IN ('teammate', 'opponent')),
    covariance        DOUBLE NOT NULL,
    PRIMARY KEY (model_version, player_uid_a, player_uid_b)
);
