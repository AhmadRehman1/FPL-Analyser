-- M2 -- Minutes Model.
-- Per-run metadata + the three-state P(0)/P(1-59)/P(60+) output, one row per player per run.

CREATE SEQUENCE IF NOT EXISTS seq_minutes_model_version START 1;

CREATE TABLE IF NOT EXISTS minutes_model_versions (
    model_version                  INTEGER PRIMARY KEY DEFAULT nextval('seq_minutes_model_version'),
    calibration_asof_date          DATE NOT NULL,
    target_season                  VARCHAR NOT NULL,
    decay_params_version           INTEGER NOT NULL,   -- minutes_model_decay_params
    adjustment_params_version      INTEGER NOT NULL,   -- minutes_adjustment_params
    shrinkage_params_version       INTEGER NOT NULL,   -- minutes_model_shrinkage_params (the "10" threshold)
    fact_multiplier_params_version INTEGER NOT NULL,
    lookback_seasons                VARCHAR NOT NULL,  -- JSON list
    created_at                      TIMESTAMP NOT NULL DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS minutes_model_outputs (
    model_version                        INTEGER NOT NULL REFERENCES minutes_model_versions (model_version),
    player_uid                           VARCHAR NOT NULL REFERENCES dim_player (player_uid),
    position                             VARCHAR,
    p_start_historical_own                DOUBLE,   -- NULL if the player has zero historical rows
    p_start_historical_position_avg       DOUBLE NOT NULL,
    weight_own                            DOUBLE NOT NULL,
    p_start_historical_final              DOUBLE NOT NULL,
    logit_adjustment_total                DOUBLE NOT NULL,
    p_start_final                         DOUBLE NOT NULL,
    p_used_as_sub_given_not_started        DOUBLE NOT NULL,
    p_0min                                 DOUBLE NOT NULL,
    p_1_59min                              DOUBLE NOT NULL,
    p_60plus_min                           DOUBLE NOT NULL,
    competitive_matches_last_2_seasons     INTEGER NOT NULL,
    PRIMARY KEY (model_version, player_uid)
);
