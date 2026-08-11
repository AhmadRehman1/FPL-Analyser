-- M3 -- Expected Points Engine.
-- Every category is its own sub-model, all conditioned on the same M2 minutes
-- distribution; category expectations are summed for total EP (linearity of expectation).

CREATE SEQUENCE IF NOT EXISTS seq_ep_model_version START 1;

CREATE TABLE IF NOT EXISTS ep_model_versions (
    model_version                INTEGER PRIMARY KEY DEFAULT nextval('seq_ep_model_version'),
    calibration_asof_date        DATE NOT NULL,
    target_season                VARCHAR NOT NULL,
    team_strength_model_version  INTEGER NOT NULL REFERENCES team_strength_model_versions (model_version),
    minutes_model_version        INTEGER NOT NULL REFERENCES minutes_model_versions (model_version),
    scoring_matrix_params_version INTEGER NOT NULL,
    bps_params_version           INTEGER NOT NULL,
    bps_tau_params_version       INTEGER NOT NULL,
    created_at                   TIMESTAMP NOT NULL DEFAULT current_timestamp
);

-- One row per player per upcoming fixture (a player can have 0 fixtures in a blank
-- gameweek or 1 in a normal one; DGW/multi-fixture handling is out of scope for v1, per
-- M8's own research finding that 2026-27 currently has no scheduled doubles/blanks).
CREATE TABLE IF NOT EXISTS ep_outputs (
    model_version      INTEGER NOT NULL REFERENCES ep_model_versions (model_version),
    player_uid         VARCHAR NOT NULL REFERENCES dim_player (player_uid),
    fixture_match_id   VARCHAR NOT NULL REFERENCES fact_match (match_id),
    ep_appearance      DOUBLE NOT NULL,
    ep_goals           DOUBLE NOT NULL,
    ep_assists         DOUBLE NOT NULL,
    ep_clean_sheet     DOUBLE NOT NULL,
    ep_goals_conceded  DOUBLE NOT NULL,
    ep_defcon          DOUBLE NOT NULL,
    ep_bonus           DOUBLE NOT NULL,
    ep_saves           DOUBLE NOT NULL,
    ep_penalty_save    DOUBLE NOT NULL,
    ep_cards           DOUBLE NOT NULL,
    ep_own_goal        DOUBLE NOT NULL,
    ep_total           DOUBLE NOT NULL,
    expected_bps       DOUBLE NOT NULL,  -- mu_i feeding the Plackett-Luce sub-model
    PRIMARY KEY (model_version, player_uid, fixture_match_id)
);
