-- M8 -- Transfer & Chip Strategy Planner.
-- Operates on an EXISTING squad, distinct from M5's from-scratch problem. Nothing in M0-M7
-- tracks "what the manager actually owns" -- manager_state_versions/manager_squad_holdings is
-- new, genuinely persisted state, bootstrapped once from a real squad_optimizer_runs
-- selection and evolved forward by M8's own accepted recommendations from then on (never
-- silently re-derived from a fresh M5 call -- that's the whole reason transfer planning is a
-- distinct problem from just re-running M5 every gameweek).

CREATE SEQUENCE IF NOT EXISTS seq_manager_state_version START 1;

-- produced_by_run_id (not a transfer_plan_runs.output_state_version column pointing the other
-- way) is a deliberate direction choice, not just a naming pick: DuckDB refuses to UPDATE a
-- row that has FK-referencing children in another table (transfer_recommendations/
-- chip_evaluations both FK into transfer_plan_runs), even when the updated column isn't the
-- key -- confirmed by direct testing, same category of limitation as the ALTER TABLE block
-- found earlier building M8. Setting this column once at INSERT time (apply_recommendation())
-- avoids ever needing to UPDATE a row with FK dependents.
CREATE TABLE IF NOT EXISTS manager_state_versions (
    state_version               INTEGER PRIMARY KEY DEFAULT nextval('seq_manager_state_version'),
    season                      VARCHAR NOT NULL,
    as_of_gameweek              INTEGER NOT NULL,   -- the gameweek this state is valid entering
    free_transfers_available    INTEGER NOT NULL,
    chips_used_set1             VARCHAR NOT NULL DEFAULT '[]',   -- JSON list, e.g. ["wildcard"]
    chips_used_set2              VARCHAR NOT NULL DEFAULT '[]',
    derived_from_state_version   INTEGER REFERENCES manager_state_versions (state_version),  -- NULL for the bootstrap row
    produced_by_run_id            INTEGER,  -- NULL for the bootstrap row; which transfer_plan_runs.run_id led here otherwise.
                                             -- No inline FK: transfer_plan_runs.input_state_version already references
                                             -- this table, so a reverse FK here would be circular within one CREATE TABLE
                                             -- ordering; enforced at the application layer instead (apply_recommendation()
                                             -- always sets it from a real transfer_plan_runs.run_id it just read).
    created_at                   TIMESTAMP NOT NULL DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS manager_squad_holdings (
    state_version   INTEGER NOT NULL REFERENCES manager_state_versions (state_version),
    player_uid      VARCHAR NOT NULL REFERENCES dim_player (player_uid),
    in_xi           BOOLEAN NOT NULL,
    is_captain      BOOLEAN NOT NULL,
    is_vice         BOOLEAN NOT NULL,
    PRIMARY KEY (state_version, player_uid)
);

-- One row per planning invocation. ep_model_versions/uncertainty_model_versions are JSON
-- OBJECTS keyed by str(gameweek) (one entry per horizon gameweek -- see transfer_planner.run()'s
-- own INSERT: json.dumps({str(gw): v for ...})) rather than a fixed-column shape, since
-- horizon_gameweeks is itself a versioned parameter, not a schema-time constant. Read them with
-- json.loads(...).get(str(target_gameweek)), the way decision_engine.py does.
CREATE SEQUENCE IF NOT EXISTS seq_transfer_plan_run START 1;

CREATE TABLE IF NOT EXISTS transfer_plan_runs (
    run_id                          INTEGER PRIMARY KEY DEFAULT nextval('seq_transfer_plan_run'),
    calibration_asof_date           DATE NOT NULL,
    target_season                   VARCHAR NOT NULL,
    target_gameweek                 INTEGER NOT NULL,
    input_state_version              INTEGER NOT NULL REFERENCES manager_state_versions (state_version),
    horizon_params_version           INTEGER NOT NULL,
    transfer_cost_params_version     INTEGER NOT NULL,
    ep_model_versions                VARCHAR NOT NULL,   -- JSON object {str(gameweek): ep_model_version}
    uncertainty_model_versions       VARCHAR NOT NULL,
    created_at                       TIMESTAMP NOT NULL DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS transfer_recommendations (
    run_id               INTEGER NOT NULL REFERENCES transfer_plan_runs (run_id),
    rank                 INTEGER NOT NULL,   -- 1 = best
    player_out           VARCHAR NOT NULL REFERENCES dim_player (player_uid),
    player_in            VARCHAR NOT NULL REFERENCES dim_player (player_uid),
    horizon_value_gain   DOUBLE NOT NULL,    -- sum(EP_new - EP_old) over the horizon, before cost
    transfer_cost        DOUBLE NOT NULL,    -- 0 if within free allocation, else points_per_hit * n_paid
    net_value            DOUBLE NOT NULL,
    PRIMARY KEY (run_id, rank)
);

CREATE TABLE IF NOT EXISTS chip_evaluations (
    run_id             INTEGER NOT NULL REFERENCES transfer_plan_runs (run_id),
    chip_type          VARCHAR NOT NULL CHECK (chip_type IN ('wildcard', 'free_hit', 'triple_captain', 'bench_boost')),
    recommended        BOOLEAN NOT NULL,
    score_or_gain       DOUBLE,
    detail              VARCHAR,   -- JSON, e.g. {"captain_candidate": "...", "target_gameweek": ...} for TC/BB
    gw19_urgent_flag     BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (run_id, chip_type)
);
