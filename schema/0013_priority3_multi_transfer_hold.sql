-- Priority 3 -- bounded 2-for-2 combinatorial multi-transfer search + "hold this transfer"
-- recommendation.

-- Mirrors transfer_recommendations' own shape, but players_out/players_in are JSON lists of
-- exactly 2 player_uids each (sorted) rather than two more player_uid columns per side --
-- widening this table to fixed player_out_a/player_out_b/player_in_a/player_in_b columns
-- would bake in an assumption (exactly 2-for-2, never a different combo size) this project
-- already knows won't necessarily hold if a future extension goes beyond 2-for-2.
CREATE TABLE IF NOT EXISTS multi_transfer_recommendations (
    run_id               INTEGER NOT NULL REFERENCES transfer_plan_runs (run_id),
    rank                 INTEGER NOT NULL,
    players_out          VARCHAR NOT NULL,   -- JSON list of 2 player_uids, sorted
    players_in           VARCHAR NOT NULL,   -- JSON list of 2 player_uids, sorted
    combined_price_out   DOUBLE NOT NULL,
    combined_price_in    DOUBLE NOT NULL,
    horizon_value_gain   DOUBLE NOT NULL,
    transfer_cost        DOUBLE NOT NULL,
    net_value            DOUBLE NOT NULL,
    PRIMARY KEY (run_id, rank)
);

-- One row per planning invocation -- the hold-vs-transfer-now decision itself, not a ranked
-- list (there is exactly one recommended action: "hold", "transfer_now", or
-- "no_action_available"). detail carries the full evaluate_hold_recommendation() return dict
-- (best_transfer_now / best_hold_single_next_week / best_hold_multi_next_week), the same
-- JSON-blob pattern chip_evaluations.detail already uses for exactly this reason.
CREATE TABLE IF NOT EXISTS hold_recommendations (
    run_id               INTEGER PRIMARY KEY REFERENCES transfer_plan_runs (run_id),
    recommended_action   VARCHAR NOT NULL CHECK (recommended_action IN ('hold', 'transfer_now', 'no_action_available')),
    transfer_now_value   DOUBLE,
    hold_value           DOUBLE,
    detail               VARCHAR NOT NULL   -- JSON, full evaluate_hold_recommendation() return dict
);
