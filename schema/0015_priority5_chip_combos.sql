-- Priority 5 -- chip-combo sequencing (Wildcard-into-Bench-Boost, Free-Hit-into-Triple-
-- Captain). One row per (run, combo_type) -- exactly two combo_types exist, each producing
-- one recommendation per planning invocation, same "one row per real decision" shape
-- hold_recommendations already established (schema/0013) rather than a ranked list like
-- transfer_recommendations (there's nothing to rank here, only one combo per type).
CREATE TABLE IF NOT EXISTS chip_combo_evaluations (
    run_id             INTEGER NOT NULL REFERENCES transfer_plan_runs (run_id),
    combo_type         VARCHAR NOT NULL CHECK (combo_type IN ('wildcard_bench_boost', 'free_hit_triple_captain')),
    recommended_combo  BOOLEAN NOT NULL,
    detail             VARCHAR NOT NULL,   -- JSON, full evaluate_*_combo() return dict
    PRIMARY KEY (run_id, combo_type)
);
