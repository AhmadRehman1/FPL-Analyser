-- Priority 10 Phase B (partial) -- rank instrumentation, ahead of the full joint field
-- simulator. See docs/priority10_field_simulator_design.md.
--
-- The model-managed team's real goal is overall rank ~top 100k. nightly_backtest.yml only
-- ever scores model_squad_realized_points - avg_manager_points (a synthetic POINTS delta) --
-- which does not measure rank at all. This table records, per completed 2026-27 gameweek,
-- where a squad actually landed against a real, stratified sample of rival squads
-- (fact_rival_squad_sample, schema/0017), plus a decomposition of the rank gap into
-- captaincy / template-coverage / differential / bench. Forward-only by construction: FPL's
-- API serves only current-season picks, so there is no way to rank-score the 2024-25/2025-26
-- walk-forward -- this starts accumulating from 2026-27 GW1.
--
-- Unlike fact_rival_squad_sample (immutable observations of real people's picks), a rank
-- score is a DERIVED quantity -- the attribution logic here will improve over time, so
-- re-scoring a (season, event, subject, subject_id) REPLACES its row rather than being a
-- no-op. scripts/rank_autopsy.py deletes then re-inserts.
CREATE TABLE IF NOT EXISTS fact_squad_rank_score (
    season               VARCHAR NOT NULL,
    event                  INTEGER NOT NULL,   -- FPL's own name for gameweek number
    subject                  VARCHAR NOT NULL CHECK (subject IN ('model_optimal', 'account')),
    subject_id                 VARCHAR NOT NULL,   -- squad_optimizer_runs.run_id (model_optimal) or entry_id (account), as text
    realized_points              INTEGER NOT NULL,   -- the subject XI's real FPL score that gameweek, captain multiplier applied
    n_rivals                       INTEGER NOT NULL,  -- sampled rival squads with a settled score to compare against
    n_beaten                         INTEGER NOT NULL,
    n_tied                             INTEGER NOT NULL,
    percentile                          DOUBLE NOT NULL,   -- 0..100, (n_beaten + 0.5*n_tied) / n_rivals * 100
    estimated_rank                        INTEGER,          -- percentile projected onto total_players; NULL if total_players unknown
    total_players                          INTEGER,          -- bootstrap-static total_players at scoring time
    real_overall_rank                       INTEGER,         -- accounts only: FPL's own settled overall rank for that gameweek; NULL for model_optimal
    sample_bands                              VARCHAR NOT NULL,  -- JSON: the ACHIEVED rank bands behind this sample, [[start, end, n], ...]
    attribution                                 VARCHAR NOT NULL,  -- JSON: {captaincy: {...}, template_coverage: {...}, differentials: {...}, bench: {...}}
    scored_at                                     TIMESTAMP NOT NULL DEFAULT current_timestamp,
    PRIMARY KEY (season, event, subject, subject_id)
);
