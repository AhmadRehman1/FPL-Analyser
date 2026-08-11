-- FPL Quant v2 -- M0 core schema.
-- Three-layer model: fact_raw (dynamic, see ingest_csv.py) -> fact_reconciled -> evidence_claims.
-- Every module's versioned parameters resolve through the single param_versions mechanism
-- (kickoff notes item 2) rather than one bespoke table per module.

-- ============================================================
-- Layer 1: fact_raw ingestion log
-- Actual raw tables (one per (season, source_relpath), all-VARCHAR, append-only across
-- ingestion runs) are created dynamically by ingest_csv.py. This log is the durable,
-- statically-defined record of what has been ingested and when.
-- ============================================================

CREATE SEQUENCE IF NOT EXISTS seq_ingestion_batch START 1;

CREATE TABLE IF NOT EXISTS fact_raw_ingestion_log (
    batch_id         BIGINT PRIMARY KEY DEFAULT nextval('seq_ingestion_batch'),
    raw_table_name   VARCHAR NOT NULL,
    season           VARCHAR NOT NULL,
    source_relpath   VARCHAR NOT NULL,
    source_file_hash VARCHAR NOT NULL,
    row_count        BIGINT NOT NULL,
    ingested_at      TIMESTAMP NOT NULL DEFAULT current_timestamp,
    -- byte-identical re-ingestion of the same file is a no-op, not a new batch
    UNIQUE (raw_table_name, source_file_hash)
);

-- ============================================================
-- Layer 2: fact_reconciled
-- Entity identity is not stable across seasons (M0 research finding) -- everything
-- joins on (normalized_name, team_code, season) via the alias tables below.
-- ============================================================

CREATE TABLE IF NOT EXISTS dim_team (
    team_uid       VARCHAR PRIMARY KEY,
    canonical_name VARCHAR NOT NULL,
    short_name     VARCHAR,
    -- context/audit flag only; the substantive promoted-club prior mechanism is M1's, not M0's
    prior_division VARCHAR,
    UNIQUE (canonical_name)
);

CREATE TABLE IF NOT EXISTS team_alias (
    alias_name   VARCHAR NOT NULL,
    season       VARCHAR NOT NULL,
    team_uid     VARCHAR NOT NULL REFERENCES dim_team (team_uid),
    alias_source VARCHAR,
    PRIMARY KEY (alias_name, season)
);

CREATE TABLE IF NOT EXISTS dim_player (
    player_uid     VARCHAR PRIMARY KEY,
    canonical_name VARCHAR NOT NULL,
    position       VARCHAR,
    UNIQUE (canonical_name)
);

CREATE TABLE IF NOT EXISTS player_alias (
    alias_name            VARCHAR NOT NULL,
    -- external sources (the evidence workbook, in particular) spell names differently
    -- (accents, middle names, nicknames) than FPL-Core-Insights does; lookups match on
    -- this normalized form, not the literal alias_name string.
    normalized_alias_name VARCHAR NOT NULL,
    team_code             VARCHAR NOT NULL,
    season                VARCHAR NOT NULL,
    player_uid            VARCHAR NOT NULL REFERENCES dim_player (player_uid),
    source_player_id      VARCHAR,
    PRIMARY KEY (alias_name, team_code, season)
);

CREATE INDEX IF NOT EXISTS idx_player_alias_normalized ON player_alias (normalized_alias_name);

-- match_id deduplication (rescheduled-fixture double-filing, M1 research finding) happens
-- here, before anything downstream sees a row.
CREATE TABLE IF NOT EXISTS fact_match (
    match_id      VARCHAR PRIMARY KEY,
    season        VARCHAR NOT NULL,
    gameweek      INTEGER,          -- 0 = preseason friendly (GW0)
    kickoff_time  TIMESTAMP,
    home_team_uid VARCHAR NOT NULL REFERENCES dim_team (team_uid),
    away_team_uid VARCHAR NOT NULL REFERENCES dim_team (team_uid),
    home_score    INTEGER,
    away_score    INTEGER,
    home_team_elo DOUBLE,
    away_team_elo DOUBLE,
    finished      BOOLEAN,
    competition   VARCHAR,
    _ingested_at  TIMESTAMP NOT NULL
);

-- Deliberate initial column subset: the fields M1-M9's frozen specs actually reference
-- (start/finish min for M2, goals/assists/DefCon components for M3, etc.), not a full
-- mirror of every playermatchstats/playerstats column. Full fidelity remains in fact_raw;
-- extend this table as later modules need more fields.
CREATE TABLE IF NOT EXISTS fact_player_match_stats (
    player_uid          VARCHAR NOT NULL REFERENCES dim_player (player_uid),
    match_id             VARCHAR NOT NULL REFERENCES fact_match (match_id),
    season               VARCHAR NOT NULL,
    start_min            INTEGER,
    finish_min           INTEGER,
    minutes_played       INTEGER,
    goals                INTEGER,
    assists              INTEGER,
    saves                INTEGER,
    goals_conceded       INTEGER,
    team_goals_conceded  INTEGER,
    tackles              INTEGER,
    clearances           INTEGER,
    interceptions        INTEGER,
    recoveries           INTEGER,
    blocks               INTEGER,
    _ingested_at         TIMESTAMP NOT NULL,
    PRIMARY KEY (player_uid, match_id)
);

CREATE TABLE IF NOT EXISTS fact_player_season_stats (
    player_uid                    VARCHAR NOT NULL REFERENCES dim_player (player_uid),
    season                        VARCHAR NOT NULL,
    gw                            INTEGER NOT NULL,
    now_cost                      DOUBLE,
    selected_by_percent           DOUBLE,
    ep_next                       DOUBLE,
    chance_of_playing_next_round  DOUBLE,
    status                        VARCHAR,
    minutes                       INTEGER,
    goals_scored                  INTEGER,
    assists                       INTEGER,
    bps                           INTEGER,
    expected_goals                DOUBLE,
    expected_assists              DOUBLE,
    expected_goals_per_90         DOUBLE,
    expected_assists_per_90       DOUBLE,
    defensive_contribution        DOUBLE,
    defensive_contribution_per_90 DOUBLE,
    saves_per_90                  DOUBLE,
    total_points                  INTEGER,
    event_points                  INTEGER,
    _ingested_at                  TIMESTAMP NOT NULL,
    PRIMARY KEY (player_uid, season, gw)
);

-- playerstats.csv mixes live and cumulative-to-date columns in the same file (M0 research
-- finding). This classifies fact_reconciled columns, not individual cells/rows, so a
-- consumer cannot mistake a zeroed pre-season stat for a real zero.
CREATE TABLE IF NOT EXISTS fact_reconciled_column_semantics (
    table_name  VARCHAR NOT NULL,
    column_name VARCHAR NOT NULL,
    semantics   VARCHAR NOT NULL CHECK (semantics IN ('live', 'cumulative_to_date', 'preseason_null')),
    notes       VARCHAR,
    PRIMARY KEY (table_name, column_name)
);

-- ============================================================
-- Layer 3: evidence_claims
-- ============================================================

CREATE TABLE IF NOT EXISTS sources (
    source_id              VARCHAR PRIMARY KEY,
    source_name             VARCHAR NOT NULL,
    source_type             VARCHAR NOT NULL
        CHECK (source_type IN ('official', 'journalist', 'specialist', 'community', 'system-derived')),
    base_reliability_score  DOUBLE,
    citation_count          BIGINT,
    source_notes            VARCHAR,
    last_reviewed_date      DATE,
    UNIQUE (source_name)
);

-- claim_id is a UUID generated at insert time (Python-side, for portability).
CREATE TABLE IF NOT EXISTS evidence_claims (
    claim_id               VARCHAR PRIMARY KEY,
    subject_entity_type    VARCHAR NOT NULL CHECK (subject_entity_type IN ('player', 'team', 'fixture')),
    -- polymorphic: dim_player.player_uid / dim_team.team_uid / fact_match.match_id depending
    -- on subject_entity_type above. DuckDB has no cross-table conditional FK; enforced in tests.
    subject_entity_id      VARCHAR NOT NULL,
    claim_type             VARCHAR NOT NULL,
    claim_value             VARCHAR,   -- JSON payload as text
    claim_value_numeric     DOUBLE,
    information_type        VARCHAR CHECK (information_type IN ('FACT', 'OPINION')),
    source_id               VARCHAR NOT NULL REFERENCES sources (source_id),
    -- snapshotted at ingestion -- never live-joined, so later re-scoring of a source never
    -- silently reweights old claims (M0 spec).
    source_reliability_score DOUBLE NOT NULL,
    confidence               DOUBLE,   -- normalized 0-1 from the workbook's 1-10 scale
    observed_date             DATE,
    ingested_date              TIMESTAMP NOT NULL,
    superseded_by               VARCHAR REFERENCES evidence_claims (claim_id),
    tab_origin                   VARCHAR,
    row_origin                    INTEGER,
    raw_text                       VARCHAR   -- audit pointer back to the source cell
);

-- Compound free-text cells mixing multiple assertions (6_Manager Database, 17_Pre-season
-- Match Reports -- M1b research finding, generalizes across both tabs). Decomposition into
-- atomic evidence_claims is a permanent, human-curated process (M1b spec), not NLP
-- extraction -- rows land here until a human produces the atomic claims below.
CREATE SEQUENCE IF NOT EXISTS seq_pending_decomp START 1;

CREATE TABLE IF NOT EXISTS claims_pending_manual_decomposition (
    id                BIGINT PRIMARY KEY DEFAULT nextval('seq_pending_decomp'),
    subject_hint      VARCHAR,   -- e.g. Club (tab 6) or Fixture (tab 17) free-text field
    raw_text          VARCHAR NOT NULL,
    information_type  VARCHAR,
    source_id         VARCHAR,
    source_date       DATE,
    confidence_raw    DOUBLE,
    fpl_relevance     VARCHAR,
    tab_origin        VARCHAR,
    row_origin        INTEGER,
    ingested_date     TIMESTAMP NOT NULL,
    decomposed        BOOLEAN NOT NULL DEFAULT FALSE
);

-- Deprecation enforcement: the ingestion loader's tab allowlist, made structural and
-- testable rather than an unwritten rule. Deprecated tabs are never processed for claims --
-- structurally absent, not silently ignored.
CREATE TABLE IF NOT EXISTS workbook_tab_allowlist (
    tab_name VARCHAR PRIMARY KEY,
    status   VARCHAR NOT NULL CHECK (status IN (
        'ingest_claims', 'ingest_manual_decomposition', 'reference_only',
        'audit_metadata', 'documentation', 'excluded_deprecated'
    )),
    reason   VARCHAR NOT NULL
);

-- ============================================================
-- Generic versioned-parameter mechanism (kickoff notes item 2)
-- claim_type_decay_params, source_tier_weights, model_decay_params,
-- minutes_adjustment_params, risk_aversion_params, simulation_params,
-- tc_risk_aversion_params, planning_horizon_params, etc. all resolve through this one
-- table, distinguished by param_family. New tuning = new params_version row, never an edit.
-- ============================================================

CREATE SEQUENCE IF NOT EXISTS seq_param_id START 1;

CREATE TABLE IF NOT EXISTS param_versions (
    id             BIGINT PRIMARY KEY DEFAULT nextval('seq_param_id'),
    param_family   VARCHAR NOT NULL,
    param_version  INTEGER NOT NULL,
    effective_date DATE NOT NULL,
    -- canonical (sorted-key) JSON string; '{}' for singleton params with no sub-dimension
    dimensions     VARCHAR NOT NULL DEFAULT '{}',
    param_key      VARCHAR NOT NULL,
    value_numeric  DOUBLE,
    value_text     VARCHAR,
    created_at     TIMESTAMP NOT NULL DEFAULT current_timestamp,
    UNIQUE (param_family, param_version, dimensions, param_key)
);

-- Named, read-only views over the shared mechanism, matching each module spec's own
-- table name exactly -- so callers reading the spec can find the table they expect,
-- without a second physical implementation existing behind it.
CREATE VIEW IF NOT EXISTS claim_type_decay_params AS
SELECT
    json_extract_string(dimensions, '$.claim_type') AS claim_type,
    value_numeric AS decay_half_life_days,
    param_version,
    effective_date
FROM param_versions
WHERE param_family = 'claim_type_decay_params' AND param_key = 'decay_half_life_days';

CREATE VIEW IF NOT EXISTS source_tier_weights AS
SELECT
    json_extract_string(dimensions, '$.source_type') AS source_type,
    value_numeric AS tier_weight,
    param_version,
    effective_date
FROM param_versions
WHERE param_family = 'source_tier_weights' AND param_key = 'tier_weight';

-- ============================================================
-- Snapshot discipline (M0 / exercised by M7)
-- Every model run pins a data_asof timestamp and a config snapshot of param versions;
-- fact_reconciled and evidence_claims are queried "as of" that timestamp only.
-- ============================================================

CREATE SEQUENCE IF NOT EXISTS seq_model_run START 1;

CREATE TABLE IF NOT EXISTS model_runs (
    run_id             BIGINT PRIMARY KEY DEFAULT nextval('seq_model_run'),
    data_asof          TIMESTAMP NOT NULL,
    created_at         TIMESTAMP NOT NULL DEFAULT current_timestamp,
    notes              VARCHAR,
    param_version_pins VARCHAR   -- JSON: {"risk_aversion_params": 1, ...}
);
