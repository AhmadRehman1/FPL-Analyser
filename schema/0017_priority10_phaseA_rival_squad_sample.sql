-- Priority 10 Phase A -- a real, bounded sample of rival squads from FPL's own public API
-- (leagues-classic standings + entry picks), per docs/priority10_field_simulator_design.md.
-- Deliberately minimal fields: entry_id (an opaque numeric FPL id, needed only for row
-- identity/dedup), player picks, captain flag, and the entry's public league rank at sample
-- time -- no manager name, team name, or any other personally-identifying field is fetched or
-- stored, even though the source API exposes them. This is intentionally the smallest table
-- that lets a later Phase B/C reconstruct "who did the sampled field own and captain," nothing
-- more.
CREATE TABLE IF NOT EXISTS fact_rival_squad_sample (
    entry_id      BIGINT NOT NULL,
    season        VARCHAR NOT NULL,
    event         INTEGER NOT NULL,   -- FPL's own name for gameweek number, matching its API
    player_uid    VARCHAR NOT NULL REFERENCES dim_player (player_uid),
    is_captain    BOOLEAN NOT NULL,
    multiplier    INTEGER NOT NULL,   -- 0 (not in the XI that week, e.g. an unused bench slot), 1, 2, or 3 (Triple Captain)
    league_rank   INTEGER,            -- this entry's Overall-league rank at sample time, for reference/filtering only
    _ingested_at  TIMESTAMP NOT NULL,
    PRIMARY KEY (entry_id, season, event, player_uid)
);
