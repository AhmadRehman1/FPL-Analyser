-- Priority 7a -- Understat season-cumulative xG/xA/npxG/xGChain/xGBuildup, a genuinely
-- second, independent xG source alongside FPL-Core-Insights' own expected_goals/
-- expected_assists already in fact_player_season_stats. Deliberately its OWN table, not
-- folded into fact_player_season_stats: this is a live network fetch (one per season, not a
-- locally-provided file), so it doesn't fit ingest_csv.py's file-scanning fact_raw pattern,
-- and blending two independently-fitted goal-rate estimates into one number without a
-- principled combination rule would be exactly the silent-double-counting risk this project's
-- own conventions warn against -- see ingest_understat.py's own module docstring for how it's
-- actually used (informational second-opinion + a genuinely new xGChain/xGBuildup signal,
-- never silently blended into ep_goals/ep_assists).
CREATE TABLE IF NOT EXISTS fact_understat_player_season (
    player_uid           VARCHAR NOT NULL REFERENCES dim_player (player_uid),
    season                VARCHAR NOT NULL,
    understat_player_id      VARCHAR NOT NULL,  -- Understat's own numeric player id, for audit/re-fetch
    source_player_name         VARCHAR NOT NULL,  -- the raw name Understat used, for resolution audit
    games                         INTEGER NOT NULL,
    minutes                         INTEGER NOT NULL,
    goals                             INTEGER NOT NULL,
    assists                            INTEGER NOT NULL,
    xg                                   DOUBLE NOT NULL,
    npxg                                   DOUBLE NOT NULL,  -- non-penalty xG
    xa                                       DOUBLE NOT NULL,
    xgchain                                    DOUBLE NOT NULL,  -- xG of every possession this player touched
    xgbuildup                                    DOUBLE NOT NULL,  -- xGChain excluding key passes/assists/goals themselves
    shots                                          INTEGER NOT NULL,
    key_passes                                       INTEGER NOT NULL,
    _ingested_at                                       TIMESTAMP NOT NULL,
    PRIMARY KEY (player_uid, season)
);
