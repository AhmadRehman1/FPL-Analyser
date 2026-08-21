-- Part 2 of the rank-relative work: real net-transfers-in/out and price-change columns.
--
-- These already exist in the raw, already-ingested playerstats.csv (transfers_in,
-- transfers_out, transfers_in_event, transfers_out_event, cost_change_event,
-- cost_change_start) -- confirmed by checking the actual CSV column list, same gap
-- selected_by_percent itself was in before an earlier round of this project's work wired it
-- through. This is "reconcile already-ingested-but-dropped columns," not new ingestion.
--
-- transfers_in/transfers_out are season-cumulative (like goals_scored, minutes);
-- transfers_in_event/transfers_out_event/cost_change_event are real per-gameweek deltas
-- (live, like event_points); cost_change_start is the cumulative price change since the
-- season's launch price (also live -- it's "current state relative to launch," not a
-- running total of anything).
ALTER TABLE fact_player_season_stats ADD COLUMN IF NOT EXISTS transfers_in INTEGER;
ALTER TABLE fact_player_season_stats ADD COLUMN IF NOT EXISTS transfers_out INTEGER;
ALTER TABLE fact_player_season_stats ADD COLUMN IF NOT EXISTS transfers_in_event INTEGER;
ALTER TABLE fact_player_season_stats ADD COLUMN IF NOT EXISTS transfers_out_event INTEGER;
ALTER TABLE fact_player_season_stats ADD COLUMN IF NOT EXISTS cost_change_event DOUBLE;
ALTER TABLE fact_player_season_stats ADD COLUMN IF NOT EXISTS cost_change_start DOUBLE;
