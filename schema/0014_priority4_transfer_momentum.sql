-- Priority 4 -- forward-looking price-change-timing signal.
--
-- price_momentum_by_player() (M8) is retrospective (reports now_cost/selected_by_percent
-- movement that already happened) and explicitly informational-only. This gives the
-- forward-looking layer a real input to work from: this-gameweek-only net transfer activity
-- (FPL's own transfers_in_event/transfers_out_event field names, reused verbatim -- not
-- invented -- since a source CSV mirroring the official API is expected to use the same
-- names), the same signal FPL's own price-change algorithm is actually driven by.
--
-- ADD COLUMN with no inline constraint, same idempotent pattern already established in
-- 0010/0011/0012 -- safe even though fact_player_season_stats has no FK dependents blocking
-- it, kept consistent with the rest of this project's migrations regardless.
ALTER TABLE fact_player_season_stats ADD COLUMN IF NOT EXISTS transfers_in_event DOUBLE;
ALTER TABLE fact_player_season_stats ADD COLUMN IF NOT EXISTS transfers_out_event DOUBLE;
