-- The EP recipe: the versions expected_points.run() was called with that change its outputs
-- but were never recorded, so M4/M6 (and anyone re-reading ep_outputs) couldn't tell which
-- k_minutes, fixture scaling, set-piece uplift or assist calibration produced a row. NULL on
-- rows written before this migration means "not recorded" (readers fall back to their own
-- argument), not "off".
ALTER TABLE ep_model_versions ADD COLUMN IF NOT EXISTS set_piece_params_version INTEGER;
ALTER TABLE ep_model_versions ADD COLUMN IF NOT EXISTS fixture_params_version INTEGER;
ALTER TABLE ep_model_versions ADD COLUMN IF NOT EXISTS rate_shrinkage_params_version INTEGER;
ALTER TABLE ep_model_versions ADD COLUMN IF NOT EXISTS assist_calibration_params_version INTEGER;
-- written by expected_points.run() from now on, so a NULL recipe column is unambiguous
ALTER TABLE ep_model_versions ADD COLUMN IF NOT EXISTS recipe_recorded BOOLEAN DEFAULT FALSE;
