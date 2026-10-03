-- EP recipe: the BPS calibration expected_points.run() applied (NULL = off, or a row written
-- before this column existed) -- see expected_points._bps_residual_table().
ALTER TABLE ep_model_versions ADD COLUMN IF NOT EXISTS bps_calibration_params_version INTEGER;
