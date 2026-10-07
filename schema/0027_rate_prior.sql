-- EP recipe: the rate prior expected_points.run() applied (NULL = off, or a row written before
-- this column existed) -- see expected_points.resolve_rate_prior().
ALTER TABLE ep_model_versions ADD COLUMN IF NOT EXISTS rate_prior_params_version INTEGER;
