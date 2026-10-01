-- EP recipe: the finishing-skill calibration expected_points.run() applied (NULL = off, or a
-- row written before this column existed). Monte Carlo and uncertainty read it back so they
-- simulate the same goal/assist rates the EP was built with.
ALTER TABLE ep_model_versions ADD COLUMN IF NOT EXISTS finishing_skill_params_version INTEGER;
