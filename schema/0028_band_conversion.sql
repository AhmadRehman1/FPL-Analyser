-- EP recipe: the pooled band conversion expected_points.run() applied (conversion_params,
-- pseudo_xg; NULL = off, or a row written before this column existed). Each position and price
-- band's goals per xG and assists per xA in earlier seasons scale that band's goal and assist
-- rates. Monte Carlo and uncertainty read it back so they simulate the same rates.
ALTER TABLE ep_model_versions ADD COLUMN IF NOT EXISTS conversion_params_version INTEGER;
