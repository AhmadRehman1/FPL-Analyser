-- A4: ep_model_versions previously had no record of which (if any) opt-in qualitative-
-- evidence params a given run used -- set_piece_params_version, decay_params_version,
-- fact_multiplier_params_version (A2/A3), role_shift_params_version (A2). Without these,
-- explain_qualitative_adjustment() has no way to know, after the fact, whether a stored
-- ep_model_version's e_goals/e_assists numbers were qualitative-adjusted at all, let alone
-- which claim-weighting params produced them -- the same "what actually fed this number"
-- gap M9's provenance-trail spec exists to close for minutes_model. All nullable: every
-- pre-existing ep_model_versions row (and every future run that leaves the adjustment
-- opted out) legitimately has none of these set.
ALTER TABLE ep_model_versions ADD COLUMN IF NOT EXISTS set_piece_params_version INTEGER;
ALTER TABLE ep_model_versions ADD COLUMN IF NOT EXISTS decay_params_version INTEGER;
ALTER TABLE ep_model_versions ADD COLUMN IF NOT EXISTS fact_multiplier_params_version INTEGER;
ALTER TABLE ep_model_versions ADD COLUMN IF NOT EXISTS role_shift_params_version INTEGER;
