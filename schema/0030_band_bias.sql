-- EP recipe: lever L2, the band bias (docs/plans/2026-10_beat_the_template.md, Phase 6).
-- expected_points.run() adds each player's (position, price band) adjustment
-- (fpl_quant.residual_levers.band_adjustments) to ep_total after the components are summed, split
-- across a double gameweek's fixtures in proportion to their EP and never taking a fixture's EP
-- below zero, as L1 does. ep_band_adj keeps that share on its own, so
-- ep_total - ep_crowd_adj - ep_band_adj is the projection before every lever. NULL / 0 = off.
ALTER TABLE ep_model_versions ADD COLUMN IF NOT EXISTS band_bias_params_version INTEGER;
ALTER TABLE ep_outputs ADD COLUMN IF NOT EXISTS ep_band_adj DOUBLE DEFAULT 0.0;

-- schema/0029_crowd_prior.sql's per-gameweek view, with the band adjustment summed too.
CREATE OR REPLACE VIEW ep_gameweek_outputs AS
SELECT
    model_version,
    player_uid,
    count(*)                         AS n_fixtures,
    sum(ep_appearance)               AS ep_appearance,
    sum(ep_goals)                    AS ep_goals,
    sum(ep_assists)                  AS ep_assists,
    sum(ep_clean_sheet)              AS ep_clean_sheet,
    sum(ep_goals_conceded)           AS ep_goals_conceded,
    sum(ep_defcon)                   AS ep_defcon,
    sum(ep_bonus)                    AS ep_bonus,
    sum(ep_saves)                    AS ep_saves,
    sum(ep_penalty_save)             AS ep_penalty_save,
    sum(ep_cards)                    AS ep_cards,
    sum(ep_own_goal)                 AS ep_own_goal,
    sum(ep_total)                    AS ep_total,
    sum(expected_bps)                AS expected_bps,
    sum(coalesce(ep_crowd_adj, 0.0)) AS ep_crowd_adj,
    sum(coalesce(ep_band_adj, 0.0))  AS ep_band_adj
FROM ep_outputs
GROUP BY model_version, player_uid;
