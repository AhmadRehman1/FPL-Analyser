-- Per-player, per-gameweek totals over ep_outputs / uncertainty_outputs, which stay one row per
-- player per FIXTURE. In a double gameweek a player has two fixture rows; every reader that
-- wants "this player's gameweek" (the squad optimizer's candidate pool, bench order, bands,
-- scoring against event_points) reads these views instead, so a double counts both fixtures
-- rather than crashing a (player, run) key or silently keeping one row.
--
-- Means and variances add across fixtures (fixtures treated as independent; the same player's
-- minutes are correlated across them, which this ignores). Quantiles, skew and kurtosis don't
-- add, so a multi-fixture row uses a normal approximation around the summed mean and variance;
-- a single-fixture row passes the stored Cornish-Fisher values through unchanged.

CREATE OR REPLACE VIEW ep_gameweek_outputs AS
SELECT
    model_version,
    player_uid,
    count(*)                 AS n_fixtures,
    sum(ep_appearance)       AS ep_appearance,
    sum(ep_goals)            AS ep_goals,
    sum(ep_assists)          AS ep_assists,
    sum(ep_clean_sheet)      AS ep_clean_sheet,
    sum(ep_goals_conceded)   AS ep_goals_conceded,
    sum(ep_defcon)           AS ep_defcon,
    sum(ep_bonus)            AS ep_bonus,
    sum(ep_saves)            AS ep_saves,
    sum(ep_penalty_save)     AS ep_penalty_save,
    sum(ep_cards)            AS ep_cards,
    sum(ep_own_goal)         AS ep_own_goal,
    sum(ep_total)            AS ep_total,
    sum(expected_bps)        AS expected_bps
FROM ep_outputs
GROUP BY model_version, player_uid;

CREATE OR REPLACE VIEW uncertainty_gameweek_outputs AS
WITH per_fixture AS (
    SELECT u.*, o.ep_total
    FROM uncertainty_outputs u
    JOIN uncertainty_model_versions umv ON umv.model_version = u.model_version
    JOIN ep_outputs o
        ON o.model_version = umv.ep_model_version
        AND o.player_uid = u.player_uid
        AND o.fixture_match_id = u.fixture_match_id
),
summed AS (
    SELECT
        model_version,
        player_uid,
        count(*)                   AS n_fixtures,
        sum(ep_total)              AS mean_total,
        sum(var_appearance)        AS var_appearance,
        sum(var_goals)             AS var_goals,
        sum(var_assists)           AS var_assists,
        sum(var_clean_sheet)       AS var_clean_sheet,
        sum(var_goals_conceded)    AS var_goals_conceded,
        sum(var_defcon)            AS var_defcon,
        sum(var_bonus)             AS var_bonus,
        sum(var_saves)             AS var_saves,
        sum(var_total)             AS var_total,
        any_value(skew)            AS skew_1,
        any_value(excess_kurtosis) AS excess_kurtosis_1,
        any_value(quantile_05)     AS quantile_05_1,
        any_value(quantile_25)     AS quantile_25_1,
        any_value(quantile_75)     AS quantile_75_1,
        any_value(quantile_95)     AS quantile_95_1
    FROM per_fixture
    GROUP BY model_version, player_uid
)
SELECT
    model_version, player_uid, n_fixtures, mean_total,
    var_appearance, var_goals, var_assists, var_clean_sheet, var_goals_conceded,
    var_defcon, var_bonus, var_saves, var_total,
    CASE WHEN n_fixtures = 1 THEN skew_1 ELSE 0.0 END                                   AS skew,
    CASE WHEN n_fixtures = 1 THEN excess_kurtosis_1 ELSE 0.0 END                        AS excess_kurtosis,
    CASE WHEN n_fixtures = 1 THEN quantile_05_1 ELSE mean_total - 1.6448536 * sqrt(var_total) END AS quantile_05,
    CASE WHEN n_fixtures = 1 THEN quantile_25_1 ELSE mean_total - 0.6744898 * sqrt(var_total) END AS quantile_25,
    CASE WHEN n_fixtures = 1 THEN quantile_75_1 ELSE mean_total + 0.6744898 * sqrt(var_total) END AS quantile_75,
    CASE WHEN n_fixtures = 1 THEN quantile_95_1 ELSE mean_total + 1.6448536 * sqrt(var_total) END AS quantile_95
FROM summed;
