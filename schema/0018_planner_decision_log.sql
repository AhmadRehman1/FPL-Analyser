-- Roadmap P1 item -- planner decision backtest (hold vs. use), per
-- docs/plans/2026-08_roadmap_plan.md Track C. Logs, per real tracked manager and gameweek, what
-- scripts/run_transfer_planner_for_real_squad.py actually recommended at generation time.
-- This is new: neither transfer_plan_runs (this project's own dashboard snapshot, overwritten
-- every run -- see data/dashboard/real_squad_<entry_id>.json) nor data/report_history/ (the
-- generic, non-manager-specific squad snapshot) persists a recommendation across gameweeks
-- today, so "did following this recommendation work" has never been answerable.
--
-- actual_action_taken/realized_points_* stay NULL until Plan Phase C-2 fills them in, once the
-- next gameweek's real squad (what the manager actually did) and real results are both known.
CREATE TABLE IF NOT EXISTS planner_decision_log (
    entry_id                                    BIGINT NOT NULL,
    target_season                               VARCHAR NOT NULL,
    target_gameweek                             INTEGER NOT NULL,  -- the gameweek this recommendation is FOR (the run's plan_for_gameweek)
    run_id                                      INTEGER REFERENCES transfer_plan_runs (run_id),
    recommended_action                          VARCHAR NOT NULL CHECK (recommended_action IN ('hold', 'transfer_now', 'no_action_available')),
    recommended_transfer_out                    VARCHAR,  -- top-ranked transfer_recommendations.player_out, only when recommended_action = 'transfer_now'
    recommended_transfer_in                     VARCHAR,
    recommended_chip                            VARCHAR,  -- chip_type of any chip_evaluations row with recommended = TRUE this run, else NULL
    recommended_captain                         VARCHAR,  -- suggested captain, only when it differs from the manager's actual current captain
    actual_action_taken                         VARCHAR,
    realized_points_actual                      DOUBLE,
    realized_points_if_recommendation_followed  DOUBLE,
    logged_at                                   TIMESTAMP NOT NULL,
    PRIMARY KEY (entry_id, target_season, target_gameweek)
);
