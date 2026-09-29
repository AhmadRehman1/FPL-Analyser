-- M7 recalibration gate (docs/reports/2026-10_model_status_refresh.md, Finding 5). A proposal
-- now carries the score on gameweek-steps its own search never saw (purged, embargoed K-fold)
-- and the searched grid's bounds, so the gate can refuse in-sample-only and grid-edge winners.
-- Nullable: older proposals (and refits not yet wired for held-out scoring) leave them NULL,
-- and the gate treats a missing held-out score as "hold", never as a pass.
ALTER TABLE recalibration_proposals ADD COLUMN IF NOT EXISTS holdout_metric_before DOUBLE DEFAULT NULL;
ALTER TABLE recalibration_proposals ADD COLUMN IF NOT EXISTS holdout_metric_after DOUBLE DEFAULT NULL;
ALTER TABLE recalibration_proposals ADD COLUMN IF NOT EXISTS grid_min DOUBLE DEFAULT NULL;
ALTER TABLE recalibration_proposals ADD COLUMN IF NOT EXISTS grid_max DOUBLE DEFAULT NULL;
