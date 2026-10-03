-- The steps a walk-forward run set out to walk and the seasons it covers (backtest.run()), so a
-- run stopped by its time budget -- or cut off by a job limit -- reads as incomplete rather than
-- as a shorter run, and can be resumed (docs/reports/2026-10_open_issues.md: the minutes price
-- prior arm hit the 330-minute job limit and left no summary at all).
ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS steps_planned INTEGER;
ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS seasons VARCHAR;
