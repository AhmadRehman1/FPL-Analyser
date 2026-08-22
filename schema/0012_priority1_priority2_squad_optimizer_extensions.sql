-- Priority 1 (ownership/EO, field-covariance) + Priority 2 (bench-quality floor,
-- team-concentration risk, solve-quality transparency) additions to M5's squad_optimizer_runs.
--
-- All nullable, no-default columns: every one of these features is opt-in at the run() call
-- site (squad_optimizer.run()'s own docstring). A run that doesn't pass the corresponding
-- *_params_version keyword argument leaves that column NULL, and solve()'s own behavior for
-- that feature is an exact no-op -- see squad_optimizer.solve()'s own per-term comments.
--
-- ADD COLUMN with no inline constraint is safe and idempotent even on a table with FK
-- dependents (monte_carlo_run_versions, squad_optimizer_selections both reference
-- squad_optimizer_runs) -- same pattern already established in 0010/0011.
ALTER TABLE squad_optimizer_runs ADD COLUMN IF NOT EXISTS ownership_params_version INTEGER;
ALTER TABLE squad_optimizer_runs ADD COLUMN IF NOT EXISTS risk_posture_params_version INTEGER;
ALTER TABLE squad_optimizer_runs ADD COLUMN IF NOT EXISTS field_covariance_params_version INTEGER;
ALTER TABLE squad_optimizer_runs ADD COLUMN IF NOT EXISTS bench_quality_params_version INTEGER;
ALTER TABLE squad_optimizer_runs ADD COLUMN IF NOT EXISTS concentration_risk_params_version INTEGER;

-- Solve-quality transparency (Priority 2): SCIP's own proven relative gap at termination --
-- 0.0 iff the solve was proven globally optimal (matches solver_status='optimal' already
-- being SCIP's own claim of a proven-optimal solution; stored explicitly anyway so a
-- time/gap-limited run's ACTUAL gap number is visible, not just "not optimal").
ALTER TABLE squad_optimizer_runs ADD COLUMN IF NOT EXISTS mip_gap DOUBLE;
