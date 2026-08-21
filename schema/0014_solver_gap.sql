-- Part 4 of the squad-quality-guardrails work: surface whether SCIP PROVED optimality (gap
-- <= ~0, a real proof) or hit its own limits/time=300 cutoff and returned a feasible-but-
-- unproven incumbent instead -- solver_status alone ("optimal" vs "timelimit") already exists,
-- but SCIP itself can report status="optimal" once it clears its own internal gap tolerance,
-- which is a genuine proof, distinct from a caller-side guarantee that no better solution
-- could exist without reading the actual numeric gap. Nullable: legitimately absent for any
-- run recorded before this column existed, and for the early-return no-solution-found case
-- (see solve()'s own "status not in (optimal, timelimit)" branch).
ALTER TABLE squad_optimizer_runs ADD COLUMN IF NOT EXISTS solver_gap DOUBLE;
ALTER TABLE squad_optimizer_runs ADD COLUMN IF NOT EXISTS proven_optimal BOOLEAN;
