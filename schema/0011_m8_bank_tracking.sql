-- M8 -- Bank (leftover budget) tracking.
--
-- Real bug fixed here: evaluate_transfers() previously required price_in <= price_out with no
-- way to draw on money already saved up by prior transfers -- the module's own docstring
-- disclosed this as "no banked-budget tracking exists yet, so this conservatively assumes zero
-- bank." Practical effect: a manager holding several mid-priced players could never be
-- recommended an upgrade to a genuinely premium player (e.g. selling down three ~7.5m forwards
-- but never being offered the one ~15.5m elite forward those sales could fund), since no single
-- outgoing player was ever expensive enough on its own. bank now accumulates
-- (price_out - price_in) across accepted transfers in apply_recommendation(), and
-- evaluate_transfers() relaxes its constraint to price_in <= price_out + bank.
--
-- ADD COLUMN with a DEFAULT (no inline NOT NULL -- DuckDB doesn't support adding a column with
-- an inline constraint) is safe and idempotent even on a table with FK dependents, same pattern
-- as 0010's is_manager_snapshot addition.
ALTER TABLE manager_state_versions ADD COLUMN IF NOT EXISTS bank DOUBLE DEFAULT 0.0;

-- Prices captured at recommendation time so apply_recommendation() can compute the bank delta
-- directly from the stored recommendation, without re-deriving prices from horizon EP data that
-- may have moved on by the time a recommendation is actually accepted.
ALTER TABLE transfer_recommendations ADD COLUMN IF NOT EXISTS price_out DOUBLE DEFAULT 0.0;
ALTER TABLE transfer_recommendations ADD COLUMN IF NOT EXISTS price_in DOUBLE DEFAULT 0.0;
