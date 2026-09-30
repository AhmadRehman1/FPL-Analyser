# Prompt: implement model improvements (for Kimi)

Use this after Kimi has run the review prompt (`kimi_model_review.md`). Paste everything below
the line into a Kimi session that can clone and push to `AhmadRehman1/FPL-Analyser`. If Kimi
can't push, it should output each change as a patch instead, and you apply it.

---

You are implementing model improvements in `AhmadRehman1/FPL-Analyser`, a Python + DuckDB
Fantasy Premier League model (`src/fpl_quant/`). Your earlier review produced a ranked list of
changes. Implement them **one at a time**, each as its own branch and pull request, and prove
each one improves the model before asking for it to be merged.

## The goal

More season-long FPL points. Today the walk-forward loses to the average manager by **-0.71
pts/GW**. The biggest known problems are premium attackers under-predicted (+0.93 residual at
£9m+), cheap defenders and keepers over-predicted, and a minutes model that scores worse than a
uniform baseline (log score -1.256 vs -1.099). Details are in
`docs/reports/2026-10_model_status_refresh.md`.

## Workflow for each change

1. **Branch.** Start from the latest `master`. Use a `bt/<short-name>` branch for the change.
   Pushing to `bt/**` runs `.github/workflows/branch_walkforward.yml`, which runs the full
   walk-forward against the cached DB and uploads a scoreboard (`scripts/walkforward_summary.py`).
   That scoreboard is your evidence.
2. **Keep it small.** One idea per branch. Don't mix a minutes-model fix with an optimizer change,
   or you can't tell which one moved the numbers.
3. **Make it testable.** If the change is a parameter or a toggle, add it as a flag to
   `scripts/run_walkforward.py` (see the existing `--lambda`, `--assist-prior-xa` flags) so the
   old and new behaviour can run side by side. New parameters go through the versioned mechanism
   in `params.py`, not as hardcoded constants.
4. **Run the checks locally before pushing:**
   ```bash
   pip install -r requirements.lock -r requirements-dev.txt
   ruff check .
   mypy src/
   PYTHONPATH=src python -m pytest tests/ -q
   npm test
   ```
   Add or update tests for anything you change. Never skip, delete or loosen an existing test to
   get green; if a test fails, either your change is wrong or the test encodes a rule you need to
   understand first.
5. **Compare against the control.** The control is the same walk-forward on `master` with no
   flags. Report the before/after for at least: beats-avg-manager pts/GW, minutes log score and
   Brier, price-band residuals (`<5.0`, `5.0-7.0`, `7.0-9.0`, `9.0+`), position residuals, and
   global EP bias. `scripts/compare_backtest_metrics.py` diffs two metric dumps into a table.
6. **Open a pull request into `master`** only if the headline pts/GW improves, or a calibration
   metric improves clearly without the headline getting worse. Otherwise, write up the negative
   result in the PR description and leave it open as a draft, since that is still useful.

## Pull request description

Each PR must include:
- the problem it fixes, with the evidence (`file:line` and numbers)
- what changed and why this approach
- the before/after metric table from the walk-forward, with the workflow run link
- risks, and anything it might make worse
- how to roll it back (usually: revert the param version or drop the flag)

## Hard rules

- **Never merge your own PR and never push to `master`.** The owner merges after reading the
  numbers and CI.
- **No leakage.** Every feature must be knowable at the prediction deadline. Don't bypass
  `backtest.asof_scope()` or query fact tables without it inside the walk-forward.
  `tests/test_schema_invariants.py` guards this; it must stay green.
- **Don't change how the walk-forward scores** (`score_gameweek()`, the headline metric) in the
  same PR as a model change. If the scoring itself needs fixing, that's its own PR, so model
  gains can't come from moving the goalposts.
- **Don't edit generated data** (`data/dashboard/*.json`, `data/recalibration/*.json`,
  `report_history/`). Scheduled jobs own those files.
- **Don't auto-confirm recalibration proposals.** If a change needs a new parameter value, propose
  it and show the evidence; the owner confirms it.
- Keep `README.md` and the relevant `docs/` report up to date with what you changed.

## Suggested order

Start with whatever your review ranked highest. If you have no ranking, use this order:
1. Make the walk-forward pass the live `rate_shrinkage_params` version to `ep.run()`, so
   backtest and live use the same `k_minutes` (`run_gameweek_step()` in `backtest.py`). This is
   a correctness fix, and it makes every later comparison trustworthy.
2. Minutes model: replace the hard `avail == 0 => p_0min = 1` rule with a calibrated
   probability, and fix regular starters getting too little P(60+).
3. Premium under-prediction: shrink per-90 rates toward a price- or role-conditioned prior
   instead of the plain position average.
4. Captaincy: pick the captain by expected points (optionally adjusted for effective
   ownership) and test dropping the 3x captain variance weight.
5. Fit `risk_aversion_params.lambda_value` with the walk-forward instead of the invented 0.15.

When you finish a change, stop and summarise: branch, PR link, before/after table, and what you
would do next.
