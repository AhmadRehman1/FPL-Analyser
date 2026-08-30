"""M7 walk-forward backtest ONLY -- the `backtest.run()` half of scripts/run_backtest.py,
without the recalibration tail.

Why this exists: `research/ml/experiment.py` reads `backtest_gameweek_steps` (the walk-forward's
per-gameweek asof predictions) and nothing else -- it never touches recalibration proposals. But
`scripts/run_backtest.py` runs `backtest.run()` AND `backtest.recalibrate()`, and the recalibration
tail (the lambda-grid SCIP re-solves) alone exceeds a GitHub Actions job's 6-hour hard cap (a real
`weekly_backtest.yml` dispatch on 2026-08-25 was cancelled at 5h having not finished). So the ML
experiment could never be provisioned with a walk-forward on a cloud runner.

`backtest.run()` on its own is the ~1-2 hour part (README's own estimate) and fits comfortably in
one Actions job. This script runs exactly that and prints the same run-summary block
`run_backtest.py` does, so `.github/workflows/ml_experiment.yml` can cache the resulting DB (now
carrying `backtest_gameweek_steps`) and hand it to `python -m research.ml.experiment`.

Recalibration is a separate concern and stays in `scripts/run_backtest.py` -- run that locally, or
on a runner without the 6h limit, when you actually want new `recalibration_proposals`.

Usage (from repo root):
    PYTHONPATH=src python scripts/run_walkforward.py
"""

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import backtest, db  # noqa: E402

# Reuse run_backtest.py's own version-resolution verbatim -- the walk-forward must measure the
# model against the same git-committed confirmed-seed versions every other script uses, not a
# hardcoded literal (see run_backtest.py's own comment on _param_versions).
from run_backtest import _param_versions, RECALIBRATION_SEED_DIR  # noqa: E402


def main() -> None:
    con = db.connect()
    active = backtest.active_recalibratable_versions(RECALIBRATION_SEED_DIR)
    param_versions = _param_versions(active)

    t0 = time.time()
    backtest_run_id = backtest.run(
        con, **param_versions, n_antithetic_pairs=5000, run_monte_carlo=True,
        notes="M7 walk-forward (ml_experiment.yml provisioning -- no recalibration)",
    )
    print(f"[backtest.run] {time.time() - t0:.1f}s -> backtest_run_id={backtest_run_id}")

    steps = con.execute(
        "SELECT tier, count(*), sum(CASE WHEN divergence_check_passed THEN 1 ELSE 0 END) "
        "FROM backtest_gameweek_steps WHERE backtest_run_id = ? GROUP BY tier ORDER BY tier",
        [backtest_run_id],
    ).fetchall()
    for tier, n, n_passed in steps:
        print(f"  tier={tier}: {n} steps, {n_passed} divergence-check passes")

    metrics = con.execute(
        "SELECT tier, metric_name, count(*), avg(metric_value) FROM backtest_metrics "
        "WHERE backtest_run_id = ? AND metric_name NOT LIKE 'realized%' GROUP BY tier, metric_name ORDER BY metric_name, tier",
        [backtest_run_id],
    ).fetchall()
    for tier, name, n, avg in metrics:
        print(f"  [{tier}] {name}: n={n} mean={avg:.4f}")

    n_pred = con.execute(
        "SELECT count(*) FROM backtest_gameweek_steps WHERE backtest_run_id = ? AND ep_model_version IS NOT NULL",
        [backtest_run_id],
    ).fetchone()[0]
    print(f"[walk-forward] {n_pred} gameweek steps carry an ep_model_version -- research.ml.experiment can now run")

    con.close()


if __name__ == "__main__":
    main()
