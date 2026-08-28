"""24/7 continuous simulation runner: loop the ML experiment forever, each iteration with a
fresh seed, logging every run and tracking the best FPL-manager points found so far.

    python -m research.ml.run_continuous
    python -m research.ml.run_continuous --sleep 60 --fold-mode gameweek

This is the "keep doing machine learning simulating previous years through walk-forward as many
times as possible, 24/7, acting as an FPL manager trying to get the most points" engine. It
writes one timestamped results subdir per run under results/runs/ and appends a row to the
rolling experiment_runs.csv each iteration, so progress is never lost on a crash/restart.

Requires the repo's DuckDB to be populated first (scripts/run_ingestion.py +
scripts/run_backtest.py). Research-only: does not touch the live Quant model or production
recommendations (spec: "do not prematurely integrate ML into the production app").
"""

from __future__ import annotations

import argparse
import time
import traceback
from datetime import datetime, timezone

from . import contract as C
from .experiment import run_experiment, redirect_results_to
from .experiment import _append_run_log  # noqa: F401  (re-exported helper, used below)


def _best_so_far() -> dict | None:
    try:
        import pandas as pd
        if not C.RUN_LOG_CSV.exists():
            return None
        df = pd.read_csv(C.RUN_LOG_CSV)
        if df.empty:
            return None
        best = df.loc[df["ml_manager_points"].idxmax()].to_dict()
        return best
    except Exception:
        return None


def run_forever(sleep_seconds: float, fold_mode: str, base_seed: int, seasons: tuple[str, ...] | None, failure_backoff: float = 60.0, max_iterations: int | None = None) -> int:
    """Run the experiment in a loop forever (or until `max_iterations` is reached). Returns the
    number of iterations attempted. Each success is logged to experiment_runs.csv; each failure
    is swallowed and the loop backs off so a persistently-unavailable DB never spins."""
    run_index = 0
    while max_iterations is None or run_index < max_iterations:
        seed = base_seed + run_index
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = C.RUNS_DIR / f"run_{run_index:04d}_{ts}"
        try:
            with redirect_results_to(run_dir):
                result = run_experiment(seasons=seasons, con=None, random_seed=seed, fold_mode=fold_mode)
            m = result["manifest"]
            sp = m["season_points"]
            _append_run_log({
                "run_index": run_index, "seed": seed,
                "timestamp_utc": m["run_timestamp_utc"],
                "fold_mode": fold_mode, "n_folds": m["n_walk_forward_folds"],
                "dataset_rows": m["dataset_rows"],
                "quant_manager_points": sp["quant_manager"],
                "ml_manager_points": sp["ml_manager"],
                "ml_beats_quant": sp["ml_beats_quant"],
                "run_dir": str(run_dir),
            })
            best = _best_so_far()
            best_pts = best.get("ml_manager_points") if best else sp["ml_manager"]
            print(
                f"[24/7] run #{run_index} seed={seed} | ML {sp['ml_manager']:.1f} vs Quant "
                f"{sp['quant_manager']:.1f} pts ({'ML wins' if sp['ml_beats_quant'] else 'quant holds'}) "
                f"| best ML so far: {best_pts:.1f} pts",
                flush=True,
            )
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
        except Exception as exc:  # never let a single run kill the loop
            print(f"[24/7] run #{run_index} seed={seed} FAILED: {exc}", flush=True)
            traceback.print_exc()
            # back off so a persistently-unavailable DB (or any recurring error) does not spin
            time.sleep(max(failure_backoff, sleep_seconds))
        run_index += 1
    return run_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ML FPL-manager simulation 24/7.")
    parser.add_argument("--sleep", type=float, default=0.0, help="seconds to sleep between runs (default: 0 = back-to-back)")
    parser.add_argument("--fold-mode", choices=["gameweek", "season"], default="gameweek")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--seasons", nargs="*", default=None)
    parser.add_argument("--max-iterations", type=int, default=None, help="stop after N iterations (default: none = run forever)")
    args = parser.parse_args()
    seasons = tuple(args.seasons) if args.seasons else None
    print(f"[24/7] starting continuous simulation (fold_mode={args.fold_mode}, sleep={args.sleep}s, "
          f"max_iterations={args.max_iterations}). Ctrl+C to stop. Results: {C.RUNS_DIR}", flush=True)
    try:
        run_forever(args.sleep, args.fold_mode, args.base_seed, seasons, max_iterations=args.max_iterations)
    except KeyboardInterrupt:
        print("[24/7] stopped by user.", flush=True)


if __name__ == "__main__":
    main()
