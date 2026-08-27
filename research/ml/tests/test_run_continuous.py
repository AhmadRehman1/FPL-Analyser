"""24/7 continuous runner tests: the loop runs repeatedly with fresh seeds, logs every run to
the rolling log, tracks the best, and never crashes on a per-run failure."""

from __future__ import annotations

import pandas as pd

from research.ml import contract as C
from research.ml import run_continuous as rc


def _fake_manifest(ml_points: float) -> dict:
    return {
        "run_timestamp_utc": "2026-01-01T00:00:00Z",
        "n_walk_forward_folds": 5,
        "dataset_rows": 108,
        "season_points": {"quant_manager": 100.0, "ml_manager": ml_points, "ml_beats_quant": ml_points > 100.0},
    }


def test_run_forever_loops_logs_and_tracks_best(seeded_db, monkeypatch, tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(C, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(C, "RUN_LOG_CSV", tmp_path / "experiment_runs.csv")

    # second run scores higher -> becomes the best
    points = [100.0, 110.0]

    def fake_run(seasons=None, con=None, random_seed=42, fold_mode="gameweek"):
        i = random_seed - 42
        return {"manifest": _fake_manifest(points[i]), "comparison": pd.DataFrame(),
                "improvement": pd.DataFrame(), "season_points": pd.DataFrame()}

    monkeypatch.setattr(rc, "run_experiment", fake_run)

    n = rc.run_forever(sleep_seconds=0, fold_mode="gameweek", base_seed=42,
                      seasons=["2024-2025", "2025-2026"], failure_backoff=0, max_iterations=2)

    assert n == 2
    log = pd.read_csv(tmp_path / "experiment_runs.csv")
    assert len(log) == 2
    assert list(log["run_index"]) == [0, 1]
    assert list(log["seed"]) == [42, 43]
    assert list(log["ml_manager_points"]) == [100.0, 110.0]


def test_run_forever_survives_failure_and_continues(seeded_db, monkeypatch, tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(C, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(C, "RUN_LOG_CSV", tmp_path / "experiment_runs.csv")

    def fake_run(seasons=None, con=None, random_seed=42, fold_mode="gameweek"):
        i = random_seed - 42
        if i == 0:
            raise RuntimeError("database missing")  # first run fails, must not kill the loop
        return {"manifest": _fake_manifest(100.0), "comparison": pd.DataFrame(),
                "improvement": pd.DataFrame(), "season_points": pd.DataFrame()}

    monkeypatch.setattr(rc, "run_experiment", fake_run)

    n = rc.run_forever(sleep_seconds=0, fold_mode="gameweek", base_seed=42,
                      seasons=None, failure_backoff=0, max_iterations=2)

    assert n == 2
    log = pd.read_csv(tmp_path / "experiment_runs.csv")
    assert len(log) == 1  # only the successful run is logged
    assert log["ml_manager_points"].iloc[0] == 100.0
