"""Dataset-builder tests. Verifies the player×gameweek join, the five leakage-invariant
assertions, and that DGW rows are skipped before reaching downstream code."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.ml.dataset_builder import build_dataset, build_minimal_dataset
from research.ml.leakage_checks import assert_minimal_dataset_invariants

COLS = ["player_uid", "season", "gameweek", "quant_prediction", "actual_points", "residual"]


def test_minimal_dataset_builds_and_has_required_columns(seeded_db):
    df = build_minimal_dataset(seeded_db)
    assert len(df) > 0
    for col in COLS + ["prediction_timestamp", "position", "team_uid", "opponent_uid", "home_away"]:
        assert col in df.columns, f"missing column {col}"
    assert df.attrs["seasons"] == ["2024-2025", "2025-2026"]
    # residual = actual - quant
    assert np.allclose(df["residual"], df["actual_points"] - df["quant_prediction"])
    # one row per (player, season, gameweek) for players who had a fixture
    counts = df.groupby(["player_uid", "season", "gameweek"]).size()
    assert (counts == 1).all()


def test_minimal_dataset_runs_leakage_invariants(seeded_db):
    # build_minimal_dataset calls assert_minimal_dataset_invariants internally; this also
    # exercises the direct call path.
    df = build_minimal_dataset(seeded_db)
    assert_minimal_dataset_invariants(df, seeded_db, df.attrs["skip_log"])


def test_full_dataset_includes_features(seeded_db):
    df = build_dataset(seeded_db, with_features=True)
    assert len(df) == len(build_minimal_dataset(seeded_db))
    assert "rolling_points_5" in df.columns
    assert "is_home" in df.columns
    assert "fixture_difficulty" in df.columns
    # no forbidden leakage columns leaked in
    forbidden = {"minutes", "goals_scored", "total_points", "bps", "expected_goals"}
    assert not (forbidden & set(df.columns))


def test_labels_use_event_points_not_total_points(seeded_db):
    """actual_points must come from event_points (per-GW), never total_points (cumulative)."""
    df = build_minimal_dataset(seeded_db)
    raw = seeded_db.execute(
        "SELECT player_uid, season, gw, event_points, total_points FROM fact_player_season_stats"
    ).fetchdf()
    merged = df.merge(raw, left_on=["player_uid", "season", "gameweek"], right_on=["player_uid", "season", "gw"])
    assert np.allclose(merged["actual_points"], merged["event_points"])
    # the label is NOT the cumulative total_points
    assert not np.allclose(merged["actual_points"], merged["total_points"]) or merged["event_points"].equals(
        merged["total_points"]
    )


def test_prediction_timestamp_strictly_before_kickoff(seeded_db):
    """The data_asof (prediction timestamp) must precede every kickoff in the gameweek."""
    df = build_minimal_dataset(seeded_db)
    for row in df.itertuples():
        kickoff = seeded_db.execute(
            "SELECT min(kickoff_time) FROM fact_match WHERE season = ? AND gameweek = ?",
            [row.season, row.gameweek],
        ).fetchone()[0]
        assert pd.Timestamp(row.prediction_timestamp) < pd.Timestamp(kickoff)


def test_dgw_steps_are_skipped_and_logged(seeded_db):
    """A gameweek that becomes a double-gameweek (a second fixture for the same player under the
    same ep_model_version) is skipped before reaching the dataset and recorded in skip_log --
    the builder never silently duplicates rows or trains on undefined DGW aggregation."""
    con = seeded_db
    step = con.execute(
        "SELECT season, gameweek, ep_model_version FROM backtest_gameweek_steps ORDER BY season, gameweek LIMIT 1"
    ).fetchone()
    season, gw, ep_mv = step
    # a second match in the same gameweek -> makes it a double gameweek
    con.execute(
        "INSERT INTO fact_match (match_id, season, gameweek, kickoff_time, home_team_uid, away_team_uid, "
        "home_score, away_score, finished, competition, _ingested_at) "
        "VALUES ('m_dup', ?, ?, '2024-08-19 15:00:00', 'team_a', 'team_b', 1, 1, TRUE, 'Premier League', '2024-08-10')",
        [season, gw]
    )
    con.execute(
        "INSERT INTO ep_outputs (model_version, player_uid, fixture_match_id, ep_appearance, ep_goals, "
        "ep_assists, ep_clean_sheet, ep_goals_conceded, ep_defcon, ep_bonus, ep_saves, "
        "ep_penalty_save, ep_cards, ep_own_goal, ep_total, expected_bps) "
        "VALUES (?, 'p1', 'm_dup', 1.0, 0.3, 0.1, 0, 0, 0, 0.2, 0, 0, 0, 0, 3.0, 20.0)",
        [ep_mv]
    )
    df = build_minimal_dataset(con)
    skip_log = df.attrs["skip_log"]
    dgw_skips = [s for s in skip_log if s.get("reason") == "double_gameweek"]
    assert any(s["season"] == season and int(s["gameweek"]) == int(gw) for s in dgw_skips)
    # no duplicate player rows in the dataset (the DGW was excluded, not aggregated)
    dup = df.groupby(["player_uid", "season", "gameweek"]).size()
    assert (dup <= 1).all()
