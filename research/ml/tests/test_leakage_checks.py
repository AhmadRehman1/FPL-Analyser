"""Leakage-check tests. Each check is tested in both its passing and failing (negative) form."""

from __future__ import annotations

import pandas as pd
import pytest

from research.ml import contract as C
from research.ml.dataset_builder import build_minimal_dataset
from research.ml.leakage_checks import (
    LeakageError,
    check_chronological_split,
    check_ep_model_version_matches_step,
    check_label_not_a_feature,
    check_no_forbidden_feature_columns,
    check_no_unskipped_dgw,
    check_prediction_precedes_outcome,
)


def test_ep_model_version_matches_step(seeded_db):
    df = build_minimal_dataset(seeded_db)
    # happy path: no exception
    check_ep_model_version_matches_step(df, seeded_db)


def test_ep_model_version_mismatch_raises(seeded_db):
    df = build_minimal_dataset(seeded_db)
    # tamper: claim a different ep_model_version for one row
    df = df.copy()
    df.loc[df.index[0], C.COL_EP_MODEL_VERSION] = 999999
    with pytest.raises(LeakageError, match="ep_model_version"):
        check_ep_model_version_matches_step(df, seeded_db)


def test_prediction_precedes_outcome(seeded_db):
    df = build_minimal_dataset(seeded_db)
    check_prediction_precedes_outcome(df, seeded_db)


def test_prediction_not_before_kickoff_raises(seeded_db):
    df = build_minimal_dataset(seeded_db)
    df = df.copy()
    # move prediction timestamp to after kickoff (use a far-future date)
    df[C.COL_PRED_TIMESTAMP] = pd.Timestamp("2030-01-01")
    with pytest.raises(LeakageError, match="not strictly before"):
        check_prediction_precedes_outcome(df, seeded_db)


def test_no_forbidden_feature_columns_passes():
    check_no_forbidden_feature_columns(["rolling_points_5", "is_home", "player_uid"])  # no exception


def test_no_forbidden_feature_columns_detects_leak():
    with pytest.raises(LeakageError, match="forbidden"):
        check_no_forbidden_feature_columns(["rolling_points_5", "minutes"])


def test_label_not_a_feature():
    feats = ["rolling_points_5", "player_uid"]
    check_label_not_a_feature(feats)  # no exception
    with pytest.raises(LeakageError, match="label column"):
        check_label_not_a_feature(["rolling_points_5", C.COL_ACTUAL])


def test_no_unskipped_dgw(seeded_db):
    df = build_minimal_dataset(seeded_db)
    check_no_unskipped_dgw(df, seeded_db)  # no exception -- synthetic seasons are single-GW


def test_chronological_split_passes(seeded_db):
    df = build_minimal_dataset(seeded_db)
    train = df[df["season"] == "2024-2025"]
    test = df[df["season"] == "2025-2026"]
    check_chronological_split(train, test)  # no exception


def test_chronological_split_detects_overlap(seeded_db):
    df = build_minimal_dataset(seeded_db)
    # train and test both include 2024-2025 -> not a valid chronological split
    train = df[df["season"] == "2024-2025"]
    test = df[df["season"].isin(["2024-2025", "2025-2026"])]
    with pytest.raises(LeakageError, match="walk-forward split"):
        check_chronological_split(train, test)
