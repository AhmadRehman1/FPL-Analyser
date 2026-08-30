"""Feature-engineering tests: asof-safety of rolling features, absence of forbidden columns,
and that prior-history never reaches into the prediction's own gameweek."""

from __future__ import annotations


from research.ml.dataset_builder import build_dataset
from research.ml import contract as C
from research.ml.feature_engineering import feature_columns
from research.ml.leakage_checks import assert_feature_matrix_invariants


def test_feature_columns_are_deterministic_and_no_label():
    cols = feature_columns()
    assert cols == feature_columns()  # cached / deterministic
    forbidden = {"minutes", "goals_scored", "assists", "bps", "total_points", "saves", "expected_goals"}
    assert not (forbidden & set(cols))
    assert C.COL_ACTUAL not in cols
    assert C.COL_RESIDUAL not in cols


def test_position_is_a_feature_column():
    # position is an approved static-identity feature (EXISTING_MODEL_AUDIT.md §9,
    # LEAKAGE_PROTOCOL.md §4) that was previously carried through the dataset for slicing only
    # and never fed to the residual model -- must be present so the model can condition on it.
    assert "position" in feature_columns()


def test_position_populated_for_every_row(seeded_db):
    df = build_dataset(seeded_db, with_features=True)
    assert df["position"].notna().all()
    assert set(df["position"].unique()) <= {"Goalkeeper", "Defender", "Midfielder", "Forward"}


def test_add_features_preserves_row_count(seeded_db):
    df = build_dataset(seeded_db, with_features=True)
    minimal = build_dataset(seeded_db, with_features=False)
    assert len(df) == len(minimal)
    assert set(minimal.columns) - set(df.columns) == set()  # all original columns retained


def test_feature_matrix_invariants_hold(seeded_db):
    # the feature matrix invariants (no forbidden columns, label not a feature) must hold
    assert_feature_matrix_invariants(feature_columns())


def test_rolling_features_are_nan_for_first_gameweek(seeded_db):
    """A player's first appearance in the dataset has no prior history -> rolling windows NaN."""
    df = build_dataset(seeded_db, with_features=True)
    gw1 = df[df["gameweek"] == 1]
    assert gw1["rolling_points_3"].isna().all()
    assert gw1["rolling_points_5"].isna().all()


def test_rolling_features_populate_after_history(seeded_db):
    """By gameweek 3, players who played in GW1-2 should have populated rolling features."""
    df = build_dataset(seeded_db, with_features=True)
    gw3 = df[df["gameweek"] == 3]
    # at least some players should have non-null rolling points
    assert gw3["rolling_points_3"].notna().any()


def test_no_feature_uses_future_gameweek(seeded_db):
    """A player's rolling feature for gameweek N must not incorporate any value from gameweek N
    itself (would be leakage). Verify by checking GW1 features are all NaN even though GW1
    has realized event_points in the DB."""
    df = build_dataset(seeded_db, with_features=True)
    gw1 = df[df["gameweek"] == 1]
    # rolling_points_5 for GW1 must be NaN -- the GW1 event_points must NOT have leaked in
    assert gw1["rolling_points_5"].isna().all()
    # also verify the DB actually has realized GW1 points (so NaN is due to asof, not missing data)
    assert seeded_db.execute(
        "SELECT count(*) FROM fact_player_season_stats WHERE gw = 1 AND event_points IS NOT NULL"
    ).fetchone()[0] > 0


def test_rolling_defcon_is_in_feature_columns_and_asof_safe(seeded_db):
    """rolling_defcon_{3,5,10}: mean per-match defensive actions over prior matches. Sits beside
    rolling_goals_{w} in _rolling_match_features and follows the same asof discipline -- prior
    match rows only (cross-season history is valid, exactly like rolling_goals), and the very
    first match ever (2024-2025 GW1) has none -> NaN."""
    assert {"rolling_defcon_3", "rolling_defcon_5", "rolling_defcon_10"} <= set(feature_columns())
    df = build_dataset(seeded_db, with_features=True)
    first_ever = df[(df["season"] == "2024-2025") & (df["gameweek"] == 1)]
    assert len(first_ever) > 0
    assert first_ever["rolling_defcon_3"].isna().all()
    gw3 = df[df["gameweek"] == 3]
    populated = gw3[gw3["rolling_defcon_3"].notna()]
    assert len(populated) > 0
    # fixture seeds tackles+clearances+interceptions+recoveries+blocks = 1+2+1+3+1 = 8 per match
    assert (populated["rolling_defcon_3"] - 8.0).abs().max() < 1e-9
