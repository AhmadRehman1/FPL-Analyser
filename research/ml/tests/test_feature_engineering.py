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
    forbidden = {"minutes", "goals_scored", "assists", "bps", "total_points", "saves",
                 "expected_goals", "expected_assists"}
    assert not (forbidden & set(cols))
    assert C.COL_ACTUAL not in cols
    assert C.COL_RESIDUAL not in cols
    # the recent-xG/xA delta features are present and named so they never collide with the
    # forbidden raw same-gameweek outcome columns
    assert {"xg_last_3", "xg_last_5", "xa_last_3", "xa_last_5"} <= set(cols)


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


def test_recent_xg_xa_deltas_are_nan_before_two_prior_snapshots(seeded_db):
    """xg_last_{3,5} / xa_last_{3,5} need >=2 prior gw < G snapshots to difference. GW1 has no
    prior snapshot, GW2 has exactly one -> both must be NaN, even though the DB carries a
    cumulative expected_goals for every gameweek (so NaN is asof discipline, not missing data)."""
    df = build_dataset(seeded_db, with_features=True)
    for col in ("xg_last_3", "xg_last_5", "xa_last_3", "xa_last_5"):
        assert df.loc[df["gameweek"] == 1, col].isna().all()
        assert df.loc[df["gameweek"] == 2, col].isna().all()
    assert seeded_db.execute(
        "SELECT count(*) FROM fact_player_season_stats WHERE gw IN (1, 2) AND expected_goals IS NOT NULL"
    ).fetchone()[0] > 0


def test_recent_xg_xa_deltas_recover_the_per_gameweek_value(seeded_db):
    """By GW3 a player with GW1+GW2 snapshots has one computable delta. The fixture accrues a
    flat 0.2 xG / 0.1 xA per gameweek cumulatively, so the recovered per-gameweek delta is
    exactly that -- and never the target GW's own xG (which would make it 0.2*3 - 0.2*2 anyway,
    but the point is only gw < G rows are touched)."""
    df = build_dataset(seeded_db, with_features=True)
    gw3 = df[df["gameweek"] == 3]
    populated = gw3[gw3["xg_last_3"].notna()]
    assert len(populated) > 0
    assert (populated["xg_last_3"] - 0.2).abs().max() < 1e-9
    assert (populated["xa_last_3"] - 0.1).abs().max() < 1e-9
