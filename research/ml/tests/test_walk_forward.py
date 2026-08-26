"""Walk-forward validation tests: expanding-window season folds, chronological ordering,
no train/test overlap, and supplementary gameweek folds."""

from __future__ import annotations


from research.ml.dataset_builder import build_dataset
from research.ml.walk_forward import (
    default_folds,
    exhaustive_gameweek_folds,
    gameweek_folds,
    season_folds,
)


def test_season_folds_produce_expanding_window(seeded_db):
    df = build_dataset(seeded_db, with_features=True)
    folds = season_folds(df)
    assert len(folds) == 1  # 2 seasons -> 1 expanding fold (S1 train, S2 test)
    f = folds[0]
    assert list(f.train_seasons) == ["2024-2025"]
    assert f.test_season == "2025-2026"
    assert set(f.train_seasons) & {f.test_season} == set()  # no overlap


def test_season_folds_train_precedes_test_chronologically(seeded_db):
    df = build_dataset(seeded_db, with_features=True)
    for f in season_folds(df):
        for ts in f.train_seasons:
            assert ts < f.test_season  # string comparison works for ISO season keys


def test_default_folds_run_split_invariants(seeded_db):
    # default_folds calls assert_split_invariants internally -> exercises leakage check
    df = build_dataset(seeded_db, with_features=True)
    folds = default_folds(df)
    assert len(folds) >= 1
    for f in folds:
        assert len(f.train_df) > 0
        assert len(f.test_df) > 0
        # no row appears in both train and test
        shared = f.train_df.merge(f.test_df, on=["player_uid", "season", "gameweek"], how="inner")
        assert len(shared) == 0


def test_gameweek_folds_respect_step(seeded_db):
    df = build_dataset(seeded_db, with_features=True)
    folds = gameweek_folds(df, step=2)
    # with 2 seasons x 3 gameweeks = 6 GWs and step=2, we expect at least one fold
    assert len(folds) >= 1
    for f in folds:
        assert len(f.train_df) > 0
        assert len(f.test_df) > 0


def test_gameweek_folds_train_before_test(seeded_db):
    df = build_dataset(seeded_db, with_features=True)
    for f in gameweek_folds(df, step=2):
        train_max_gw = f.train_df.groupby("season")["gameweek"].max()
        test_min_gw = f.test_df.groupby("season")["gameweek"].min()
        for season in test_min_gw.index:
            if season in train_max_gw.index:
                assert train_max_gw[season] < test_min_gw[season]


def test_default_folds_is_exhaustive_gameweek(seeded_db):
    # the default fold mode tests one gameweek at a time -- the maximum number of simulations
    df = build_dataset(seeded_db, with_features=True)
    folds = default_folds(df, fold_mode="gameweek")
    # 2 seasons x 3 gameweeks = 6 steps; the first has no prior training -> 5 folds
    assert len(folds) == 5
    for f in folds:
        # each fold tests exactly one (season, gameweek)
        assert f.test_gameweek is not None
        assert f.test_df["gameweek"].nunique() == 1
        assert f.test_df["season"].nunique() == 1
        # training set is strictly earlier in chronological key space
        train_keys = set(f.train_df[["season", "gameweek"]].drop_duplicates().itertuples(index=False, name=None))
        test_keys = set(f.test_df[["season", "gameweek"]].drop_duplicates().itertuples(index=False, name=None))
        assert train_keys & test_keys == set()  # no shared gameweek


def test_exhaustive_folds_more_than_season_folds(seeded_db):
    df = build_dataset(seeded_db, with_features=True)
    assert len(exhaustive_gameweek_folds(df)) > len(season_folds(df))


def test_default_folds_season_mode(seeded_db):
    df = build_dataset(seeded_db, with_features=True)
    folds = default_folds(df, fold_mode="season")
    assert len(folds) == 1  # one expanding season fold

