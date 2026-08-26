"""Walk-forward validation for the Phase-0 residual ML experiment.

FPL is a time-series. Random train/test splits (sklearn train_test_split) are forbidden
(spec §6) -- they let future information bleed into training. This module produces CHRONOLOGICAL
splits: train on every row strictly before the test step, evaluate on the held-out test step.

Splits are derived from the dataset's own (season, gameweek, prediction_timestamp) columns --
never from a re-shuffle. Each fold is leakage-checked (check_chronological_split) before it is
returned.

The default fold mode is **exhaustive gameweek walk-forward**: test on exactly one (season,
gameweek) at a time, re-training on all prior rows. This maximises the number of historical
simulations -- every gameweek with enough prior training data becomes an out-of-sample test
point. Season-level folds remain available for a coarser summary view.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import contract as C
from .leakage_checks import assert_split_invariants


@dataclass(frozen=True)
class WalkForwardFold:
    name: str
    train_seasons: tuple[str, ...]
    test_season: str
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    test_gameweek: int | None = None


def _available_seasons(df: pd.DataFrame) -> list[str]:
    return sorted(df[C.COL_SEASON].unique().tolist(), key=C.season_sort_key)


def _season_gameweek_keys(df: pd.DataFrame) -> list[tuple[str, int]]:
    """Chronologically ordered (season, gameweek) steps present in the dataset."""
    keys = (
        df[[C.COL_SEASON, C.COL_GAMEWEEK]]
        .drop_duplicates()
        .sort_values([C.COL_SEASON, C.COL_GAMEWEEK])
        .apply(lambda r: (r[C.COL_SEASON], int(r[C.COL_GAMEWEEK])), axis=1)
        .tolist()
    )
    keys.sort(key=lambda k: (C.season_sort_key(k[0]), k[1]))
    return keys


def season_folds(df: pd.DataFrame) -> list[WalkForwardFold]:
    """Expanding-window walk-forward folds at season granularity: for each test season (except
    the earliest, which has nothing to train on), train on all prior seasons and test on it.
    Mirrors the spec's Experiment 1-5 table (train 2017/18->X, test season X+1)."""
    seasons = _available_seasons(df)
    folds: list[WalkForwardFold] = []
    for i in range(1, len(seasons)):
        train_seasons = tuple(seasons[:i])
        test_season = seasons[i]
        train_df = df[df[C.COL_SEASON].isin(train_seasons)].copy()
        test_df = df[df[C.COL_SEASON] == test_season].copy()
        assert_split_invariants(train_df, test_df)
        folds.append(WalkForwardFold(
            name=f"train_{'_'.join(train_seasons)}_test_{test_season}",
            train_seasons=train_seasons, test_season=test_season,
            train_df=train_df, test_df=test_df,
        ))
    return folds


def gameweek_folds(df: pd.DataFrame, *, step: int = 4) -> list[WalkForwardFold]:
    """A coarser within-season expanding window: train on all gameweeks strictly before a
    cutoff, test on the next `step` gameweeks. Requires at least two seasons so early-season
    folds still have training data from prior seasons."""
    seasons = _available_seasons(df)
    if len(seasons) < 2:
        return []
    folds: list[WalkForwardFold] = []
    for season in seasons[1:]:  # earliest season has no prior training data
        season_df = df[df[C.COL_SEASON] == season].sort_values(C.COL_GAMEWEEK)
        gws = sorted(season_df[C.COL_GAMEWEEK].unique().tolist())
        prior = df[df[C.COL_SEASON].isin([s for s in seasons if C.season_sort_key(s) < C.season_sort_key(season)])]
        for start in range(0, len(gws) - 1, step):
            test_gws = gws[start:start + step]
            if not test_gws:
                break
            train_df = pd.concat([prior, season_df[season_df[C.COL_GAMEWEEK] < test_gws[0]]])
            test_df = season_df[season_df[C.COL_GAMEWEEK].isin(test_gws)]
            if train_df.empty or test_df.empty:
                continue
            assert_split_invariants(train_df, test_df)
            folds.append(WalkForwardFold(
                name=f"gw_{season}_{test_gws[0]}-{test_gws[-1]}",
                train_seasons=tuple(sorted(train_df[C.COL_SEASON].unique().tolist(), key=C.season_sort_key)),
                test_season=season,
                train_df=train_df, test_df=test_df,
                test_gameweek=test_gws[0],
            ))
    return folds


def exhaustive_gameweek_folds(df: pd.DataFrame, *, min_train_rows: int = 1) -> list[WalkForwardFold]:
    """Exhaustive expanding gameweek walk-forward: for every (season, gameweek) step that has
    at least `min_train_rows` of strictly-prior training data, train on ALL prior rows and test
    on exactly that one gameweek. This is the maximum-simulation mode -- every historical
    gameweek becomes an out-of-sample test point."""
    keys = _season_gameweek_keys(df)
    folds: list[WalkForwardFold] = []
    for i in range(1, len(keys)):
        test_season, test_gw = keys[i]
        train_mask = pd.Series(False, index=df.index)
        for season, gw in keys[:i]:
            train_mask |= ((df[C.COL_SEASON] == season) & (df[C.COL_GAMEWEEK] == gw))
        train_df = df[train_mask].copy()
        test_df = df[(df[C.COL_SEASON] == test_season) & (df[C.COL_GAMEWEEK] == test_gw)].copy()
        if len(train_df) < min_train_rows or test_df.empty:
            continue
        assert_split_invariants(train_df, test_df)
        folds.append(WalkForwardFold(
            name=f"gw_{test_season}_{test_gw}",
            train_seasons=tuple(sorted(train_df[C.COL_SEASON].unique().tolist(), key=C.season_sort_key)),
            test_season=test_season,
            train_df=train_df, test_df=test_df,
            test_gameweek=int(test_gw),
        ))
    return folds


def default_folds(df: pd.DataFrame, *, fold_mode: str = "gameweek") -> list[WalkForwardFold]:
    """The folds the experiment actually evaluates.

    fold_mode="gameweek" (default): exhaustive one-gameweek-at-a-time walk-forward -- the
    maximum number of historical simulations. fold_mode="season": one fold per season.
    """
    if fold_mode == "season":
        folds = season_folds(df)
    elif fold_mode == "gameweek":
        folds = exhaustive_gameweek_folds(df)
    else:
        raise ValueError(f"unknown fold_mode {fold_mode!r} (expected 'gameweek' or 'season')")
    if not folds:
        raise RuntimeError(
            "no walk-forward folds could be built -- the dataset needs at least two seasons "
            "with backtest steps (the earliest season trains nothing)."
        )
    return folds
