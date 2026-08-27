"""Season manager simulation tests: pick a starting XI from a prediction signal, accumulate
actual points (captain doubles), and compare signals by total season points."""

from __future__ import annotations

import pandas as pd

from research.ml import contract as C
from research.ml.season_sim import (
    season_points_table,
    select_starting_xi,
    simulate_gameweek,
    simulate_season,
)


def _gw_df(preds: dict[str, float], actuals: dict[str, float]) -> pd.DataFrame:
    """One gameweek of players: preds/actuals keyed by player_uid. Mixed positions and teams."""
    rows = []
    teams = ["t1", "t2", "t3", "t4", "t5", "t6"]
    pos_cycle = ["GK", "DEF", "MID", "FWD"]
    for i, uid in enumerate(preds):
        rows.append({
            C.COL_PLAYER_UID: uid,
            C.COL_SEASON: "2024-2025",
            C.COL_GAMEWEEK: 1,
            C.COL_TEAM_UID: teams[i % len(teams)],
            C.COL_POSITION: pos_cycle[i % len(pos_cycle)],
            "now_cost": 50,
            "predicted": preds[uid],
            C.COL_ACTUAL: actuals[uid],
        })
    return pd.DataFrame(rows)


def test_select_starting_xi_picks_eleven_balanced():
    preds = {f"p{i}": float(i) for i in range(30)}
    actuals = {f"p{i}": 0.0 for i in range(30)}
    df = _gw_df(preds, actuals)
    xi, captain = select_starting_xi(df, "predicted")
    assert len(xi) == 11
    assert captain is not None
    # captain is the highest-predicted selected player
    rows = df[df[C.COL_PLAYER_UID].isin(xi)]
    assert captain == str(rows.loc[rows["predicted"].idxmax(), C.COL_PLAYER_UID])


def test_simulate_gameweek_captain_doubles():
    preds = {f"p{i}": float(i) for i in range(30)}
    # p29 is highest predicted -> captain; give it actual 10
    actuals = {f"p{i}": 1.0 for i in range(30)}
    actuals["p29"] = 10.0
    df = _gw_df(preds, actuals)
    xi, captain = select_starting_xi(df, "predicted")
    pts = simulate_gameweek(df, "predicted")
    # base = sum of XI actuals (10 starters at 1.0 + captain at 10.0) = 10*1 + 10 ... wait
    base_actual = float(df[df[C.COL_PLAYER_UID].isin(xi)][C.COL_ACTUAL].sum())
    assert pts == base_actual + 10.0  # captain doubles: base includes captain once, +captain again


def test_simulate_season_accumulates_across_gameweeks():
    rows = []
    for gw in (1, 2, 3):
        preds = {f"p{i}_{gw}": float(i) for i in range(30)}
        actuals = {f"p{i}_{gw}": 2.0 for i in range(30)}
        df = _gw_df(preds, actuals)
        df[C.COL_GAMEWEEK] = gw
        rows.append(df)
    season_df = pd.concat(rows, ignore_index=True)
    sim = simulate_season(season_df, "predicted")
    assert sim["total_points"] > 0
    assert len(sim["per_gameweek"]) == 3


def test_season_points_table_compares_signals():
    rows = []
    for gw in (1, 2):
        preds = {f"p{i}_{gw}": float(i) for i in range(30)}
        actuals = {f"p{i}_{gw}": 2.0 for i in range(30)}
        df = _gw_df(preds, actuals)
        df[C.COL_GAMEWEEK] = gw
        rows.append(df)
    season_df = pd.concat(rows, ignore_index=True)
    # a deliberately better signal: predicted == actual for the top players
    season_df["perfect"] = season_df[C.COL_ACTUAL]
    table = season_points_table(season_df, {"predicted": "predicted", "perfect": "perfect"})
    assert set(table["signal"]) == {"predicted", "perfect"}
    perfect_pts = table.loc[table["signal"] == "perfect", "total_points"].iloc[0]
    pred_pts = table.loc[table["signal"] == "predicted", "total_points"].iloc[0]
    assert perfect_pts >= pred_pts
