"""Season simulation: act as an FPL manager using a prediction signal, accumulate real points.

This is a RESEARCH-ONLY proxy (spec: "do not integrate ML into the live FPL recommendation
engine yet"). It does not call the production scip optimizer. Instead it greedily selects a
starting XI from each gameweek's prediction rows using a fixed prediction signal, under
simplified FPL constraints (position balance, 3-per-club, captain doubles actual), and sums the
*actual* event_points of the selected XI across the season.

The point is a comparable "how many points would a manager using signal X have scored" number
per season -- so we can ask whether the ML-augmented signal beats the Quant signal at the thing
that ultimately matters: fantasy points, not just prediction error.
"""

from __future__ import annotations

import pandas as pd

from . import contract as C

# FPL starting-XI formation constraints (exactly 1 GK; DEF+MID+FWD = 10). Keyed by the real
# position strings this repo uses everywhere else (contract.POSITIONS, dim_player.position,
# dataset_builder's attached `position` column) -- NOT FPL's short codes ("GK"/"DEF"/...), which
# never appear in this data and would silently make every position check below a no-op (see the
# bug this comment replaces: the greedy loop's `if pos not in _POS_MIN: continue` matched every
# real row, so selected stayed empty and every gameweek fell through to the position-blind
# "backfill" path -- picking whichever 11 players had the highest predicted value with no
# regard for a legal FPL formation, for the entire history of this module).
_POS_MAX = {"Defender": 5, "Midfielder": 5, "Forward": 3}
_POS_MIN = {"Goalkeeper": 1, "Defender": 3, "Midfielder": 2, "Forward": 1}
_CLUB_CAP = 3
_XI_SIZE = 11


def _feasible_after_add(counts: dict[str, int], adding_pos: str, current_size: int) -> bool:
    """After adding one player of `adding_pos`, can the remaining slots still meet every
    position's minimum?"""
    new_counts = dict(counts)
    new_counts[adding_pos] = new_counts.get(adding_pos, 0) + 1
    remaining_slots = _XI_SIZE - (current_size + 1)
    unmet = sum(max(0, _POS_MIN[p] - new_counts.get(p, 0)) for p in _POS_MIN)
    return remaining_slots >= unmet


def select_starting_xi(gw_df: pd.DataFrame, pred_col: str) -> tuple[list[str], str | None]:
    """Greedily pick an 11-man starting XI maximising `pred_col`, respecting position balance
    (1 GK; DEF 3-5, MID 2-5, FWD 1-3), 3-per-club. Returns (player_uids, captain_uid)."""
    pool = gw_df.dropna(subset=[pred_col, C.COL_ACTUAL]).copy()
    if pool.empty:
        return [], None
    pool = pool.sort_values(pred_col, ascending=False)
    selected: list[str] = []
    club_count: dict[str, int] = {}
    pos_count: dict[str, int] = {}
    for _, p in pool.iterrows():
        if len(selected) >= _XI_SIZE:
            break
        pos = str(p[C.COL_POSITION])
        if pos not in _POS_MIN:
            continue
        if pos == "Goalkeeper":
            if pos_count.get("Goalkeeper", 0) >= 1:
                continue
        else:
            if pos_count.get(pos, 0) >= _POS_MAX[pos]:
                continue
        club = str(p[C.COL_TEAM_UID])
        if club_count.get(club, 0) >= _CLUB_CAP:
            continue
        if not _feasible_after_add(pos_count, pos, len(selected)):
            continue
        uid = str(p[C.COL_PLAYER_UID])
        selected.append(uid)
        pos_count[pos] = pos_count.get(pos, 0) + 1
        club_count[club] = club_count.get(club, 0) + 1
    # backfill any unfilled slots (data too small) with best remaining, ignoring max caps
    if len(selected) < _XI_SIZE:
        chosen = set(selected)
        for _, p in pool.iterrows():
            if len(selected) >= _XI_SIZE:
                break
            uid = str(p[C.COL_PLAYER_UID])
            if uid in chosen:
                continue
            club = str(p[C.COL_TEAM_UID])
            if club_count.get(club, 0) >= _CLUB_CAP:
                continue
            selected.append(uid)
            chosen.add(uid)
            club_count[club] = club_count.get(club, 0) + 1
    captain = None
    if selected:
        rows = gw_df[gw_df[C.COL_PLAYER_UID].isin(selected)]
        if not rows.empty:
            captain = str(rows.loc[rows[pred_col].idxmax(), C.COL_PLAYER_UID])
    return selected, captain


def simulate_gameweek(gw_df: pd.DataFrame, pred_col: str) -> float:
    """Pick a starting XI using `pred_col`, return the actual points scored (captain doubles)."""
    xi, captain = select_starting_xi(gw_df, pred_col)
    if not xi:
        return 0.0
    xi_df = gw_df[gw_df[C.COL_PLAYER_UID].isin(xi)]
    base = float(xi_df[C.COL_ACTUAL].sum())
    if captain is not None:
        cap_row = xi_df[xi_df[C.COL_PLAYER_UID] == captain]
        if not cap_row.empty:
            base += float(cap_row[C.COL_ACTUAL].iloc[0])
    return base


def simulate_season(season_df: pd.DataFrame, pred_col: str) -> dict:
    """Accumulate actual points across every gameweek in `season_df`, picking from `pred_col`
    each gameweek. Returns total points + per-gameweek breakdown."""
    per_gw: list[dict] = []
    total = 0.0
    for gw, gw_df in season_df.sort_values(C.COL_GAMEWEEK).groupby(C.COL_GAMEWEEK):
        pts = simulate_gameweek(gw_df, pred_col)
        total += pts
        per_gw.append({"gameweek": int(gw), "points": pts})
    return {"total_points": total, "per_gameweek": per_gw}


def season_points_table(df: pd.DataFrame, signal_cols: dict[str, str]) -> pd.DataFrame:
    """For each (signal_name -> prediction column) and each season, simulate a manager picking
    from that signal and return total season points. One row per (season, signal)."""
    rows: list[dict] = []
    for season, season_df in df.groupby(C.COL_SEASON):
        for signal_name, pred_col in signal_cols.items():
            sim = simulate_season(season_df, pred_col)
            rows.append({
                "season": season, "signal": signal_name,
                "total_points": sim["total_points"],
                "n_gameweeks": len(sim["per_gameweek"]),
            })
    return pd.DataFrame(rows)
