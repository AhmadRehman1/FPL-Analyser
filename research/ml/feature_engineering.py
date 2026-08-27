"""Asof-safe rolling feature engineering for the Phase-0 residual ML dataset.

Adds the approved pre-prediction feature columns (LEAKAGE_PROTOCOL.md §4) to the minimal
player x gameweek dataset. Features are computed PER WALK-FORWARD STEP: the dataset is
grouped by (season, gameweek), and for each group the builder re-enters
fpl_quant.backtest.asof_scope() so the three fact tables are shadowed to strictly-pre-deadline
rows. Every query below therefore reads only knowable-asof data -- the asof boundary is
enforced at the data layer, not by each feature being careful.

Design choice (per the audit's §9 / leakage protocol §5): features are recomputed directly
from the reconciled fact tables with explicit gw/season < target filters, NOT by calling
Quant internals like _player_rate_pool(). Divergence from the Quant model's own feature
values is acceptable here -- these are residual EXPLANATORY features, not a replica of the
Quant model's inputs, and recomputation keeps the dataset independently leakage-checkable.
"""

from __future__ import annotations

import math
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd

from . import contract as C

from fpl_quant import backtest as bt


# ============================================================
# Asof-safe bulk fetchers (run INSIDE asof_scope -- tables are shadowed)
# ============================================================

def _prior_match_stats(con: duckdb.DuckDBPyConnection, player_uids: list[str]) -> pd.DataFrame:
    """All prior-match rows for the step's players, newest first. fact_player_match_stats is
    shadowed to kickoff_time < deadline by asof_scope, so only knowable matches return."""
    if not player_uids:
        return pd.DataFrame()
    placeholders = ", ".join(["?"] * len(player_uids))
    return con.execute(
        f"""
        SELECT s.player_uid, s.match_id, m.kickoff_time, s.start_min, s.minutes_played,
               s.goals, s.assists
        FROM fact_player_match_stats s JOIN fact_match m ON m.match_id = s.match_id
        WHERE s.player_uid IN ({placeholders})
        ORDER BY s.player_uid, m.kickoff_time DESC
        """,
        player_uids,
    ).fetchdf()


def _prior_season_snapshots(con: duckdb.DuckDBPyConnection, season: str, gameweek: int, player_uids: list[str]) -> pd.DataFrame:
    """Per-gameweek season snapshots strictly before the target gw, newest first. The shadow
    truncates the in-progress season to gw < target, so event_points here is the realised
    points for PRIOR gameweeks only -- a valid asof feature source."""
    if not player_uids:
        return pd.DataFrame()
    placeholders = ", ".join(["?"] * len(player_uids))
    return con.execute(
        f"""
        SELECT player_uid, gw, now_cost, selected_by_percent, chance_of_playing_next_round,
               status, expected_goals_per_90, expected_assists_per_90,
               defensive_contribution_per_90, saves_per_90, bps, event_points
        FROM fact_player_season_stats
        WHERE season = ? AND gw < ? AND player_uid IN ({placeholders})
        ORDER BY player_uid, gw DESC
        """,
        [season, gameweek, *player_uids],
    ).fetchdf()


def _m2_minutes_probs(con: duckdb.DuckDBPyConnection, mm_model_version: int | None, player_uids: list[str]) -> pd.DataFrame:
    """M2 minutes probabilities for the target gameweek. These are a Quant PREDICTION made asof
    the deadline -- permitted as a feature (an input the residual model can correct), distinct
    from the realised outcome."""
    if mm_model_version is None or not player_uids:
        return pd.DataFrame(columns=["player_uid", "p_start_final", "p_60plus_min"])
    placeholders = ", ".join(["?"] * len(player_uids))
    return con.execute(
        f"""
        SELECT player_uid, p_start_final, p_60plus_min
        FROM minutes_model_outputs
        WHERE model_version = ? AND player_uid IN ({placeholders})
        """,
        [mm_model_version, *player_uids],
    ).fetchdf()


def _team_prior_matches(con: duckdb.DuckDBPyConnection, team_uids: list[str], deadline) -> pd.DataFrame:
    """Prior finished matches for the given teams, newest first. fact_match is shadowed so
    scores are only visible where kickoff_time < deadline (finished prior matches)."""
    if not team_uids:
        return pd.DataFrame()
    placeholders = ", ".join(["?"] * len(team_uids))
    return con.execute(
        f"""
        SELECT match_id, home_team_uid, away_team_uid, kickoff_time, home_score, away_score
        FROM fact_match
        WHERE competition = ? AND finished = TRUE AND home_score IS NOT NULL AND away_score IS NOT NULL
          AND kickoff_time < ?
          AND (home_team_uid IN ({placeholders}) OR away_team_uid IN ({placeholders}))
        ORDER BY kickoff_time DESC
        """,
        [C.PL, deadline, *team_uids, *team_uids],
    ).fetchdf()


# ============================================================
# Rolling computations in pandas (vectorised, no per-row SQL)
# ============================================================

def _rolling_match_features(stats: pd.DataFrame, windows: Iterable[int]) -> pd.DataFrame:
    """Per-player rolling means of goals/assists/minutes and start counts over N prior matches.
    Computed by taking the first N rows per player (already ordered newest-first) and averaging
    -- a true walk-back rolling window, never including the target match (which is absent)."""
    if stats.empty:
        return pd.DataFrame(columns=["player_uid"])
    stats = stats.copy()
    stats["started"] = (stats["start_min"] == 0).astype(float)
    out_rows = []
    grouped = stats.groupby("player_uid", sort=False)
    for player_uid, g in grouped:
        g = g.reset_index(drop=True)
        rec: dict = {"player_uid": player_uid}
        for w in windows:
            head = g.head(w)
            rec[f"rolling_goals_{w}"] = head["goals"].mean() if not head.empty else 0.0
            rec[f"rolling_assists_{w}"] = head["assists"].mean() if not head.empty else 0.0
            rec[f"rolling_minutes_{w}"] = head["minutes_played"].mean() if not head.empty else 0.0
            rec[f"rolling_starts_{w}"] = head["started"].mean() if not head.empty else 0.0
        # discrete windows the spec calls out explicitly
        rec["starts_last_3"] = int(g.head(3)["started"].sum()) if not g.empty else 0
        rec["starts_last_5"] = int(g.head(5)["started"].sum()) if not g.empty else 0
        rec["minutes_last_5"] = float(g.head(5)["minutes_played"].sum()) if not g.empty else 0.0
        out_rows.append(rec)
    return pd.DataFrame(out_rows)


def _latest_snapshot_features(snaps: pd.DataFrame, windows: Iterable[int]) -> pd.DataFrame:
    """Per-player most-recent asof snapshot fields (price/ownership/status/per-90 rates) plus a
    rolling mean of prior event_points (the only valid source of a 'rolling_points' feature --
    per-gameweek FPL points, not the cumulative total_points)."""
    if snaps.empty:
        return pd.DataFrame(columns=["player_uid"])
    snaps = snaps.copy()
    out_rows = []
    for player_uid, g in snaps.groupby("player_uid", sort=False):
        g = g.reset_index(drop=True)  # newest-first from the query
        latest = g.iloc[0] if not g.empty else None
        rec: dict = {"player_uid": player_uid}
        if latest is not None:
            rec["now_cost"] = float(latest["now_cost"]) if pd.notna(latest["now_cost"]) else np.nan
            rec["selected_by_percent"] = float(latest["selected_by_percent"]) if pd.notna(latest["selected_by_percent"]) else np.nan
            rec["chance_of_playing_next_round"] = float(latest["chance_of_playing_next_round"]) if pd.notna(latest["chance_of_playing_next_round"]) else np.nan
            rec["status"] = latest["status"] if pd.notna(latest["status"]) else "unknown"
            rec["rolling_xg_per90"] = float(latest["expected_goals_per_90"]) if pd.notna(latest["expected_goals_per_90"]) else np.nan
            rec["rolling_xa_per90"] = float(latest["expected_assists_per_90"]) if pd.notna(latest["expected_assists_per_90"]) else np.nan
            rec["rolling_defcon_per90"] = float(latest["defensive_contribution_per_90"]) if pd.notna(latest["defensive_contribution_per_90"]) else np.nan
            rec["rolling_saves_per90"] = float(latest["saves_per_90"]) if pd.notna(latest["saves_per_90"]) else np.nan
            rec["rolling_bps"] = float(latest["bps"]) if pd.notna(latest["bps"]) else np.nan
        for w in windows:
            head = g.head(w)
            rec[f"rolling_points_{w}"] = head["event_points"].mean() if not head.empty else 0.0
        out_rows.append(rec)
    return pd.DataFrame(out_rows)


def _team_features(team_matches: pd.DataFrame, team_uid: str | None, deadline) -> dict:
    """Goals for/against over a team's last 10 prior finished matches + congestion counts.
    Returns NaNs if the team has no prior history (a genuinely new/promoted side)."""
    if team_uid is None or team_matches.empty:
        return {
            "team_goals_for_last_10": np.nan, "team_goals_against_last_10": np.nan,
            "matches_last_7_days": 0, "matches_last_14_days": 0,
        }
    tm = team_matches[
        (team_matches["home_team_uid"] == team_uid) | (team_matches["away_team_uid"] == team_uid)
    ].head(10)
    if tm.empty:
        return {
            "team_goals_for_last_10": np.nan, "team_goals_against_last_10": np.nan,
            "matches_last_7_days": 0, "matches_last_14_days": 0,
        }
    gf, ga = [], []
    for _, r in tm.iterrows():
        if r["home_team_uid"] == team_uid:
            gf.append(r["home_score"])
            ga.append(r["away_score"])
        else:
            gf.append(r["away_score"])
            ga.append(r["home_score"])
    deadline_ts = pd.Timestamp(deadline)
    congestion = team_matches[
        (team_matches["home_team_uid"] == team_uid) | (team_matches["away_team_uid"] == team_uid)
    ]
    last_7 = (deadline_ts - congestion["kickoff_time"]).dt.total_seconds() <= 7 * 86400
    last_14 = (deadline_ts - congestion["kickoff_time"]).dt.total_seconds() <= 14 * 86400
    return {
        "team_goals_for_last_10": float(np.mean(gf)),
        "team_goals_against_last_10": float(np.mean(ga)),
        "matches_last_7_days": int(last_7.sum()),
        "matches_last_14_days": int(last_14.sum()),
    }


def _opponent_features(team_matches: pd.DataFrame, opp_uid: str | None) -> dict:
    """Opponent attacking/defensive strength from their last 10 prior matches (goals scored /
    conceded). Fixture difficulty is the opponent's goals-against (defensive weakness): a leakier
    opponent => an easier fixture for the player's team to score in."""
    if opp_uid is None or team_matches.empty:
        return {"opponent_goals_for_last_10": np.nan, "opponent_goals_against_last_10": np.nan, "fixture_difficulty": np.nan}
    tm = team_matches[
        (team_matches["home_team_uid"] == opp_uid) | (team_matches["away_team_uid"] == opp_uid)
    ].head(10)
    if tm.empty:
        return {"opponent_goals_for_last_10": np.nan, "opponent_goals_against_last_10": np.nan, "fixture_difficulty": np.nan}
    gf, ga = [], []
    for _, r in tm.iterrows():
        if r["home_team_uid"] == opp_uid:
            gf.append(r["home_score"])
            ga.append(r["away_score"])
        else:
            gf.append(r["away_score"])
            ga.append(r["home_score"])
    opp_ga = float(np.mean(ga))
    # higher opponent goals-against => easier fixture => lower difficulty score
    difficulty = -opp_ga if not math.isnan(opp_ga) else np.nan
    return {
        "opponent_goals_for_last_10": float(np.mean(gf)),
        "opponent_goals_against_last_10": opp_ga,
        "fixture_difficulty": difficulty,
    }


# ============================================================
# Per-step assembly
# ============================================================

def _compute_step_features(
    con: duckdb.DuckDBPyConnection, season: str, gameweek: int, grp: pd.DataFrame, mm_mv
) -> pd.DataFrame:
    player_uids = grp["player_uid"].unique().tolist()
    deadline = bt.gameweek_deadline(con, season, gameweek)

    stats = _prior_match_stats(con, player_uids)
    snaps = _prior_season_snapshots(con, season, gameweek, player_uids)
    m2 = _m2_minutes_probs(con, mm_mv, player_uids)

    roll_match = _rolling_match_features(stats, C.ROLLING_WINDOWS)
    snap_feat = _latest_snapshot_features(snaps, C.ROLLING_WINDOWS)

    team_uids = [t for t in grp[C.COL_TEAM_UID].unique().tolist() if pd.notna(t)]
    opp_uids = [t for t in grp[C.COL_OPPONENT_UID].unique().tolist() if pd.notna(t)]
    team_matches = _team_prior_matches(con, team_uids + opp_uids, deadline)

    team_feat_rows = []
    for team_uid, opp_uid in zip(grp[C.COL_TEAM_UID], grp[C.COL_OPPONENT_UID]):
        tf = _team_features(team_matches, team_uid if pd.notna(team_uid) else None, deadline)
        of = _opponent_features(team_matches, opp_uid if pd.notna(opp_uid) else None)
        team_feat_rows.append({**tf, **of})

    team_feat_df = pd.DataFrame(team_feat_rows).reset_index(drop=True)

    feats = grp[["player_uid"]].reset_index(drop=True)
    feats = feats.merge(roll_match, on="player_uid", how="left")
    feats = feats.merge(snap_feat, on="player_uid", how="left")
    feats = feats.merge(m2, on="player_uid", how="left")
    feats["is_home"] = (grp[C.COL_HOME_AWAY].reset_index(drop=True) == "home").astype(float)
    # Attach the per-row team/opponent features (fixture_difficulty, goals for/against, congestion).
    # Built positionally against grp's row order -- reset_index keeps the alignment 1:1 with feats.
    feats = pd.concat([feats, team_feat_df], axis=1)
    # fill discrete count NaNs that arise from no-history players
    for col in ("starts_last_3", "starts_last_5", "matches_last_7_days", "matches_last_14_days"):
        if col in feats.columns:
            feats[col] = feats[col].fillna(0)
    return feats


# ============================================================
# Public entrypoint
# ============================================================

def feature_columns() -> list[str]:
    """The ordered, documented list of feature columns this module produces. Used by
    leakage_checks.assert_feature_matrix_invariants and by the model layer to know X."""
    cols: list[str] = ["is_home"]
    for w in C.ROLLING_WINDOWS:
        cols += [f"rolling_points_{w}", f"rolling_goals_{w}", f"rolling_assists_{w}",
                 f"rolling_minutes_{w}", f"rolling_starts_{w}"]
    cols += [
        "starts_last_3", "starts_last_5", "minutes_last_5",
        "now_cost", "selected_by_percent", "chance_of_playing_next_round", "status",
        "rolling_xg_per90", "rolling_xa_per90", "rolling_defcon_per90", "rolling_saves_per90",
        "rolling_bps", "p_start_final", "p_60plus_min",
        "team_goals_for_last_10", "team_goals_against_last_10",
        "opponent_goals_for_last_10", "opponent_goals_against_last_10",
        "fixture_difficulty", "matches_last_7_days", "matches_last_14_days",
    ]
    return cols


def add_features(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> pd.DataFrame:
    """Add asof-safe feature columns to the minimal dataset, per walk-forward step.

    Groups by (season, gameweek) and re-enters asof_scope for each group so feature queries see
    only pre-deadline rows. The merge back to the minimal dataset is on (player_uid, season,
    gameweek); players with no prior history get NaN features (handled explicitly by the model
    layer, never silently dropped).
    """
    if df.empty:
        return df
    parts: list[pd.DataFrame] = []
    for (season, gw), grp in df.groupby([C.COL_SEASON, C.COL_GAMEWEEK], sort=False):
        mm_mv = grp["_mm_model_version"].iloc[0]
        mm_mv = int(mm_mv) if pd.notna(mm_mv) else None
        with bt.asof_scope(con, season, int(gw)):
            feats = _compute_step_features(con, season, int(gw), grp.reset_index(drop=True), mm_mv)
        feats[C.COL_SEASON] = season
        feats[C.COL_GAMEWEEK] = int(gw)
        parts.append(feats)

    feats_df = pd.concat(parts, ignore_index=True)
    keys = ["player_uid", C.COL_SEASON, C.COL_GAMEWEEK]
    out = df.merge(feats_df, on=keys, how="left")
    out.attrs.update(df.attrs)
    return out
