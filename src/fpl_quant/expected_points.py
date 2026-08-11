"""M3: Expected Points Engine.

Every FPL scoring category is its own sub-model, all conditioned on the same M2 minutes
distribution; category expectations are summed for total EP (linearity of expectation
holds regardless of correlation between categories -- correlation is M4's job).

Scope limitation, stated plainly rather than silently approximated: the BPS mu_i formula
below uses the components backed by what fact_reconciled actually carries (goals, assists,
saves, CBI/recoveries via fact_player_match_stats, appearance minutes) -- not the full
official 32-stat formula. Passing/crossing/key-pass/foul granularity was never reconciled
into fact_reconciled (M0 scoped fact_player_match_stats to a deliberate column subset), so
those BPS components are omitted here rather than faked from data that doesn't exist.

Verified against current 2026/27 rules before writing any of this (kickoff notes' hard
precondition) -- see seed_v1_params() for the full source list and the one genuine
unresolved ambiguity (outside-box GK saves in BPS).
"""

import math
from datetime import date, datetime, timezone

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import poisson

from . import minutes_model as mm
from . import params as params_mod
from . import reconcile as reconcile_mod

PL = "Premier League"
POSITIONS = ["Goalkeeper", "Defender", "Midfielder", "Forward"]


# ============================================================
# v1 params: base scoring matrix + BPS formula (verified 2026/27 rules)
# ============================================================

def seed_v1_params(con: duckdb.DuckDBPyConnection) -> None:
    w = lambda key, value, dims=None: params_mod.write_param(  # noqa: E731
        con, "base_scoring_matrix", 1, "2026-08-10", key, value_numeric=value, dimensions=dims
    )
    for pos, pts in {"Goalkeeper": 10, "Defender": 6, "Midfielder": 5, "Forward": 4}.items():
        w("goal_points", pts, {"position": pos})
    for pos, pts in {"Goalkeeper": 4, "Defender": 4, "Midfielder": 1, "Forward": 0}.items():
        w("clean_sheet_points", pts, {"position": pos})
    w("assist_points", 3.0)
    w("appearance_points_1_59", 1.0)
    w("appearance_points_60plus", 2.0)
    w("saves_per_point", 3.0)
    w("goals_conceded_per_point", 2.0)
    w("penalty_save_points", 5.0)
    w("penalty_miss_points", -2.0)
    w("own_goal_points", -2.0)
    w("yellow_card_points", -1.0)
    w("red_card_points", -3.0)
    for pos, thr in {"Defender": 10, "Midfielder": 12, "Forward": 12}.items():
        w("defcon_threshold", thr, {"position": pos})
    w("defcon_points", 2.0)

    b = lambda key, value, dims=None: params_mod.write_param(  # noqa: E731
        con, "bps_formula_params", 1, "2026-08-10", key, value_numeric=value, dimensions=dims
    )
    for pos, v in {"Goalkeeper": 12, "Defender": 12, "Midfielder": 18, "Forward": 24}.items():
        b("goal", v, {"position": pos})
    b("penalty_goal", 12)
    b("assist", 9)
    b("save_inside_box", 3)
    # 2026/27-confirmed deltas (Premier League's own announcement + Fantasy Football
    # Scout, cross-checked against the workbook's own 13_Rule Changes Database):
    b("penalty_save", 7)              # reduced from 8
    b("gk_save_cross_deflection", 2)  # new category
    b("gk_save_big_chance", 1)        # new category
    b("cbi_per_point", 3.0)           # was 1 point per 2 CBI; now per 3
    b("being_tackled", 0.0)           # the -1 "being tackled" penalty was removed entirely
    # Genuine unresolved ambiguity: the workbook's research says outside-box GK saves are
    # removed from bonus scoring entirely; external 2026/27 sources suggest a base +2 BPS
    # is retained with only the box-specific bonus removed. Going with the workbook's more
    # specific, directly-sourced claim (0), flagged here rather than silently picked.
    b("save_outside_box", 0.0)
    b("recoveries_per_point", 3.0)    # legacy, unchanged
    b("playing_1_60", 3)
    b("playing_60plus", 6)
    b("goal_conceded_gk_def", -4)     # legacy, unchanged

    # tau (Plackett-Luce dispersion): invented v1 default, no literature to cite -- flagged
    # for M7 recalibration once real 2026-27 BPS outcomes exist to fit against (per spec).
    params_mod.write_param(con, "bps_dispersion_params", 1, "2026-08-10", "tau", value_numeric=10.0)


def _sm(con, key, params_version, position=None):
    dims = {"position": position} if position else None
    v, _ = params_mod.resolve_param(con, "base_scoring_matrix", key, params_version, dimensions=dims)
    return v


def _bp(con, key, params_version, position=None):
    dims = {"position": position} if position else None
    v, _ = params_mod.resolve_param(con, "bps_formula_params", key, params_version, dimensions=dims)
    return v


# ============================================================
# per-90 rate sourcing: pooled across the lookback window, shrunk toward the position
# average by sample size. A per-90 rate extrapolated from a handful of minutes is noise,
# not signal -- real example this project hit: a player with a single 2-minute cameo and
# one lucky xG contribution extrapolated to expected_goals_per_90=3.6, which without
# shrinkage briefly made him rank above Haaland for a gameweek's expected goals.
# ============================================================

# Invented v1 default (no literature to cite, same status as every other invented constant
# in this project): the sample-minutes point at which a player's own rate and the position
# average get equal weight. Flagged for M7 recalibration.
RATE_SHRINKAGE_K_MINUTES = 450.0


def _shrink_rate(own_rate: float, sample_minutes: float, position_avg_rate: float, k: float = RATE_SHRINKAGE_K_MINUTES) -> float:
    weight_own = sample_minutes / (sample_minutes + k)
    return weight_own * own_rate + (1 - weight_own) * position_avg_rate


def _player_rate_pool(con: duckdb.DuckDBPyConnection, player_uid: str, season_priority: list[str]) -> dict:
    """Pools each lookback season's latest (most complete cumulative) row, weighted by that
    season's own total minutes -- not a single cherry-picked season."""
    total_minutes = total_goals = total_assists = total_saves_weighted = 0.0
    for season in season_priority:
        row = con.execute(
            "SELECT minutes, expected_goals, expected_assists, saves_per_90 "
            "FROM fact_player_season_stats WHERE player_uid = ? AND season = ? ORDER BY gw DESC LIMIT 1",
            [player_uid, season],
        ).fetchone()
        if not row or not row[0]:
            continue
        minutes, xg, xa, saves_p90 = row
        total_minutes += minutes
        total_goals += xg or 0.0
        total_assists += xa or 0.0
        total_saves_weighted += (saves_p90 or 0.0) * minutes
    if total_minutes <= 0:
        return {"expected_goals_per_90": 0.0, "expected_assists_per_90": 0.0, "saves_per_90": 0.0, "sample_minutes": 0.0}
    return {
        "expected_goals_per_90": total_goals / total_minutes * 90,
        "expected_assists_per_90": total_assists / total_minutes * 90,
        "saves_per_90": total_saves_weighted / total_minutes,
        "sample_minutes": total_minutes,
    }


def _position_average_rates(con: duckdb.DuckDBPyConnection, position: str, season_priority: list[str]) -> dict:
    placeholders = ",".join(["?"] * len(season_priority))
    row = con.execute(
        f"""
        SELECT avg(expected_goals_per_90), avg(expected_assists_per_90), avg(saves_per_90)
        FROM fact_player_season_stats fps
        JOIN dim_player dp ON dp.player_uid = fps.player_uid
        WHERE dp.position = ? AND fps.season IN ({placeholders}) AND fps.minutes > 0
        """,
        [position, *season_priority],
    ).fetchone()
    return {
        "expected_goals_per_90": row[0] or 0.0,
        "expected_assists_per_90": row[1] or 0.0,
        "saves_per_90": row[2] or 0.0,
    }


def player_rates_shrunk(con: duckdb.DuckDBPyConnection, player_uid: str, position: str, season_priority: list[str]) -> dict:
    own = _player_rate_pool(con, player_uid, season_priority)
    pos_avg = _position_average_rates(con, position, season_priority)
    return {
        key: _shrink_rate(own[key], own["sample_minutes"], pos_avg[key])
        for key in ("expected_goals_per_90", "expected_assists_per_90", "saves_per_90")
    }


def _defensive_action_rates_per_90(con: duckdb.DuckDBPyConnection, player_uid: str, position: str, seasons: list[str]) -> dict:
    """CBI (tackles+clearances+interceptions+blocks) and recoveries, per 90 minutes, from
    fact_player_match_stats -- the only place these are reconciled at per-match grain.
    Shrunk toward the position average the same way and for the same reason as the goals/
    assists/saves rates above."""
    placeholders = ",".join(["?"] * len(seasons))
    row = con.execute(
        f"""
        SELECT
            sum(coalesce(tackles,0) + coalesce(clearances,0) + coalesce(interceptions,0) + coalesce(blocks,0)) AS cbi_total,
            sum(coalesce(recoveries,0)) AS recoveries_total,
            sum(minutes_played) AS minutes_total
        FROM fact_player_match_stats
        WHERE player_uid = ? AND season IN ({placeholders})
        """,
        [player_uid, *seasons],
    ).fetchone()
    cbi_total, recoveries_total, minutes_total = row
    own_cbi = (cbi_total or 0) / minutes_total * 90 if minutes_total else 0.0
    own_recoveries = (recoveries_total or 0) / minutes_total * 90 if minutes_total else 0.0

    pos_row = con.execute(
        f"""
        SELECT
            sum(coalesce(pmst.tackles,0) + coalesce(pmst.clearances,0) + coalesce(pmst.interceptions,0) + coalesce(pmst.blocks,0)),
            sum(coalesce(pmst.recoveries,0)), sum(pmst.minutes_played)
        FROM fact_player_match_stats pmst
        JOIN dim_player dp ON dp.player_uid = pmst.player_uid
        WHERE dp.position = ? AND pmst.season IN ({placeholders})
        """,
        [position, *seasons],
    ).fetchone()
    pos_cbi_total, pos_recoveries_total, pos_minutes_total = pos_row
    pos_avg_cbi = (pos_cbi_total or 0) / pos_minutes_total * 90 if pos_minutes_total else 0.0
    pos_avg_recoveries = (pos_recoveries_total or 0) / pos_minutes_total * 90 if pos_minutes_total else 0.0

    sample_minutes = minutes_total or 0.0
    return {
        "cbi_per_90": _shrink_rate(own_cbi, sample_minutes, pos_avg_cbi),
        "recoveries_per_90": _shrink_rate(own_recoveries, sample_minutes, pos_avg_recoveries),
    }


# ============================================================
# expected minutes, conditional on playing
# ============================================================

def _mean_minutes_by_bucket(con: duckdb.DuckDBPyConnection) -> dict:
    """Empirical mean minutes_played conditional on landing in the 1-59 vs 60+ bucket --
    derived from real match data rather than assumed round numbers."""
    row = con.execute(
        """
        SELECT
            avg(CASE WHEN minutes_played BETWEEN 1 AND 59 THEN minutes_played END),
            avg(CASE WHEN minutes_played >= 60 THEN minutes_played END)
        FROM fact_player_match_stats
        """
    ).fetchone()
    return {"mean_1_59": row[0] or 30.0, "mean_60plus": row[1] or 85.0}


def expected_minutes_given_played(p_1_59: float, p_60plus: float, mean_minutes: dict) -> float:
    p_played = p_1_59 + p_60plus
    if p_played <= 0:
        return 0.0
    unconditional = mean_minutes["mean_1_59"] * p_1_59 + mean_minutes["mean_60plus"] * p_60plus
    return unconditional / p_played


# ============================================================
# fixture-level team strength lookup
# ============================================================

def _fixture_lambdas(con: duckdb.DuckDBPyConnection, team_uid: str, match_id: str, ts_model_version: int):
    match = con.execute(
        "SELECT home_team_uid, away_team_uid FROM fact_match WHERE match_id = ?", [match_id]
    ).fetchone()
    home_uid, away_uid = match
    is_home = team_uid == home_uid
    opp_uid = away_uid if is_home else home_uid

    home_adv = con.execute(
        "SELECT home_advantage FROM team_strength_model_versions WHERE model_version = ?", [ts_model_version]
    ).fetchone()[0]

    own = con.execute(
        "SELECT final_attack, final_defence FROM team_strength_snapshots WHERE model_version = ? AND team_uid = ?",
        [ts_model_version, team_uid],
    ).fetchone()
    opp = con.execute(
        "SELECT final_attack, final_defence FROM team_strength_snapshots WHERE model_version = ? AND team_uid = ?",
        [ts_model_version, opp_uid],
    ).fetchone()
    own_attack, own_defence = own
    opp_attack, opp_defence = opp

    adv_own = home_adv if is_home else 0.0
    adv_opp = home_adv if not is_home else 0.0
    lambda_for = math.exp(own_attack - opp_defence + adv_own)
    lambda_against = math.exp(opp_attack - own_defence + adv_opp)
    return lambda_for, lambda_against, is_home


def _expected_floor_half(lam: float, max_k: int = 15) -> float:
    """E[floor(X/2)] for X ~ Poisson(lam) -- the exact expectation under FPL's -1-per-2-
    conceded rule, not a linear approximation."""
    return sum((k // 2) * poisson.pmf(k, lam) for k in range(max_k + 1))


# ============================================================
# Plackett-Luce bonus sub-model
# ============================================================

def plackett_luce_bonus(strengths: dict[str, float]) -> dict[str, float]:
    """E[bonus_i] = 3*P(rank1=i) + 2*P(rank2=i) + 1*P(rank3=i), via sequential
    marginalization (M3 spec's formula, applied literally)."""
    players = list(strengths.keys())
    total = sum(strengths.values())
    if total <= 0 or len(players) == 0:
        return {p: 0.0 for p in players}

    p_rank1 = {p: strengths[p] / total for p in players}

    p_rank2 = {p: 0.0 for p in players}
    for k in players:
        remaining_total = total - strengths[k]
        if remaining_total <= 0:
            continue
        for i in players:
            if i == k:
                continue
            p_rank2[i] += p_rank1[k] * (strengths[i] / remaining_total)

    p_rank3 = {p: 0.0 for p in players}
    if len(players) >= 3:
        for k in players:
            for j in players:
                if j == k:
                    continue
                remaining_after_k = total - strengths[k]
                remaining_after_kj = remaining_after_k - strengths[j]
                if remaining_after_k <= 0 or remaining_after_kj <= 0:
                    continue
                p_k = p_rank1[k]
                p_j_given_k = strengths[j] / remaining_after_k
                for i in players:
                    if i in (k, j):
                        continue
                    p_i_given_kj = strengths[i] / remaining_after_kj
                    p_rank3[i] += p_k * p_j_given_k * p_i_given_kj

    return {p: 3 * p_rank1[p] + 2 * p_rank2.get(p, 0.0) + 1 * p_rank3.get(p, 0.0) for p in players}


# ============================================================
# per-player, per-fixture sub-models (everything except bonus, which needs the whole fixture)
# ============================================================

def compute_player_fixture_components(
    con: duckdb.DuckDBPyConnection, player_uid: str, position: str, team_uid: str, match_id: str,
    p_0: float, p_1_59: float, p_60plus: float,
    ts_model_version: int, scoring_params_version: int, bps_params_version: int,
    season_priority: list[str], mean_minutes: dict,
) -> dict:
    rates = player_rates_shrunk(con, player_uid, position, season_priority)
    def_rates = _defensive_action_rates_per_90(con, player_uid, position, season_priority)
    e_min_played = expected_minutes_given_played(p_1_59, p_60plus, mean_minutes)
    p_played = p_1_59 + p_60plus

    lambda_for, lambda_against, is_home = _fixture_lambdas(con, team_uid, match_id, ts_model_version)

    # ---- appearance ----
    ep_appearance = (
        _sm(con, "appearance_points_1_59", scoring_params_version) * p_1_59
        + _sm(con, "appearance_points_60plus", scoring_params_version) * p_60plus
    )

    # ---- goals / assists ----
    e_goals = rates["expected_goals_per_90"] * e_min_played / 90.0 * p_played
    e_assists = rates["expected_assists_per_90"] * e_min_played / 90.0 * p_played
    ep_goals = e_goals * _sm(con, "goal_points", scoring_params_version, position)
    ep_assists = e_assists * _sm(con, "assist_points", scoring_params_version)

    # ---- clean sheet (exact binary 60+ gate, GK/DEF/MID only per base scoring matrix) ----
    p_clean_sheet = math.exp(-lambda_against) * p_60plus
    ep_clean_sheet = p_clean_sheet * _sm(con, "clean_sheet_points", scoring_params_version, position)

    # ---- goals conceded (approximate binary 60+ gate, GK/DEF only) ----
    ep_goals_conceded = 0.0
    if position in ("Goalkeeper", "Defender"):
        e_floor_half_conceded = _expected_floor_half(lambda_against) * p_60plus
        ep_goals_conceded = -1.0 * e_floor_half_conceded

    # ---- DefCon (count-distribution, thresholded, gated by minutes) ----
    ep_defcon = 0.0
    if position in ("Defender", "Midfielder", "Forward"):
        defcon_rate = (def_rates["cbi_per_90"] + def_rates["recoveries_per_90"]) * e_min_played / 90.0
        threshold = _sm(con, "defcon_threshold", scoring_params_version, position)
        p_over_threshold = 1.0 - poisson.cdf(threshold - 1, max(defcon_rate, 1e-9)) if defcon_rate > 0 else 0.0
        ep_defcon = p_over_threshold * p_played * _sm(con, "defcon_points", scoring_params_version)

    # ---- saves / penalty saves (goalkeepers only) ----
    ep_saves = 0.0
    ep_penalty_save = 0.0
    if position == "Goalkeeper":
        e_saves = rates["saves_per_90"] * e_min_played / 90.0 * p_played
        ep_saves = e_saves / _sm(con, "saves_per_point", scoring_params_version)
        # No penalty-taker/penalties-faced rate reconciled -- left at 0 rather than guessed.

    # ---- expected BPS (mu_i), components backed by reconciled data only (see module docstring) ----
    mu = 0.0
    mu += _bp(con, "playing_1_60", bps_params_version) * p_1_59
    mu += _bp(con, "playing_60plus", bps_params_version) * p_60plus
    mu += e_goals * _bp(con, "goal", bps_params_version, position)
    mu += e_assists * _bp(con, "assist", bps_params_version)
    e_cbi = def_rates["cbi_per_90"] * e_min_played / 90.0 * p_played
    e_recoveries = def_rates["recoveries_per_90"] * e_min_played / 90.0 * p_played
    mu += e_cbi / _bp(con, "cbi_per_point", bps_params_version)
    mu += e_recoveries / _bp(con, "recoveries_per_point", bps_params_version)
    if position in ("Goalkeeper", "Defender"):
        mu += _bp(con, "goal_conceded_gk_def", bps_params_version) * (_expected_floor_half(lambda_against) * 2) * p_60plus
    if position == "Goalkeeper":
        e_saves = rates["saves_per_90"] * e_min_played / 90.0 * p_played
        mu += e_saves * _bp(con, "save_inside_box", bps_params_version)

    return {
        "position": position, "match_id": match_id,
        "ep_appearance": ep_appearance, "ep_goals": ep_goals, "ep_assists": ep_assists,
        "ep_clean_sheet": ep_clean_sheet, "ep_goals_conceded": ep_goals_conceded,
        "ep_defcon": ep_defcon, "ep_saves": ep_saves, "ep_penalty_save": ep_penalty_save,
        "ep_cards": 0.0, "ep_own_goal": 0.0,  # no reconciled per-90 rate for these -- left at 0, not guessed
        "expected_bps": mu, "p_played": p_played,
    }


# ============================================================
# orchestrator
# ============================================================

def run(
    con: duckdb.DuckDBPyConnection,
    calibration_asof_date: date,
    target_season: str,
    target_gameweek: int,
    ts_model_version: int,
    mm_model_version: int,
    scoring_params_version: int,
    bps_params_version: int,
    tau_params_version: int,
    lookback_seasons: tuple[str, ...] = ("2026-2027", "2025-2026", "2024-2025"),
) -> int:
    tau, _ = params_mod.resolve_param(con, "bps_dispersion_params", "tau", tau_params_version)
    mean_minutes = _mean_minutes_by_bucket(con)

    fixtures = con.execute(
        "SELECT match_id, home_team_uid, away_team_uid FROM fact_match "
        "WHERE season = ? AND gameweek = ? AND competition = ?",
        [target_season, target_gameweek, PL],
    ).fetchall()
    if not fixtures:
        raise ValueError(f"no {PL} fixtures found for {target_season} GW{target_gameweek}")

    model_version = con.execute(
        """
        INSERT INTO ep_model_versions
            (calibration_asof_date, target_season, team_strength_model_version, minutes_model_version,
             scoring_matrix_params_version, bps_params_version, bps_tau_params_version)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        RETURNING model_version
        """,
        [calibration_asof_date, target_season, ts_model_version, mm_model_version,
         scoring_params_version, bps_params_version, tau_params_version],
    ).fetchone()[0]

    for match_id, home_uid, away_uid in fixtures:
        fixture_rows = []
        for team_uid in (home_uid, away_uid):
            roster = con.execute(
                """
                SELECT DISTINCT dp.player_uid, dp.position
                FROM player_alias pa
                JOIN dim_player dp ON dp.player_uid = pa.player_uid
                JOIN "{}" t ON t.code = pa.team_code
                JOIN team_alias ta ON ta.alias_name = t.name AND ta.season = pa.season
                WHERE pa.season = ? AND ta.team_uid = ?
                """.format(reconcile_mod._season_root_table(con, target_season, "teams.csv")[1]),
                [target_season, team_uid],
            ).fetchall()
            for player_uid, position in roster:
                if position not in POSITIONS:
                    # a small number of non-player rows (managers, blank positions) leak
                    # into players.csv -- not a real squad player, skip rather than crash
                    # on an unseeded scoring-matrix lookup.
                    continue
                mrow = con.execute(
                    "SELECT p_0min, p_1_59min, p_60plus_min FROM minutes_model_outputs "
                    "WHERE model_version = ? AND player_uid = ?", [mm_model_version, player_uid],
                ).fetchone()
                if not mrow:
                    continue
                p_0, p_1_59, p_60plus = mrow
                comp = compute_player_fixture_components(
                    con, player_uid, position, team_uid, match_id, p_0, p_1_59, p_60plus,
                    ts_model_version, scoring_params_version, bps_params_version,
                    list(lookback_seasons), mean_minutes,
                )
                comp["player_uid"] = player_uid
                fixture_rows.append(comp)

        strengths = {r["player_uid"]: math.exp(r["expected_bps"] / tau) * r["p_played"] for r in fixture_rows}
        bonus_by_player = plackett_luce_bonus(strengths)

        for r in fixture_rows:
            ep_bonus = bonus_by_player.get(r["player_uid"], 0.0)
            ep_total = (
                r["ep_appearance"] + r["ep_goals"] + r["ep_assists"] + r["ep_clean_sheet"]
                + r["ep_goals_conceded"] + r["ep_defcon"] + ep_bonus + r["ep_saves"]
                + r["ep_penalty_save"] + r["ep_cards"] + r["ep_own_goal"]
            )
            con.execute(
                """
                INSERT INTO ep_outputs
                    (model_version, player_uid, fixture_match_id, ep_appearance, ep_goals, ep_assists,
                     ep_clean_sheet, ep_goals_conceded, ep_defcon, ep_bonus, ep_saves, ep_penalty_save,
                     ep_cards, ep_own_goal, ep_total, expected_bps)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (model_version, player_uid, fixture_match_id) DO NOTHING
                """,
                [model_version, r["player_uid"], r["match_id"], r["ep_appearance"], r["ep_goals"],
                 r["ep_assists"], r["ep_clean_sheet"], r["ep_goals_conceded"], r["ep_defcon"], ep_bonus,
                 r["ep_saves"], r["ep_penalty_save"], r["ep_cards"], r["ep_own_goal"], ep_total, r["expected_bps"]],
            )

    return model_version


# ============================================================
# non-double-counting audit (required, testable invariant per M3 spec)
# ============================================================

# Every raw stat this engine actually consumes, and every EP/BPS category it feeds. A stat
# feeding more than one category is flagged explicit and reasoned about, not left to be
# noticed by accident -- e.g. a CBI action legitimately feeds both DefCon (a real FPL
# scoring rule) and the BPS estimate (a real, separate FPL scoring rule); that's two
# genuinely distinct mechanisms, not the same points counted twice.
NON_DOUBLE_COUNTING_AUDIT = [
    {"raw_stat": "minutes (P(1-59)/P(60+) from M2)", "feeds": ["ep_appearance", "expected_bps (playing_1_60/60plus)"],
     "intentional_dual_use": True, "note": "appearance points and playing-time BPS are separate real FPL mechanisms"},
    {"raw_stat": "expected_goals_per_90", "feeds": ["ep_goals", "expected_bps (goal)"],
     "intentional_dual_use": True, "note": "goal points and goal BPS are separate real FPL mechanisms"},
    {"raw_stat": "expected_assists_per_90", "feeds": ["ep_assists", "expected_bps (assist)"],
     "intentional_dual_use": True, "note": "assist points and assist BPS are separate real FPL mechanisms"},
    {"raw_stat": "opponent lambda (M1)", "feeds": ["ep_clean_sheet", "ep_goals_conceded", "expected_bps (goal_conceded_gk_def)"],
     "intentional_dual_use": True, "note": "clean sheet, goals-conceded points, and goals-conceded BPS are three separate real FPL mechanisms off the same underlying goal count"},
    {"raw_stat": "CBI (tackles+clearances+interceptions+blocks)", "feeds": ["ep_defcon", "expected_bps (cbi_per_point)"],
     "intentional_dual_use": True, "note": "the exact example named in M3's own spec -- DefCon and BPS are separate real FPL scoring mechanisms"},
    {"raw_stat": "recoveries", "feeds": ["ep_defcon", "expected_bps (recoveries_per_point)"],
     "intentional_dual_use": True, "note": "same reasoning as CBI above"},
    {"raw_stat": "saves_per_90", "feeds": ["ep_saves", "expected_bps (save_inside_box)"],
     "intentional_dual_use": True, "note": "save points and save BPS are separate real FPL mechanisms"},
    {"raw_stat": "ep_bonus (Plackett-Luce over expected_bps)", "feeds": ["ep_total"],
     "intentional_dual_use": False, "note": "expected_bps itself is a ranking input, not points -- only realized bonus points enter ep_total, so this is not additional double counting"},
]

_NOT_MODELED_FOR_LACK_OF_RECONCILED_DATA = [
    "cards", "own_goal", "penalty_save", "penalty_miss",
    "passing/crossing/key-pass BPS components", "goalline_clearance", "winning_goal",
]


def non_double_counting_audit() -> list[dict]:
    return NON_DOUBLE_COUNTING_AUDIT
