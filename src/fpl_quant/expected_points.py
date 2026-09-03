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

import json
import math
from datetime import date, datetime, timezone

import duckdb
from scipy.stats import poisson

from . import params as params_mod
from . import reconcile as reconcile_mod
from . import snapshot as snapshot_mod

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

    # Invented v1 default (no reconciled penalty-frequency/conversion data to derive it from,
    # same honest gap as GK penalty saves being left at 0 above) for the optional confirmed-
    # primary-penalty-taker goal-rate uplift -- see _set_piece_goal_uplift_multiplier(). A
    # modest ~15% boost: enough to matter for a real early-season/new-signing case, deliberately
    # not large enough to swamp the real historical xG signal it's applied on top of.
    params_mod.write_param(con, "set_piece_evidence_params", 1, "2026-08-10", "penalty_taker_goal_rate_multiplier", value_numeric=1.15)
    # Priority 7b: invented v1 defaults, same status/reasoning as the penalty multiplier above
    # -- see _set_piece_goal_uplift_multiplier()/_set_piece_assist_uplift_multiplier(). A
    # direct free-kick is real but far rarer/lower-probability than a penalty, so a more modest
    # ~5% boost; a confirmed corner/free-kick delivery specialist's real value (many attempts
    # per match, not an occasional penalty) is a genuinely larger ~20% boost to e_assists.
    params_mod.write_param(con, "set_piece_evidence_params", 1, "2026-08-10", "free_kick_taker_goal_rate_multiplier", value_numeric=1.05)
    params_mod.write_param(con, "set_piece_evidence_params", 1, "2026-08-10", "set_piece_deliverer_assist_rate_multiplier", value_numeric=1.20)

    # Fixture-strength scaling of e_goals/e_assists (see _fixture_attack_multiplier). v1 = 1.0
    # is the full first-order adjustment (multiplier = lambda_for(this fixture) / team's own
    # season-mean lambda_for); the walk-forward measures whether it wants damping. Invented v1
    # default, flagged for M7 recalibration -- same status as tau / the set-piece multipliers.
    params_mod.write_param(con, "fixture_strength_params", 1, "2026-08-10", "attack_sensitivity", value_numeric=1.0)


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


def _season_match_minutes(con: duckdb.DuckDBPyConnection, player_uid: str, season: str) -> float:
    """A player's total real minutes that season, from the per-match grain -- the only place
    2024-2025 minutes exist (its playerstats.csv snapshot predates the season-total `minutes`
    column 2025-26+ has; see reconcile.build_fact_player_season_stats)."""
    row = con.execute(
        "SELECT sum(minutes_played) FROM fact_player_match_stats WHERE player_uid = ? AND season = ?",
        [player_uid, season],
    ).fetchone()
    return float(row[0]) if row and row[0] else 0.0


def _player_rate_pool(con: duckdb.DuckDBPyConnection, player_uid: str, season_priority: list[str]) -> dict:
    """Pools each lookback season's latest (most complete cumulative) row, weighted by that
    season's own total minutes -- not a single cherry-picked season.

    Two source schemas: 2025-26+ publishes a season-total `minutes` + `expected_goals`
    (cumulative), while 2024-2025's snapshot publishes `expected_goals_per_90` directly but no
    season-total minutes or xG. The old code required `minutes`, so it silently dropped ALL of
    2024-2025 -- halving the attacking-rate sample for every player and shrinking premiums
    (high own rate, small sample) hardest toward the position average, exactly the EP
    compression the DefCon rate (which reads fact_player_match_stats and DOES see 2024-25) does
    not suffer. This now recovers 2024-25 from the per-90 rate + match-grain minutes."""
    total_minutes = total_goals = total_assists = total_saves_weighted = saves_minutes = 0.0
    for season in season_priority:
        row = con.execute(
            "SELECT minutes, expected_goals, expected_assists, saves_per_90, "
            "expected_goals_per_90, expected_assists_per_90 "
            "FROM fact_player_season_stats WHERE player_uid = ? AND season = ? ORDER BY gw DESC LIMIT 1",
            [player_uid, season],
        ).fetchone()
        if not row:
            continue
        minutes, xg, xa, saves_p90, xg90, xa90 = row
        if minutes and xg is not None:
            total_minutes += minutes
            total_goals += xg or 0.0
            total_assists += xa or 0.0
            if saves_p90 is not None:
                total_saves_weighted += saves_p90 * minutes
                saves_minutes += minutes
        elif xg90 is not None or xa90 is not None:
            mins = _season_match_minutes(con, player_uid, season)
            if mins <= 0:
                continue
            total_minutes += mins
            total_goals += (xg90 or 0.0) / 90.0 * mins
            total_assists += (xa90 or 0.0) / 90.0 * mins
            # saves_per_90 genuinely isn't in this schema -- a snapshot-only season contributes
            # nothing to the saves anchor rather than a fabricated 0 that would drag it down.
    if total_minutes <= 0:
        return {"expected_goals_per_90": 0.0, "expected_assists_per_90": 0.0, "saves_per_90": 0.0, "sample_minutes": 0.0}
    return {
        "expected_goals_per_90": total_goals / total_minutes * 90,
        "expected_assists_per_90": total_assists / total_minutes * 90,
        "saves_per_90": total_saves_weighted / saves_minutes if saves_minutes > 0 else 0.0,
        "sample_minutes": total_minutes,
    }


def _position_average_rates(con: duckdb.DuckDBPyConnection, position: str, season_priority: list[str]) -> dict:
    """The shrinkage anchor for goals/assists/saves. Minutes-weighted from each player's
    LATEST (most complete cumulative) row per season -- the exact construction _player_rate_pool()
    uses for a player's own rate, and _defensive_action_rates_per_90() uses for the CBI/recoveries
    anchor right below. The previous version did an unweighted avg() over every per-gameweek
    cumulative snapshot (fact_player_season_stats is one row per (player, season, gw)), which
    (a) counted a 200-minute fringe player the same as a 3000-minute regular and (b) folded in
    the very noisy early-season snapshots at full weight -- both pull the anchor toward zero,
    and _shrink_rate() then compresses every player's rate toward that too-low anchor (the same
    EP-compression failure mode as the DefCon/minutes fixes)."""
    placeholders = ",".join(["?"] * len(season_priority))
    # Two source schemas, same as _player_rate_pool: the richer 2025-26+ rows carry a
    # season-total `minutes` + `expected_goals`; 2024-2025's snapshot carries
    # `expected_goals_per_90` directly but NULL minutes. The old query's `fps.minutes > 0`
    # filter dropped every 2024-25 player from the anchor -- so the anchor (and every rate
    # shrunk toward it) was fit on one season while _defensive_action_rates_per_90()'s anchor
    # saw two. The snapshot branch recovers those players via the per-90 rate x match-grain
    # minutes; `xg_total`/`xa_total` are the implied season counts so the minutes-weighted
    # aggregate below stays a single consistent formula across both branches.
    row = con.execute(
        f"""
        WITH latest AS (
            SELECT fps.expected_goals AS xg_total, fps.expected_assists AS xa_total,
                   fps.saves_per_90, fps.minutes AS mins
            FROM fact_player_season_stats fps
            JOIN dim_player dp ON dp.player_uid = fps.player_uid
            WHERE dp.position = ? AND fps.season IN ({placeholders}) AND fps.minutes > 0
            QUALIFY row_number() OVER (PARTITION BY fps.player_uid, fps.season ORDER BY fps.gw DESC) = 1

            UNION ALL

            SELECT s.xg90 / 90.0 * m.mins AS xg_total, s.xa90 / 90.0 * m.mins AS xa_total,
                   NULL AS saves_per_90, m.mins
            FROM (
                SELECT fps.player_uid, fps.season,
                       fps.expected_goals_per_90 AS xg90, fps.expected_assists_per_90 AS xa90
                FROM fact_player_season_stats fps
                JOIN dim_player dp ON dp.player_uid = fps.player_uid
                WHERE dp.position = ? AND fps.season IN ({placeholders}) AND fps.minutes IS NULL
                  AND (fps.expected_goals_per_90 IS NOT NULL OR fps.expected_assists_per_90 IS NOT NULL)
                QUALIFY row_number() OVER (PARTITION BY fps.player_uid, fps.season ORDER BY fps.gw DESC) = 1
            ) s
            JOIN (
                SELECT player_uid, season, sum(minutes_played) AS mins
                FROM fact_player_match_stats GROUP BY player_uid, season
            ) m ON m.player_uid = s.player_uid AND m.season = s.season
            WHERE m.mins > 0
        )
        SELECT
            sum(coalesce(xg_total, 0)) / nullif(sum(mins), 0) * 90,
            sum(coalesce(xa_total, 0)) / nullif(sum(mins), 0) * 90,
            sum(coalesce(saves_per_90, 0) * mins) / nullif(sum(CASE WHEN saves_per_90 IS NOT NULL THEN mins ELSE 0 END), 0)
        FROM latest
        """,
        [position, *season_priority, position, *season_priority],
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

# ============================================================
# fixture-strength scaling of a player's attacking output.
#
# THE BUG this fixes: compute_player_fixture_components() computes lambda_for/lambda_against
# from M1's Dixon-Coles team strength, but only ever USES lambda_against (clean sheet, goals
# conceded, the BPS conceded term). A player's e_goals / e_assists were their flat
# season-average per-90 rate x minutes -- identical against Coventry or Man City. So a premium
# attacker never got their easy-fixture ceiling, while a defender's clean-sheet points WERE
# fixture-adjusted (via lambda_against) -- which is exactly why the walk-forward showed the
# model under-predicts £9m+ players by ~1 pt/game and a defender's good-fixture clean-sheet
# spike floats up next to premium attackers in the captain ranking.
#
# THE FIX: scale e_goals / e_assists by how favourable this fixture is for the player's team
# relative to a league-average opponent -- lambda_for(this fixture) / lambda_for(this team vs
# an average defence, half-home). A player's per-90 rate is ~proportional to team goals, so
# this is the first-order correct adjustment. `attack_sensitivity` (fixture_strength_params,
# v1 default 1.0 = full) damps it; the multiplier is clipped to [0.4, 2.5] so one extreme
# projected scoreline can't dominate. Backtest-gated -- flagged for M7 recalibration.
# ============================================================

_FIXTURE_REF_CACHE: dict = {}


def _league_defence_and_home_adv(con: duckdb.DuckDBPyConnection, ts_model_version: int) -> tuple[float, float]:
    """(mean final_defence across the league, home_advantage) for this snapshot set -- cached.
    Dixon-Coles centres mean ATTACK at 0 but not mean defence (see team_strength's own design
    note), so the 'average opponent' a player's flat rate is measured against has defence =
    this mean, not 0."""
    cached = _FIXTURE_REF_CACHE.get(ts_model_version)
    if cached is None:
        mean_def = con.execute(
            "SELECT avg(final_defence) FROM team_strength_snapshots WHERE model_version = ?",
            [ts_model_version],
        ).fetchone()[0]
        home_adv = con.execute(
            "SELECT home_advantage FROM team_strength_model_versions WHERE model_version = ?",
            [ts_model_version],
        ).fetchone()[0]
        cached = (float(mean_def or 0.0), float(home_adv or 0.0))
        _FIXTURE_REF_CACHE[ts_model_version] = cached
    return cached


def _fixture_attack_multiplier(
    con: duckdb.DuckDBPyConnection, team_uid: str, match_id: str, target_season: str,
    ts_model_version: int, fixture_params_version: int,
) -> float:
    """lambda_for(this fixture) / lambda_for(this team vs a league-average opponent, half-home).

    lambda_for = exp(own_attack - opp_defence + adv_own); the reference cancels own_attack, so
    the ratio is exp(mean_defence - opp_defence + adv_own - home_adv/2) -- i.e. purely how much
    weaker/stronger THIS opponent's defence is than average, plus the home/away swing. Needs
    only team-strength params (no fixture history), so it composes cleanly with asof_scope().
    `target_season` is accepted for signature symmetry / future use."""
    sensitivity, _ = params_mod.resolve_param(
        con, "fixture_strength_params", "attack_sensitivity", fixture_params_version,
    )
    if sensitivity == 0.0:
        return 1.0
    lambda_for, _lambda_against, is_home = _fixture_lambdas(con, team_uid, match_id, ts_model_version)
    mean_def, home_adv = _league_defence_and_home_adv(con, ts_model_version)
    own_attack = con.execute(
        "SELECT final_attack FROM team_strength_snapshots WHERE model_version = ? AND team_uid = ?",
        [ts_model_version, team_uid],
    ).fetchone()[0]
    ref_lambda = math.exp(own_attack - mean_def + home_adv / 2.0)
    if ref_lambda <= 0:
        return 1.0
    mult = (lambda_for / ref_lambda) ** sensitivity
    return max(0.4, min(2.5, mult))


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

def plackett_luce_rank_distribution(strengths: dict[str, float]) -> dict[str, tuple[float, float, float]]:
    """P(rank1=i), P(rank2=i), P(rank3=i) via sequential marginalization (M3 spec's
    formula, applied literally). Exposed separately from plackett_luce_bonus() so M4 can
    reconstruct bonus's full categorical distribution over {0,1,2,3} points, not just its
    mean -- Var[bonus] needs the whole distribution, not E[bonus] alone."""
    players = list(strengths.keys())
    total = sum(strengths.values())
    if total <= 0 or len(players) == 0:
        return {p: (0.0, 0.0, 0.0) for p in players}

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

    return {p: (p_rank1[p], p_rank2.get(p, 0.0), p_rank3.get(p, 0.0)) for p in players}


def plackett_luce_bonus(strengths: dict[str, float]) -> dict[str, float]:
    """E[bonus_i] = 3*P(rank1=i) + 2*P(rank2=i) + 1*P(rank3=i)."""
    dist = plackett_luce_rank_distribution(strengths)
    return {p: 3 * p1 + 2 * p2 + 1 * p3 for p, (p1, p2, p3) in dist.items()}


# ============================================================
# per-player, per-fixture sub-models (everything except bonus, which needs the whole fixture)
# ============================================================

# ============================================================
# optional set-piece rate uplift -- ingested but previously unused. ingest_research_pull.py's
# ingest_set_piece_takers() has written real claim_type="set_piece_order_override" claims
# ({club, duty, order: primary/secondary}) into evidence_claims since the module existed, but
# grepping the whole src/ tree turns up zero readers of that claim_type anywhere -- confirmed
# ingested and dormant, not a hypothetical gap. Originally scoped narrowly to confirmed PRIMARY
# penalty duty (the single highest-signal, best-understood case: penalty conversion is close
# to deterministic, and a summer transfer's new penalty duty won't yet show up in pure
# historical expected_goals_per_90, especially on a small early-season sample) -- free-kick/
# corner duty claims existed in the same tab but were deliberately left alone, a smaller,
# separately-scoped extension "if ever wanted" per that original comment. Priority 7b is that
# extension: free-kick duty gets its own (smaller) e_goals uplift alongside penalties -- a
# direct free-kick is a real, if far rarer, scoring opportunity for the taker, same mechanism,
# different invented magnitude -- and confirmed corner/free-kick DELIVERY duty gets a new
# e_assists uplift below, since a set-piece deliverer's real value is chances created for
# teammates, not goals for themselves.
#
# `duty` is free text lifted straight from a curated Excel tab (no fixed vocabulary enforced
# anywhere upstream), so this matches by substring the same permissive way the original
# penalty check already did ("penalt" in duty.lower()), not an exact-string enum.
# ============================================================

def _set_piece_goal_uplift_multiplier(
    con: duckdb.DuckDBPyConnection, player_uid: str, asof: datetime, set_piece_params_version: int,
) -> float:
    """1.0 (no-op) unless an asof-visible set_piece_order_override claim confirms this player
    as the PRIMARY penalty OR free-kick taker, in which case a small, versioned multiplicative
    uplift is applied to e_goals (a different, smaller magnitude for free-kicks -- direct FK
    conversion is real but much rarer than penalty conversion). No real set-piece-frequency/
    conversion data is reconciled anywhere in this project (same honest gap expected_points.py's
    own module docstring already names for GK penalty saves: "left at 0 rather than guessed")
    -- both uplift magnitudes are therefore invented v1 defaults, same status as every other
    unpinned constant here, flagged for M7 recalibration once real per-taker outcome data
    exists to fit them against. Checks penalty duty first (the higher-signal, more-established
    case) so a claim naming both somehow still resolves to the larger, more-defensible number."""
    claims = snapshot_mod.get_claims_asof(
        con, asof, subject_entity_type="player", subject_entity_id=player_uid, claim_type="set_piece_order_override",
    ).to_dict("records")
    penalty_claim, free_kick_claim = False, False
    for c in claims:
        if not c["claim_value"]:
            continue
        payload = json.loads(c["claim_value"])
        duty = (payload.get("duty") or "").lower()
        if payload.get("order") != "primary":
            continue
        if "penalt" in duty:
            penalty_claim = True
        elif "free kick" in duty or "free-kick" in duty or "freekick" in duty:
            free_kick_claim = True
    if penalty_claim:
        multiplier, _ = params_mod.resolve_param(
            con, "set_piece_evidence_params", "penalty_taker_goal_rate_multiplier", set_piece_params_version,
        )
        return multiplier
    if free_kick_claim:
        multiplier, _ = params_mod.resolve_param(
            con, "set_piece_evidence_params", "free_kick_taker_goal_rate_multiplier", set_piece_params_version,
        )
        return multiplier
    return 1.0


def _set_piece_assist_uplift_multiplier(
    con: duckdb.DuckDBPyConnection, player_uid: str, asof: datetime, set_piece_params_version: int,
) -> float:
    """1.0 (no-op) unless an asof-visible claim confirms this player as the PRIMARY corner OR
    free-kick DELIVERY taker, in which case a single versioned multiplicative uplift is applied
    to e_assists. Corners and free-kicks are treated as the same delivered-set-piece assist
    opportunity here, one shared multiplier rather than two separately-invented ones -- neither
    is reconciled with enough real outcome data in this project to justify tuning them apart, a
    genuine, disclosed simplification (a "free kick taker" claim doesn't distinguish direct-shot
    duty from out-swinging delivery duty in the source data anyway, so a free-kick claim
    legitimately contributes to BOTH this and the goal uplift above -- both are real possible
    sources of extra value from that role, not double-counting the same one)."""
    claims = snapshot_mod.get_claims_asof(
        con, asof, subject_entity_type="player", subject_entity_id=player_uid, claim_type="set_piece_order_override",
    ).to_dict("records")
    for c in claims:
        if not c["claim_value"]:
            continue
        payload = json.loads(c["claim_value"])
        duty = (payload.get("duty") or "").lower()
        if payload.get("order") != "primary":
            continue
        if "corner" in duty or "free kick" in duty or "free-kick" in duty or "freekick" in duty:
            multiplier, _ = params_mod.resolve_param(
                con, "set_piece_evidence_params", "set_piece_deliverer_assist_rate_multiplier", set_piece_params_version,
            )
            return multiplier
    return 1.0


def compute_player_fixture_components(
    con: duckdb.DuckDBPyConnection, player_uid: str, position: str, team_uid: str, match_id: str,
    p_0: float, p_1_59: float, p_60plus: float,
    ts_model_version: int, scoring_params_version: int, bps_params_version: int,
    season_priority: list[str], mean_minutes: dict,
    *, asof: datetime | None = None, set_piece_params_version: int | None = None,
    fixture_params_version: int | None = 1, target_season: str | None = None,
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
    # fixture-strength scaling: a player's flat season per-90 rate, adjusted for how favourable
    # THIS opponent is vs the team's average fixture (see _fixture_attack_multiplier). Was the
    # single biggest gap -- e_goals/e_assists were opponent-blind while clean sheets weren't.
    if fixture_params_version is not None:
        fixture_mult = _fixture_attack_multiplier(
            con, team_uid, match_id, target_season or season_priority[0],
            ts_model_version, fixture_params_version,
        )
        e_goals *= fixture_mult
        e_assists *= fixture_mult
    if asof is not None and set_piece_params_version is not None:
        e_goals *= _set_piece_goal_uplift_multiplier(con, player_uid, asof, set_piece_params_version)
        e_assists *= _set_piece_assist_uplift_multiplier(con, player_uid, asof, set_piece_params_version)
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
    # FPL's Defensive Contribution rule counts a different action set per position (verified
    # against the 2026/27 rules): a DEFENDER needs 10+ CBIT -- clearances, blocks,
    # interceptions, tackles -- and ball recoveries do NOT count toward it; a MIDFIELDER or
    # FORWARD needs 12+ of CBIT *plus* ball recoveries. Adding recoveries to a defender's rate
    # (as this originally did, unconditionally) roughly doubled the modelled action rate for
    # high-recovery centre-backs and full-backs, making almost every nailed starting defender
    # a near-certain +2 every week -- the single biggest reason defenders outranked premium
    # forwards for captaincy.
    ep_defcon = 0.0
    if position in ("Defender", "Midfielder", "Forward"):
        defcon_actions_per_90 = def_rates["cbi_per_90"]
        if position in ("Midfielder", "Forward"):
            defcon_actions_per_90 += def_rates["recoveries_per_90"]
        defcon_rate = defcon_actions_per_90 * e_min_played / 90.0
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
    set_piece_params_version: int | None = 1,
    fixture_params_version: int | None = 1,
) -> int:
    # set_piece_params_version defaults to 1 (was None): the confirmed-primary penalty/free-kick
    # taker e_goals/e_assists uplift (_set_piece_goal_uplift_multiplier, built as Priority 7b but
    # never actually called by any live entrypoint) is now ON. It is a per-player no-op unless an
    # asof-visible set_piece_order_override primary claim exists -- so historical seasons with no
    # such claims are unaffected. Pass None to opt out.
    tau, _ = params_mod.resolve_param(con, "bps_dispersion_params", "tau", tau_params_version)
    mean_minutes = _mean_minutes_by_bucket(con)
    # end-of-day, not start-of-day: same "as of this date" convention minutes_model.run()
    # already established -- a claim ingested at 09:34 on the asof date itself is legitimately
    # knowable "as of" that date. Only used when set_piece_params_version opts the uplift in.
    asof = datetime.combine(calibration_asof_date, datetime.max.time(), tzinfo=timezone.utc)

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
                    asof=asof, set_piece_params_version=set_piece_params_version,
                    fixture_params_version=fixture_params_version, target_season=target_season,
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
    {"raw_stat": "recoveries", "feeds": ["ep_defcon (Midfielder/Forward only)", "expected_bps (recoveries_per_point)"],
     "intentional_dual_use": True, "note": "same reasoning as CBI above; recoveries only count toward DefCon for MID/FWD, never for a Defender (whose DefCon threshold is CBIT-only per FPL rules)"},
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


# ============================================================
# M9 adapter -- category-level EP breakdown, display-ready shape
# ============================================================

def explain_player_ep(con: duckdb.DuckDBPyConnection, ep_model_version: int, player_uid: str) -> dict | None:
    """M9's category-level EP breakdown section: "not a single blended number, so a human can
    see e.g. this defender's value is mostly DefCon-driven." Pure read against ep_outputs,
    labeled by category rather than raw column name -- no new computation. Returns None if the
    player has no fixture at this model_version (a legitimate blank gameweek, not an error)."""
    row = con.execute(
        "SELECT fixture_match_id, ep_appearance, ep_goals, ep_assists, ep_clean_sheet, "
        "ep_goals_conceded, ep_defcon, ep_bonus, ep_saves, ep_penalty_save, ep_cards, "
        "ep_own_goal, ep_total, expected_bps FROM ep_outputs WHERE model_version = ? AND player_uid = ?",
        [ep_model_version, player_uid],
    ).fetchone()
    if row is None:
        return None
    (fixture_match_id, appearance, goals, assists, clean_sheet, goals_conceded, defcon,
     bonus, saves, penalty_save, cards, own_goal, total, expected_bps) = row
    return {
        "player_uid": player_uid, "fixture_match_id": fixture_match_id,
        "categories": {
            "appearance": appearance, "goals": goals, "assists": assists,
            "clean_sheet": clean_sheet, "goals_conceded": goals_conceded, "defcon": defcon,
            "bonus": bonus, "saves": saves, "penalty_save": penalty_save, "cards": cards,
            "own_goal": own_goal,
        },
        "total": total, "expected_bps": expected_bps,
    }
