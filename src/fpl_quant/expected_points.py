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
import numpy as np
import pandas as pd
from scipy.stats import poisson

from . import evidence_blend as eb
from . import minutes_model as mm
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

    # A3: extends the same set_piece_evidence_params family -- confirmed SECONDARY penalty duty
    # (previously ingested, silently ignored) now applies a bounded demotion, deliberately a
    # smaller magnitude than the primary uplift's +15% (a confirmed secondary taker still
    # sometimes scores penalties when the primary is unavailable, so full symmetric downside
    # isn't warranted). Free-kick/corner delivery duty is a new, separate assist-side signal
    # (this tab's rows were previously read into `duty` but never consumed at all) -- a modest
    # uplift/demotion on e_assists specifically, not e_goals: delivering a set piece creates a
    # teammate's chance, it isn't a personal shot the way a penalty is. All three invented v1
    # defaults, same status as every other unpinned constant here, flagged for M7 recalibration.
    params_mod.write_param(con, "set_piece_evidence_params", 1, "2026-08-10", "secondary_penalty_taker_goal_rate_multiplier", value_numeric=0.90)
    params_mod.write_param(con, "set_piece_evidence_params", 1, "2026-08-10", "set_piece_deliverer_assist_rate_multiplier", value_numeric=1.12)
    params_mod.write_param(con, "set_piece_evidence_params", 1, "2026-08-10", "secondary_set_piece_deliverer_assist_rate_multiplier", value_numeric=0.92)

    # A2: qualitative "predicted role" adjustment -- see _role_shift_multiplier(). exp_position
    # ("Expected Position") from 18_Predicted XI Database claims is a genuinely different
    # evidence dimension than the same claim's numeric start_conf (which minutes_model already
    # consumes for P(start)): this is about *what role* a starting player is expected to play,
    # not *whether* they start, so reading both off the same predicted_xi claim is not double
    # counting -- see NON_DOUBLE_COUNTING_AUDIT below. Both constants are invented v1 defaults,
    # same status as every other unpinned constant here, flagged for the same eventual M7
    # recalibration once real per-role attacking-output deltas exist to fit against. Deliberately
    # small and capped: this shades the historical xG/xA rate for a specific fixture's predicted
    # role, it doesn't override it.
    params_mod.write_param(con, "role_shift_params", 1, "2026-08-10", "per_rank_multiplier_step", value_numeric=0.08)
    params_mod.write_param(con, "role_shift_params", 1, "2026-08-10", "max_multiplier_delta", value_numeric=0.16)


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
# set-piece rate adjustments -- ingested but previously unused. ingest_research_pull.py's
# ingest_set_piece_takers() has written real claim_type="set_piece_order_override" claims
# ({club, duty, order: primary/secondary}) into evidence_claims since the module existed, but
# grepping the whole src/ tree turned up zero readers of that claim_type anywhere before this
# was first wired in -- confirmed ingested and dormant, not a hypothetical gap.
#
# A3 extends the original PRIMARY-penalty-only uplift two ways:
#   - a SECONDARY penalty claim (previously read and silently ignored) is now real evidence of
#     reduced personal conversion likelihood -- a bounded demotion, not just the absence of a
#     boost. Smaller magnitude than the primary uplift: a confirmed secondary taker still
#     sometimes scores penalties in practice (primary injured/subbed/off form), so full
#     symmetric downside isn't justified.
#   - free-kick/corner delivery duty (same tab, same claim_type, previously left alone entirely
#     as "a smaller, separately-scoped extension") is now a real e_assists adjustment --
#     deliberately kept off e_goals: delivering a set piece creates a teammate's chance, it
#     isn't a personal shot the way a penalty is.
#
# Both directions are now decided by real evidence STRENGTH (evidence_blend.effective_weight --
# source reliability x confidence x decay) rather than "whichever claim happens to be iterated
# first": when multiple sources disagree on a player's duty, the side with more/stronger
# supporting evidence wins; a genuine tie (including "no evidence either way") stays a no-op.
# No real penalty-frequency/conversion or set-piece-assist-rate data is reconciled anywhere in
# this project (same honest gap this module's own docstring already names for GK penalty
# saves) -- every multiplier below is an invented v1 default, flagged for M7 recalibration once
# real per-duty outcome data exists to fit it against.
# ============================================================

def _set_piece_duty_claims(
    con: duckdb.DuckDBPyConnection, player_uid: str, asof: datetime, duty_keywords: tuple[str, ...],
) -> list[dict]:
    """asof-visible set_piece_order_override claims whose duty string contains ANY of
    duty_keywords -- checked as a single any-of match (not one pass per keyword)
    specifically because the real data contains combined duty strings like
    "corners/free-kicks", which a per-keyword-summed loop would double-count. Each returned
    dict carries the parsed payload as claim['_payload'] alongside the raw claim fields."""
    claims = snapshot_mod.get_claims_asof(
        con, asof, subject_entity_type="player", subject_entity_id=player_uid, claim_type="set_piece_order_override",
    ).to_dict("records")
    matched = []
    for c in claims:
        if not c["claim_value"]:
            continue
        payload = json.loads(c["claim_value"])
        duty = (payload.get("duty") or "").lower()
        if not any(kw in duty for kw in duty_keywords):
            continue
        matched.append({**c, "_payload": payload})
    return matched


def _set_piece_duty_evidence(
    con: duckdb.DuckDBPyConnection, player_uid: str, asof: datetime, duty_keywords: tuple[str, ...],
    decay_params_version: int, fact_multiplier_params_version: int,
) -> tuple[float, float]:
    """Sums effective_weight() per order ('primary'/'secondary') across the claims
    _set_piece_duty_claims() matches."""
    primary_weight = secondary_weight = 0.0
    for c in _set_piece_duty_claims(con, player_uid, asof, duty_keywords):
        w = eb.effective_weight(con, c, asof, decay_params_version, fact_multiplier_params_version)
        order = c["_payload"].get("order")
        if order == "primary":
            primary_weight += w
        elif order == "secondary":
            secondary_weight += w
    return primary_weight, secondary_weight


def _set_piece_goal_uplift_multiplier(
    con: duckdb.DuckDBPyConnection, player_uid: str, asof: datetime, set_piece_params_version: int,
    decay_params_version: int = 1, fact_multiplier_params_version: int = 1,
) -> float:
    """1.0 (no-op) unless asof-visible set_piece_order_override evidence, weighed by
    effective_weight, favors PRIMARY or SECONDARY penalty duty for this player -- a versioned
    multiplicative uplift (primary) or demotion (secondary) applied to e_goals. decay/fact-
    multiplier params default to v1 (the standard "no confirmed recalibration yet" fallback
    used throughout this project) so every pre-existing caller keeps working unchanged while
    still genuinely routing through real evidence weighting, not a flat on/off switch."""
    primary_w, secondary_w = _set_piece_duty_evidence(
        con, player_uid, asof, ("penalt",), decay_params_version, fact_multiplier_params_version,
    )
    if primary_w > secondary_w and primary_w > 0.0:
        multiplier, _ = params_mod.resolve_param(
            con, "set_piece_evidence_params", "penalty_taker_goal_rate_multiplier", set_piece_params_version,
        )
        return multiplier
    if secondary_w > primary_w and secondary_w > 0.0:
        try:
            multiplier, _ = params_mod.resolve_param(
                con, "set_piece_evidence_params", "secondary_penalty_taker_goal_rate_multiplier", set_piece_params_version,
            )
        except params_mod.ParamNotFoundError:
            return 1.0
        return multiplier
    return 1.0  # no evidence either way, or a genuine tie -- don't guess a direction


def _set_piece_assist_uplift_multiplier(
    con: duckdb.DuckDBPyConnection, player_uid: str, asof: datetime, set_piece_params_version: int,
    decay_params_version: int = 1, fact_multiplier_params_version: int = 1,
) -> float:
    """Same evidence-strength-decided shape as _set_piece_goal_uplift_multiplier above, but for
    free-kick/corner delivery duty and applied to e_assists, not e_goals -- new capability, this
    duty was previously ingested and read into `duty` but never consumed by anything."""
    primary_w, secondary_w = _set_piece_duty_evidence(
        con, player_uid, asof, ("free-kick", "corner"), decay_params_version, fact_multiplier_params_version,
    )
    if primary_w > secondary_w and primary_w > 0.0:
        try:
            multiplier, _ = params_mod.resolve_param(
                con, "set_piece_evidence_params", "set_piece_deliverer_assist_rate_multiplier", set_piece_params_version,
            )
        except params_mod.ParamNotFoundError:
            return 1.0
        return multiplier
    if secondary_w > primary_w and secondary_w > 0.0:
        try:
            multiplier, _ = params_mod.resolve_param(
                con, "set_piece_evidence_params", "secondary_set_piece_deliverer_assist_rate_multiplier", set_piece_params_version,
            )
        except params_mod.ParamNotFoundError:
            return 1.0
        return multiplier
    return 1.0


# ============================================================
# A2: role-shift adjustment -- a bounded, explainable multiplier on attacking output when
# 18_Predicted XI Database evidence says a player is expected to play a fixture-specific role
# more/less advanced than their FPL-registered position. Registered `position` still decides
# every points-category bucket (clean_sheet_points, defcon_threshold, etc. -- FPL scores off
# registered position, not matchday role, so this deliberately never touches those), but a
# genuine role-shift ("this Defender is playing as a auxiliary wing-back/winger this week") is
# real evidence about attacking involvement that the season-long historical xG/xA rate alone
# can't see. Consumes evidence_blend.blend_categorical() -- previously ingested, dead code
# outside its own tests -- as its first real caller.
# ============================================================

_POSITION_RANK = {"Goalkeeper": 0, "Defender": 1, "Midfielder": 2, "Forward": 3}


def _role_shift_multiplier(
    con: duckdb.DuckDBPyConnection, player_uid: str, registered_position: str, asof: datetime,
    decay_params_version: int, fact_multiplier_params_version: int, role_shift_params_version: int,
) -> float:
    """1.0 (no-op) unless asof-visible predicted_xi evidence assigns this player an exp_position
    on the opposite side of the attacking-order hierarchy (GK < DEF < MID < FWD) from their
    registered position. blend_categorical's weighted distribution over exp_position values
    handles multiple/conflicting source claims the same way every other evidence-blended signal
    in this project does; an exp_position string outside the four canonical positions (e.g. a
    free-text "Right-back/Wing-back") is silently excluded from the weighted sum rather than
    guessed at -- understates confidence slightly rather than risking a wrong-direction shift."""
    if registered_position not in _POSITION_RANK:
        return 1.0
    dist = eb.blend_categorical(
        con, "player", player_uid, "predicted_xi", "exp_position", asof,
        decay_params_version, fact_multiplier_params_version,
    )
    if not dist:
        return 1.0
    registered_rank = _POSITION_RANK[registered_position]
    expected_rank_shift = sum(
        weight * (_POSITION_RANK[pos] - registered_rank)
        for pos, weight in dist.items() if pos in _POSITION_RANK
    )
    if expected_rank_shift == 0.0:
        return 1.0
    try:
        per_rank, _ = params_mod.resolve_param(con, "role_shift_params", "per_rank_multiplier_step", role_shift_params_version)
        cap, _ = params_mod.resolve_param(con, "role_shift_params", "max_multiplier_delta", role_shift_params_version)
    except params_mod.ParamNotFoundError:
        return 1.0
    delta = max(-cap, min(cap, expected_rank_shift * per_rank))
    return 1.0 + delta


def compute_player_fixture_components(
    con: duckdb.DuckDBPyConnection, player_uid: str, position: str, team_uid: str, match_id: str,
    p_0: float, p_1_59: float, p_60plus: float,
    ts_model_version: int, scoring_params_version: int, bps_params_version: int,
    season_priority: list[str], mean_minutes: dict,
    *, asof: datetime | None = None, set_piece_params_version: int | None = None,
    decay_params_version: int | None = None, fact_multiplier_params_version: int | None = None,
    role_shift_params_version: int | None = None,
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
    if asof is not None and set_piece_params_version is not None:
        _sp_decay_v = decay_params_version if decay_params_version is not None else 1
        _sp_fact_v = fact_multiplier_params_version if fact_multiplier_params_version is not None else 1
        e_goals *= _set_piece_goal_uplift_multiplier(con, player_uid, asof, set_piece_params_version, _sp_decay_v, _sp_fact_v)
        e_assists *= _set_piece_assist_uplift_multiplier(con, player_uid, asof, set_piece_params_version, _sp_decay_v, _sp_fact_v)
    if (
        asof is not None and decay_params_version is not None
        and fact_multiplier_params_version is not None and role_shift_params_version is not None
    ):
        role_multiplier = _role_shift_multiplier(
            con, player_uid, position, asof, decay_params_version, fact_multiplier_params_version, role_shift_params_version,
        )
        e_goals *= role_multiplier
        e_assists *= role_multiplier
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
    # Real FPL rule: defenders clear a CBIT threshold (clearances+blocks+interceptions+tackles
    # only, no recoveries) at 10; midfielders/forwards clear a CBIRT threshold (+recoveries) at
    # 12 -- defcon_threshold above already encodes this split (10 vs 12), the rate composition
    # must match it. Recoveries still feed mu/BPS for every position a few lines down (a real,
    # separate FPL scoring category unrelated to the defensive-contribution threshold), so this
    # only narrows defcon_rate, not e_recoveries itself.
    ep_defcon = 0.0
    if position in ("Defender", "Midfielder", "Forward"):
        own_defcon_rate = def_rates["cbi_per_90"]
        if position != "Defender":
            own_defcon_rate += def_rates["recoveries_per_90"]
        defcon_rate = own_defcon_rate * e_min_played / 90.0
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
    set_piece_params_version: int | None = None,
    decay_params_version: int | None = None, fact_multiplier_params_version: int | None = None,
    role_shift_params_version: int | None = None,
) -> int:
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
             scoring_matrix_params_version, bps_params_version, bps_tau_params_version,
             set_piece_params_version, decay_params_version, fact_multiplier_params_version,
             role_shift_params_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        RETURNING model_version
        """,
        [calibration_asof_date, target_season, ts_model_version, mm_model_version,
         scoring_params_version, bps_params_version, tau_params_version,
         set_piece_params_version, decay_params_version, fact_multiplier_params_version,
         role_shift_params_version],
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
                    decay_params_version=decay_params_version,
                    fact_multiplier_params_version=fact_multiplier_params_version,
                    role_shift_params_version=role_shift_params_version,
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
    {"raw_stat": "predicted_xi claim: claim_value_numeric (start_conf)", "feeds": ["minutes_model P(start) shift (compute_logit_adjustment)"],
     "intentional_dual_use": False,
     "note": "consumed entirely in minutes_model.py, never read again here -- see the exp_position row below for why this is a genuinely different evidence dimension off the same claim, not the same evidence counted twice"},
    {"raw_stat": "predicted_xi claim: claim_value['exp_position']", "feeds": ["e_goals (role-shift multiplier)", "e_assists (role-shift multiplier)", "expected_bps (via e_goals/e_assists)"],
     "intentional_dual_use": True,
     "note": "A2: a genuinely different question (what ROLE a starting player plays) than the same claim's numeric start_conf (WHETHER they start, consumed only by minutes_model -- row above). Registered position, not exp_position, still decides every points-category bucket (clean_sheet_points, defcon_threshold, ...) -- this only shades the attacking-rate magnitude, bounded and capped by role_shift_params."},
    {"raw_stat": "set_piece_order_override claim (penalty duty)", "feeds": ["ep_goals (via e_goals multiplier)", "expected_bps (via e_goals)"],
     "intentional_dual_use": True,
     "note": "goal points and goal BPS are separate real FPL mechanisms off the same (evidence-adjusted) e_goals -- same reasoning as the plain expected_goals_per_90 row above, just with an evidence multiplier applied first."},
    {"raw_stat": "set_piece_order_override claim (free-kick/corner duty)", "feeds": ["ep_assists (via e_assists multiplier)", "expected_bps (via e_assists)"],
     "intentional_dual_use": True,
     "note": "A3: same shape as the penalty-duty row above but for e_assists -- delivering a set piece is an assist-creation signal, deliberately never applied to e_goals."},
]

_NOT_MODELED_FOR_LACK_OF_RECONCILED_DATA = [
    "cards", "own_goal", "penalty_save", "penalty_miss",
    "passing/crossing/key-pass BPS components", "goalline_clearance", "winning_goal",
]


def non_double_counting_audit() -> list[dict]:
    return NON_DOUBLE_COUNTING_AUDIT


# ============================================================
# A4: M9 adapter -- qualitative-adjustment provenance trail. Extends the same "raw_stat ->
# feeds" transparency NON_DOUBLE_COUNTING_AUDIT documents at the audit-table level down to
# "for this player, which specific evidence_claims rows moved the number, by how much
# confidence/reliability, and which side lost" -- mirroring minutes_model.
# explain_player_adjustment()'s shape (source, confidence, reliability tier, information_type,
# included vs merely considered).
#
# Deliberately calls the real _role_shift_multiplier()/_set_piece_goal_uplift_multiplier()/
# _set_piece_assist_uplift_multiplier() functions for the reported multiplier itself, rather
# than re-deriving it from the per-claim rows below: blend_categorical's weighted-distribution
# normalization is opaque from outside without recomputing it, and re-deriving that number by
# hand here would risk a provenance trail that quietly drifts from what was actually applied --
# worse than no explain function at all. The per-claim rows are for "what evidence contributed"
# transparency; the multiplier is the actual multiplier, from the actual function.
# ============================================================

def explain_qualitative_adjustment(con: duckdb.DuckDBPyConnection, ep_model_version: int, player_uid: str) -> dict:
    """Returns {} when this ep_model_version ran with every qualitative-adjustment param left
    unset (a legitimate opted-out run, not a missing player) -- same "distinguish absence of
    the feature from absence of data" shape used elsewhere in this project."""
    run_row = con.execute(
        "SELECT calibration_asof_date, set_piece_params_version, decay_params_version, "
        "fact_multiplier_params_version, role_shift_params_version FROM ep_model_versions WHERE model_version = ?",
        [ep_model_version],
    ).fetchone()
    if not run_row:
        return {}
    (calibration_asof_date, set_piece_params_version, decay_params_version,
     fact_multiplier_params_version, role_shift_params_version) = run_row
    if all(v is None for v in (set_piece_params_version, decay_params_version, fact_multiplier_params_version, role_shift_params_version)):
        return {}
    asof = datetime.combine(calibration_asof_date, datetime.max.time(), tzinfo=timezone.utc)
    decay_v = decay_params_version if decay_params_version is not None else 1
    fact_v = fact_multiplier_params_version if fact_multiplier_params_version is not None else 1

    position_row = con.execute("SELECT position FROM dim_player WHERE player_uid = ?", [player_uid]).fetchone()
    registered_position = position_row[0] if position_row else None

    sources_by_id = {r[0]: (r[1], r[2]) for r in con.execute("SELECT source_id, source_name, source_type FROM sources").fetchall()}

    def _source_info(source_id):
        row = sources_by_id.get(source_id)
        return {"source_name": row[0], "source_type": row[1]} if row else {"source_name": None, "source_type": None}

    def _claim_base(c):
        return {
            "claim_id": c["claim_id"], "claim_type": c["claim_type"], **_source_info(c["source_id"]),
            "information_type": c["information_type"], "confidence": c["confidence"],
            "reliability_score": c["source_reliability_score"], "observed_date": c["observed_date"],
            "raw_text": c["raw_text"],
        }

    result = {
        "role_shift": {"applied": False, "multiplier": 1.0, "registered_position": registered_position, "claims": []},
        "set_piece_goal": {"applied": False, "multiplier": 1.0, "claims": []},
        "set_piece_assist": {"applied": False, "multiplier": 1.0, "claims": []},
    }

    # ---- role shift (A2) ----
    if role_shift_params_version is not None and registered_position in _POSITION_RANK:
        multiplier = _role_shift_multiplier(
            con, player_uid, registered_position, asof, decay_v, fact_v, role_shift_params_version,
        )
        claims = snapshot_mod.get_claims_asof(
            con, asof, subject_entity_type="player", subject_entity_id=player_uid, claim_type="predicted_xi",
        ).to_dict("records")
        rows = []
        for c in claims:
            payload = json.loads(c["claim_value"]) if c["claim_value"] else {}
            exp_position = payload.get("exp_position")
            base = {**_claim_base(c), "exp_position": exp_position}
            if exp_position not in _POSITION_RANK:
                rows.append({**base, "included": False, "exclusion_reason": "exp_position missing or not one of the four canonical positions"})
                continue
            w = eb.effective_weight(con, c, asof, decay_v, fact_v)
            rows.append({
                **base, "included": True, "exclusion_reason": None, "weight": w,
                "rank_delta": _POSITION_RANK[exp_position] - _POSITION_RANK[registered_position],
            })
        result["role_shift"] = {
            "applied": multiplier != 1.0, "multiplier": multiplier,
            "registered_position": registered_position, "claims": rows,
        }

    # ---- set-piece goal/assist (A3) ----
    for key, duty_keywords, uplift_fn in (
        ("set_piece_goal", ("penalt",), _set_piece_goal_uplift_multiplier),
        ("set_piece_assist", ("free-kick", "corner"), _set_piece_assist_uplift_multiplier),
    ):
        if set_piece_params_version is None:
            continue
        multiplier = uplift_fn(con, player_uid, asof, set_piece_params_version, decay_v, fact_v)
        matched = _set_piece_duty_claims(con, player_uid, asof, duty_keywords)
        # arbitrary when multiplier == 1.0 (no net directional evidence) -- doesn't matter,
        # `included` below is forced False in that case regardless of this value.
        winning_order = "primary" if multiplier >= 1.0 else "secondary"
        rows = []
        for c in matched:
            order = c["_payload"].get("order")
            w = eb.effective_weight(con, c, asof, decay_v, fact_v)
            included = multiplier != 1.0 and order == winning_order
            rows.append({
                **_claim_base(c), "duty": c["_payload"].get("duty"), "order": order, "weight": w,
                "included": included,
                "exclusion_reason": None if included else "outweighed by stronger opposing-order evidence" if multiplier != 1.0 else "no net directional evidence",
            })
        result[key] = {"applied": multiplier != 1.0, "multiplier": multiplier, "claims": rows}

    return result


# ============================================================
# M9 adapter -- category-level EP breakdown, display-ready shape
# ============================================================

def explain_player_ep(con: duckdb.DuckDBPyConnection, ep_model_version: int, player_uid: str) -> dict | None:
    """M9's category-level EP breakdown section: "not a single blended number, so a human can
    see e.g. this defender's value is mostly DefCon-driven." Pure read against ep_outputs,
    labeled by category rather than raw column name -- no new computation, except for the
    qualitative_adjustments key (A4), which extends the same provenance-trail idea
    minutes_model.explain_player_adjustment() already established for minutes into EP's own
    evidence-driven multipliers. Returns None if the player has no fixture at this
    model_version (a legitimate blank gameweek, not an error)."""
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
        "qualitative_adjustments": explain_qualitative_adjustment(con, ep_model_version, player_uid),
    }
