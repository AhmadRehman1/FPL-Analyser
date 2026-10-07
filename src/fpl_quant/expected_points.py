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

import functools
import json
import math
from datetime import date, datetime, timezone

import duckdb
from scipy.stats import poisson

from . import minutes_model as minutes_mod
from . import params as params_mod
from . import reconcile as reconcile_mod
from . import season_rules
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
    # GK saves scale ~1:1 with shots faced (= opponent attack); DefCon actions correlate with
    # being under pressure but far more loosely, so damped. Both invented v1, M7-recalibratable.
    params_mod.write_param(con, "fixture_strength_params", 1, "2026-08-10", "save_sensitivity", value_numeric=1.0)
    params_mod.write_param(con, "fixture_strength_params", 1, "2026-08-10", "defcon_sensitivity", value_numeric=0.5)

    # Invented v1 default (was a bare module constant, RATE_SHRINKAGE_K_MINUTES, until this --
    # see that name's own docstring history), now a real recalibratable param so M7 can actually
    # tune it: the walk-forward's segment_calibration shows ep_total_calibration_mean_resid
    # growing monotonically with price (£9.0m+ under-predicted by +0.84 pts/GW vs <£5.0m at
    # -0.24), i.e. exactly the "premiums shrunk hardest toward the position average" failure
    # this constant was already flagged (but never wired) for -- see
    # docs/plans/2026-09_ep_attacker_defender_imbalance.md, Lead B.
    params_mod.write_param(con, "rate_shrinkage_params", 1, "2026-08-10", "k_minutes", value_numeric=RATE_SHRINKAGE_K_MINUTES)


def _sm(con, key, params_version, position=None):
    dims = {"position": position} if position else None
    v, _ = params_mod.resolve_param(con, "base_scoring_matrix", key, params_version, dimensions=dims)
    return v


def _bp(con, key, params_version, position=None, season=None):
    """A bps_formula_params weight, or `season`'s own value where that season's BPS differed
    from the configured version (season_rules.bps_override())."""
    override = season_rules.bps_override(season, key)
    if override is not None:
        return override
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

# Invented v1 default -- now also the "rate_shrinkage_params"/"k_minutes" v1 row (seed_v1_params()
# above), so M7 can recalibrate it. This module constant remains the fallback used whenever no
# rate_shrinkage_params_version is supplied (every pre-existing caller), so the two stay in sync
# by construction: DEFAULT_RATE_SHRINKAGE_K_MINUTES is what v1 is seeded to equal.
DEFAULT_RATE_SHRINKAGE_K_MINUTES = 450.0
RATE_SHRINKAGE_K_MINUTES = DEFAULT_RATE_SHRINKAGE_K_MINUTES  # back-compat alias; prefer the DEFAULT_ name in new code


def _resolve_shrinkage_k(con: duckdb.DuckDBPyConnection, rate_shrinkage_params_version: int | None) -> float:
    """None means "no version pinned" -- every caller before this param existed, and still the
    default for run()/compute_player_fixture_components() -- so behavior for them is byte-for-byte
    unchanged: the same hardcoded constant as before. A real version resolves the versioned param
    instead, which is what backtest.refit_rate_shrinkage() pins to search candidate k values."""
    if rate_shrinkage_params_version is None:
        return DEFAULT_RATE_SHRINKAGE_K_MINUTES
    value, _ = params_mod.resolve_param(con, "rate_shrinkage_params", "k_minutes", rate_shrinkage_params_version)
    return value


def _shrink_rate(own_rate: float, sample_minutes: float, position_avg_rate: float, k: float = DEFAULT_RATE_SHRINKAGE_K_MINUTES) -> float:
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


def _player_rate_pool(
    con: duckdb.DuckDBPyConnection, player_uid: str, season_priority: list[str],
    season_weights: dict[str, float] | None = None,
) -> dict:
    """Pools each lookback season's latest (most complete cumulative) row, weighted by that
    season's own total minutes -- not a single cherry-picked season. season_weights (the opt-in
    rate prior's recency, season_weights_for()) also scales each season's minutes and returns,
    so sample_minutes is the weighted evidence; None weights every season 1.

    Two source schemas: 2025-26+ publishes a season-total `minutes` + `expected_goals`
    (cumulative), while 2024-2025's snapshot publishes `expected_goals_per_90` directly but no
    season-total minutes or xG. The old code required `minutes`, so it silently dropped ALL of
    2024-2025 -- halving the attacking-rate sample for every player and shrinking premiums
    (high own rate, small sample) hardest toward the position average, exactly the EP
    compression the DefCon rate (which reads fact_player_match_stats and DOES see 2024-25) does
    not suffer. This now recovers 2024-25 from the per-90 rate + match-grain minutes."""
    total_minutes = total_goals = total_assists = total_saves_weighted = saves_minutes = 0.0
    for season in season_priority:
        w = season_weights.get(season, 1.0) if season_weights else 1.0
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
            total_minutes += w * minutes
            total_goals += w * (xg or 0.0)
            total_assists += w * (xa or 0.0)
            if saves_p90 is not None:
                total_saves_weighted += saves_p90 * w * minutes
                saves_minutes += w * minutes
        elif xg90 is not None or xa90 is not None:
            mins = _season_match_minutes(con, player_uid, season)
            if mins <= 0:
                continue
            total_minutes += w * mins
            total_goals += w * ((xg90 or 0.0) / 90.0 * mins)
            total_assists += w * ((xa90 or 0.0) / 90.0 * mins)
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


def new_memo() -> dict:
    """A cache for one asof view of the data, shared by every run() / uncertainty.run() /
    monte_carlo.run() call made against it (docs/reports/2026-10_open_issues.md, issue 3).

    A planner call projects five or more gameweeks from the same asof cutoff, and each one used
    to rebuild every player's rate pool, the position anchors and the fixture lambdas from
    scratch: player_rates_shrunk() ran ~10k times for two season-simulation gameweeks. None of
    those depend on the target gameweek, only on the data visible at the cutoff, so one memo
    serves the whole horizon.

    Only share a memo between calls that see the same data: the same asof_scope() block, with
    no TEMP TABLE shadow (scenario.py, decision_engine) switched on or off in between. The memo
    is deliberately an explicit argument rather than a module-level cache, which once returned
    another run's values (see _league_defence_and_home_adv())."""
    return {}


def _memo_get(memo: dict | None, key: tuple, compute):
    """compute() once per key when a memo is given, on every call when it isn't."""
    if memo is None:
        return compute()
    if key not in memo:
        memo[key] = compute()
    return memo[key]


# ============================================================
# rate prior (opt-in experiment, rate_prior_params; None = the rates above, unchanged). Early in
# 2026-27 the model ranked the season's template picks 100th-500th: a player new to the league
# has a few hundred minutes of own rate, shrunk with k_minutes toward the flat position average,
# and every premium is shrunk toward that same average (the walk-forward under-predicts 9.0+ by
# ~0.8 pts/GW). Three knobs, one bundle version, so each walk-forward arm is one flag set:
#   price_anchor           1 shrinks goals/assists toward a per-position rate rising with price
#                          (price_anchor_fits()) instead of the position average; saves keep the
#                          average (cheap keepers on weak sides face more shots)
#   current_season_weight  multiplies the newest lookback season's minutes and returns in the
#                          player's pool, so this season's evidence counts for more
#   season_decay           the weight of each season further back (0.5: last season half, the
#                          one before a quarter)
# ============================================================

RATE_PRIOR_KEYS = ("price_anchor", "current_season_weight", "season_decay")
# A position needs this much pooled history before it gets a price anchor (else the average).
PRICE_ANCHOR_MIN_MINUTES = 5000.0


def resolve_rate_prior(con: duckdb.DuckDBPyConnection, rate_prior_params_version: int | None) -> dict | None:
    """The rate_prior_params bundle as {key: value}, or None when off."""
    if rate_prior_params_version is None:
        return None
    values = {
        key: params_mod.resolve_param(con, "rate_prior_params", key, rate_prior_params_version)[0]
        for key in RATE_PRIOR_KEYS
    }
    return {
        "price_anchor": bool(values["price_anchor"]),
        "current_season_weight": float(values["current_season_weight"]),
        "season_decay": float(values["season_decay"]),
    }


def season_weights_for(season_priority: list[str], rate_prior: dict | None) -> dict[str, float] | None:
    """{season: weight} for _player_rate_pool(): the newest season current_season_weight, each
    one further back season_decay times the last. None when the prior is off or every weight
    is 1."""
    if rate_prior is None:
        return None
    newest_first = sorted(set(season_priority), reverse=True)
    weights = {
        season: rate_prior["current_season_weight"] if i == 0 else rate_prior["season_decay"] ** i
        for i, season in enumerate(newest_first)
    }
    return None if all(w == 1.0 for w in weights.values()) else weights


def price_anchor_fits(
    con: duckdb.DuckDBPyConnection, position: str, season_priority: list[str], *, memo: dict | None = None,
) -> dict[str, tuple[float, float, float]] | None:
    """{rate key: (intercept, slope, pivot)} for goals and assists per 90 at `position`: a
    minutes-weighted least-squares line of each player's pooled own rate (_player_rate_pool(),
    unweighted by recency) on his price, centred on the weighted mean price (the pivot), so the
    intercept is the weighted mean rate. Slopes below 0 are set to 0. None when fewer than
    PRICE_ANCHOR_MIN_MINUTES of priced history exist (the caller keeps the position average).
    Prices are minutes_model.latest_price_by_player(), the price as of an asof_scope deadline."""
    seasons = list(season_priority)
    placeholders = ",".join(["?"] * len(seasons))
    uids = [r[0] for r in con.execute(
        f"""
        SELECT DISTINCT fps.player_uid FROM fact_player_season_stats fps
        JOIN dim_player dp ON dp.player_uid = fps.player_uid
        WHERE dp.position = ? AND fps.season IN ({placeholders}) ORDER BY 1
        """,
        [position, *seasons],
    ).fetchall()]
    prices = _memo_get(memo, ("latest_prices",), lambda: minutes_mod.latest_price_by_player(con))
    rows = []
    for uid in uids:
        pool = _memo_get(memo, ("rate_pool", uid, tuple(seasons)), lambda uid=uid: _player_rate_pool(con, uid, seasons))
        if pool["sample_minutes"] > 0 and uid in prices:
            rows.append((prices[uid], pool["sample_minutes"], pool["expected_goals_per_90"], pool["expected_assists_per_90"]))
    total = sum(r[1] for r in rows)
    if total < PRICE_ANCHOR_MIN_MINUTES:
        return None
    pivot = sum(r[0] * r[1] for r in rows) / total
    var = sum(r[1] * (r[0] - pivot) ** 2 for r in rows)
    fits = {}
    for i, key in ((2, "expected_goals_per_90"), (3, "expected_assists_per_90")):
        mean = sum(r[i] * r[1] for r in rows) / total
        cov = sum(r[1] * (r[0] - pivot) * (r[i] - mean) for r in rows)
        fits[key] = (mean, max(cov / var, 0.0) if var > 0 else 0.0, pivot)
    return fits


def player_rates_shrunk(
    con: duckdb.DuckDBPyConnection, player_uid: str, position: str, season_priority: list[str],
    rate_shrinkage_params_version: int | None = None, finishing_prior_xg: float | None = None,
    *, memo: dict | None = None, rate_prior: dict | None = None,
) -> dict:
    seasons = list(season_priority)
    weights = season_weights_for(seasons, rate_prior)
    if weights is None:
        own = _memo_get(memo, ("rate_pool", player_uid, tuple(seasons)), lambda: _player_rate_pool(con, player_uid, seasons))
    else:
        own = _memo_get(
            memo, ("rate_pool", player_uid, tuple(seasons), tuple(sorted(weights.items()))),
            lambda: _player_rate_pool(con, player_uid, seasons, weights),
        )
    pos_avg = _memo_get(memo, ("position_rates", position, tuple(seasons)), lambda: _position_average_rates(con, position, seasons))
    anchor = dict(pos_avg)
    if rate_prior is not None and rate_prior["price_anchor"]:
        fits = _memo_get(memo, ("price_anchor", position, tuple(seasons)), lambda: price_anchor_fits(con, position, seasons, memo=memo))
        price = _memo_get(memo, ("latest_prices",), lambda: minutes_mod.latest_price_by_player(con)).get(player_uid)
        if fits is not None and price is not None:
            for key, (intercept, slope, pivot) in fits.items():
                anchor[key] = max(intercept + slope * (price - pivot), 0.0)
    k = _resolve_shrinkage_k(con, rate_shrinkage_params_version)
    rates = {
        key: _shrink_rate(own[key], own["sample_minutes"], anchor[key], k=k)
        for key in ("expected_goals_per_90", "expected_assists_per_90", "saves_per_90")
    }
    if finishing_prior_xg is not None:
        goal_ratio, assist_ratio = _memo_get(
            memo, ("finishing", player_uid, tuple(seasons), finishing_prior_xg),
            lambda: finishing_ratios(con, player_uid, seasons, finishing_prior_xg),
        )
        rates["expected_goals_per_90"] *= goal_ratio
        rates["expected_assists_per_90"] *= assist_ratio
    return rates


def defcon_in_force(season: str | None) -> bool:
    """True when `season`'s scoring awards DefCon points (2 points for reaching a CBIT / CBIRT
    threshold), which only exist from 2025-26: a 2024-25 walk-forward step that predicted them
    was scoring players on points that season could not award. Read from season_rules, the one
    table of rules by season; None (no season known) keeps the current rules."""
    return season_rules.rules_for(season).defcon


MAX_FINISHING_RATIO = 2.0


def finishing_ratios(
    con: duckdb.DuckDBPyConnection, player_uid: str, season_priority: list[str], prior_xg: float,
) -> tuple[float, float]:
    """(goals ratio, assists ratio) for one player: (goals + prior) / (xG + prior) and
    (assists + prior) / (xA + prior), pooled over the lookback seasons. The walk-forward's
    clean baseline under-predicts 9.0+ players by ~0.8 pts/GW (goals, assists and bonus all
    short) and over-predicts <5.0 players: xG/xA rates miss persistent finishing and
    assist skill. prior_xg pulls a small sample back to 1; the ratio is capped to
    [1/MAX_FINISHING_RATIO, MAX_FINISHING_RATIO].

    Same two source schemas as _player_rate_pool(): 2025-26+ rows carry season-total goals,
    xG, assists and xA; 2024-2025's snapshot carries only per-90 xG/xA, so that season's
    totals are rebuilt from the per-90 rates x match-grain minutes, with goals and assists
    summed from fact_player_match_stats (match-feed assists, close to FPL's)."""
    goals = xg = assists = xa = 0.0
    for season in season_priority:
        row = con.execute(
            "SELECT goals_scored, expected_goals, assists, expected_assists, "
            "expected_goals_per_90, expected_assists_per_90 "
            "FROM fact_player_season_stats WHERE player_uid = ? AND season = ? ORDER BY gw DESC LIMIT 1",
            [player_uid, season],
        ).fetchone()
        if not row:
            continue
        g, season_xg, a, season_xa, xg90, xa90 = row
        if g is not None and season_xg is not None:
            goals += g
            xg += season_xg
            assists += a or 0
            xa += season_xa or 0.0
        elif xg90 is not None or xa90 is not None:
            match = con.execute(
                "SELECT sum(minutes_played), sum(coalesce(goals, 0)), sum(coalesce(assists, 0)) "
                "FROM fact_player_match_stats WHERE player_uid = ? AND season = ?",
                [player_uid, season],
            ).fetchone()
            mins = float(match[0] or 0.0) if match else 0.0
            if mins <= 0:
                continue
            goals += float(match[1] or 0)
            assists += float(match[2] or 0)
            xg += (xg90 or 0.0) / 90.0 * mins
            xa += (xa90 or 0.0) / 90.0 * mins

    def ratio(actual: float, expected: float) -> float:
        r = (actual + prior_xg) / (expected + prior_xg)
        return min(max(r, 1.0 / MAX_FINISHING_RATIO), MAX_FINISHING_RATIO)

    return ratio(goals, xg), ratio(assists, xa)


def finishing_ratio_report(
    con: duckdb.DuckDBPyConnection, season_priority: list[str], prior_xg: float, min_expected: float = 1.0,
) -> dict:
    """How finishing_ratios() spreads over the players with at least min_expected xG (or xA)
    in the window: counts pinned at MAX_FINISHING_RATIO and at its floor, quantiles, and who is
    at the cap. The finishing-prior arm changed sign when 2024-25 joined the window
    (docs/reports/2026-10_open_issues.md, issue 5); comparing windows here shows whether the
    ratios themselves move, and how many players the hard clamp decides."""
    placeholders = ",".join(["?"] * len(season_priority))
    uids = [r[0] for r in con.execute(
        f"SELECT DISTINCT player_uid FROM fact_player_season_stats WHERE season IN ({placeholders}) ORDER BY 1",
        list(season_priority),
    ).fetchall()]
    cap, floor = MAX_FINISHING_RATIO, 1.0 / MAX_FINISHING_RATIO
    out: dict = {"seasons": list(season_priority), "prior_xg": prior_xg}
    ratios: dict[str, list[tuple[str, float]]] = {"goals": [], "assists": []}
    for uid in uids:
        # finishing_ratios() pools goals/xG over the window; recover xG to apply min_expected
        xg = xa = 0.0
        for season in season_priority:
            row = con.execute(
                "SELECT expected_goals, expected_assists, expected_goals_per_90, expected_assists_per_90, minutes "
                "FROM fact_player_season_stats WHERE player_uid = ? AND season = ? ORDER BY gw DESC LIMIT 1",
                [uid, season],
            ).fetchone()
            if not row:
                continue
            season_xg, season_xa, xg90, xa90, minutes = row
            if season_xg is not None and minutes:
                xg += season_xg
                xa += season_xa or 0.0
            elif xg90 is not None or xa90 is not None:
                mins = _season_match_minutes(con, uid, season)
                xg += (xg90 or 0.0) / 90.0 * mins
                xa += (xa90 or 0.0) / 90.0 * mins
        goal_ratio, assist_ratio = finishing_ratios(con, uid, list(season_priority), prior_xg)
        if xg >= min_expected:
            ratios["goals"].append((uid, goal_ratio))
        if xa >= min_expected:
            ratios["assists"].append((uid, assist_ratio))
    for kind, values in ratios.items():
        vals = sorted(v for _uid, v in values)
        n = len(vals)
        out[kind] = {
            "n": n,
            "at_cap": sum(1 for v in vals if v >= cap - 1e-12),
            "at_floor": sum(1 for v in vals if v <= floor + 1e-12),
            "quantiles": {q: vals[min(n - 1, int(q * n))] for q in (0.05, 0.25, 0.5, 0.75, 0.95)} if n else {},
            "capped_players": sorted(uid for uid, v in values if v >= cap - 1e-12),
        }
    return out


def resolve_finishing_prior(con: duckdb.DuckDBPyConnection, finishing_skill_params_version: int | None) -> float | None:
    """None (the default everywhere) keeps xG/xA rates as they are."""
    if finishing_skill_params_version is None:
        return None
    value, _ = params_mod.resolve_param(con, "finishing_skill_params", "prior_xg", finishing_skill_params_version)
    return value


# ============================================================
# BPS calibration (opt-in): the bonus points a player earns that the estimate below can't see.
#
# The estimate scores playing time, goals, assists, CBI, recoveries, saves and goals conceded
# only -- not the clean-sheet bonus, chances created, dribbles, passing, winning goals or the
# negatives. Premium attackers earn most of what's missing, so they get more bonus than their
# estimated BPS implies: the walk-forward under-predicts the 9.0+ band's bonus-and-other points
# by ~0.27 per player-gameweek (docs/reports/2026-10_open_issues.md, issue 6).
#
# fact_player_season_stats carries each player's real season BPS. Taking away what the
# estimate's own terms would have given for the matches he actually played (at that season's
# weights) leaves the BPS the estimate misses; per 90 minutes and shrunk toward his position's
# average, it is added to his expected BPS. Seasons whose match-grain minutes don't match the
# season total within 10% (missing match rows) are skipped, as are seasons with no BPS total
# (2024-25's snapshot has none).
# ============================================================

BPS_CALIBRATION_MINUTES_TOLERANCE = 0.10


def _bps_residual_table(
    con: duckdb.DuckDBPyConnection, season_priority: list[str], bps_params_version: int,
) -> dict:
    """{"player": {uid: (residual_bps, minutes)}, "position": {position: residual per 90}} over
    the lookback seasons: residual = real season BPS - the estimate's terms on the same matches."""
    placeholders = ",".join(["?"] * len(season_priority))
    rows = con.execute(
        f"""
        WITH season_total AS (
            SELECT player_uid, season, bps, minutes FROM fact_player_season_stats
            WHERE season IN ({placeholders}) AND bps IS NOT NULL AND minutes IS NOT NULL
            QUALIFY row_number() OVER (PARTITION BY player_uid, season ORDER BY gw DESC) = 1
        ),
        matches AS (
            SELECT player_uid, season,
                   sum(minutes_played) AS mins,
                   sum(CASE WHEN minutes_played BETWEEN 1 AND 59 THEN 1 ELSE 0 END) AS n_1_59,
                   sum(CASE WHEN minutes_played >= 60 THEN 1 ELSE 0 END) AS n_60,
                   sum(coalesce(goals, 0)) AS goals, sum(coalesce(assists, 0)) AS assists,
                   sum(coalesce(saves, 0)) AS saves,
                   -- the estimate's own conceded term: 2*floor(X/2) per 60+ minute appearance
                   sum(CASE WHEN minutes_played >= 60 THEN 2 * (coalesce(goals_conceded, 0) // 2) ELSE 0 END) AS conceded,
                   sum(coalesce(tackles, 0) + coalesce(clearances, 0) + coalesce(interceptions, 0) + coalesce(blocks, 0)) AS cbi,
                   sum(coalesce(recoveries, 0)) AS recoveries
            FROM fact_player_match_stats WHERE season IN ({placeholders})
            GROUP BY player_uid, season
        )
        SELECT t.player_uid, t.season, dp.position, t.bps, t.minutes, m.mins, m.n_1_59, m.n_60,
               m.goals, m.assists, m.saves, m.conceded, m.cbi, m.recoveries
        FROM season_total t
        JOIN matches m ON m.player_uid = t.player_uid AND m.season = t.season
        JOIN dim_player dp ON dp.player_uid = t.player_uid
        WHERE m.mins > 0
        ORDER BY t.player_uid, t.season
        """,
        [*season_priority, *season_priority],
    ).fetchall()
    player: dict[str, list[float]] = {}
    position_totals: dict[str, list[float]] = {}
    for (uid, season, position, bps, minutes, mins, n_1_59, n_60, goals, assists, saves, conceded, cbi,
         recoveries) in rows:
        if position not in POSITIONS or abs(mins - minutes) > BPS_CALIBRATION_MINUTES_TOLERANCE * max(minutes, 1):
            continue

        def weight(key: str, pos: str | None = None) -> float:
            return _bp(con, key, bps_params_version, pos, season=season)

        modelled = (
            weight("playing_1_60") * n_1_59 + weight("playing_60plus") * n_60
            + weight("goal", position) * goals + weight("assist") * assists
            + cbi / weight("cbi_per_point") + recoveries / weight("recoveries_per_point")
        )
        if position in ("Goalkeeper", "Defender"):
            modelled += weight("goal_conceded_gk_def") * conceded
        if position == "Goalkeeper":
            modelled += weight("save_inside_box") * saves
        residual = bps - modelled
        acc = player.setdefault(uid, [0.0, 0.0])
        acc[0] += residual
        acc[1] += mins
        pos_acc = position_totals.setdefault(position, [0.0, 0.0])
        pos_acc[0] += residual
        pos_acc[1] += mins
    return {
        "player": {uid: (r, m) for uid, (r, m) in player.items()},
        "position": {pos: r / m * 90 for pos, (r, m) in position_totals.items() if m > 0},
    }


def bps_residual_per_90(
    con: duckdb.DuckDBPyConnection, player_uid: str, position: str, season_priority: list[str],
    bps_params_version: int, k_minutes: float, *, memo: dict | None = None,
) -> float:
    """The BPS per 90 minutes this player earns beyond the estimate's own terms, shrunk toward
    his position's average by sample size (the same weight _shrink_rate() uses). 0.0 with no
    calibration data at all for the position."""
    seasons = list(season_priority)
    table = _memo_get(
        memo, ("bps_residuals", tuple(seasons), bps_params_version),
        lambda: _bps_residual_table(con, seasons, bps_params_version),
    )
    position_rate = table["position"].get(position)
    if position_rate is None:
        return 0.0
    residual, minutes = table["player"].get(player_uid, (0.0, 0.0))
    own_rate = residual / minutes * 90 if minutes > 0 else 0.0
    return _shrink_rate(own_rate, minutes, position_rate, k=k_minutes)


def resolve_bps_calibration(con: duckdb.DuckDBPyConnection, bps_calibration_params_version: int | None) -> float | None:
    """The calibration's shrinkage k_minutes, or None (off, the default everywhere)."""
    if bps_calibration_params_version is None:
        return None
    value, _ = params_mod.resolve_param(con, "bps_calibration_params", "k_minutes", bps_calibration_params_version)
    return value


def _defensive_action_rates_per_90(
    con: duckdb.DuckDBPyConnection, player_uid: str, position: str, seasons: list[str],
    rate_shrinkage_params_version: int | None = None, *, memo: dict | None = None,
) -> dict:
    """CBI (tackles+clearances+interceptions+blocks) and recoveries, per 90 minutes, from
    fact_player_match_stats -- the only place these are reconciled at per-match grain.
    Shrunk toward the position average the same way and for the same reason as the goals/
    assists/saves rates above."""
    seasons = list(seasons)
    placeholders = ",".join(["?"] * len(seasons))

    def own_totals():
        return con.execute(
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

    def position_totals():
        return con.execute(
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

    cbi_total, recoveries_total, minutes_total = _memo_get(memo, ("defensive_totals", player_uid, tuple(seasons)), own_totals)
    own_cbi = (cbi_total or 0) / minutes_total * 90 if minutes_total else 0.0
    own_recoveries = (recoveries_total or 0) / minutes_total * 90 if minutes_total else 0.0

    pos_cbi_total, pos_recoveries_total, pos_minutes_total = _memo_get(
        memo, ("defensive_position_totals", position, tuple(seasons)), position_totals,
    )
    pos_avg_cbi = (pos_cbi_total or 0) / pos_minutes_total * 90 if pos_minutes_total else 0.0
    pos_avg_recoveries = (pos_recoveries_total or 0) / pos_minutes_total * 90 if pos_minutes_total else 0.0

    sample_minutes = minutes_total or 0.0
    k = _resolve_shrinkage_k(con, rate_shrinkage_params_version)
    return {
        "cbi_per_90": _shrink_rate(own_cbi, sample_minutes, pos_avg_cbi, k=k),
        "recoveries_per_90": _shrink_rate(own_recoveries, sample_minutes, pos_avg_recoveries, k=k),
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

def _league_defence_and_home_adv(
    con: duckdb.DuckDBPyConnection, ts_model_version: int, *, memo: dict | None = None,
) -> tuple[float, float]:
    """(mean final_defence across the league, home_advantage) for this snapshot set. Two
    indexed single-row lookups -- deliberately NOT memoised on a module global (a stale
    per-model_version cache silently returned another run's/test's values; the cost of just
    re-reading is negligible next to the SCIP solves and Monte Carlo anyway).

    Dixon-Coles centres mean ATTACK at 0 but not mean defence (see team_strength's own design
    note), so the 'average opponent' a player's flat rate is measured against has defence =
    this mean, not 0. (A per-call `memo`, see new_memo(), is fine: it never outlives the run.)"""
    def compute():
        mean_def = con.execute(
            "SELECT avg(final_defence) FROM team_strength_snapshots WHERE model_version = ?",
            [ts_model_version],
        ).fetchone()[0]
        home_adv = con.execute(
            "SELECT home_advantage FROM team_strength_model_versions WHERE model_version = ?",
            [ts_model_version],
        ).fetchone()[0]
        return float(mean_def or 0.0), float(home_adv or 0.0)

    return _memo_get(memo, ("league_defence", ts_model_version), compute)


def _fixture_attack_multiplier(
    con: duckdb.DuckDBPyConnection, team_uid: str, match_id: str, target_season: str,
    ts_model_version: int, fixture_params_version: int, *, memo: dict | None = None,
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

    def compute():
        lambda_for, _lambda_against, _is_home = _fixture_lambdas(con, team_uid, match_id, ts_model_version, memo=memo)
        mean_def, home_adv = _league_defence_and_home_adv(con, ts_model_version, memo=memo)
        own_attack = _team_strength(con, ts_model_version, team_uid, memo)[0]
        ref_lambda = math.exp(own_attack - mean_def + home_adv / 2.0)
        if ref_lambda <= 0:
            return 1.0
        mult = (lambda_for / ref_lambda) ** sensitivity
        return max(0.4, min(2.5, mult))

    return _memo_get(memo, ("attack_mult", team_uid, match_id, ts_model_version, sensitivity), compute)


def _fixture_defensive_multiplier(
    con: duckdb.DuckDBPyConnection, team_uid: str, match_id: str,
    ts_model_version: int, fixture_params_version: int, sensitivity_key: str, *, memo: dict | None = None,
) -> float:
    """lambda_against(this fixture) / lambda_against(this team vs a league-average attack,
    half-away) = exp(opp_attack + adv_opp - home_adv/2) (mean attack is 0 by Dixon-Coles
    centring). How much more/less this team is expected to concede than in an average fixture.

    Scales GK saves (shots faced ~ opponent attacking strength) at sensitivity_key
    "save_sensitivity" (v1 1.0), and DefCon actions -- a back line under pressure makes more
    blocks/clearances/interceptions, but the link is looser than saves-to-shots -- at
    "defcon_sensitivity" (v1 0.5, damped). Same [0.4, 2.5] clip and asof-clean construction as
    _fixture_attack_multiplier(); the goals-conceded term already uses lambda_against directly."""
    sensitivity, _ = params_mod.resolve_param(
        con, "fixture_strength_params", sensitivity_key, fixture_params_version,
    )
    if sensitivity == 0.0:
        return 1.0

    def compute():
        _lf, lambda_against, _is_home = _fixture_lambdas(con, team_uid, match_id, ts_model_version, memo=memo)
        _mean_def, home_adv = _league_defence_and_home_adv(con, ts_model_version, memo=memo)
        own_defence = _team_strength(con, ts_model_version, team_uid, memo)[1]
        ref_lambda = math.exp(-own_defence + home_adv / 2.0)
        if ref_lambda <= 0:
            return 1.0
        mult = (lambda_against / ref_lambda) ** sensitivity
        return max(0.4, min(2.5, mult))

    return _memo_get(memo, ("defensive_mult", team_uid, match_id, ts_model_version, sensitivity), compute)


def _team_strength(con: duckdb.DuckDBPyConnection, ts_model_version: int, team_uid: str, memo: dict | None) -> tuple:
    """(final_attack, final_defence) of one team in one team-strength snapshot."""
    return _memo_get(memo, ("team_strength", ts_model_version, team_uid), lambda: con.execute(
        "SELECT final_attack, final_defence FROM team_strength_snapshots WHERE model_version = ? AND team_uid = ?",
        [ts_model_version, team_uid],
    ).fetchone())


def _fixture_lambdas(
    con: duckdb.DuckDBPyConnection, team_uid: str, match_id: str, ts_model_version: int, *, memo: dict | None = None,
):
    def compute():
        match = con.execute(
            "SELECT home_team_uid, away_team_uid FROM fact_match WHERE match_id = ?", [match_id]
        ).fetchone()
        home_uid, away_uid = match
        is_home = team_uid == home_uid
        opp_uid = away_uid if is_home else home_uid

        home_adv = _memo_get(memo, ("home_advantage", ts_model_version), lambda: con.execute(
            "SELECT home_advantage FROM team_strength_model_versions WHERE model_version = ?", [ts_model_version]
        ).fetchone()[0])

        own_attack, own_defence = _team_strength(con, ts_model_version, team_uid, memo)
        opp_attack, opp_defence = _team_strength(con, ts_model_version, opp_uid, memo)

        adv_own = home_adv if is_home else 0.0
        adv_opp = home_adv if not is_home else 0.0
        lambda_for = math.exp(own_attack - opp_defence + adv_own)
        lambda_against = math.exp(opp_attack - own_defence + adv_opp)
        return lambda_for, lambda_against, is_home

    return _memo_get(memo, ("fixture_lambdas", team_uid, match_id, ts_model_version), compute)


@functools.lru_cache(maxsize=8192)
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
    mean -- Var[bonus] needs the whole distribution, not E[bonus] alone.

    Players are summed in sorted order, not the caller's dict order: the strengths usually come
    from a DISTINCT/JOIN query with no fixed row order, and float sums taken in a different
    order differ in the last bits, which was enough to change the optimizer's squad between
    otherwise identical runs (docs/reports/2026-10_open_issues.md, issue 2)."""
    players = sorted(strengths)
    total = sum(strengths[p] for p in players)
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

def _player_claims(
    con: duckdb.DuckDBPyConnection, asof: datetime, player_uid: str, claim_type: str, memo: dict | None,
) -> list[dict]:
    """The player's claims of one type visible as of `asof`. With a memo, one query fetches
    every player's claims of that type and later calls read from it -- the per-player query
    and its DataFrame round trip ran ~17k times in two season-simulation gameweeks."""
    if memo is None:
        return snapshot_mod.get_claims_asof(
            con, asof, subject_entity_type="player", subject_entity_id=player_uid, claim_type=claim_type,
        ).to_dict("records")

    def by_player():
        out: dict[str, list[dict]] = {}
        records = snapshot_mod.get_claims_asof(
            con, asof, subject_entity_type="player", claim_type=claim_type,
        ).to_dict("records")
        for c in records:
            out.setdefault(c["subject_entity_id"], []).append(c)
        return out

    return _memo_get(memo, ("player_claims", asof, claim_type), by_player).get(player_uid, [])


def _set_piece_goal_uplift_multiplier(
    con: duckdb.DuckDBPyConnection, player_uid: str, asof: datetime, set_piece_params_version: int,
    *, memo: dict | None = None,
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
    claims = _player_claims(con, asof, player_uid, "set_piece_order_override", memo)
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
    *, memo: dict | None = None,
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
    claims = _player_claims(con, asof, player_uid, "set_piece_order_override", memo)
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


# FPL awards assists more generously than Opta's xA counts them (second assists on won
# penalties, deflections, rebounds). Over 2024-26, outfield FPL assists ran at ~1.38x xA while
# goals ran at ~0.99x xG, so an xA-based rate under-predicts FPL assist points, most for the
# attackers who create the most (Finding 2). The ratio is fitted per position from whatever
# season stats are visible as of the run (point-in-time inside the walk-forward), shrunk toward
# 1.0 by a pseudo-count of prior_xa expected assists.
PLACEHOLDER_ASSIST_PRIOR_XA = 30.0


def seed_assist_calibration_params(con: duckdb.DuckDBPyConnection) -> None:
    params_mod.write_param(
        con, "fpl_assist_calibration_params", 1, "2026-09-29", "prior_xa", value_numeric=PLACEHOLDER_ASSIST_PRIOR_XA,
    )


def fpl_assist_ratio_by_position(con: duckdb.DuckDBPyConnection, seasons: list[str], prior_xa: float) -> dict[str, float]:
    """{position: (FPL assists + prior_xa) / (expected assists + prior_xa)} from each player's
    latest cumulative season row. Positions with no data get 1.0."""
    placeholders = ",".join(["?"] * len(seasons))
    rows = con.execute(
        f"""
        WITH latest AS (
            SELECT dp.position, fps.assists, fps.expected_assists
            FROM fact_player_season_stats fps JOIN dim_player dp ON dp.player_uid = fps.player_uid
            WHERE fps.season IN ({placeholders}) AND fps.assists IS NOT NULL AND fps.expected_assists IS NOT NULL
            QUALIFY row_number() OVER (PARTITION BY fps.player_uid, fps.season ORDER BY fps.gw DESC) = 1
        )
        SELECT position, sum(assists), sum(expected_assists) FROM latest GROUP BY position
        """,
        list(seasons),
    ).fetchall()
    out = {}
    for position, a, xa in rows:
        out[position] = ((a or 0) + prior_xa) / ((xa or 0.0) + prior_xa)
    return out


def compute_player_fixture_components(
    con: duckdb.DuckDBPyConnection, player_uid: str, position: str, team_uid: str, match_id: str,
    p_0: float, p_1_59: float, p_60plus: float,
    ts_model_version: int, scoring_params_version: int, bps_params_version: int,
    season_priority: list[str], mean_minutes: dict,
    *, asof: datetime | None = None, set_piece_params_version: int | None = None,
    fixture_params_version: int | None = 1, target_season: str | None = None,
    rate_shrinkage_params_version: int | None = None,
    assist_ratio: float = 1.0,
    finishing_prior_xg: float | None = None,
    memo: dict | None = None,
    bps_calibration_k: float | None = None,
    rate_prior: dict | None = None,
) -> dict:
    rates = player_rates_shrunk(
        con, player_uid, position, season_priority, rate_shrinkage_params_version, finishing_prior_xg=finishing_prior_xg,
        memo=memo, rate_prior=rate_prior,
    )
    def_rates = _defensive_action_rates_per_90(
        con, player_uid, position, season_priority, rate_shrinkage_params_version, memo=memo,
    )
    e_min_played = expected_minutes_given_played(p_1_59, p_60plus, mean_minutes)
    p_played = p_1_59 + p_60plus

    lambda_for, lambda_against, is_home = _fixture_lambdas(con, team_uid, match_id, ts_model_version, memo=memo)

    # ---- appearance ----
    ep_appearance = (
        _sm(con, "appearance_points_1_59", scoring_params_version) * p_1_59
        + _sm(con, "appearance_points_60plus", scoring_params_version) * p_60plus
    )

    # ---- goals / assists ----
    e_goals = rates["expected_goals_per_90"] * e_min_played / 90.0 * p_played
    e_assists = rates["expected_assists_per_90"] * e_min_played / 90.0 * p_played * assist_ratio
    # fixture-strength scaling: a player's flat season per-90 rate, adjusted for how favourable
    # THIS opponent is vs the team's average fixture (see _fixture_attack_multiplier). Was the
    # single biggest gap -- e_goals/e_assists were opponent-blind while clean sheets weren't.
    if fixture_params_version is not None:
        fixture_mult = _fixture_attack_multiplier(
            con, team_uid, match_id, target_season or season_priority[0],
            ts_model_version, fixture_params_version, memo=memo,
        )
        e_goals *= fixture_mult
        e_assists *= fixture_mult
    if asof is not None and set_piece_params_version is not None:
        e_goals *= _set_piece_goal_uplift_multiplier(con, player_uid, asof, set_piece_params_version, memo=memo)
        e_assists *= _set_piece_assist_uplift_multiplier(con, player_uid, asof, set_piece_params_version, memo=memo)
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
    # Fixture-strength scaling of the DEFENSIVE output (was opponent-blind -- a defender under
    # siege makes more clearances/blocks, a keeper vs a strong attack faces more shots -- while
    # the clean-sheet / goals-conceded terms above already use lambda_against directly).
    defence_defcon_mult = defence_saves_mult = 1.0
    if fixture_params_version is not None:
        defence_defcon_mult = _fixture_defensive_multiplier(
            con, team_uid, match_id, ts_model_version, fixture_params_version, "defcon_sensitivity", memo=memo,
        )
        defence_saves_mult = _fixture_defensive_multiplier(
            con, team_uid, match_id, ts_model_version, fixture_params_version, "save_sensitivity", memo=memo,
        )

    ep_defcon = 0.0
    if position in ("Defender", "Midfielder", "Forward") and defcon_in_force(target_season or season_priority[0]):
        defcon_actions_per_90 = def_rates["cbi_per_90"]
        if position in ("Midfielder", "Forward"):
            defcon_actions_per_90 += def_rates["recoveries_per_90"]
        defcon_rate = defcon_actions_per_90 * defence_defcon_mult * e_min_played / 90.0
        threshold = _sm(con, "defcon_threshold", scoring_params_version, position)
        p_over_threshold = 1.0 - poisson.cdf(threshold - 1, max(defcon_rate, 1e-9)) if defcon_rate > 0 else 0.0
        ep_defcon = p_over_threshold * p_played * _sm(con, "defcon_points", scoring_params_version)

    # ---- saves / penalty saves (goalkeepers only) ----
    ep_saves = 0.0
    ep_penalty_save = 0.0
    if position == "Goalkeeper":
        e_saves = rates["saves_per_90"] * defence_saves_mult * e_min_played / 90.0 * p_played
        ep_saves = e_saves / _sm(con, "saves_per_point", scoring_params_version)
        # No penalty-taker/penalties-faced rate reconciled -- left at 0 rather than guessed.

    # ---- expected BPS (mu_i), components backed by reconciled data only (see module docstring) ----
    # The target season's own BPS weights where they differed (season_rules).
    rules_season = target_season or season_priority[0]
    mu = 0.0
    mu += _bp(con, "playing_1_60", bps_params_version, season=rules_season) * p_1_59
    mu += _bp(con, "playing_60plus", bps_params_version, season=rules_season) * p_60plus
    mu += e_goals * _bp(con, "goal", bps_params_version, position, season=rules_season)
    mu += e_assists * _bp(con, "assist", bps_params_version, season=rules_season)
    # same fixture-scaled defensive rates as the ep_* terms above (BPS's "intentional dual use"
    # of the e_* expectations -- see the module docstring's non-double-counting note).
    e_cbi = def_rates["cbi_per_90"] * defence_defcon_mult * e_min_played / 90.0 * p_played
    e_recoveries = def_rates["recoveries_per_90"] * defence_defcon_mult * e_min_played / 90.0 * p_played
    mu += e_cbi / _bp(con, "cbi_per_point", bps_params_version, season=rules_season)
    mu += e_recoveries / _bp(con, "recoveries_per_point", bps_params_version, season=rules_season)
    if position in ("Goalkeeper", "Defender"):
        mu += _bp(con, "goal_conceded_gk_def", bps_params_version, season=rules_season) * (_expected_floor_half(lambda_against) * 2) * p_60plus
    if position == "Goalkeeper":
        e_saves = rates["saves_per_90"] * defence_saves_mult * e_min_played / 90.0 * p_played
        mu += e_saves * _bp(con, "save_inside_box", bps_params_version, season=rules_season)
    # opt-in: the BPS this player earns that the terms above can't see (see _bps_residual_table())
    if bps_calibration_k is not None:
        mu += bps_residual_per_90(
            con, player_uid, position, season_priority, bps_params_version, bps_calibration_k, memo=memo,
        ) * e_min_played / 90.0 * p_played

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

RECIPE_KEYS = (
    "set_piece_params_version", "fixture_params_version",
    "rate_shrinkage_params_version", "assist_calibration_params_version",
    "finishing_skill_params_version", "bps_calibration_params_version", "rate_prior_params_version",
)


def recipe_of(con: duckdb.DuckDBPyConnection, ep_model_version: int) -> dict | None:
    """The EP recipe run() recorded for this model version ({key: version or None}), or None
    for a row written before recipes were recorded -- the caller then keeps its own
    arguments. M4 (uncertainty) and M6 (Monte Carlo) read this so they describe the same
    k_minutes / fixture scaling / assist calibration the EP they sit on was built with."""
    row = con.execute(
        f"SELECT recipe_recorded, {', '.join(RECIPE_KEYS)} FROM ep_model_versions WHERE model_version = ?",
        [ep_model_version],
    ).fetchone()
    if row is None or not row[0]:
        return None
    return dict(zip(RECIPE_KEYS, row[1:]))


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
    rate_shrinkage_params_version: int | None = None,
    assist_calibration_params_version: int | None = None,
    finishing_skill_params_version: int | None = None,
    memo: dict | None = None,
    bps_calibration_params_version: int | None = None,
    rate_prior_params_version: int | None = None,
) -> int:
    # rate_prior_params_version (opt-in, None = off): the price anchor and season recency on
    # each player's goal/assist rates -- see the rate prior block above player_rates_shrunk().
    # bps_calibration_params_version (opt-in, None = off): add each player's BPS the estimate
    # can't see -- see _bps_residual_table().
    # memo (see new_memo()): pass one memo to every run() -- and the uncertainty.run() calls --
    # made against the same asof view, so each player's rates are built once for the whole
    # planning horizon rather than once per gameweek. None builds a fresh one for this call.
    if memo is None:
        memo = new_memo()
    # set_piece_params_version defaults to 1 (was None): the confirmed-primary penalty/free-kick
    # taker e_goals/e_assists uplift (_set_piece_goal_uplift_multiplier, built as Priority 7b but
    # never actually called by any live entrypoint) is now ON. It is a per-player no-op unless an
    # asof-visible set_piece_order_override primary claim exists -- so historical seasons with no
    # such claims are unaffected. Pass None to opt out.
    tau, _ = params_mod.resolve_param(con, "bps_dispersion_params", "tau", tau_params_version)
    mean_minutes = _memo_get(memo, ("mean_minutes",), lambda: _mean_minutes_by_bucket(con))
    # None keeps xA-based assists unchanged; a version applies the fitted FPL/xA ratio.
    assist_ratio_by_position: dict[str, float] = {}
    if assist_calibration_params_version is not None:
        prior_xa, _ = params_mod.resolve_param(con, "fpl_assist_calibration_params", "prior_xa", assist_calibration_params_version)
        assist_ratio_by_position = _memo_get(
            memo, ("assist_ratio", tuple(lookback_seasons), prior_xa),
            lambda: fpl_assist_ratio_by_position(con, list(lookback_seasons), prior_xa),
        )
    finishing_prior_xg = resolve_finishing_prior(con, finishing_skill_params_version)
    bps_calibration_k = resolve_bps_calibration(con, bps_calibration_params_version)
    rate_prior = resolve_rate_prior(con, rate_prior_params_version)
    # end-of-day, not start-of-day: same "as of this date" convention minutes_model.run()
    # already established -- a claim ingested at 09:34 on the asof date itself is legitimately
    # knowable "as of" that date. Only used when set_piece_params_version opts the uplift in.
    asof = datetime.combine(calibration_asof_date, datetime.max.time(), tzinfo=timezone.utc)

    fixtures = con.execute(
        "SELECT match_id, home_team_uid, away_team_uid FROM fact_match "
        "WHERE season = ? AND gameweek = ? AND competition = ? ORDER BY kickoff_time, match_id",
        [target_season, target_gameweek, PL],
    ).fetchall()
    if not fixtures:
        raise ValueError(f"no {PL} fixtures found for {target_season} GW{target_gameweek}")
    teams_table = reconcile_mod._season_root_table(con, target_season, "teams.csv")[1]
    minutes_by_player = _memo_get(memo, ("minutes", mm_model_version), lambda: {
        uid: (p_0, p_1_59, p_60plus) for uid, p_0, p_1_59, p_60plus in con.execute(
            "SELECT player_uid, p_0min, p_1_59min, p_60plus_min FROM minutes_model_outputs WHERE model_version = ?",
            [mm_model_version],
        ).fetchall()
    })

    model_version = con.execute(
        """
        INSERT INTO ep_model_versions
            (calibration_asof_date, target_season, team_strength_model_version, minutes_model_version,
             scoring_matrix_params_version, bps_params_version, bps_tau_params_version,
             set_piece_params_version, fixture_params_version, rate_shrinkage_params_version,
             assist_calibration_params_version, finishing_skill_params_version, bps_calibration_params_version,
             rate_prior_params_version, recipe_recorded)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, TRUE)
        RETURNING model_version
        """,
        [calibration_asof_date, target_season, ts_model_version, mm_model_version,
         scoring_params_version, bps_params_version, tau_params_version,
         set_piece_params_version, fixture_params_version, rate_shrinkage_params_version,
         assist_calibration_params_version, finishing_skill_params_version, bps_calibration_params_version,
         rate_prior_params_version],
    ).fetchone()[0]

    for match_id, home_uid, away_uid in fixtures:
        fixture_rows = []
        for team_uid in (home_uid, away_uid):
            # ORDER BY: DISTINCT returns rows in a different order on every run once DuckDB uses
            # more than one thread, and the order reaches the bonus model's float sums.
            roster = con.execute(
                """
                SELECT DISTINCT dp.player_uid, dp.position
                FROM player_alias pa
                JOIN dim_player dp ON dp.player_uid = pa.player_uid
                JOIN "{}" t ON t.code = pa.team_code
                JOIN team_alias ta ON ta.alias_name = t.name AND ta.season = pa.season
                WHERE pa.season = ? AND ta.team_uid = ?
                ORDER BY dp.player_uid
                """.format(teams_table),
                [target_season, team_uid],
            ).fetchall()
            for player_uid, position in roster:
                if position not in POSITIONS:
                    # a small number of non-player rows (managers, blank positions) leak
                    # into players.csv -- not a real squad player, skip rather than crash
                    # on an unseeded scoring-matrix lookup.
                    continue
                mrow = minutes_by_player.get(player_uid)
                if not mrow:
                    continue
                p_0, p_1_59, p_60plus = mrow
                comp = compute_player_fixture_components(
                    con, player_uid, position, team_uid, match_id, p_0, p_1_59, p_60plus,
                    ts_model_version, scoring_params_version, bps_params_version,
                    list(lookback_seasons), mean_minutes,
                    asof=asof, set_piece_params_version=set_piece_params_version,
                    fixture_params_version=fixture_params_version, target_season=target_season,
                    rate_shrinkage_params_version=rate_shrinkage_params_version,
                    assist_ratio=assist_ratio_by_position.get(position, 1.0),
                    finishing_prior_xg=finishing_prior_xg,
                    memo=memo,
                    bps_calibration_k=bps_calibration_k,
                    rate_prior=rate_prior,
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
