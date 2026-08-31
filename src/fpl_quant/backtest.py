"""M7: Walk-Forward Backtesting Framework.

Runs the full M1->M6 pipeline against ~76 historical gameweeks (2024-25 + 2025-26, now
confirmed as two fully-completed seasons -- see M7 spec's own research correction), tiered
cold/warm/mature and scored against realized outcomes once known.

Every module downstream of M0 (team_strength, minutes_model, expected_points, uncertainty,
squad_optimizer, monte_carlo) stores calibration_asof_date for audit but -- until this module
-- never actually filters a fact-table query with it (confirmed by reading every one of
those six modules: only minutes_model.py's evidence-claim path calls a real asof filter,
snapshot.get_claims_asof). That's been invisible because every live run's historical facts
trivially predate data_asof; this is the first real exercise of the guarantee, so the
enforcement mechanism below (asof_scope) has to actually work, not just be plausible.
"""

import json
import math
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import poisson

from . import expected_points as ep
from . import minutes_model
from . import monte_carlo
from . import ownership as ownership_mod
from . import params as params_mod
from . import squad_optimizer
from . import team_strength
from . import transfer_planner
from . import uncertainty

PL = "Premier League"

# 2024-25 GW1 is the earliest gameweek in the whole repo -- its own deadline has zero prior
# matches under it by construction (team_strength.fit_dixon_coles() cannot MLE-fit an empty
# set; calibrate() raises ValueError on matches.empty). This floor is what backtest_runs'
# warm_up_gameweeks records: the count of leading gameweeks skipped for exactly this reason,
# not an arbitrary tuning knob.
COLD_START_MATCH_FLOOR = 1

# No real Premier League fixture has ever approached this expected-goals level -- a generous
# upper bound (the real single-match scoreline record is 9-0) used only to detect a diverged
# cold-tier MLE fit (see score_gameweek()'s M1 Poisson-calibration guard), not a plausible
# real value to clip toward.
MAX_PHYSICAL_LAMBDA = 15.0

TIER_BOUNDARIES = {
    "2024-2025": {"cold_through_gw": 9},
    "2025-2026": {"warm_through_gw": 15},
}


def tier_for(season: str, gameweek: int) -> str:
    """Implements the spec's tiering table literally: 2024-25 GW1-9 cold, 2024-25 GW10-38 +
    2025-26 GW1-~15 warm, rest of 2025-26 mature.

    2026-2027 is "mature" from GW1: every one of its gameweeks sits on top of two complete
    prior Premier League seasons (760 finished matches) under any asof cutoff, so its
    calibration-data volume is already well past the bar 2025-2026 only cleared at GW16. The
    cold/warm split existed to flag genuinely data-starved early gameweeks in the first
    loaded season, a condition that cannot recur once two full seasons precede the target."""
    if season == "2024-2025":
        return "cold" if gameweek <= TIER_BOUNDARIES["2024-2025"]["cold_through_gw"] else "warm"
    if season == "2025-2026":
        return "warm" if gameweek <= TIER_BOUNDARIES["2025-2026"]["warm_through_gw"] else "mature"
    if season == "2026-2027":
        return "mature"
    raise ValueError(f"no tier definition for season {season!r} -- backtest only covers 2024-25/2025-26/2026-27")


def fit_seasons_for(season: str) -> tuple[str, ...]:
    """team_strength.calibrate()'s Elo-regression eligibility threshold is
    min(seasons_threshold, len(fit_seasons)) -- its own hardcoded live default,
    fit_seasons=("2024-2025","2025-2026"), assumes both seasons can contribute. Backtesting
    2024-2025 itself with that default is a real, structural (not just early-gameweek) bug:
    "2025-2026" is always entirely in the future under any 2024-2025 asof cutoff, so it always
    shadows to zero rows and no team can ever reach the len(fit_seasons)=2 threshold, no matter
    how late in the season -- calibrate() would hard-crash in fit_elo_regression on every single
    2024-2025 step, not just the earliest ones. fit_seasons must instead be only the seasons
    actually reachable as of the season being backtested.

    2026-2027 adds itself to the pair: under a 2026-2027 asof cutoff none of the three
    seasons shadows entirely to empty (2024-25 and 2025-26 are complete; 2026-27 has its own
    played gameweeks), so len(fit_seasons)=3 makes effective_threshold=min(3,3)=3 and the
    eligible-teams population is the clubs present in all three seasons -- 15 in the real
    ingested DB, a healthy Elo-regression population (fit_elo_regression needs >=2), not the
    empty set the 2024-2025 bug produced. Early in 2026-2027 its own match rows are sparse,
    but fit_dixon_coles' xi-decay weighting and weight_own_data=min(1,seasons/3) already
    govern how much that thin recent slice is trusted; fit_seasons only decides what data is
    on the table, and all reachable data is the consistent answer."""
    if season == "2024-2025":
        return ("2024-2025",)
    if season == "2025-2026":
        return ("2024-2025", "2025-2026")
    if season == "2026-2027":
        return ("2024-2025", "2025-2026", "2026-2027")
    raise ValueError(f"no fit_seasons definition for season {season!r} -- backtest only covers 2024-25/2025-26/2026-27")


def gameweek_deadline(con: duckdb.DuckDBPyConnection, season: str, gameweek: int):
    """Earliest kickoff_time among that gameweek's real fixtures, standing in for the actual
    FPL transfer deadline -- no deadline field exists anywhere in the ingested data (grepped
    repo-wide for "deadline", zero hits). A documented approximation, not a precise value."""
    row = con.execute(
        "SELECT min(kickoff_time) FROM fact_match WHERE season = ? AND gameweek = ? AND competition = ?",
        [season, gameweek, PL],
    ).fetchone()
    return row[0] if row else None


def has_double_gameweek(con: duckdb.DuckDBPyConnection, season: str, gameweek: int) -> bool:
    """True iff any team has more than one fixture under this gameweek's own label -- a real,
    if infrequent, historical occurrence (rearranged fixtures squeezed into the same FPL
    gameweek slot: confirmed via the actual ingested data at 2024-25 GW25 and 2025-26 GW26/33/36,
    4 of the 76 backtest gameweeks). expected_points.run() emits one ep_outputs row per player
    per fixture by its own explicit v1 design ("DGW/multi-fixture handling is out of scope for
    v1, per M8's own research finding that 2026-27 currently has no scheduled doubles/blanks" --
    expected_points.py's own module docstring) -- that finding was scoped to the live target
    gameweek specifically, not a guarantee about the historical seasons M7 walks through, and a
    DGW player's duplicate ep_outputs rows crash squad_optimizer_selections' primary key
    (player_uid, run_id) with no aggregation semantics defined for what a DGW player's combined
    squad value should even mean. Extending that same existing v1 scope boundary into the
    backtest loop (skip, don't invent DGW-aggregation modeling here) is the consistent fix, not
    a new decision improvised mid-M7."""
    row = con.execute(
        """
        SELECT count(*) FROM (
            SELECT team_uid FROM (
                SELECT home_team_uid AS team_uid FROM fact_match WHERE season = ? AND gameweek = ? AND competition = ?
                UNION ALL
                SELECT away_team_uid FROM fact_match WHERE season = ? AND gameweek = ? AND competition = ?
            ) GROUP BY team_uid HAVING count(*) > 1
        )
        """,
        [season, gameweek, PL, season, gameweek, PL],
    ).fetchone()
    return row[0] > 0


def has_fittable_history(con: duckdb.DuckDBPyConnection, season: str, gameweek: int, floor: int = COLD_START_MATCH_FLOOR) -> bool:
    """True iff at least `floor` finished PL matches exist strictly before this gameweek's
    deadline. Checked before attempting a step at all -- team_strength.calibrate() has no
    graceful empty-data path of its own, by design (an unfittable request should be loud, not
    silently defaulted), so the walk-forward loop is what has to know to skip it instead."""
    deadline = gameweek_deadline(con, season, gameweek)
    if deadline is None:
        return False
    n = con.execute(
        "SELECT count(*) FROM fact_match WHERE competition = ? AND finished = TRUE "
        "AND home_score IS NOT NULL AND away_score IS NOT NULL AND kickoff_time < ?",
        [PL, deadline],
    ).fetchone()[0]
    return n >= floor


@contextmanager
def asof_scope(con: duckdb.DuckDBPyConnection, season: str, gameweek: int, *, schedule_horizon_gameweeks: int = 1):
    """Connection-scoped TEMP TABLE shadowing of the three fact tables, truncated to what was
    knowable strictly before this gameweek's deadline. DuckDB resolves an unqualified table
    name against `temp` before `main`, so every existing M1-M5 query -- all of which read
    fact_match/fact_player_match_stats/fact_player_season_stats by bare name -- runs completely
    unmodified and sees only asof-valid rows. main.* is never touched; only this connection's
    temp schema is rebuilt, and dropped again on exit.

    fact_player_season_stats has PRIMARY KEY (player_uid, season, gw) -- already a per-gameweek
    cumulative snapshot, not a single season-aggregate row -- so truncating the in-progress
    season to `gw < gameweek` is exact, no on-the-fly re-aggregation needed. Prior, fully
    completed seasons pass through whole (season <> the one being backtested is never
    date-sensitive relative to this gameweek's cutoff).

    fact_match gets one deliberate exception, not a strict kickoff_time cutoff: the target
    gameweek's own fixture *schedule* (match_id/home_team_uid/away_team_uid/kickoff_time) stays
    visible with home_score/away_score/finished nulled out -- a real fix, found by running this
    against the actual DB, not a hypothetical. A strict `kickoff_time < deadline` cutoff hides
    the very fixtures being predicted (their kickoff_time is exactly the deadline), and
    expected_points.run()/monte_carlo.run() both need that gameweek's schedule to know which
    players face which fixture -- that's the whole point of the prediction, not a leak. Only the
    *result* is unknowable in advance; the schedule is announced well before any deadline.

    schedule_horizon_gameweeks (default 1, M7's original single-gameweek behavior, exact and
    unchanged): widens that same schedule-only exception to cover [gameweek, gameweek +
    schedule_horizon_gameweeks), not just gameweek itself. Needed for M8's compute_horizon_ep(),
    which plans several gameweeks ahead in one call -- a real manager planning at gameweek G's
    deadline genuinely does know gameweeks G+1..G+4's fixture schedules (the whole season's
    calendar is announced well before a ball is kicked), just not their results, so revealing
    only the schedule that far ahead is not a look-ahead leak, the same reasoning the single-
    gameweek case already rests on, just extended over a wider, still-schedule-only window.
    Every gameweek beyond that window stays fully invisible, schedule included, exactly as
    before.

    Yields the deadline timestamp used for the shadow, for callers that also need it (e.g. to
    stamp the calibration_asof_date passed into M1-M6).
    """
    deadline = gameweek_deadline(con, season, gameweek)
    if deadline is None:
        raise ValueError(f"no {PL} fixtures found for {season} GW{gameweek} -- cannot pin a deadline")
    schedule_horizon_end_gameweek = gameweek + schedule_horizon_gameweeks - 1

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE fact_match AS
        SELECT match_id, season, gameweek, kickoff_time, home_team_uid, away_team_uid,
               CASE WHEN kickoff_time < ? THEN home_score ELSE NULL END AS home_score,
               CASE WHEN kickoff_time < ? THEN away_score ELSE NULL END AS away_score,
               home_team_elo, away_team_elo,
               CASE WHEN kickoff_time < ? THEN finished ELSE FALSE END AS finished,
               competition, _ingested_at
        FROM main.fact_match
        WHERE kickoff_time < ? OR (season = ? AND gameweek BETWEEN ? AND ?)
        """,
        [deadline, deadline, deadline, deadline, season, gameweek, schedule_horizon_end_gameweek],
    )
    con.execute(
        """CREATE OR REPLACE TEMP TABLE fact_player_match_stats AS
           SELECT s.* FROM main.fact_player_match_stats s JOIN main.fact_match m USING (match_id)
           WHERE m.kickoff_time < ?""",
        [deadline],
    )
    con.execute(
        "CREATE OR REPLACE TEMP TABLE fact_player_season_stats AS "
        "SELECT * FROM main.fact_player_season_stats WHERE (season = ? AND gw < ?) OR season <> ?",
        [season, gameweek, season],
    )
    try:
        yield deadline
    finally:
        con.execute("DROP TABLE IF EXISTS fact_match")
        con.execute("DROP TABLE IF EXISTS fact_player_match_stats")
        con.execute("DROP TABLE IF EXISTS fact_player_season_stats")


# ============================================================
# one walk-forward step: run M1->M6 pinned to one historical gameweek's asof cutoff
# ============================================================

def run_gameweek_step(
    con: duckdb.DuckDBPyConnection,
    backtest_run_id: int,
    season: str,
    gameweek: int,
    *,
    xi_params_version: int,
    rho_params_version: int,
    decay_params_version: int,
    adjustment_params_version: int,
    shrinkage_params_version: int,
    fact_multiplier_params_version: int,
    scoring_params_version: int,
    bps_params_version: int,
    tau_params_version: int,
    rho_residual_params_version: int,
    corr_params_version: int,
    lambda_params_version: int,
    guardrail_params_version: int,
    n_antithetic_pairs: int = 2000,
    run_monte_carlo: bool = True,
) -> None:
    """One walk-forward step. Inside asof_scope, calls the exact same M1-M6 entrypoints a live
    run calls, completely unmodified -- the shadow is what makes every one of those calls
    asof-safe, not a parallel asof-aware code path. Two outcomes are caught and recorded rather
    than aborting the whole step, both real/expected, not defensive over-catching:

    - DivergenceCheckFailedError: M7's spec requires this check to run every walk-forward pass
      and its result recorded, not gate the loop (M5's own squad_optimizer.run() already
      refuses to store a squad selection on failure -- so_run_id stays None, exactly like a
      live run would leave it).
    - candidate pool < 15 priced players (squad_optimizer.fetch_candidate_pool): expected in
      genuinely early-season gameweeks where too few ep_outputs rows exist yet to fill a squad,
      not a bug -- recorded as a skipped optimizer stage (divergence_check_passed stays NULL,
      distinct from an explicit False).
    """
    tier = tier_for(season, gameweek)
    deadline = gameweek_deadline(con, season, gameweek)
    if deadline is None:
        raise ValueError(f"no {PL} fixtures for {season} GW{gameweek}")
    calibration_asof_date = deadline.date()

    ts_model_version = mm_model_version = ep_model_version = un_model_version = None
    so_run_id = mc_model_version = None
    divergence_passed = None

    with asof_scope(con, season, gameweek):
        ts_model_version = team_strength.calibrate(
            con, calibration_asof_date, xi_params_version, rho_params_version,
            target_season=season, fit_seasons=fit_seasons_for(season),
        )
        mm_model_version = minutes_model.run(
            con, calibration_asof_date, season, decay_params_version, adjustment_params_version,
            shrinkage_params_version, fact_multiplier_params_version,
        )
        ep_model_version = ep.run(
            con, calibration_asof_date, season, gameweek, ts_model_version, mm_model_version,
            scoring_params_version, bps_params_version, tau_params_version,
        )
        un_model_version = uncertainty.run(
            con, calibration_asof_date, ep_model_version, mm_model_version, ts_model_version,
            scoring_params_version, bps_params_version, tau_params_version,
            rho_residual_params_version, corr_params_version,
        )
        try:
            so_run_id = squad_optimizer.run(
                con, calibration_asof_date, season, gameweek, ep_model_version, un_model_version,
                lambda_params_version, guardrail_params_version,
            )
            divergence_passed = True
        except squad_optimizer.DivergenceCheckFailedError:
            divergence_passed = False
        except ValueError:
            divergence_passed = None

        if so_run_id is not None and run_monte_carlo:
            mc_model_version = monte_carlo.run(
                con, calibration_asof_date, so_run_id, ep_model_version, mm_model_version, ts_model_version,
                un_model_version, scoring_params_version, tau_params_version, rho_residual_params_version,
                n_antithetic_pairs=n_antithetic_pairs,
            )

    con.execute(
        """
        INSERT INTO backtest_gameweek_steps
            (backtest_run_id, season, gameweek, tier, data_asof, ts_model_version, mm_model_version,
             ep_model_version, un_model_version, so_run_id, mc_model_version, divergence_check_passed)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [backtest_run_id, season, gameweek, tier, deadline, ts_model_version, mm_model_version,
         ep_model_version, un_model_version, so_run_id, mc_model_version, divergence_passed],
    )


# ============================================================
# scoring: log score / Brier / Poisson-calibration helpers (pure functions)
# ============================================================

_EPS = 1e-9


def log_score_bernoulli(p: float, outcome: bool) -> float:
    p = min(max(p, _EPS), 1 - _EPS)
    return math.log(p) if outcome else math.log(1 - p)


def brier_bernoulli(p: float, outcome: bool) -> float:
    return (p - (1.0 if outcome else 0.0)) ** 2


def log_score_categorical(probs: dict, observed_key: str) -> float:
    p = min(max(probs.get(observed_key, 0.0), _EPS), 1.0)
    return math.log(p)


def brier_categorical(probs: dict, observed_key: str) -> float:
    return sum((p - (1.0 if k == observed_key else 0.0)) ** 2 for k, p in probs.items())


def log_score_poisson(lam: float, observed_count: int) -> float:
    return float(poisson.logpmf(observed_count, max(lam, _EPS)))


def _minutes_state(minutes_played: int) -> str:
    if minutes_played <= 0:
        return "0"
    return "60plus" if minutes_played >= 60 else "1_59"


def _realized_player_match_outcome(con: duckdb.DuckDBPyConnection, player_uid: str, match_id: str) -> dict:
    """Reads main.fact_player_match_stats directly (score_gameweek runs after asof_scope has
    already exited, so bare table names resolve to main.* again -- this gameweek's results are
    now real, not shadowed). A missing row means the player didn't feature at all (0 minutes),
    not missing data -- most squad members are unused substitutes most gameweeks."""
    row = con.execute(
        "SELECT minutes_played, goals, assists, team_goals_conceded FROM fact_player_match_stats "
        "WHERE player_uid = ? AND match_id = ?",
        [player_uid, match_id],
    ).fetchone()
    if row is None or row[0] is None:
        return {"minutes_played": 0, "goals": 0, "assists": 0, "team_goals_conceded": None}
    minutes_played, goals, assists, team_goals_conceded = row
    return {
        "minutes_played": minutes_played, "goals": goals or 0, "assists": assists or 0,
        "team_goals_conceded": team_goals_conceded,
    }


# ============================================================
# Priority 9b: segment-tiering. A SEPARATE axis from tier_for()'s time-based cold/warm/mature
# classification above -- this project's own "realized_goals_assists:{player_uid}" /
# "realized_points:{player_uid}" metric names (score_gameweek's own established convention,
# not a new one invented here) already show backtest_metrics' metric_name column is meant to
# carry a colon-suffixed sub-dimension; segment membership reuses that same convention
# (":promoted_team", ":new_signing", ":set_piece_taker") rather than a new schema column.
# ============================================================

def _previous_season(con: duckdb.DuckDBPyConnection, season: str) -> str | None:
    """The season immediately before `season` among seasons that actually have fact_match
    data -- ordered by the season string itself (YYYY-YYYY sorts chronologically), not parsed
    year arithmetic, so this doesn't assume anything about which seasons exist beyond what's
    actually loaded. None for the earliest loaded season -- there's genuinely nothing to
    compare it against, not a false "not promoted" default."""
    seasons = [r[0] for r in con.execute(
        "SELECT DISTINCT season FROM fact_match WHERE competition = ? ORDER BY season", [PL]
    ).fetchall()]
    if season not in seasons:
        return None
    idx = seasons.index(season)
    return seasons[idx - 1] if idx > 0 else None


def _is_newly_promoted_team(con: duckdb.DuckDBPyConnection, team_uid: str, season: str) -> bool | None:
    """True iff team_uid has real fixtures in `season` but had none in the immediately
    preceding loaded season -- derived from fact_match itself (no external promoted-teams
    list to maintain or get stale). None (not False) when there's no preceding season loaded
    to compare against -- "can't tell" is a genuinely different claim from "not promoted"."""
    prev_season = _previous_season(con, season)
    if prev_season is None:
        return None
    n_prev = con.execute(
        "SELECT count(*) FROM fact_match WHERE season = ? AND competition = ? "
        "AND (home_team_uid = ? OR away_team_uid = ?)",
        [prev_season, PL, team_uid, team_uid],
    ).fetchone()[0]
    return n_prev == 0


def _is_new_signing(con: duckdb.DuckDBPyConnection, player_uid: str, season: str) -> bool | None:
    """True iff the player's club (player_alias.team_code, season-grain -- the same grain
    minutes_model already keys lookback seasons on) differs from, or didn't exist in, the
    immediately preceding loaded season. None when there's no preceding season to compare
    against, or when a player has more than one distinct team_code within `season` itself
    (a genuine mid-season transfer makes "new signing" for the WHOLE season ambiguous --
    skipped rather than guessed, same simplification-disclosure pattern as everywhere else
    in this project)."""
    prev_season = _previous_season(con, season)
    if prev_season is None:
        return None
    cur_codes = {r[0] for r in con.execute(
        "SELECT DISTINCT team_code FROM player_alias WHERE player_uid = ? AND season = ?", [player_uid, season]
    ).fetchall()}
    if len(cur_codes) != 1:
        return None
    prev_codes = {r[0] for r in con.execute(
        "SELECT DISTINCT team_code FROM player_alias WHERE player_uid = ? AND season = ?", [player_uid, prev_season]
    ).fetchall()}
    return cur_codes != prev_codes


def _player_segments(
    con: duckdb.DuckDBPyConnection, player_uid: str, team_uid: str | None, season: str,
    asof, set_piece_params_version: int | None, promoted_team_cache: dict,
) -> set[str]:
    segments = set()
    if team_uid is not None:
        if team_uid not in promoted_team_cache:
            promoted_team_cache[team_uid] = _is_newly_promoted_team(con, team_uid, season)
        if promoted_team_cache[team_uid]:
            segments.add("promoted_team")
    if _is_new_signing(con, player_uid, season):
        segments.add("new_signing")
    if set_piece_params_version is not None and asof is not None:
        if ep._set_piece_goal_uplift_multiplier(con, player_uid, asof, set_piece_params_version) > 1.0:
            segments.add("set_piece_taker")
    return segments


def _record_segment_metrics(
    con: duckdb.DuckDBPyConnection, backtest_run_id: int, season: str, gameweek: int, tier: str,
    metric_name: str, scored: list[tuple[str, float]], segment_of: dict[str, set[str]],
) -> None:
    """scored: [(player_uid, score_value), ...] for one metric family this gameweek --
    buckets by segment and records a mean per segment actually represented (never a segment
    with zero members this gameweek, never a fabricated 0.0)."""
    by_segment: dict[str, list[float]] = {}
    for player_uid, value in scored:
        for segment in segment_of.get(player_uid, ()):
            by_segment.setdefault(segment, []).append(value)
    for segment, values in by_segment.items():
        _record_metric(con, backtest_run_id, season, gameweek, tier, f"{metric_name}:{segment}", sum(values) / len(values))


# ============================================================
# Priority 9c: "beats the ownership-weighted average manager" -- the real bar is beating the
# crowd, not clearing zero (roadmap's own framing).
# ============================================================

def _avg_manager_benchmark_points(
    con: duckdb.DuckDBPyConnection, season: str, gameweek: int, ep_model_version: int, ownership_params_version: int,
) -> float | None:
    """A SYNTHETIC 'average manager' benchmark, not FPL's own real published average entry
    score (that aggregate isn't ingested anywhere in this project -- grepped). Reuses Priority
    1's exact EO mechanism (ownership.compute_eo_for_pool) -- every player with a real fixture
    this gameweek contributes eo_pct/100 * their REAL realized fact_player_season_stats.
    event_points, approximating "what the ownership-weighted field actually scored" the same
    way field_covariance.py's synthetic EO-weighted portfolio approximates field risk exposure
    elsewhere -- one EO mechanism, two consumers, not two independently-invented ones.

    Players with no real selected_by_percent data this gameweek are excluded from the sum (a
    genuine, disclosed simplification, not a fabricated 0% assumption) -- None (not 0.0) when
    NO player in the pool has usable ownership data at all, since that's "not computable," not
    "the field scored zero.\""""
    captaincy_concentration, _ = params_mod.resolve_param(
        con, "ownership_params", "captaincy_concentration", ownership_params_version,
    )
    rows = con.execute(
        "SELECT o.player_uid, dp.position, o.ep_total, fps.selected_by_percent "
        "FROM ep_outputs o JOIN dim_player dp ON dp.player_uid = o.player_uid "
        "LEFT JOIN fact_player_season_stats fps ON fps.player_uid = o.player_uid AND fps.season = ? AND fps.gw = ? "
        "WHERE o.model_version = ?",
        [season, gameweek, ep_model_version],
    ).fetchall()
    candidates = [
        {"player_uid": uid, "position": pos, "mu": ep_total, "selected_by_percent": sbp}
        for uid, pos, ep_total, sbp in rows
    ]
    eo_by_uid = ownership_mod.compute_eo_for_pool(candidates, captaincy_concentration)

    total, any_eo = 0.0, False
    for uid, _pos, _ep_total, _sbp in rows:
        eo = eo_by_uid.get(uid)
        if eo is None:
            continue
        any_eo = True
        row = con.execute(
            "SELECT event_points FROM fact_player_season_stats WHERE player_uid = ? AND season = ? AND gw = ?",
            [uid, season, gameweek],
        ).fetchone()
        pts = row[0] if row and row[0] is not None else 0.0
        total += (eo / 100.0) * pts
    return total if any_eo else None


def _record_metric(con: duckdb.DuckDBPyConnection, backtest_run_id: int, season: str, gameweek: int, tier: str, metric_name: str, metric_value: float) -> None:
    con.execute(
        "INSERT INTO backtest_metrics (backtest_run_id, season, gameweek, tier, metric_name, metric_value) "
        "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
        [backtest_run_id, season, gameweek, tier, metric_name, float(metric_value)],
    )


def score_gameweek(
    con: duckdb.DuckDBPyConnection,
    backtest_run_id: int,
    season: str,
    gameweek: int,
    ep_model_version: int,
    mm_model_version: int,
    ts_model_version: int,
    scoring_params_version: int,
    so_run_id: int | None = None,
    *,
    compute_segments: bool = False,
    set_piece_params_version: int | None = None,
    ownership_params_version: int | None = None,
) -> None:
    """Scores one already-completed walk-forward step against that gameweek's now-real
    results. Four metric families for this first implementation pass (deliberately scoped --
    DefCon/bonus/saves each need their own realized-outcome reconstruction and are left as a
    mechanical extension of the same helpers above, not built here):

    - M1 Poisson calibration: mean(realized goal count - fitted lambda), signed (bias
      direction, not just magnitude) -- reuses expected_points._fixture_lambdas() unchanged.
    - M2 minutes distribution: categorical log score + Brier vs realized 0/1-59/60+ state.
    - M3 clean sheet: Bernoulli log score + Brier, using the *same* joint gate M3 itself
      predicts with (p_clean_sheet = exp(-lambda_against)*p_60plus) -- realized outcome is
      therefore also the joint (team conceded 0) AND (this player played 60+), not either
      alone, so the comparison is apples-to-apples with what was actually predicted.
    - M3 goals/assists: Poisson log score vs realized counts, recovering the predicted lambda
      from ep_outputs via the same points/rate inversion monte_carlo.compute_lambda_representative()
      already established (ep_goals/goal_points = lambda).

    Only scores players expected_points.run() itself determined had a fixture that gameweek
    (i.e. iterates ep_outputs, not the full player pool) -- reuses M3's own fixture-
    participation scope rather than re-deriving team rosters here.

    When so_run_id is given (a squad_optimizer selection exists for this step), also records
    each squad player's *realized* FPL points that gameweek (fact_player_season_stats.
    event_points -- the real, already-computed FPL gameweek score, not a reconstruction) as
    per-player backtest_metrics rows. This is raw material for recalibrate()'s cross-gameweek
    moment-matching of rho_residual/Z_fixture against realized covariance -- a single
    gameweek's realized points for a pair of players is just two scalars, not a covariance;
    the aggregation across many gameweeks happens at recalibration time, not here.

    Priority 9b (opt-in via compute_segments, default off -- every existing caller/backtest
    run is unaffected): additionally buckets each of the four metric families above by
    segment (promoted_team, new_signing, set_piece_taker -- see _player_segments()) and
    records a segment-suffixed metric_name (matching this function's own established
    "realized_goals_assists:{player_uid}" colon-suffix convention below) for whichever
    segments actually have members this gameweek. set_piece_taker specifically also needs
    set_piece_params_version -- without it, that one segment is skipped, the other two are not.

    Priority 9c (opt-in via ownership_params_version, needs so_run_id too -- there's no
    squad to compare against otherwise): records model_squad_realized_points,
    avg_manager_benchmark_points, and their delta (beats_crowd_points_delta, positive = beat
    the synthetic EO-weighted crowd that week) -- see _avg_manager_benchmark_points().
    """
    tier = tier_for(season, gameweek)

    # -------- M1: Poisson calibration --------
    fixtures = con.execute(
        "SELECT match_id, home_team_uid, away_team_uid, home_score, away_score FROM fact_match "
        "WHERE season = ? AND gameweek = ? AND competition = ? AND home_score IS NOT NULL AND away_score IS NOT NULL",
        [season, gameweek, PL],
    ).fetchall()

    segment_of: dict[str, set[str]] = {}
    if compute_segments:
        team_of: dict[str, str] = {}
        promoted_team_cache: dict[str, bool | None] = {}
        asof = gameweek_deadline(con, season, gameweek)
        for match_id, home_uid, away_uid, _hs, _as in fixtures:
            team_of.update(monte_carlo._team_of_for_fixture(con, home_uid, away_uid, season))
        for player_uid, team_uid in team_of.items():
            segment_of[player_uid] = _player_segments(
                con, player_uid, team_uid, season, asof, set_piece_params_version, promoted_team_cache,
            )
    resids, n_degenerate = [], 0
    for match_id, home_uid, away_uid, home_score, away_score in fixtures:
        lam_home, lam_away, _ = ep._fixture_lambdas(con, home_uid, match_id, ts_model_version)
        for lam, score in ((lam_home, home_score), (lam_away, away_score)):
            # fit_dixon_coles()'s unconstrained MLE (M1, tested/frozen against the live fit --
            # not touched here) can genuinely diverge on cold-tier's sparse early data, a real
            # quasi-separation problem (a team with zero conceded goals in its first 1-2
            # matches has no likelihood penalty against its defence parameter running to +inf).
            # No real Premier League fixture has ever approached MAX_PHYSICAL_LAMBDA goals --
            # excluded here rather than silently averaged into a meaningless number, and
            # counted so the instability is visible in the tiered report, not hidden.
            if not math.isfinite(lam) or lam > MAX_PHYSICAL_LAMBDA:
                n_degenerate += 1
                continue
            resids.append(score - lam)
    if resids:
        _record_metric(con, backtest_run_id, season, gameweek, tier, "poisson_calibration_mean_resid", sum(resids) / len(resids))
    if n_degenerate:
        _record_metric(con, backtest_run_id, season, gameweek, tier, "poisson_calibration_degenerate_count", n_degenerate)

    # -------- M2: minutes distribution --------
    mm_rows = con.execute(
        "SELECT m.player_uid, m.p_0min, m.p_1_59min, m.p_60plus_min FROM minutes_model_outputs m "
        "WHERE m.model_version = ? AND m.player_uid IN (SELECT DISTINCT player_uid FROM ep_outputs WHERE model_version = ?)",
        [mm_model_version, ep_model_version],
    ).fetchall()
    ep_fixture_of = dict(con.execute(
        "SELECT player_uid, fixture_match_id FROM ep_outputs WHERE model_version = ?", [ep_model_version]
    ).fetchall())
    minutes_log, minutes_brier = [], []
    minutes_log_by_player, minutes_brier_by_player = [], []
    for player_uid, p0, p1, p2 in mm_rows:
        match_id = ep_fixture_of.get(player_uid)
        if match_id is None:
            continue
        outcome = _realized_player_match_outcome(con, player_uid, match_id)
        state = _minutes_state(outcome["minutes_played"])
        probs = {"0": p0, "1_59": p1, "60plus": p2}
        log_val, brier_val = log_score_categorical(probs, state), brier_categorical(probs, state)
        minutes_log.append(log_val)
        minutes_brier.append(brier_val)
        minutes_log_by_player.append((player_uid, log_val))
        minutes_brier_by_player.append((player_uid, brier_val))
    if minutes_log:
        _record_metric(con, backtest_run_id, season, gameweek, tier, "log_score_minutes_mean", sum(minutes_log) / len(minutes_log))
        _record_metric(con, backtest_run_id, season, gameweek, tier, "brier_minutes_mean", sum(minutes_brier) / len(minutes_brier))
    if compute_segments:
        _record_segment_metrics(con, backtest_run_id, season, gameweek, tier, "log_score_minutes_mean", minutes_log_by_player, segment_of)
        _record_segment_metrics(con, backtest_run_id, season, gameweek, tier, "brier_minutes_mean", minutes_brier_by_player, segment_of)

    # -------- M3: clean sheet (Bernoulli), goals/assists (Poisson) --------
    squad_uids = set()
    if so_run_id is not None:
        squad_uids = {r[0] for r in con.execute(
            "SELECT player_uid FROM squad_optimizer_selections WHERE run_id = ? AND in_squad", [so_run_id]
        ).fetchall()}

    ep_rows = con.execute(
        "SELECT o.player_uid, dp.position, o.fixture_match_id, o.ep_clean_sheet, o.ep_goals, o.ep_assists "
        "FROM ep_outputs o JOIN dim_player dp ON dp.player_uid = o.player_uid WHERE o.model_version = ?",
        [ep_model_version],
    ).fetchall()
    cs_log, cs_brier, goals_log, assists_log = [], [], [], []
    cs_log_bp, cs_brier_bp, goals_log_bp, assists_log_bp = [], [], [], []
    for player_uid, position, match_id, ep_cs, ep_g, ep_a in ep_rows:
        outcome = _realized_player_match_outcome(con, player_uid, match_id)

        cs_pts = ep._sm(con, "clean_sheet_points", scoring_params_version, position)
        if cs_pts:
            p_cs = ep_cs / cs_pts
            realized_cs = outcome["team_goals_conceded"] == 0 and outcome["minutes_played"] >= 60
            log_val, brier_val = log_score_bernoulli(p_cs, realized_cs), brier_bernoulli(p_cs, realized_cs)
            cs_log.append(log_val)
            cs_brier.append(brier_val)
            cs_log_bp.append((player_uid, log_val))
            cs_brier_bp.append((player_uid, brier_val))

        goal_pts = ep._sm(con, "goal_points", scoring_params_version, position)
        if goal_pts:
            goals_val = log_score_poisson(ep_g / goal_pts, outcome["goals"])
            goals_log.append(goals_val)
            goals_log_bp.append((player_uid, goals_val))

        assist_pts = ep._sm(con, "assist_points", scoring_params_version)
        if assist_pts:
            assists_val = log_score_poisson(ep_a / assist_pts, outcome["assists"])
            assists_log.append(assists_val)
            assists_log_bp.append((player_uid, assists_val))

        # rho_residual/Z_fixture are calibrated against realized goals+assists COUNTS
        # specifically (monte_carlo.compute_lambda_representative()'s own definition), not
        # total points -- captured here, in the same loop that already computed `outcome`,
        # rather than a second pass over fact_player_match_stats.
        if player_uid in squad_uids:
            _record_metric(
                con, backtest_run_id, season, gameweek, tier,
                f"realized_goals_assists:{player_uid}", outcome["goals"] + outcome["assists"],
            )

    for name, values in (
        ("log_score_clean_sheet_mean", cs_log), ("brier_clean_sheet_mean", cs_brier),
        ("log_score_goals_mean", goals_log), ("log_score_assists_mean", assists_log),
    ):
        if values:
            _record_metric(con, backtest_run_id, season, gameweek, tier, name, sum(values) / len(values))
    if compute_segments:
        for name, values_bp in (
            ("log_score_clean_sheet_mean", cs_log_bp), ("brier_clean_sheet_mean", cs_brier_bp),
            ("log_score_goals_mean", goals_log_bp), ("log_score_assists_mean", assists_log_bp),
        ):
            _record_segment_metrics(con, backtest_run_id, season, gameweek, tier, name, values_bp, segment_of)

    # -------- per-squad-player realized total points (raw material for M5's realized_sharpe) --------
    for player_uid in squad_uids:
        row = con.execute(
            "SELECT event_points FROM fact_player_season_stats WHERE player_uid = ? AND season = ? AND gw = ?",
            [player_uid, season, gameweek],
        ).fetchone()
        if row is not None and row[0] is not None:
            _record_metric(con, backtest_run_id, season, gameweek, tier, f"realized_points:{player_uid}", row[0])

    # -------- Priority 9c: beats-the-crowd benchmark --------
    if so_run_id is not None and ownership_params_version is not None:
        xi_uids = frozenset(r[0] for r in con.execute(
            "SELECT player_uid FROM squad_optimizer_selections WHERE run_id = ? AND in_xi", [so_run_id]
        ).fetchall())
        captain_row = con.execute(
            "SELECT player_uid FROM squad_optimizer_selections WHERE run_id = ? AND is_captain", [so_run_id]
        ).fetchone()
        captain_uid = captain_row[0] if captain_row else None
        model_points = _realized_xi_points(con, season, gameweek, xi_uids, captain_uid)
        avg_manager_points = _avg_manager_benchmark_points(con, season, gameweek, ep_model_version, ownership_params_version)
        if avg_manager_points is not None:
            _record_metric(con, backtest_run_id, season, gameweek, tier, "model_squad_realized_points", model_points)
            _record_metric(con, backtest_run_id, season, gameweek, tier, "avg_manager_benchmark_points", avg_manager_points)
            _record_metric(con, backtest_run_id, season, gameweek, tier, "beats_crowd_points_delta", model_points - avg_manager_points)


# ============================================================
# top-level orchestrator
# ============================================================

# Both seasons ship a full 38-gameweek Premier League calendar (confirmed: 2025-2026's own
# fact_match spans gameweek 1-38) -- 76 gameweeks total, matching the spec's own count.
ALL_SEASON_GAMEWEEKS = [("2024-2025", gw) for gw in range(1, 39)] + [("2025-2026", gw) for gw in range(1, 39)]


def run(
    con: duckdb.DuckDBPyConnection,
    *,
    xi_params_version: int,
    rho_params_version: int,
    decay_params_version: int,
    adjustment_params_version: int,
    shrinkage_params_version: int,
    fact_multiplier_params_version: int,
    scoring_params_version: int,
    bps_params_version: int,
    tau_params_version: int,
    rho_residual_params_version: int,
    corr_params_version: int,
    lambda_params_version: int,
    guardrail_params_version: int,
    n_antithetic_pairs: int = 2000,
    run_monte_carlo: bool = True,
    notes: str | None = None,
    compute_segments: bool = False,
    set_piece_params_version: int | None = None,
    ownership_params_version: int | None = None,
) -> int:
    """Full walk-forward pass over both historical seasons. Skips any (season, gameweek) that
    fails has_fittable_history() (2024-2025 GW1 in practice, per the cold-start guard) or that
    has_double_gameweek() (4 real historical gameweeks -- see that function's docstring for why
    DGWs are out of scope here, same as expected_points.py's own existing v1 scope boundary)
    rather than attempting and crashing -- warm_up_gameweeks records the total skipped, of
    either kind.

    compute_segments/set_piece_params_version/ownership_params_version: Priority 9b/9c
    opt-in, passed straight through to score_gameweek() -- see its own docstring. Default off,
    same backward-compatible convention as every other opt-in feature in this project."""
    steps = [
        (s, gw) for s, gw in ALL_SEASON_GAMEWEEKS
        if has_fittable_history(con, s, gw) and not has_double_gameweek(con, s, gw)
    ]
    warm_up_gameweeks = len(ALL_SEASON_GAMEWEEKS) - len(steps)

    backtest_run_id = con.execute(
        "INSERT INTO backtest_runs (warm_up_gameweeks, notes) VALUES (?, ?) RETURNING backtest_run_id",
        [warm_up_gameweeks, notes],
    ).fetchone()[0]

    for season, gameweek in steps:
        run_gameweek_step(
            con, backtest_run_id, season, gameweek,
            xi_params_version=xi_params_version, rho_params_version=rho_params_version,
            decay_params_version=decay_params_version, adjustment_params_version=adjustment_params_version,
            shrinkage_params_version=shrinkage_params_version, fact_multiplier_params_version=fact_multiplier_params_version,
            scoring_params_version=scoring_params_version, bps_params_version=bps_params_version, tau_params_version=tau_params_version,
            rho_residual_params_version=rho_residual_params_version, corr_params_version=corr_params_version,
            lambda_params_version=lambda_params_version, guardrail_params_version=guardrail_params_version,
            n_antithetic_pairs=n_antithetic_pairs, run_monte_carlo=run_monte_carlo,
        )
        ep_mv, mm_mv, ts_mv, so_run_id = con.execute(
            "SELECT ep_model_version, mm_model_version, ts_model_version, so_run_id FROM backtest_gameweek_steps "
            "WHERE backtest_run_id = ? AND season = ? AND gameweek = ?",
            [backtest_run_id, season, gameweek],
        ).fetchone()
        score_gameweek(
            con, backtest_run_id, season, gameweek, ep_mv, mm_mv, ts_mv, scoring_params_version, so_run_id=so_run_id,
            compute_segments=compute_segments, set_piece_params_version=set_piece_params_version,
            ownership_params_version=ownership_params_version,
        )

    return backtest_run_id


# ============================================================
# season simulation: an evolving M8 manager, not a fresh M5 solve every step.
#
# Every walk-forward mechanism above this point (run(), refit_lambda(), report_
# concentration_sensitivity()) re-solves squad_optimizer fresh at every single gameweek --
# exactly the gap the README's own Design notes name explicitly: "M7's walk-forward squad is
# M5's from-scratch pick every single step, not an evolving manager holding, so no equivalent
# 'what would the manager have owned at gameweek N' state exists ... to compare a wildcard's
# gain against." Season-long rank depends on cumulative points from ONE evolving squad making
# real transfer/chip decisions week to week, which is what this section actually builds --
# reusing asof_scope() (extended above with schedule_horizon_gameweeks, not replaced) and
# M8's own transfer_planner.run()/apply_recommendation() unmodified, not a parallel mechanism.
# ============================================================

CHIP_PRIORITY = ("wildcard", "free_hit", "bench_boost", "triple_captain")

# Which chip_evaluations.detail field carries each chip's own per-visible-gameweek value
# trajectory (see evaluate_wildcard()/evaluate_free_hit()/evaluate_triple_captain()/
# evaluate_bench_boost()'s own docstrings for how each is built -- all reuse already-computed
# horizon EP, zero extra solves/simulations), and whether "now" being genuinely the best time
# to play means the LOWEST point on that trajectory (wildcard/free_hit -- the current squad's
# own value is at its worst, the real situation a rebuild exists to fix) or the HIGHEST
# (bench_boost/triple_captain -- the bench's or the captain candidate's own value peaks now).
CHIP_TIMING_FIELD = {
    "wildcard": ("current_squad_value_per_gw", "min"),
    "free_hit": ("current_xi_value_per_gw", "min"),
    "bench_boost": ("all_gameweeks", "max"),
    "triple_captain": ("captain_value_per_gw", "max"),
}


def _is_best_gameweek_in_visible_horizon(per_gw: dict, target_gameweek: int, prefer: str) -> bool:
    """A real, if myopic, "is now better than waiting" signal built entirely from the model's
    own already-computed forward EP for the gameweeks currently visible in the planning
    horizon -- never a peek at a chip's actual realized future outcome (nothing here reads a
    real result; per_gw is projected mu, computed the same asof-safe way as everything else
    inside run_season_simulation()'s own asof_scope). "Visible" is the operative word: at
    gameweek G the model only ever sees [G, G+horizon_gameweeks), so this can only ever compare
    against the SAME window every other part of this planning call already can see -- it
    genuinely cannot know whether gameweek G+10 will be better than G+2 when planning at G,
    exactly as a real manager's own forward-looking judgement is bounded too.

    Missing/empty per_gw (a chip evaluator call that had nothing to compare against, e.g. an
    old-shape or ex-{} detail payload) or target_gameweek absent from it (a rare edge case --
    the DB dict's own JSON keys are stringified ints, handled below) defers to True: "can't
    assess timing here, fall back to the existing threshold-only check" rather than silently
    suppressing an otherwise-real recommendation just because comparison data is missing.
    """
    per_gw = {int(k): v for k, v in per_gw.items()} if per_gw else {}
    if not per_gw or target_gameweek not in per_gw:
        return True
    epsilon = 1e-9
    if prefer == "min":
        return per_gw[target_gameweek] <= min(per_gw.values()) + epsilon
    return per_gw[target_gameweek] >= max(per_gw.values()) - epsilon


def _decide_gameweek_action(
    con: duckdb.DuckDBPyConnection, plan_run_id: int, chips_used_set1: set, chips_used_set2: set,
    target_gameweek: int, accept_transfer_if_net_value_above: float,
) -> tuple[int | None, str | None]:
    """The harness's own explicit decision rule, deliberately simple and auditable rather than
    a second optimization layer: accept the #1 ranked transfer iff its net_value clears
    accept_transfer_if_net_value_above (default 0.0 -- any genuine expected gain), or accept
    one recommended chip in a fixed priority order (CHIP_PRIORITY) skipping any chip already
    spent from the set covering this gameweek. Never both a transfer and a chip in the same
    week -- Wildcard/Free Hit already are the "transfer" for that week, and combining Bench
    Boost/Triple Captain with an ordinary transfer in the same week is a real thing an expert
    manager sometimes does, but deliberately out of scope for this v1 decision rule (named
    here, not silently modeled as if it were handled).

    Set-1 gameweeks (target_gameweek < GW19) also weigh a recommended chip's current-week
    value against holding it (see _is_best_gameweek_in_visible_horizon() above): clearing
    evaluate_*()'s own gain threshold answers "is this worth it at all," this answers "is now
    actually better than waiting," a genuinely different question a fixed per-week threshold
    alone can never answer. A chip that clears its threshold but isn't (per the model's own
    currently-visible horizon) the best week to play it is held, not taken -- the loop tries
    the next-priority chip instead, then falls through to a transfer, exactly as when nothing
    was recommended at all. Set-2 gameweeks (GW19+) keep the original threshold-only check --
    deliberately out of scope for this round, named here rather than silently extended."""
    is_set1 = target_gameweek < transfer_planner.GW19_DEADLINE_GAMEWEEK
    used_this_set = chips_used_set1 if is_set1 else chips_used_set2
    rows = con.execute(
        "SELECT chip_type, recommended, detail FROM chip_evaluations WHERE run_id = ?", [plan_run_id]
    ).fetchall()
    recommended = {chip_type: json.loads(detail or "{}") for chip_type, is_rec, detail in rows if is_rec}

    for candidate in CHIP_PRIORITY:
        if candidate not in recommended or candidate in used_this_set:
            continue
        if is_set1:
            field, prefer = CHIP_TIMING_FIELD[candidate]
            per_gw = recommended[candidate].get(field, {})
            if not _is_best_gameweek_in_visible_horizon(per_gw, target_gameweek, prefer):
                continue  # a later week within the model's currently-visible horizon looks better -- hold
        return None, candidate

    top = con.execute(
        "SELECT rank, net_value FROM transfer_recommendations WHERE run_id = ? ORDER BY rank LIMIT 1", [plan_run_id]
    ).fetchone()
    if top and top[1] > accept_transfer_if_net_value_above:
        return top[0], None
    return None, None


def run_season_simulation(
    con: duckdb.DuckDBPyConnection,
    season: str,
    start_gameweek: int,
    end_gameweek: int,
    *,
    xi_params_version: int,
    rho_params_version: int,
    decay_params_version: int,
    adjustment_params_version: int,
    shrinkage_params_version: int,
    fact_multiplier_params_version: int,
    scoring_params_version: int,
    bps_params_version: int,
    tau_params_version: int,
    rho_residual_params_version: int,
    corr_params_version: int,
    lambda_params_version: int,
    guardrail_params_version: int,
    horizon_params_version: int,
    transfer_cost_params_version: int,
    wildcard_threshold_params_version: int,
    free_hit_threshold_params_version: int,
    kappa_tc_params_version: int,
    accept_transfer_if_net_value_above: float = 0.0,
    n_antithetic_pairs: int = 2000,
) -> dict:
    """Bootstraps a real M5 squad at start_gameweek, then walks forward to end_gameweek making
    one real M8 transfer_planner.run()-informed decision per gameweek (see
    _decide_gameweek_action()), applying it via apply_recommendation() -- one continuously
    evolving squad across the whole window, mirroring how a real manager actually plays,
    instead of a fresh from-scratch squad every gameweek.

    Every planning call for gameweek G+1 runs inside asof_scope(con, season, G+1,
    schedule_horizon_gameweeks=horizon_gameweeks) -- pinned to G+1's OWN deadline, after G's
    results are known, with G+1's own horizon of fixture schedules (not results) visible, the
    same asof-safety guarantee every M7 walk-forward step already carries, extended (not
    bypassed) to cover a multi-gameweek horizon. Real historical Double Gameweeks are skipped
    for planning (same has_double_gameweek() v1 scope boundary M7's own walk-forward loop
    already applies -- squad_optimizer_selections' primary key cannot represent a DGW player's
    duplicate ep_outputs rows) but NOT for scoring: that gameweek's real, already-aggregated
    event_points total is still read and counted, holdings just don't change that week.

    Chip scoring effects that don't touch persisted holdings are applied for exactly the one
    gameweek they're accepted: Free Hit scores off its own fresh one-off squad (never
    persisted -- see apply_recommendation()'s own docstring for why leaving holdings unchanged
    on accept_chip="free_hit" is correct, not a gap); Bench Boost scores the full 15-player
    squad instead of just the XI; Triple Captain triples (not doubles) the captain's points.
    Wildcard is the one chip that DOES persist -- apply_recommendation() already rebuilds
    holdings for it, so the following gameweek's scoring reads it like any other transfer.

    Returns {"weekly_points": [...], "gameweeks": [...], "final_state_version": int,
    "actions": [{"gameweek", "action", "detail"} ...], "skipped_dgw_gameweeks": [...]}
    -- actions is the real per-gameweek decision log, for auditing what the simulated manager
    actually did, not just the final score.
    """
    if not has_fittable_history(con, season, start_gameweek):
        raise ValueError(f"{season} GW{start_gameweek} has insufficient prior history to bootstrap from -- pick a later start_gameweek")
    horizon_gameweeks, _ = params_mod.resolve_param(con, "planning_horizon_params", "horizon_gameweeks", horizon_params_version)
    horizon_gameweeks = int(horizon_gameweeks)

    with asof_scope(con, season, start_gameweek, schedule_horizon_gameweeks=horizon_gameweeks) as deadline:
        calibration_asof_date = deadline.date()
        ts_mv = team_strength.calibrate(
            con, calibration_asof_date, xi_params_version, rho_params_version,
            target_season=season, fit_seasons=fit_seasons_for(season),
        )
        mm_mv = minutes_model.run(
            con, calibration_asof_date, season, decay_params_version, adjustment_params_version,
            shrinkage_params_version, fact_multiplier_params_version,
        )
        ep_mv = ep.run(
            con, calibration_asof_date, season, start_gameweek, ts_mv, mm_mv,
            scoring_params_version, bps_params_version, tau_params_version,
        )
        un_mv = uncertainty.run(
            con, calibration_asof_date, ep_mv, mm_mv, ts_mv, scoring_params_version, bps_params_version,
            tau_params_version, rho_residual_params_version, corr_params_version,
        )
        bootstrap_run_id = squad_optimizer.run(
            con, calibration_asof_date, season, start_gameweek, ep_mv, un_mv,
            lambda_params_version, guardrail_params_version,
        )
        # Real look-ahead leak, fixed here: bootstrap_from_squad_optimizer_run() -> its own
        # _compute_bank_for_squad() prices each held player via `ORDER BY gw DESC` with no
        # ceiling -- correct for a real live run (no future gameweeks exist yet to leak from),
        # but this call used to sit AFTER the with-block exited, so inside a season simulation
        # walking historical gameweeks it read main.fact_player_season_stats completely
        # unshadowed -- the manager's very first bank figure could be computed from a LATER
        # gameweek's price than start_gameweek's own deadline. Kept inside the shadow now, so
        # its unqualified fact_player_season_stats read resolves to the same asof-safe TEMP
        # TABLE (season = ? AND gw < start_gameweek) every other M1-M5 call in this block
        # already gets -- no change needed in transfer_planner.py itself, this composes with
        # the existing mechanism exactly as asof_scope()'s own docstring promises.
        state_version = transfer_planner.bootstrap_from_squad_optimizer_run(con, bootstrap_run_id)

    weekly_points: list[float] = []
    gameweeks_scored: list[int] = []
    actions: list[dict] = []
    skipped_dgw: list[int] = []

    for gw in range(start_gameweek, end_gameweek + 1):
        accept_chip = None
        free_hit_squad = None

        if has_double_gameweek(con, season, gw):
            skipped_dgw.append(gw)
        elif gw > start_gameweek:
            state_row = con.execute(
                "SELECT free_transfers_available, chips_used_set1, chips_used_set2, bank FROM manager_state_versions WHERE state_version = ?",
                [state_version],
            ).fetchone()
            _fts, chips_used_set1_json, chips_used_set2_json, _bank = state_row
            chips_used_set1 = set(json.loads(chips_used_set1_json))
            chips_used_set2 = set(json.loads(chips_used_set2_json))

            with asof_scope(con, season, gw, schedule_horizon_gameweeks=horizon_gameweeks) as deadline:
                calibration_asof_date = deadline.date()
                ts_mv = team_strength.calibrate(
                    con, calibration_asof_date, xi_params_version, rho_params_version,
                    target_season=season, fit_seasons=fit_seasons_for(season),
                )
                mm_mv = minutes_model.run(
                    con, calibration_asof_date, season, decay_params_version, adjustment_params_version,
                    shrinkage_params_version, fact_multiplier_params_version,
                )
                plan_run_id = transfer_planner.run(
                    con, calibration_asof_date, season, gw, state_version, ts_mv, mm_mv,
                    horizon_params_version, scoring_params_version, bps_params_version, tau_params_version,
                    rho_residual_params_version, corr_params_version, transfer_cost_params_version,
                    lambda_params_version, guardrail_params_version, wildcard_threshold_params_version,
                    free_hit_threshold_params_version, kappa_tc_params_version,
                )
                accept_transfer_rank, accept_chip = _decide_gameweek_action(
                    con, plan_run_id, chips_used_set1, chips_used_set2, gw, accept_transfer_if_net_value_above,
                )

                if accept_chip == "free_hit":
                    free_hit_squad = transfer_planner.read_fresh_chip_squad(con, plan_run_id, "free_hit")

                # Same look-ahead leak as the bootstrap call above, same fix: apply_recommendation()'s
                # Wildcard-accept branch calls _compute_bank_for_squad() too, and this call used to
                # sit after the with-block exited -- kept inside the shadow now so a Wildcard
                # accepted while walking through gameweek gw prices the fresh squad off gw's own
                # asof-safe price snapshot (gw' < gw), never a later gameweek's price. No effect on
                # the non-Wildcard paths (they never call _compute_bank_for_squad() at all).
                state_version = transfer_planner.apply_recommendation(
                    con, plan_run_id, accept_transfer_rank=accept_transfer_rank, accept_chip=accept_chip,
                )
            actions.append({
                "gameweek": gw, "accepted_transfer_rank": accept_transfer_rank, "accepted_chip": accept_chip,
                "plan_run_id": plan_run_id,
            })

        holdings = transfer_planner._read_holdings(con, state_version)
        if accept_chip == "free_hit" and free_hit_squad is not None:
            xi_uids = frozenset(h["player_uid"] for h in free_hit_squad if h["in_xi"])
            captain_uid = next((h["player_uid"] for h in free_hit_squad if h["is_captain"]), None)
            captain_multiplier = 2
        elif accept_chip == "bench_boost":
            xi_uids = frozenset(h["player_uid"] for h in holdings)  # full 15, not just the XI
            captain_uid = next((h["player_uid"] for h in holdings if h["is_captain"]), None)
            captain_multiplier = 2
        else:
            xi_uids = frozenset(h["player_uid"] for h in holdings if h["in_xi"])
            captain_uid = next((h["player_uid"] for h in holdings if h["is_captain"]), None)
            captain_multiplier = 3 if accept_chip == "triple_captain" else 2

        points = _realized_xi_points(con, season, gw, xi_uids, captain_uid, captain_multiplier=captain_multiplier)
        weekly_points.append(points)
        gameweeks_scored.append(gw)

    return {
        "weekly_points": weekly_points, "gameweeks": gameweeks_scored, "final_state_version": state_version,
        "actions": actions, "skipped_dgw_gameweeks": skipped_dgw, "bootstrap_run_id": bootstrap_run_id,
    }


def report_season_simulation_sensitivity(
    con: duckdb.DuckDBPyConnection,
    season: str,
    start_gameweek: int,
    end_gameweek: int,
    base_versions: dict,
    *,
    lambda_grid: tuple[float, ...] = (),
    guardrail_cap_grid: tuple[float, ...] = (),
    effective_date: str = "2026-08-11",
) -> dict:
    """Read-only reporting, deliberately separate from any writing/proposal mechanism --
    mirrors report_concentration_sensitivity()'s own read-only pattern, extended to season-long
    metrics. Runs run_season_simulation() once per lambda_grid candidate (guardrail_cap held
    fixed at base_versions' own pinned value, for exactly the reason refit_lambda() already
    holds it fixed: re-tuning a redundant backstop against the same signal the primary risk
    dial is tuned against would erode the protection it exists for) and once per
    guardrail_cap_grid candidate (lambda held fixed, same reasoning in reverse) -- never both
    varied together, matching the existing precedent's own two-separate-questions split, not a
    full cross-product grid (season_simulation is materially more expensive per point than a
    single-gameweek solve: a real MIQP solve plus Monte Carlo simulation every gameweek in the
    window, not once).

    base_versions must carry exactly the keyword arguments run_season_simulation() itself
    requires (including its own lambda_params_version/guardrail_params_version, held fixed on
    whichever side of a given sweep isn't the one varying). Each grid candidate gets its own
    freshly written param_versions row (via params_mod.write_param(), the normal immutable-
    versioning mechanism -- writing a version never activates it) rather than mutating the live
    pinned version, so nothing here can accidentally change what a real run resolves.

    This does not, on its own, justify changing lambda_value or xi_club_concentration_cap --
    that requires the same real-data discipline the README's own Design notes already insist on
    for the existing cross-sectional lambda finding (a synthetic run is honest evidence the
    MACHINERY works, not evidence about what the real pinned values should be).
    """
    results: dict[str, dict] = {"lambda": {}, "guardrail_cap": {}}
    if lambda_grid:
        for lam in lambda_grid:
            trial_version = _next_param_version(con, "risk_aversion_params")
            params_mod.write_param(con, "risk_aversion_params", trial_version, effective_date, "lambda_value", value_numeric=lam)
            trial_versions = {**base_versions, "lambda_params_version": trial_version}
            sim = run_season_simulation(con, season, start_gameweek, end_gameweek, **trial_versions)
            results["lambda"][lam] = {**season_cumulative_metrics(sim["weekly_points"]), "actions": sim["actions"]}

    if guardrail_cap_grid:
        for cap in guardrail_cap_grid:
            trial_version = _next_param_version(con, "squad_optimizer_guardrail_params")
            params_mod.write_param(
                con, "squad_optimizer_guardrail_params", trial_version, effective_date,
                "xi_club_concentration_cap", value_numeric=cap,
            )
            trial_versions = {**base_versions, "guardrail_params_version": trial_version}
            sim = run_season_simulation(con, season, start_gameweek, end_gameweek, **trial_versions)
            results["guardrail_cap"][cap] = {**season_cumulative_metrics(sim["weekly_points"]), "actions": sim["actions"]}

    return results


# ============================================================
# beats_baseline: does the model's architecture actually beat simpler alternatives, over the
# same walked window and the same asof_scope() discipline -- the headline question the elaborate
# M1-M6 pipeline's edge was previously ASSERTED to answer, not measured. Three baselines:
#   1. "recent_points"        -- pick by recent real form, no model at all.
#   2. "ownership_popularity" -- pick the most-owned ("template") squad, no model at all.
#   3. "crowd"                -- the ownership-weighted average manager (Priority 9c, already
#                                 built as a per-gameweek metric; this is where it becomes an
#                                 actual season-level comparator instead of just a metric).
# ============================================================

def _naive_candidate_pool(
    con: duckdb.DuckDBPyConnection, target_season: str, signal: str, lookback_gameweeks: int = 3,
) -> list[dict]:
    """A squad_optimizer.solve()-compatible candidate pool ({"player_uid","position","mu","var",
    "club","price"}) built from real, already-known data only -- NOT ep_outputs, no model
    involved at all, per this being a genuinely model-free baseline. Reads fact_player_season_stats
    by bare name, so it inherits whatever asof_scope() shadow (or lack of one) is active at the
    call site -- callers MUST call this inside the SAME asof_scope() the real model's own
    candidate pool would be built in, for an apples-to-apples "what was knowable at this
    deadline" comparison, never main.* directly.

    signal="recent_points": mu = mean of the last lookback_gameweeks real event_points (a
    rolling form metric). signal="ownership": mu = the most recent real selected_by_percent (a
    popularity/"template team" pick). Either way mu feeds squad_optimizer.solve() the exact
    same way ep_total normally does -- reusing the real M5 budget/formation/club-cap machinery
    gives a fair, same-constraints comparison, not a strawman unconstrained ranking. var is
    always 0.0 (a naive baseline carries no risk model; solve() is always called with lam=0.0
    for these pools regardless, making var moot).

    A player missing from the resulting mu map (no recent_points history yet, or no ownership
    data yet) is excluded from the pool entirely -- missing data is never treated as a
    fabricated 0, matching this project's discipline everywhere else.
    """
    if signal not in ("recent_points", "ownership"):
        raise ValueError(f"signal must be 'recent_points' or 'ownership', got {signal!r}")

    team_of = dict(con.execute(
        "SELECT DISTINCT player_uid, team_code FROM player_alias WHERE season = ?", [target_season],
    ).fetchall())
    positions = dict(con.execute("SELECT player_uid, position FROM dim_player").fetchall())
    prices = dict(con.execute(
        "SELECT player_uid, now_cost FROM fact_player_season_stats WHERE season = ? AND now_cost IS NOT NULL "
        "QUALIFY row_number() OVER (PARTITION BY player_uid ORDER BY gw DESC) = 1",
        [target_season],
    ).fetchall())

    if signal == "ownership":
        mu_by_uid = dict(con.execute(
            "SELECT player_uid, selected_by_percent FROM fact_player_season_stats "
            "WHERE season = ? AND selected_by_percent IS NOT NULL "
            "QUALIFY row_number() OVER (PARTITION BY player_uid ORDER BY gw DESC) = 1",
            [target_season],
        ).fetchall())
    else:
        rows = con.execute(
            "SELECT player_uid, event_points FROM fact_player_season_stats "
            "WHERE season = ? AND event_points IS NOT NULL "
            "QUALIFY row_number() OVER (PARTITION BY player_uid ORDER BY gw DESC) <= ?",
            [target_season, lookback_gameweeks],
        ).fetchall()
        by_player: dict[str, list[float]] = {}
        for player_uid, event_points in rows:
            by_player.setdefault(player_uid, []).append(event_points)
        mu_by_uid = {uid: sum(vals) / len(vals) for uid, vals in by_player.items()}

    candidates = []
    for player_uid, price in prices.items():
        position = positions.get(player_uid)
        club = team_of.get(player_uid)
        mu = mu_by_uid.get(player_uid)
        if position not in squad_optimizer.POSITIONS or club is None or mu is None:
            continue
        candidates.append({
            "player_uid": player_uid, "position": position, "mu": mu, "var": 0.0, "club": club, "price": price,
        })
    return candidates


def beats_baseline(
    con: duckdb.DuckDBPyConnection,
    season: str,
    start_gameweek: int,
    end_gameweek: int,
    model_gameweeks: list[int],
    model_weekly_points: list[float],
    *,
    xi_params_version: int,
    rho_params_version: int,
    decay_params_version: int,
    adjustment_params_version: int,
    shrinkage_params_version: int,
    fact_multiplier_params_version: int,
    scoring_params_version: int,
    bps_params_version: int,
    tau_params_version: int,
    ownership_params_version: int,
    guardrail_cap: float = 3.0,
    recent_points_lookback_gameweeks: int = 3,
) -> dict:
    """Scores three model-free baselines over [start_gameweek, end_gameweek] using the SAME
    asof_scope() discipline every other walk-forward step in this module uses, then compares
    each against model_gameweeks/model_weekly_points -- the trajectory an already-completed
    run_season_simulation() call produced for the SAME window (passed in rather than
    recomputed here, since re-running the real M5 MIQP + M6 Monte Carlo a second time just to
    get a number this function already has cheaper access to would be pure waste; this
    function's own walk only needs M1-M3, not M5/M6, for any of the three baselines).

    Per gameweek: recent_points and ownership_popularity each pick a squad via
    squad_optimizer.solve() against a _naive_candidate_pool() (lam=0.0 -- no risk model, and
    none of these three baselines needs squad_optimizer_runs/selections rows persisted, since
    nothing downstream ever needs to look one back up), scored on that squad's REAL realized
    points the same way run_season_simulation()'s own model squad is (_realized_xi_points(),
    called after the asof_scope() block exits so it reads real, no-longer-shadowed results).
    crowd reuses _avg_manager_benchmark_points() unchanged (Priority 9c) -- already a real,
    computed season-level number, just not previously compared against the model's own
    trajectory at the season level.

    A gameweek where a baseline's own candidate pool can't fill a legal squad (SCIP status not
    in ('optimal','timelimit') -- e.g. genuinely early-season, too few players have
    lookback_gameweeks of real history yet) is skipped for THAT baseline only, not treated as a
    fabricated 0 -- n_gameweeks/n_gameweeks_skipped report this honestly. model_beats_baseline_by
    is computed only over the gameweeks a baseline actually has data for (the model's own points
    on those SAME gameweeks, not its full-window total), so a baseline with partial coverage is
    never penalized twice -- once for its own missing weeks, again for a lopsided total
    comparison.
    """
    if len(model_gameweeks) != len(model_weekly_points):
        raise ValueError("model_gameweeks and model_weekly_points must be the same length")
    model_points_by_gw = dict(zip(model_gameweeks, model_weekly_points))

    baseline_weekly: dict[str, list[float]] = {"recent_points": [], "ownership_popularity": [], "crowd": []}
    baseline_gameweeks: dict[str, list[int]] = {"recent_points": [], "ownership_popularity": [], "crowd": []}

    for gw in range(start_gameweek, end_gameweek + 1):
        with asof_scope(con, season, gw) as deadline:
            calibration_asof_date = deadline.date()
            ts_mv = team_strength.calibrate(
                con, calibration_asof_date, xi_params_version, rho_params_version,
                target_season=season, fit_seasons=fit_seasons_for(season),
            )
            mm_mv = minutes_model.run(
                con, calibration_asof_date, season, decay_params_version, adjustment_params_version,
                shrinkage_params_version, fact_multiplier_params_version,
            )
            ep_mv = ep.run(
                con, calibration_asof_date, season, gw, ts_mv, mm_mv,
                scoring_params_version, bps_params_version, tau_params_version,
            )

            recent_pool = _naive_candidate_pool(con, season, "recent_points", recent_points_lookback_gameweeks)
            recent_solve = (
                squad_optimizer.solve(recent_pool, {}, 0.0, guardrail_cap) if len(recent_pool) >= 15 else None
            )
            ownership_pool = _naive_candidate_pool(con, season, "ownership")
            ownership_solve = (
                squad_optimizer.solve(ownership_pool, {}, 0.0, guardrail_cap) if len(ownership_pool) >= 15 else None
            )

        # _avg_manager_benchmark_points needs THIS gameweek's own real selected_by_percent/
        # event_points (fps.gw = the current gameweek) -- asof_scope()'s shadow truncates to
        # gw < the current gameweek by design, so calling it inside the block above would
        # always see nothing for gw itself and silently return None every time (a real bug
        # this exact reasoning caught before it shipped). Called out here, after the shadow
        # drops, the same way _realized_xi_points() below reads real, no-longer-shadowed
        # results -- ep_mv itself is still valid to use here, it names a real, already-written
        # ep_outputs model_version regardless of which schema temp/main currently resolves to.
        crowd_points = _avg_manager_benchmark_points(con, season, gw, ep_mv, ownership_params_version)

        if recent_solve is not None and recent_solve["status"] in ("optimal", "timelimit"):
            baseline_weekly["recent_points"].append(_realized_xi_points(con, season, gw, recent_solve["xi"], recent_solve["captain"]))
            baseline_gameweeks["recent_points"].append(gw)
        if ownership_solve is not None and ownership_solve["status"] in ("optimal", "timelimit"):
            baseline_weekly["ownership_popularity"].append(_realized_xi_points(con, season, gw, ownership_solve["xi"], ownership_solve["captain"]))
            baseline_gameweeks["ownership_popularity"].append(gw)
        if crowd_points is not None:
            baseline_weekly["crowd"].append(crowd_points)
            baseline_gameweeks["crowd"].append(gw)

    total_window_gameweeks = end_gameweek - start_gameweek + 1
    summary = {"model": {"total_points": sum(model_weekly_points), "n_gameweeks": len(model_gameweeks)}}
    for name, weekly in baseline_weekly.items():
        gws = baseline_gameweeks[name]
        comparable_model_total = sum(model_points_by_gw[gw] for gw in gws if gw in model_points_by_gw)
        comparable_baseline_total = sum(pts for gw, pts in zip(gws, weekly) if gw in model_points_by_gw)
        summary[name] = {
            "total_points": sum(weekly),
            "n_gameweeks": len(gws),
            "n_gameweeks_skipped": total_window_gameweeks - len(gws),
            "model_total_points_same_gameweeks": comparable_model_total,
            "model_beats_baseline_by": comparable_model_total - comparable_baseline_total,
        }
    return summary


# ============================================================
# recalibration: proposal-writing gate + per-family refit techniques
# ============================================================

def _next_param_version(con: duckdb.DuckDBPyConnection, param_family: str) -> int:
    row = con.execute("SELECT max(param_version) FROM param_versions WHERE param_family = ?", [param_family]).fetchone()
    return (row[0] or 0) + 1


def propose_recalibration(
    con: duckdb.DuckDBPyConnection,
    backtest_run_id: int,
    param_family: str,
    param_key: str,
    new_value: float,
    metric_name: str,
    metric_before: float,
    metric_after: float,
    *,
    dimensions: dict | None = None,
    old_params_version: int | None = None,
    effective_date: str = "2026-08-11",
) -> int:
    """Writes a candidate value as a normal new immutable param_versions row (write_param() is
    unchanged -- writing a version never activates it, resolve_param() is explicit-version-only
    per params.py's own docstring) plus one recalibration_proposals row recording *why* --
    status stays 'pending' until a human reviews it (see review_recalibration.py). Activating a
    confirmed proposal still means a human editing the explicit version-number argument
    scripts/run_ingestion.py passes for that family -- the same discipline as every other
    version bump in this project, not a new mechanism.
    """
    new_params_version = _next_param_version(con, param_family)
    old_value = None
    if old_params_version is not None:
        try:
            old_value, _ = params_mod.resolve_param(con, param_family, param_key, old_params_version, dimensions=dimensions)
        except params_mod.ParamNotFoundError:
            old_value = None

    params_mod.write_param(con, param_family, new_params_version, effective_date, param_key, value_numeric=new_value, dimensions=dimensions)

    return con.execute(
        """
        INSERT INTO recalibration_proposals
            (backtest_run_id, param_family, param_key, dimensions, old_params_version, new_params_version,
             old_value, new_value, metric_name, metric_before, metric_after)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        RETURNING proposal_id
        """,
        [backtest_run_id, param_family, param_key, json.dumps(dimensions, sort_keys=True) if dimensions else None,
         old_params_version, new_params_version, old_value, new_value, metric_name, metric_before, metric_after],
    ).fetchone()[0]


def write_recalibration_seed_file(con: duckdb.DuckDBPyConnection, backtest_run_id: int, seed_dir: Path | str) -> Path:
    """Writes every recalibration_proposals row for this backtest_run_id to a committed JSON
    file (seed_dir/seeds_<backtest_run_id>.json) -- a durable copy of what a real backtest run
    found, independent of the DuckDB file itself.

    The exact gap this closes: fact_type_multiplier_params v8 / minutes_model_shrinkage_params
    v10's real winning values only ever existed as param_versions rows inside a local
    db/fpl_quant_v2.duckdb that was later lost, with no other durable record of what those
    values actually were (see README's Design notes and run_ingestion.py's own comment on the
    same incident) -- a git-committed seed file survives exactly that failure mode.

    Written status is a live read of recalibration_proposals.status at call time, not frozen at
    'pending' -- review_recalibration.py's set_status() re-calls this after a human
    confirms/rejects a proposal, so the committed file and the DB never drift out of sync.
    load_confirmed_recalibration_seeds() only ever reads 'confirmed' entries back out, so a
    'pending' or 'rejected' proposal captured here is disclosed provenance, not an activated
    default -- the same human-gate discipline recalibration_proposals.status already enforces
    in the DB, applied identically to its file-backed copy.
    """
    rows = con.execute(
        "SELECT proposal_id, param_family, param_key, dimensions, old_params_version, new_params_version, "
        "old_value, new_value, metric_name, metric_before, metric_after, status, reviewed_by, reviewed_at "
        "FROM recalibration_proposals WHERE backtest_run_id = ? ORDER BY proposal_id",
        [backtest_run_id],
    ).fetchall()
    proposals = []
    for (proposal_id, family, key, dims, old_v, new_v, old_val, new_val, metric_name,
         metric_before, metric_after, status, reviewed_by, reviewed_at) in rows:
        proposals.append({
            "proposal_id": proposal_id, "param_family": family, "param_key": key,
            "dimensions": json.loads(dims) if dims else None,
            "old_params_version": old_v, "new_params_version": new_v,
            "old_value": old_val, "new_value": new_val,
            "metric_name": metric_name, "metric_before": metric_before, "metric_after": metric_after,
            "status": status, "reviewed_by": reviewed_by,
            "reviewed_at": reviewed_at.isoformat() if reviewed_at else None,
        })
    seed_dir = Path(seed_dir)
    seed_dir.mkdir(parents=True, exist_ok=True)
    out_path = seed_dir / f"seeds_{backtest_run_id}.json"
    out_path.write_text(json.dumps(
        {"backtest_run_id": backtest_run_id, "written_at": datetime.now(timezone.utc).isoformat(), "proposals": proposals},
        indent=2, sort_keys=True,
    ) + "\n")
    return out_path


def resolve_active_version(
    param_family: str, default_version: int, seed_dir: Path | str, param_key: str | tuple[str, ...],
) -> int:
    """Roadmap P1 item (Track B, docs/plans/2026-08_roadmap_plan.md [A3]/[A10]): the single
    source of truth for "what params_version is currently active" for a recalibratable
    (param_family, param_key) -- the highest new_params_version among CONFIRMED recalibration
    seeds for it (see load_confirmed_recalibration_seeds()'s own docstring: only 'confirmed'
    rows are ever returned), or default_version if none has ever been confirmed.

    This deliberately never touches the DB -- it reads only the git-committed seed files, so it
    gives the identical answer on a fresh checkout as it does mid-run (per [A10]: a DB table
    alone doesn't survive between separate scheduled_pipeline.yml/weekly_backtest.yml jobs, so
    the committed seed files, not the DB, have to be the real source of truth).

    param_key as a tuple is for the one real case (minutes_adjustment_params, whose 'magnitude'
    and 'cap' keys share one version-argument, adjustment_params_version) where a single
    version-argument governs more than one param_key. Pass the family's COMPLETE, fixed set of
    shared keys (e.g. ("magnitude", "cap")) -- not "whatever's been proposed so far" -- see the
    correctness note below for why that distinction is load-bearing.

    CRITICAL correctness note for the tuple case (found in review before this shipped):
    propose_recalibration() assigns each (family, key) proposal its own next-available
    new_params_version via _next_param_version(), which is scoped to the FAMILY, not the
    (family, key) pair -- so when magnitude and cap are proposed together (recalibrate()'s own
    MINUTES_PARAM_GRIDS loop does exactly this), they get two DIFFERENT version numbers (e.g. N
    and N+1), even though minutes_model.run() resolves both keys at the SAME
    adjustment_params_version. Matching "any confirmed key present" -- or even "every key
    confirmed so far agrees," if one of the two has simply never been proposed yet -- would
    therefore often pick a version where the OTHER key has no row at all, and resolve_param()
    hard-errors (no fallback) on it, crashing the whole real-data pipeline the first time this
    family is ever confirmed. So this only returns a version that is confirmed for EVERY key in
    the caller's supplied param_key tuple (never a version safe for only some of them). In
    practice this means adjustment_params_version stays at default_version until both magnitude
    and cap have been confirmed at the same version number -- which propose_recalibration()'s
    current per-family (not per-family-per-shared-arg) versioning never produces on its own; a
    real, disclosed limitation (docs/plans/2026-08_roadmap_plan.md's Open Items), not silently
    papered over.

    Fixes a real, already-existing inconsistency this project had before Track B: several
    scripts already hand-updated to xi_params_version=2/rho_residual_params_version=2 after
    those were confirmed (see run_ingestion.py's own comment on commit 7bf7604), while others
    (run_backtest.py, run_season_simulation.py, run_transfer_planner_for_real_squad.py, among
    others) were never updated and still passed the stale default=1 literal. Every call site
    resolving through this same function is what actually closes that gap, not just automates
    future promotions.
    """
    confirmed = [s for s in load_confirmed_recalibration_seeds(seed_dir) if s["param_family"] == param_family]
    if isinstance(param_key, str):
        versions = [s["new_params_version"] for s in confirmed if s["param_key"] == param_key]
        return max(versions, default=default_version)

    versions_by_key = [
        {s["new_params_version"] for s in confirmed if s["param_key"] == key} for key in param_key
    ]
    common_versions = set.intersection(*versions_by_key) if versions_by_key else set()
    return max(common_versions, default=default_version)


RECALIBRATABLE_VERSION_ARGS: dict[str, tuple[str, str | tuple[str, ...]]] = {
    # version-argument name -> (param_family, param_key). A tuple param_key means "this
    # version-argument governs every one of these keys together" -- only true for
    # minutes_adjustment_params (magnitude/cap share one version_field), and only safe because
    # resolve_active_version() requires ALL of them confirmed at the same version -- see its
    # own docstring for why "any key" would be unsafe here.
    "xi_params_version": ("model_decay_params", "xi"),
    "rho_params_version": ("model_decay_params", "rho"),
    "rho_residual_params_version": ("correlation_params", "rho_residual"),
    "fact_multiplier_params_version": ("fact_type_multiplier_params", "multiplier"),
    "shrinkage_params_version": ("minutes_model_shrinkage_params", "competitive_matches_threshold"),
    "adjustment_params_version": ("minutes_adjustment_params", ("magnitude", "cap")),
    "lambda_params_version": ("risk_aversion_params", "lambda_value"),
    "kappa_tc_params_version": ("tc_risk_aversion_params", "kappa_tc"),
}


def active_recalibratable_versions(seed_dir: Path | str, default_version: int = 1) -> dict[str, int]:
    """Roadmap P1 item (Track B): every real-data script that needs one or more of the 8
    recalibratable version-arguments (RECALIBRATABLE_VERSION_ARGS above) calls this once and
    reads the result, rather than each hardcoding its own literal (or each calling
    resolve_active_version() 8 times with hand-copied family/key strings -- exactly the
    duplication that let xi_params_version/rho_residual_params_version drift out of sync across
    a dozen scripts before this function existed, see resolve_active_version()'s own docstring).
    One place owns the family/key mapping; every caller reads the same answer."""
    return {
        arg_name: resolve_active_version(family, default_version, seed_dir, param_key=key)
        for arg_name, (family, key) in RECALIBRATABLE_VERSION_ARGS.items()
    }


# metric_name -> "higher_is_better" for every value recalibrate() actually writes into
# recalibration_proposals.metric_name. Two are NOT a comparable before/after score at all:
# rho_hat's metric_before/metric_after are the actual old/new rho_residual VALUE (moment-
# matched directly against realized covariance, not a grid search over a scored objective --
# see refit_rho_residual()'s own call site, which -- unlike xi/rho/lambda/kappa_tc -- has no
# `if new != current:` guard, so a proposal can even represent no real change at all). Those
# are handled as a special case in evaluate_and_promote_proposal() below, not via this table.
_METRIC_DIRECTION = {
    "neg_log_likelihood": "lower_is_better",
    "log_score_minutes_mean": "higher_is_better",
    "log_score_minutes_mean_holdout": "higher_is_better",
    "realized_sharpe": "higher_is_better",
}
_NOT_A_SCORE_METRICS = {"rho_hat"}


def evaluate_and_promote_proposal(
    con: duckdb.DuckDBPyConnection, proposal_id: int, seed_dir: Path | str,
    min_relative_improvement: float = 0.0, reviewed_by: str = "auto-regression-gate",
) -> dict:
    """Roadmap P1 item (Track B, docs/plans/2026-08_roadmap_plan.md [A2]): the automated
    counterpart to review_recalibration.py's human --confirm/--reject -- evaluates one pending
    proposal and either promotes it (same set_status() mechanism review_recalibration.py already
    uses: UPDATE recalibration_proposals + re-write the committed seed file, so the DB and the
    git-committed record never drift apart) or leaves it pending with a logged reason.

    Honest scope, disclosed rather than hidden behind a reassuring-looking threshold: this
    compares each proposal's OWN before/after value in the metric its own refit already
    optimized (metric_before/metric_after, already computed by recalibrate() -- see
    _METRIC_DIRECTION above for the direction convention per metric_name). It does NOT re-run a
    full walk-forward backtest to check whether the change regresses some OTHER standard metric
    (log_score_clean_sheet_mean, etc.) -- each scripts/run_backtest.py run is a real ~1-2 hour
    job (see its own module docstring), so re-running one per pending proposal is not something
    an automated per-week gate can afford. That residual risk -- a proposal that improves its
    own target metric but regresses an unrelated one -- is named in
    docs/plans/2026-08_roadmap_plan.md's Open Items, not silently assumed away.

    Every family except rho_residual (see _METRIC_DIRECTION/_NOT_A_SCORE_METRICS above) only
    ever gets a proposal when the refit already found a strict improvement (refit_xi_rho() etc.
    only call propose_recalibration() `if result[...] != current[...]`), so
    min_relative_improvement's real job is filtering out numerically-noisy near-ties (a
    grid-search "improvement" of 1e-6 is not a real signal), not catching genuine regressions --
    there are none to catch in this metric space by construction. rho_residual's proposals are
    promoted whenever the value actually changed (no score to compare), since moment-matching
    against realized covariance IS the validation there, not a searched-and-scored candidate.

    Returns a dict: {"proposal_id", "action": "promoted" | "held", "reason"}.
    """
    row = con.execute(
        "SELECT status, backtest_run_id, param_family, param_key, metric_name, metric_before, metric_after "
        "FROM recalibration_proposals WHERE proposal_id = ?", [proposal_id],
    ).fetchone()
    if row is None:
        raise ValueError(f"no recalibration_proposals row for proposal_id={proposal_id}")
    status, backtest_run_id, param_family, param_key, metric_name, metric_before, metric_after = row
    if status != "pending":
        return {"proposal_id": proposal_id, "action": "held", "reason": f"already {status}, not pending"}

    if metric_name in _NOT_A_SCORE_METRICS:
        # metric_before/metric_after ARE the old/new value directly for this metric (see
        # refit_rho_residual()'s own call site) -- not old_value/new_value, which resolve
        # through a separate old_params_version lookup and needn't even be set.
        if metric_before == metric_after:
            return {"proposal_id": proposal_id, "action": "held", "reason": f"{metric_name}: value unchanged ({metric_before}), nothing to promote"}
        reason = f"{metric_name}: {metric_before} -> {metric_after} (moment-matched refit, no score to gate)"
    else:
        direction = _METRIC_DIRECTION.get(metric_name)
        if direction is None:
            return {"proposal_id": proposal_id, "action": "held", "reason": f"unrecognized metric_name {metric_name!r} -- refusing to guess a direction"}
        improved = metric_after < metric_before if direction == "lower_is_better" else metric_after > metric_before
        if not improved:
            return {
                "proposal_id": proposal_id, "action": "held",
                "reason": f"{metric_name}: {metric_before} -> {metric_after} is not an improvement ({direction})",
            }
        denom = abs(metric_before) if metric_before != 0 else abs(metric_after) if metric_after != 0 else None
        relative_improvement = abs(metric_after - metric_before) / denom if denom else float("inf")
        if relative_improvement < min_relative_improvement:
            return {
                "proposal_id": proposal_id, "action": "held",
                "reason": f"{metric_name}: improvement {relative_improvement:.4%} below the {min_relative_improvement:.0%} noise floor",
            }
        reason = f"{metric_name}: {metric_before} -> {metric_after} ({relative_improvement:.2%} improvement)"

    con.execute(
        "UPDATE recalibration_proposals SET status = 'confirmed', reviewed_by = ?, reviewed_at = ? WHERE proposal_id = ?",
        [reviewed_by, datetime.now(timezone.utc), proposal_id],
    )
    write_recalibration_seed_file(con, backtest_run_id, seed_dir)
    return {"proposal_id": proposal_id, "action": "promoted", "reason": reason}


def auto_promote_pending_proposals(
    con: duckdb.DuckDBPyConnection, backtest_run_id: int, seed_dir: Path | str,
    min_relative_improvement: float = 0.0, reviewed_by: str = "auto-regression-gate",
) -> list[dict]:
    """Runs evaluate_and_promote_proposal() over every pending proposal for one backtest run --
    the whole-run entry point scripts/run_backtest.py calls right after recalibrate()."""
    proposal_ids = [
        r[0] for r in con.execute(
            "SELECT proposal_id FROM recalibration_proposals WHERE backtest_run_id = ? AND status = 'pending' ORDER BY proposal_id",
            [backtest_run_id],
        ).fetchall()
    ]
    return [
        evaluate_and_promote_proposal(con, pid, seed_dir, min_relative_improvement, reviewed_by)
        for pid in proposal_ids
    ]


def load_confirmed_recalibration_seeds(seed_dir: Path | str) -> list[dict]:
    """Reads every seeds_*.json file in seed_dir and returns only 'confirmed' proposals -- the
    set safe to auto-write into a fresh/empty DB (see run_ingestion.py's own use of this). A
    'pending' or 'rejected' entry is real, disclosed history, but never auto-activated here.
    Returns [] (not an error) when seed_dir doesn't exist yet -- a fresh checkout with no
    recalibration run behind it is a normal, expected state, not a failure."""
    seed_dir = Path(seed_dir)
    if not seed_dir.is_dir():
        return []
    confirmed = []
    for path in sorted(seed_dir.glob("seeds_*.json")):
        payload = json.loads(path.read_text())
        for proposal in payload.get("proposals", []):
            if proposal.get("status") == "confirmed":
                confirmed.append(proposal)
    return confirmed


def refit_xi_rho(
    con: duckdb.DuckDBPyConnection,
    fit_seasons: tuple[str, ...] = ("2024-2025", "2025-2026"),
    xi_grid: tuple[float, ...] = (0.0005, 0.001, 0.0018, 0.003, 0.005),
    rho_grid: tuple[float, ...] = (-0.05, -0.10, -0.13, -0.16, -0.20),
    reference_team_uid: str | None = None,
    asof_date: date | None = None,
) -> dict:
    """Expanding-window MLE re-fit of xi/rho against every realized match now available.

    team_strength.fit_dixon_coles() itself only fits attack/defence/home_advantage GIVEN xi
    and rho -- they're pinned inputs to its likelihood, not its own fitted outputs. Refitting
    them "via the same MLE" therefore means profiling the same neg-log-likelihood over a second
    axis: for each (xi, rho) candidate, call fit_dixon_coles() completely unchanged and keep
    whichever pair yields the lowest likelihood (scipy's own optimizer result, unchanged) --
    still the identical likelihood and optimizer M1 already uses, not a new fitting method.
    """
    matches = team_strength.fetch_calibration_matches(con, fit_seasons)
    if matches.empty:
        raise ValueError(f"no matches to refit xi/rho against for seasons {fit_seasons}")
    if asof_date is None:
        asof_date = pd.to_datetime(matches["kickoff_time"]).max().date()
    teams = sorted(set(matches.home_team_uid) | set(matches.away_team_uid))
    if reference_team_uid is None:
        reference_team_uid = teams[0]

    best_xi, best_rho, best_nll = None, None, None
    for xi in xi_grid:
        for rho in rho_grid:
            _, _, _, opt = team_strength.fit_dixon_coles(matches, xi, rho, asof_date, reference_team_uid)
            if best_nll is None or opt.fun < best_nll:
                best_xi, best_rho, best_nll = xi, rho, float(opt.fun)
    return {"xi": best_xi, "rho": best_rho, "neg_log_likelihood": best_nll}


def refit_rho_residual(
    con: duckdb.DuckDBPyConnection, backtest_run_id: int, tiers: tuple[str, ...] = ("warm", "mature"), min_shared_gameweeks: int = 5,
) -> dict:
    """Moment-matches rho_residual against REALIZED goals+assists count covariance, using the
    exact closed-form monte_carlo.z_fixture_variance() inverts -- solved in the other
    direction (rho recovered from an observed covariance, not sigma_z^2 derived from an assumed
    rho): rho = cov / (lambda + cov). Reuses the realized_goals_assists:{player_uid} rows
    score_gameweek() already recorded.

    Scope, stated plainly: "realized outcome covariance from actual results" (the M7 spec's own
    phrase) is inherently a cross-gameweek estimate -- a single gameweek's paired realization is
    two scalars, not a covariance -- so this pools every gameweek a given pair of players were
    *both* in the optimizer's chosen squad (squad membership itself changes week to week, unlike
    M6's within-fixture Z_fixture mechanism), restricted to pairs with at least
    min_shared_gameweeks of overlap to avoid a two-observation "covariance." Cold tier is
    excluded from the fitting objective (data-starved by construction, not a fair calibration
    target), same reasoning as the M1b/M2 objective below -- still fully visible in backtest_metrics
    for reporting.
    """
    placeholders = ",".join("?" * len(tiers))
    rows = con.execute(
        f"SELECT season, gameweek, metric_name, metric_value FROM backtest_metrics "
        f"WHERE backtest_run_id = ? AND tier IN ({placeholders}) AND metric_name LIKE 'realized_goals_assists:%'",
        [backtest_run_id, *tiers],
    ).fetchdf()
    if rows.empty:
        raise ValueError(f"no realized_goals_assists metrics recorded for backtest_run_id={backtest_run_id}")

    rows["player_uid"] = rows["metric_name"].str.split(":", n=1).str[1]
    wide = rows.pivot_table(index=["season", "gameweek"], columns="player_uid", values="metric_value")
    lambda_representative = float(rows["metric_value"].mean())
    cov = wide.cov(ddof=0, min_periods=min_shared_gameweeks)

    players = list(wide.columns)
    covs = [cov.loc[a, b] for i, a in enumerate(players) for b in players[i + 1:] if pd.notna(cov.loc[a, b])]
    if not covs:
        raise ValueError(
            f"no player pairs shared >= {min_shared_gameweeks} gameweeks in the same squad -- "
            "cannot moment-match rho_residual from this backtest_run_id"
        )
    empirical_cov = float(np.mean(covs))
    denom = lambda_representative + empirical_cov
    rho_hat = max(0.0, min(empirical_cov / denom, 0.99)) if denom > 0 else 0.0
    return {
        "rho_residual": rho_hat, "lambda_representative": lambda_representative,
        "empirical_cov": empirical_cov, "n_pairs": len(covs),
    }


def _minutes_log_score_for_step(
    con: duckdb.DuckDBPyConnection, season: str, gameweek: int, ep_model_version: int,
    decay_params_version: int, adjustment_params_version: int, shrinkage_params_version: int, fact_multiplier_params_version: int,
) -> float | None:
    """Re-runs only minutes_model.run() (cheap -- SQL aggregation, no SCIP/scipy) inside a
    fresh asof_scope for this one step, and returns its mean minutes log score against that
    gameweek's now-real outcome. Reuses the *existing* ep_model_version from the original
    walk-forward pass (which fixture each player had doesn't depend on M1b/M2's parameters),
    so a coordinate-descent evaluation never has to re-run team_strength/expected_points at
    all -- this is what keeps this refit tractable across many candidates x many gameweeks."""
    with asof_scope(con, season, gameweek):
        mm_model_version = minutes_model.run(
            con, gameweek_deadline(con, season, gameweek).date(), season,
            decay_params_version, adjustment_params_version, shrinkage_params_version, fact_multiplier_params_version,
        )
    ep_fixture_of = dict(con.execute(
        "SELECT player_uid, fixture_match_id FROM ep_outputs WHERE model_version = ?", [ep_model_version]
    ).fetchall())
    mm_rows = con.execute(
        "SELECT player_uid, p_0min, p_1_59min, p_60plus_min FROM minutes_model_outputs WHERE model_version = ?", [mm_model_version]
    ).fetchall()
    scores = []
    for player_uid, p0, p1, p2 in mm_rows:
        match_id = ep_fixture_of.get(player_uid)
        if match_id is None:
            continue
        outcome = _realized_player_match_outcome(con, player_uid, match_id)
        state = _minutes_state(outcome["minutes_played"])
        scores.append(log_score_categorical({"0": p0, "1_59": p1, "60plus": p2}, state))
    return sum(scores) / len(scores) if scores else None


def _write_family_version_with_override(
    con: duckdb.DuckDBPyConnection, param_family: str, base_version: int, new_version: int, effective_date: str,
    override_key: str, override_dimensions: dict | None, override_value: float,
) -> None:
    """Copies every row of base_version into a new version, substituting one (key,
    dimensions) row's value. Needed because resolve_param() requires every key in a family to
    exist at whatever version is pinned -- testing one candidate value for one key still needs
    a complete version, not a partial one."""
    override_dims_json = params_mod._canonical_dimensions(override_dimensions)
    rows = con.execute(
        "SELECT param_key, dimensions, value_numeric, value_text FROM param_versions WHERE param_family = ? AND param_version = ?",
        [param_family, base_version],
    ).fetchall()
    for key, dims_json, val_num, val_text in rows:
        if key == override_key and dims_json == override_dims_json:
            val_num = override_value
        dims = json.loads(dims_json) if dims_json != "{}" else None
        params_mod.write_param(con, param_family, new_version, effective_date, key, value_numeric=val_num, value_text=val_text, dimensions=dims)


def refit_minutes_and_evidence_params(
    con: duckdb.DuckDBPyConnection,
    eval_steps: list[tuple[str, int]],
    ep_model_version_by_step: dict[tuple[str, int], int],
    base_versions: dict,
    param_grids: list[dict],
    n_rounds: int = 1,
    holdout_steps: list[tuple[str, int]] | None = None,
) -> dict:
    """Block coordinate descent over M1b's tier weights/fact_type_multiplier and M2's
    threshold/adjustment magnitudes, all against one shared objective: mean minutes log score
    on realized outcomes across eval_steps. One shared objective because both blocks are
    causally chained into the same output (tier weight -> effective_weight/blend confidence ->
    adjustment magnitude application -> P(60+) shift) -- inventing two separate objectives for
    parameters that feed the same pipeline stage would be arbitrary.

    base_versions: {"decay_params_version", "adjustment_params_version",
    "shrinkage_params_version", "fact_multiplier_params_version"} -- the versions to start from.
    param_grids: one dict per block: {"param_family", "param_key", "dimensions" (or None),
    "candidates": [...], "version_field": which base_versions key this family's version lives
    under}. eval_steps and ep_model_version_by_step are the caller's explicit choice of which
    already-run walk-forward steps to evaluate against -- deliberately not defaulted to "all 76"
    here, since the real cost (n_rounds x n_blocks x n_candidates x len(eval_steps) minutes_model
    runs) is the caller's to size, not this function's to assume.

    Real overfitting risk, disclosed and addressed rather than left implicit (analogous to
    refit_lambda()'s own out-of-sample framing, but a materially bigger version of the same
    risk here): this is a multi-round, multi-family block coordinate descent, not a single
    7-point grid search -- selecting the best candidate per block against eval_steps and then
    reporting that same eval_steps score as evidence is optimistic by construction (the classic
    "graded on the set you were selected on" bias), and the risk compounds with every extra
    round/family/candidate this function is given. `holdout_steps`, when provided, must be
    disjoint from eval_steps (the caller's responsibility -- typically an entire later season
    never touched by the descent, matching this project's forward-chronological walk-forward
    discipline) -- the descent itself still only ever searches against eval_steps (a coordinate
    descent needs a stable objective to hill-climb; splitting the search itself would just add
    noise, not rigor), but the final chosen versions are ALSO scored against holdout_steps and
    both scores are returned, so a human reviewing the proposal sees whether the improvement
    actually generalizes to gameweeks the descent never saw, not just the in-sample number it
    was picked to maximize. ep_model_version_by_step must cover holdout_steps too when supplied.
    """
    current = dict(base_versions)

    def _mean_score(versions: dict, steps: list[tuple[str, int]] = eval_steps) -> float:
        scores = []
        for season, gw in steps:
            s = _minutes_log_score_for_step(
                con, season, gw, ep_model_version_by_step[(season, gw)],
                versions["decay_params_version"], versions["adjustment_params_version"],
                versions["shrinkage_params_version"], versions["fact_multiplier_params_version"],
            )
            if s is not None:
                scores.append(s)
        return sum(scores) / len(scores) if scores else float("-inf")

    best_score = _mean_score(current)
    history = [{"round": 0, "log_score": best_score, "versions": dict(current)}]

    for round_num in range(1, n_rounds + 1):
        for block in param_grids:
            version_field = block["version_field"]
            base_version = current[version_field]
            best_candidate_score, best_candidate_version = best_score, None
            for candidate in block["candidates"]:
                trial_version = _next_param_version(con, block["param_family"])
                _write_family_version_with_override(
                    con, block["param_family"], base_version, trial_version, "2026-08-11",
                    block["param_key"], block.get("dimensions"), candidate,
                )
                trial = dict(current)
                trial[version_field] = trial_version
                trial_score = _mean_score(trial)
                if trial_score > best_candidate_score:
                    best_candidate_score, best_candidate_version = trial_score, trial_version
            if best_candidate_version is not None:
                current[version_field] = best_candidate_version
                best_score = best_candidate_score
        history.append({"round": round_num, "log_score": best_score, "versions": dict(current)})

    result = {"versions": current, "log_score": best_score, "history": history}
    if holdout_steps:
        result["holdout_log_score_before"] = _mean_score(base_versions, steps=holdout_steps)
        result["holdout_log_score_after"] = _mean_score(current, steps=holdout_steps)
        result["n_holdout_steps"] = len(holdout_steps)
    return result


def _realized_xi_points(
    con: duckdb.DuckDBPyConnection, season: str, gameweek: int, xi_uids: frozenset, captain_uid: str | None,
    captain_multiplier: int = 2,
) -> float:
    """Real FPL scoring: only the starting XI's points count, and the captain's points double
    -- summing the full 15-player squad (bench included) would overstate what a squad actually
    scored. Uses fact_player_season_stats.event_points, the real ground truth, not a
    reconstruction from raw stats. captain_multiplier defaults to 2 (the real rule for every
    normal gameweek); run_season_simulation() passes 3 for a gameweek Triple Captain was
    accepted on -- the one real FPL rule change captaincy makes to this formula, not a second
    scoring function."""
    total = 0.0
    for player_uid in xi_uids:
        row = con.execute(
            "SELECT event_points FROM fact_player_season_stats WHERE player_uid = ? AND season = ? AND gw = ?",
            [player_uid, season, gameweek],
        ).fetchone()
        pts = row[0] if row and row[0] is not None else 0.0
        total += pts * captain_multiplier if player_uid == captain_uid else pts
    return total


# ============================================================
# season-long scoring -- one evolving manager's own weekly trajectory, not independent
# from-scratch squads across many backtest steps (see run_season_simulation())
# ============================================================

def season_cumulative_metrics(weekly_points: list[float]) -> dict:
    """Scores a season-long trajectory of realized XI points -- one manager's own week-by-week
    sequence from a real evolving squad, not the cross-sectional population refit_lambda()'s
    realized_sharpe was built to compare (independent from-scratch M5 squads, one per backtest
    gameweek). realized_sharpe here reuses that exact same formula (mean/std) unchanged; what's
    different is only the population it's computed over -- a genuine season-consistency read on
    one continuous squad's own trajectory, the real quant-finance sense of a single track
    record's Sharpe ratio, not a cross-sectional one.

    Sharpe alone doesn't capture drawdown though. Real FPL weekly points are (virtually) always
    non-negative, so cumulative points only ever accumulate -- a literal price-style drawdown
    computed directly on cumulative points would always be ~0 and tell you nothing. Instead,
    this applies the standard "underwater curve" technique to cumulative SURPLUS over the
    season's own mean weekly score (weekly_points - mean_points, cumulatively summed): the
    deepest fall from that surplus curve's own prior peak, in points. A genuinely distinct risk
    signal from Sharpe -- a manager who banks most of their points in one purple patch then
    grinds through a long, deep cold streak scores worse here than one with the same total and
    even the same variance spread evenly across the season, which Sharpe's single whole-season
    standard deviation can't always tell apart from the smoother trajectory.
    """
    if not weekly_points:
        return {"total_points": 0.0, "mean_points": 0.0, "n_gameweeks": 0, "realized_sharpe": float("-inf"), "max_drawdown": 0.0}

    arr = np.array(weekly_points, dtype=float)
    total = float(arr.sum())
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    sharpe = mean / std if std > 0 else float("-inf")

    cumulative_surplus = np.cumsum(arr - mean)
    running_peak = np.maximum.accumulate(cumulative_surplus)
    max_drawdown = float((running_peak - cumulative_surplus).max())

    return {
        "total_points": total, "mean_points": mean, "n_gameweeks": len(weekly_points),
        "realized_sharpe": sharpe, "max_drawdown": max_drawdown,
    }


def refit_lambda(
    con: duckdb.DuckDBPyConnection,
    eval_steps: list[tuple[str, int]],
    ep_model_version_by_step: dict[tuple[str, int], int],
    uncertainty_model_version_by_step: dict[tuple[str, int], int],
    guardrail_cap: float,
    lambda_grid: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50),
) -> dict:
    """Out-of-sample grid search on realized_sharpe = mean(realized XI points)/std(same)
    across eval_steps, maximized over lambda_grid -- per the M7 spec's own self-critique
    ("out-of-sample grid search on risk-adjusted return, since lambda is a preference
    parameter rather than a data-fit one"). Re-solves squad_optimizer.solve() per gameweek per
    candidate (the expensive part, ~19s per solve per README's own live-run numbers) but never
    re-runs team_strength/minutes/expected_points/uncertainty -- lambda only changes the
    optimizer's risk term, not the EP/Sigma inputs already computed by the original
    walk-forward pass, so this reuses fetch_candidate_pool()/fetch_sigma_pairs() against the
    existing ep_model_version/uncertainty_model_version for each step.

    Deliberately excludes xi_club_concentration_cap from this search (see
    report_concentration_sensitivity() instead) -- guardrail_cap is held fixed at whatever the
    live pin currently is, not itself a grid dimension. Re-tuning a redundant safety backstop
    against the same signal the primary risk dial is tuned against would erode the exact
    protection it exists to provide if the primary mechanism ever silently degenerated.
    """
    grid_results = {}
    for lam in lambda_grid:
        gameweek_points = []
        for season, gw in eval_steps:
            ep_mv = ep_model_version_by_step.get((season, gw))
            un_mv = uncertainty_model_version_by_step.get((season, gw))
            if ep_mv is None or un_mv is None:
                continue
            candidates = squad_optimizer.fetch_candidate_pool(con, ep_mv, un_mv, season)
            if len(candidates) < 15:
                continue
            player_uids = {c["player_uid"] for c in candidates}
            sigma_pairs = squad_optimizer.fetch_sigma_pairs(con, un_mv, player_uids)
            result = squad_optimizer.solve(candidates, sigma_pairs, lam, guardrail_cap)
            if not result["xi"]:
                continue
            gameweek_points.append(_realized_xi_points(con, season, gw, result["xi"], result["captain"]))

        if len(gameweek_points) >= 2:
            arr = np.array(gameweek_points)
            std = float(arr.std(ddof=0))
            sharpe = float(arr.mean() / std) if std > 0 else float("-inf")
            grid_results[lam] = {"realized_sharpe": sharpe, "mean_points": float(arr.mean()), "n_gameweeks": len(gameweek_points)}
        else:
            grid_results[lam] = {"realized_sharpe": float("-inf"), "mean_points": None, "n_gameweeks": len(gameweek_points)}

    best_lambda = max(grid_results, key=lambda lam_val: grid_results[lam_val]["realized_sharpe"])
    return {"best_lambda": best_lambda, "grid": grid_results}


def _xi_uids_by_step(con: duckdb.DuckDBPyConnection, backtest_run_id: int) -> dict[tuple[str, int], set[str]]:
    rows = con.execute(
        "SELECT season, gameweek, so_run_id FROM backtest_gameweek_steps WHERE backtest_run_id = ? AND so_run_id IS NOT NULL",
        [backtest_run_id],
    ).fetchall()
    out = {}
    for season, gw, so_run_id in rows:
        xi = {
            r[0] for r in con.execute(
                "SELECT player_uid FROM squad_optimizer_selections WHERE run_id = ? AND in_xi", [so_run_id]
            ).fetchall()
        }
        out[(season, gw)] = xi
    return out


def refit_kappa_tc(
    con: duckdb.DuckDBPyConnection,
    eval_steps: list[tuple[str, int]],
    mc_model_version_by_step: dict[tuple[str, int], int],
    xi_uids_by_step: dict[tuple[str, int], set[str]],
    kappa_tc_grid: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50),
) -> dict:
    """M8's own spec flags kappa_tc for M7 recalibration explicitly ("same status as lambda...
    flagged for M7 recalibration"). Out-of-sample grid search on realized_sharpe =
    mean(realized captain points)/std(same) across eval_steps -- the same technique
    refit_lambda() already established for this class of risk-preference parameter.

    Unlike lambda, kappa_tc never changes which squad/XI gets picked -- only which XI player
    would be captained -- so this needs zero re-solving. Every backtest gameweek step already
    has a real Monte Carlo simulation of its own XI (monte_carlo_player_summary, linked via
    backtest_gameweek_steps.mc_model_version), and captain choice is a pure argmax read
    against that existing data, the exact same TC_score formula
    transfer_planner.evaluate_triple_captain() uses live. `_realized_xi_points()` with a
    single-player set doubles that one player's realized points -- exactly what captaining
    them would have scored, isolated from the rest of the team's outcome.

    wildcard_gain_threshold_params (M8's other invented risk-adjacent default) is deliberately
    NOT covered here -- backtesting it would mean re-running M8's manager-state bootstrap and
    evolution across the whole backtest history (a materially larger, separately-scoped
    undertaking: M7's own walk-forward squad is M5's from-scratch pick every step, not an
    evolving manager holding), not a small extension of this function. Named as a real,
    disclosed gap, not silently skipped.
    """
    grid_results = {}
    for kappa in kappa_tc_grid:
        gameweek_points = []
        for season, gw in eval_steps:
            mc_mv = mc_model_version_by_step.get((season, gw))
            xi_uids = xi_uids_by_step.get((season, gw))
            if not mc_mv or not xi_uids:
                continue
            rows = con.execute(
                "SELECT player_uid, mean_total, var_total FROM monte_carlo_player_summary WHERE model_version = ?",
                [mc_mv],
            ).fetchall()
            scored = [(uid, mean_total - kappa * (var_total ** 0.5)) for uid, mean_total, var_total in rows if uid in xi_uids]
            if not scored:
                continue
            captain_uid = max(scored, key=lambda c: c[1])[0]
            gameweek_points.append(_realized_xi_points(con, season, gw, frozenset({captain_uid}), captain_uid))

        if len(gameweek_points) >= 2:
            arr = np.array(gameweek_points)
            std = float(arr.std(ddof=0))
            sharpe = float(arr.mean() / std) if std > 0 else float("-inf")
            grid_results[kappa] = {"realized_sharpe": sharpe, "mean_points": float(arr.mean()), "n_gameweeks": len(gameweek_points)}
        else:
            grid_results[kappa] = {"realized_sharpe": float("-inf"), "mean_points": None, "n_gameweeks": len(gameweek_points)}

    best_kappa = max(grid_results, key=lambda k: grid_results[k]["realized_sharpe"])
    return {"best_kappa_tc": best_kappa, "grid": grid_results}


def report_concentration_sensitivity(
    con: duckdb.DuckDBPyConnection,
    eval_steps: list[tuple[str, int]],
    ep_model_version_by_step: dict[tuple[str, int], int],
    uncertainty_model_version_by_step: dict[tuple[str, int], int],
    lambda_value: float,
    cap_grid: tuple[float, ...] = (2, 3, 4, 5),
) -> dict:
    """Read-only reporting, deliberately separate from recalibrate(): how would realized_sharpe
    have differed at each club-concentration cap value historically? xi_club_concentration_cap
    is a deliberate redundant backstop against lambda's mechanism silently failing (stub Sigma,
    solver falling back to linear-only -- the project's own documented history), not a
    statistical fit -- see README's Design notes for the full reasoning. This function never
    writes a param_versions or recalibration_proposals row; it exists purely to inform the M7
    spec's own qualitative "could a human beat this by eye" review step.
    """
    grid_results = {}
    for cap in cap_grid:
        gameweek_points = []
        for season, gw in eval_steps:
            ep_mv = ep_model_version_by_step.get((season, gw))
            un_mv = uncertainty_model_version_by_step.get((season, gw))
            if ep_mv is None or un_mv is None:
                continue
            candidates = squad_optimizer.fetch_candidate_pool(con, ep_mv, un_mv, season)
            if len(candidates) < 15:
                continue
            player_uids = {c["player_uid"] for c in candidates}
            sigma_pairs = squad_optimizer.fetch_sigma_pairs(con, un_mv, player_uids)
            result = squad_optimizer.solve(candidates, sigma_pairs, lambda_value, cap)
            if not result["xi"]:
                continue
            gameweek_points.append(_realized_xi_points(con, season, gw, result["xi"], result["captain"]))

        if len(gameweek_points) >= 2:
            arr = np.array(gameweek_points)
            std = float(arr.std(ddof=0))
            sharpe = float(arr.mean() / std) if std > 0 else float("-inf")
            grid_results[cap] = {"realized_sharpe": sharpe, "mean_points": float(arr.mean()), "n_gameweeks": len(gameweek_points)}
        else:
            grid_results[cap] = {"realized_sharpe": float("-inf"), "mean_points": None, "n_gameweeks": len(gameweek_points)}
    return {"grid": grid_results}


# ============================================================
# top-level recalibration orchestrator
# ============================================================

def _eval_steps_for(con: duckdb.DuckDBPyConnection, backtest_run_id: int, tiers: tuple[str, ...] = ("warm", "mature")) -> list[tuple[str, int]]:
    """Steps to evaluate refit objectives against, defaulting to every step that actually ran
    in warm/mature tiers -- cold tier is excluded from every fitting objective in this module
    (data-starved by construction, not a fair calibration target), though it stays fully
    visible in backtest_metrics for reporting."""
    placeholders = ",".join("?" * len(tiers))
    rows = con.execute(
        f"SELECT season, gameweek FROM backtest_gameweek_steps WHERE backtest_run_id = ? AND tier IN ({placeholders}) ORDER BY season, gameweek",
        [backtest_run_id, *tiers],
    ).fetchall()
    return [(s, gw) for s, gw in rows]


def _model_version_map(con: duckdb.DuckDBPyConnection, backtest_run_id: int, column: str) -> dict:
    rows = con.execute(
        f"SELECT season, gameweek, {column} FROM backtest_gameweek_steps WHERE backtest_run_id = ?", [backtest_run_id]
    ).fetchall()
    return {(s, gw): v for s, gw, v in rows if v is not None}


def recalibrate(
    con: duckdb.DuckDBPyConnection,
    backtest_run_id: int,
    *,
    current_xi_version: int, current_rho_version: int,
    current_rho_residual_version: int,
    current_minutes_versions: dict,
    current_lambda_version: int,
    guardrail_cap: float,
    minutes_param_grids: list[dict],
    lambda_grid: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50),
    xi_grid: tuple[float, ...] = (0.0005, 0.001, 0.0018, 0.003, 0.005),
    rho_grid: tuple[float, ...] = (-0.05, -0.10, -0.13, -0.16, -0.20),
    effective_date: str = "2026-08-11",
    refit_xi_rho_flag: bool = True,
    refit_rho_residual_flag: bool = True,
    refit_minutes_flag: bool = True,
    refit_lambda_flag: bool = True,
    current_kappa_tc_version: int | None = None,
    kappa_tc_grid: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50),
    refit_kappa_tc_flag: bool = False,
    minutes_select_seasons: tuple[str, ...] = ("2024-2025",),
    minutes_holdout_flag: bool = True,
    seed_dir: Path | str | None = None,
) -> list[int]:
    """Runs whichever refit techniques are enabled against this backtest_run_id's results and
    writes one propose_recalibration() row per changed parameter -- never activates anything
    (see propose_recalibration()'s own docstring). Each technique is individually toggleable:
    their real costs differ by roughly two orders of magnitude (xi/rho and rho_residual are
    cheap closed-form/profile-likelihood work; the minutes coordinate descent re-runs
    minutes_model.run() per candidate per gameweek; the lambda grid search re-solves the SCIP
    MIQP -- ~19s each per README's own live-run numbers -- per candidate per gameweek), so a
    caller may reasonably want to run them separately rather than pay for all four in one call.

    xi_club_concentration_cap is never recalibrated here by design -- see
    report_concentration_sensitivity() and the README's Design notes for why.

    refit_kappa_tc_flag defaults False (unlike the M7-native techniques above, which default
    True): it is the one M8 extension in this function, opt-in via current_kappa_tc_version so
    that existing M7-only callers are unaffected. It is by far the cheapest technique here --
    a pure argmax read against monte_carlo_player_summary rows this backtest run already wrote
    per step, no re-solving of anything -- see refit_kappa_tc()'s own docstring.
    wildcard_gain_threshold_params is NOT covered by any technique here; see refit_kappa_tc()'s
    docstring for why that is a disclosed scope decision, not an oversight.

    minutes_holdout_flag (default True): refit_minutes_and_evidence_params()'s coordinate
    descent is a genuinely larger overfitting risk than the single-dimension grid searches
    above (see that function's own docstring) -- when this is on, its eval_steps are split by
    season into a select set (minutes_select_seasons, default 2024-2025's warm gameweeks --
    chronologically earlier) and a holdout set (every other warm/mature step, i.e. all of
    2025-2026 -- chronologically later, never touched by the descent), and the proposal's
    logged before/after metric is the holdout score, not the in-sample one the descent
    actually climbed. Off reverts to the prior in-sample-only behavior.

    seed_dir (optional, default None -- exact no-op, prior behavior unchanged): when given,
    every proposal_recalibration() call above also lands in a committed JSON seed file (see
    write_recalibration_seed_file()) after this function finishes, independent of whatever
    happens to the DuckDB file that holds the same rows. Written unconditionally when set (even
    with zero new proposals) -- "we ran this and nothing changed" is real, worth-recording
    information too, not just a non-event.
    """
    proposal_ids = []
    eval_steps = _eval_steps_for(con, backtest_run_id)
    ep_by_step = _model_version_map(con, backtest_run_id, "ep_model_version")
    un_by_step = _model_version_map(con, backtest_run_id, "un_model_version")

    if refit_xi_rho_flag:
        current_xi, _ = params_mod.resolve_param(con, "model_decay_params", "xi", current_xi_version)
        current_rho, _ = params_mod.resolve_param(con, "model_decay_params", "rho", current_rho_version)
        matches = team_strength.fetch_calibration_matches(con, ("2024-2025", "2025-2026"))
        asof_date = pd.to_datetime(matches["kickoff_time"]).max().date()
        teams = sorted(set(matches.home_team_uid) | set(matches.away_team_uid))
        _, _, _, opt_before = team_strength.fit_dixon_coles(matches, current_xi, current_rho, asof_date, teams[0])
        result = refit_xi_rho(con, xi_grid=tuple(set(xi_grid) | {current_xi}), rho_grid=tuple(set(rho_grid) | {current_rho}))
        if result["xi"] != current_xi:
            proposal_ids.append(propose_recalibration(
                con, backtest_run_id, "model_decay_params", "xi", result["xi"],
                "neg_log_likelihood", float(opt_before.fun), result["neg_log_likelihood"],
                old_params_version=current_xi_version, effective_date=effective_date,
            ))
        if result["rho"] != current_rho:
            proposal_ids.append(propose_recalibration(
                con, backtest_run_id, "model_decay_params", "rho", result["rho"],
                "neg_log_likelihood", float(opt_before.fun), result["neg_log_likelihood"],
                old_params_version=current_rho_version, effective_date=effective_date,
            ))

    if refit_rho_residual_flag:
        current_rho_residual, _ = params_mod.resolve_param(con, "correlation_params", "rho_residual", current_rho_residual_version)
        result = refit_rho_residual(con, backtest_run_id)
        proposal_ids.append(propose_recalibration(
            con, backtest_run_id, "correlation_params", "rho_residual", result["rho_residual"],
            "rho_hat", current_rho_residual, result["rho_residual"],
            old_params_version=current_rho_residual_version, effective_date=effective_date,
        ))

    if refit_minutes_flag:
        minutes_select_steps = [s for s in eval_steps if s[0] in minutes_select_seasons]
        minutes_holdout_steps = (
            [s for s in eval_steps if s[0] not in minutes_select_seasons] if minutes_holdout_flag else None
        )
        result = refit_minutes_and_evidence_params(
            con, minutes_select_steps, ep_by_step, current_minutes_versions, minutes_param_grids,
            holdout_steps=minutes_holdout_steps,
        )
        if minutes_holdout_steps:
            metric_name = "log_score_minutes_mean_holdout"
            score_before, score_after = result["holdout_log_score_before"], result["holdout_log_score_after"]
        else:
            metric_name = "log_score_minutes_mean"
            score_before, score_after = result["history"][0]["log_score"], result["log_score"]
        for block in minutes_param_grids:
            new_version = result["versions"][block["version_field"]]
            old_version = current_minutes_versions[block["version_field"]]
            if new_version == old_version:
                continue
            new_value, _ = params_mod.resolve_param(con, block["param_family"], block["param_key"], new_version, dimensions=block.get("dimensions"))
            proposal_ids.append(propose_recalibration(
                con, backtest_run_id, block["param_family"], block["param_key"], new_value,
                metric_name, score_before, score_after,
                dimensions=block.get("dimensions"), old_params_version=old_version, effective_date=effective_date,
            ))

    if refit_lambda_flag:
        current_lambda, _ = params_mod.resolve_param(con, "risk_aversion_params", "lambda_value", current_lambda_version)
        grid = tuple(set(lambda_grid) | {current_lambda})
        result = refit_lambda(con, eval_steps, ep_by_step, un_by_step, guardrail_cap, lambda_grid=grid)
        if result["best_lambda"] != current_lambda:
            proposal_ids.append(propose_recalibration(
                con, backtest_run_id, "risk_aversion_params", "lambda_value", result["best_lambda"],
                "realized_sharpe", result["grid"][current_lambda]["realized_sharpe"], result["grid"][result["best_lambda"]]["realized_sharpe"],
                old_params_version=current_lambda_version, effective_date=effective_date,
            ))

    if refit_kappa_tc_flag:
        if current_kappa_tc_version is None:
            raise ValueError("refit_kappa_tc_flag=True requires current_kappa_tc_version")
        current_kappa_tc, _ = params_mod.resolve_param(con, "tc_risk_aversion_params", "kappa_tc", current_kappa_tc_version)
        mc_by_step = _model_version_map(con, backtest_run_id, "mc_model_version")
        xi_uids_by_step = _xi_uids_by_step(con, backtest_run_id)
        grid = tuple(set(kappa_tc_grid) | {current_kappa_tc})
        result = refit_kappa_tc(con, eval_steps, mc_by_step, xi_uids_by_step, kappa_tc_grid=grid)
        if result["best_kappa_tc"] != current_kappa_tc:
            proposal_ids.append(propose_recalibration(
                con, backtest_run_id, "tc_risk_aversion_params", "kappa_tc", result["best_kappa_tc"],
                "realized_sharpe", result["grid"][current_kappa_tc]["realized_sharpe"], result["grid"][result["best_kappa_tc"]]["realized_sharpe"],
                old_params_version=current_kappa_tc_version, effective_date=effective_date,
            ))

    if seed_dir is not None:
        write_recalibration_seed_file(con, backtest_run_id, seed_dir)

    return proposal_ids


# ============================================================
# M9 adapter -- backtest performance summary
# ============================================================

def explain_backtest_summary(con: duckdb.DuckDBPyConnection, backtest_run_id: int) -> dict:
    """M9's backtest-summary section: "M7's tiered (cold/warm/mature) metrics, both log score
    and Brier score, so a human sees the system's actual track record, not just this week's
    output in isolation." No canonical "latest" backtest_run_id exists anywhere in this
    project (matching params.py's explicit-version-only discipline) -- the caller always
    states which backtest run to report from. Note this module's own documented scope: there
    is no Brier score for goals/assists, only log score (score_gameweek()'s own choice) -- the
    per-metric rows below simply won't include a brier_goals_mean/brier_assists_mean key,
    not a gap in this adapter.
    """
    run_row = con.execute(
        "SELECT started_at, warm_up_gameweeks, notes FROM backtest_runs WHERE backtest_run_id = ?", [backtest_run_id]
    ).fetchone()
    if not run_row:
        raise ValueError(f"no backtest_runs row for backtest_run_id={backtest_run_id}")
    started_at, warm_up_gameweeks, notes = run_row

    step_counts = con.execute(
        "SELECT tier, count(*), sum(CASE WHEN divergence_check_passed THEN 1 ELSE 0 END), "
        "sum(CASE WHEN divergence_check_passed IS FALSE THEN 1 ELSE 0 END) "
        "FROM backtest_gameweek_steps WHERE backtest_run_id = ? GROUP BY tier", [backtest_run_id],
    ).fetchall()
    steps_by_tier = {tier: {"n_steps": n, "divergence_passed": passed, "divergence_failed": failed} for tier, n, passed, failed in step_counts}

    metric_rows = con.execute(
        "SELECT tier, metric_name, count(*), avg(metric_value) FROM backtest_metrics "
        "WHERE backtest_run_id = ? AND metric_name NOT LIKE 'realized%' GROUP BY tier, metric_name", [backtest_run_id],
    ).fetchall()
    metrics_by_tier: dict = {}
    for tier, name, n, avg_value in metric_rows:
        metrics_by_tier.setdefault(tier, {})[name] = {"n": n, "mean": avg_value}

    n_pending_proposals = con.execute(
        "SELECT count(*) FROM recalibration_proposals WHERE backtest_run_id = ? AND status = 'pending'", [backtest_run_id]
    ).fetchone()[0]

    return {
        "backtest_run_id": backtest_run_id, "started_at": started_at, "warm_up_gameweeks": warm_up_gameweeks,
        "notes": notes, "steps_by_tier": steps_by_tier, "metrics_by_tier": metrics_by_tier,
        "n_pending_recalibration_proposals": n_pending_proposals,
    }
