"""M8 addition: rolling multi-gameweek fixture-difficulty ("fixture swing") signal.

Reuses M1's Dixon-Coles team-strength snapshots and reproduces M3's exact per-fixture
lambda formula (expected_points._fixture_lambdas) so "difficulty" here means the same
thing the EP engine already means by it -- not a separate, inconsistent ad hoc metric.
Deliberately NOT imported from expected_points.py: that would create an M8 -> M3 module
dependency in the wrong direction for this codebase's layering. Kept in sync by
tests/test_fixture_swing.py::test_lambda_formula_matches_expected_points, which asserts
agreement against the real M3 function on the same inputs.

"Swing" specifically means short-window average difficulty minus long-window average
difficulty for the SAME starting gameweek -- i.e. is the near-term run easier or harder
than the fuller window that contains it -- not "difficulty vs. season average," which
would just be a difficulty score, not a directional swing signal.
"""

import math
from dataclasses import dataclass, field

import duckdb


def team_uid_by_player(con: duckdb.DuckDBPyConnection, target_season: str) -> dict[str, str]:
    """{player_uid: team_uid} for target_season, via the same player_alias -> raw teams
    table -> team_alias -> dim_team join squad_optimizer.explain_run() already uses for
    club-name display -- reused here rather than re-derived, so this module can never
    disagree with M5's own idea of which team a player belongs to."""
    from . import reconcile as reconcile_mod

    found = reconcile_mod._season_root_table(con, target_season, "teams.csv")
    if not found:
        return {}
    rows = con.execute(
        """
        SELECT DISTINCT pa.player_uid, ta.team_uid
        FROM player_alias pa
        JOIN "{}" t ON t.code = pa.team_code
        JOIN team_alias ta ON ta.alias_name = t.name AND ta.season = pa.season
        WHERE pa.season = ?
        """.format(found[1]),
        [target_season],
    ).fetchall()
    return {player_uid: team_uid for player_uid, team_uid in rows}


def _team_fixture_lambdas(con: duckdb.DuckDBPyConnection, team_uid: str, match_id: str, ts_model_version: int):
    """Deliberately duplicated from expected_points._fixture_lambdas (see module
    docstring). Returns None, not a raised error, if the schedule/snapshot data needed
    isn't there (unscheduled fixture, team not in this ts_model_version's fit) --
    swing computation degrades to "insufficient data" for that one fixture, not a crash
    for the whole rolling window."""
    match = con.execute(
        "SELECT home_team_uid, away_team_uid FROM fact_match WHERE match_id = ?", [match_id]
    ).fetchone()
    if match is None:
        return None
    home_uid, away_uid = match
    is_home = team_uid == home_uid
    opp_uid = away_uid if is_home else home_uid

    home_adv_row = con.execute(
        "SELECT home_advantage FROM team_strength_model_versions WHERE model_version = ?", [ts_model_version]
    ).fetchone()
    if home_adv_row is None:
        return None
    home_adv = home_adv_row[0]

    own = con.execute(
        "SELECT final_attack, final_defence FROM team_strength_snapshots WHERE model_version = ? AND team_uid = ?",
        [ts_model_version, team_uid],
    ).fetchone()
    opp = con.execute(
        "SELECT final_attack, final_defence FROM team_strength_snapshots WHERE model_version = ? AND team_uid = ?",
        [ts_model_version, opp_uid],
    ).fetchone()
    if own is None or opp is None:
        return None
    own_attack, own_defence = own
    opp_attack, opp_defence = opp

    adv_own = home_adv if is_home else 0.0
    adv_opp = home_adv if not is_home else 0.0
    lambda_for = math.exp(own_attack - opp_defence + adv_own)
    lambda_against = math.exp(opp_attack - own_defence + adv_opp)
    return lambda_for, lambda_against, is_home, opp_uid


def team_fixtures_in_window(
    con: duckdb.DuckDBPyConnection, team_uid: str, season: str, start_gw: int, end_gw: int,
) -> list[tuple[str, int]]:
    """[(match_id, gameweek), ...] for team_uid in [start_gw, end_gw] inclusive, in kickoff
    order. A blank gameweek (0 fixtures) and a double gameweek (2+ fixtures) both surface
    naturally here -- neither is silently collapsed to one row per gameweek, since both
    matter for real swing/chip-timing decisions."""
    return con.execute(
        """
        SELECT match_id, gameweek
        FROM fact_match
        WHERE season = ?
          AND gameweek BETWEEN ? AND ?
          AND (home_team_uid = ? OR away_team_uid = ?)
        ORDER BY gameweek, kickoff_time
        """,
        [season, start_gw, end_gw, team_uid, team_uid],
    ).fetchall()


@dataclass
class FixtureDifficulty:
    match_id: str
    gameweek: int
    is_home: bool
    opponent_uid: str
    difficulty: float  # expected goals against minus expected goals for; higher = harder


def fixture_difficulty_for_match(
    con: duckdb.DuckDBPyConnection, team_uid: str, match_id: str, gameweek: int, ts_model_version: int,
) -> FixtureDifficulty | None:
    lambdas = _team_fixture_lambdas(con, team_uid, match_id, ts_model_version)
    if lambdas is None:
        return None
    lambda_for, lambda_against, is_home, opp_uid = lambdas
    return FixtureDifficulty(match_id, gameweek, is_home, opp_uid, lambda_against - lambda_for)


@dataclass
class SwingScore:
    team_uid: str
    as_of_gameweek: int
    short_window: int
    long_window: int
    short_avg_difficulty: float | None
    long_avg_difficulty: float | None
    swing_score: float | None  # negative = swinging into EASIER fixtures; positive = swinging into HARDER ones
    n_short_fixtures: int
    n_long_fixtures: int
    fixtures: list = field(default_factory=list)


def rolling_swing_score(
    con: duckdb.DuckDBPyConnection,
    team_uid: str,
    season: str,
    as_of_gameweek: int,
    ts_model_version: int,
    short_window: int = 3,
    long_window: int = 6,
) -> SwingScore:
    """Rolling fixture-swing score for team_uid as of as_of_gameweek (inclusive).

    swing_score = short_window_avg_difficulty - long_window_avg_difficulty. Negative means
    the near-term run is easier than the fuller window containing it (a good time to hold
    or buy in); positive means it's harder (a good time to sell or avoid captaining) --
    directional relative to the team's OWN upcoming run, not relative to league-wide
    average difficulty, which would be a difficulty score rather than a swing.

    None fields mean genuinely insufficient data (e.g. fixtures not yet scheduled, or the
    team missing from this ts_model_version's fit) -- never silently coerced to 0.0, since
    0.0 is a real, different claim ("difficulty is neutral") from "unknown."
    """
    if long_window < short_window:
        raise ValueError(f"long_window ({long_window}) must be >= short_window ({short_window})")

    short_fixtures = team_fixtures_in_window(con, team_uid, season, as_of_gameweek, as_of_gameweek + short_window - 1)
    long_fixtures = team_fixtures_in_window(con, team_uid, season, as_of_gameweek, as_of_gameweek + long_window - 1)

    short_diffs = [d for d in (
        fixture_difficulty_for_match(con, team_uid, mid, gw, ts_model_version) for mid, gw in short_fixtures
    ) if d is not None]
    long_diffs = [d for d in (
        fixture_difficulty_for_match(con, team_uid, mid, gw, ts_model_version) for mid, gw in long_fixtures
    ) if d is not None]

    short_avg = sum(d.difficulty for d in short_diffs) / len(short_diffs) if short_diffs else None
    long_avg = sum(d.difficulty for d in long_diffs) / len(long_diffs) if long_diffs else None
    swing = (short_avg - long_avg) if (short_avg is not None and long_avg is not None) else None

    return SwingScore(
        team_uid=team_uid, as_of_gameweek=as_of_gameweek, short_window=short_window, long_window=long_window,
        short_avg_difficulty=short_avg, long_avg_difficulty=long_avg, swing_score=swing,
        n_short_fixtures=len(short_diffs), n_long_fixtures=len(long_diffs), fixtures=short_diffs,
    )


def swing_scores_by_team(
    con: duckdb.DuckDBPyConnection,
    season: str,
    as_of_gameweek: int,
    ts_model_version: int,
    short_window: int = 3,
    long_window: int = 6,
) -> dict[str, SwingScore]:
    """rolling_swing_score() for every team_uid with a snapshot under ts_model_version --
    the bulk form callers (transfer_planner's evaluators) actually want, so they pay one
    query fan-out instead of N individual calls."""
    team_uids = [r[0] for r in con.execute(
        "SELECT DISTINCT team_uid FROM team_strength_snapshots WHERE model_version = ?", [ts_model_version],
    ).fetchall()]
    return {
        team_uid: rolling_swing_score(con, team_uid, season, as_of_gameweek, ts_model_version, short_window, long_window)
        for team_uid in team_uids
    }
