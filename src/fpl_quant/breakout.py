"""The breakout group (docs/plans/2026-10_breakout_players.md, R2): players who have won a
starting place this season but have little Premier League starting record before it -- new to
the league, or a backup turned starter. Early 2026-27 the live model ranked several of them
100th-500th while they started every match, so the walk-forward measures this group on its
own, and a fix for it is judged on it.

A player-step (season S, gameweek G) is in the group when, using only league matches that
kicked off before G's deadline:
  (a) his club in S has played at least RECENT_MATCHES league matches, and he started at least
      MIN_RECENT_STARTS of the last RECENT_MATCHES;
  (b) across the seasons before S he played fewer than PRIOR_MINUTES_MAX league minutes, or
      started fewer than PRIOR_START_SHARE_MAX of the league matches of the clubs he was at;
  (c) his club in S is not newly promoted -- otherwise every regular at a promoted club meets
      (b), and the club-strength bias would swamp the group. Those players are labelled
      PROMOTED, reported apart and never decide anything.
A season with no earlier loaded season can't apply (b), so classify() returns None for it.

Clubs come from minutes_model's _player_season_team spells (a weekly roster snapshot puts him
at the club at G). A season without weekly snapshots gives the season-root club for the whole
season, so a mid-season mover there is judged at his later club -- a known simplification.
"""

import duckdb

from . import backtest
from . import minutes_model

RECENT_MATCHES = 4
MIN_RECENT_STARTS = 3
PRIOR_MINUTES_MAX = 900.0
PRIOR_START_SHARE_MAX = 0.40
# Declared in advance (plan, Edge Cases): the one widening allowed if the 2025-26 group is too small.
WIDENED = {"recent_matches": 3, "min_recent_starts": 2}
MIN_GROUP_SIZE = 200

BREAKOUT = "breakout"
PROMOTED = "breakout_promoted"


def build_club_spells(con: duckdb.DuckDBPyConnection, seasons: tuple[str, ...]) -> None:
    """(Re)creates the _player_season_team temp table classify() reads, for these seasons."""
    minutes_model._build_player_season_team_map(con, tuple(seasons))


def prior_seasons(con: duckdb.DuckDBPyConnection, season: str) -> list[str]:
    """Loaded league seasons before `season` (season strings sort chronologically)."""
    return [r[0] for r in con.execute(
        "SELECT DISTINCT season FROM fact_match WHERE competition = ? AND season < ? ORDER BY season",
        [backtest.PL, season],
    ).fetchall()]


def classify(
    con: duckdb.DuckDBPyConnection, season: str, gameweek: int, deadline,
    *, recent_matches: int = RECENT_MATCHES, min_recent_starts: int = MIN_RECENT_STARTS,
    promoted_memo: dict | None = None,
) -> dict[str, str] | None:
    """{player_uid: BREAKOUT | PROMOTED} for the players in the group at this step (everyone
    else is absent), or None when `season` has no earlier loaded season. Needs the
    _player_season_team temp table for `season` and every earlier season (build_club_spells())."""
    earlier = prior_seasons(con, season)
    if not earlier:
        return None
    rows = con.execute(
        """
        WITH cur AS (
            SELECT player_uid, team_uid FROM _player_season_team
            WHERE season = ? AND first_gw <= ? AND last_gw >= ?
            QUALIFY row_number() OVER (PARTITION BY player_uid ORDER BY first_gw DESC) = 1
        ),
        club_matches AS (
            SELECT t.team_uid, m.match_id,
                   row_number() OVER (PARTITION BY t.team_uid ORDER BY m.kickoff_time DESC) AS rn,
                   count(*) OVER (PARTITION BY t.team_uid) AS n_played
            FROM (SELECT DISTINCT team_uid FROM cur) t
            JOIN fact_match m ON t.team_uid IN (m.home_team_uid, m.away_team_uid)
            WHERE m.season = ? AND m.competition = ? AND m.kickoff_time < ? AND m.home_score IS NOT NULL
        ),
        recent AS (
            SELECT c.player_uid, c.team_uid, max(cm.n_played) AS n_played,
                   count(pms.match_id) FILTER (WHERE pms.start_min = 0) AS recent_starts
            FROM cur c
            JOIN club_matches cm ON cm.team_uid = c.team_uid AND cm.rn <= ?
            LEFT JOIN fact_player_match_stats pms ON pms.player_uid = c.player_uid AND pms.match_id = cm.match_id
            GROUP BY c.player_uid, c.team_uid
        ),
        prior_club_matches AS (
            SELECT DISTINCT s.player_uid, m.match_id
            FROM _player_season_team s
            JOIN fact_match m ON m.season = s.season AND m.competition = ?
                AND s.team_uid IN (m.home_team_uid, m.away_team_uid)
                AND coalesce(m.gameweek, 0) BETWEEN s.first_gw AND s.last_gw
            WHERE s.season < ?
        ),
        prior_starts AS (
            SELECT p.player_uid, count(*) AS club_matches,
                   count(pms.match_id) FILTER (WHERE pms.start_min = 0) AS starts
            FROM prior_club_matches p
            LEFT JOIN fact_player_match_stats pms ON pms.player_uid = p.player_uid AND pms.match_id = p.match_id
            GROUP BY p.player_uid
        ),
        prior_minutes AS (
            SELECT pms.player_uid, sum(coalesce(pms.minutes_played, 0)) AS minutes
            FROM fact_player_match_stats pms
            JOIN fact_match m ON m.match_id = pms.match_id
            WHERE m.season < ? AND m.competition = ?
            GROUP BY pms.player_uid
        )
        SELECT r.player_uid, r.team_uid,
               coalesce(pm.minutes, 0) AS prior_minutes,
               coalesce(ps.starts, 0) AS prior_starts, coalesce(ps.club_matches, 0) AS prior_club_matches
        FROM recent r
        LEFT JOIN prior_minutes pm ON pm.player_uid = r.player_uid
        LEFT JOIN prior_starts ps ON ps.player_uid = r.player_uid
        WHERE r.n_played >= ? AND r.recent_starts >= ?
        """,
        [season, gameweek, gameweek, season, backtest.PL, deadline, recent_matches,
         backtest.PL, season, season, backtest.PL, recent_matches, min_recent_starts],
    ).fetchall()
    memo = promoted_memo if promoted_memo is not None else {}
    out: dict[str, str] = {}
    for uid, team_uid, prior_minutes, prior_starts, prior_club_matches in rows:
        share = prior_starts / prior_club_matches if prior_club_matches else 0.0
        if not (prior_minutes < PRIOR_MINUTES_MAX or share < PRIOR_START_SHARE_MAX):
            continue
        key = (team_uid, season)
        if key not in memo:
            memo[key] = bool(backtest._is_newly_promoted_team(con, team_uid, season))
        out[uid] = PROMOTED if memo[key] else BREAKOUT
    return out
