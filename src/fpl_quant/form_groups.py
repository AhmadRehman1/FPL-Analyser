"""The in-form and cooling groups (docs/plans/2026-10_premium_and_form_returns.md, RQ4): established
starters whose underlying attacking output this season runs well above (or below) their own
earlier seasons'. Early 2026-27 the live model held Mbeumo's goal rate at 0.43 per 90 against a
season xG of 0.61, so the walk-forward measures these players on their own, and a change aimed at
them is judged on them.

A player-step (season S, gameweek G) is IN_FORM when, using only rows from before G's deadline:
  (a) he started at least MIN_RECENT_STARTS of his club's last RECENT_MATCHES league matches
      (breakout.py's counting);
  (b) he has at least MIN_SEASON_MINUTES league minutes in S, his earlier loaded seasons hold at
      least MIN_EARLIER_MINUTES, and his (xG + xA) per 90 in S is at least RATIO times the earlier
      seasons'. S is read from his latest fact_player_season_stats row with gw < G (row G already
      holds G's own matches); the earlier seasons from expected_points._player_rate_pool(), which
      rebuilds 2024-25's per-90-only rows from match minutes;
  (c) he is a midfielder or forward (fixed 2026-10-11 in Phase 1, before any arm: with defenders
      in, the live group was mostly defenders whose (xG + xA) per 90 went from 0.03 to 0.10, a
      ratio of noise on a scoring rate that barely moves their points), and not in the breakout
      group (breakout.classify, declared rule, either label): those players have no earlier
      record to be in form against.
COOLING is the same with the ratio at most 1 / RATIO. An earlier rate of 0 classifies nobody.
A season with no earlier loaded season returns None.

Needs breakout.build_club_spells() for S and every earlier season.
"""

import duckdb

from . import backtest
from . import breakout
from . import expected_points as ep

RECENT_MATCHES = breakout.RECENT_MATCHES
MIN_RECENT_STARTS = breakout.MIN_RECENT_STARTS
MIN_SEASON_MINUTES = 360.0
MIN_EARLIER_MINUTES = 900.0
RATIO = 1.30
# Declared in advance (plan, RQ4): the one widening allowed if the 2025-26 group is too small.
WIDENED = {"ratio": 1.20}
MIN_GROUP_SIZE = 150

IN_FORM = "in_form"
COOLING = "cooling"


def _season_rates(con: duckdb.DuckDBPyConnection, season: str, gameweek: int, deadline, uids: list[str]) -> dict:
    """{player_uid: (league minutes, (xG + xA) per 90)} in `season` before G: his latest
    fact_player_season_stats row with gw < G. A row without season totals (2024-25's shape) takes
    its per-90 rates and his league minutes from the match feed before the deadline."""
    if not uids:
        return {}
    rows = con.execute(
        "SELECT player_uid, minutes, expected_goals, expected_assists, expected_goals_per_90, expected_assists_per_90 "
        "FROM fact_player_season_stats WHERE season = ? AND gw < ? AND player_uid IN (SELECT unnest(?::VARCHAR[])) "
        "QUALIFY row_number() OVER (PARTITION BY player_uid ORDER BY gw DESC) = 1",
        [season, gameweek, uids],
    ).fetchall()
    out = {}
    for uid, minutes, xg, xa, xg90, xa90 in rows:
        if minutes and xg is not None:
            out[uid] = (float(minutes), ((xg or 0.0) + (xa or 0.0)) / minutes * 90.0)
        elif xg90 is not None or xa90 is not None:
            mins = con.execute(
                "SELECT sum(coalesce(s.minutes_played, 0)) FROM fact_player_match_stats s "
                "JOIN fact_match m ON m.match_id = s.match_id "
                "WHERE s.player_uid = ? AND m.season = ? AND m.competition = ? AND m.kickoff_time < ?",
                [uid, season, backtest.PL, deadline],
            ).fetchone()[0]
            out[uid] = (float(mins or 0.0), (xg90 or 0.0) + (xa90 or 0.0))
    return out


def classify(
    con: duckdb.DuckDBPyConnection, season: str, gameweek: int, deadline,
    *, ratio: float = RATIO, recent_matches: int = RECENT_MATCHES, min_recent_starts: int = MIN_RECENT_STARTS,
    memo: dict | None = None, details: dict | None = None,
) -> dict[str, str] | None:
    """{player_uid: IN_FORM | COOLING} for the players in either group at this step (everyone else
    is absent), or None when `season` has no earlier loaded season. `memo` carries the earlier-season
    rates and promoted-club lookups across steps; `details`, when given, is filled with
    {player_uid: {season_minutes, season_per90, earlier_minutes, earlier_per90, ratio}} for every
    player who reached the rate comparison."""
    earlier = breakout.prior_seasons(con, season)
    if not earlier:
        return None
    memo = memo if memo is not None else {}
    starters = [r[0] for r in con.execute(
        f"""
        WITH {breakout.RECENT_STARTS_CTES}
        SELECT r.player_uid FROM recent r
        JOIN dim_player dp ON dp.player_uid = r.player_uid
        WHERE r.n_played >= ? AND r.recent_starts >= ? AND dp.position IN ('Midfielder', 'Forward')
        ORDER BY r.player_uid
        """,
        [season, gameweek, gameweek, season, backtest.PL, deadline, recent_matches,
         recent_matches, min_recent_starts],
    ).fetchall()]
    breakouts = breakout.classify(con, season, gameweek, deadline, promoted_memo=memo.setdefault("promoted", {}))
    starters = [uid for uid in starters if uid not in breakouts]
    this_season = _season_rates(con, season, gameweek, deadline, starters)
    pools = memo.setdefault("earlier", {})
    out: dict[str, str] = {}
    for uid in starters:
        if uid not in this_season:
            continue
        season_minutes, season_per90 = this_season[uid]
        key = (uid, tuple(earlier))
        if key not in pools:
            pool = ep._player_rate_pool(con, uid, earlier)
            pools[key] = (pool["sample_minutes"], pool["expected_goals_per_90"] + pool["expected_assists_per_90"])
        earlier_minutes, earlier_per90 = pools[key]
        if season_minutes < MIN_SEASON_MINUTES or earlier_minutes < MIN_EARLIER_MINUTES or earlier_per90 <= 0:
            continue
        r = season_per90 / earlier_per90
        if details is not None:
            details[uid] = {
                "season_minutes": season_minutes, "season_per90": round(season_per90, 4),
                "earlier_minutes": earlier_minutes, "earlier_per90": round(earlier_per90, 4), "ratio": round(r, 4),
            }
        if r >= ratio:
            out[uid] = IN_FORM
        elif r <= 1.0 / ratio:
            out[uid] = COOLING
    return out
