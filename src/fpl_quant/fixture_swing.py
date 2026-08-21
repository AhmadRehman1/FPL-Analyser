"""A7: rolling fixture-swing signal.

FPL's own Fixture Difficulty Rating (FDR) is a static, hand-assigned 1-5 scale per fixture
that never updates as a team's actual attacking/defensive strength evolves through the
season. This project already fits a live Dixon-Coles attack/defence rating per team every
time team_strength.calibrate() runs (see M1/M1a) -- expected_points._fixture_lambdas() turns
that into a per-fixture lambda_for/lambda_against for exactly the fixture EP itself is computed
from. This module reuses that SAME live-fitted source of truth for a genuinely different
question EP's per-gameweek loop doesn't answer on its own: not "how good is this one fixture",
but "is this team's near-term fixture run trending easier or harder than their own longer-term
baseline" -- a trajectory signal, not a snapshot.

Deliberately NOT sourced from 30_Fixture Swing GW1-10 (the evidence workbook's own tab covering
this exact topic): that tab is marked EXCLUDED_DEPRECATED in ingest_workbook.py's M0 allowlist
(a static, one-time hand analysis that goes stale the moment real results start moving the
Dixon-Coles ratings) and is never resurrected here -- every number this module returns is a
live function of team_strength_snapshots as of whichever ts_model_version is passed in, so it
updates automatically every time the model is recalibrated, the same guarantee every other M1
consumer in this project already has.

short/long window sizes are plain function defaults (3, 6 gameweeks), not DB-versioned params:
unlike lambda_value/kappa_captain (which change what gets selected/stored), these only shape a
derived reporting signal computed on demand, so there's no stored-output provenance need for
versioning them the way params_mod.write_param()'s discipline exists for.
"""

import duckdb

from . import expected_points as ep_mod


def favorability(
    con: duckdb.DuckDBPyConnection, team_uid: str, gameweek: int, target_season: str, ts_model_version: int,
) -> float | None:
    """lambda_for - lambda_against for this team's fixture in `gameweek`, reusing M1's own
    _fixture_lambdas() -- the same live Dixon-Coles-fitted team strengths every other consumer
    of fixture difficulty in this project already uses. Positive = attack-favored fixture,
    negative = defense-favored (tough) fixture. None for a genuine blank gameweek (no fixture
    that gameweek for this team) -- absence of a fixture isn't a neutral/0.0 one, and silently
    treating it as 0.0 would flatten a real blank-gameweek signal into a fake "average" fixture.
    """
    match = con.execute(
        "SELECT match_id FROM fact_match WHERE season = ? AND gameweek = ? AND competition = ? "
        "AND (home_team_uid = ? OR away_team_uid = ?)",
        [target_season, gameweek, ep_mod.PL, team_uid, team_uid],
    ).fetchone()
    if not match:
        return None
    lambda_for, lambda_against, _is_home = ep_mod._fixture_lambdas(con, team_uid, match[0], ts_model_version)
    return lambda_for - lambda_against


def rolling_avg(
    con: duckdb.DuckDBPyConnection, team_uid: str, start_gameweek: int, window: int,
    target_season: str, ts_model_version: int,
) -> float | None:
    """Mean favorability() over [start_gameweek, start_gameweek + window). Blank gameweeks
    (favorability() returning None) are skipped, not counted as 0.0 -- a missing fixture
    shouldn't drag the average toward "neutral", it should simply not contribute a data point.
    None only if every gameweek in the window is blank (nothing at all to average)."""
    values = [
        v for gw in range(start_gameweek, start_gameweek + window)
        if (v := favorability(con, team_uid, gw, target_season, ts_model_version)) is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def swing_score(
    con: duckdb.DuckDBPyConnection, team_uid: str, gameweek: int, target_season: str, ts_model_version: int,
    short: int = 3, long: int = 6,
) -> float | None:
    """rolling_avg(short window) - rolling_avg(long window), both starting at `gameweek`.
    Positive = this team's near-term fixture run is MORE favorable than their own longer-term
    baseline (a genuine "swing toward easier" -- a real signal about trajectory, not just "this
    team has easy fixtures", which a single-window average would already capture on its own).
    Negative = trending harder soon. None if either window has no fixtures to average at all
    (e.g. querying past the end of the currently-scheduled fixture list)."""
    short_avg = rolling_avg(con, team_uid, gameweek, short, target_season, ts_model_version)
    long_avg = rolling_avg(con, team_uid, gameweek, long, target_season, ts_model_version)
    if short_avg is None or long_avg is None:
        return None
    return short_avg - long_avg


def swing_score_by_team(
    con: duckdb.DuckDBPyConnection, gameweek: int, target_season: str, ts_model_version: int,
    short: int = 3, long: int = 6,
) -> dict[str, float]:
    """swing_score() for every team with a team_strength_snapshots row at this ts_model_version,
    in one batch -- the shape transfer_planner/A9-A11's per-gameweek callers actually need
    (avoiding one query-per-team-per-candidate). Teams with no computable swing_score (both
    windows blank) are simply absent from the returned dict, not included as 0.0."""
    team_uids = [
        r[0] for r in con.execute(
            "SELECT DISTINCT team_uid FROM team_strength_snapshots WHERE model_version = ?", [ts_model_version],
        ).fetchall()
    ]
    out = {}
    for team_uid in team_uids:
        score = swing_score(con, team_uid, gameweek, target_season, ts_model_version, short, long)
        if score is not None:
            out[team_uid] = score
    return out
