from datetime import datetime, timedelta

from fpl_quant import form_groups

NOW = datetime(2026, 1, 1)
PRIOR, SEASON = "2024-2025", "2025-2026"
# (xG + xA) per 90: (2024-25, 2025-26). hot 2x, warm 1.25x, steady 1.1x, cold 0.5x; the keeper's
# ratio is 4x but keepers are never classified.
RATES = {
    "hot": (0.30, 0.60),
    "warm": (0.40, 0.50),
    "steady": (0.40, 0.44),
    "cold": (0.60, 0.30),
    "keeper": (0.05, 0.20),
}


def _match(con, mid, season, gw, home, away, kickoff):
    con.execute(
        "INSERT INTO fact_match (match_id, season, gameweek, kickoff_time, home_team_uid, away_team_uid, "
        "home_score, away_score, finished, competition, _ingested_at) "
        "VALUES (?, ?, ?, ?, ?, ?, 1, 0, TRUE, 'Premier League', ?)",
        [mid, season, gw, kickoff, home, away, NOW],
    )


def _played(con, uid, mid, season, started, minutes):
    con.execute(
        "INSERT INTO fact_player_match_stats (player_uid, match_id, season, start_min, minutes_played, _ingested_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [uid, mid, season, 0 if started else 90 - minutes, minutes, NOW],
    )


def _season_row(con, uid, season, gw, *, minutes=None, xg=None, xa=None, xg90=None, xa90=None):
    con.execute(
        "INSERT INTO fact_player_season_stats (player_uid, season, gw, minutes, expected_goals, expected_assists, "
        "expected_goals_per_90, expected_assists_per_90, _ingested_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [uid, season, gw, minutes, xg, xa, xg90, xa90, NOW],
    )


def _seed(con):
    """2024-25: A v B, 20 matches, everyone (but new) starts all of them. Its season rows carry
    per-90 rates only, with NULL minutes and totals (the real 2024-25 shape). 2025-26: A v B,
    5 gameweeks, everyone starts; cumulative rows per gameweek with season totals.
    new -- at A, no earlier league minutes: a breakout player, never in form."""
    for team in ("team_a", "team_b"):
        con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES (?, ?)", [team, team])
    players = {"hot": "team_a", "warm": "team_a", "steady": "team_a", "cold": "team_b", "keeper": "team_b", "new": "team_a"}
    for uid in players:
        position = "Goalkeeper" if uid == "keeper" else "Midfielder"
        con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, ?)", [uid, uid, position])
    con.execute(
        "CREATE OR REPLACE TEMP TABLE _player_season_team "
        "(player_uid VARCHAR, season VARCHAR, team_uid VARCHAR, first_gw INTEGER, last_gw INTEGER)"
    )
    for uid, team in players.items():
        con.execute("INSERT INTO _player_season_team VALUES (?, ?, ?, 0, 99)", [uid, SEASON, team])
        if uid != "new":
            con.execute("INSERT INTO _player_season_team VALUES (?, ?, ?, 0, 99)", [uid, PRIOR, team])
    start = datetime(2024, 8, 17)
    for gw in range(1, 21):
        mid = f"prior_{gw}"
        _match(con, mid, PRIOR, gw, "team_a", "team_b", start + timedelta(days=7 * gw))
        for uid in RATES:
            _played(con, uid, mid, PRIOR, True, 90)
    for uid, (prior_rate, _) in RATES.items():
        _season_row(con, uid, PRIOR, 20, xg90=prior_rate * 2 / 3, xa90=prior_rate / 3)
    start = datetime(2025, 8, 16)
    for gw in range(1, 6):
        mid = f"ab_{gw}"
        _match(con, mid, SEASON, gw, "team_a", "team_b", start + timedelta(days=7 * gw))
        for uid in players:
            _played(con, uid, mid, SEASON, True, 90)
            rate = RATES.get(uid, (0.0, 0.5))[1]
            mins = 90 * gw
            _season_row(con, uid, SEASON, gw, minutes=mins, xg=rate * mins / 90 * 2 / 3, xa=rate * mins / 90 / 3)


def _deadline_after(gw):
    return datetime(2025, 8, 16) + timedelta(days=7 * gw + 1)


def test_in_form_and_cooling_from_a_per_90_only_earlier_season(con):
    _seed(con)
    out = form_groups.classify(con, SEASON, 6, _deadline_after(5))
    # the earlier season has NULL minutes and totals; its rate comes from per-90 x match minutes.
    # warm (1.25x) and steady (1.1x) are in neither group, the keeper never is, new is a breakout
    assert out == {"hot": form_groups.IN_FORM, "cold": form_groups.COOLING}


def test_the_declared_widening(con):
    _seed(con)
    out = form_groups.classify(con, SEASON, 6, _deadline_after(5), **form_groups.WIDENED)
    assert out == {"hot": form_groups.IN_FORM, "warm": form_groups.IN_FORM, "cold": form_groups.COOLING}


def test_needs_360_league_minutes_this_season(con):
    _seed(con)
    # GW4: the latest row before it is GW3's, 270 minutes
    assert form_groups.classify(con, SEASON, 4, _deadline_after(3)) == {}
    # GW5: GW4's row, 360 minutes
    assert form_groups.classify(con, SEASON, 5, _deadline_after(4)) == {"hot": form_groups.IN_FORM, "cold": form_groups.COOLING}


def test_needs_three_starts_in_the_last_four(con):
    _seed(con)
    con.execute("UPDATE fact_player_match_stats SET start_min = 60 WHERE player_uid = 'hot' AND match_id IN ('ab_4', 'ab_5')")
    assert "hot" not in form_groups.classify(con, SEASON, 6, _deadline_after(5))


def test_rows_from_the_gameweek_on_change_nothing(con):
    """Row G already holds G's own matches: only rows with gw < G count, so changing row G (or
    adding later ones) leaves the step alone."""
    _seed(con)
    before = form_groups.classify(con, SEASON, 5, _deadline_after(4))
    con.execute("UPDATE fact_player_season_stats SET expected_goals = 0, expected_assists = 0 "
                "WHERE player_uid = 'hot' AND season = ? AND gw = 5", [SEASON])
    _season_row(con, "cold", SEASON, 6, minutes=540, xg=20.0, xa=10.0)
    assert form_groups.classify(con, SEASON, 5, _deadline_after(4)) == before == {
        "hot": form_groups.IN_FORM, "cold": form_groups.COOLING,
    }


def test_details_and_memo(con):
    _seed(con)
    memo, details = {}, {}
    form_groups.classify(con, SEASON, 6, _deadline_after(5), memo=memo, details=details)
    assert details["hot"]["ratio"] == 2.0
    assert details["hot"]["earlier_minutes"] == 1800.0
    assert "keeper" not in details and "new" not in details
    # the earlier-season pool is reused across steps
    assert ("hot", (PRIOR,)) in memo["earlier"]


def test_a_season_without_an_earlier_one_returns_none(con):
    _seed(con)
    assert form_groups.classify(con, PRIOR, 6, datetime(2024, 10, 1)) is None
