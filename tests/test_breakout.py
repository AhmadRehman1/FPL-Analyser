from datetime import datetime, timedelta

from fpl_quant import breakout, db

NOW = datetime(2026, 1, 1)
PRIOR, SEASON = "2024-2025", "2025-2026"


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


def _seed(con):
    """2024-25: A v B, 20 matches. 2025-26: A v B and P v C (both newly promoted), 5 gameweeks.
    new      -- at A, no earlier league minutes, starts every 2025-26 match
    backup   -- at A, 2024-25: 4 starts of 20 (1320 minutes), starts every 2025-26 match
    regular  -- at B, started all 20 in 2024-25, starts every 2025-26 match
    promoted -- at P (promoted), no earlier league minutes, starts every 2025-26 match
    few      -- at D (D v E, also new), starts all of D's matches, but D has played only GW1-3"""
    for team in ("team_a", "team_b", "team_p", "team_c", "team_d", "team_e"):
        con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES (?, ?)", [team, team])
    for uid in ("new", "backup", "regular", "promoted", "few"):
        con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, 'Midfielder')", [uid, uid])
    con.execute(
        "CREATE OR REPLACE TEMP TABLE _player_season_team "
        "(player_uid VARCHAR, season VARCHAR, team_uid VARCHAR, first_gw INTEGER, last_gw INTEGER)"
    )
    for uid, season, team in [
        ("backup", PRIOR, "team_a"), ("regular", PRIOR, "team_b"),
        ("new", SEASON, "team_a"), ("backup", SEASON, "team_a"), ("regular", SEASON, "team_b"),
        ("promoted", SEASON, "team_p"), ("few", SEASON, "team_d"),
    ]:
        con.execute("INSERT INTO _player_season_team VALUES (?, ?, ?, 0, 99)", [uid, season, team])
    start = datetime(2024, 8, 17)
    for gw in range(1, 21):
        mid = f"prior_{gw}"
        _match(con, mid, PRIOR, gw, "team_a", "team_b", start + timedelta(days=7 * gw))
        _played(con, "backup", mid, PRIOR, gw <= 4, 90 if gw <= 4 else 60)
        _played(con, "regular", mid, PRIOR, True, 90)
    start = datetime(2025, 8, 16)
    for gw in range(1, 6):
        kickoff = start + timedelta(days=7 * gw)
        _match(con, f"ab_{gw}", SEASON, gw, "team_a", "team_b", kickoff)
        _match(con, f"pc_{gw}", SEASON, gw, "team_p", "team_c", kickoff)
        for uid in ("new", "backup"):
            _played(con, uid, f"ab_{gw}", SEASON, True, 90)
        _played(con, "regular", f"ab_{gw}", SEASON, True, 90)
        _played(con, "promoted", f"pc_{gw}", SEASON, True, 90)
        if gw <= 3:
            _match(con, f"de_{gw}", SEASON, gw, "team_d", "team_e", kickoff)
            _played(con, "few", f"de_{gw}", SEASON, True, 90)


def _deadline_after(gw):
    return datetime(2025, 8, 16) + timedelta(days=7 * gw + 1)


def test_new_and_backup_starters_are_breakout_promoted_apart_regular_and_few_out(con):
    _seed(con)
    out = breakout.classify(con, SEASON, 6, _deadline_after(5))
    # few: his club has played 3 matches by this deadline, not 4
    assert out == {"new": breakout.BREAKOUT, "backup": breakout.BREAKOUT, "promoted": breakout.PROMOTED}


def test_needs_four_club_matches_before_the_deadline(con):
    _seed(con)
    # only 3 club matches played by GW4's deadline: nobody qualifies yet
    assert breakout.classify(con, SEASON, 4, _deadline_after(3)) == {}
    # the declared widening (2 of the last 3) lets them in
    widened = breakout.classify(con, SEASON, 4, _deadline_after(3), **breakout.WIDENED)
    assert widened == {
        "new": breakout.BREAKOUT, "backup": breakout.BREAKOUT, "promoted": breakout.PROMOTED,
        "few": breakout.PROMOTED,  # D is new to the league too
    }


def test_a_heavy_prior_player_without_club_matches_is_judged_on_minutes(con):
    _seed(con)
    # 1800 earlier league minutes but no club spell on record that season (a join miss)
    con.execute("DELETE FROM _player_season_team WHERE player_uid = 'regular' AND season = ?", [PRIOR])
    assert "regular" not in breakout.classify(con, SEASON, 6, _deadline_after(5))


def test_club_spells_build_on_a_read_only_connection(tmp_path):
    """The walk-forward scoreboard opens the DB read-only; building the spells must not write.
    An older DB also carries a stored norm_id macro; the session's own must win."""
    path = tmp_path / "ro.duckdb"
    con = db.connect(path)
    con.execute('CREATE TABLE "raw_teams" (code VARCHAR, id VARCHAR, name VARCHAR)')
    con.execute("INSERT INTO raw_teams VALUES ('3.0', '1', 'Alpha')")
    con.execute('CREATE TABLE "raw_gw1_players" (player_code VARCHAR, team_code VARCHAR)')
    con.execute("INSERT INTO raw_gw1_players VALUES ('77', '3')")
    con.execute(
        "INSERT INTO fact_raw_ingestion_log (raw_table_name, season, source_relpath, source_file_hash, row_count) "
        "VALUES ('raw_teams', ?, 'teams.csv', 'h1', 1), ('raw_gw1_players', ?, 'By Gameweek/GW1/players.csv', 'h2', 1)",
        [SEASON, SEASON],
    )
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('team_alpha', 'Alpha')")
    con.execute("INSERT INTO team_alias (alias_name, season, team_uid) VALUES ('Alpha', ?, 'team_alpha')", [SEASON])
    con.execute("INSERT INTO dim_player (player_uid, canonical_name) VALUES ('p77', 'P')")
    con.execute(
        "INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid, source_player_id) "
        "VALUES ('P', 'p', '3', ?, 'p77', '77')", [SEASON],
    )
    con.execute("CREATE OR REPLACE MACRO norm_id(x) AS x")  # a stale stored copy: no '3.0' -> '3'
    con.close()

    ro = db.connect(path, read_only=True)
    try:
        breakout.build_club_spells(ro, (SEASON,))
        assert ro.execute("SELECT player_uid, team_uid, first_gw FROM _player_season_team").fetchall() == [
            ("p77", "team_alpha", 1),
        ]
    finally:
        ro.close()


def test_only_matches_before_the_deadline_count(con):
    _seed(con)
    # GW5's own match is after GW5's deadline; a benching in it must not change the GW5 step
    con.execute("UPDATE fact_player_match_stats SET start_min = 60 WHERE player_uid = 'new' AND match_id = 'ab_5'")
    assert breakout.classify(con, SEASON, 5, _deadline_after(4))["new"] == breakout.BREAKOUT
    # ...but once it is in the past, 3 of the last 4 still holds; two benchings break it
    con.execute("UPDATE fact_player_match_stats SET start_min = 60 WHERE player_uid = 'new' AND match_id = 'ab_4'")
    assert "new" not in breakout.classify(con, SEASON, 6, _deadline_after(5))


def test_no_earlier_season_gives_none(con):
    _seed(con)
    assert breakout.classify(con, PRIOR, 10, datetime(2024, 11, 1)) is None
