from fpl_quant import fixture_swing as fs

TARGET_SEASON = "2026-2027"


def _seed_team_strength(con, teams: dict[str, tuple[float, float]], target_season=TARGET_SEASON):
    """teams: {team_uid: (final_attack, final_defence)}. Minimal fixture -- fixture_swing only
    ever reads fact_match + team_strength_model_versions + team_strength_snapshots (via
    expected_points._fixture_lambdas), never the raw-ingestion/player_alias machinery
    monte_carlo's field-covariance fixtures need."""
    for team_uid in teams:
        con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES (?, ?)", [team_uid, team_uid])
    con.execute(
        "INSERT INTO team_strength_model_versions (calibration_asof_date, home_advantage, xi_params_version, "
        "rho_params_version, reference_team_uid) VALUES ('2026-08-10', 0.0, 1, 1, ?)",
        [next(iter(teams))],
    )
    ts_mv = con.execute("SELECT max(model_version) FROM team_strength_model_versions").fetchone()[0]
    for team_uid, (attack, defence) in teams.items():
        con.execute(
            "INSERT INTO team_strength_snapshots (model_version, team_uid, final_attack, final_defence, "
            "seasons_of_topflight_data, weight_own_data) VALUES (?, ?, ?, ?, 2, 1.0)",
            [ts_mv, team_uid, attack, defence],
        )
    return ts_mv


def _seed_fixture(con, gameweek, home_uid, away_uid, target_season=TARGET_SEASON):
    con.execute(
        "INSERT INTO fact_match (match_id, season, gameweek, home_team_uid, away_team_uid, finished, "
        "competition, kickoff_time, _ingested_at) VALUES (?, ?, ?, ?, ?, FALSE, 'Premier League', "
        "'2026-08-24', current_timestamp)",
        [f"m_{home_uid}_{away_uid}_{gameweek}", target_season, gameweek, home_uid, away_uid],
    )


def test_favorability_none_for_a_blank_gameweek(con):
    ts_mv = _seed_team_strength(con, {"team_x": (0.0, 0.0), "opp": (0.0, 0.0)})
    _seed_fixture(con, 1, "team_x", "opp")
    assert fs.favorability(con, "team_x", 2, TARGET_SEASON, ts_mv) is None


def test_favorability_positive_for_a_weak_opponent_negative_for_a_strong_one(con):
    ts_mv = _seed_team_strength(con, {
        "team_x": (0.0, 0.0), "weak_opp": (-1.0, -1.0), "strong_opp": (1.0, 1.0),
    })
    _seed_fixture(con, 1, "team_x", "weak_opp")
    _seed_fixture(con, 2, "team_x", "strong_opp")
    assert fs.favorability(con, "team_x", 1, TARGET_SEASON, ts_mv) > 0
    assert fs.favorability(con, "team_x", 2, TARGET_SEASON, ts_mv) < 0


def test_rolling_avg_skips_blank_gameweeks_rather_than_zero_filling(con):
    ts_mv = _seed_team_strength(con, {"team_x": (0.0, 0.0), "weak_opp": (-1.0, -1.0)})
    _seed_fixture(con, 1, "team_x", "weak_opp")
    # GW2, 3 deliberately left blank for team_x
    _seed_fixture(con, 4, "team_x", "weak_opp")
    avg_with_blanks = fs.rolling_avg(con, "team_x", 1, 4, TARGET_SEASON, ts_mv)
    single_fixture = fs.favorability(con, "team_x", 1, TARGET_SEASON, ts_mv)
    assert avg_with_blanks == single_fixture  # the two real fixtures are identical -- their
    # mean should equal either one, not be diluted by the two blank gameweeks


def test_rolling_avg_none_when_every_gameweek_in_window_is_blank(con):
    ts_mv = _seed_team_strength(con, {"team_x": (0.0, 0.0), "opp": (0.0, 0.0)})
    _seed_fixture(con, 10, "team_x", "opp")  # outside the window queried below
    assert fs.rolling_avg(con, "team_x", 1, 3, TARGET_SEASON, ts_mv) is None


def _seed_swing_league(con, weak_first: bool):
    """team_x plays 6 distinct opponents across GW1-6: either 3 weak-then-3-strong
    (weak_first=True) or 3 strong-then-3-weak (weak_first=False). A distinct opponent per
    gameweek (not one reused team) avoids any accidental team_strength_snapshots overwrite."""
    weak, strong = (-1.0, -1.0), (1.0, 1.0)
    sequence = [weak] * 3 + [strong] * 3 if weak_first else [strong] * 3 + [weak] * 3
    teams = {"team_x": (0.0, 0.0)}
    for i, (attack, defence) in enumerate(sequence, start=1):
        teams[f"opp{i}"] = (attack, defence)
    ts_mv = _seed_team_strength(con, teams)
    for i in range(1, 7):
        _seed_fixture(con, i, "team_x", f"opp{i}")
    return ts_mv


def test_swing_score_positive_when_near_term_run_is_easier_than_the_longer_baseline(con):
    """3 weak opponents (GW1-3) then 3 strong (GW4-6): at GW1, the short (1-3) window is all-
    weak (favorable), while the long (1-6) window averages in the 3 strong ones too -- the
    near-term run is genuinely easier than the team's own longer-term baseline."""
    ts_mv = _seed_swing_league(con, weak_first=True)
    score = fs.swing_score(con, "team_x", 1, TARGET_SEASON, ts_mv, short=3, long=6)
    assert score > 0


def test_swing_score_negative_when_near_term_run_is_harder_than_the_longer_baseline(con):
    """The exact mirror scenario, proving direction genuinely flips (not just "always
    positive" from some unrelated bias): 3 strong opponents (GW1-3) then 3 weak (GW4-6) --
    at GW1 the near-term run is harder than the longer-term baseline."""
    ts_mv = _seed_swing_league(con, weak_first=False)
    score = fs.swing_score(con, "team_x", 1, TARGET_SEASON, ts_mv, short=3, long=6)
    assert score < 0


def test_swing_score_none_when_a_window_has_no_fixtures(con):
    ts_mv = _seed_team_strength(con, {"team_x": (0.0, 0.0), "opp": (0.0, 0.0)})
    _seed_fixture(con, 10, "team_x", "opp")  # outside both the short and long windows queried below
    assert fs.swing_score(con, "team_x", 1, TARGET_SEASON, ts_mv, short=3, long=6) is None


def test_swing_score_by_team_batches_every_team_with_a_snapshot(con):
    ts_mv = _seed_swing_league(con, weak_first=True)
    scores = fs.swing_score_by_team(con, 1, TARGET_SEASON, ts_mv, short=3, long=6)
    assert "team_x" in scores
    assert scores["team_x"] > 0
    # opponent teams only got ONE fixture each seeded (vs team_x) -- both windows collapse to
    # that single fixture for them, so their own swing_score is exactly 0.0, still present
    # (not silently dropped) since 0.0 is a real, computed value, not a missing one.
    assert scores["opp1"] == 0.0
