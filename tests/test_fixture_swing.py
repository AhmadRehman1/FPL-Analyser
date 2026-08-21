import pytest

from fpl_quant import expected_points as ep
from fpl_quant import fixture_swing as fs


def _seed_teams_and_strength(con, team_attack_defence: list[tuple[str, float, float]]) -> int:
    """team_attack_defence: list of (team_uid, attack, defence) all under one fresh
    team_strength_model_versions row. Returns the new model_version."""
    con.execute(
        "INSERT INTO team_strength_model_versions (calibration_asof_date, home_advantage, xi_params_version, "
        "rho_params_version, reference_team_uid) VALUES ('2026-08-10', 0.2, 1, 1, ?)",
        [team_attack_defence[0][0]],
    )
    ts_mv = con.execute("SELECT max(model_version) FROM team_strength_model_versions").fetchone()[0]
    for team_uid, attack, defence in team_attack_defence:
        con.execute(
            "INSERT INTO dim_team (team_uid, canonical_name) VALUES (?, ?) ON CONFLICT DO NOTHING",
            [team_uid, team_uid],
        )
        con.execute(
            "INSERT INTO team_strength_snapshots (model_version, team_uid, final_attack, final_defence, "
            "seasons_of_topflight_data, weight_own_data) VALUES (?, ?, ?, ?, 2, 1.0)",
            [ts_mv, team_uid, attack, defence],
        )
    return ts_mv


def _insert_fixture(con, match_id, season, gw, home_uid, away_uid):
    con.execute(
        "INSERT INTO fact_match (match_id, season, gameweek, home_team_uid, away_team_uid, finished, "
        "competition, kickoff_time, _ingested_at) VALUES (?, ?, ?, ?, ?, FALSE, 'Premier League', "
        "'2026-08-24', current_timestamp)",
        [match_id, season, gw, home_uid, away_uid],
    )


# ============================================================
# lambda formula must match M3 exactly -- prevents this module silently drifting from
# what "difficulty" means everywhere else in the pipeline
# ============================================================

def test_lambda_formula_matches_expected_points(con):
    ts_mv = _seed_teams_and_strength(con, [("team_a", 0.35, -0.05), ("team_b", -0.10, 0.12)])
    _insert_fixture(con, "m1", "2026-2027", 3, "team_a", "team_b")

    ep_lambdas = ep._fixture_lambdas(con, "team_a", "m1", ts_mv)
    fs_lambdas = fs._team_fixture_lambdas(con, "team_a", "m1", ts_mv)

    assert fs_lambdas[0] == ep_lambdas[0]  # lambda_for
    assert fs_lambdas[1] == ep_lambdas[1]  # lambda_against
    assert fs_lambdas[2] == ep_lambdas[2]  # is_home


def test_fixture_difficulty_sign_matches_intuition(con):
    """A strong home team (high attack, low defence-conceded) against a weak away team
    should show NEGATIVE difficulty (easier than average) -- lambda_for > lambda_against."""
    ts_mv = _seed_teams_and_strength(con, [("strong", 0.8, -0.3), ("weak", -0.6, 0.5)])
    _insert_fixture(con, "m1", "2026-2027", 1, "strong", "weak")

    d = fs.fixture_difficulty_for_match(con, "strong", "m1", 1, ts_mv)
    assert d is not None
    assert d.difficulty < 0  # easier than a neutral fixture
    assert d.is_home is True
    assert d.opponent_uid == "weak"

    d_weak = fs.fixture_difficulty_for_match(con, "weak", "m1", 1, ts_mv)
    assert d_weak.difficulty > 0  # the away weak team faces the harder fixture


def test_fixture_difficulty_for_unscheduled_match_returns_none(con):
    ts_mv = _seed_teams_and_strength(con, [("team_a", 0.1, 0.0), ("team_b", -0.1, 0.1)])
    assert fs.fixture_difficulty_for_match(con, "team_a", "does-not-exist", 1, ts_mv) is None


# ============================================================
# rolling swing score -- direction must flip at the right point
# ============================================================

def test_swing_score_detects_run_getting_harder_near_term(con):
    """team_x faces 3 hard fixtures (short window) then, looking further out (long window),
    3 easy ones dilute the average -- so the near-term run is HARDER than the fuller
    window, swing_score should be positive."""
    ts_mv = _seed_teams_and_strength(con, [
        ("team_x", 0.0, 0.0), ("hard1", 0.9, -0.4), ("hard2", 0.9, -0.4), ("hard3", 0.9, -0.4),
        ("easy1", -0.9, 0.4), ("easy2", -0.9, 0.4), ("easy3", -0.9, 0.4),
    ])
    for i, opp in enumerate(["hard1", "hard2", "hard3", "easy1", "easy2", "easy3"], start=1):
        _insert_fixture(con, f"m{i}", "2026-2027", i, "team_x", opp)

    score = fs.rolling_swing_score(con, "team_x", "2026-2027", as_of_gameweek=1, ts_model_version=ts_mv,
                                    short_window=3, long_window=6)
    assert score.n_short_fixtures == 3
    assert score.n_long_fixtures == 6
    assert score.short_avg_difficulty > score.long_avg_difficulty
    assert score.swing_score > 0  # near-term (hard-heavy) run is harder than the full window


def test_swing_score_detects_run_getting_easier_near_term(con):
    """Same shape, reversed: 3 easy fixtures first, then 3 hard ones further out -- near-term
    should be EASIER than the fuller window, swing_score negative."""
    ts_mv = _seed_teams_and_strength(con, [
        ("team_y", 0.0, 0.0), ("easy1", -0.9, 0.4), ("easy2", -0.9, 0.4), ("easy3", -0.9, 0.4),
        ("hard1", 0.9, -0.4), ("hard2", 0.9, -0.4), ("hard3", 0.9, -0.4),
    ])
    for i, opp in enumerate(["easy1", "easy2", "easy3", "hard1", "hard2", "hard3"], start=1):
        _insert_fixture(con, f"m{i}", "2026-2027", i, "team_y", opp)

    score = fs.rolling_swing_score(con, "team_y", "2026-2027", as_of_gameweek=1, ts_model_version=ts_mv,
                                    short_window=3, long_window=6)
    assert score.swing_score < 0


def test_swing_score_none_when_no_fixtures_scheduled(con):
    ts_mv = _seed_teams_and_strength(con, [("lonely", 0.0, 0.0)])
    score = fs.rolling_swing_score(con, "lonely", "2026-2027", as_of_gameweek=1, ts_model_version=ts_mv)
    assert score.short_avg_difficulty is None
    assert score.long_avg_difficulty is None
    assert score.swing_score is None
    assert score.n_short_fixtures == 0


def test_swing_score_handles_double_and_blank_gameweeks(con):
    """A double gameweek (2 fixtures in one gw) contributes 2 rows, not 1 -- neither
    collapsed away nor double-averaged incorrectly."""
    ts_mv = _seed_teams_and_strength(con, [("team_z", 0.0, 0.0), ("opp1", 0.0, 0.0), ("opp2", 0.0, 0.0)])
    _insert_fixture(con, "m1", "2026-2027", 1, "team_z", "opp1")
    _insert_fixture(con, "m2", "2026-2027", 1, "team_z", "opp2")
    # gw2 is a blank -- no fixture inserted for team_z there.

    score = fs.rolling_swing_score(con, "team_z", "2026-2027", as_of_gameweek=1, ts_model_version=ts_mv,
                                    short_window=2, long_window=2)
    assert score.n_short_fixtures == 2


def test_rolling_swing_score_rejects_long_window_shorter_than_short_window(con):
    ts_mv = _seed_teams_and_strength(con, [("team_a", 0.0, 0.0)])
    with pytest.raises(ValueError):
        fs.rolling_swing_score(con, "team_a", "2026-2027", as_of_gameweek=1, ts_model_version=ts_mv,
                                short_window=6, long_window=3)


def test_swing_scores_by_team_covers_every_team_in_the_model_version(con):
    ts_mv = _seed_teams_and_strength(con, [("team_a", 0.1, -0.1), ("team_b", -0.1, 0.1)])
    _insert_fixture(con, "m1", "2026-2027", 1, "team_a", "team_b")

    scores = fs.swing_scores_by_team(con, "2026-2027", as_of_gameweek=1, ts_model_version=ts_mv)
    assert set(scores) == {"team_a", "team_b"}
    assert scores["team_a"].n_short_fixtures == 1
