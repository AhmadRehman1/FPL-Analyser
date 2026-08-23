from datetime import date

import pytest

from fpl_quant import expected_points as ep
from fpl_quant import field_covariance as fc
from fpl_quant import uncertainty as un


def _seed_pool(con):
    """Two fixtures in the same gameweek: match_a (2 players) and match_b (2 players), all
    with real ep_goals/ep_assists so compute_lambda_representative has real data to average,
    and real ep_outputs.fixture_match_id rows so compute_field_covariance can resolve each
    candidate to their fixture. Returns (players, ep_model_version)."""
    ep.seed_v1_params(con)
    un.seed_v1_params(con)

    con.execute(
        "INSERT INTO dim_team (team_uid, canonical_name) VALUES "
        "('t1','T1'), ('t2','T2'), ('t3','T3'), ('t4','T4')"
    )
    con.execute(
        "INSERT INTO fact_match (match_id, season, gameweek, home_team_uid, away_team_uid, finished, "
        "competition, kickoff_time, _ingested_at) VALUES "
        "('match_a', '2026-2027', 3, 't1', 't2', FALSE, 'Premier League', '2026-09-01', current_timestamp), "
        "('match_b', '2026-2027', 3, 't3', 't4', FALSE, 'Premier League', '2026-09-01', current_timestamp)"
    )

    con.execute(
        "INSERT INTO team_strength_model_versions (calibration_asof_date, home_advantage, xi_params_version, "
        "rho_params_version, reference_team_uid) VALUES ('2026-08-10', 0.2, 1, 1, 't1')"
    )
    ts_mv = con.execute("SELECT max(model_version) FROM team_strength_model_versions").fetchone()[0]
    con.execute(
        "INSERT INTO minutes_model_versions (calibration_asof_date, target_season, decay_params_version, "
        "adjustment_params_version, shrinkage_params_version, fact_multiplier_params_version, lookback_seasons) "
        "VALUES ('2026-08-10', '2026-2027', 1, 1, 1, 1, '[]')"
    )
    mm_mv = con.execute("SELECT max(model_version) FROM minutes_model_versions").fetchone()[0]
    ep_model_version = con.execute(
        "INSERT INTO ep_model_versions (calibration_asof_date, target_season, team_strength_model_version, "
        "minutes_model_version, scoring_matrix_params_version, bps_params_version, bps_tau_params_version) "
        "VALUES ('2026-08-10', '2026-2027', ?, ?, 1, 1, 1) RETURNING model_version", [ts_mv, mm_mv],
    ).fetchone()[0]

    players = {
        # uid: (position, mu, ep_goals, ep_assists, fixture)
        "high_a": ("Forward", 8.0, 0.5, 0.2, "match_a"),
        "low_a": ("Defender", 3.0, 0.05, 0.05, "match_a"),
        "high_b": ("Midfielder", 6.0, 0.3, 0.3, "match_b"),
        "low_b": ("Goalkeeper", 2.0, 0.0, 0.0, "match_b"),
    }
    for uid, (position, mu, eg, ea, fixture) in players.items():
        con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, ?)", [uid, uid, position])
        con.execute(
            "INSERT INTO ep_outputs (model_version, player_uid, fixture_match_id, ep_appearance, ep_goals, "
            "ep_assists, ep_clean_sheet, ep_goals_conceded, ep_defcon, ep_bonus, ep_saves, ep_penalty_save, "
            "ep_cards, ep_own_goal, ep_total, expected_bps) "
            "VALUES (?, ?, ?, 0, ?, ?, 0,0,0,0,0,0,0,0, ?, 5.0)",
            [ep_model_version, uid, fixture, eg, ea, mu],
        )
    return players, ep_model_version


def _candidates_from(players):
    return [
        {"player_uid": uid, "position": pos, "mu": mu}
        for uid, (pos, mu, _eg, _ea, _fx) in players.items()
    ]


def test_compute_field_covariance_zero_for_unowned_candidate(con):
    """A candidate with eo=0 (nobody owns them) contributes nothing to the field portfolio,
    so their own realized points can still vary freely without correlating with a field that
    doesn't include them -- Cov should land at (or extremely near) zero when NO candidate has
    nonzero EO, since the field portfolio itself is then identically zero."""
    players, ep_model_version = _seed_pool(con)
    candidates = _candidates_from(players)
    eo_by_uid = {uid: 0.0 for uid in players}

    cov = fc.compute_field_covariance(
        con, candidates, target_season="2026-2027", target_gameweek=3, ep_model_version=ep_model_version,
        scoring_params_version=1, rho_residual_params_version=1, eo_by_uid=eo_by_uid,
        calibration_asof_date=date(2026, 8, 10), n_antithetic_pairs=500,
    )
    for uid in players:
        assert cov[uid] == pytest.approx(0.0)


def test_compute_field_covariance_positive_for_a_fully_owned_fixture_pair(con):
    """high_a and low_a share match_a; if BOTH are effectively fully owned (eo=100), the
    field portfolio's match_a contribution is entirely made of players sharing high_a's own
    Z_fixture draw, so high_a's own points must be POSITIVELY covariant with the field."""
    players, ep_model_version = _seed_pool(con)
    candidates = _candidates_from(players)
    eo_by_uid = {"high_a": 100.0, "low_a": 100.0, "high_b": 0.0, "low_b": 0.0}

    cov = fc.compute_field_covariance(
        con, candidates, target_season="2026-2027", target_gameweek=3, ep_model_version=ep_model_version,
        scoring_params_version=1, rho_residual_params_version=1, eo_by_uid=eo_by_uid,
        calibration_asof_date=date(2026, 8, 10), n_antithetic_pairs=20000,
    )
    assert cov["high_a"] > 0
    assert cov["low_a"] > 0
    # high_b/low_b share a DIFFERENT fixture (match_b) with zero EO weight in the field --
    # their own Z_fixture draw is independent of match_a's, so the TRUE covariance with a
    # field made entirely of match_a players is zero. A finite Monte Carlo sample of two
    # independent quantities never lands at EXACTLY zero sample covariance though (that's
    # real, expected estimation noise, not a bug) -- asserted here as "much smaller than the
    # real same-fixture signal," not "exactly zero," with a large sample size to keep that
    # noise floor low relative to the real signal.
    assert abs(cov["high_b"]) < 0.05 * cov["high_a"]
    assert abs(cov["low_b"]) < 0.05 * cov["low_a"]


def test_compute_field_covariance_reproducible_across_repeated_calls(con):
    """Same determinism discipline as squad_optimizer.solve() (Priority 0): identical inputs
    must produce identical output across repeated calls, since this feeds the MIQP as a
    precomputed coefficient and any run-to-run drift here would reintroduce exactly the kind
    of nondeterminism Priority 0 root-caused and fixed elsewhere."""
    players, ep_model_version = _seed_pool(con)
    candidates = _candidates_from(players)
    eo_by_uid = {"high_a": 60.0, "low_a": 20.0, "high_b": 40.0, "low_b": 5.0}

    kwargs = dict(
        target_season="2026-2027", target_gameweek=3, ep_model_version=ep_model_version,
        scoring_params_version=1, rho_residual_params_version=1, eo_by_uid=eo_by_uid,
        calibration_asof_date=date(2026, 8, 10), n_antithetic_pairs=500,
    )
    first = fc.compute_field_covariance(con, candidates, **kwargs)
    for _ in range(3):
        again = fc.compute_field_covariance(con, candidates, **kwargs)
        assert again == first


def test_compute_field_covariance_zero_for_candidate_with_no_fixture_this_gameweek(con):
    """A blank-GW candidate (no row in fact_match for this gameweek) contributes nothing and
    gets Cov=0 -- a legitimate exclusion, not an error, matching monte_carlo.run()'s own
    stance on the same situation."""
    players, ep_model_version = _seed_pool(con)
    candidates = _candidates_from(players) + [{"player_uid": "blank_gw_player", "position": "Forward", "mu": 5.0}]
    eo_by_uid = {"high_a": 60.0, "low_a": 20.0, "high_b": 40.0, "low_b": 5.0, "blank_gw_player": 30.0}

    cov = fc.compute_field_covariance(
        con, candidates, target_season="2026-2027", target_gameweek=3, ep_model_version=ep_model_version,
        scoring_params_version=1, rho_residual_params_version=1, eo_by_uid=eo_by_uid,
        calibration_asof_date=date(2026, 8, 10), n_antithetic_pairs=500,
    )
    assert cov["blank_gw_player"] == pytest.approx(0.0)
