import json
from datetime import date, datetime, timezone

import pandas as pd
import pytest

from fpl_quant import params, team_strength as ts


def test_tau_formula_known_cells():
    rho = -0.13
    lam_h, lam_a = 1.4, 1.1
    assert ts.tau(0, 0, lam_h, lam_a, rho) == pytest.approx(1 - lam_h * lam_a * rho)
    assert ts.tau(0, 1, lam_h, lam_a, rho) == pytest.approx(1 + lam_h * rho)
    assert ts.tau(1, 0, lam_h, lam_a, rho) == pytest.approx(1 + lam_a * rho)
    assert ts.tau(1, 1, lam_h, lam_a, rho) == pytest.approx(1 - rho)


def test_tau_is_1_outside_low_score_cells():
    assert ts.tau(2, 0, 1.4, 1.1, -0.13) == 1.0
    assert ts.tau(3, 3, 1.4, 1.1, -0.13) == 1.0


def _seed_teams(con, names):
    uids = {}
    for name in names:
        uid = f"team_{name.lower()}"
        con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES (?, ?)", [uid, name])
        uids[name] = uid
    return uids


def _round_robin_matches(uids, results):
    """results: list of (home, away, home_goals, away_goals)."""
    rows = []
    for i, (h, a, hg, ag) in enumerate(results):
        rows.append({
            "match_id": f"m{i}", "season": "2024-2025", "home_team_uid": uids[h],
            "away_team_uid": uids[a], "home_score": hg, "away_score": ag,
            "kickoff_time": pd.Timestamp("2025-01-01") + pd.Timedelta(days=i),
        })
    return pd.DataFrame(rows)


def test_fit_dixon_coles_recovers_known_strength_ordering():
    uids = {"A": "team_a", "B": "team_b", "C": "team_c", "D": "team_d"}
    # A is a strong side (scores a lot, concedes little); D is weak (reverse).
    results = [
        ("A", "B", 3, 0), ("B", "A", 0, 2),
        ("A", "C", 4, 1), ("C", "A", 0, 3),
        ("A", "D", 5, 0), ("D", "A", 0, 4),
        ("B", "C", 1, 1), ("C", "B", 1, 1),
        ("B", "D", 2, 0), ("D", "B", 0, 2),
        ("C", "D", 2, 0), ("D", "C", 0, 2),
    ]
    matches = _round_robin_matches(uids, results)
    attack, defence, home_adv, _ = ts.fit_dixon_coles(matches, xi=0.0018, rho=-0.13, asof_date=date(2025, 6, 1), reference_team_uid=uids["A"])

    assert attack[uids["A"]] > attack[uids["B"]] > attack[uids["D"]]
    assert defence[uids["A"]] > defence[uids["D"]]  # A concedes far less -> higher defence value


def test_fit_dixon_coles_zero_centered_attack_mean():
    uids = {"A": "team_a", "B": "team_b"}
    results = [("A", "B", 2, 1), ("B", "A", 1, 1), ("A", "B", 3, 0), ("B", "A", 0, 2)]
    matches = _round_robin_matches(uids, results)
    attack, _defence, _ha, _ = ts.fit_dixon_coles(matches, xi=0.0018, rho=-0.13, asof_date=date(2025, 6, 1), reference_team_uid=uids["A"])
    mean_attack = sum(attack.values()) / len(attack)
    assert abs(mean_attack) < 1e-6


def test_fit_dixon_coles_warns_on_non_convergence(monkeypatch, capsys):
    uids = {"A": "team_a", "B": "team_b"}
    results = [("A", "B", 2, 1), ("B", "A", 1, 1)]
    matches = _round_robin_matches(uids, results)

    class FakeResult:
        def __init__(self, x):
            self.x = x
            self.success = False
            self.message = "fake non-convergence for test"
            self.nit = 0

    monkeypatch.setattr(ts, "minimize", lambda fn, x0, method: FakeResult(x0))
    ts.fit_dixon_coles(matches, xi=0.0018, rho=-0.13, asof_date=date(2025, 6, 1), reference_team_uid=uids["A"])
    out = capsys.readouterr().out
    assert "::warning::" in out
    assert "did not converge" in out


def test_fit_dixon_coles_warns_on_tau_floor_clipping(monkeypatch, capsys):
    """tau(x,y;rho) going <=0 and hitting the 1e-10 floor means (xi,rho) is in an invalid-tau
    region for the data -- forced here via a monkeypatched tau() rather than hunting for real
    (xi,rho,data) that happens to trigger it, since the floor-clipping behavior being tested is
    orthogonal to which real inputs would produce it."""
    uids = {"A": "team_a", "B": "team_b"}
    results = [("A", "B", 0, 0), ("B", "A", 1, 1)]
    matches = _round_robin_matches(uids, results)
    monkeypatch.setattr(ts, "tau", lambda x, y, lh, la, rho: -5.0)
    ts.fit_dixon_coles(matches, xi=0.0018, rho=-0.13, asof_date=date(2025, 6, 1), reference_team_uid=uids["A"])
    out = capsys.readouterr().out
    assert "::warning::" in out
    assert "floor" in out.lower()


def test_fit_dixon_coles_no_warning_on_clean_fit(capsys):
    uids = {"A": "team_a", "B": "team_b", "C": "team_c"}
    results = [("A", "B", 2, 1), ("B", "A", 1, 1), ("A", "C", 3, 0), ("C", "A", 0, 2), ("B", "C", 1, 1)]
    matches = _round_robin_matches(uids, results)
    ts.fit_dixon_coles(matches, xi=0.0018, rho=-0.13, asof_date=date(2025, 6, 1), reference_team_uid=uids["A"])
    assert "::warning::" not in capsys.readouterr().out


def test_compute_seasons_of_topflight_data(con):
    uids = _seed_teams(con, ["A", "B", "X", "C"])
    now = datetime.now(timezone.utc)
    # A: played in both prior seasons. B: only 2025-2026 (e.g. promoted mid-window,
    # facing stand-in opponent X in 2024-2025 instead). C: neither (a fresh promotion).
    con.execute(
        "INSERT INTO fact_match (match_id, season, home_team_uid, away_team_uid, finished, competition, _ingested_at) "
        "VALUES ('m1', '2024-2025', ?, ?, TRUE, 'Premier League', ?)", [uids["A"], uids["X"], now]
    )
    con.execute(
        "INSERT INTO fact_match (match_id, season, home_team_uid, away_team_uid, finished, competition, _ingested_at) "
        "VALUES ('m2', '2025-2026', ?, ?, TRUE, 'Premier League', ?)", [uids["A"], uids["B"], now]
    )
    result = ts.compute_seasons_of_topflight_data(con, list(uids.values()), ("2024-2025", "2025-2026"))
    assert result[uids["A"]] == 2
    assert result[uids["B"]] == 1
    assert result[uids["C"]] == 0


def test_elo_regression_positive_slope():
    attack_mle = {"t1": 0.5, "t2": 0.1, "t3": -0.3, "t4": -0.5}
    defence_mle = {"t1": 0.4, "t2": 0.0, "t3": -0.2, "t4": -0.4}
    elo = {"t1": 2000, "t2": 1900, "t3": 1800, "t4": 1700}
    a0, a1, b0, b1, n = ts.fit_elo_regression(attack_mle, defence_mle, elo, list(elo.keys()))
    assert n == 4
    assert a1 > 0  # higher Elo -> higher attack
    assert b1 > 0  # higher Elo -> higher (better) defence


def test_elo_regression_raises_with_insufficient_teams():
    with pytest.raises(ValueError):
        ts.fit_elo_regression({"t1": 0.1}, {"t1": 0.1}, {"t1": 2000}, ["t1"])


def test_calibrate_end_to_end_promoted_team_gets_pure_elo_prior(con, tmp_path, monkeypatch):
    params.write_param(con, "model_decay_params", 1, "2026-08-10", "xi", value_numeric=0.0018)
    params.write_param(con, "model_decay_params", 1, "2026-08-10", "rho", value_numeric=-0.13)

    uids = _seed_teams(con, ["A", "B", "C", "Promoted"])
    now = datetime.now(timezone.utc)
    results = [
        ("A", "B", 2, 0), ("B", "A", 0, 1), ("A", "C", 3, 1), ("C", "A", 0, 2),
        ("B", "C", 1, 1), ("C", "B", 1, 0),
    ]
    for i, (h, a, hg, ag) in enumerate(results):
        for season in ("2024-2025", "2025-2026"):
            con.execute(
                "INSERT INTO fact_match (match_id, season, home_team_uid, away_team_uid, home_score, "
                "away_score, finished, competition, kickoff_time, _ingested_at) "
                "VALUES (?, ?, ?, ?, ?, ?, TRUE, 'Premier League', ?, ?)",
                [f"m{season}_{i}", season, uids[h], uids[a], hg, ag,
                 datetime(2025 if season == "2024-2025" else 2026, 1, 1 + i), now],
            )
    # 2026-2027: same 3 established teams plus the new promotion
    for i, (h, a) in enumerate([("A", "B"), ("C", "Promoted")]):
        con.execute(
            "INSERT INTO fact_match (match_id, season, home_team_uid, away_team_uid, finished, "
            "competition, _ingested_at) VALUES (?, '2026-2027', ?, ?, FALSE, 'Premier League', ?)",
            [f"m2026_{i}", uids[h], uids[a], now],
        )

    monkeypatch.setattr(
        ts, "fetch_current_elo",
        lambda con, season: {uids["A"]: 2000, uids["B"]: 1900, uids["C"]: 1850, uids["Promoted"]: 1500},
    )

    model_version = ts.calibrate(con, date(2026, 8, 10), xi_params_version=1, rho_params_version=1)
    snap = con.execute(
        "SELECT team_uid, attack_mle, final_attack, seasons_of_topflight_data, weight_own_data "
        "FROM team_strength_snapshots WHERE model_version = ?", [model_version],
    ).fetchdf().set_index("team_uid")

    promoted_row = snap.loc[uids["Promoted"]]
    assert pd.isna(promoted_row["attack_mle"])  # never played in our loaded history
    assert promoted_row["seasons_of_topflight_data"] == 0
    assert promoted_row["weight_own_data"] == 0.0

    established_row = snap.loc[uids["A"]]
    assert established_row["seasons_of_topflight_data"] == 2
    assert established_row["weight_own_data"] == pytest.approx(2 / 3)


def test_calibrate_falls_back_to_prior_season_elo_when_target_season_elo_all_blank(con, monkeypatch):
    """A real, live-CI-observed condition: FPL-Core-Insights' target-season teams.csv ships an
    `elo` column present but entirely blank early in a season -- fetch_current_elo() correctly
    returns {} for that, which must not permanently break the Elo regression."""
    params.write_param(con, "model_decay_params", 1, "2026-08-10", "xi", value_numeric=0.0018)
    params.write_param(con, "model_decay_params", 1, "2026-08-10", "rho", value_numeric=-0.13)

    uids = _seed_teams(con, ["A", "B", "C"])
    now = datetime.now(timezone.utc)
    results = [("A", "B", 2, 0), ("B", "A", 0, 1), ("A", "C", 3, 1), ("C", "A", 0, 2)]
    for i, (h, a, hg, ag) in enumerate(results):
        for season in ("2024-2025", "2025-2026"):
            con.execute(
                "INSERT INTO fact_match (match_id, season, home_team_uid, away_team_uid, home_score, "
                "away_score, finished, competition, kickoff_time, _ingested_at) "
                "VALUES (?, ?, ?, ?, ?, ?, TRUE, 'Premier League', ?, ?)",
                [f"m{season}_{i}", season, uids[h], uids[a], hg, ag,
                 datetime(2025 if season == "2024-2025" else 2026, 1, 1 + i), now],
            )
    for i, (h, a) in enumerate([("A", "B"), ("B", "C")]):
        con.execute(
            "INSERT INTO fact_match (match_id, season, home_team_uid, away_team_uid, finished, "
            "competition, _ingested_at) VALUES (?, '2026-2027', ?, ?, FALSE, 'Premier League', ?)",
            [f"m2026_{i}", uids[h], uids[a], now],
        )

    prior_season_elo = {uids["A"]: 2000, uids["B"]: 1900, uids["C"]: 1850}
    monkeypatch.setattr(
        ts, "fetch_current_elo",
        lambda con, season: {} if season == "2026-2027" else prior_season_elo,
    )

    model_version = ts.calibrate(con, date(2026, 8, 10), xi_params_version=1, rho_params_version=1)
    n_teams = con.execute(
        "SELECT count(*) FROM team_strength_snapshots WHERE model_version = ?", [model_version]
    ).fetchone()[0]
    assert n_teams == 3

    # backtest.fit_seasons_for("2026-2027") appends the target season itself, so fit_seasons[-1]
    # IS "2026-2027" -- the blank-Elo season this fallback exists to skip. The fallback must
    # still find a prior season's Elo, not stop at fit_seasons[-1] and crash (the failure that
    # killed chip_timing_analysis.yml run 33453667681 and forward_season_sim.yml run
    # 33432100674 against a freshly-ingested CI DB).
    model_version_3s = ts.calibrate(
        con, date(2026, 8, 10), xi_params_version=1, rho_params_version=1,
        target_season="2026-2027", fit_seasons=("2024-2025", "2025-2026", "2026-2027"),
    )
    assert con.execute(
        "SELECT count(*) FROM team_strength_snapshots WHERE model_version = ?", [model_version_3s]
    ).fetchone()[0] == 3


def test_calibrate_uses_league_average_when_team_has_no_mle_and_no_elo(con, monkeypatch, capsys):
    """The real root cause this project actually hit: a club spelled differently across seasons
    (e.g. "Ipswich" in 2024-25's source data vs "Ipswich Town" in 2026-27's) normalizes to two
    different team_uids, so the 2026-27 uid has neither an MLE fit (its 2024-25 history is
    attached to the OTHER uid) nor an Elo prior (also keyed by the wrong uid) -- must not crash
    the whole calibration over one team."""
    params.write_param(con, "model_decay_params", 1, "2026-08-10", "xi", value_numeric=0.0018)
    params.write_param(con, "model_decay_params", 1, "2026-08-10", "rho", value_numeric=-0.13)

    uids = _seed_teams(con, ["A", "B", "Renamed"])
    now = datetime.now(timezone.utc)
    results = [("A", "B", 2, 0), ("B", "A", 0, 1)]
    for i, (h, a, hg, ag) in enumerate(results):
        for season in ("2024-2025", "2025-2026"):
            con.execute(
                "INSERT INTO fact_match (match_id, season, home_team_uid, away_team_uid, home_score, "
                "away_score, finished, competition, kickoff_time, _ingested_at) "
                "VALUES (?, ?, ?, ?, ?, ?, TRUE, 'Premier League', ?, ?)",
                [f"m{season}_{i}", season, uids[h], uids[a], hg, ag,
                 datetime(2025 if season == "2024-2025" else 2026, 1, 1 + i), now],
            )
    # "Renamed" only ever appears under this team_uid in the target season -- no fit_seasons
    # history, and (unlike test_calibrate_end_to_end_promoted_team_gets_pure_elo_prior) no Elo
    # for it either, in any season. A and B also need a real 2026-27 fixture each (not just
    # against Renamed) so they're both in target_teams/eligible_teams -- otherwise the Elo
    # regression itself can't fit (a different, already-covered failure mode).
    for i, (h, a) in enumerate([("A", "Renamed"), ("A", "B")]):
        con.execute(
            "INSERT INTO fact_match (match_id, season, home_team_uid, away_team_uid, finished, "
            "competition, _ingested_at) VALUES (?, '2026-2027', ?, ?, FALSE, 'Premier League', ?)",
            [f"m2026_{i}", uids[h], uids[a], now],
        )

    monkeypatch.setattr(
        ts, "fetch_current_elo",
        lambda con, season: {uids["A"]: 2000, uids["B"]: 1900},
    )

    model_version = ts.calibrate(con, date(2026, 8, 10), xi_params_version=1, rho_params_version=1)
    snap = con.execute(
        "SELECT team_uid, attack_mle, final_attack, final_defence "
        "FROM team_strength_snapshots WHERE model_version = ?", [model_version],
    ).fetchdf().set_index("team_uid")

    renamed_row = snap.loc[uids["Renamed"]]
    assert pd.isna(renamed_row["attack_mle"])  # confirms it genuinely has no MLE fit
    real_fits = snap.drop(uids["Renamed"])
    expected_fallback = real_fits["attack_mle"].mean()
    assert renamed_row["final_attack"] == pytest.approx(expected_fallback)
    assert "::warning::" in capsys.readouterr().out


# ============================================================
# team_strength_guard_params: docs/reports/2026-10_promoted_club_strength.md
# ============================================================

FIT_3 = ("2024-2025", "2025-2026", "2026-2027")
ELO = {"A": 2000.0, "B": 1900.0, "C": 1850.0}


def _insert_match(con, match_id, season, home, away, hg, ag, kickoff, *, finished=True, home_elo=None, away_elo=None):
    con.execute(
        "INSERT INTO fact_match (match_id, season, home_team_uid, away_team_uid, home_score, away_score, "
        "home_team_elo, away_team_elo, finished, competition, kickoff_time, _ingested_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'Premier League', ?, ?)",
        [match_id, season, home, away, hg, ag, home_elo, away_elo, finished, kickoff, datetime.now(timezone.utc)],
    )


def _seed_guard_league(con, *, newcomer_elo=None, newcomer_results=((0, 2), (0, 1), (0, 3))):
    """A, B, C play each other in 2024-25 and 2025-26 (with their Elo on every match); in
    2026-27 promoted N joins and loses every match without scoring."""
    params.write_param(con, "model_decay_params", 1, "2026-08-10", "xi", value_numeric=0.0018)
    params.write_param(con, "model_decay_params", 1, "2026-08-10", "rho", value_numeric=-0.13)
    uids = _seed_teams(con, ["A", "B", "C", "N"])
    results = [("A", "B", 2, 0), ("B", "A", 0, 1), ("A", "C", 3, 1), ("C", "A", 0, 2), ("B", "C", 1, 1), ("C", "B", 1, 0)]
    for season, year in (("2024-2025", 2025), ("2025-2026", 2026)):
        for i, (h, a, hg, ag) in enumerate(results):
            _insert_match(con, f"m{season}_{i}", season, uids[h], uids[a], hg, ag, datetime(year, 1, 1 + i),
                          home_elo=ELO[h], away_elo=ELO[a])
    for i, (h, a, hg, ag) in enumerate([("A", "B", 1, 1), ("B", "C", 2, 1), ("C", "A", 0, 1)]):
        _insert_match(con, f"m2026_{i}", "2026-2027", uids[h], uids[a], hg, ag, datetime(2026, 8, 20 + i),
                      home_elo=ELO[h], away_elo=ELO[a])
    for i, (opponent, (n_goals, opp_goals)) in enumerate(zip(("A", "B", "C"), newcomer_results)):
        _insert_match(con, f"m2026_n{i}", "2026-2027", uids["N"], uids[opponent], n_goals, opp_goals,
                      datetime(2026, 8, 24 + i), home_elo=newcomer_elo, away_elo=ELO[opponent])
    return uids


def _snapshot(con, model_version):
    return con.execute(
        "SELECT team_uid, attack_mle, defence_mle, attack_elo_prior, defence_elo_prior, final_attack, final_defence, "
        "seasons_of_topflight_data, weight_own_data, elo_at_calibration FROM team_strength_snapshots WHERE model_version = ?",
        [model_version],
    ).fetchdf().set_index("team_uid")


def _guard_version(con, arm):
    return params.get_or_create_bundle_version(con, ts.GUARD_FAMILY, ts.GUARD_ARMS[arm], "2026-10-05")


def test_point_in_time_elo_is_each_clubs_latest_finished_match(con):
    uids = _seed_teams(con, ["A", "B"])
    _insert_match(con, "old", "2025-2026", uids["A"], uids["B"], 1, 0, datetime(2026, 5, 1), home_elo=1900.0, away_elo=1700.0)
    _insert_match(con, "new", "2026-2027", uids["B"], uids["A"], 0, 0, datetime(2026, 8, 20), home_elo=1710.0)
    # the next fixture's Elo is not known yet: only finished matches count
    _insert_match(con, "next", "2026-2027", uids["A"], uids["B"], None, None, datetime(2026, 9, 1),
                  finished=False, home_elo=1950.0, away_elo=1750.0)
    assert ts.fetch_point_in_time_elo(con) == {uids["A"]: 1900.0, uids["B"]: 1710.0}


def test_unguarded_fit_lets_a_club_that_has_not_scored_run_off(con, monkeypatch):
    """The failure the guard exists for: with 2026-27 in the fit and no Elo, the newcomer's raw
    fit goes in unshrunk."""
    uids = _seed_guard_league(con)
    # as live: 2026-27's teams.csv Elo blank, 2025-26's lists only the established clubs
    monkeypatch.setattr(ts, "fetch_current_elo", lambda con, season: (
        {} if season == "2026-2027" else {uids[name]: elo for name, elo in ELO.items()}
    ))
    snap = _snapshot(con, ts.calibrate(con, date(2026, 9, 1), 1, 1, target_season="2026-2027", fit_seasons=FIT_3))
    # the fit centres attack on every club, so read strength against the established clubs
    centre = snap.loc[[uids["A"], uids["B"], uids["C"]], "attack_mle"].mean()
    assert snap.loc[uids["N"], "final_attack"] - centre < -3
    assert snap.loc[uids["A"], "weight_own_data"] == pytest.approx(1.0)  # the in-progress season counted


def test_guard_holds_a_first_season_club_near_its_prior(con, capsys):
    uids = _seed_guard_league(con)
    snap = _snapshot(con, ts.calibrate(con, date(2026, 9, 1), 1, 1, target_season="2026-2027", fit_seasons=FIT_3,
                                       guard_params_version=_guard_version(con, "fix")))
    newcomer = snap.loc[uids["N"]]
    established = snap.loc[[uids["A"], uids["B"], uids["C"]]]
    # no Elo anywhere for N: the promoted-club prior, the established mean plus the offsets
    assert pd.isna(newcomer["elo_at_calibration"])
    assert newcomer["attack_elo_prior"] == pytest.approx(established["attack_mle"].mean() - 0.39)
    assert newcomer["defence_elo_prior"] == pytest.approx(established["defence_mle"].mean() - 0.32)
    # three matches: weight 3 / 13, its runaway fit held 1.0 below the prior first
    assert newcomer["weight_own_data"] == pytest.approx(3 / 13)
    assert newcomer["final_attack"] == pytest.approx(newcomer["attack_elo_prior"] - 3 / 13)
    # nothing more than 1.5 from the established clubs (the fit centres attack on every club,
    # so a runaway club shifts all stored values alike; strengths are read against each other)
    for final, fitted in (("final_attack", "attack_mle"), ("final_defence", "defence_mle")):
        assert (snap[final] - established[fitted].mean()).abs().max() < 1.5
    # the established clubs keep 2/3 although 2026-27 is in the fit
    assert established["weight_own_data"].tolist() == pytest.approx([2 / 3] * 3)
    assert "promoted" in capsys.readouterr().out


def test_guard_uses_a_promoted_clubs_own_match_elo_unless_withheld(con):
    uids = _seed_guard_league(con, newcomer_elo=1700.0)
    fix = _snapshot(con, ts.calibrate(con, date(2026, 9, 1), 1, 1, target_season="2026-2027", fit_seasons=FIT_3,
                                      guard_params_version=_guard_version(con, "fix")))
    assert fix.loc[uids["N"], "elo_at_calibration"] == 1700.0
    withheld = _snapshot(con, ts.calibrate(con, date(2026, 9, 1), 1, 1, target_season="2026-2027", fit_seasons=FIT_3,
                                           guard_params_version=_guard_version(con, "fix-withheld")))
    assert pd.isna(withheld.loc[uids["N"], "elo_at_calibration"])
    # established clubs keep theirs either way
    assert withheld.loc[uids["A"], "elo_at_calibration"] == 2000.0


def test_guard_gives_an_unplayed_newcomer_the_promoted_club_prior(con):
    uids = _seed_guard_league(con, newcomer_results=())
    _insert_match(con, "n_next", "2026-2027", uids["N"], uids["A"], None, None, datetime(2026, 9, 5), finished=False)
    snap = _snapshot(con, ts.calibrate(con, date(2026, 9, 1), 1, 1, target_season="2026-2027", fit_seasons=FIT_3,
                                       guard_params_version=_guard_version(con, "fix")))
    newcomer = snap.loc[uids["N"]]
    assert pd.isna(newcomer["attack_mle"]) and newcomer["weight_own_data"] == 0.0
    assert newcomer["final_attack"] == pytest.approx(newcomer["attack_elo_prior"])
    assert newcomer["final_defence"] == pytest.approx(newcomer["defence_elo_prior"])


def test_live_like_fits_prior_seasons_and_leaves_a_promoted_club_at_league_average(con, capsys):
    """What live did before the fix: no 2026-27 match in the fit, no Elo for the promoted club."""
    uids = _seed_guard_league(con, newcomer_elo=1700.0)
    model_version = ts.calibrate(con, date(2026, 9, 1), 1, 1, target_season="2026-2027", fit_seasons=FIT_3,
                                 guard_params_version=_guard_version(con, "live-like"))
    seasons_fit = con.execute(
        "SELECT seasons_fit FROM team_strength_model_versions WHERE model_version = ?", [model_version],
    ).fetchone()[0]
    assert json.loads(seasons_fit) == ["2024-2025", "2025-2026"]
    snap = _snapshot(con, model_version)
    assert pd.isna(snap.loc[uids["N"], "attack_mle"]) and pd.isna(snap.loc[uids["N"], "elo_at_calibration"])
    assert "league-average" in capsys.readouterr().out


def test_guard_v1_is_the_recommended_blend_and_arms_reuse_their_versions(con):
    ts.seed_team_strength_guard_params(con)
    assert ts.resolve_guard_params(con, 1) == ts.GUARD_RECOMMENDED
    assert _guard_version(con, "fix") == 1
    honest = _guard_version(con, "honest")
    assert honest == 2 and _guard_version(con, "honest") == honest
    assert set(ts.GUARD_ARMS) == {"honest", "live-like", "fix", "fix-withheld"}
    for values in ts.GUARD_ARMS.values():
        assert set(values) == set(ts.GUARD_RECOMMENDED)
