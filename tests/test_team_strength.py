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


def _seed_raw_teams_table(con, season, rows):
    """rows: list of (name, elo). Mirrors ingest_csv.py's own raw-table naming/logging
    convention (fact_raw_ingestion_log.raw_table_name), the same mechanism
    reconcile._season_root_table() looks up -- not a shortcut around it."""
    table = f"raw_{season.replace('-', '_')}_teams"
    con.execute(f'CREATE TABLE "{table}" (name VARCHAR, elo VARCHAR)')
    for name, elo in rows:
        con.execute(f'INSERT INTO "{table}" VALUES (?, ?)', [name, elo])
    con.execute(
        "INSERT INTO fact_raw_ingestion_log (raw_table_name, season, source_relpath, source_file_hash, row_count) "
        "VALUES (?, ?, 'teams.csv', 'x', ?)",
        [table, season, len(rows)],
    )


def test_calibrate_gives_zero_signal_promoted_team_the_weakest_known_elo_instead_of_raising(con, tmp_path, monkeypatch):
    """Real case: Coventry City/Hull City for 2026-2027 have no MLE fit (never in fit_seasons'
    top flight) and no Elo anywhere in the loaded data (never PL-tracked). This used to raise
    and block calibration for every OTHER team too -- must now get a real, disclosed, weakest-
    known-elo-floor prior instead."""
    params.write_param(con, "model_decay_params", 1, "2026-08-10", "xi", value_numeric=0.0018)
    params.write_param(con, "model_decay_params", 1, "2026-08-10", "rho", value_numeric=-0.13)

    uids = _seed_teams(con, ["A", "B", "Promoted"])
    now = datetime.now(timezone.utc)
    results = [("A", "B", 2, 0), ("B", "A", 0, 1), ("A", "B", 3, 1), ("B", "A", 0, 2)]
    for i, (h, a, hg, ag) in enumerate(results):
        for season in ("2024-2025", "2025-2026"):
            con.execute(
                "INSERT INTO fact_match (match_id, season, home_team_uid, away_team_uid, home_score, "
                "away_score, finished, competition, kickoff_time, _ingested_at) "
                "VALUES (?, ?, ?, ?, ?, ?, TRUE, 'Premier League', ?, ?)",
                [f"m{season}_{i}", season, uids[h], uids[a], hg, ag,
                 datetime(2025 if season == "2024-2025" else 2026, 1, 1 + i), now],
            )
    # 2026-2027: A, B, and a genuinely brand-new promoted team with zero prior history
    for i, (h, a) in enumerate([("A", "B"), ("B", "Promoted")]):
        con.execute(
            "INSERT INTO fact_match (match_id, season, home_team_uid, away_team_uid, finished, "
            "competition, _ingested_at) VALUES (?, '2026-2027', ?, ?, FALSE, 'Premier League', ?)",
            [f"m2026_{i}", uids[h], uids[a], now],
        )

    monkeypatch.setattr(
        ts, "fetch_current_elo",
        lambda con, season, fallback_seasons=(): {uids["A"]: 2000, uids["B"]: 1700},
    )

    model_version = ts.calibrate(con, date(2026, 8, 10), xi_params_version=1, rho_params_version=1)
    row = con.execute(
        "SELECT elo_at_calibration, final_attack, final_defence FROM team_strength_snapshots "
        "WHERE model_version = ? AND team_uid = ?", [model_version, uids["Promoted"]]
    ).fetchone()
    assert row is not None, "promoted team must get a real snapshot row, not a raised exception"
    elo_used, final_attack, final_defence = row
    assert elo_used == 1700.0  # the weaker of A/B's two real Elo values, not invented out of thin air
    assert final_attack is not None and final_defence is not None


def test_fetch_current_elo_falls_back_to_prior_season_when_target_season_blank(con):
    uids = _seed_teams(con, ["A", "B"])
    # Pre-season: 2026-2027's own teams.csv ships with elo entirely blank (real
    # FPL-Core-Insights behavior before that season's own matches exist to derive it from).
    _seed_raw_teams_table(con, "2026-2027", [("A", ""), ("B", "")])
    _seed_raw_teams_table(con, "2025-2026", [("A", "2000"), ("B", "1800")])
    for season in ("2026-2027", "2025-2026"):
        con.execute("INSERT INTO team_alias (alias_name, season, team_uid) VALUES ('A', ?, ?)", [season, uids["A"]])
        con.execute("INSERT INTO team_alias (alias_name, season, team_uid) VALUES ('B', ?, ?)", [season, uids["B"]])

    out = ts.fetch_current_elo(con, "2026-2027", fallback_seasons=("2025-2026",))
    assert out == {uids["A"]: 2000.0, uids["B"]: 1800.0}


def test_fetch_current_elo_prefers_target_season_over_fallback_when_both_have_data(con):
    uids = _seed_teams(con, ["A"])
    _seed_raw_teams_table(con, "2026-2027", [("A", "2100")])
    _seed_raw_teams_table(con, "2025-2026", [("A", "2000")])
    for season in ("2026-2027", "2025-2026"):
        con.execute("INSERT INTO team_alias (alias_name, season, team_uid) VALUES ('A', ?, ?)", [season, uids["A"]])

    out = ts.fetch_current_elo(con, "2026-2027", fallback_seasons=("2025-2026",))
    assert out == {uids["A"]: 2100.0}


def test_fetch_current_elo_falls_back_per_team_not_per_season(con):
    """A team absent from the most-preferred fallback season (e.g. relegated after it) but
    present in an older one still needs ITS OWN most recent data -- real case this caught:
    Ipswich Town was in the 2024-2025 Premier League (elo=1589) but not 2025-2026 (relegated),
    while most other clubs' most recent data IS 2025-2026. A "first season with any data wins"
    fallback would use 2025-2026 for everyone and silently drop Ipswich entirely."""
    uids = _seed_teams(con, ["Stayed", "Ipswich"])
    _seed_raw_teams_table(con, "2026-2027", [("Stayed", ""), ("Ipswich", "")])
    _seed_raw_teams_table(con, "2025-2026", [("Stayed", "2000")])  # Ipswich not in the PL this season
    _seed_raw_teams_table(con, "2024-2025", [("Stayed", "1950"), ("Ipswich", "1589")])
    for season in ("2026-2027", "2025-2026", "2024-2025"):
        con.execute("INSERT INTO team_alias (alias_name, season, team_uid) VALUES ('Stayed', ?, ?)", [season, uids["Stayed"]])
    for season in ("2026-2027", "2024-2025"):
        con.execute("INSERT INTO team_alias (alias_name, season, team_uid) VALUES ('Ipswich', ?, ?)", [season, uids["Ipswich"]])

    out = ts.fetch_current_elo(con, "2026-2027", fallback_seasons=("2025-2026", "2024-2025"))
    assert out == {uids["Stayed"]: 2000.0, uids["Ipswich"]: 1589.0}


def test_fetch_current_elo_returns_empty_when_no_season_has_data(con):
    uids = _seed_teams(con, ["Promoted"])
    _seed_raw_teams_table(con, "2026-2027", [("Promoted", "")])
    con.execute(
        "INSERT INTO team_alias (alias_name, season, team_uid) VALUES ('Promoted', '2026-2027', ?)",
        [uids["Promoted"]],
    )

    assert ts.fetch_current_elo(con, "2026-2027", fallback_seasons=("2025-2026", "2024-2025")) == {}


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
        lambda con, season, fallback_seasons=(): {uids["A"]: 2000, uids["B"]: 1900, uids["C"]: 1850, uids["Promoted"]: 1500},
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
