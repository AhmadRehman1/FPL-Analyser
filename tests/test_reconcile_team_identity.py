"""Regression: FPL-Core-Insights spells the same club differently across seasons' teams.csv
("Ipswich" in 2024-25, "Ipswich Town" in 2026-27 -- same FPL team code 40). Keyed by name
alone those split into two team_uids, so the 2024-25 history never attaches to the 2026-27
team and team_strength.calibrate() falls back to a league-average forecast for a club it
genuinely has (weak) data for. build_dim_team() now keys cross-season identity on the stable
FPL team `code`.
"""

from pathlib import Path

from fpl_quant import ingest_csv, reconcile

TEAMS_HEADER = "code,id,name,short_name"
MATCHES_HEADER = "match_id,gameweek,kickoff_time,home_team,away_team,home_score,away_score,home_team_elo,away_team_elo,finished"


def _write(path: Path, header: str, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")


def _ingest_teams(con, tmp_path, season, rows):
    p = tmp_path / season / "teams.csv"
    _write(p, TEAMS_HEADER, rows)
    ingest_csv.ingest_csv_file(con, season, "teams.csv", p)


def test_same_club_different_spelling_same_code_unifies_to_one_team_uid(con, tmp_path):
    _ingest_teams(con, tmp_path, "2024-2025", ["3,1,Arsenal,ARS", "40,10,Ipswich,IPS"])
    _ingest_teams(con, tmp_path, "2026-2027", ["3,1,Arsenal,ARS", "40,12,Ipswich Town,IPS"])

    reconcile.build_dim_team(con)

    # code 40 resolves to a SINGLE team_uid across both seasons
    uids = {r[0] for r in con.execute("SELECT DISTINCT team_uid FROM _team_code_map WHERE code = '40'").fetchall()}
    assert len(uids) == 1, f"code 40 split into {uids}"
    ipswich_uid = uids.pop()

    # dim_team has one row for it (not two)
    assert con.execute("SELECT count(*) FROM dim_team WHERE team_uid = ?", [ipswich_uid]).fetchone()[0] == 1
    assert con.execute("SELECT count(*) FROM dim_team WHERE canonical_name LIKE 'Ipswich%'").fetchone()[0] == 1

    # both season spellings are recorded as aliases pointing at the one uid
    alias_rows = dict(con.execute(
        "SELECT alias_name, team_uid FROM team_alias WHERE alias_name LIKE 'Ipswich%'"
    ).fetchall())
    assert alias_rows == {"Ipswich": ipswich_uid, "Ipswich Town": ipswich_uid}


def test_genuinely_new_club_gets_its_own_team_uid(con, tmp_path):
    _ingest_teams(con, tmp_path, "2024-2025", ["3,1,Arsenal,ARS"])
    _ingest_teams(con, tmp_path, "2026-2027", ["3,1,Arsenal,ARS", "88,11,Hull City,HUL"])

    reconcile.build_dim_team(con)

    hull = con.execute("SELECT DISTINCT team_uid FROM _team_code_map WHERE code = '88'").fetchall()
    assert len(hull) == 1 and hull[0][0] == reconcile.er.team_uid_for("Hull City")
    # Arsenal (code 3, both seasons) is still a single uid
    ars = con.execute("SELECT DISTINCT team_uid FROM _team_code_map WHERE code = '3'").fetchall()
    assert len(ars) == 1


def test_fact_match_attributes_both_seasons_ipswich_fixtures_to_one_uid(con, tmp_path):
    _ingest_teams(con, tmp_path, "2024-2025", ["3,1,Arsenal,ARS", "40,10,Ipswich,IPS"])
    _ingest_teams(con, tmp_path, "2026-2027", ["3,1,Arsenal,ARS", "40,12,Ipswich Town,IPS"])
    # matches.csv references team by CODE (verified design note in reconcile.py)
    m24 = tmp_path / "2024-2025" / "matches" / "GW1" / "matches.csv"
    _write(m24, MATCHES_HEADER, ["m_2425_1,1,2024-08-17T14:00:00Z,3,40,2,0,1800,1500,true"])
    m26 = tmp_path / "2026-2027" / "By Tournament" / "Premier League" / "GW1" / "matches.csv"
    _write(m26, MATCHES_HEADER, ["m_2627_1,1,2026-08-15T14:00:00Z,40,3,1,1,1520,1810,true"])
    ingest_csv.ingest_csv_file(con, "2024-2025", "matches/GW1/matches.csv", m24)
    ingest_csv.ingest_csv_file(con, "2026-2027", "By Tournament/Premier League/GW1/matches.csv", m26)

    reconcile.build_dim_team(con)
    reconcile.build_fact_match(con)

    ipswich_uid = con.execute("SELECT team_uid FROM _team_code_map WHERE code = '40' LIMIT 1").fetchone()[0]
    involved = con.execute(
        "SELECT season, home_team_uid, away_team_uid FROM fact_match ORDER BY season"
    ).fetchall()
    assert involved == [
        ("2024-2025", reconcile.er.team_uid_for("Arsenal"), ipswich_uid),
        ("2026-2027", ipswich_uid, reconcile.er.team_uid_for("Arsenal")),
    ]
