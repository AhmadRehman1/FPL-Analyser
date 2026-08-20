"""Regression test for a real bug in reconcile.build_dim_team(): team_uid was derived purely
from er.team_uid_for(name) per season, with no cross-season identity beyond that. Real case
this hit ingesting FPL-Core-Insights: Ipswich Town appears as "Ipswich" in 2024-2025's
teams.csv and "Ipswich Town" in 2026-2027's, but both rows carry the same FPL-permanent
numeric `code` (40). Name-only derivation silently split one real club into two distinct
dim_team rows with no shared history -- team_strength.calibrate() then genuinely could not
find any Elo or MLE-fit signal for the 2026-2027 identity, since all of its own 2024-2025
match history lived under the other, orphaned uid. Fixed by keying team_uid on code (FPL's own
stable identifier) once a code has been seen, reusing the first uid minted for it -- code is
already the documented join key for matches.csv (see README's design notes), just not
previously used for identity derivation itself.
"""

from pathlib import Path

from fpl_quant import entity_resolution as er
from fpl_quant import ingest_csv, reconcile

TEAMS_HEADER = "code,id,name,short_name"


def _write_csv(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([TEAMS_HEADER, *rows]) + "\n", encoding="utf-8")


def test_build_dim_team_unifies_same_code_across_a_season_name_change(con, tmp_path):
    _write_csv(tmp_path / "2024-2025" / "teams.csv", ["40,10,Ipswich,IPS"])
    _write_csv(tmp_path / "2026-2027" / "teams.csv", ["40,12,Ipswich Town,IPS"])
    for season in ("2024-2025", "2026-2027"):
        ingest_csv.ingest_csv_file(con, season, "teams.csv", tmp_path / season / "teams.csv")

    reconcile.build_dim_team(con)

    rows = con.execute("SELECT team_uid, canonical_name FROM dim_team").fetchall()
    assert len(rows) == 1, f"expected one unified team, got {rows}"
    uid = rows[0][0]

    aliases = dict(con.execute("SELECT season, alias_name FROM team_alias WHERE team_uid = ?", [uid]).fetchall())
    assert aliases == {"2024-2025": "Ipswich", "2026-2027": "Ipswich Town"}


def test_build_dim_team_still_separates_genuinely_different_codes(con, tmp_path):
    _write_csv(tmp_path / "2024-2025" / "teams.csv", ["3,1,Arsenal,ARS", "43,11,Man City,MCI"])
    ingest_csv.ingest_csv_file(con, "2024-2025", "teams.csv", tmp_path / "2024-2025" / "teams.csv")

    reconcile.build_dim_team(con)

    rows = con.execute("SELECT canonical_name FROM dim_team ORDER BY canonical_name").fetchall()
    assert [r[0] for r in rows] == ["Arsenal", "Man City"]


def test_team_uid_for_is_unchanged_for_a_single_name(con):
    assert er.team_uid_for("Arsenal") == "team_arsenal"
