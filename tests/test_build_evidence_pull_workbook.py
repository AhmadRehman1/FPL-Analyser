import sys
from pathlib import Path

import openpyxl

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_evidence_pull_workbook as bw  # noqa: E402

TEAMS = [{"id": 1, "name": "Coventry City"}, {"id": 2, "name": "Aston Villa"}, {"id": 3, "name": "Leeds"},
         {"id": 4, "name": "Brentford"}, {"id": 5, "name": "Man Utd"}, {"id": 6, "name": "Man City"}]
ELEMENTS = [
    {"first_name": "Haji", "second_name": "Wright", "web_name": "Wright", "team": 1},
    {"first_name": "James", "second_name": "Wright", "web_name": "Wright", "team": 2},
    {"first_name": "Harry", "second_name": "Wilson", "web_name": "Wilson", "team": 3},
    {"first_name": "Callum", "second_name": "Wilson", "web_name": "Wilson", "team": 4},
    {"first_name": "Dominic", "second_name": "Calvert-Lewin", "web_name": "Calvert-Lewin", "team": 3},
    {"first_name": "Amad", "second_name": "Diallo", "web_name": "Amad", "team": 5},
    {"first_name": "Phil", "second_name": "Foden", "web_name": "Foden", "team": 6},
]


def test_a_name_two_players_share_maps_to_neither():
    variants, ambiguous, _ = bw._name_map(ELEMENTS, TEAMS)
    assert "wright" not in variants and "wright" in ambiguous
    assert variants["haji wright"] == "Haji Wright"
    assert variants["calvert-lewin"] == "Dominic Calvert-Lewin"
    assert variants["amad"] == "Amad Diallo"


def test_same_club_reads_fpl_short_names_and_whole_words_only():
    assert bw._same_club("Leeds United", "Leeds")
    assert bw._same_club("Ipswich", "Ipswich Town")
    assert bw._same_club("Manchester United", "Man Utd")
    assert bw._same_club("Nottingham Forest", "Nott'm Forest")
    assert bw._same_club("Tottenham", "Spurs")
    assert not bw._same_club("Manchester City", "Man Utd")
    assert not bw._same_club("Leeds United", "Brentford")


def _set_pieces_md(tmp_path, rows: list[str]) -> Path:
    md = tmp_path / "pull.md"
    md.write_text(
        "## Table 5 — SetPieces\n\n"
        "| club | duty | primary_taker | secondary_taker | source_name | confidence_1_10 | observed_date |\n"
        "|---|---|---|---|---|---|---|\n" + "".join(r + "\n" for r in rows),
        encoding="utf-8",
    )
    return md


def test_build_resolves_a_shared_bare_name_by_the_rows_club(tmp_path, monkeypatch):
    monkeypatch.setattr(bw, "_fetch_fpl_name_map", lambda: bw._name_map(ELEMENTS, TEAMS))
    md = _set_pieces_md(tmp_path, [
        "| Coventry City | Penalties | Wright | | fplcopilot | 5 | 2026-08-06 |",
        "| Leeds United | Corners | Wilson | Calvert-Lewin | FootballDream | 4 | 2026-08-27 |",
        "| Arsenal | Penalties | Wilson | | Somebody | 3 | 2026-08-27 |",
    ])
    report = bw.build(md, tmp_path / "out.xlsx")
    sheet = openpyxl.load_workbook(tmp_path / "out.xlsx")["SetPieces"]
    takers = [(row[2], row[3]) for row in sheet.iter_rows(min_row=2, values_only=True)]
    assert takers == [("Haji Wright", None), ("Harry Wilson", "Dominic Calvert-Lewin"), ("Wilson", None)]
    assert report["unresolved_single_names"] == ["Wilson"]  # no Wilson at Arsenal: left as written
