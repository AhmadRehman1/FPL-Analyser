"""Expert-consensus ingester: parsing, strict validation, coercion, claims, point-in-time."""

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from fpl_quant import ingest_expert_consensus as iec

FIXTURE = Path(__file__).parent / "fixtures" / "expert_consensus" / "expert_consensus_GW6_2026-10-02.md"

TEAMS = [
    {"id": 1, "name": "Man City", "short_name": "MCI"},
    {"id": 2, "name": "Man Utd", "short_name": "MUN"},
    {"id": 3, "name": "Bournemouth", "short_name": "BOU"},
    {"id": 4, "name": "Brighton", "short_name": "BHA"},
    {"id": 5, "name": "Chelsea", "short_name": "CHE"},
    {"id": 6, "name": "Crystal Palace", "short_name": "CRY"},
]


def _p(pid, first, second, web, team):
    return {"id": pid, "first_name": first, "second_name": second, "web_name": web, "team": team}


BOOTSTRAP = {
    "teams": TEAMS,
    "elements": [
        _p(1, "Erling", "Haaland", "Haaland", 1),
        _p(2, "Bruno Miguel", "Borges Fernandes", "B.Fernandes", 2),
        _p(3, "Antoine", "Semenyo", "Semenyo", 1),          # moved to Man City
        _p(4, "Gianluigi", "Donnarumma", "Donnarumma", 1),
        _p(5, "Pascal", "Groß", "Groß", 4),
        _p(6, "Cole", "Palmer", "Palmer", 5),
        _p(7, "Maxence", "Lacroix", "Lacroix", 6),
        _p(8, "Robert", "Lynch Sánchez", "Sánchez", 5),
        _p(9, "Yankuba", "Minteh", "Minteh", 4),
    ],
}


@pytest.fixture
def parsed():
    return iec.parse_markdown(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture
def validated(parsed):
    return iec.validate(parsed, iec.FplIndex(BOOTSTRAP))


def test_parses_consensus_and_standard_tables(parsed):
    assert set(parsed) >= {"CaptainPicks", "TransferTargets", "Differentials", "Avoid", "Sources", "Injuries", "PredictedXI"}
    assert len(parsed["CaptainPicks"]) == 4
    assert parsed["Sources"][0]["weight_1_10"] == "8"


@pytest.mark.parametrize("name", ["Wouter Lacroix", "Alessandro Donnarumma", "Ismaila Minteh", "Robert Sanchez / Martinez"])
def test_known_bad_names_are_rejected_not_guessed(name):
    p, reason = iec.FplIndex(BOOTSTRAP).match(name)
    assert p is None and reason


def test_good_names_match(validated):
    fpl = iec.FplIndex(BOOTSTRAP)
    assert fpl.match("Bruno Fernandes")[0]["id"] == 2
    assert fpl.match("Pascal Gross")[0]["id"] == 5   # sharp-s folding
    assert fpl.match("Haaland")[0]["id"] == 1


def test_rejected_rows_are_logged_and_dropped(validated):
    clean, log = validated
    rejected = [e["detail"] for e in log if e["action"] == "rejected"]
    assert any("Wouter Lacroix" in d for d in rejected)
    assert any("Alessandro Donnarumma" in d for d in rejected)
    assert any("Ismaila Minteh" in d for d in rejected)
    assert any("Sanchez / Martinez" in d for d in rejected)
    assert [r["player"] for r in clean["CaptainPicks"]] == ["Erling Haaland", "Bruno Miguel Borges Fernandes", "Antoine Semenyo"]


def test_fpl_club_wins_over_the_file(validated):
    clean, log = validated
    semenyo = next(r for r in clean["CaptainPicks"] if r["fpl_id"] == 3)
    assert semenyo["club"] == "Man City"
    assert any(e["action"] == "club_fixed" and "Bournemouth" in e["detail"] for e in log)


def test_messy_values_are_coerced_and_logged(validated):
    clean, log = validated
    pxi = {r["player"]: r for r in clean["PredictedXI"]}
    assert pxi["Pascal Groß"]["start_confidence_pct"] == 76.0
    assert pxi["Erling Haaland"]["start_confidence_pct"] == 95.0
    assert pxi["Bruno Miguel Borges Fernandes"]["start_confidence_pct"] is None
    assert pxi["Cole Palmer"]["predicted_starter"] == "No" and "Yes, if fit" in pxi["Cole Palmer"]["notes"]
    assert pxi["Bruno Miguel Borges Fernandes"]["predicted_starter"] == "No"
    assert clean["Injuries"][0]["expected_return"] == "unknown"
    semenyo = next(r for r in clean["CaptainPicks"] if r["fpl_id"] == 3)
    assert semenyo["share"] == pytest.approx(0.1)   # 'n/a' -> derived from counts
    diff = next(r for r in clean["Differentials"] if r["fpl_id"] == 5)
    assert diff["ownership_pct"] == 95.0
    coerced = [e["detail"] for e in log if e["action"] == "coerced"]
    assert any("'72-80' -> 76" in d for d in coerced)
    assert any("'95+' -> 95" in d for d in coerced)
    assert any("Yes, if fit" in d for d in coerced)
    assert any("expected_return" in d for d in coerced)


@pytest.mark.parametrize("raw,val", [("95+", 95.0), ("72-80", 76.0), ("n/a", None), ("", None), ("3.1", 3.1), ("~40%", 40.0)])
def test_coerce_number(raw, val):
    assert iec.coerce_number(raw)[0] == val


def test_consensus_claims_carry_real_numbers(validated):
    clean, _ = validated
    claims = {(c["claim_type"], c["player"]): c["numeric"] for c in iec.consensus_claims(clean)}
    assert claims[("expert_captain_share", "Erling Haaland")] == pytest.approx(0.7)
    assert claims[("expert_transfer_net", "Pascal Groß")] == pytest.approx(6 / 3)
    assert claims[("expert_transfer_net", "Cole Palmer")] == pytest.approx(-1.0)
    assert claims[("expert_avoid", "Cole Palmer")] == pytest.approx(-1.0)
    assert iec.consensus_confidence(clean) == pytest.approx((8 + 6 + 2) / 3 / 10)


def test_standard_tables_round_trip_through_the_existing_workbook_parser(validated):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "bepw", Path(__file__).resolve().parents[1] / "scripts" / "build_evidence_pull_workbook.py")
    bepw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bepw)
    clean, _ = validated
    sheets = bepw._parse_markdown_tables(iec.standard_tables_markdown(clean))
    assert set(sheets) == {"Injuries", "PredictedXI"}
    assert {r["player"] for r in sheets["PredictedXI"]} == {
        "Pascal Groß", "Erling Haaland", "Cole Palmer", "Bruno Miguel Borges Fernandes"}
    assert next(r for r in sheets["PredictedXI"] if r["player"] == "Pascal Groß")["start_confidence_pct"] == "76"


def test_point_in_time_rule():
    gw, d = iec.file_meta(FIXTURE)
    assert (gw, d) == (6, date(2026, 10, 2))
    assert iec.usable_for_gameweek(d, datetime(2026, 10, 3, 17, 30, tzinfo=timezone.utc))
    assert not iec.usable_for_gameweek(d, datetime(2026, 10, 2, 17, 30, tzinfo=timezone.utc))
    assert not iec.usable_for_gameweek(d, datetime(2026, 9, 26, 17, 30, tzinfo=timezone.utc))


def test_ingest_writes_numeric_claims(con, validated):
    clean, _ = validated
    for pid, name in ((1, "Erling Haaland"), (6, "Cole Palmer")):
        uid = f"player_{pid}"
        con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, 'Forward')", [uid, name])
        con.execute(
            "INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) VALUES (?, ?, '1', '2026-2027', ?)",
            [name, name.lower(), uid],
        )
    out = iec.ingest_consensus_claims(con, clean, date(2026, 10, 2), datetime(2026, 10, 2, 12, tzinfo=timezone.utc))
    assert out["inserted"] == 3   # Haaland captain share; Palmer transfer-net + avoid (others not in this DB)
    rows = dict(con.execute(
        "SELECT claim_type || ':' || subject_entity_id, claim_value_numeric FROM evidence_claims "
        "WHERE tab_origin LIKE 'expert_consensus:%'").fetchall())
    assert rows["expert_captain_share:player_1"] == pytest.approx(0.7)
    assert rows["expert_avoid:player_6"] == pytest.approx(-1.0)
    # re-ingest is a no-op
    again = iec.ingest_consensus_claims(con, clean, date(2026, 10, 2), datetime(2026, 10, 2, 12, tzinfo=timezone.utc))
    assert con.execute("SELECT count(*) FROM evidence_claims WHERE tab_origin LIKE 'expert_consensus:%'").fetchone()[0] == 3
    assert again["inserted"] == 0
