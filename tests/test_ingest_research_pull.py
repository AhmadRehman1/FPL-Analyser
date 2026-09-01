"""ingest_research_pull -- the manual web-research evidence pull (v2 Perplexity format +
v1 backward compatibility). Builds a small workbook in memory, ingests it against a fresh
DB, and checks the right evidence_claims rows land with the right claim_type and payload.
"""

from __future__ import annotations

import json

import openpyxl
import pytest

from fpl_quant import entity_resolution as er
from fpl_quant import ingest_research_pull as irp


def _seed(con):
    con.execute("INSERT INTO sources (source_id, source_name, source_type, base_reliability_score) "
                "VALUES ('src_the_athletic', 'The Athletic', 'journalist', 0.8)")
    for uid, name, pos in [
        ("p_haaland", "Erling Haaland", "Forward"),
        ("p_gross", "Pascal Gross", "Midfielder"),
        ("p_saka", "Bukayo Saka", "Midfielder"),
        ("p_rice", "Declan Rice", "Midfielder"),
    ]:
        con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, ?)", [uid, name, pos])
        con.execute(
            "INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) "
            "VALUES (?, ?, '1', '2026-2027', ?)", [name, er.normalize_name(name), uid],
        )


def _wb(sheets: dict[str, list[list]]) -> openpyxl.Workbook:
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(name)
        for r in rows:
            ws.append(r)
    return wb


def _write(tmp_path, wb) -> str:
    p = tmp_path / "pull.xlsx"
    wb.save(p)
    return str(p)


def _claims(con, claim_type=None):
    q = "SELECT subject_entity_id, claim_type, claim_value, claim_value_numeric, information_type, tab_origin FROM evidence_claims"
    if claim_type:
        q += f" WHERE claim_type = '{claim_type}'"
    return [
        {"uid": r[0], "type": r[1], "payload": json.loads(r[2]) if r[2] else {},
         "numeric": r[3], "info": r[4], "tab": r[5]}
        for r in con.execute(q).fetchall()
    ]


# ------------------------------------------------------------------ v2 tabs

def test_injuries_tab_maps_status_to_the_minutes_model_category(con, tmp_path):
    _seed(con)
    wb = _wb({"Injuries": [
        ["player", "club", "status", "issue", "date_reported", "expected_return",
         "source_name", "source_type", "confidence_1_10", "information_type", "observed_date", "notes"],
        ["Erling Haaland", "Man City", "Doubt", "knock", "2026-08-30", "unknown",
         "The Athletic", "journalist", 7, "OPINION", "2026-08-31", "trained partially"],
    ]})
    irp.ingest_all(con, _write(tmp_path, wb), source_tier_params_version=1)
    inj = _claims(con, "injury_status")
    assert len(inj) == 1
    assert inj[0]["uid"] == "p_haaland"
    assert inj[0]["payload"]["category"] == "Doubt"       # the key minutes_adjustment_params keys off
    assert inj[0]["payload"]["issue"] == "knock"
    assert inj[0]["tab"] == "research_pull:Injuries"


def test_predicted_xi_tab_carries_start_confidence_as_a_0_1_numeric(con, tmp_path):
    _seed(con)
    wb = _wb({"PredictedXI": [
        ["player", "club", "predicted_starter", "start_confidence_pct", "expected_minutes",
         "position", "source_name", "source_type", "confidence_1_10", "information_type", "observed_date", "notes"],
        ["Pascal Gross", "Brighton", "Yes", 92, "75-90", "CM",
         "The Athletic", "journalist", 8, "OPINION", "2026-08-31", "nailed"],
    ]})
    irp.ingest_all(con, _write(tmp_path, wb), source_tier_params_version=1)
    px = _claims(con, "predicted_xi")
    assert len(px) == 1 and px[0]["uid"] == "p_gross"
    assert px[0]["numeric"] == pytest.approx(0.92)
    assert px[0]["payload"]["expected_minutes"] == "75-90"


def test_rotation_tab_requires_a_valence_and_maps_to_manager_tendency(con, tmp_path):
    _seed(con)
    wb = _wb({"Rotation": [
        ["player", "club", "manager", "valence", "pattern", "trigger",
         "source_name", "source_type", "confidence_1_10", "information_type", "observed_date", "notes"],
        ["Bukayo Saka", "Arsenal", "Arteta", "negative", "rested for CL group games", "midweek CL",
         "The Athletic", "journalist", 6, "OPINION", "2026-08-31", ""],
        ["Declan Rice", "Arsenal", "Arteta", "", "unclear", "",   # no valence -> skipped
         "The Athletic", "journalist", 4, "OPINION", "2026-08-31", ""],
    ]})
    res = irp.ingest_all(con, _write(tmp_path, wb), source_tier_params_version=1)
    mt = _claims(con, "manager_tendency")
    assert [c["uid"] for c in mt] == ["p_saka"]
    assert mt[0]["payload"]["valence"] == "negative"
    assert res["rotation"] == {"inserted": 1, "skipped": 1}


def test_role_change_routes_by_change_value(con, tmp_path):
    _seed(con)
    wb = _wb({"RoleChange": [
        ["player", "club", "change", "cause", "effective_from",
         "source_name", "source_type", "confidence_1_10", "information_type", "observed_date", "notes"],
        ["Pascal Gross", "Brighton", "frozen_out", "new signing", "2026-08-20",
         "The Athletic", "journalist", 6, "OPINION", "2026-08-31", ""],
        ["Declan Rice", "Arsenal", "likely_january_exit", "wants out", "",
         "The Athletic", "journalist", 3, "OPINION", "2026-08-31", ""],
    ]})
    irp.ingest_all(con, _write(tmp_path, wb), source_tier_params_version=1)
    by_uid = {c["uid"]: c for c in _claims(con)}
    assert by_uid["p_gross"]["type"] == "manager_tendency"
    assert by_uid["p_gross"]["payload"]["valence"] == "negative"
    assert by_uid["p_rice"]["type"] == "transfer_likelihood"
    assert by_uid["p_rice"]["payload"]["status"] == "Expected"   # NOT "Complete" -> minutes_model still counts it


def test_set_pieces_tab_emits_a_claim_per_taker_with_order(con, tmp_path):
    _seed(con)
    wb = _wb({"SetPieces": [
        ["club", "duty", "primary_taker", "secondary_taker", "deputy_if_primary_absent",
         "source_name", "source_type", "confidence_1_10", "information_type", "observed_date", "notes"],
        ["Arsenal", "Penalties", "Bukayo Saka", "Declan Rice", "Bukayo Saka",
         "The Athletic", "journalist", 9, "FACT", "2026-08-31", "Saka scored the GW2 pen"],
        ["Man City", "Corners", "-", "", "",   # no takers -> nothing emitted
         "The Athletic", "journalist", 5, "OPINION", "2026-08-31", ""],
    ]})
    irp.ingest_all(con, _write(tmp_path, wb), source_tier_params_version=1)
    sp = _claims(con, "set_piece_order_override")
    orders = {(c["uid"], c["payload"]["order"]) for c in sp}
    assert orders == {("p_saka", "primary"), ("p_rice", "secondary")}
    assert all(c["payload"]["duty"] == "Penalties" for c in sp)
    assert next(c for c in sp if c["payload"]["order"] == "primary")["info"] == "FACT"


def test_a_bad_row_is_skipped_not_raised(con, tmp_path):
    _seed(con)
    wb = _wb({"Injuries": [
        ["player", "club", "status", "issue", "date_reported", "expected_return",
         "source_name", "source_type", "confidence_1_10", "information_type", "observed_date", "notes"],
        ["Nonexistent Player", "Nowhere", "Out", "x", "", "", "The Athletic", "journalist", 5, "FACT", "2026-08-31", ""],
        ["Erling Haaland", "Man City", "Out", "acl", "", "", "", "", 5, "FACT", "2026-08-31", ""],  # no source
        ["Erling Haaland", "Man City", "Out", "acl", "", "", "The Athletic", "journalist", 9, "FACT", "2026-08-31", ""],
    ]})
    res = irp.ingest_all(con, _write(tmp_path, wb), source_tier_params_version=1)
    assert res["injuries"] == {"inserted": 1, "skipped": 2}


def test_missing_tabs_are_skipped_cleanly(con, tmp_path):
    _seed(con)
    wb = _wb({"README": [["notes"], ["nothing here"]]})
    res = irp.ingest_all(con, _write(tmp_path, wb), source_tier_params_version=1)
    assert res["tabs_present"] == []
    assert res["injuries"]["inserted"] == 0


def test_flexible_date_parsing():
    from datetime import date
    assert irp._parse_flexible_date("2026-08-10") == date(2026, 8, 10)
    assert irp._parse_flexible_date("2026-08-xx") == date(2026, 8, 15)
    assert irp._parse_flexible_date("2026-summer") == date(2026, 7, 15)
    assert irp._parse_flexible_date("garbage") is None
    assert irp._parse_flexible_date(None) is None


# ------------------------------------------------------------------ v1 backward compatibility

def test_v1_workbook_still_ingests(con, tmp_path):
    _seed(con)
    wb = _wb({
        "Injuries": [
            ["subject", "club", "claim_type", "status", "claim_value", "source_name", "source_type",
             "confidence_1_10", "information_type", "observed_date", "notes"],
            ["Erling Haaland", "Man City", "injury_status", "Doubt", "ankle", "The Athletic", "journalist",
             7, "OPINION", "2026-08-31", ""],
        ],
        "SetPieceTakers": [
            ["club", "duty", "primary_taker", "secondary_taker", "claim_value", "source_name", "source_type",
             "confidence_1_10", "information_type", "observed_date", "notes"],
            ["Arsenal", "Penalties", "Bukayo Saka", "-", "", "The Athletic", "journalist", 9, "FACT", "2026-08-31", ""],
        ],
        "PriceNotes": [
            ["subject", "club", "claim_type", "claim_value_numeric_gbp_m", "claim_value", "source_name",
             "source_type", "confidence_1_10", "information_type", "observed_date", "notes"],
            ["Erling Haaland", "Man City", "fpl_price_note", 15.6, "rising", "The Athletic", "journalist",
             6, "OPINION", "2026-08-31", ""],
        ],
    })
    res = irp.ingest_all(con, _write(tmp_path, wb), source_tier_params_version=1)
    assert res["injuries"]["inserted"] == 1
    assert res["set_pieces"]["inserted"] == 1
    assert res["price_watch"]["inserted"] == 1
    assert {c["type"] for c in _claims(con)} == {"injury_status", "set_piece_order_override", "fpl_price_note"}
