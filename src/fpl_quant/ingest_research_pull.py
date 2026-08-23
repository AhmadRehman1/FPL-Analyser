"""Ingests the ad-hoc web-research evidence pull (Transfers/Injuries/SetPieceTakers/
PriceNotes) into evidence_claims -- a second, differently-shaped evidence source
alongside the main 40-tab workbook (flat columns per tab, not the tab-per-fact-type
layout), so it gets its own small ingestion module rather than being shoehorned into
ingest_workbook.py's per-tab functions. Same destination schema (evidence_claims),
same source-tier classification and reliability-scoring mechanism (M1b).

PriceNotes is ingested for audit/provenance visibility only -- claim_type='fpl_price_note'
-- never promoted into fact_reconciled's authoritative now_cost. That boundary is M0's own
architectural principle ("a stat is never conflated with an opinion, even a well-sourced
one"), and this sheet's own PriceNotes rows explicitly flag themselves as sometimes
unverified ("do not treat this number as confirmed"), which only reinforces it.
"""

import json
import re
from datetime import date, datetime, timezone

import duckdb
import openpyxl

from . import ingest_workbook as iw


def _parse_flexible_date(val) -> date | None:
    """Real research-note dates are messy: '2026-08-10' (clean), '2026-06-xx' (unknown
    day), '2026-summer' (unknown month entirely). Approximated to a representative day
    rather than dropped -- mid-month for a known month, mid-July for a bare season/summer
    reference -- and documented here as an approximation, not treated as exact."""
    if val is None:
        return None
    s = str(val).strip()
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        pass
    m = re.match(r"^(\d{4})-(\d{2})-xx$", s)
    if m:
        return date(int(m.group(1)), int(m.group(2)), 15)
    m = re.match(r"^(\d{4})-summer$", s, re.IGNORECASE)
    if m:
        return date(int(m.group(1)), 7, 15)
    return None


def register_new_sources(con: duckdb.DuckDBPyConnection, wb: openpyxl.Workbook, params_version: int) -> int:
    existing = {r[0] for r in con.execute("SELECT source_name FROM sources").fetchall()}
    max_citations = con.execute("SELECT max(citation_count) FROM sources").fetchone()[0] or 1

    names: set[str] = set()
    for tab in ("Transfers", "Injuries", "SetPieceTakers", "PriceNotes"):
        ws = wb[tab]
        header = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
        idx = header.index("source_name")
        for row in ws.iter_rows(min_row=2, values_only=True):
            val = row[idx] if idx < len(row) else None
            if val and isinstance(val, str):
                names.add(val.strip())

    new_names = names - existing
    n = 0
    for name in sorted(new_names):
        source_type = iw.classify_source_type(name)
        row = con.execute(
            "SELECT value_numeric FROM param_versions WHERE param_family = 'source_tier_weights' "
            "AND param_version = ? AND param_key = 'tier_weight' AND dimensions = ?",
            [params_version, json.dumps({"source_type": source_type}, sort_keys=True, separators=(",", ":"))],
        ).fetchone()
        tier_weight = row[0] if row else 0.4
        # no citation-count index for this source pull -- default to 1 (lowest observed
        # weight elsewhere in the system), not silently inflated.
        base_score = tier_weight * iw._log_scaled(1, max_citations)
        source_id = "src_" + name.lower().replace(" ", "_").replace("/", "_")
        con.execute(
            "INSERT INTO sources (source_id, source_name, source_type, base_reliability_score, "
            "citation_count, source_notes, last_reviewed_date) VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (source_name) DO NOTHING",
            [source_id, name, source_type, base_score, 1, None, None],
        )
        n += 1
    return n


def ingest_transfers(con, wb, ingested_date) -> dict:
    ws = wb["Transfers"]
    inserted, skipped = 0, 0
    for i, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
        subject, club_from, club_to, _claim_type, claim_value, source_name, _source_type, \
            conf, info_type, observed, notes = row[:11]
        if subject is None:
            continue
        player_uid = iw._resolve_player(con, subject)
        source_id = iw._source_id_for(con, source_name)
        if not player_uid or not source_id:
            skipped += 1
            continue
        ok = iw._insert_claim(
            con, subject_entity_type="player", subject_entity_id=player_uid,
            claim_type="transfer_likelihood",
            claim_value={"status": "Complete", "old_club": club_from, "new_club": club_to,
                         "description": claim_value, "notes": notes},
            claim_value_numeric=1.0, information_type=info_type, source_id=source_id,
            source_reliability_score=iw._reliability_for(con, source_id), confidence=iw._confidence_0_1(conf),
            observed_date=_parse_flexible_date(observed), ingested_date=ingested_date,
            tab_origin="research_pull:Transfers", row_origin=i,
        )
        inserted += 1 if ok else 0
        skipped += 0 if ok else 1
    return {"inserted": inserted, "skipped": skipped}


def ingest_injuries(con, wb, ingested_date) -> dict:
    ws = wb["Injuries"]
    inserted, skipped = 0, 0
    for i, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
        subject, club, _claim_type, status, claim_value, source_name, _source_type, \
            conf, info_type, observed, notes = row[:11]
        if subject is None:
            continue
        player_uid = iw._resolve_player(con, subject)
        source_id = iw._source_id_for(con, source_name)
        if not player_uid or not source_id:
            skipped += 1
            continue
        ok = iw._insert_claim(
            con, subject_entity_type="player", subject_entity_id=player_uid,
            claim_type="injury_status",
            claim_value={"category": status, "issue": claim_value, "notes": notes},
            claim_value_numeric=None, information_type=info_type, source_id=source_id,
            source_reliability_score=iw._reliability_for(con, source_id), confidence=iw._confidence_0_1(conf),
            observed_date=_parse_flexible_date(observed), ingested_date=ingested_date,
            tab_origin="research_pull:Injuries", row_origin=i,
        )
        inserted += 1 if ok else 0
        skipped += 0 if ok else 1
    return {"inserted": inserted, "skipped": skipped}


def ingest_set_piece_takers(con, wb, ingested_date) -> dict:
    ws = wb["SetPieceTakers"]
    inserted, skipped = 0, 0
    for i, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
        club, duty, primary, secondary, claim_value, source_name, _source_type, \
            conf, info_type, observed, notes = row[:11]
        source_id = iw._source_id_for(con, source_name)
        if not source_id:
            skipped += 1
            continue
        for taker, order in ((primary, "primary"), (secondary, "secondary")):
            if not taker or str(taker).strip() in ("-", ""):
                continue
            player_uid = iw._resolve_player(con, taker)
            if not player_uid:
                skipped += 1
                continue
            ok = iw._insert_claim(
                con, subject_entity_type="player", subject_entity_id=player_uid,
                claim_type="set_piece_order_override",
                claim_value={"club": club, "duty": duty, "order": order, "description": claim_value, "notes": notes},
                claim_value_numeric=None, information_type=info_type, source_id=source_id,
                source_reliability_score=iw._reliability_for(con, source_id), confidence=iw._confidence_0_1(conf),
                observed_date=_parse_flexible_date(observed), ingested_date=ingested_date,
                tab_origin="research_pull:SetPieceTakers", row_origin=i,
            )
            inserted += 1 if ok else 0
            skipped += 0 if ok else 1
    return {"inserted": inserted, "skipped": skipped}


def ingest_price_notes(con, wb, ingested_date) -> dict:
    ws = wb["PriceNotes"]
    inserted, skipped = 0, 0
    for i, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
        subject, club, _claim_type, numeric_gbp, claim_value, source_name, _source_type, \
            conf, info_type, observed, notes = row[:11]
        if subject is None:
            continue
        player_uid = iw._resolve_player(con, subject)
        source_id = iw._source_id_for(con, source_name)
        if not player_uid or not source_id:
            skipped += 1
            continue
        ok = iw._insert_claim(
            con, subject_entity_type="player", subject_entity_id=player_uid,
            claim_type="fpl_price_note",
            claim_value={"description": claim_value, "notes": notes},
            claim_value_numeric=float(numeric_gbp) if numeric_gbp is not None else None,
            information_type=info_type, source_id=source_id,
            source_reliability_score=iw._reliability_for(con, source_id), confidence=iw._confidence_0_1(conf),
            observed_date=_parse_flexible_date(observed), ingested_date=ingested_date,
            tab_origin="research_pull:PriceNotes", row_origin=i,
        )
        inserted += 1 if ok else 0
        skipped += 0 if ok else 1
    return {"inserted": inserted, "skipped": skipped}


def ingest_all(con: duckdb.DuckDBPyConnection, xlsx_path: str, source_tier_params_version: int) -> dict:
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    n_sources = register_new_sources(con, wb, source_tier_params_version)
    ingested_date = datetime.now(timezone.utc)
    return {
        "new_sources_registered": n_sources,
        "transfers": ingest_transfers(con, wb, ingested_date),
        "injuries": ingest_injuries(con, wb, ingested_date),
        "set_piece_takers": ingest_set_piece_takers(con, wb, ingested_date),
        "price_notes": ingest_price_notes(con, wb, ingested_date),
    }
