"""Regression tests for a real bug: ingest_workbook._insert_claim() previously always
inserted a fresh claim_id with no existence check, and evidence_claims has no uniqueness
constraint on anything content-identifying -- so every re-ingestion of the workbook (which
the project's own weekly-update workflow explicitly expects) silently duplicated every
still-present claim. evidence_blend.py's weighted-average/categorical-distribution blending
then over-weighted whichever evidence happened to survive across the most re-ingestion runs,
not the most reliable or most recent evidence.

These tests exercise _insert_claim() directly against a minimal synthetic DB (a `sources`
row + a `dim_player` row), independent of the real evidence workbook fixture used by the rest
of this file's module (which is skipped when data/external/ isn't present, as it isn't in a
fresh checkout) -- so this coverage runs everywhere, not just when the real workbook is staged.
"""

from datetime import date, datetime, timezone

from fpl_quant import ingest_workbook as iw


def _seed_player_and_source(con):
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p1', 'Test Player', 'Midfielder')")
    con.execute(
        "INSERT INTO sources (source_id, source_name, source_type, base_reliability_score) "
        "VALUES ('src1', 'Test Source', 'official', 0.9)"
    )


def _claim_kwargs(**overrides):
    kwargs = dict(
        subject_entity_type="player", subject_entity_id="p1", claim_type="injury_status",
        claim_value={"category": "Doubt", "issue": "hamstring"}, claim_value_numeric=None,
        information_type="FACT", source_id="src1", source_reliability_score=0.9, confidence=0.8,
        observed_date=date(2026, 8, 10), ingested_date=datetime(2026, 8, 10, tzinfo=timezone.utc),
        tab_origin="4_Injury Database", row_origin=7,
    )
    kwargs.update(overrides)
    return kwargs


def test_reingesting_identical_content_is_a_genuine_noop(con):
    _seed_player_and_source(con)
    ok1 = iw._insert_claim(con, **_claim_kwargs())
    ok2 = iw._insert_claim(con, **_claim_kwargs())  # identical re-ingestion, e.g. a repeated run
    assert ok1 is True
    assert ok2 is False, "unchanged content re-ingested must not create a duplicate claim"
    count = con.execute("SELECT count(*) FROM evidence_claims WHERE subject_entity_id = 'p1'").fetchone()[0]
    assert count == 1


def test_reingesting_changed_content_supersedes_not_duplicates(con):
    _seed_player_and_source(con)
    iw._insert_claim(con, **_claim_kwargs())
    ok2 = iw._insert_claim(con, **_claim_kwargs(claim_value={"category": "Out", "issue": "hamstring"}))
    assert ok2 is True

    rows = con.execute(
        "SELECT claim_value, superseded_by FROM evidence_claims WHERE subject_entity_id = 'p1'"
    ).fetchall()
    assert len(rows) == 2, "changed content must land as a new claim, not overwrite the old row in place"
    by_value = {value: superseded_by for value, superseded_by in rows}
    old_value = next(v for v in by_value if "Doubt" in v)
    new_value = next(v for v in by_value if "Out" in v)
    assert by_value[old_value] is not None, "the stale claim must be marked superseded, not left live alongside the new one"
    assert by_value[new_value] is None


def test_different_workbook_rows_never_collide(con):
    """Two genuinely different workbook rows for the same player/claim_type/source must both
    be kept -- the identity key is scoped to (tab_origin, row_origin), not just
    (subject_entity_id, claim_type, source_id)."""
    _seed_player_and_source(con)
    iw._insert_claim(con, **_claim_kwargs(row_origin=7))
    iw._insert_claim(con, **_claim_kwargs(row_origin=8, claim_value={"category": "Doubt", "issue": "ankle"}))
    count = con.execute("SELECT count(*) FROM evidence_claims WHERE subject_entity_id = 'p1'").fetchone()[0]
    assert count == 2
