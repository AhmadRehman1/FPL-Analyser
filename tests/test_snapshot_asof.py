from datetime import datetime

from fpl_quant import snapshot


def _seed(con):
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p1', 'Test Player', 'MID')")
    con.execute("INSERT INTO sources (source_id, source_name, source_type) VALUES ('s1', 'Test Source', 'official')")


def test_claim_visible_after_its_own_ingested_date(con):
    _seed(con)
    con.execute(
        "INSERT INTO evidence_claims (claim_id, subject_entity_type, subject_entity_id, claim_type, "
        "source_id, source_reliability_score, observed_date, ingested_date) "
        "VALUES ('c1', 'player', 'p1', 'injury_status', 's1', 0.8, '2026-01-01', '2026-01-01 09:00:00')"
    )
    before = snapshot.get_claims_asof(con, datetime(2025, 12, 31))
    after = snapshot.get_claims_asof(con, datetime(2026, 1, 2))
    assert len(before) == 0
    assert len(after) == 1


def test_supersession_look_ahead_safe(con):
    """A correction (C2) superseding an original claim (C1) must not erase C1 from a
    snapshot taken before C2 was itself known -- otherwise a backtest run before the
    correction landed would incorrectly see no evidence at all, which is a real
    look-ahead bug, not just an edge case."""
    _seed(con)
    con.execute(
        "INSERT INTO evidence_claims (claim_id, subject_entity_type, subject_entity_id, claim_type, "
        "source_id, source_reliability_score, observed_date, ingested_date) "
        "VALUES ('c2', 'player', 'p1', 'injury_status', 's1', 0.8, '2026-01-05', '2026-01-05 09:00:00')"
    )
    con.execute(
        "INSERT INTO evidence_claims (claim_id, subject_entity_type, subject_entity_id, claim_type, "
        "source_id, source_reliability_score, observed_date, ingested_date, superseded_by) "
        "VALUES ('c1', 'player', 'p1', 'injury_status', 's1', 0.8, '2026-01-01', '2026-01-01 09:00:00', 'c2')"
    )

    asof_before_correction = snapshot.get_claims_asof(con, datetime(2026, 1, 3))
    asof_after_correction = snapshot.get_claims_asof(con, datetime(2026, 1, 10))

    assert list(asof_before_correction["claim_id"]) == ["c1"]
    assert list(asof_after_correction["claim_id"]) == ["c2"]


def test_filters_by_subject_and_claim_type(con):
    _seed(con)
    con.execute(
        "INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p2', 'Other Player', 'DEF')"
    )
    con.execute(
        "INSERT INTO evidence_claims (claim_id, subject_entity_type, subject_entity_id, claim_type, "
        "source_id, source_reliability_score, observed_date, ingested_date) "
        "VALUES ('c1', 'player', 'p1', 'injury_status', 's1', 0.8, '2026-01-01', '2026-01-01 09:00:00')"
    )
    con.execute(
        "INSERT INTO evidence_claims (claim_id, subject_entity_type, subject_entity_id, claim_type, "
        "source_id, source_reliability_score, observed_date, ingested_date) "
        "VALUES ('c2', 'player', 'p2', 'injury_status', 's1', 0.8, '2026-01-01', '2026-01-01 09:00:00')"
    )
    result = snapshot.get_claims_asof(con, datetime(2026, 6, 1), subject_entity_id="p1")
    assert list(result["claim_id"]) == ["c1"]
