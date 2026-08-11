import duckdb
import pytest


def test_core_tables_exist(con):
    tables = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    required = {
        "fact_raw_ingestion_log", "dim_team", "team_alias", "dim_player", "player_alias",
        "fact_match", "fact_player_match_stats", "fact_player_season_stats",
        "fact_reconciled_column_semantics", "sources", "evidence_claims",
        "claims_pending_manual_decomposition", "workbook_tab_allowlist", "param_versions",
        "model_runs",
    }
    assert required.issubset(tables)


def test_named_param_views_exist(con):
    views = {r[0] for r in con.execute("SELECT table_name FROM information_schema.views").fetchall()}
    assert {"claim_type_decay_params", "source_tier_weights"}.issubset(views)


def test_evidence_claims_subject_entity_type_check(con):
    con.execute(
        "INSERT INTO sources (source_id, source_name, source_type) VALUES ('s1', 'Test Source', 'official')"
    )
    with pytest.raises(duckdb.Error):
        con.execute(
            "INSERT INTO evidence_claims (claim_id, subject_entity_type, subject_entity_id, claim_type, "
            "source_id, source_reliability_score, ingested_date) "
            "VALUES ('c1', 'not_a_valid_type', 'p1', 'injury_status', 's1', 0.5, current_timestamp)"
        )


def test_evidence_claims_information_type_check(con):
    con.execute(
        "INSERT INTO sources (source_id, source_name, source_type) VALUES ('s1', 'Test Source', 'official')"
    )
    with pytest.raises(duckdb.Error):
        con.execute(
            "INSERT INTO evidence_claims (claim_id, subject_entity_type, subject_entity_id, claim_type, "
            "information_type, source_id, source_reliability_score, ingested_date) "
            "VALUES ('c1', 'player', 'p1', 'injury_status', 'MAYBE', 's1', 0.5, current_timestamp)"
        )


def test_column_semantics_enum_check(con):
    with pytest.raises(duckdb.Error):
        con.execute(
            "INSERT INTO fact_reconciled_column_semantics (table_name, column_name, semantics) "
            "VALUES ('t', 'c', 'not_a_real_semantic')"
        )


def test_source_type_check(con):
    with pytest.raises(duckdb.Error):
        con.execute(
            "INSERT INTO sources (source_id, source_name, source_type) VALUES ('s1', 'Bad', 'not_a_tier')"
        )


def test_workbook_tab_allowlist_status_check(con):
    with pytest.raises(duckdb.Error):
        con.execute(
            "INSERT INTO workbook_tab_allowlist (tab_name, status, reason) VALUES ('x', 'not_a_status', 'r')"
        )
