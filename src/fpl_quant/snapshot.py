"""data_asof snapshot discipline (M0), exercised for real by M7's walk-forward backtest.

Every model run pins a data_asof timestamp; fact_reconciled and evidence_claims are
queried "as of" that timestamp only, so a backtest can never see data that wasn't yet
knowable at the point it's simulating a decision.
"""

from datetime import datetime

import duckdb


def get_claims_asof(
    con: duckdb.DuckDBPyConnection,
    asof: datetime,
    *,
    subject_entity_type: str | None = None,
    subject_entity_id: str | None = None,
    claim_type: str | None = None,
):
    """Look-ahead-safe evidence_claims query.

    A claim is visible iff ingested_date <= asof, observed_date <= asof (or null), and it
    has not been superseded by a correction that was *itself* already known as of asof.

    Implementation note: M0's spec says a superseded claim is "excluded outright" -- read
    literally (unconditional on asof) that would make a backtest run before the correction
    landed see nothing at all, which breaks the look-ahead-prevention guarantee M0 exists
    to provide. "Outright" is treated here as the present-day end state; under data_asof
    snapshot discipline it means "outright as of what's knowable at asof."
    """
    sql = """
        SELECT c.*
        FROM evidence_claims c
        LEFT JOIN evidence_claims s ON s.claim_id = c.superseded_by
        WHERE c.ingested_date <= ?
          AND (c.observed_date IS NULL OR c.observed_date <= ?)
          AND (c.superseded_by IS NULL OR s.ingested_date > ?)
    """
    params = [asof, asof, asof]
    if subject_entity_type is not None:
        sql += " AND c.subject_entity_type = ?"
        params.append(subject_entity_type)
    if subject_entity_id is not None:
        sql += " AND c.subject_entity_id = ?"
        params.append(subject_entity_id)
    if claim_type is not None:
        sql += " AND c.claim_type = ?"
        params.append(claim_type)
    return con.execute(sql, params).fetchdf()


def get_matches_asof(con: duckdb.DuckDBPyConnection, asof: datetime):
    return con.execute(
        "SELECT * FROM fact_match WHERE _ingested_at <= ?", [asof]
    ).fetchdf()


def get_player_match_stats_asof(con: duckdb.DuckDBPyConnection, asof: datetime):
    return con.execute(
        "SELECT * FROM fact_player_match_stats WHERE _ingested_at <= ?", [asof]
    ).fetchdf()


def get_player_season_stats_asof(con: duckdb.DuckDBPyConnection, asof: datetime):
    return con.execute(
        "SELECT * FROM fact_player_season_stats WHERE _ingested_at <= ?", [asof]
    ).fetchdf()
