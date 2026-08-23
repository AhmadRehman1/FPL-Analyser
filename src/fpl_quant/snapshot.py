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
    enforce_ingested_date: bool = True,
):
    """Look-ahead-safe evidence_claims query.

    A claim is visible iff ingested_date <= asof, observed_date <= asof (or null), and it
    has not been superseded by a correction that was *itself* already known as of asof.

    Implementation note: M0's spec says a superseded claim is "excluded outright" -- read
    literally (unconditional on asof) that would make a backtest run before the correction
    landed see nothing at all, which breaks the look-ahead-prevention guarantee M0 exists
    to provide. "Outright" is treated here as the present-day end state; under data_asof
    snapshot discipline it means "outright as of what's knowable at asof."

    enforce_ingested_date=False (M7's backtest usage only, live callers keep the default):
    ingested_date is stamped wall-clock at whichever real-world day this repo's ingestion
    script last ran -- ingest_workbook.py:202, one shared timestamp per ingestion batch, not
    per-claim knowability. For any live run (calibration_asof_date=date.today(), per
    run_ingestion.py) that's always <= asof and the check is a harmless no-op. For a backtest
    asof pinned into 2024-25/2025-26 it would be strictly AFTER every historical claim's
    ingested_date, so the default condition would silently exclude *all* evidence for *every*
    backtest gameweek -- the opposite failure from a look-ahead leak, but just as invalidating.
    Backtesting has no real substitute for "when would this have been knowable" beyond
    observed_date itself, so this flag drops the ingested_date condition and relies on
    observed_date (+ the superseded_by asof-relative logic, which is genuinely date-based and
    still correct) alone.
    """
    sql = """
        SELECT c.*
        FROM evidence_claims c
        LEFT JOIN evidence_claims s ON s.claim_id = c.superseded_by
        WHERE (c.observed_date IS NULL OR c.observed_date <= ?)
          AND (c.superseded_by IS NULL OR s.ingested_date > ?)
    """
    params: list[object] = [asof, asof]
    if enforce_ingested_date:
        sql += " AND c.ingested_date <= ?"
        params.append(asof)
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
