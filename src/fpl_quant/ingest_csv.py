"""fact_raw ingestion: every CSV under FPL-Core-Insights becomes its own untouched table.

One table per (season, source_relpath) -- stable name, append-only across ingestion runs
(the CSVs refresh twice daily per the source repo's README; a data_asof query needs to be
able to tell what was known as of a given ingestion, not just the latest pull). Rows are
never mutated or deleted; a byte-identical re-ingestion is a no-op (see UNIQUE constraint
on fact_raw_ingestion_log).
"""

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

import duckdb

SEASON_DIR_RE = re.compile(r"^\d{4}-\d{4}$")


def _slugify(relpath: str) -> str:
    slug = relpath.lower()
    slug = re.sub(r"\.csv$", "", slug)
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    return slug


def raw_table_name(season: str, source_relpath: str) -> str:
    season_slug = season.replace("-", "_")
    return f"raw_{season_slug}_{_slugify(source_relpath)}"


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_csv_files(data_root: Path) -> list[tuple[str, str, Path]]:
    """Returns (season, source_relpath, absolute_path) for every CSV under data_root.

    data_root is expected to be the FPL-Core-Insights repo's `data/` directory.
    """
    out = []
    for season_dir in sorted(data_root.iterdir()):
        if not season_dir.is_dir() or not SEASON_DIR_RE.match(season_dir.name):
            continue
        season = season_dir.name
        for csv_path in sorted(season_dir.rglob("*.csv")):
            source_relpath = csv_path.relative_to(season_dir).as_posix()
            out.append((season, source_relpath, csv_path))
    return out


def ingest_csv_file(
    con: duckdb.DuckDBPyConnection, season: str, source_relpath: str, abs_path: Path
) -> dict:
    """Ingests a single CSV into its fact_raw table. Idempotent on identical file content."""
    file_hash = _file_hash(abs_path)
    table = raw_table_name(season, source_relpath)

    already = con.execute(
        "SELECT 1 FROM fact_raw_ingestion_log WHERE raw_table_name = ? AND source_file_hash = ?",
        [table, file_hash],
    ).fetchone()
    if already:
        return {"table": table, "status": "skipped_identical", "rows": 0}

    table_exists = con.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = ?", [table]
    ).fetchone()

    ingested_at = datetime.now(timezone.utc)
    batch_id = con.execute("SELECT nextval('seq_ingestion_batch')").fetchone()[0]
    abs_path_sql = str(abs_path).replace("'", "''")
    ingested_at_sql = ingested_at.strftime("%Y-%m-%d %H:%M:%S.%f")

    select_sql = f"""
        SELECT
            *,
            {batch_id} AS _batch_id,
            TIMESTAMP '{ingested_at_sql}' AS _ingested_at,
            '{file_hash}' AS _source_file_hash,
            row_number() OVER () AS _row_num
        FROM read_csv('{abs_path_sql}', ALL_VARCHAR=TRUE, header=TRUE)
    """

    if not table_exists:
        con.execute(f'CREATE TABLE "{table}" AS {select_sql}')
    else:
        con.execute(f'INSERT INTO "{table}" {select_sql}')

    row_count = con.execute(f'SELECT count(*) FROM "{table}" WHERE _batch_id = ?', [batch_id]).fetchone()[0]

    con.execute(
        """
        INSERT INTO fact_raw_ingestion_log
            (batch_id, raw_table_name, season, source_relpath, source_file_hash, row_count, ingested_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [batch_id, table, season, source_relpath, file_hash, row_count, ingested_at],
    )
    return {"table": table, "status": "ingested", "rows": row_count}


def ingest_all(con: duckdb.DuckDBPyConnection, data_root: Path) -> list[dict]:
    results = []
    for season, source_relpath, abs_path in discover_csv_files(data_root):
        results.append(ingest_csv_file(con, season, source_relpath, abs_path))
    return results
