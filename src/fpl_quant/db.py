"""DuckDB connection helper. One local file, schema applied idempotently on connect."""

from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = REPO_ROOT / "db" / "fpl_quant_v2.duckdb"
SCHEMA_DIR = REPO_ROOT / "schema"


def connect(db_path: Path | str = DEFAULT_DB_PATH, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path), read_only=read_only)
    if not read_only:
        apply_schema(con)
    return con


def apply_schema(con: duckdb.DuckDBPyConnection) -> None:
    for sql_file in sorted(SCHEMA_DIR.glob("*.sql")):
        con.execute(sql_file.read_text(encoding="utf-8"))
