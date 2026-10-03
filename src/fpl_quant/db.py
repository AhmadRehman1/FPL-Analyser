"""DuckDB connection helper. One local file, schema applied idempotently on connect."""

from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = REPO_ROOT / "db" / "fpl_quant_v2.duckdb"
SCHEMA_DIR = REPO_ROOT / "schema"

# One DuckDB worker thread by default. With more than one, DuckDB does not give the same answer
# twice: the row order of DISTINCT / GROUP BY / hash-join results changes from run to run (on
# any table size), and floating-point sums over more than one row group are combined in a
# different order. The pipeline feeds those rows into order-sensitive arithmetic (Plackett-Luce
# bonus sums, the Dixon-Coles likelihood, pandas group sums) and into a MILP with many tied
# optima, so the same inputs produced a different squad from one run to the next
# (docs/reports/2026-10_open_issues.md, issue 2). The workload is thousands of small queries,
# so one thread is also faster: a synthetic 4-gameweek season simulation took 28s on one
# thread against 36s on DuckDB's default of four. Pass threads=None for DuckDB's default.
DEFAULT_THREADS = 1


def connect(
    db_path: Path | str = DEFAULT_DB_PATH, read_only: bool = False, threads: int | None = DEFAULT_THREADS,
) -> duckdb.DuckDBPyConnection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path), read_only=read_only)
    if threads is not None:
        con.execute(f"SET threads = {int(threads)}")
    if not read_only:
        apply_schema(con)
    return con


def apply_schema(con: duckdb.DuckDBPyConnection) -> None:
    for sql_file in sorted(SCHEMA_DIR.glob("*.sql")):
        con.execute(sql_file.read_text(encoding="utf-8"))
