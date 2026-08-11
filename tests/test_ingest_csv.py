from pathlib import Path

from fpl_quant import ingest_csv


def _write_csv(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_ingest_creates_table_with_row_count(con, tmp_path):
    csv_path = tmp_path / "2024-2025" / "teams.csv"
    _write_csv(csv_path, ["code,id,name", "3,1,Arsenal", "7,2,Aston Villa"])

    result = ingest_csv.ingest_csv_file(con, "2024-2025", "teams.csv", csv_path)
    assert result["status"] == "ingested"
    assert result["rows"] == 2

    table = ingest_csv.raw_table_name("2024-2025", "teams.csv")
    rows = con.execute(f'SELECT code, id, name FROM "{table}" ORDER BY id').fetchall()
    assert rows == [("3", "1", "Arsenal"), ("7", "2", "Aston Villa")]


def test_reingest_identical_file_is_noop(con, tmp_path):
    csv_path = tmp_path / "2024-2025" / "teams.csv"
    _write_csv(csv_path, ["code,id,name", "3,1,Arsenal"])
    ingest_csv.ingest_csv_file(con, "2024-2025", "teams.csv", csv_path)
    result = ingest_csv.ingest_csv_file(con, "2024-2025", "teams.csv", csv_path)
    assert result["status"] == "skipped_identical"

    table = ingest_csv.raw_table_name("2024-2025", "teams.csv")
    assert con.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0] == 1


def test_reingest_changed_file_appends_new_batch(con, tmp_path):
    csv_path = tmp_path / "2024-2025" / "teams.csv"
    _write_csv(csv_path, ["code,id,name", "3,1,Arsenal"])
    ingest_csv.ingest_csv_file(con, "2024-2025", "teams.csv", csv_path)

    _write_csv(csv_path, ["code,id,name", "3,1,Arsenal", "7,2,Aston Villa"])
    result = ingest_csv.ingest_csv_file(con, "2024-2025", "teams.csv", csv_path)
    assert result["status"] == "ingested"

    table = ingest_csv.raw_table_name("2024-2025", "teams.csv")
    # append-only: both batches' rows survive, never mutated/deleted
    assert con.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0] == 3
    batches = con.execute(f'SELECT count(DISTINCT _batch_id) FROM "{table}"').fetchone()[0]
    assert batches == 2


def test_raw_columns_are_all_varchar_untouched(con, tmp_path):
    csv_path = tmp_path / "2024-2025" / "teams.csv"
    _write_csv(csv_path, ["code,id,name", "03,1,Arsenal"])  # leading zero -- must not become 3
    ingest_csv.ingest_csv_file(con, "2024-2025", "teams.csv", csv_path)
    table = ingest_csv.raw_table_name("2024-2025", "teams.csv")
    code = con.execute(f'SELECT code FROM "{table}"').fetchone()[0]
    assert code == "03"


def test_ingestion_log_records_batch(con, tmp_path):
    csv_path = tmp_path / "2024-2025" / "teams.csv"
    _write_csv(csv_path, ["code,id,name", "3,1,Arsenal"])
    ingest_csv.ingest_csv_file(con, "2024-2025", "teams.csv", csv_path)
    log_rows = con.execute(
        "SELECT season, source_relpath, row_count FROM fact_raw_ingestion_log"
    ).fetchall()
    assert log_rows == [("2024-2025", "teams.csv", 1)]
