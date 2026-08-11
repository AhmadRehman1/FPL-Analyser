import sys
from datetime import date, datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import db  # noqa: E402


@pytest.fixture
def con(tmp_path):
    c = db.connect(tmp_path / "test.duckdb")
    yield c
    c.close()
