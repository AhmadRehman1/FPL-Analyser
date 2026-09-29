"""scripts/walkforward_summary.py runs against the real schema."""

import importlib.util
from pathlib import Path

import duckdb

from fpl_quant import db

_SPEC = importlib.util.spec_from_file_location(
    "walkforward_summary", Path(__file__).resolve().parents[1] / "scripts" / "walkforward_summary.py"
)
wfs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(wfs)


def test_summarize_empty_run():
    con = duckdb.connect(":memory:")
    db.apply_schema(con)
    out = wfs.summarize(con, 1)
    assert out["headline"]["beats_crowd_points_delta"] is None
    assert set(out["price_band"]) == {"<5.0", "5.0-7.0", "7.0-9.0", "9.0+"}
    assert out["captain"] == {"n": 0}
    assert out["per_gameweek"] == []
