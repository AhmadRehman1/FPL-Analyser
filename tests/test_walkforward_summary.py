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
    assert out["captain_counterfactuals"] == {"n": 0}
    assert out["per_gameweek"] == []


def test_captain_rule_points_scores_each_rule_on_the_same_xi():
    # (ep, var, p95, realized): the steady 6.0 player vs the boom-or-bust 5.5 one
    gw1 = [(6.0, 4.0, 9.0, 2.0), (5.5, 25.0, 14.0, 15.0), (3.0, 1.0, 5.0, 4.0)]
    gw2 = [(6.0, 4.0, 9.0, 8.0), (5.5, 25.0, 14.0, 1.0), (3.0, 1.0, 5.0, 3.0)]
    out = wfs.captain_rule_points([gw1, gw2])
    assert out["n"] == 2
    assert out["points_per_gw"]["top_ep"] == 5.0           # (2 + 8) / 2
    assert out["points_per_gw"]["top_p95"] == 8.0          # (15 + 1) / 2
    assert out["points_per_gw"]["ep_plus_half_sd"] == 8.0  # 5.5 + 2.5 beats 6.0 + 1.0
    assert out["vs_top_ep"]["top_p95"]["mean"] == 3.0
    assert out["hindsight_points_per_gw"] == 11.5          # (15 + 8) / 2


def test_captain_rule_points_skips_gameweeks_with_missing_data():
    assert wfs.captain_rule_points([[(None, 1.0, 2.0, 3.0)]]) == {"n": 0}
