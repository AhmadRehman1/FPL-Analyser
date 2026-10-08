"""scripts/compare_arms.py: the breakout promotion rule (docs/plans/2026-10_breakout_players.md, R4)."""

import copy
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import compare_arms as ca  # noqa: E402

S = "2025-2026"


def _run(beats_real, mae=1.08, resid=0.60, n=300, widened_n=400, key="duckdb-1"):
    return {
        "progress": {"complete": True},
        "db_cache_key": key,
        "headline_by_season": {S: {"ep_total_calibration_mae": mae}},
        "per_gameweek": [{"season": S, "gw": gw, "beats_real": v} for gw, v in enumerate(beats_real, start=1)]
        + [{"season": "2024-2025", "gw": 5, "beats_real": None}],
        "breakout": {S: {
            "breakout": {"n_player_steps": n, "mean_resid": resid, "mae": 2.0},
            "breakout_widened": {"n_player_steps": widened_n, "mean_resid": resid, "mae": 2.0},
        }, "2024-2025": None},
    }


CONTROL = [10.0, 12.0, 8.0, 11.0]


def test_pass_when_points_level_mae_flat_and_breakout_cut_by_a_third():
    out = ca.compare(_run(CONTROL), _run([11.0, 12.0, 9.0, 11.0], mae=1.081, resid=0.39))
    assert out["verdict"] == "PASS"
    assert out["checks"]["points"]["mean"] == pytest.approx(0.5)
    assert out["checks"]["breakout"]["limit"] == pytest.approx(0.4)


@pytest.mark.parametrize("arm, failing", [
    (_run([9.0, 11.0, 7.0, 10.0], resid=0.2), "points"),         # m = -1.0
    (_run([9.9, 11.9, 7.9, 10.9], resid=0.2), "points"),         # m = -0.1 every week: m + SE < 0
    (_run(CONTROL, mae=1.0825, resid=0.2), "mae"),               # +0.0025
    (_run(CONTROL, resid=0.41), "breakout"),                     # above two thirds of 0.60
])
def test_each_failing_reason(arm, failing):
    out = ca.compare(_run(CONTROL), arm)
    assert out["verdict"] == "FAIL"
    assert [k for k, c in out["checks"].items() if not c["pass"]] == [failing]


def test_m_plus_se_rule_lets_a_noisy_small_loss_through_but_not_a_large_one():
    noisy = _run([10.5, 11.0, 8.6, 11.5], resid=0.2)  # diffs +0.5 -1.0 +0.6 +0.5: m = 0.15
    assert ca.compare(_run(CONTROL), noisy)["checks"]["points"]["pass"]
    assert not ca.compare(_run(CONTROL), _run([9.6, 11.6, 7.6, 10.6], resid=0.2))["checks"]["points"]["pass"]  # m = -0.4


def test_control_not_under_predicting_the_group_stops():
    out = ca.compare(_run(CONTROL, resid=-0.1), _run(CONTROL, resid=-0.3))
    assert out["verdict"] == "FAIL"
    assert "stop and report" in out["checks"]["breakout"]["reason"]


def test_small_group_falls_back_to_the_declared_widening_then_stops():
    widened = ca.compare(_run(CONTROL, n=150), _run(CONTROL, n=150, resid=0.3))
    assert widened["checks"]["breakout"]["group"].startswith("breakout_widened")
    both_small = ca.compare(_run(CONTROL, n=150, widened_n=180), _run(CONTROL, resid=0.3))
    assert "too small" in both_small["checks"]["breakout"]["reason"]


@pytest.mark.parametrize("mutate, message", [
    (lambda r: r["progress"].update(complete=False), "incomplete"),
    (lambda r: r.update(db_cache_key=None), "no db_cache_key"),
    (lambda r: r.update(db_cache_key="duckdb-2"), "different cached DBs"),
])
def test_refuses_runs_that_cannot_be_compared(mutate, message):
    arm = copy.deepcopy(_run(CONTROL))
    mutate(arm)
    with pytest.raises(ca.Refused, match=message):
        ca.compare(_run(CONTROL), arm)


def test_cli_exit_codes(tmp_path, capsys):
    import json
    control, arm = tmp_path / "c.json", tmp_path / "a.json"
    control.write_text(json.dumps(_run(CONTROL)))
    arm.write_text(json.dumps(_run([11.0, 12.0, 9.0, 11.0], resid=0.3)))
    assert ca.main([str(control), str(arm)]) == 0
    assert capsys.readouterr().out.startswith("PASS")
    arm.write_text(json.dumps(_run(CONTROL, key="duckdb-9")))
    assert ca.main([str(control), str(arm)]) == 2
