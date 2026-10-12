import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import export_leaderboard as el  # noqa: E402


@pytest.fixture
def dashboard(tmp_path, monkeypatch):
    d = tmp_path / "dashboard"
    monkeypatch.setattr(el, "DASHBOARD_DIR", d)
    return d


# ---- _resolve_end_gameweek ---------------------------------------------------

def test_resolve_end_gameweek_uses_last_finished_gameweek(monkeypatch):
    monkeypatch.setattr(el.ax, "fetch_bootstrap_static", lambda: {"events": [
        {"id": 1, "finished": True}, {"id": 2, "finished": True}, {"id": 3, "finished": False},
    ]})
    assert el._resolve_end_gameweek() == 2


def test_resolve_end_gameweek_falls_back_when_the_live_fetch_fails(monkeypatch):
    def boom():
        raise RuntimeError("network blocked")
    monkeypatch.setattr(el.ax, "fetch_bootstrap_static", boom)
    assert el._resolve_end_gameweek() == el.END_GAMEWEEK_FALLBACK


def test_resolve_end_gameweek_falls_back_when_no_gameweek_finished_yet(monkeypatch):
    monkeypatch.setattr(el.ax, "fetch_bootstrap_static", lambda: {"events": [{"id": 1, "finished": False}]})
    assert el._resolve_end_gameweek() == el.END_GAMEWEEK_FALLBACK


# ---- main(): the insufficient-data short circuit ----------------------------

def test_main_writes_insufficient_data_payload_without_touching_the_db(dashboard, monkeypatch):
    # GW2 -> GW2 is a 1-gameweek span, below MIN_GAMEWEEK_SPAN -- must NOT call db.connect().
    monkeypatch.setattr(el.db, "connect", lambda *a, **k: pytest.fail("db.connect() called on the insufficient-data path"))
    monkeypatch.setattr(sys, "argv", ["export_leaderboard.py", "2", "2"])

    el.main()

    latest = json.loads((dashboard / "leaderboard_latest.json").read_text())
    assert latest["status"] == "insufficient_data"
    assert latest["start_gameweek"] == 2 and latest["end_gameweek"] == 2
    assert latest["completed_gameweeks"] == 1
    assert latest["min_gameweeks"] == el.MIN_GAMEWEEK_SPAN
    assert latest["rows"] == []
    # dated sibling written too
    dated = list(dashboard.glob("leaderboard_2*.json"))
    assert len(dated) == 1
    assert json.loads(dated[0].read_text())["status"] == "insufficient_data"


def test_main_insufficient_data_payload_carries_a_data_asof(dashboard, monkeypatch):
    monkeypatch.setattr(el.db, "connect", lambda *a, **k: pytest.fail("db.connect() should not be called"))
    monkeypatch.setattr(sys, "argv", ["export_leaderboard.py", "2", "3"])  # span 2, still < 3
    el.main()
    latest = json.loads((dashboard / "leaderboard_latest.json").read_text())
    assert latest["status"] == "insufficient_data"
    assert latest["data_asof"]  # non-empty ISO date string


# ---- the like-for-like rows (docs/plans/2026-10_beat_the_template.md, Req 3) ----

MODEL = {2: 50.0, 3: 60.0, 4: 40.0}
BASELINES = {
    "recent_points": {2: 45.0, 3: 55.0, 4: 50.0},
    "ownership_popularity": {2: 52.0, 4: 48.0},  # GW3 skipped
    "crowd": {2: 60.0, 3: 62.0, 4: 58.0},
}
ORACLE = {"model_oracle": {2: 58.0, 3: 66.0, 4: 51.0}, "template_gated": {2: 55.0, 3: 61.0, 4: 49.0}}


def test_rows_keep_the_evolving_model_first_and_append_the_oracle_rows_last():
    rows = el.build_rows(MODEL, BASELINES, ORACLE, window_gameweeks=3)
    assert [r["name"] for r in rows] == [
        "FPL-Analyser (model)", "recent-points baseline", "price/ownership baseline",
        "synthetic EO crowd (not FPL's average)", "model (oracle)", "ownership template (available)",
    ]
    assert [r["is_baseline"] for r in rows] == [False, True, True, True, True, True]
    # the app takes the first non-baseline row as the model
    assert next(r for r in rows if not r["is_baseline"])["name"] == "FPL-Analyser (model)"
    assert [r.get("kind") for r in rows[4:]] == ["model_oracle", "template_gated"]


def test_every_baseline_row_carries_numeric_totals_and_gaps():
    rows = el.build_rows(MODEL, BASELINES, ORACLE, window_gameweeks=3)
    for r in rows[1:]:
        for key in ("total_points", "n_gameweeks", "n_gameweeks_skipped", "model_beats_baseline_by"):
            assert isinstance(r[key], (int, float)), (r["name"], key)
    by_kind = {r.get("kind"): r for r in rows}
    # model_beats_baseline_by keeps its meaning: the evolving model minus the row
    assert by_kind["model_oracle"]["model_beats_baseline_by"] == pytest.approx(150.0 - 175.0)
    assert by_kind["template_gated"]["model_beats_baseline_by"] == pytest.approx(150.0 - 165.0)
    assert by_kind["ownership_popularity"]["model_beats_baseline_by"] == pytest.approx(90.0 - 100.0)  # GW2 and GW4
    assert by_kind["ownership_popularity"]["n_gameweeks_skipped"] == 1
    # the oracle model against each oracle row, like for like
    assert by_kind["template_gated"]["oracle_model_beats_baseline_by"] == pytest.approx(175.0 - 165.0)
    assert by_kind["crowd"]["oracle_model_beats_baseline_by"] == pytest.approx(175.0 - 180.0)
    assert by_kind["ownership_popularity"]["oracle_model_beats_baseline_by"] == pytest.approx(109.0 - 100.0)
    assert "oracle_model_beats_baseline_by" not in by_kind["model_oracle"]


def test_rows_carry_their_weekly_points_and_the_models_hits():
    rows = el.build_rows(MODEL, BASELINES, ORACLE, window_gameweeks=3, model_hits={3: 4.0})
    assert rows[0]["weekly_points"] == {"2": 50.0, "3": 60.0, "4": 40.0}
    assert rows[0]["weekly_hits"] == {"3": 4.0}
    assert rows[2]["weekly_points"] == {"2": 52.0, "4": 48.0}
    assert rows[4]["weekly_points"] == {"2": 58.0, "3": 66.0, "4": 51.0}


def test_rows_survive_an_oracle_row_with_no_gameweeks():
    rows = el.build_rows(MODEL, BASELINES, {"model_oracle": {}, "template_gated": {}}, window_gameweeks=3)
    oracle_row = next(r for r in rows if r.get("kind") == "model_oracle")
    assert oracle_row["total_points"] == 0 and oracle_row["n_gameweeks_skipped"] == 3
    assert oracle_row["model_beats_baseline_by"] == 0


def test_oracle_weeks_scores_the_walk_forward_step_and_the_gated_template(con):
    from tests.test_backtest import _SEASON_SIM_VERSIONS, _seed_season_simulation_league

    _seed_season_simulation_league(con)
    con.execute("UPDATE fact_player_season_stats SET selected_by_percent = "
                "10.0 + now_cost * 3 + CAST(substr(player_uid, 2) AS INTEGER) * 0.1")
    out = el.oracle_weeks(con, dict(_SEASON_SIM_VERSIONS), "2025-2026", 2, 3)
    assert set(out["model_oracle"]) == {2, 3} and set(out["template_gated"]) == {2, 3}
    assert all(v >= 0 for v in out["model_oracle"].values())
    notes = con.execute("SELECT notes FROM backtest_runs").fetchall()
    assert notes == [(el.ORACLE_RUN_NOTES,)]
