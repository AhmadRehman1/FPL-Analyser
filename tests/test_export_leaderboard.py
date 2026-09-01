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
