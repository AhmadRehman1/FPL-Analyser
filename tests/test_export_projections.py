"""The projections table (and its captain_ranking) must START at current_event + 1 -- the
first gameweek a decision made now can still affect. Before this it started AT current_event,
so once a gameweek finished (FPL's is_current stays put until the next deadline) the app's
"Top captain picks" and the Planner sheet showed an already-played gameweek for the whole
pre-deadline window.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import export_projections as ep  # noqa: E402


def test_start_gameweek_is_current_event_plus_one_from_the_cli_arg():
    assert ep.resolve_start_gameweek(["export_projections.py", "2"]) == 3
    assert ep.resolve_start_gameweek(["export_projections.py", "3"]) == 4


def test_start_gameweek_falls_back_to_live_current_event_plus_one(monkeypatch):
    monkeypatch.setattr(ep.ax, "fetch_bootstrap_static", lambda: {"events": [
        {"id": 2, "is_current": True}, {"id": 3, "is_current": False},
    ]})
    assert ep.resolve_start_gameweek(["export_projections.py"]) == 3


def test_start_gameweek_survives_a_failed_live_fetch(monkeypatch):
    def boom():
        raise RuntimeError("network blocked")
    monkeypatch.setattr(ep.ax, "fetch_bootstrap_static", boom)
    assert ep.resolve_start_gameweek(["export_projections.py"]) == ep.TARGET_GAMEWEEK + 1
