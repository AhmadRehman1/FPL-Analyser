import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from check_deadline_alerts import build_alert_body  # noqa: E402


def test_build_alert_body_none_when_no_previous_snapshot():
    diff = {"has_previous": False}
    assert build_alert_body(diff) is None


def test_build_alert_body_none_when_nothing_newly_doubtful():
    diff = {"has_previous": True, "newly_doubtful_flags": []}
    assert build_alert_body(diff) is None


def test_build_alert_body_present_when_newly_doubtful():
    diff = {
        "has_previous": True, "previous_gameweek": 5, "current_gameweek": 6,
        "newly_doubtful_flags": ["nailed_attacking_return"],
    }
    body = build_alert_body(diff)
    assert body is not None
    assert "nailed_attacking_return" in body
    assert "GW5" in body and "GW6" in body
