import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from check_deadline_alerts import _next_deadline_utc, build_alert_body, compute_push_payload  # noqa: E402

NOW = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)


# ---- channel 1: the model-report GitHub Issue (unchanged) -------------------

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


# ---- channel 2: held-player push alerts (app gap 1) ------------------------

def test_next_deadline_picks_the_soonest_future_gameweek():
    fixtures = {"gameweeks": [
        {"gameweek": 1, "deadline_time": "2026-08-21T17:30:00Z"},
        {"gameweek": 3, "deadline_time": "2026-09-01T14:00:00Z"},
        {"gameweek": 4, "deadline_time": "2026-09-08T17:30:00Z"},
    ]}
    assert _next_deadline_utc(fixtures, NOW) == datetime(2026, 9, 1, 14, 0, tzinfo=timezone.utc)
    assert _next_deadline_utc({"gameweeks": []}, NOW) is None
    assert _next_deadline_utc(None, NOW) is None


def _write_account_fixtures(d: Path, *, captain_status="a", deadline_hours_out=2.0):
    (d / "app_players.json").write_text(json.dumps({"players": [
        {"id": 2, "web_name": "Haaland", "status": captain_status, "chance_of_playing_next_round": None, "news": "Knock" if captain_status != "a" else None},
    ]}))
    (d / "app_fixtures.json").write_text(json.dumps({"gameweeks": [
        {"gameweek": 3, "deadline_time": (NOW + timedelta(hours=deadline_hours_out)).strftime("%Y-%m-%dT%H:%M:%SZ")},
    ]}))
    (d / "app_price_watch.json").write_text(json.dumps({"risers": [], "fallers": []}))
    for eid in (7139944, 1305242):
        (d / f"app_team_{eid}.json").write_text(json.dumps({"entry_id": eid, "squad": [
            {"player_id": 2, "web_name": "Haaland", "is_captain": True, "is_vice_captain": False},
        ]}))
        (d / f"real_squad_{eid}.json").write_text(json.dumps({"hold_vs_transfer_now": {"recommended_action": "hold"}, "chip_evaluations": []}))


def test_compute_push_payload_none_when_all_quiet(tmp_path):
    _write_account_fixtures(tmp_path, captain_status="a", deadline_hours_out=48)
    assert compute_push_payload(tmp_path, NOW, 3.0) is None


def test_compute_push_payload_fires_for_an_injured_captain(tmp_path):
    _write_account_fixtures(tmp_path, captain_status="i", deadline_hours_out=48)
    payload = compute_push_payload(tmp_path, NOW, 3.0)
    assert payload is not None
    assert "Captain doubt: Haaland" in payload["title"]
    # both tracked accounts hold Haaland as captain -> 2 alerts, one shown + "(+1 more)"
    assert "more alert" in payload["body"]


def test_compute_push_payload_survives_a_completely_empty_dashboard_dir(tmp_path):
    # the "first scheduled run, nothing ingested yet" shape -- must not raise
    assert compute_push_payload(tmp_path, NOW, 3.0) is None
