import json
import sys
from datetime import date, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from verify_dashboard_feeds import MAX_ASOF_AGE_DAYS, main, verify_feed  # noqa: E402

TODAY = date(2026, 9, 1)


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload))
    return path


def _fresh_projections(**overrides) -> dict:
    payload = {
        "data_asof": TODAY.isoformat(),
        "gameweeks": [3, 4, 5],
        "players": [{"player_uid": "p1", "ep": 4.2}],
    }
    payload.update(overrides)
    return payload


def _fresh_elite(**overrides) -> dict:
    payload = {"data_asof": TODAY.isoformat(), "status": "not_configured", "managers": []}
    payload.update(overrides)
    return payload


# ---- verify_feed ---------------------------------------------------------------

def test_healthy_projections_feed_has_no_problems(tmp_path):
    path = _write(tmp_path / "projections_latest.json", _fresh_projections())
    assert verify_feed(path, kind="projections", today=TODAY) == []


def test_healthy_elite_feed_not_configured_is_ok(tmp_path):
    path = _write(tmp_path / "elite_divergence_latest.json", _fresh_elite())
    assert verify_feed(path, kind="elite", today=TODAY) == []


def test_healthy_elite_feed_ok_status_is_ok(tmp_path):
    path = _write(tmp_path / "elite_divergence_latest.json", _fresh_elite(status="ok", managers=[{"name": "x"}]))
    assert verify_feed(path, kind="elite", today=TODAY) == []


def test_missing_feed_is_a_problem(tmp_path):
    problems = verify_feed(tmp_path / "projections_latest.json", kind="projections", today=TODAY)
    assert len(problems) == 1 and "missing" in problems[0]


def test_unparseable_feed_is_a_problem(tmp_path):
    path = tmp_path / "projections_latest.json"
    path.write_text("{not json")
    problems = verify_feed(path, kind="projections", today=TODAY)
    assert len(problems) == 1 and "unparseable" in problems[0].lower()


def test_stale_data_asof_is_a_problem(tmp_path):
    stale = (TODAY - timedelta(days=MAX_ASOF_AGE_DAYS + 1)).isoformat()
    path = _write(tmp_path / "projections_latest.json", _fresh_projections(data_asof=stale))
    problems = verify_feed(path, kind="projections", today=TODAY)
    assert any("stale" in p for p in problems)


def test_asof_within_grace_window_is_ok(tmp_path):
    ok_edge = (TODAY - timedelta(days=MAX_ASOF_AGE_DAYS)).isoformat()
    path = _write(tmp_path / "projections_latest.json", _fresh_projections(data_asof=ok_edge))
    assert verify_feed(path, kind="projections", today=TODAY) == []


def test_projections_with_no_players_is_a_problem(tmp_path):
    path = _write(tmp_path / "projections_latest.json", _fresh_projections(players=[]))
    problems = verify_feed(path, kind="projections", today=TODAY)
    assert any("no player rows" in p for p in problems)


def test_elite_with_unexpected_status_is_a_problem(tmp_path):
    path = _write(tmp_path / "elite_divergence_latest.json", _fresh_elite(status="weird"))
    problems = verify_feed(path, kind="elite", today=TODAY)
    assert any("unexpected status" in p for p in problems)


def test_missing_data_asof_is_a_problem(tmp_path):
    payload = _fresh_projections()
    del payload["data_asof"]
    path = _write(tmp_path / "projections_latest.json", payload)
    problems = verify_feed(path, kind="projections", today=TODAY)
    assert any("no data_asof" in p for p in problems)


# ---- main (exit code contract) ------------------------------------------------

def test_main_returns_0_when_both_feeds_healthy(tmp_path, capsys):
    _write(tmp_path / "projections_latest.json", _fresh_projections())
    _write(tmp_path / "elite_divergence_latest.json", _fresh_elite())
    # verify_feed inside main() uses the real date.today(); write today's date so it passes.
    (tmp_path / "projections_latest.json").write_text(json.dumps(_fresh_projections(data_asof=date.today().isoformat())))
    (tmp_path / "elite_divergence_latest.json").write_text(json.dumps(_fresh_elite(data_asof=date.today().isoformat())))
    assert main(["verify_dashboard_feeds.py", str(tmp_path)]) == 0


def test_main_returns_1_and_prints_error_when_a_feed_is_missing(tmp_path, capsys):
    (tmp_path / "projections_latest.json").write_text(json.dumps(_fresh_projections(data_asof=date.today().isoformat())))
    # elite feed absent
    assert main(["verify_dashboard_feeds.py", str(tmp_path)]) == 1
    assert "::error::" in capsys.readouterr().out
