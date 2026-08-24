import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import export_live_data as eld  # noqa: E402
from fpl_quant import app_export as ax  # noqa: E402


def _bootstrap():
    return {
        "events": [{"id": 5, "is_current": True}],
        "elements": [],
        "total_players": 10_000_000,
    }


def test_main_falls_back_to_stale_snapshot_on_upstream_failure(tmp_path, monkeypatch):
    """Review Feature 6 / B7: a live fetch that fails even after retry/backoff must fall back
    to the account's own last-known rank with stale=True, never raise and never fabricate a
    rank for that poll cycle."""
    monkeypatch.setattr(eld, "DASHBOARD_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["export_live_data.py", "123:Test account"])

    # Pre-seed a last-known snapshot so the fallback has something real to reuse.
    ax.append_live_rank_snapshot(
        tmp_path, 123, 5, ts="2026-08-23T14:00:00Z", overall_rank=500_000,
        mini_league_pos=None, live_points=10, stale=False, data_asof="2026-08-23",
    )

    monkeypatch.setattr(ax, "fetch_bootstrap_static", lambda: _bootstrap())
    monkeypatch.setattr(ax, "fetch_fixtures", lambda event: [{"started": True, "finished": False}])
    monkeypatch.setattr(eld.lt, "is_any_fixture_live", lambda fixtures: True)
    monkeypatch.setattr(ax, "fetch_event_live", lambda event: {"elements": []})
    monkeypatch.setattr(ax, "build_player_directory", lambda bootstrap: [])

    def _boom(entry_id, event):
        raise ax.UpstreamUnavailableError("simulated: fetch_entry_picks exhausted retries")

    monkeypatch.setattr(ax, "fetch_entry_picks", _boom)

    eld.main()  # must not raise

    payload = json.loads((tmp_path / "live_rank_123_5.json").read_text())
    assert payload["stale"] is True
    assert len(payload["snapshots"]) == 2  # append-only -- the pre-seeded row survives
    assert payload["snapshots"][0]["overall_rank"] == 500_000
    assert payload["snapshots"][1]["overall_rank"] == 500_000  # reused the last-known rank, not fabricated
    # no app_live_123.json written for this failed cycle -- the fallback path returns early
    assert not (tmp_path / "app_live_123.json").exists()


def test_main_skips_gracefully_with_no_prior_snapshot_and_upstream_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(eld, "DASHBOARD_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["export_live_data.py", "999:No history"])

    monkeypatch.setattr(ax, "fetch_bootstrap_static", lambda: _bootstrap())
    monkeypatch.setattr(ax, "fetch_fixtures", lambda event: [{"started": True, "finished": False}])
    monkeypatch.setattr(eld.lt, "is_any_fixture_live", lambda fixtures: True)
    monkeypatch.setattr(ax, "fetch_event_live", lambda event: {"elements": []})
    monkeypatch.setattr(ax, "build_player_directory", lambda bootstrap: [])

    def _boom(entry_id, event):
        raise ax.UpstreamUnavailableError("simulated failure")

    monkeypatch.setattr(ax, "fetch_entry_picks", _boom)

    eld.main()  # must not raise, even with nothing to fall back to
    assert not (tmp_path / "live_rank_999_5.json").exists()
