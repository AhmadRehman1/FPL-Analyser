"""The not-configured path of scripts/track_elite.py: it must still write an explicit feed
(status: "not_configured") so index.html can tell "elite tracking is switched off" apart from
"the feed failed to load this run". The configured path needs a real ingested DB + live FPL
fetches and is exercised by the pipeline, not here."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import track_elite as te  # noqa: E402
from fpl_quant import db as db_mod  # noqa: E402


def test_no_elite_managers_configured_writes_a_not_configured_feed(tmp_path, monkeypatch):
    dashboard = tmp_path / "dashboard"
    empty_cfg = tmp_path / "elite_managers.json"
    empty_cfg.write_text(json.dumps({"managers": []}))

    real_connect = db_mod.connect  # capture before patching -- te.db IS db_mod (same module)
    monkeypatch.setattr(te, "DASHBOARD_DIR", dashboard)
    monkeypatch.setattr(te, "ELITE_MANAGERS_PATH", empty_cfg)
    monkeypatch.setattr(te.db, "connect", lambda *a, **k: real_connect(tmp_path / "t.duckdb"))
    monkeypatch.setattr(sys, "argv", ["track_elite.py"])

    te.main()

    latest = json.loads((dashboard / "elite_divergence_latest.json").read_text())
    assert latest["status"] == "not_configured"
    assert latest["managers"] == []
    assert latest["data_asof"]
    # dated sibling too
    dated = list(dashboard.glob("elite_divergence_2*.json"))
    assert len(dated) == 1 and json.loads(dated[0].read_text())["status"] == "not_configured"
