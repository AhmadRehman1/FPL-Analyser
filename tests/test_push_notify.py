"""App gap 1: scripts/push_notify.py must NO-OP cleanly (exit 0) whenever push isn't set up --
no payload, missing secrets, or pywebpush absent -- so a scheduled run never fails over it."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import push_notify  # noqa: E402

_ALL_SECRETS = ["PUSH_PAYLOAD_JSON", "VAPID_PRIVATE_KEY", "VAPID_PUBLIC_KEY", "VAPID_SUBJECT",
                "PUSH_SUBSCRIPTIONS_GIST", "GIST_PAT"]


def _clear(monkeypatch):
    for k in _ALL_SECRETS:
        monkeypatch.delenv(k, raising=False)


def test_no_payload_exits_zero(monkeypatch):
    _clear(monkeypatch)
    assert push_notify.main() == 0


def test_payload_but_no_secrets_exits_zero_without_raising(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("PUSH_PAYLOAD_JSON", '{"title": "x", "body": "y", "url": "https://z", "tag": "t"}')
    assert push_notify.main() == 0


def test_malformed_payload_is_a_loud_failure(monkeypatch, capsys):
    _clear(monkeypatch)
    monkeypatch.setenv("PUSH_PAYLOAD_JSON", "{not json")
    assert push_notify.main() == 1
    assert "::error::" in capsys.readouterr().out


def test_partial_secrets_still_no_ops(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("PUSH_PAYLOAD_JSON", '{"title": "x", "body": "y"}')
    monkeypatch.setenv("VAPID_PUBLIC_KEY", "pub")
    monkeypatch.setenv("VAPID_PRIVATE_KEY", "priv")
    # VAPID_SUBJECT / gist / PAT still missing
    assert push_notify.main() == 0
