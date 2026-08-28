"""research/ml/ollama_client.py -- unit tests using a fake `requests` module. No real Ollama
server or network access is used or required; this only tests the client's own request-building
and error-handling logic."""

from __future__ import annotations

import pytest
import requests

from research.ml.ollama_client import OllamaUnavailableError, chat, is_reachable


class _FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json_data = json_data or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self):
        return self._json_data


def test_is_reachable_true_on_200(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(200))
    assert is_reachable() is True


def test_is_reachable_false_on_connection_error(monkeypatch):
    def _raise(*a, **k):
        raise requests.ConnectionError("refused")
    monkeypatch.setattr(requests, "get", _raise)
    assert is_reachable() is False


def test_chat_raises_actionable_error_when_server_unreachable(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(500))
    with pytest.raises(OllamaUnavailableError, match="ollama serve"):
        chat("system", "user")


def test_chat_returns_message_content_on_success(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(200))
    monkeypatch.setattr(
        requests, "post",
        lambda *a, **k: _FakeResponse(200, {"message": {"content": "a plain-English summary"}}),
    )
    result = chat("system", "user", model="llama3.1")
    assert result.text == "a plain-English summary"
    assert result.model == "llama3.1"


def test_chat_raises_when_response_json_has_no_content(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(200))
    monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(200, {"message": {}}))
    with pytest.raises(OllamaUnavailableError, match="empty response"):
        chat("system", "user")


def test_chat_raises_when_post_itself_fails(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(200))

    def _raise(*a, **k):
        raise requests.Timeout("timed out")
    monkeypatch.setattr(requests, "post", _raise)
    with pytest.raises(OllamaUnavailableError, match="ollama pull"):
        chat("system", "user")
