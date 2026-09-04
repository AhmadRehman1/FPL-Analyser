"""PR #137 §"Activate web push" follow-up: scripts/verify_push_setup.py -- the non-fatal
consistency check between index.html's VAPID_PUBLIC_KEY literal and the VAPID/gist repo
secrets. It must never fail the run (always exit 0) and must warn ONLY on a half-configured
state, staying silent in the repo's default (neither half set up)."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import verify_push_setup as vps  # noqa: E402

REAL_KEY = "B" + "x" * 86  # ~87 url-safe-base64 chars, like a real P-256 public key
OTHER_KEY = "B" + "y" * 86
ALL_SECRETS = {n: "set" for n in vps.REQUIRED_SECRETS}


def _html(vapid_value):
    """A minimal index.html-shaped string. None => the declaration is absent entirely."""
    decl = "" if vapid_value is None else f'const VAPID_PUBLIC_KEY = "{vapid_value}";'
    return f"<script>\nconst PUSH_SUB_KEY = 'x';\n{decl}\nfunction pushConfigured() {{}}\n</script>"


# ---- client_vapid_key --------------------------------------------------------

def test_client_vapid_key_reads_the_literal():
    assert vps.client_vapid_key(_html(REAL_KEY)) == REAL_KEY


def test_client_vapid_key_empty_string_is_empty_not_none():
    assert vps.client_vapid_key(_html("")) == ""


def test_client_vapid_key_missing_declaration_is_none():
    assert vps.client_vapid_key(_html(None)) is None


def test_client_vapid_key_single_quotes_ok():
    assert vps.client_vapid_key("const VAPID_PUBLIC_KEY = 'abc';") == "abc"


# ---- evaluate: the repo default (nothing configured) -------------------------

def test_default_state_is_not_configured_and_silent():
    state, warnings = vps.evaluate(_html(""), {})
    assert state == "not_configured"
    assert warnings == []


def test_placeholder_short_key_still_counts_as_not_configured():
    state, warnings = vps.evaluate(_html("PASTE_ME"), {})
    assert state == "not_configured"
    assert warnings == []


# ---- evaluate: half-configured (the state this script exists to surface) -----

def test_secrets_set_but_client_key_empty_warns():
    state, warnings = vps.evaluate(_html(""), ALL_SECRETS)
    assert state == "mismatch"
    assert len(warnings) == 1
    assert "still" in warnings[0] and "empty" in warnings[0]


def test_client_key_set_but_secrets_missing_lists_the_missing_names():
    state, warnings = vps.evaluate(_html(REAL_KEY), {"VAPID_SUBJECT": "set"})
    assert state == "mismatch"
    assert any(
        "VAPID_PRIVATE_KEY" in w and "PUSH_SUBSCRIPTIONS_GIST" in w and "GIST_PAT" in w
        for w in warnings
    )
    assert not any("VAPID_SUBJECT" in w for w in warnings)  # that one IS set


def test_blank_secret_value_counts_as_missing():
    env = {**ALL_SECRETS, "GIST_PAT": "   "}
    state, warnings = vps.evaluate(_html(REAL_KEY), env)
    assert state == "mismatch"
    assert any("GIST_PAT" in w for w in warnings)


def test_missing_declaration_always_warns():
    state, warnings = vps.evaluate(_html(None), {})
    assert any("no longer contains" in w for w in warnings)


# ---- evaluate: fully configured --------------------------------------------

def test_fully_configured_keys_agree_is_clean():
    env = {**ALL_SECRETS, "VAPID_PUBLIC_KEY": REAL_KEY}
    state, warnings = vps.evaluate(_html(REAL_KEY), env)
    assert state == "configured"
    assert warnings == []


def test_fully_configured_keys_disagree_warns():
    env = {**ALL_SECRETS, "VAPID_PUBLIC_KEY": OTHER_KEY}
    state, warnings = vps.evaluate(_html(REAL_KEY), env)
    assert state == "mismatch"
    assert any("does not match" in w for w in warnings)


# ---- main: exit-code contract (never fails the run) ------------------------

def test_main_returns_0_in_the_default_state(tmp_path, capsys, monkeypatch):
    for n in vps.REQUIRED_SECRETS:
        monkeypatch.delenv(n, raising=False)
    idx = tmp_path / "index.html"
    idx.write_text(_html(""), encoding="utf-8")
    assert vps.main(["verify_push_setup.py", str(idx)]) == 0
    out = capsys.readouterr().out
    assert "::warning::" not in out
    assert "not configured" in out


def test_main_returns_0_and_warns_on_mismatch(tmp_path, capsys, monkeypatch):
    for n in vps.REQUIRED_SECRETS:
        monkeypatch.setenv(n, "set")
    idx = tmp_path / "index.html"
    idx.write_text(_html(""), encoding="utf-8")
    assert vps.main(["verify_push_setup.py", str(idx)]) == 0
    assert "::warning::" in capsys.readouterr().out


def test_main_returns_0_when_index_html_unreadable(tmp_path, capsys):
    missing = tmp_path / "nope.html"
    assert vps.main(["verify_push_setup.py", str(missing)]) == 0
    assert "could not read" in capsys.readouterr().out


def test_real_repo_index_html_is_in_a_consistent_state():
    """Guard: the checked-in index.html + a no-secrets environment must not warn -- catches a
    future change that leaves a real key in the client with no CI secrets, or vice versa."""
    real_index = (REPO_ROOT / "index.html").read_text(encoding="utf-8")
    state, warnings = vps.evaluate(real_index, {})
    assert state == "not_configured"
    assert warnings == []
