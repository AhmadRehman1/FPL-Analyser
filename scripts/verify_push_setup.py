"""App gap 1 follow-up (PR #137 §"Activate web push"): a one-time, non-fatal consistency check
between the *client* half of Web Push (the `VAPID_PUBLIC_KEY` literal pasted into index.html)
and the *server* half (the VAPID / gist repo secrets scheduled_pipeline.yml passes to
scripts/push_notify.py).

Web Push only works when BOTH halves are in place:
  - index.html ships a real `const VAPID_PUBLIC_KEY = "B..."` so the browser can `subscribe()`;
  - the repo has VAPID_PRIVATE_KEY / VAPID_PUBLIC_KEY / VAPID_SUBJECT / PUSH_SUBSCRIPTIONS_GIST
    / GIST_PAT so the pipeline can actually send.

Either half alone is silently inert: a client key with no secrets means `subscribe()` succeeds
but every send no-ops; secrets with an empty client key means the pipeline is ready to send but
no device can ever register. Before this script, that half-configured state produced no signal
at all -- push just quietly didn't work. This runs in the pipeline and logs a GitHub
`::warning::` describing exactly which half is missing, so the repo owner gets a nudge instead
of silent inertness. It NEVER fails the run (always exit 0): a push-config mismatch is an
operator to-do, not a pipeline error.

It does NOT print secret values -- only whether each expected name is non-empty in the
environment (the workflow maps `secrets.*` -> env for exactly the five names push_notify.py
reads).

Usage (from scheduled_pipeline.yml, with the VAPID/gist secrets mapped into env):
    PYTHONPATH=src python scripts/verify_push_setup.py [path/to/index.html]
"""

import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = REPO_ROOT / "index.html"

# The exact set scripts/push_notify.py requires before it will attempt a send (see its module
# docstring) -- keep in lockstep with that list.
REQUIRED_SECRETS = (
    "VAPID_PRIVATE_KEY",
    "VAPID_PUBLIC_KEY",
    "VAPID_SUBJECT",
    "PUSH_SUBSCRIPTIONS_GIST",
    "GIST_PAT",
)

# A real VAPID P-256 public key is ~87 url-safe-base64 chars; anything this short is a
# placeholder. Mirrors index.html's own pushConfigured() length gate (> 20).
_MIN_REAL_KEY_LEN = 20

_VAPID_LITERAL_RE = re.compile(
    r"""const\s+VAPID_PUBLIC_KEY\s*=\s*(["'])(?P<key>.*?)\1""",
    re.DOTALL,
)


def client_vapid_key(index_html_text: str) -> str | None:
    """The `VAPID_PUBLIC_KEY` string literal from index.html, or None if the declaration isn't
    found at all (a structural change worth its own warning)."""
    m = _VAPID_LITERAL_RE.search(index_html_text)
    if m is None:
        return None
    return m.group("key").strip()


def _present(env: Mapping[str, str], name: str) -> bool:
    return bool((env.get(name) or "").strip())


def evaluate(index_html_text: str, env: Mapping[str, str]) -> tuple[str, list[str]]:
    """Returns (state, warnings). state is one of:
        not_configured  -- neither half set up (the repo default; no warning)
        configured      -- both halves present and the two public keys agree
        mismatch        -- exactly one half set up, or the two public keys disagree
    warnings is a list of human-readable ::warning:: lines (empty unless state == "mismatch").
    """
    client_key = client_vapid_key(index_html_text)
    client_ready = client_key is not None and len(client_key) >= _MIN_REAL_KEY_LEN

    missing = [n for n in REQUIRED_SECRETS if not _present(env, n)]
    secrets_ready = not missing

    warnings: list[str] = []

    if client_key is None:
        warnings.append(
            "index.html no longer contains a `const VAPID_PUBLIC_KEY = \"...\"` declaration -- "
            "the client cannot register for Web Push. See docs/PUSH_SETUP.md step 3."
        )

    if not client_ready and not secrets_ready:
        # The clean, expected default: push simply isn't set up. Nothing to warn about.
        return "not_configured", warnings

    if client_ready and secrets_ready:
        secret_key = (env.get("VAPID_PUBLIC_KEY") or "").strip()
        if secret_key and client_key != secret_key:
            warnings.append(
                "the VAPID public key in index.html does not match the VAPID_PUBLIC_KEY repo "
                "secret -- the push service will reject subscriptions signed with the other key. "
                "Re-paste the same public key in both places (docs/PUSH_SETUP.md steps 1 & 3)."
            )
            return "mismatch", warnings
        return "configured", warnings

    # Exactly one half is ready -> half-configured, the state this script exists to surface.
    if secrets_ready and not client_ready:
        warnings.append(
            "the Web Push repo secrets are configured but index.html's VAPID_PUBLIC_KEY is still "
            "empty/placeholder -- no device can subscribe, so every send will no-op. Paste the "
            "public key into index.html (docs/PUSH_SETUP.md step 3) and redeploy the shell."
        )
    elif client_ready and not secrets_ready:
        warnings.append(
            "index.html ships a real VAPID_PUBLIC_KEY but these repo secrets are missing: "
            f"{', '.join(missing)} -- devices can subscribe but the pipeline cannot send. "
            "Add the secrets (docs/PUSH_SETUP.md steps 1 & 2)."
        )
    return "mismatch", warnings


def main(argv: list[str]) -> int:
    index_path = Path(argv[1]) if len(argv) > 1 else INDEX_HTML
    try:
        text = index_path.read_text(encoding="utf-8")
    except OSError as e:
        print(f"::warning::verify_push_setup: could not read {index_path} ({e}) -- skipping check")
        return 0

    state, warnings = evaluate(text, os.environ)

    for w in warnings:
        print(f"::warning::verify_push_setup: {w}")

    if state == "not_configured":
        print("[verify_push_setup] Web Push not configured (neither client key nor repo secrets) "
              "-- expected default, nothing to do. See docs/PUSH_SETUP.md to enable.")
    elif state == "configured":
        print("[verify_push_setup] Web Push fully configured -- client key and repo secrets agree.")
    else:
        print("[verify_push_setup] Web Push is half-configured -- see the ::warning:: above.")

    # Never fail the run: a push-config mismatch is an operator to-do, not a pipeline error.
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
