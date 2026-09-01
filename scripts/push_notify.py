"""App gap 1: send the deadline / injury Web Push the scheduled pipeline decided is warranted.

Environment (all optional -- this script NO-OPS cleanly, exit 0, when any of them is absent, so
a scheduled run never fails because push isn't set up yet):

  PUSH_PAYLOAD_JSON        the notification JSON scripts/check_deadline_alerts.py emitted
                           (scripts/push_alerts.build_push_payload() shape). Empty/"" => nothing
                           to send.
  VAPID_PRIVATE_KEY        repo secret -- the private half of the VAPID keypair
  VAPID_PUBLIC_KEY         repo secret -- the public half (also pasted into index.html)
  VAPID_SUBJECT            "mailto:you@example.com" (push services require a contact)
  PUSH_SUBSCRIPTIONS_GIST  id of the SECRET gist holding subscriptions.json
                           ([{endpoint, keys:{p256dh, auth}}, ...])
  GIST_PAT                 a PAT with the `gist` scope, to read + rewrite that gist

On send it prunes any subscription the push service reports as gone (404/410) and writes the
trimmed list back to the gist. See docs/PUSH_SETUP.md for the one-time setup.

Usage:  PYTHONPATH=src python scripts/push_notify.py
"""

import json
import os
import sys

GIST_FILENAME = "subscriptions.json"


def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def _load_subscriptions(gist_id: str, pat: str) -> tuple[list[dict], object]:
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        f"https://api.github.com/gists/{gist_id}",
        headers={"Authorization": f"Bearer {pat}", "Accept": "application/vnd.github+json",
                 "User-Agent": "fpl-quant-push"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            gist = json.load(resp)
    except urllib.error.HTTPError as e:
        print(f"::warning::push_notify: could not read subscriptions gist ({e.code}) -- skipping send")
        return [], None
    files = gist.get("files") or {}
    entry = files.get(GIST_FILENAME)
    if not entry:
        print(f"::warning::push_notify: gist has no {GIST_FILENAME} yet -- no subscribers")
        return [], gist
    try:
        subs = json.loads(entry.get("content") or "[]")
    except json.JSONDecodeError:
        print(f"::warning::push_notify: {GIST_FILENAME} is not valid JSON -- skipping send")
        return [], gist
    return [s for s in subs if isinstance(s, dict) and s.get("endpoint")], gist


def _rewrite_gist(gist_id: str, pat: str, subs: list[dict]) -> None:
    import urllib.request

    body = json.dumps({"files": {GIST_FILENAME: {"content": json.dumps(subs, indent=2)}}}).encode()
    req = urllib.request.Request(
        f"https://api.github.com/gists/{gist_id}", data=body, method="PATCH",
        headers={"Authorization": f"Bearer {pat}", "Accept": "application/vnd.github+json",
                 "User-Agent": "fpl-quant-push", "Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=30).read()
        print(f"[push_notify] pruned subscriptions gist -> {len(subs)} live")
    except Exception as e:  # noqa: BLE001 -- best effort: a failed prune must not fail the run
        print(f"::warning::push_notify: could not rewrite subscriptions gist ({e})")


def main() -> int:
    raw_payload = _env("PUSH_PAYLOAD_JSON")
    if not raw_payload:
        print("[push_notify] no PUSH_PAYLOAD_JSON -- nothing to send")
        return 0
    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError as e:
        print(f"::error::push_notify: PUSH_PAYLOAD_JSON is not valid JSON: {e}")
        return 1

    vapid_priv, vapid_pub, subject = _env("VAPID_PRIVATE_KEY"), _env("VAPID_PUBLIC_KEY"), _env("VAPID_SUBJECT")
    gist_id, pat = _env("PUSH_SUBSCRIPTIONS_GIST"), _env("GIST_PAT")
    missing = [n for n, v in [
        ("VAPID_PRIVATE_KEY", vapid_priv), ("VAPID_PUBLIC_KEY", vapid_pub), ("VAPID_SUBJECT", subject),
        ("PUSH_SUBSCRIPTIONS_GIST", gist_id), ("GIST_PAT", pat),
    ] if not v]
    if missing:
        print(f"[push_notify] push not configured (missing: {', '.join(missing)}) -- "
              f"the GitHub Issue alert still fired. See docs/PUSH_SETUP.md.")
        return 0

    try:
        from pywebpush import WebPushException, webpush
    except ImportError:
        print("::warning::push_notify: pywebpush not installed -- skipping send")
        return 0

    subs, _gist = _load_subscriptions(gist_id, pat)
    if not subs:
        print("[push_notify] no subscribers -- nothing to send")
        return 0

    data = json.dumps({k: payload[k] for k in ("title", "body", "url", "tag") if k in payload})
    vapid_claims = {"sub": subject}
    live: list[dict] = []
    sent = 0
    for sub in subs:
        try:
            webpush(subscription_info=sub, data=data, vapid_private_key=vapid_priv, vapid_claims=dict(vapid_claims))
            sent += 1
            live.append(sub)
        except WebPushException as e:
            status = getattr(e.response, "status_code", None)
            if status in (404, 410):
                print(f"[push_notify] dropping expired subscription ({status})")
            else:
                print(f"::warning::push_notify: send failed ({status}) -- keeping subscription: {e}")
                live.append(sub)
        except Exception as e:  # noqa: BLE001
            print(f"::warning::push_notify: unexpected send error -- keeping subscription: {e}")
            live.append(sub)

    print(f"[push_notify] sent {sent}/{len(subs)}")
    if len(live) != len(subs):
        _rewrite_gist(gist_id, pat, live)
    return 0


if __name__ == "__main__":
    sys.exit(main())
