# Web Push setup (app gap 1)

Closed-app push notifications for deadline / injury / price alerts. All of this is $0 and
consistent with the project's free-tier design — the subscription store is a **secret GitHub
Gist**, sends go out from the existing scheduled Actions run via `pywebpush`.

Until these steps are done the app still works: it does **local** notifications (fired while
the app is open or backgrounded), and `scripts/push_notify.py` no-ops cleanly on every
scheduled run. Nothing is broken by leaving push unconfigured.

**iOS note:** Safari only delivers Web Push to a PWA that has been **Added to Home Screen**
(iOS 16.4+). On desktop / Android Chrome it works from the browser tab.

---

## 1. Generate a VAPID keypair (once)

```bash
pip install pywebpush py-vapid
vapid --gen                 # writes private_key.pem / public_key.pem
vapid --applicationServerKey   # prints the URL-safe base64 public key for the browser
```

or, in Node:

```bash
npx web-push generate-vapid-keys
```

You need three values:

| value | where it goes |
|---|---|
| **public key** (URL-safe base64, ~87 chars) | `index.html` — set `const VAPID_PUBLIC_KEY = "..."` (it is public, safe to commit) **and** repo secret `VAPID_PUBLIC_KEY` |
| **private key** | repo secret `VAPID_PRIVATE_KEY` (the PEM contents, or the base64 form `pywebpush` accepts) |
| a contact URL, e.g. `mailto:you@example.com` | repo secret `VAPID_SUBJECT` |

## 2. Create the subscriptions gist (once)

- Create a **secret** gist (gist.github.com → "Create secret gist") with one file
  `subscriptions.json` containing exactly `[]`.
- Copy the gist id from its URL (`https://gist.github.com/<user>/<THIS>`).
- Create a **fine-grained PAT** (github.com → Settings → Developer settings → Tokens) with
  **Gists: read and write** and nothing else. (A classic token with the `gist` scope also
  works.)

Add repo secrets (Settings → Secrets and variables → Actions):

| secret | value |
|---|---|
| `PUSH_SUBSCRIPTIONS_GIST` | the gist id |
| `GIST_PAT` | the PAT |
| `VAPID_PUBLIC_KEY` / `VAPID_PRIVATE_KEY` / `VAPID_SUBJECT` | from step 1 |

## 3. Paste the public key into the client

In `index.html`:

```js
const VAPID_PUBLIC_KEY = "BPa...your...key...";
```

Commit + push — `deploy_pages.yml` redeploys the shell.

## 4. Register a device

In the app, tap the bell → **allow notifications**. The app opens a pre-filled
`[push-subscribe]` GitHub issue; submit it. `push_subscribe.yml` parses the subscription into
the gist and closes the issue with a confirmation. The bell turns **gold** once a device is
linked. Repeat on each device you want alerts on.

## How a send happens

`scheduled_pipeline.yml` runs `check_deadline_alerts.py` twice a day. For each tracked account
it computes held-player alerts (`src/fpl_quant/push_alerts.py`): captain/vice flagged doubtful,
a price move on a squad player, or — within `PUSH_ALERT_LEAD_HOURS` (default 3) of the deadline
— an unconfirmed pending transfer/chip recommendation. If anything fires, the run sends **one**
Web Push (highest-priority alert, "+N more") to every subscription in the gist and prunes any
the push service reports as expired.

## Setup consistency check

`scheduled_pipeline.yml` also runs `scripts/verify_push_setup.py` early in every run. It's
stdlib-only, never fails the run, and stays silent while push is unconfigured (the default). It
logs a GitHub `::warning::` only for a **half-configured** state — the VAPID/gist secrets are
set but `index.html`'s `VAPID_PUBLIC_KEY` is still empty (or vice versa), or the client key and
the `VAPID_PUBLIC_KEY` secret disagree — because that state is otherwise silently inert. If you
do steps 1–3 above and see no warning on the next scheduled run, both halves match.
