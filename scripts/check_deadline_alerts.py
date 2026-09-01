"""Priority 8b + app gap 1: decides whether a deadline alert is warranted and, if so, what to
say. Two independent channels, both emitted to GITHUB_OUTPUT for the calling workflow:

  has_alert / alert_body   -- the ORIGINAL channel: a nailed starter in the MODEL's own
                              from-scratch squad newly flagged doubtful (reads run_report.py's
                              latest_diff.json). Opens a GitHub Issue.

  push_payload             -- NEW: the held-player alerts the app-feature-gaps prompt asks for,
                              computed off the tracked accounts' own data/dashboard/*.json via
                              src/fpl_quant/push_alerts.py -- captain/vice doubtful, a price
                              change on a squad player, and (inside a configurable lead time)
                              an unconfirmed pending transfer/chip recommendation. Sent as a
                              real Web Push by scripts/push_notify.py. Empty string when there
                              is nothing to push.

Kept as a standalone script (not baked into run_report.py) so it stays usable/testable outside
any particular CI product.

Usage: python scripts/check_deadline_alerts.py [report_history_dir]
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import push_alerts  # noqa: E402

DASHBOARD_DIR = REPO_ROOT / "data" / "dashboard"
# The two recurring accounts scheduled_pipeline.yml tracks -- same list its planner step uses.
TRACKED_ENTRY_IDS = (7139944, 1305242)
PUSH_LEAD_HOURS = float(os.environ.get("PUSH_ALERT_LEAD_HOURS", push_alerts.DEFAULT_LEAD_HOURS))


def build_alert_body(diff: dict) -> str | None:
    """None when there's nothing alert-worthy -- a real absence, not an empty-string sentinel
    a caller could mistake for "alert with no content"."""
    if not diff.get("has_previous"):
        return None
    flags = diff.get("newly_doubtful_flags") or []
    if not flags:
        return None
    return (
        f"**Newly doubtful starters** (GW{diff['previous_gameweek']} -> GW{diff['current_gameweek']}):\n\n"
        + "\n".join(f"- `{f}`" for f in flags)
        + "\n\nSee the full report for which players triggered this."
    )


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _next_deadline_utc(fixtures: dict | None, now: datetime) -> datetime | None:
    """The soonest gameweek deadline still in the future, from app_fixtures.json."""
    if not fixtures:
        return None
    upcoming = []
    for gw in fixtures.get("gameweeks") or []:
        raw = gw.get("deadline_time")
        if not raw:
            continue
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            continue
        if dt > now:
            upcoming.append(dt)
    return min(upcoming) if upcoming else None


def compute_push_payload(dashboard_dir: Path, now: datetime, lead_hours: float) -> dict | None:
    """Runs push_alerts.compute_alerts() for every tracked account off the committed dashboard
    JSON and collapses the result to the single notification to send (or None)."""
    players = _load(dashboard_dir / "app_players.json") or {}
    players_by_id = {p["id"]: p for p in (players.get("players") or []) if "id" in p}
    fixtures = _load(dashboard_dir / "app_fixtures.json")
    price_watch = _load(dashboard_dir / "app_price_watch.json")
    next_deadline = _next_deadline_utc(fixtures, now)

    all_alerts: list[dict] = []
    for entry_id in TRACKED_ENTRY_IDS:
        alerts = push_alerts.compute_alerts(
            real_squad=_load(dashboard_dir / f"real_squad_{entry_id}.json"),
            team=_load(dashboard_dir / f"app_team_{entry_id}.json"),
            players_by_id=players_by_id,
            price_watch=price_watch,
            next_deadline_utc=next_deadline,
            now_utc=now,
            lead_hours=lead_hours,
        )
        all_alerts.extend(alerts)
    return push_alerts.build_push_payload(all_alerts)


def main() -> None:
    history_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "data" / "report_history"
    now = datetime.now(timezone.utc)

    # --- channel 1: the model-report newly-doubtful GitHub Issue (unchanged) ---
    diff_path = history_dir / "latest_diff.json"
    body = None
    if diff_path.exists():
        body = build_alert_body(json.loads(diff_path.read_text()))
    else:
        print(f"[check_deadline_alerts] no diff file at {diff_path} -- skipping the issue channel")

    # --- channel 2: held-player push alerts for the tracked accounts (app gap 1) ---
    push_payload = None
    try:
        push_payload = compute_push_payload(DASHBOARD_DIR, now, PUSH_LEAD_HOURS)
    except Exception as e:  # noqa: BLE001 -- a bug here must not stop the issue channel or the run
        print(f"::warning::check_deadline_alerts: push-alert computation failed ({e}) -- issue channel unaffected")

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"has_alert={'true' if body else 'false'}\n")
            if body:
                f.write(f"alert_body<<GHADELIM\n{body}\nGHADELIM\n")
            f.write(f"push_payload={json.dumps(push_payload) if push_payload else ''}\n")

    if body:
        print("[check_deadline_alerts] ISSUE ALERT:")
        print(body)
    else:
        print("[check_deadline_alerts] nothing newly doubtful in the model report -- no issue")
    if push_payload:
        print(f"[check_deadline_alerts] PUSH: {push_payload['title']} -- kinds={push_payload['alert_kinds']}")
    else:
        print("[check_deadline_alerts] no held-player push alerts")


if __name__ == "__main__":
    main()
