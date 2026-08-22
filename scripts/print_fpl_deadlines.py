"""Prints every real gameweek's deadline_time from FPL's own bootstrap-static, plus the
UTC cron expression for "1 day before that deadline" -- the actual data this project's
scheduled workflow's cron list is generated from (see .github/workflows/scheduled_pipeline.yml's
own comment on why this isn't dynamic self-scheduling: GitHub Actions cron can't be rescheduled
from inside a run, so this is a manual re-sync tool instead, re-run whenever the real season
schedule changes, e.g. postponements/rescheduled fixtures).

Usage (needs open internet -- run via GitHub Actions, not this project's own dev sandbox,
which blocks fantasy.premierleague.com by policy):
    python scripts/print_fpl_deadlines.py
"""

from datetime import datetime, timedelta, timezone

import requests

FPL_API_BASE = "https://fantasy.premierleague.com/api"


def main() -> None:
    resp = requests.get(f"{FPL_API_BASE}/bootstrap-static/", timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    events = resp.json()["events"]

    now = datetime.now(timezone.utc)
    print(f"{'GW':>3}  {'deadline_time (UTC)':<20}  {'1 day before (UTC)':<20}  cron (min hour day month *)  {'status'}")
    for e in events:
        deadline = datetime.fromisoformat(e["deadline_time"].replace("Z", "+00:00"))
        one_day_before = deadline - timedelta(days=1)
        cron = f"{one_day_before.minute} {one_day_before.hour} {one_day_before.day} {one_day_before.month} *"
        status = "past" if deadline < now else ("NEXT" if not e.get("finished") and one_day_before < now < deadline else "")
        print(f"{e['id']:>3}  {deadline.strftime('%Y-%m-%d %H:%M'):<20}  {one_day_before.strftime('%Y-%m-%d %H:%M'):<20}  {cron:<28}  {status}")


if __name__ == "__main__":
    main()
