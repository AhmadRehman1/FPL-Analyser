"""Gap 2 (docs/BUSINESS_PLAN.md 2.4): a loud post-export check that the optional dashboard
feeds scheduled_pipeline.yml produces are actually present, parseable, and fresh.

Before this, a generator script that errored mid-run left a stale (or absent) `_latest.json`,
and index.html's own `.catch(() => null)` on every one of those fetches meant a real regression
degraded silently -- the section just vanished, indistinguishable from "this feature isn't
built". This script runs right after the export steps and FAILS the workflow (non-zero exit +
`::error::`) if any expected feed is missing, unparseable, or older than a scheduled run should
ever leave it.

Only covers the feeds scheduled_pipeline.yml itself writes every run:
  - data/dashboard/projections_latest.json   (scripts/export_projections.py)
  - data/dashboard/elite_divergence_latest.json (scripts/track_elite.py)
leaderboard_latest.json is deliberately NOT checked here -- it's written by the separate,
weekly weekly_backtest.yml job, on its own cadence, and would be legitimately stale relative
to a twice-daily run.

Usage (from repo root):
    PYTHONPATH=src python scripts/verify_dashboard_feeds.py [dashboard_dir]
"""

import json
import sys
from datetime import date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = REPO_ROOT / "data" / "dashboard"

# A scheduled run writes these every invocation, so a `data_asof` older than this many calendar
# days means the step that should have refreshed it silently didn't. 2 (not 1) absorbs the
# UTC-vs-local and near-midnight boundary between the run and this check.
MAX_ASOF_AGE_DAYS = 2

_ELITE_OK_STATUSES = {"ok", "not_configured"}


def _asof_age_days(data_asof: str, today: date) -> int | None:
    try:
        parsed = datetime.strptime(data_asof, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None
    return (today - parsed).days


def verify_feed(path: Path, *, kind: str, today: date | None = None) -> list[str]:
    """Returns a list of human-readable problems with this feed -- empty means it's healthy."""
    today = today or date.today()
    if not path.exists():
        return [f"{path.name}: missing -- its generator script did not write it this run"]
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return [f"{path.name}: unreadable/unparseable ({e})"]

    problems: list[str] = []
    data_asof = payload.get("data_asof")
    age = _asof_age_days(data_asof, today) if data_asof is not None else None
    if data_asof is None:
        problems.append(f"{path.name}: no data_asof field")
    elif age is None:
        problems.append(f"{path.name}: unparseable data_asof {data_asof!r}")
    elif age > MAX_ASOF_AGE_DAYS:
        problems.append(f"{path.name}: stale -- data_asof {data_asof} is {age} days old (max {MAX_ASOF_AGE_DAYS})")

    if kind == "projections":
        if not payload.get("players"):
            problems.append(f"{path.name}: no player rows -- export produced an empty projection table")
        if not payload.get("gameweeks"):
            problems.append(f"{path.name}: no gameweeks listed")
    elif kind == "elite":
        status = payload.get("status")
        if status not in _ELITE_OK_STATUSES:
            problems.append(f"{path.name}: unexpected status {status!r} (want one of {sorted(_ELITE_OK_STATUSES)})")
    return problems


def main(argv: list[str]) -> int:
    dashboard_dir = Path(argv[1]) if len(argv) > 1 else DASHBOARD_DIR
    checks = [
        (dashboard_dir / "projections_latest.json", "projections"),
        (dashboard_dir / "elite_divergence_latest.json", "elite"),
    ]
    all_problems: list[str] = []
    for path, kind in checks:
        problems = verify_feed(path, kind=kind)
        for p in problems:
            all_problems.append(p)
        if not problems:
            print(f"[verify_dashboard_feeds] OK: {path.name}")

    if all_problems:
        for p in all_problems:
            print(f"::error::verify_dashboard_feeds: {p}")
        return 1
    print("[verify_dashboard_feeds] all expected dashboard feeds present and fresh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
