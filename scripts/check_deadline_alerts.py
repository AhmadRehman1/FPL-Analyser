"""Priority 8b: reads scripts/run_report.py's own latest_diff.json and decides whether a
deadline alert is warranted -- specifically "a nailed starter in the current squad gets
flagged doubtful in the window before deadline" (the roadmap's own exact wording), not every
week-over-week change (squad/captain churn is expected and already visible in the diff report
itself, not alert-worthy on its own).

Writes GITHUB_OUTPUT (when set, i.e. running inside a GH Actions step) so the calling workflow
step can conditionally open a GitHub Issue -- kept as a separate, standalone script rather than
baked into run_report.py so it stays usable/testable outside any particular CI product.

Usage: python scripts/check_deadline_alerts.py [report_history_dir]
"""

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


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


def main() -> None:
    history_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "data" / "report_history"
    diff_path = history_dir / "latest_diff.json"
    if not diff_path.exists():
        print(f"[check_deadline_alerts] no diff file at {diff_path} -- nothing to check")
        return

    diff = json.loads(diff_path.read_text())
    body = build_alert_body(diff)

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"has_alert={'true' if body else 'false'}\n")
            if body:
                delimiter = "GHADELIM"
                f.write(f"alert_body<<{delimiter}\n{body}\n{delimiter}\n")

    if body:
        print("[check_deadline_alerts] ALERT:")
        print(body)
    else:
        print("[check_deadline_alerts] nothing newly doubtful -- no alert")


if __name__ == "__main__":
    main()
