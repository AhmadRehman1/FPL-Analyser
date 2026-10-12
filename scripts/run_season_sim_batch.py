"""Runs several season-simulation arms one after another on this one runner: season_sim_arms.yml's
same_runner mode (docs/plans/2026-10_beat_the_template.md, Req 7). The same arms on different
runners have split by +-1.81 pts/GW on the same DB and params
(docs/reports/2026-10_promoted_club_strength.md), so an arm judged against a control must run
beside it; a second control (control_b) on the same runner measures what noise is left.

Each arm is its own scripts/run_season_sim_arm.py process writing
data/season_sim_arms/<tag>.json, so the aggregator reads them exactly like matrix jobs' files.
An arm that fails doesn't stop the ones after it; the exit code says whether any failed.

Usage (from repo root):
    PYTHONPATH=src python scripts/run_season_sim_batch.py '[{"label": "control", "flags": "", "season": "2025-2026",
        "start": 2, "end": 38, "tag": "control_2025-2026_gw2-38"}, ...]'
"""

import json
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_ARMS = 5


def arm_command(arm: dict, out_dir: Path) -> list[str]:
    return [
        sys.executable, str(REPO_ROOT / "scripts" / "run_season_sim_arm.py"),
        "--label", arm["label"], "--season", arm["season"],
        "--start-gw", str(arm["start"]), "--end-gw", str(arm["end"]),
        "--out", str(out_dir / f"{arm['tag']}.json"),
        *shlex.split(arm.get("flags") or ""),
    ]


def main(argv: list[str] | None = None, run=subprocess.run, out_dir: Path | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print(__doc__)
        return 2
    batch = json.loads(args[0])
    if not batch or len(batch) > MAX_ARMS:
        print(f"::error::a same-runner batch takes 1 to {MAX_ARMS} arms, got {len(batch)}")
        return 2
    out_dir = out_dir or REPO_ROOT / "data" / "season_sim_arms"
    out_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for arm in batch:
        print(f"[season-sim-batch] {arm['tag']}: {arm.get('flags') or '(control flags)'}", flush=True)
        if run(arm_command(arm, out_dir)).returncode != 0:
            failed.append(arm["tag"])
            print(f"::error::season-sim arm {arm['tag']} failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
