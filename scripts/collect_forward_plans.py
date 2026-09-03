"""Merge the per-entity forward-plan files (data/forward_plan/<slug>.json, one per
forward_plan.yml matrix arm) into a single data/forward_plan/forward_plan_latest.json the app
fetches once. Missing arms are skipped (an arm that timed out just isn't in the merged file);
if none landed, exits non-zero so the workflow surfaces it.

Usage (from repo root):
    PYTHONPATH=src python scripts/collect_forward_plans.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "forward_plan"
ENTITY_SLUGS = ["model_team", "7139944", "1305242"]


def main() -> None:
    entities: dict[str, dict] = {}
    for slug in ENTITY_SLUGS:
        path = OUT_DIR / f"{slug}.json"
        if not path.exists():
            print(f"::warning::forward plan missing for {slug} ({path.name}) -- skipped")
            continue
        entities[slug] = json.loads(path.read_text(encoding="utf-8"))

    if not entities:
        raise SystemExit("no per-entity forward-plan files found -- every matrix arm failed")

    merged = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_season": next(iter(entities.values())).get("target_season"),
        "entities": entities,
    }
    out_path = OUT_DIR / "forward_plan_latest.json"
    out_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    weeks = {k: len(v.get("weeks", [])) for k, v in entities.items()}
    print(f"[forward_plan] merged {len(entities)} entities {weeks} -> {out_path}")


if __name__ == "__main__":
    main()
