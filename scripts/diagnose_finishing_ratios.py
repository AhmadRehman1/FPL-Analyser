"""How the finishing-skill prior's goals/xG and assists/xA ratios spread, window by window.

The --finishing-prior-xg walk-forward arm flipped from +1.43 to -1.62 points a gameweek against
FPL's real average when 2024-25 joined its data window (docs/reports/2026-10_open_issues.md,
issue 5). This prints, for each lookback window, how many players sit on the hard
MAX_FINISHING_RATIO clamp (and its floor), the ratio quantiles, and who is capped -- so it shows
whether the ratios themselves move when 2024-25 is added, and how much the clamp decides.

Usage (from repo root, against an ingested DB; read-only):
    PYTHONPATH=src python scripts/diagnose_finishing_ratios.py [--prior-xg 10] [--min-expected 1.0]
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import db  # noqa: E402
from fpl_quant import expected_points as ep  # noqa: E402

WINDOWS = (("2024-2025",), ("2025-2026",), ("2025-2026", "2024-2025"), ("2026-2027", "2025-2026", "2024-2025"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prior-xg", type=float, default=10.0)
    parser.add_argument("--min-expected", type=float, default=1.0, help="only players with at least this xG (xA)")
    args = parser.parse_args(argv)
    con = db.connect(read_only=True)
    reports = [ep.finishing_ratio_report(con, list(w), args.prior_xg, args.min_expected) for w in WINDOWS]
    print(json.dumps(reports, indent=2))
    con.close()


if __name__ == "__main__":
    main()
