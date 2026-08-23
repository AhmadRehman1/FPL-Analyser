"""Review B2 / roadmap Feature 7: publishes the model's real backtested edge (or lack of it)
over three baselines a manager could follow with none of this project's M1-M6 machinery --
the change that converts the architecture from a disclosure ("65 of 71 parameters still
invented") into evidence. See backtest.beats_baseline() for the actual computation; this
script only reads a completed backtest_run_id's already-recorded metrics and writes the
PWA-facing JSON.

Needs a backtest_run_id from a real scripts/run_backtest.py run made WITH
ownership_params_version set (Priority 9c) -- without that, score_gameweek() never records
the metrics beats_baseline() reads, and this script raises loudly rather than writing an
empty/fabricated leaderboard.

Usage (from repo root, after scripts/run_backtest.py has produced a real backtest_run_id):
    PYTHONPATH=src python scripts/export_leaderboard.py [backtest_run_id]

With no argument, uses the most recent backtest_run_id in backtest_runs.
"""

import json
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import backtest, db  # noqa: E402

DASHBOARD_DIR = REPO_ROOT / "data" / "dashboard"


def main() -> None:
    con = db.connect()

    if len(sys.argv) > 1:
        backtest_run_id = int(sys.argv[1])
    else:
        row = con.execute("SELECT max(backtest_run_id) FROM backtest_runs").fetchone()
        if row is None or row[0] is None:
            raise SystemExit("no backtest_runs rows exist -- run scripts/run_backtest.py first")
        backtest_run_id = row[0]

    result = backtest.beats_baseline(con, backtest_run_id)
    data_asof = date.today().isoformat()
    payload = {"data_asof": data_asof, **result}

    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DASHBOARD_DIR / f"leaderboard_{data_asof}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[export_leaderboard] backtest_run_id={backtest_run_id} -> wrote {out_path}")
    for row_data in result["rows"]:
        print(f"  {row_data['name']}: ep={row_data['ep']:.2f} [{row_data['ci_low']:.2f}, {row_data['ci_high']:.2f}]")
    if result["honest_losses"]:
        print(f"  honest_losses: {result['honest_losses']}")

    con.close()


if __name__ == "__main__":
    main()
