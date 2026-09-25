"""Phase 1A/1B diagnostics: price-band calibration + selection-residual ("optimiser curse")
reports, recomputed from raw walk-forward rows (never from backtest_metrics' pre-aggregated
segment means -- see src/fpl_quant/calibration_diagnostics.py's module docstring).

Writes two versioned JSON artefacts to data/calibration_diagnostics/, named with the backtest
run ID and a UTC timestamp so a run is never silently overwritten by a later one -- each
artefact already carries its own git SHA / seed / tier / scoring_params_version / gameweek-range
header (this project's artefact-versioning convention; see docs/audit_2026-09_calibration_
captain_headline.md for why several other artefacts in this repo currently DON'T do this).

Read-only against the DB (writes only the two JSON files).

Usage (from repo root):
    PYTHONPATH=src python scripts/run_calibration_diagnostics.py [backtest_run_id]

With no argument, uses max(backtest_run_id).
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import calibration_diagnostics as cd  # noqa: E402
from fpl_quant import db  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "data" / "calibration_diagnostics"


def _gameweek_range(rows: list[dict]) -> dict:
    by_season: dict = {}
    for r in rows:
        lo, hi = by_season.get(r["season"], (r["gameweek"], r["gameweek"]))
        by_season[r["season"]] = (min(lo, r["gameweek"]), max(hi, r["gameweek"]))
    return {season: {"min_gw": lo, "max_gw": hi} for season, (lo, hi) in sorted(by_season.items())}


def main() -> None:
    con = db.connect()
    run_id = int(sys.argv[1]) if len(sys.argv) > 1 else con.execute(
        "SELECT max(backtest_run_id) FROM backtest_runs"
    ).fetchone()[0]
    if run_id is None:
        print("no backtest_runs row (run scripts/run_walkforward.py, or nightly_backtest.yml).")
        con.close()
        return

    price_band = cd.price_band_calibration_report(con, run_id, tier="mature")
    price_band["gameweek_range"] = _gameweek_range(price_band["rows"])
    curse = cd.selection_curse_report(con, run_id, tier="mature")
    con.close()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    price_band_path = OUTPUT_DIR / f"price_band_calibration_run{run_id}_{stamp}.json"
    curse_path = OUTPUT_DIR / f"selection_curse_run{run_id}_{stamp}.json"
    price_band_path.write_text(json.dumps(price_band, indent=2, default=str))
    curse_path.write_text(json.dumps(curse, indent=2, default=str))

    print(f"backtest_run_id={run_id}  n_rows={price_band['overall']['n']}  "
          f"gameweek_range={price_band['gameweek_range']}")
    print(f"overall mean_resid={price_band['overall']['mean_resid']:.4f}  "
          f"ci95={price_band['overall']['mean_resid_ci95_gw_clustered']}")
    for band, summary in sorted(price_band["by_price_band"].items()):
        print(f"  price_band={band:8}  n={summary['n']:6}  mean_resid={summary['mean_resid']:+.4f}  "
              f"mae={summary['mae']:.4f}  ci95={summary['mean_resid_ci95_gw_clustered']}")
    print(f"\nselection curse (XI, price-band-matched): n_gameweeks="
          f"{curse['variants']['price_band__xi']['n_gameweeks']}  "
          f"mean_curse={curse['variants']['price_band__xi']['mean_curse']}  "
          f"ci95={curse['variants']['price_band__xi']['curse_ci95_gw_clustered']}  "
          f"frac_positive={curse['variants']['price_band__xi']['frac_positive']}")
    print(f"\nwrote {price_band_path}")
    print(f"wrote {curse_path}")


if __name__ == "__main__":
    main()
