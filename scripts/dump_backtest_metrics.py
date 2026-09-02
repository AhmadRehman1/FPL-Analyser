"""Dump the latest walk-forward backtest run's aggregated metrics to a small JSON file.

Used by ab_evidence_strength.yml: each A/B arm runs its own ingestion + run_walkforward.py,
then this writes {metric_name: {mean, n}} plus provenance so the compare step can diff two
arms without either DB. Whole-population metrics only ("name:segment" rows are dropped -- those
only feed recalibrate()).

Usage (from repo root):
    PYTHONPATH=src python scripts/dump_backtest_metrics.py <out_path.json> [label]
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import db  # noqa: E402


def main() -> None:
    out_path = Path(sys.argv[1])
    label = sys.argv[2] if len(sys.argv) > 2 else out_path.stem

    con = db.connect()
    row = con.execute("SELECT max(backtest_run_id) FROM backtest_runs").fetchone()
    if row is None or row[0] is None:
        raise SystemExit("no backtest_runs row -- run scripts/run_walkforward.py first")
    backtest_run_id = row[0]

    n_steps = con.execute(
        "SELECT count(*) FROM backtest_gameweek_steps WHERE backtest_run_id = ?", [backtest_run_id]
    ).fetchone()[0]
    seasons = [
        r[0] for r in con.execute(
            "SELECT DISTINCT season FROM backtest_gameweek_steps WHERE backtest_run_id = ? ORDER BY season",
            [backtest_run_id],
        ).fetchall()
    ]
    metrics = {
        name: {"mean": round(mean, 5), "n": n}
        for name, mean, n in con.execute(
            "SELECT metric_name, avg(metric_value), count(*) FROM backtest_metrics "
            "WHERE backtest_run_id = ? AND metric_name NOT LIKE 'realized%' AND metric_name NOT LIKE '%:%' "
            "GROUP BY metric_name ORDER BY metric_name",
            [backtest_run_id],
        ).fetchall()
    }
    # the predicted_xi pull + tier weights this DB was actually ingested with, so the compare
    # step can label the arms from the artifacts alone.
    try:
        pull = con.execute(
            "SELECT value_numeric FROM param_versions WHERE param_family = 'minutes_adjustment_params' "
            "AND param_key = 'magnitude' AND dimensions = ?",
            [json.dumps({"claim_type": "predicted_xi"}, sort_keys=True, separators=(",", ":"))],
        ).fetchone()
        predicted_xi_pull = pull[0] if pull else None
    except Exception:  # noqa: BLE001
        predicted_xi_pull = None
    try:
        tw = con.execute(
            "SELECT value_numeric FROM param_versions WHERE param_family = 'source_tier_weights' "
            "AND param_key = 'tier_weight' AND dimensions = ?",
            [json.dumps({"source_type": "official"}, sort_keys=True, separators=(",", ":"))],
        ).fetchone()
        official_tier_weight = tw[0] if tw else None
    except Exception:  # noqa: BLE001
        official_tier_weight = None

    payload = {
        "label": label,
        "backtest_run_id": backtest_run_id,
        "n_gameweek_steps": n_steps,
        "seasons_covered": seasons,
        "config": {
            "predicted_xi_pull_strength": predicted_xi_pull,
            "official_tier_weight": official_tier_weight,
        },
        "metrics": metrics,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[dump_backtest_metrics] {label}: run {backtest_run_id}, {n_steps} steps, "
          f"{len(metrics)} metrics -> {out_path}")
    for name, m in metrics.items():
        print(f"  {name}: {m['mean']} (n={m['n']})")
    con.close()


if __name__ == "__main__":
    main()
