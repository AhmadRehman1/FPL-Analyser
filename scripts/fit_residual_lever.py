"""Fits lever L1, the crowd prior (docs/plans/2026-10_beat_the_template.md, Phase 5), from a
walk-forward's residual export and writes its coefficient file.

Input: residual_features.csv from `run_walkforward.py --export-residuals` (Req 12): one row per
player-gameweek with realized points, the projection before any lever (ep_pre_adjustment) and the
crowd features own_pct, log_sbp and xfer_pct from row G-lag.

Fit (residual_levers.fit_crowd_prior): ridge on the standardized features, alpha 10, no
intercept, of realized - ep_pre_adjustment, once per fold -- each season's own rows and both
seasons pooled. residual_levers.fold_for() applies each historical season with the OTHER season's
fit, so no season is scored with coefficients fitted on its own residuals; 2026-27 (live) uses
the pooled fit. A fold with fewer than min_rows complete rows is not fitted (no adjustment).

Output: data/recalibration/crowd_prior_coeffs_<DATE>.json with the features, lag, alpha,
min_rows, clip, the source file's sha256 and run id, each fold's fit, and a cross-fold check:
each season's residual MAE before and after the adjustment fitted on the other season, applied
with expected_points.run()'s floor (an adjusted projection never goes below zero).

Usage (from repo root):
    PYTHONPATH=src python scripts/fit_residual_lever.py --residuals residual_features.csv \\
        --date 20261012 --lag 2 [--source-run 38269659997] [--clip 1.5] [--alpha 10] [--min-rows 2000]
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import residual_levers as rl  # noqa: E402

SEASONS = ("2024-2025", "2025-2026")


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _complete(row: dict, features: tuple[str, ...]) -> bool:
    return all(row.get(k) not in (None, "") for k in (*features, "realized_points", "ep_pre_adjustment"))


def cross_fold_check(rows: list[dict], folds: dict, features: tuple[str, ...], clip: float) -> dict:
    """Per season: residual MAE and mean before and after the adjustment from the fold that scores
    it, applied as expected_points.run() applies it (an adjusted projection is floored at zero, or
    at itself when already below), over the rows with a realized score and a projection. Rows
    without a full feature set keep their residual (no adjustment). share_floored counts the
    adjusted rows the floor cut, unfloored_mae_after the MAE without it."""
    out = {}
    for season in SEASONS:
        fit = folds.get(rl.fold_for(season))
        season_rows = [r for r in rows if r.get("season") == season
                       and r.get("realized_points") not in (None, "") and r.get("ep_pre_adjustment") not in (None, "")]
        if fit is None or not season_rows:
            out[season] = {"n": len(season_rows), "fitted_fold": None}
            continue
        before, after, unfloored, adjusted, floored, adjs = [], [], [], 0, 0, []
        for r in season_rows:
            ep_pre = float(r["ep_pre_adjustment"])
            resid = float(r["realized_points"]) - ep_pre
            adj = rl.adjustment(fit, {f: float(r[f]) for f in features if r.get(f) not in (None, "")}, features, clip)
            applied = 0.0 if adj is None else max(adj, -max(ep_pre, 0.0))
            before.append(resid)
            after.append(resid - applied)
            unfloored.append(resid - (adj or 0.0))
            if adj is not None:
                adjusted += 1
                floored += applied != adj
                adjs.append(applied)
        n = len(before)
        out[season] = {
            "n": n, "n_adjusted": adjusted, "fitted_fold": rl.fold_for(season),
            "mae_before": round(sum(abs(x) for x in before) / n, 4),
            "mae_after": round(sum(abs(x) for x in after) / n, 4),
            "unfloored_mae_after": round(sum(abs(x) for x in unfloored) / n, 4),
            "rmse_before": round((sum(x * x for x in before) / n) ** 0.5, 4),
            "rmse_after": round((sum(x * x for x in after) / n) ** 0.5, 4),
            "mean_resid_before": round(sum(before) / n, 4),
            "mean_resid_after": round(sum(after) / n, 4),
            "mean_adjustment": round(sum(adjs) / len(adjs), 4) if adjs else None,
            "share_clipped": round(sum(1 for a in adjs if abs(a) >= clip) / len(adjs), 4) if adjs else None,
            "share_floored": round(floored / adjusted, 4) if adjusted else None,
        }
    return out


def build(rows: list[dict], lag: int, alpha: float, min_rows: int, clip: float,
          source: dict, features: tuple[str, ...] = rl.FEATURES) -> dict:
    folds = rl.fit_crowd_prior(rows, alpha=alpha, min_rows=min_rows, features=features)
    return {
        "lever": "L1 crowd prior (docs/plans/2026-10_beat_the_template.md)",
        "features": list(features), "lag": lag, "alpha": alpha, "min_rows": min_rows, "clip": clip,
        "source": source,
        "n_rows": {s: sum(1 for r in rows if r.get("season") == s) for s in SEASONS},
        "n_complete": {s: sum(1 for r in rows if r.get("season") == s and _complete(r, features)) for s in SEASONS},
        "folds": folds,
        "fold_for": {s: rl.fold_for(s) for s in (*SEASONS, "2026-2027")},
        "cross_fold_check": cross_fold_check(rows, folds, features, clip),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--residuals", required=True, type=Path, help="residual_features.csv")
    parser.add_argument("--date", required=True, type=int, help="YYYYMMDD; names the coefficient file")
    parser.add_argument("--lag", required=True, type=int, choices=(1, 2),
                        help="the feature row the export read (G minus this); must match the run's --template-lag")
    parser.add_argument("--source-run", default=None, help="the walk-forward's Actions run id, for the record")
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--min-rows", type=int, default=2000)
    parser.add_argument("--clip", type=float, default=1.5)
    parser.add_argument("--out", type=Path, default=None, help="default data/recalibration/crowd_prior_coeffs_<DATE>.json")
    args = parser.parse_args(argv)
    rows = read_rows(args.residuals)
    source = {"file": args.residuals.name, "sha256": hashlib.sha256(args.residuals.read_bytes()).hexdigest(),
              "run_id": args.source_run}
    out = build(rows, args.lag, args.alpha, args.min_rows, args.clip, source)
    path = args.out or rl.coeff_path(args.date)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"path": str(path), "n_complete": out["n_complete"],
                      "folds": {k: (None if v is None else {"n": v["n"], "beta": v["beta"]}) for k, v in out["folds"].items()},
                      "cross_fold_check": out["cross_fold_check"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
