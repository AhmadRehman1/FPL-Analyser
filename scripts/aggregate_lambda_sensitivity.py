"""Collects every scripts/run_lambda_sensitivity_arm.py output into one comparison table.

Reads data/lambda_study/arms/*.json (whatever arms actually landed -- a divergence-check
failure or a job timeout just drops that arm, same fail-soft convention as
aggregate_chip_timing.py) and writes:

  data/lambda_study/lambda_study_latest.json  -- machine-readable, every arm + a per-season
                                                 lambda/cap comparison
  data/lambda_study/SUMMARY.md                -- the human table + an EVIDENCE-ONLY reading

This does NOT recommend a value to promote. It lays out realized total / Sharpe / drawdown /
churn per trial value so a human can weigh them against the live pin (lambda 0.15, cap 3) --
promotion stays scripts/review_recalibration.py's gate, and seeds_1.json stays parked until
the owner decides. The "reading" section flags the shape of the evidence (is a lower lambda
strictly better, or a total-vs-consistency trade-off?), nothing more.
"""

import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARMS_DIR = REPO_ROOT / "data" / "lambda_study" / "arms"
OUT_DIR = REPO_ROOT / "data" / "lambda_study"

LIVE_LAMBDA = 0.15
LIVE_CAP = 3


def _load_arms() -> list[dict]:
    arms = []
    for path in sorted(ARMS_DIR.glob("*.json")):
        try:
            arms.append(json.loads(path.read_text()))
        except (OSError, ValueError) as exc:
            print(f"::warning::could not read {path.name}: {exc}")
    return arms


def _fmt(v, spec="{:.2f}"):
    return spec.format(v) if isinstance(v, (int, float)) else "-"


def _table(rows: list[dict], value_label: str, live_value) -> list[str]:
    lines = [
        f"| {value_label} | season | total | mean/GW | sharpe | max drawdown | transfers | chips | GWs |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: (r["season"], r["value"])):
        m = r.get("metrics", {})
        ac = r.get("action_counts", {})
        mark = " (live pin)" if r["value"] == live_value else ""
        lines.append(
            f"| {r['value']}{mark} | {r['season']} | {_fmt(m.get('total_points'), '{:.0f}')} | "
            f"{_fmt(m.get('mean_points'))} | {_fmt(m.get('realized_sharpe'), '{:.3f}')} | "
            f"{_fmt(m.get('max_drawdown'))} | {ac.get('transfers', '-')} | "
            f"{ac.get('chips_played', '-')} | {m.get('n_gameweeks', '-')} |"
        )
    return lines


def _reading(lambda_rows: list[dict]) -> list[str]:
    """Evidence-only: does the season data point the same way across seasons, and is a move a
    strict improvement or a trade-off? No recommendation."""
    if not lambda_rows:
        return ["_No lambda arms landed._"]
    by_season = defaultdict(list)
    for r in lambda_rows:
        by_season[r["season"]].append(r)

    out = []
    for season, rows in sorted(by_season.items()):
        rows = [r for r in rows if isinstance(r.get("metrics", {}).get("realized_sharpe"), (int, float))]
        if not rows:
            continue
        best_sharpe = max(rows, key=lambda r: r["metrics"]["realized_sharpe"])
        best_total = max(rows, key=lambda r: r["metrics"].get("total_points", float("-inf")))
        live = next((r for r in rows if r["value"] == LIVE_LAMBDA), None)
        line = f"- **{season}**: best realized Sharpe at lambda={best_sharpe['value']} " \
               f"(sharpe {best_sharpe['metrics']['realized_sharpe']:.3f}, " \
               f"total {best_sharpe['metrics'].get('total_points', 0):.0f}); " \
               f"best total at lambda={best_total['value']} " \
               f"({best_total['metrics'].get('total_points', 0):.0f} pts)."
        if live is not None:
            line += f" Live pin 0.15 scored sharpe {live['metrics']['realized_sharpe']:.3f}, " \
                    f"total {live['metrics'].get('total_points', 0):.0f}, " \
                    f"drawdown {live['metrics'].get('max_drawdown', 0):.2f}."
            if best_sharpe["value"] != LIVE_LAMBDA and best_total["value"] != LIVE_LAMBDA:
                strict = (
                    best_sharpe["value"] == best_total["value"]
                    and best_sharpe["metrics"]["max_drawdown"] <= live["metrics"]["max_drawdown"] + 1e-9
                )
                line += (" A **strict** improvement over the pin (better on total, Sharpe AND drawdown)."
                         if strict else
                         " A **trade-off**, not a strict win -- weigh total vs consistency vs drawdown by hand.")
        out.append(line)
    out.append("")
    out.append("_This is evidence, not a decision. seeds_1.json stays parked; promotion is "
               "scripts/review_recalibration.py's human gate (README: the lambda study gates it, "
               "the 'attack' posture default, and the deferred 'protect rank' toggle)._")
    return out


def main() -> None:
    arms = _load_arms()
    if not arms:
        raise SystemExit(f"no arm files in {ARMS_DIR} -- nothing to aggregate")

    lambda_rows = [a for a in arms if a["axis"] == "lambda"]
    cap_rows = [a for a in arms if a["axis"] == "cap"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "lambda_study_latest.json").write_text(json.dumps({
        "n_arms": len(arms),
        "live_pins": {"lambda_value": LIVE_LAMBDA, "xi_club_concentration_cap": LIVE_CAP},
        "lambda_arms": lambda_rows,
        "cap_arms": cap_rows,
    }, indent=2))

    md = ["# Risk-aversion (lambda) / concentration-cap sensitivity study", ""]
    md += [f"{len(arms)} arms landed. Realized-points, evolving-manager season simulation "
           "(`backtest.run_season_simulation`). Live pins: lambda=0.15, cap=3.", ""]
    md += ["## lambda_value sweep", ""]
    md += _table(lambda_rows, "lambda", LIVE_LAMBDA) if lambda_rows else ["_No lambda arms landed._"]
    md += ["", "## xi_club_concentration_cap sweep (lambda held at the live pin)", ""]
    md += _table(cap_rows, "cap", LIVE_CAP) if cap_rows else ["_No cap arms landed._"]
    md += ["", "## Reading the evidence", ""]
    md += _reading(lambda_rows)
    md += [""]
    (OUT_DIR / "SUMMARY.md").write_text("\n".join(md))

    print(f"[aggregate] {len(arms)} arms -> {OUT_DIR/'SUMMARY.md'}, {OUT_DIR/'lambda_study_latest.json'}")
    print("\n".join(md))


if __name__ == "__main__":
    main()
