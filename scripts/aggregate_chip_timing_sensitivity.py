"""Collects every scripts/run_chip_timing_sensitivity_arm.py output ("off" vs "on") into one
comparison table -- the walk-forward evidence docs/reports/2026-09_chip_policy_and_scoring_
diagnosis.md's Workstream B asks for before triple_captain_timing_params_version/
bench_boost_timing_params_version default on anywhere (forward_season_sim._resolve_versions()
currently pins both to None -- see its own comment).

Reads data/chip_timing_study/arms/*.json (whatever arms landed -- a timeout or crash just
drops that arm, same fail-soft convention as aggregate_lambda_sensitivity.py) and writes:

  data/chip_timing_study/chip_timing_study_latest.json  -- machine-readable, every arm
  data/chip_timing_study/SUMMARY.md                     -- the human table + an EVIDENCE-ONLY
                                                            reading (off vs on, per season)

This does NOT flip any default. It lays out realized total / Sharpe / chips actually played
per arm so a human can decide whether the season-horizon timing gate is worth turning on --
that decision stays with the project owner, matching every other sensitivity study here.
"""

import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARMS_DIR = REPO_ROOT / "data" / "chip_timing_study" / "arms"
OUT_DIR = REPO_ROOT / "data" / "chip_timing_study"


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


def _table(rows: list[dict]) -> list[str]:
    lines = [
        "| arm | season | total | mean/GW | sharpe | max drawdown | transfers | chips played | GWs |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: (r["season"], r["arm"])):
        m = r.get("metrics", {})
        ac = r.get("action_counts", {})
        lines.append(
            f"| {r['arm']} | {r['season']} | {_fmt(m.get('total_points'), '{:.0f}')} | "
            f"{_fmt(m.get('mean_points'))} | {_fmt(m.get('realized_sharpe'), '{:.3f}')} | "
            f"{_fmt(m.get('max_drawdown'))} | {ac.get('transfers', '-')} | "
            f"{ac.get('chips_played', '-')} | {m.get('n_gameweeks', '-')} |"
        )
    return lines


def _chips_diff(off_row: dict | None, on_row: dict | None) -> str:
    if off_row is None or on_row is None:
        return "one of the two arms is missing -- no comparison possible."
    off_chips = {tuple(c) for c in off_row.get("chips_played", [])}
    on_chips = {tuple(c) for c in on_row.get("chips_played", [])}
    held = sorted(off_chips - on_chips)
    same = sorted(off_chips & on_chips)
    added = sorted(on_chips - off_chips)
    parts = []
    if held:
        parts.append(f"the timing gate HELD {len(held)} chip(s) the greedy arm played: {held}")
    if added:
        parts.append(f"the timing gate additionally played {len(added)} chip(s) the greedy arm didn't: {added}")
    if same:
        parts.append(f"{len(same)} chip decision(s) unchanged: {same}")
    return "; ".join(parts) if parts else "identical chip decisions in both arms."


def _reading(rows: list[dict]) -> list[str]:
    """Evidence-only: off vs on, per season -- does the wider timing window help, hurt, or do
    nothing (because it never actually held a chip the narrow check would have played)? No
    recommendation -- see this script's own module docstring."""
    if not rows:
        return ["_No arms landed._"]
    by_season = defaultdict(dict)
    for r in rows:
        by_season[r["season"]][r["arm"]] = r

    out = []
    for season, arms in sorted(by_season.items()):
        off, on = arms.get("off"), arms.get("on")
        if off is None or on is None:
            out.append(f"- **{season}**: incomplete ({'off' if off is None else 'on'} arm missing) -- no comparison.")
            continue
        off_m, on_m = off["metrics"], on["metrics"]
        delta_total = on_m.get("total_points", 0) - off_m.get("total_points", 0)
        delta_sharpe = on_m.get("realized_sharpe", 0) - off_m.get("realized_sharpe", 0)
        verdict = "improves" if delta_total > 0 else "regresses" if delta_total < 0 else "ties"
        line = (
            f"- **{season}**: timing-on {verdict} total realized points by {delta_total:+.1f} "
            f"({off_m.get('total_points', 0):.0f} -> {on_m.get('total_points', 0):.0f}), "
            f"Sharpe {delta_sharpe:+.3f}. {_chips_diff(off, on)}"
        )
        out.append(line)
    out.append("")
    out.append("_This is evidence, not a decision. triple_captain_timing_params_version/"
                "bench_boost_timing_params_version stay None (off) in forward_season_sim."
                "_resolve_versions() and backtest.run_season_simulation()'s own defaults until "
                "the project owner reviews this and explicitly turns them on._")
    return out


def main() -> None:
    arms = _load_arms()
    if not arms:
        raise SystemExit(f"no arm files in {ARMS_DIR} -- nothing to aggregate")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "chip_timing_study_latest.json").write_text(json.dumps({
        "n_arms": len(arms),
        "arms": arms,
    }, indent=2))

    md = ["# Season-horizon chip-timing sensitivity study (Workstream B's \"fuller ask\")", ""]
    md += [f"{len(arms)} arms landed. Realized-points, evolving-manager season simulation "
           "(`backtest.run_season_simulation`). \"off\" = current greedy approach (PR #173's "
           "magnitude floor only); \"on\" = adds the wider-window season-horizon timing gate "
           "(triple_captain_timing_params_version=1, bench_boost_timing_params_version=1).", ""]
    md += ["## Arms", ""]
    md += _table(arms)
    md += ["", "## Reading the evidence", ""]
    md += _reading(arms)
    md += [""]
    (OUT_DIR / "SUMMARY.md").write_text("\n".join(md))

    print(f"[aggregate] {len(arms)} arms -> {OUT_DIR/'SUMMARY.md'}, {OUT_DIR/'chip_timing_study_latest.json'}")
    print("\n".join(md))


if __name__ == "__main__":
    main()
