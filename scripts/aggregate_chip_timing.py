"""Aggregate the two-team Wildcard-timing sweep into per-team reports (chip_timing_analysis
Steps 1-6).

Reads every arm artifact `scripts/run_chip_timing_arm.py` wrote under
`data/chip_timing/arms/`, groups them by (entry, param bundle), and for each:

  * Step 1  -- `compare_wildcard_timing()` over the hold / model_choice / forced arms
  * Step 2  -- the winning forced arm's own `wildcard_followups.bench_boost_window`
  * Step 3  -- `free_hit_scan_from_hold_arm()` off the hold arm
  * Step 4  -- the winning forced arm's own `wildcard_followups.robustness`
  * Step 0.2 -- `evidence_freshness_flags()` for the entry's real held squad
  * Step 5  -- cross-bundle stability of the Step 1/2/3 answers

Writes, per team: `<slug>_gameweeks.csv`, `<slug>_wildcard_sweep.csv`, plus a combined
`SUMMARY.md` and `chip_timing_latest.json`. Needs the same real DB + live FPL API as the arm
runner (evidence_claims + squad resolution).

Usage (from repo root):
    PYTHONPATH=src python scripts/aggregate_chip_timing.py
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import app_export as ax  # noqa: E402
from fpl_quant import chip_timing_analysis as cta  # noqa: E402
from fpl_quant import db  # noqa: E402
from fpl_quant import ingest_fpl_entry_picks as ifp  # noqa: E402
from fpl_quant import ingest_workbook as iw  # noqa: E402

TARGET_SEASON = "2026-2027"
ARMS_DIR = REPO_ROOT / "data" / "chip_timing" / "arms"
OUT_DIR = REPO_ROOT / "data" / "chip_timing"

BUNDLE_LABELS = {
    "active": "active (xi v2, rho_residual=0.0, lambda=0.15, kappa_tc=0.15)",
    "confirmed_pending": "confirmed-but-unpromoted (+ rho_residual=0.0168, lambda=0.05, kappa_tc=0.5)",
    "all_v1": "all v1 (no recalibration)",
}


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def _load_arms() -> dict:
    """{(entry_id, bundle): {"hold": arm, "model_choice": arm, "forced": {gw: arm}, "raw": {...}}}"""
    grouped: dict[tuple[int, str], dict] = {}
    for path in sorted(ARMS_DIR.glob("arm_*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        key = (d["entry_id"], d["param_bundle"])
        g = grouped.setdefault(key, {"hold": None, "model_choice": None, "forced": {}, "raw": {},
                                     "label": d["label"], "start_gameweek": d["start_gameweek"],
                                     "eval_end_gameweek": d["eval_end_gameweek"],
                                     "real_chips_used_set1": d.get("real_chips_used_set1", []),
                                     "real_chips_used_set2": d.get("real_chips_used_set2", [])})
        arm = cta.WildcardArm.from_forward_sim_dict(d["forward_sim"])
        g["raw"][d["arm"]] = d
        if d["arm"] == "hold":
            g["hold"] = arm
        elif d["arm"] == "model_choice":
            g["model_choice"] = arm
        elif arm.forced_gameweek is not None:
            g["forced"][arm.forced_gameweek] = arm
    return grouped


def _held_uids(con, entry_id: int, event: int) -> list[str]:
    bootstrap = ax.fetch_bootstrap_static()
    names = ifp.fetch_bootstrap_elements(payload=bootstrap)
    payload = ax.fetch_entry_picks(entry_id, event)
    picks = payload.get("picks") if payload else None
    if not picks:
        return []
    uids = []
    for p in picks:
        uid = iw._resolve_player(con, names.get(p["element"]), TARGET_SEASON)
        if uid:
            uids.append(uid)
    return uids


class _BundleIncomplete(RuntimeError):
    pass


def _bundle_report(con, entry_id: int, bundle: str, g: dict, evidence_flags: list[dict]) -> tuple:
    hold = g["hold"]
    if hold is None:
        raise _BundleIncomplete(
            f"entry {entry_id} bundle {bundle}: no hold_wildcard arm artifact -- cannot compare"
        )
    comparison = cta.compare_wildcard_timing(
        entry_label=g["label"],
        start_gameweek=g["start_gameweek"],
        eval_end_gameweek=g["eval_end_gameweek"],
        hold_arm=hold,
        model_choice_arm=g["model_choice"],
        forced_arms=list(g["forced"].values()),
    )

    winner_gw = comparison.swept_best_gameweek
    followups = None
    if winner_gw is not None:
        raw = g["raw"].get(f"force_gw{winner_gw}", {})
        followups = raw.get("wildcard_followups")

    fh_scan = cta.free_hit_scan_from_hold_arm(hold)

    report = cta.TeamChipTimingReport(
        entry_id=entry_id,
        entry_label=g["label"],
        generated_at=datetime.now(timezone.utc).isoformat(),
        active_param_bundle=BUNDLE_LABELS.get(bundle, bundle),
        comparison=comparison.to_dict(),
        bench_boost_window=(followups or {}).get("bench_boost_window", []),
        free_hit_scan=fh_scan,
        robustness=(followups or {}).get("robustness"),
        sensitivity=None,  # filled by the cross-bundle pass
        evidence_freshness_flags=evidence_flags,
        data_flags=_data_flags(g, evidence_flags),
    )
    return report, hold, comparison


def _data_flags(g: dict, evidence_flags: list[dict]) -> list[str]:
    flags = []
    stale = [f["name"] for f in evidence_flags if f["status"] == "stale"]
    missing = [f["name"] for f in evidence_flags if f["status"] == "no_availability_claims"]
    if stale:
        flags.append(f"stale injury/rotation evidence (>14d) for held players: {', '.join(stale)}")
    if missing:
        flags.append(f"no availability evidence at all for held players: {', '.join(missing)}")
    if g["real_chips_used_set1"] or g["real_chips_used_set2"]:
        flags.append(f"entry has already used chips: set1={g['real_chips_used_set1']} set2={g['real_chips_used_set2']}")
    return flags


def _sensitivity(per_bundle: dict) -> dict:
    """Cross-bundle stability of the Step 1/2/3 answers (Step 5)."""
    picks = {b: r.comparison["swept_best_gameweek"] for b, (r, _, _) in per_bundle.items()}
    greedy = {b: r.comparison["greedy_gameweek"] for b, (r, _, _) in per_bundle.items()}
    fh = {
        b: sorted(x["gameweek"] for x in r.free_hit_scan if x["clears_threshold"])
        for b, (r, _, _) in per_bundle.items()
    }
    wc_stable = len(set(picks.values())) == 1
    note_parts = [f"Wildcard week by bundle: {picks}."]
    note_parts.append("Stable across bundles." if wc_stable
                      else "NOT stable -- the recalibration choice moves the Wildcard recommendation.")
    if len(per_bundle) > 1:
        note_parts.append(f"Greedy WC week by bundle: {greedy}. Free Hit weeks by bundle: {fh}.")
    return {
        "wildcard_week_by_bundle": picks,
        "greedy_week_by_bundle": greedy,
        "free_hit_weeks_by_bundle": fh,
        "wildcard_stable": wc_stable,
        "note": " ".join(note_parts),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grouped = _load_arms()
    if not grouped:
        raise SystemExit(f"no arm artifacts under {ARMS_DIR}")

    con = db.connect()
    entries = sorted({eid for (eid, _b) in grouped})
    bundles_present = sorted({b for (_e, b) in grouped})
    generated_at = datetime.now(timezone.utc).isoformat()

    summary_lines = [
        "# Two-team chip timing -- Wildcard / Bench Boost / Free Hit",
        "",
        f"_Generated {generated_at}. Projected expected points, not realised. "
        f"Param bundles compared: {', '.join(bundles_present)}._",
        "",
        "The full-horizon `force_wildcard_at` sweep is the primary Wildcard signal; the greedy "
        "`model_choice` walk only sees ~5 gameweeks ahead at each step and is shown for contrast. "
        "See `src/fpl_quant/chip_timing_analysis.py` for the method.",
        "",
    ]

    all_json: dict = {"target_season": TARGET_SEASON, "generated_at": generated_at, "teams": []}

    for entry_id in entries:
        # evidence freshness once per entry (bundle-independent)
        event = next((g["start_gameweek"] - 1 for (e, _b), g in grouped.items() if e == entry_id), None)
        held = _held_uids(con, entry_id, event) if event else []
        ev_flags = cta.evidence_freshness_flags(
            con, held_player_uids=held, as_of_date=datetime.now(timezone.utc).date(),
        ) if held else []

        per_bundle = {}
        for bundle in bundles_present:
            g = grouped.get((entry_id, bundle))
            if g is None:
                continue
            try:
                per_bundle[bundle] = _bundle_report(con, entry_id, bundle, g, ev_flags)
            except _BundleIncomplete as exc:
                print(f"::warning::{exc}")

        if not per_bundle:
            print(f"::warning::entry {entry_id}: no complete bundle -- skipping")
            continue
        sens = _sensitivity(per_bundle)

        # primary bundle = "active" if present, else the first
        primary = "active" if "active" in per_bundle else sorted(per_bundle)[0]
        report, hold_arm, _cmp = per_bundle[primary]
        report.sensitivity = sens

        slug = _slug(report.entry_label)
        (OUT_DIR / f"{slug}_gameweeks.csv").write_text(report.gameweek_csv(hold_arm), encoding="utf-8")
        (OUT_DIR / f"{slug}_wildcard_sweep.csv").write_text(report.sweep_csv(), encoding="utf-8")

        summary_lines.append(cta.render_team_summary(report, hold_arm))
        all_json["teams"].append({
            "entry_id": entry_id,
            "primary_bundle": primary,
            "report": report.to_dict(),
            "bundles": {b: r.to_dict() for b, (r, _, _) in per_bundle.items()},
        })

    (OUT_DIR / "SUMMARY.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    (OUT_DIR / "chip_timing_latest.json").write_text(json.dumps(all_json, indent=2), encoding="utf-8")
    date_tag = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    (OUT_DIR / f"chip_timing_{date_tag}.json").write_text(json.dumps(all_json, indent=2), encoding="utf-8")
    print(f"[write] {OUT_DIR}/SUMMARY.md + per-team CSVs + chip_timing_latest.json")

    for team in all_json["teams"]:
        r = team["report"]["comparison"]
        print(f"  {team['report']['entry_label']}: swept WC GW{r['swept_best_gameweek']} "
              f"(greedy GW{r['greedy_gameweek']}); sensitivity stable="
              f"{team['report']['sensitivity']['wildcard_stable']}")


if __name__ == "__main__":
    main()
