"""Forward season simulation + Wildcard timing for the two real FPL squads this project tracks.

For each account it runs `forward_season_sim.run_forward_season_sim()` three ways over the
window and writes the results + a plain-English `data/forward_sim/SUMMARY.md`:

  1. hold_wildcard=True  -- never plays the Wildcard, so every gameweek's `evaluate_wildcard`
     gain is measured against the *evolved* squad. The gameweek with the largest gain that
     also clears the model's threshold is the recommended Wildcard week.
  2. model_choice        -- lets the planner play the Wildcard whenever it judges best; records
     which gameweek that turned out to be.
  3. force_wildcard_at=<recommendation from run 1> vs the run-1 baseline -- the projected
     season-points delta from actually taking the Wildcard there.

Needs a real ingested DB (scripts/run_ingestion.py) and open internet (FPL API) -- same
constraints as scripts/run_transfer_planner_for_real_squad.py. Built to run on the scheduled
GitHub Actions runner, not this project's network-blocked dev sandbox.

Usage (from repo root):
    PYTHONPATH=src python scripts/run_forward_season_sim.py [end_gameweek]
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import app_export as ax  # noqa: E402
from fpl_quant import backtest as bt  # noqa: E402
from fpl_quant import db  # noqa: E402
from fpl_quant import forward_season_sim as fss  # noqa: E402
from fpl_quant import ingest_fpl_entry_picks as ifp  # noqa: E402

TARGET_SEASON = "2026-2027"
OUT_DIR = REPO_ROOT / "data" / "forward_sim"
RECALIBRATION_SEED_DIR = REPO_ROOT / "data" / "recalibration"

# The two recurring tracked accounts (same pair scheduled_pipeline.yml plans for twice daily).
ACCOUNTS = [
    (7139944, "ChatGPT template team"),
    (1305242, "Main account"),
]

DEFAULT_END_GAMEWEEK = 16

# Which sims to run per account. `hold_wildcard` alone answers the timing question; the other
# two add "what the model would do on its own" and "the projected points delta from taking it".
# Override via FWSIM_MODES=hold_wildcard (comma-separated) to run a leaner job if the full set
# risks the workflow timeout.
ALL_MODES = ("hold_wildcard", "model_choice", "forced")
MODES = tuple(m.strip() for m in os.environ.get("FWSIM_MODES", ",".join(ALL_MODES)).split(",") if m.strip())


def _fetch_real_squad(entry_id: int, event: int) -> list[dict]:
    element_names = ifp.fetch_bootstrap_elements()
    picks = ifp.fetch_entry_picks(entry_id, event)
    if not picks:
        raise SystemExit(f"no real picks found for entry_id={entry_id} at event={event}")
    return [
        {
            "player_name": element_names[p["element"]],
            "in_xi": p["position"] <= 11,
            "is_captain": bool(p.get("is_captain")),
            "is_vice": bool(p.get("is_vice_captain")),
        }
        for p in picks
    ]


def _run_account(con, entry_id: int, label: str, start_gw: int, end_gw: int, active: dict) -> dict:
    squad = _fetch_real_squad(entry_id, start_gw - 1)  # picks are set for the last completed GW
    print(f"[{label}] {len(squad)} players; simulating GW{start_gw}..{end_gw}")

    common = dict(
        entry_label=label, target_season=TARGET_SEASON,
        start_gameweek=start_gw, end_gameweek=end_gw,
        bootstrap_squad=squad, active_versions=active,
    )

    out: dict = {
        "entry_id": entry_id, "label": label, "start_gameweek": start_gw, "end_gameweek": end_gw,
        "hold_wildcard": None, "model_choice": None, "forced_at_recommendation": None,
        "season_points_delta_from_wildcard": None,
    }

    held = None
    if "hold_wildcard" in MODES:
        t0 = time.time()
        held = fss.run_forward_season_sim(con, hold_wildcard=True, **common)
        out["hold_wildcard"] = held.to_dict()
        print(f"[{label}] hold_wildcard sim: {time.time() - t0:.0f}s; "
              f"projected {held.total_projected_points:.0f} pts; WC reco {held.wildcard_recommendation}")

    if "model_choice" in MODES:
        t0 = time.time()
        model = fss.run_forward_season_sim(con, **common)
        model_wc_gw = next((r.gameweek for r in model.rows if r.action == "wildcard"), None)
        out["model_choice"] = {**model.to_dict(), "model_played_wildcard_at": model_wc_gw}
        print(f"[{label}] model_choice sim: {time.time() - t0:.0f}s; "
              f"projected {model.total_projected_points:.0f} pts; model played WC at GW{model_wc_gw}")

    reco = held.wildcard_recommendation if held is not None else None
    if "forced" in MODES and reco is not None and held is not None:
        t0 = time.time()
        forced = fss.run_forward_season_sim(con, force_wildcard_at=reco["gameweek"], **common)
        out["forced_at_recommendation"] = forced.to_dict()
        out["season_points_delta_from_wildcard"] = round(
            forced.total_projected_points - held.total_projected_points, 1)
        print(f"[{label}] force_wildcard_gw{reco['gameweek']} sim: {time.time() - t0:.0f}s; "
              f"projected {forced.total_projected_points:.0f} pts "
              f"({forced.total_projected_points - held.total_projected_points:+.0f} vs no-WC baseline)")

    return out


def _render_summary(results: list[dict], generated_at: str) -> str:
    lines = [
        "# Forward season simulation — Wildcard timing",
        "",
        f"_Generated {generated_at}. Projected EP, not realised points. Re-run weekly._",
        "",
        "Seeded from each account's **actual current squad**, walked forward one real "
        "`transfer_planner.run()` decision per gameweek, scored on projected expected points "
        "(80% band from `uncertainty_outputs`).",
        "",
    ]
    for r in results:
        held = r.get("hold_wildcard")
        model = r.get("model_choice")
        lines += [f"## {r['label']} (entry {r['entry_id']})", "", f"- Window: GW{r['start_gameweek']}–{r['end_gameweek']}"]
        if held is not None:
            reco = held["wildcard_recommendation"]
            lines.append(
                "- **Recommended Wildcard week: "
                + (f"GW{reco['gameweek']}** (projected gain +{reco['projected_gain']:.0f} pts vs "
                   "continuing weekly transfers)" if reco else "none in this window** — hold it")
            )
        if model is not None:
            lines.append(
                "- Model's own choice (free to play it): "
                + (f"GW{model['model_played_wildcard_at']}"
                   if model["model_played_wildcard_at"] else "did not play it in-window")
            )
        if r["season_points_delta_from_wildcard"] is not None:
            lines.append(
                f"- Projected window-points if Wildcarded at the recommendation: "
                f"{r['season_points_delta_from_wildcard']:+.0f} pts vs never"
            )
        if held is None:
            lines.append("")
            continue
        lines += ["", "| GW | proj pts | 80% band | action | WC gain | WC reco? |",
                  "|----|----------|----------|--------|---------|----------|"]
        for g in held["gameweeks"]:
            band = f"{g['band_low']:.0f}–{g['band_high']:.0f}"
            wg = "—" if g["wildcard_gain"] is None else f"{g['wildcard_gain']:+.0f}"
            lines.append(
                f"| {g['gameweek']} | {g['projected_points']:.0f} | {band} | "
                f"{g['action']}{(' — ' + g['action_detail']) if g['action_detail'] else ''} | "
                f"{wg} | {'yes' if g['wildcard_recommended'] else ''} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    end_gw = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_END_GAMEWEEK
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    con = db.connect()
    active = bt.active_recalibratable_versions(RECALIBRATION_SEED_DIR)
    print(f"[versions] active recalibratable: {active}")

    bootstrap = ax.fetch_bootstrap_static()
    current_event = ax.current_event(bootstrap)
    if current_event is None:
        raise SystemExit("bootstrap-static reports no current gameweek")
    start_gw = current_event + 1
    print(f"[gw] current event {current_event} -> simulate from GW{start_gw}")

    generated_at = datetime.now(timezone.utc).isoformat()
    results = []
    for entry_id, label in ACCOUNTS:
        try:
            results.append(_run_account(con, entry_id, label, start_gw, end_gw, active))
        except Exception as exc:  # one account failing should not lose the other's result
            print(f"::warning::forward sim failed for {label} ({entry_id}): {exc}")

    if not results:
        raise SystemExit("all accounts failed -- nothing to write")

    payload = {"target_season": TARGET_SEASON, "generated_at": generated_at,
               "current_event": current_event, "accounts": results}
    date_tag = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    (OUT_DIR / f"forward_sim_{date_tag}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (OUT_DIR / "forward_sim_latest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (OUT_DIR / "SUMMARY.md").write_text(_render_summary(results, generated_at), encoding="utf-8")
    print(f"\n[write] {OUT_DIR}/forward_sim_{date_tag}.json + SUMMARY.md")

    for r in results:
        reco = r["hold_wildcard"]["wildcard_recommendation"]
        print(f"  {r['label']}: wildcard "
              + (f"GW{reco['gameweek']} (+{reco['projected_gain']:.0f})" if reco else "hold"))


if __name__ == "__main__":
    main()
