"""Roadmap Feature 3 -- "Explain My Move" for a real FPL entry: the single recommended
action, why, the downside, what would change my mind, and the historical track record of
this input pattern. Writes data/dashboard/decision_<entry_id>_<gw>_<asof>.json for the PWA
and prints a plain-English paragraph.

Usage (from repo root):
    PYTHONPATH=src python scripts/explain_my_move.py <entry_id> <event>

<event> is the gameweek your CURRENT squad's picks should be read for (same convention as
run_transfer_planner_for_real_squad.py/grade_squad.py) -- the recommendation is produced for
the NEXT gameweek.

Same network-blocked-in-sandbox caveat as every other real-squad script in this project.
"""

import dataclasses
import json
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import decision_engine as de  # noqa: E402
from fpl_quant import db, ingest_fpl_entry_picks as ifp, transfer_planner as tp  # noqa: E402

TARGET_SEASON = "2026-2027"
DASHBOARD_DIR = REPO_ROOT / "data" / "dashboard"

PARAM_VERSIONS = dict(
    horizon_params_version=1, scoring_params_version=1, bps_params_version=1, tau_params_version=1,
    rho_residual_params_version=2, corr_params_version=1, transfer_cost_params_version=1,
    lambda_params_version=1, guardrail_params_version=1, wildcard_threshold_params_version=1,
    free_hit_threshold_params_version=1, kappa_tc_params_version=1,
)


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


def _explain_paragraph(entry_id: int, gw: int, decision: de.Decision) -> str:
    lines = [f"Entry {entry_id}, GW{gw}: recommended action is '{decision.action}'."]
    if decision.swaps:
        s = decision.swaps[0]
        lines.append(f"Why: {s.out_player_uid} -> {s.in_player_uid} ({s.reason}), projected gain {decision.ep_lift:+.2f} EP over the horizon.")
    else:
        lines.append(f"Why: no move clears the model's own threshold this week (projected gain {decision.ep_lift:+.2f} EP).")
    lines.append(f"Downside: 5th-95th percentile squad total is [{decision.downside_ci[0]:.1f}, {decision.downside_ci[1]:.1f}].")
    if decision.sensitivity:
        sens = decision.sensitivity[0]
        lines.append(f"What would change my mind: if {sens.if_condition}, the recommendation flips to '{sens.then_action}' ({sens.delta_ep:+.2f} EP).")
    else:
        lines.append("What would change my mind: no toggled scenario flips this recommendation.")
    tr = decision.track_record
    if tr.optimal_in_n_of_71 is None:
        lines.append(f"Track record: insufficient history for pattern '{tr.pattern}' (sample_size={tr.sample_size}).")
    else:
        lines.append(f"Track record: this pattern ('{tr.pattern}') was optimal {tr.optimal_in_n_of_71}/{tr.sample_size} times historically.")
    if decision.runner_up is not None:
        lines.append(f"Runner-up (close call): '{decision.runner_up.action}' -- consider both.")
    return " ".join(lines)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} <entry_id> <event>")
    entry_id, current_event = int(sys.argv[1]), int(sys.argv[2])
    plan_for_gameweek = current_event + 1

    con = db.connect()
    tp.seed_v1_params(con)

    print(f"[fetch] pulling real picks for entry_id={entry_id}, GW{current_event}...")
    squad = _fetch_real_squad(entry_id, current_event)
    print(f"[fetch] {len(squad)} players")

    ts_mv = con.execute("SELECT max(model_version) FROM team_strength_model_versions").fetchone()[0]
    mm_mv = con.execute("SELECT max(model_version) FROM minutes_model_versions").fetchone()[0]
    if ts_mv is None or mm_mv is None:
        raise SystemExit("no team_strength/minutes model versions found -- run scripts/run_ingestion.py first")

    calibration_asof_date = date.today()
    horizon_ep_versions = tp.compute_horizon_ep(
        con, calibration_asof_date, TARGET_SEASON, plan_for_gameweek, ts_mv, mm_mv, 1,
        PARAM_VERSIONS["scoring_params_version"], PARAM_VERSIONS["bps_params_version"], PARAM_VERSIONS["tau_params_version"],
        PARAM_VERSIONS["rho_residual_params_version"], PARAM_VERSIONS["corr_params_version"],
    )
    ep_mv, un_mv = horizon_ep_versions[plan_for_gameweek]

    state_version = tp.bootstrap_from_real_squad(con, calibration_asof_date, TARGET_SEASON, current_event, ep_mv, un_mv, squad)

    decision = de.recommend_best_move(
        con, entry_id, calibration_asof_date, TARGET_SEASON, plan_for_gameweek, state_version,
        ts_mv, mm_mv, **PARAM_VERSIONS,
    )

    data_asof = calibration_asof_date.isoformat()
    payload = dataclasses.asdict(decision)
    payload["entry_id"] = entry_id
    payload["gw"] = plan_for_gameweek
    payload["data_asof"] = data_asof

    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DASHBOARD_DIR / f"decision_{entry_id}_{plan_for_gameweek}_{data_asof}.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[explain_my_move] wrote {out_path}")
    print()
    print(_explain_paragraph(entry_id, plan_for_gameweek, decision))

    con.close()


if __name__ == "__main__":
    main()
