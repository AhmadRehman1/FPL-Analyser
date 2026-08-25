"""Roadmap Feature 2: "Rate My Team" for a real FPL entry -- grades the manager's actual
squad (fetched live, same pattern as run_transfer_planner_for_real_squad.py) against the
mathematically-optimal squad squad_optimizer's real MIQP solve finds. Writes
data/dashboard/squad_grade_<entry_id>_<asof>.json for the PWA.

Usage (from repo root):
    PYTHONPATH=src python scripts/grade_squad.py <entry_id> <event>

<event> is the gameweek your CURRENT squad's picks should be read for (same convention as
run_transfer_planner_for_real_squad.py) -- the grade itself is produced for the NEXT
gameweek, matching what a transfer decision right now would actually be graded against.

Same network-blocked-in-sandbox caveat as every other real-squad script in this project.
"""

import dataclasses
import json
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import backtest, db, ingest_fpl_entry_picks as ifp, reporting, squad_grade as sg, transfer_planner as tp  # noqa: E402

TARGET_SEASON = "2026-2027"
DASHBOARD_DIR = REPO_ROOT / "data" / "dashboard"
RECALIBRATION_SEED_DIR = REPO_ROOT / "data" / "recalibration"


# Roadmap P1 item (Track B, docs/plans/2026-08_roadmap_plan.md): rho_residual_params_version/
# lambda_params_version resolve from the git-committed confirmed-seed files, not a hardcoded
# literal -- see backtest.active_recalibratable_versions()'s own docstring.
def _param_versions(active: dict) -> dict:
    return dict(
        scoring_params_version=1, bps_params_version=1, tau_params_version=1,
        rho_residual_params_version=active["rho_residual_params_version"], corr_params_version=1,
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
    active = backtest.active_recalibratable_versions(RECALIBRATION_SEED_DIR)
    horizon_ep_versions = tp.compute_horizon_ep(
        con, calibration_asof_date, TARGET_SEASON, plan_for_gameweek, ts_mv, mm_mv, 1, **_param_versions(active),
    )
    ep_mv, un_mv = horizon_ep_versions[plan_for_gameweek]

    state_version = tp.bootstrap_from_real_squad(
        con, calibration_asof_date, TARGET_SEASON, current_event, ep_mv, un_mv, squad,
    )
    current_holdings = tp._read_holdings(con, state_version)

    grade = sg.grade_squad(
        con, entry_id, calibration_asof_date, TARGET_SEASON, plan_for_gameweek,
        current_holdings, horizon_ep_versions,
        lambda_params_version=active["lambda_params_version"], guardrail_params_version=1,
    )

    data_asof = calibration_asof_date.isoformat()
    payload = dataclasses.asdict(grade)
    payload["data_asof"] = data_asof
    # Real fix, same as explain_my_move.py: top_swaps is keyed by this project's own internal
    # player_uid, meaningless to the PWA on its own -- resolve to real display names before
    # writing (see reporting.py's own docstring on why this shared resolution lives there).
    swap_uids = {uid for s in grade.top_swaps for uid in (s.out_player_uid, s.in_player_uid)}
    names = reporting.resolve_player_names(con, swap_uids)
    payload["top_swaps"] = reporting.humanize_swap_list(payload["top_swaps"], names)

    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DASHBOARD_DIR / f"squad_grade_{entry_id}_{data_asof}.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    # Stable-named copy, same "PWA needs a fixed, predictable path" convention as
    # app_team_<id>.json -- a static site can't discover today's date-embedded filename on
    # its own. The dated file stays too, as the historical/audit record.
    (DASHBOARD_DIR / f"squad_grade_{entry_id}_latest.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(
        f"[grade_squad] entry_id={entry_id}: grade={grade.grade}, points_gap={grade.points_gap:.2f} "
        f"(optimal={grade.optimal_ep:.2f}, yours={grade.user_squad_ep:.2f}) -> wrote {out_path}"
    )
    for s in grade.top_swaps:
        print(f"  swap: {s.out_player_uid} -> {s.in_player_uid} (+{s.delta_ep:.2f} ep, {s.reason})")

    con.close()


if __name__ == "__main__":
    main()
