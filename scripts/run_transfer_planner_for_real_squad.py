"""M8, for a real FPL entry -- bootstraps manager holdings from an ACTUAL squad (fetched live
from FPL's own public API by entry ID, e.g. one built outside this project entirely), not from
M5's own from-scratch pick. See transfer_planner.bootstrap_from_real_squad()'s own docstring
for why this is a genuinely different bootstrap path from scripts/run_transfer_planner.py.

Usage (from repo root):
    PYTHONPATH=src python scripts/run_transfer_planner_for_real_squad.py <entry_id> <event>

<entry_id> is the numeric id in your FPL team's URL (fantasy.premierleague.com/entry/<entry_id>/...).
<event> is the gameweek your CURRENT squad's picks should be read for (i.e. the gameweek you've
already set your team for) -- the plan itself is produced for the NEXT gameweek automatically.

Same network-blocked-in-sandbox caveat as ingest_fpl_entry_picks.py's own module docstring:
fetch_bootstrap_elements()/fetch_entry_picks() were not verified against a live fetch in this
project's own development sandbox (that environment's network policy blocks
fantasy.premierleague.com entirely) -- this script only runs somewhere with open internet (a
real machine, or the scheduled GitHub Actions workflow).

Depends on scripts/run_ingestion.py having already run for real (needs the same real
ts_model_version=1/mm_model_version=1/etc. this project's other real-run scripts assume).
"""

import json
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import db, fixture_swing as fs, ingest_fpl_entry_picks as ifp, reporting, transfer_planner as tp  # noqa: E402

TARGET_SEASON = "2026-2027"
DASHBOARD_DIR = REPO_ROOT / "data" / "dashboard"


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
    if len(sys.argv) not in (3, 4):
        raise SystemExit(f"usage: {sys.argv[0]} <entry_id> <event> [label]")
    entry_id, current_event = int(sys.argv[1]), int(sys.argv[2])
    label = sys.argv[3] if len(sys.argv) == 4 else str(entry_id)
    plan_for_gameweek = current_event + 1

    con = db.connect()
    tp.seed_v1_params(con)

    print(f"[fetch] pulling real picks for entry_id={entry_id}, GW{current_event}...")
    squad = _fetch_real_squad(entry_id, current_event)
    print(f"[fetch] {len(squad)} players")

    state_version = tp.bootstrap_from_real_squad(
        con, date.today(), TARGET_SEASON, current_event,
        ep_model_version=1, uncertainty_model_version=1, squad=squad,
    )
    print(f"[bootstrap] state_version={state_version} from real entry_id={entry_id}")

    t0 = time.time()
    run_id = tp.run(
        con,
        calibration_asof_date=date.today(),
        target_season=TARGET_SEASON,
        target_gameweek=plan_for_gameweek,
        input_state_version=state_version,
        ts_model_version=1,
        mm_model_version=1,
        horizon_params_version=1,
        scoring_params_version=1,
        bps_params_version=1,
        tau_params_version=1,
        rho_residual_params_version=1,
        corr_params_version=1,
        transfer_cost_params_version=1,
        lambda_params_version=1,
        guardrail_params_version=1,
        wildcard_threshold_params_version=1,
        free_hit_threshold_params_version=1,
        kappa_tc_params_version=1,
        # Priority 3, opt-in: this script's whole point is a real hold-vs-transfer-now
        # recommendation, so it's worth the extra solve time here (unlike the default GW1->GW2
        # run_transfer_planner.py, which leaves this off).
        multi_transfer_pool_limit_per_position=10,
    )
    print(f"[transfer_planner.run] {time.time() - t0:.1f}s -> run_id={run_id}")

    print("\n--- top 5 transfer recommendations ---")
    recs = con.execute(
        "SELECT rank, player_out, player_in, horizon_value_gain, transfer_cost, net_value "
        "FROM transfer_recommendations WHERE run_id = ? ORDER BY rank LIMIT 5", [run_id],
    ).fetchall()
    # Real per-player club, for the dashboard's club-color dots -- same player_uid -> team_uid
    # lookup fixture_swing.py's own swing scores are keyed by, not a separate guess.
    team_by_player = fs.team_uid_by_player(con, TARGET_SEASON)
    team_names = {r[0]: r[1] for r in con.execute("SELECT team_uid, canonical_name FROM dim_team").fetchall()}

    recs_out = []
    for rank, out_uid, in_uid, gain, cost, net in recs:
        out_name = con.execute("SELECT canonical_name FROM dim_player WHERE player_uid = ?", [out_uid]).fetchone()[0]
        in_name = con.execute("SELECT canonical_name FROM dim_player WHERE player_uid = ?", [in_uid]).fetchone()[0]
        out_club = team_names.get(team_by_player.get(out_uid))
        in_club = team_names.get(team_by_player.get(in_uid))
        print(f"  #{rank}: OUT {out_name} -> IN {in_name} | gain={gain:.2f} cost={cost} net={net:.2f}")
        recs_out.append({
            "rank": rank, "player_out": out_name, "player_in": in_name,
            "player_out_club": out_club, "player_in_club": in_club,
            "gain": round(gain, 2), "cost": cost, "net": round(net, 2),
        })

    hold_rec = con.execute(
        "SELECT recommended_action, transfer_now_value, hold_value FROM hold_recommendations WHERE run_id = ?", [run_id],
    ).fetchone()
    hold_out = None
    if hold_rec:
        print(f"\n--- hold vs transfer now ---\n  {hold_rec[0]} (transfer_now={hold_rec[1]:.2f}, hold={hold_rec[2]:.2f})")
        hold_out = {
            "recommended_action": hold_rec[0],
            "transfer_now_value": round(hold_rec[1], 2),
            "hold_value": round(hold_rec[2], 2),
        }

    print("\n--- chip evaluations ---")
    chips = con.execute(
        "SELECT chip_type, recommended, score_or_gain, detail FROM chip_evaluations WHERE run_id = ?", [run_id],
    ).fetchall()
    chips_out = []
    tc_detail = None
    for chip_type, recommended, score, detail in chips:
        print(f"  {chip_type}: recommended={recommended} score={score}")
        chips_out.append({"chip_type": chip_type, "recommended": bool(recommended), "score": round(score, 3) if score is not None else None})
        if chip_type == "triple_captain" and detail:
            tc_detail = json.loads(detail)

    # A real "who should you captain" directive -- reuses the triple_captain evaluator's own
    # already-computed ranking (see reporting.build_captain_recommendation()'s own docstring),
    # compared against the manager's actual current captain from manager_squad_holdings.
    actual_captain_row = con.execute(
        "SELECT player_uid FROM manager_squad_holdings WHERE state_version = ? AND is_captain = TRUE", [state_version],
    ).fetchone()
    actual_captain_uid = actual_captain_row[0] if actual_captain_row else None
    relevant_uids = [uid for uid in {actual_captain_uid, (tc_detail or {}).get("captain_candidate")} if uid]
    name_rows = con.execute(
        "SELECT player_uid, canonical_name FROM dim_player WHERE player_uid = ANY(?)", [relevant_uids],
    ).fetchall() if relevant_uids else []
    player_name_by_uid = {uid: name for uid, name in name_rows}
    captain_recommendation = reporting.build_captain_recommendation(tc_detail, actual_captain_uid, player_name_by_uid)
    if captain_recommendation:
        print(f"\n--- captain recommendation ---\n  {captain_recommendation}")

    # M9's own explainability adapter (transfer_planner.explain_plan()) -- pure assembly over
    # this same run_id, no new computation. Exported as-is so the dashboard can show the real
    # per-chip/hold rationale (detail JSON) instead of the frontend inventing explanatory text.
    # player_uid-keyed (top_transfers/top_multi_transfers use player_uid, not display names --
    # unlike recs_out above) since explain_plan() itself is a generic, non-dashboard-specific
    # adapter; the frontend falls back to showing the uid when a name lookup isn't available.
    explain = tp.explain_plan(con, run_id, top_n=5)
    print(f"\n[explain] attached transfer_planner.explain_plan() output (gw19_urgent={explain['gw19_deadline']})")

    con.close()

    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "entry_id": entry_id,
        "label": label,
        "target_season": TARGET_SEASON,
        "current_gameweek": current_event,
        "plan_for_gameweek": plan_for_gameweek,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "transfer_recommendations": recs_out,
        "hold_vs_transfer_now": hold_out,
        "chip_evaluations": chips_out,
        "captain_recommendation": captain_recommendation,
        "explain": explain,
    }
    out_path = DASHBOARD_DIR / f"real_squad_{entry_id}.json"
    out_path.write_text(json.dumps(snapshot, indent=2))
    print(f"\n[dashboard] snapshot written to {out_path}")


if __name__ == "__main__":
    main()
