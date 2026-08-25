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

from fpl_quant import backtest as bt, db, fixture_swing as fs, ingest_fpl_entry_picks as ifp, reporting, transfer_planner as tp  # noqa: E402

TARGET_SEASON = "2026-2027"
DASHBOARD_DIR = REPO_ROOT / "data" / "dashboard"
DECISION_LOG_DIR = REPO_ROOT / "data" / "decision_log"


# More than one chip can legitimately clear its own recommendation threshold in the same
# gameweek (Bench Boost and Triple Captain both look worth playing fairly often) -- the SQL
# below has no ORDER BY, so chips_out otherwise comes back in whatever order the DB happens to
# return chip_evaluations rows in, which is NOT guaranteed to match backtest.py's own
# CHIP_PRIORITY (the one place this project already had to solve "which chip actually wins" --
# see its own docstring). The dashboard's chips.find(c => c.recommended) just takes whichever
# comes first in this array, so an unordered array silently picks the wrong winner whenever the
# DB's incidental order disagrees with CHIP_PRIORITY -- which it did: chip_evaluations rows are
# inserted wildcard/free_hit/triple_captain/bench_boost, the reverse of CHIP_PRIORITY's own
# bench_boost-before-triple_captain.
def _order_chip_evaluations(chips_out: list[dict]) -> list[dict]:
    order = {chip_type: i for i, chip_type in enumerate(bt.CHIP_PRIORITY)}
    return sorted(chips_out, key=lambda c: order.get(c["chip_type"], len(order)))


def _build_chip_preview_squad(
    preview_rows: list[dict], name_by_uid: dict[str, str], team_by_player: dict[str, str], team_names: dict[str, str],
) -> list[dict]:
    """Pure transform from transfer_planner.read_fresh_chip_squad()'s raw
    (player_uid, in_xi, is_captain, is_vice) rows to the dashboard's preview_squad shape
    (player_name/club, not player_uid -- see main()'s own comment on why: the frontend has no
    player_uid->FPL-element-id mapping, only a name lookup). Split out from main() so this
    reshaping is unit-testable without a DB connection, matching this project's own
    unit-the-math/integrate-against-real-data split (see _order_chip_evaluations() above)."""
    return [
        {
            "player_name": name_by_uid.get(p["player_uid"], p["player_uid"]),
            "club": team_names.get(team_by_player.get(p["player_uid"])),
            "in_xi": bool(p["in_xi"]), "is_captain": bool(p["is_captain"]), "is_vice": bool(p["is_vice"]),
        }
        for p in preview_rows
    ]


def _resolve_decision_log_row(
    hold_rec: tuple | None, recs_out: list[dict], chips_out: list[dict], captain_recommendation: dict | None,
) -> dict:
    """Pure transform from this run's already-computed hold/transfer/chip/captain outputs to a
    single planner_decision_log row -- split out so the 'hold' case (recommended_action='hold',
    still a real logged row, not an absent one -- see Plan Track C's own edge case) and the other
    branches are unit-testable without a DB/network, same split-out-the-pure-part convention as
    _order_chip_evaluations()/_build_chip_preview_squad() above.

    hold_rec: the raw (recommended_action, transfer_now_value, hold_value) row from
    hold_recommendations, or None. hold_recommendations.run_id is a PRIMARY KEY REFERENCES
    transfer_plan_runs, so this is always populated for a real run_id in practice; the
    'no_action_available' fallback here only guards the type, matching the same enum value
    evaluate_hold_recommendation() itself already uses for that case.
    """
    recommended_action = hold_rec[0] if hold_rec else "no_action_available"
    return {
        "recommended_action": recommended_action,
        "recommended_transfer_out": recs_out[0]["player_out"] if recommended_action == "transfer_now" and recs_out else None,
        "recommended_transfer_in": recs_out[0]["player_in"] if recommended_action == "transfer_now" and recs_out else None,
        "recommended_chip": next((c["chip_type"] for c in chips_out if c["recommended"]), None),
        "recommended_captain": (
            captain_recommendation["recommended_name"]
            if captain_recommendation and not captain_recommendation["matches_current"]
            else None
        ),
    }


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
    # Same signal this project already uses to distinguish the two accounts scheduled_pipeline.yml
    # runs twice daily (always passed a label) from the ad-hoc, workflow_dispatch-only, one-off
    # third-account path (never passed one) -- reused below to scope planner_decision_log to the
    # two recurring, tracked accounts a hold-vs-use backtest can actually mean something for, not
    # every one-off manual check anyone with push access ever runs.
    is_tracked_account = len(sys.argv) == 4
    label = sys.argv[3] if is_tracked_account else str(entry_id)
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
        entry = {"chip_type": chip_type, "recommended": bool(recommended), "score": round(score, 3) if score is not None else None}
        # Wildcard/Free Hit already ran a real, solved M5 candidate squad to produce this very
        # score (evaluate_wildcard()/evaluate_free_hit() call squad_optimizer.run()
        # unconditionally, before the recommended/not-recommended threshold check even happens)
        # -- so a manager can preview it in the app's chip detail sheet at zero extra solve
        # cost, whether or not the chip clears its recommendation threshold this week.
        # player_name/club (not player_uid) to match this script's own recs_out convention
        # above: the frontend has no player_uid->FPL-element-id mapping, only a name lookup
        # (playerIdByName()), same as transfer_recommendations' player_out/player_in already are.
        if chip_type in ("wildcard", "free_hit") and detail and json.loads(detail).get("fresh_run_id") is not None:
            try:
                preview = tp.read_fresh_chip_squad(con, run_id, chip_type)
                preview_uids = [p["player_uid"] for p in preview]
                preview_name_by_uid = dict(con.execute(
                    "SELECT player_uid, canonical_name FROM dim_player WHERE player_uid = ANY(?)", [preview_uids],
                ).fetchall()) if preview_uids else {}
                entry["preview_squad"] = _build_chip_preview_squad(preview, preview_name_by_uid, team_by_player, team_names)
                print(f"    preview_squad: {len(entry['preview_squad'])} players")
            except ValueError as exc:
                # A secondary, nice-to-have panel failing to build shouldn't break the export
                # for the whole account -- same allowed-to-fail-independently philosophy as the
                # "Grade squad + explain best move" step's own continue-on-error in the workflow.
                print(f"    preview_squad: skipped ({exc})")
        chips_out.append(entry)
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

    # Roadmap P1 item (Track C, docs/plans/2026-08_roadmap_plan.md): log this run's own
    # recommendation as a committed JSON file (reporting.save_decision_log_entry() -- see its
    # own docstring for why this is a file, not a DuckDB table: db/fpl_quant_v2.duckdb doesn't
    # survive to the next scheduled run, so a DB-only row would be gone before Phase C-2 could
    # ever read it back). Only for the two recurring, tracked accounts (is_tracked_account) --
    # an ad-hoc one-off dispatch isn't part of the twice-daily cadence a hold-vs-use comparison
    # needs, and was never in scope (see Track C's own Non-Goal). Ordered the same way the
    # dashboard snapshot below is (_order_chip_evaluations, not the raw DB-insertion order) so
    # the two outputs of this same run can never disagree about which chip was recommended.
    if is_tracked_account:
        decision_row = _resolve_decision_log_row(
            hold_rec, recs_out, _order_chip_evaluations(chips_out), captain_recommendation,
        )
        decision_row.update({
            "entry_id": entry_id,
            "target_season": TARGET_SEASON,
            "target_gameweek": plan_for_gameweek,
            "run_id": run_id,
            "actual_action_taken": None,
            "realized_points_actual": None,
            "realized_points_if_recommendation_followed": None,
            "logged_at": datetime.now(timezone.utc).isoformat(),
        })
        reporting.save_decision_log_entry(entry_id, TARGET_SEASON, plan_for_gameweek, decision_row, DECISION_LOG_DIR)
        print(f"\n[decision_log] logged '{decision_row['recommended_action']}' for entry_id={entry_id}, GW{plan_for_gameweek}")

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
        "chip_evaluations": _order_chip_evaluations(chips_out),
        "captain_recommendation": captain_recommendation,
        "explain": explain,
    }
    out_path = DASHBOARD_DIR / f"real_squad_{entry_id}.json"
    out_path.write_text(json.dumps(snapshot, indent=2))
    print(f"\n[dashboard] snapshot written to {out_path}")


if __name__ == "__main__":
    main()
