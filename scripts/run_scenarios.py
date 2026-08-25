"""Roadmap Feature 4: "what-if" scenarios for a real FPL entry, using scenario.py's own
apply_scenario() mechanism (already built and tested, but never wired into anything before
this -- unlike every other roadmap feature, no script exported it and index.html never
fetched it).

Scope, stated plainly: scenario.py's own docstring frames Feature 4 as a "standalone,
interactive" layer -- genuinely free-form interactivity (the user picks ANY player, ANY
scenario kind, gets a live answer) would need a request-handling backend this project
doesn't have; index.html is a static PWA that only ever fetches pre-committed JSON. This
script's real, disclosed v1 scope is narrower and precomputed: two bounded scenario sets,
never the whole 15-player squad (same cost-discipline reasoning `evaluate_transfers()`'s own
docstring gives for capping its search, not an arbitrary limit) --

1. "lineup_change" (the cheapest and most actionable for a manager who already has a squad --
   "would starting this specific bench player over my current XI change the model's plan?"),
   applied to every one of the manager's own real BENCH players (up to 4).
2. "injury" for the manager's own real captain and vice-captain specifically -- "if my captain
   doesn't play at all, does the plan change?" -- the two players the real FPL armband-transfer
   rule actually turns on (see transfer_planner.vice_captain_fallback_adjustment()'s own
   docstring for the EV side of this same real rule). decision_engine's own built-in
   sensitivity toggle already answers "what if my single highest-EP player is ruled out," which
   often is NOT the captain (captaincy also weighs risk, not just raw EP) -- this is a
   genuinely different, complementary question, not a duplicate.

Each additional scenario costs one real decision_engine.recommend_best_move() call (a full
transfer-search + all-4-chip-evaluation pass), so up to 6 total per account (4 bench + captain
+ vice) is bounded, not free -- deliberately NOT extended to injury/price_change scenarios for
the whole squad.

Writes data/dashboard/scenarios_<entry_id>_<gw>_<asof>.json for the PWA.

Usage (from repo root):
    PYTHONPATH=src python scripts/run_scenarios.py <entry_id> <event>

<event> is the gameweek your CURRENT squad's picks should be read for (same convention as
run_transfer_planner_for_real_squad.py/grade_squad.py/explain_my_move.py) -- scenarios are
evaluated against the NEXT gameweek's plan, matching what a transfer decision right now would
actually be weighed against.

Same network-blocked-in-sandbox caveat as every other real-squad script in this project.
"""

import json
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import db, decision_engine as de, ingest_fpl_entry_picks as ifp, reporting  # noqa: E402
from fpl_quant import scenario as scen  # noqa: E402
from fpl_quant import transfer_planner as tp  # noqa: E402

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


def bench_player_uids(current_holdings: list[dict]) -> list[str]:
    """Pure, DB-free: the real bench (in_xi=False, per manager_squad_holdings' own convention
    -- see transfer_planner._read_holdings()), in the same order they're held."""
    return [h["player_uid"] for h in current_holdings if not h["in_xi"]]


def scenario_result_row(player_uid: str, result: scen.ScenarioResult) -> dict:
    """Pure, DB-free: shapes one apply_scenario() ScenarioResult into the PWA's JSON row.
    delta_ep/flipped are exactly what scenario.ScenarioResult already computes (baseline vs
    perturbed recommend_best_move() calls) -- never re-derived here."""
    return {
        "player_uid": player_uid,
        "delta_ep": result.delta_ep,
        "flipped": result.flipped,
        "baseline_action": result.baseline_decision.action,
        "perturbed_action": result.perturbed_decision.action,
    }


def armband_uids(current_holdings: list[dict]) -> dict[str, str]:
    """Pure, DB-free: {"captain": player_uid, "vice_captain": player_uid} for whichever of the
    two are actually found in current_holdings -- a role absent from the manager's real squad
    holdings (shouldn't happen for a real, complete squad, but never assumed) is simply omitted
    from the returned dict, not defaulted to some other player."""
    out = {}
    for h in current_holdings:
        if h["is_captain"]:
            out["captain"] = h["player_uid"]
        elif h["is_vice"]:
            out["vice_captain"] = h["player_uid"]
    return out


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
    # Real perf fix (see scripts/compute_shared_horizon.py's own module docstring): reuse the
    # pipeline's shared multi-gameweek horizon if one was precomputed for this exact GW, instead
    # of this script's own single-gameweek throwaway call below. shared_horizon_for_run only
    # ends up non-None when it's a genuine full-horizon match -- never the narrower fallback
    # dict, which would silently truncate recommend_best_move()'s own planning horizon if
    # forwarded into it.
    shared_horizon = tp.load_shared_horizon_ep_versions_from_env()
    if shared_horizon is not None and plan_for_gameweek in shared_horizon:
        horizon_ep_versions, shared_horizon_for_run = shared_horizon, shared_horizon
    else:
        horizon_ep_versions = tp.compute_horizon_ep(
            con, calibration_asof_date, TARGET_SEASON, plan_for_gameweek, ts_mv, mm_mv, 1,
            PARAM_VERSIONS["scoring_params_version"], PARAM_VERSIONS["bps_params_version"], PARAM_VERSIONS["tau_params_version"],
            PARAM_VERSIONS["rho_residual_params_version"], PARAM_VERSIONS["corr_params_version"],
        )
        shared_horizon_for_run = None
    ep_mv, un_mv = horizon_ep_versions[plan_for_gameweek]

    state_version = tp.bootstrap_from_real_squad(con, calibration_asof_date, TARGET_SEASON, current_event, ep_mv, un_mv, squad)
    current_holdings = tp._read_holdings(con, state_version)

    base_state = dict(
        entry_id=entry_id, calibration_asof_date=calibration_asof_date, target_season=TARGET_SEASON,
        target_gameweek=plan_for_gameweek, input_state_version=state_version,
        ts_model_version=ts_mv, mm_model_version=mm_mv, **PARAM_VERSIONS,
    )

    # Every scenario below shares the exact same base_state, so its baseline leg (the
    # unperturbed recommend_best_move() call) is byte-for-byte the same deterministic result
    # every time -- computed once here and reused, instead of apply_scenario() silently paying
    # for a full transfer_planner.run() pass (multi-gameweek EP + all-4-chip-evaluation, not
    # cheap) again on every one of the up to 6 scenarios below. Real, measured cost cut, not a
    # hypothetical one: this was the dominant contributor to this script's own wall-clock time.
    print("[scenario] computing shared baseline decision...")
    baseline = de.recommend_best_move(con, **base_state, include_sensitivity=False, horizon_ep_versions=shared_horizon_for_run)

    rows = []
    for player_uid in bench_player_uids(current_holdings):
        print(f"[scenario] lineup_change (starting) for {player_uid}...")
        result = scen.apply_scenario(
            con, base_state, scen.Scenario(kind="lineup_change", player_uid=player_uid, starting=True), baseline=baseline,
        )
        rows.append(scenario_result_row(player_uid, result))
    # Most-actionable first: the bench player whose hypothetical start would improve the plan
    # the most.
    rows.sort(key=lambda r: r["delta_ep"], reverse=True)

    armband_rows = []
    for role, player_uid in armband_uids(current_holdings).items():
        print(f"[scenario] injury for {role} ({player_uid})...")
        result = scen.apply_scenario(con, base_state, scen.Scenario(kind="injury", player_uid=player_uid), baseline=baseline)
        row = scenario_result_row(player_uid, result)
        row["role"] = role
        armband_rows.append(row)

    data_asof = calibration_asof_date.isoformat()
    all_uids = {r["player_uid"] for r in rows} | {r["player_uid"] for r in armband_rows}
    names = reporting.resolve_player_names(con, all_uids)
    for r in rows + armband_rows:
        r["player_name"] = names.get(r["player_uid"], r["player_uid"])

    payload = {
        "entry_id": entry_id, "gw": plan_for_gameweek, "data_asof": data_asof,
        "bench_what_ifs": rows, "armband_what_ifs": armband_rows,
    }

    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DASHBOARD_DIR / f"scenarios_{entry_id}_{plan_for_gameweek}_{data_asof}.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    # Stable-named copy, same "PWA needs a fixed, predictable path" convention as every other
    # roadmap-feature export in this project (app_team_<id>.json, decision_<id>_latest.json, ...).
    (DASHBOARD_DIR / f"scenarios_{entry_id}_latest.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[run_scenarios] wrote {out_path}")
    for r in rows:
        flip_note = f" -- FLIPS plan to '{r['perturbed_action']}'" if r["flipped"] else ""
        print(f"  bench: {r['player_name']}: {r['delta_ep']:+.2f} EP{flip_note}")
    for r in armband_rows:
        flip_note = f" -- FLIPS plan to '{r['perturbed_action']}'" if r["flipped"] else ""
        print(f"  {r['role']}: {r['player_name']}: {r['delta_ep']:+.2f} EP{flip_note}")

    con.close()


if __name__ == "__main__":
    main()
