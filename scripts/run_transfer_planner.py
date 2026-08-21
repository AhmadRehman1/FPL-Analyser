"""M8: Transfer & Chip Strategy Planner -- bootstraps the manager's current holdings from
M5's real GW1 2026-27 squad_optimizer_runs selection, then plans transfers/chips for GW2.

Usage (from repo root):
    .venv/Scripts/python scripts/run_transfer_planner.py

Depends on scripts/run_ingestion.py having already run (needs a real squad_optimizer_runs row
for 2026-2027 GW1, and GW2-6 fixtures already ingested -- both already true in this project's
real database). Bootstrapping is a one-time action per manager -- running this script again
would create a second bootstrap state; a real usage pattern would call
transfer_planner.apply_recommendation() to advance state week over week instead of
re-bootstrapping, but that's a human decision (which recommendation to accept), not something
this script does automatically.
"""

import json
import sys
import time
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import db, transfer_planner as tp  # noqa: E402

TARGET_SEASON = "2026-2027"
PLAN_FOR_GAMEWEEK = 2


def main() -> None:
    con = db.connect()
    tp.seed_v1_params(con)

    real_run_id = con.execute(
        "SELECT run_id FROM squad_optimizer_runs WHERE target_season = ? AND target_gameweek = 1 "
        "AND is_manager_snapshot = FALSE",
        [TARGET_SEASON],
    ).fetchone()
    if not real_run_id:
        raise SystemExit(f"no real squad_optimizer_runs row for {TARGET_SEASON} GW1 -- run scripts/run_ingestion.py first")
    real_run_id = real_run_id[0]

    state_version = tp.bootstrap_from_squad_optimizer_run(con, real_run_id)
    n_players = len(tp._read_holdings(con, state_version))
    print(f"[bootstrap] state_version={state_version} from squad_optimizer run_id={real_run_id}, {n_players} players")

    t0 = time.time()
    run_id = tp.run(
        con,
        calibration_asof_date=date.today(),
        target_season=TARGET_SEASON,
        target_gameweek=PLAN_FOR_GAMEWEEK,
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
        # Opt in the same qualitative-evidence adjustments scripts/run_ingestion.py now turns
        # on for GW1 (see that script's expected_points.run() call) -- otherwise this multi-
        # gameweek horizon would silently plan transfers/chips off the un-adjusted EP.
        set_piece_params_version=1,
        decay_params_version=1, fact_multiplier_params_version=1, role_shift_params_version=1,
        swing_params_version=1,
    )
    print(f"[transfer_planner.run] {time.time() - t0:.1f}s -> run_id={run_id}")

    print("\n--- top 5 transfer recommendations ---")
    recs = con.execute(
        "SELECT rank, player_out, player_in, horizon_value_gain, transfer_cost, net_value "
        "FROM transfer_recommendations WHERE run_id = ? ORDER BY rank LIMIT 5", [run_id],
    ).fetchall()
    for rank, out_uid, in_uid, gain, cost, net in recs:
        out_name = con.execute("SELECT canonical_name FROM dim_player WHERE player_uid = ?", [out_uid]).fetchone()[0]
        in_name = con.execute("SELECT canonical_name FROM dim_player WHERE player_uid = ?", [in_uid]).fetchone()[0]
        print(f"  #{rank}: OUT {out_name} -> IN {in_name} | gain={gain:.2f} cost={cost} net={net:.2f}")

    print("\n--- chip evaluations ---")
    chips = con.execute(
        "SELECT chip_type, recommended, score_or_gain, gw19_urgent_flag, detail FROM chip_evaluations WHERE run_id = ?",
        [run_id],
    ).fetchall()
    for chip_type, recommended, score, urgent, detail in chips:
        print(f"  {chip_type}: recommended={recommended} score={score} gw19_urgent={urgent}")
        d = json.loads(detail)
        if chip_type == "triple_captain" and d.get("all_candidates"):
            top3 = d["all_candidates"][:3]
            for c in top3:
                name = con.execute("SELECT canonical_name FROM dim_player WHERE player_uid = ?", [c["player_uid"]]).fetchone()[0]
                print(f"      {name}: tc_score={c['tc_score']:.3f} mean={c['mean_total']:.3f} var={c['var_total']:.3f}")

    con.close()


if __name__ == "__main__":
    main()
