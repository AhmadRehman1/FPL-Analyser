"""Priority 2 addition: adversarial self-check pass.

A separate process that actively argues AGAINST a generated squad before it's shown to a
human -- re-validates the hard constraints from scratch (not trusting solve()'s own report of
success), and composes several of this project's own sibling checks (consensus-divergence,
club concentration, bench weakness) into one holistic "here's what's wrong with this squad"
pass. Per spec, findings are surfaced regardless of whether the squad passed every individual
check -- adversarial_review() always returns every finding it checked, `triggered` True or
False, deliberately NOT filtered down to only the failures. A squad passing every automated
flag (reporting.compute_automated_flags) can still be worth this second, adversarial look --
the two are not the same mechanism (compute_automated_flags is pattern-detection surfaced
alongside a human prompt; this is a standing, always-run counter-argument).
"""

from datetime import datetime

import duckdb

from . import consensus_check as cc
from . import minutes_model as mm
from . import squad_optimizer as so_mod


def adversarial_review(
    con: duckdb.DuckDBPyConnection,
    squad_optimizer_run_id: int,
    asof: datetime,
    decay_params_version: int,
    fact_multiplier_params_version: int,
    consensus_price_band: float,
    consensus_divergence_ratio_threshold: float,
    bench_p_start_threshold: float,
) -> list[dict]:
    findings = []

    n_squad, n_xi = con.execute(
        "SELECT count(*) FILTER (WHERE in_squad), count(*) FILTER (WHERE in_xi) "
        "FROM squad_optimizer_selections WHERE run_id = ?", [squad_optimizer_run_id],
    ).fetchone()
    findings.append({
        "check": "squad_completeness", "triggered": not (n_squad == 15 and n_xi == 11),
        "detail": f"{n_squad}/15 in squad, {n_xi}/11 in XI",
    })

    run_row = con.execute(
        "SELECT ep_model_version, uncertainty_model_version, target_season FROM squad_optimizer_runs WHERE run_id = ?",
        [squad_optimizer_run_id],
    ).fetchone()
    if not run_row:
        raise ValueError(f"no squad_optimizer_runs row for run_id={squad_optimizer_run_id}")
    ep_model_version, uncertainty_model_version, target_season = run_row

    candidates = so_mod.fetch_candidate_pool(con, ep_model_version, uncertainty_model_version, target_season)
    price_by_uid = {c["player_uid"]: c["price"] for c in candidates}
    squad_uids = {
        r[0] for r in con.execute(
            "SELECT player_uid FROM squad_optimizer_selections WHERE run_id = ? AND in_squad", [squad_optimizer_run_id]
        ).fetchall()
    }
    # sum() over only the players actually resolvable in the current candidate pool -- a
    # squad member who's since dropped out of the pool (e.g. a stale run re-checked against a
    # newer ep_model_version) can't have their price re-verified, and silently treating a
    # missing price as 0 would UNDER-count total spend, not over -- the wrong direction for a
    # check whose whole point is catching an impossible (too-high) budget.
    total_price = sum(price_by_uid[uid] for uid in squad_uids if uid in price_by_uid)
    findings.append({
        "check": "budget_legality", "triggered": total_price > so_mod.BUDGET + 1e-6,
        "detail": f"total squad price {total_price:.1f} vs budget {so_mod.BUDGET:.1f} "
                   f"({len(squad_uids & price_by_uid.keys())}/{len(squad_uids)} squad players priced)",
    })

    captain_row = con.execute(
        "SELECT player_uid FROM squad_optimizer_selections WHERE run_id = ? AND is_captain", [squad_optimizer_run_id]
    ).fetchone()
    captain_uid = captain_row[0] if captain_row else None
    consensus_flags = cc.flag_consensus_divergent_picks(
        con, squad_optimizer_run_id, asof, decay_params_version, fact_multiplier_params_version,
        consensus_price_band, consensus_divergence_ratio_threshold,
    )
    captain_flag = next((f for f in consensus_flags if f["selected_player_uid"] == captain_uid), None)
    findings.append({
        "check": "consensus_defying_captain", "triggered": captain_flag is not None,
        "detail": captain_flag if captain_flag else f"captain={captain_uid}: no meaningfully better-regarded same-price alternative found",
    })

    audit = so_mod.explain_run(con, squad_optimizer_run_id)
    concentration_hit = bool(audit["clubs_at_squad_cap"] or audit["clubs_at_xi_cap"])
    findings.append({
        "check": "concentration_risk", "triggered": concentration_hit,
        "detail": {"clubs_at_squad_cap": audit["clubs_at_squad_cap"], "clubs_at_xi_cap": audit["clubs_at_xi_cap"]},
    })

    mm_version_row = con.execute(
        "SELECT minutes_model_version FROM uncertainty_model_versions WHERE model_version = ?", [uncertainty_model_version]
    ).fetchone()
    bench_uids = [
        r[0] for r in con.execute(
            "SELECT player_uid FROM squad_optimizer_selections WHERE run_id = ? AND in_squad AND NOT in_xi",
            [squad_optimizer_run_id],
        ).fetchall()
    ]
    weak_bench, weak_bench_detail = False, "no p_start data available"
    if mm_version_row is not None and bench_uids:
        p_start_by_uid = mm.p_start_final_by_player(con, mm_version_row[0], bench_uids)
        weak = sorted(
            uid for uid in bench_uids
            if p_start_by_uid.get(uid) is not None and p_start_by_uid[uid] < bench_p_start_threshold
        )
        weak_bench = bool(weak)
        weak_bench_detail = f"bench players below {bench_p_start_threshold} p_start_final: {weak}"
    findings.append({"check": "weak_bench", "triggered": weak_bench, "detail": weak_bench_detail})

    return findings
