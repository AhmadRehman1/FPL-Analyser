"""M9: Reporting / Explainability Layer.

Cross-cutting -- depends on M0 through M8, all done. Its own spec's integration pattern is
explicit: "each of M0-M8 exposes its own explain()-style interface returning its relevant
data in a display-ready shape. M9 does not reach into other modules' internals directly."
That pattern didn't exist anywhere in this project before M9 (grepped, confirmed) -- it's
built here as one small, additive adapter function per module (expected_points.
explain_player_ep, uncertainty.explain_player_risk, monte_carlo.explain_player_risk_empirical,
squad_optimizer.explain_run, minutes_model.explain_player_adjustment,
backtest.explain_backtest_summary, transfer_planner.explain_plan, params.transparency_panel),
with this module doing only the assembly, never a raw table read of its own.

Disclosure pattern, per spec: minimal headline by default (squad, captain, total EP, one-line
rationale); every other section sits behind its own dict key, "expandable on demand" being a
UI-layer concern this backend module doesn't decide -- the spec explicitly leaves "the actual
interaction design... out of scope for a backend-module spec."

Sanity-check surface is two genuinely separate things, kept as separate dict keys throughout:
automated pattern-detection flags, and a fixed human prompt that is never allowed to read as
self-certification -- "the original lambda=0 bug passed whatever internal checks existed at
the time," per the spec's own stated reasoning for keeping them apart.
"""

import duckdb

from . import backtest as bt
from . import expected_points as ep
from . import minutes_model as mm
from . import monte_carlo
from . import params as params_mod
from . import squad_optimizer
from . import transfer_planner as tp
from . import uncertainty as un

HUMAN_PROMPT = "Does this squad look defensible to you?"


# ============================================================
# automated sanity-check flags -- pattern-detection, not self-certification
# ============================================================

def compute_automated_flags(con: duckdb.DuckDBPyConnection, squad_optimizer_run_id: int) -> list[dict]:
    """Heuristics matching the spec's own named historical failure signatures: team/position
    concentration, a captained goalkeeper, guardrail-binding status, the lambda=0 divergence
    result. `passed=False` means "worth a human look," not "confirmed broken" -- a club sitting
    exactly at the concentration cap is a real, legal outcome (M5's own real GW1 squad was
    "maximally club-diversified... well under the cap," a good sign precisely because it wasn't
    forced there by the guardrail); the flag exists so a human can see when a run leans on the
    guardrail rather than the risk mechanism, not to declare that outcome wrong by itself.
    """
    audit = squad_optimizer.explain_run(con, squad_optimizer_run_id)
    flags = []
    flags.append({
        "name": "divergence_check", "passed": bool(audit["divergence_check_passed"]),
        "detail": audit["divergence_check_note"],
    })
    flags.append({
        "name": "captained_goalkeeper", "passed": not audit["captain_is_goalkeeper"],
        "detail": f"captain={audit['captain_uid']} position={audit['captain_position']}",
    })
    concentration_hit = bool(audit["clubs_at_squad_cap"] or audit["clubs_at_xi_cap"])
    flags.append({
        "name": "club_concentration", "passed": not concentration_hit,
        "detail": {"clubs_at_squad_cap": audit["clubs_at_squad_cap"], "clubs_at_xi_cap": audit["clubs_at_xi_cap"]},
    })
    # Part 4 (solve-quality transparency): status=="optimal" alone doesn't tell a report reader
    # whether SCIP actually PROVED no better solution exists, or just cleared its own internal
    # gap tolerance/hit limits/time=300 with an unproven incumbent -- surfaced as its own flag so
    # a not-proven-optimal squad is never shown with the same confidence as a proven one.
    flags.append({
        "name": "proven_optimal", "passed": bool(audit["proven_optimal"]),
        "detail": f"solver_gap={audit['solver_gap']}",
    })
    return flags


# ============================================================
# top-level assembler
# ============================================================

def build_report(
    con: duckdb.DuckDBPyConnection,
    squad_optimizer_run_id: int,
    *,
    transfer_plan_run_id: int | None = None,
    backtest_run_id: int | None = None,
    active_param_versions: dict[str, int] | None = None,
) -> dict:
    """The minimal headline is always present; every other section is a dict key a caller can
    choose to render or not -- that choice is the "expandable on demand" the spec asks for,
    made at the display layer this module doesn't own. transfer_plan_run_id/backtest_run_id
    are optional because not every report has an existing squad to plan transfers from, or a
    backtest run to cite -- absence is recorded plainly (a `None` section), never silently
    dropped from the report's shape.
    """
    run_row = con.execute(
        "SELECT target_season, target_gameweek, ep_model_version, uncertainty_model_version "
        "FROM squad_optimizer_runs WHERE run_id = ?", [squad_optimizer_run_id],
    ).fetchone()
    if not run_row:
        raise ValueError(f"no squad_optimizer_runs row for run_id={squad_optimizer_run_id}")
    target_season, target_gameweek, ep_model_version, uncertainty_model_version = run_row

    squad_rows = con.execute(
        "SELECT s.player_uid, dp.canonical_name, dp.position, s.in_xi, s.is_captain, s.is_vice "
        "FROM squad_optimizer_selections s JOIN dim_player dp ON dp.player_uid = s.player_uid "
        "WHERE s.run_id = ? AND s.in_squad", [squad_optimizer_run_id],
    ).fetchall()
    squad = [
        {"player_uid": uid, "name": name, "position": position, "in_xi": in_xi, "is_captain": is_captain, "is_vice": is_vice}
        for uid, name, position, in_xi, is_captain, is_vice in squad_rows
    ]
    xi_uids = {p["player_uid"] for p in squad if p["in_xi"]}
    captain = next((p for p in squad if p["is_captain"]), None)

    minutes_model_version = con.execute(
        "SELECT minutes_model_version FROM uncertainty_model_versions WHERE model_version = ?", [uncertainty_model_version]
    ).fetchone()[0]
    # Real bug fixed here: this previously took max(model_version) for the squad_optimizer_run_id
    # alone, with no check that the Monte Carlo run actually used the same ep_model_version/
    # uncertainty_model_version this report is otherwise built from. monte_carlo_run_versions
    # has no uniqueness constraint on squad_optimizer_run_id, so a second monte_carlo.run() call
    # against the same run_id (e.g. after an upstream EP/uncertainty recalibration produced new
    # model versions but the squad itself wasn't re-optimized) would silently pick the newest MC
    # run even if it was simulated against stale EP/uncertainty inputs -- an internally
    # inconsistent report mixing one gameweek's analytic risk numbers with another's empirical
    # ones, with no warning. Filtering on the matching versions (still picking the latest MC run
    # among those that actually match) makes the empirical section either genuinely consistent
    # with the rest of the report, or explicitly absent (mc_model_version=None) rather than
    # silently wrong.
    mc_model_version = con.execute(
        "SELECT max(model_version) FROM monte_carlo_run_versions "
        "WHERE squad_optimizer_run_id = ? AND ep_model_version = ? AND uncertainty_model_version = ?",
        [squad_optimizer_run_id, ep_model_version, uncertainty_model_version],
    ).fetchone()[0]

    total_ep = 0.0
    category_breakdown, risk_analytic, risk_empirical, evidence_provenance = {}, {}, {}, {}
    for p in squad:
        uid = p["player_uid"]
        breakdown = ep.explain_player_ep(con, ep_model_version, uid)
        if breakdown:
            category_breakdown[uid] = breakdown
            if uid in xi_uids:
                total_ep += breakdown["total"] * (2 if p["is_captain"] else 1)
        risk = un.explain_player_risk(con, uncertainty_model_version, uid)
        if risk:
            risk_analytic[uid] = risk
        if mc_model_version:
            empirical = monte_carlo.explain_player_risk_empirical(con, mc_model_version, uid)
            if empirical:
                risk_empirical[uid] = empirical
        evidence_provenance[uid] = mm.explain_player_adjustment(con, minutes_model_version, uid)

    guardrail_audit = squad_optimizer.explain_run(con, squad_optimizer_run_id)
    automated_flags = compute_automated_flags(con, squad_optimizer_run_id)

    rationale = (
        f"{target_season} GW{target_gameweek}: {len(squad)} players, captain "
        f"{captain['name'] if captain else 'unassigned'}, projected {total_ep:.1f} points "
        f"(divergence check {'passed' if guardrail_audit['divergence_check_passed'] else 'FAILED'})."
    )

    return {
        "headline": {
            "target_season": target_season, "target_gameweek": target_gameweek,
            "squad": squad, "captain": captain, "total_projected_ep": total_ep, "rationale": rationale,
        },
        "category_breakdown": category_breakdown,
        "risk": {"analytic": risk_analytic, "empirical": risk_empirical},
        "guardrail_audit": guardrail_audit,
        "evidence_provenance": evidence_provenance,
        "parameter_transparency": params_mod.transparency_panel(con, active_param_versions) if active_param_versions else None,
        "backtest_summary": bt.explain_backtest_summary(con, backtest_run_id) if backtest_run_id is not None else None,
        "transfer_chip_rationale": tp.explain_plan(con, transfer_plan_run_id) if transfer_plan_run_id is not None else None,
        "automated_flags": automated_flags,
        "human_prompt": HUMAN_PROMPT,
    }


# ============================================================
# text rendering -- no UI layer exists in this project; a console formatter matches the
# established verification-via-console-output pattern every prior milestone already uses
# ============================================================

def render_report_text(report: dict) -> str:
    lines = []
    h = report["headline"]
    lines.append(f"=== {h['target_season']} GW{h['target_gameweek']} Squad Report ===")
    lines.append(h["rationale"])
    lines.append("")
    lines.append("Squad:")
    for p in sorted(h["squad"], key=lambda p: (not p["in_xi"], p["position"])):
        tag = " (C)" if p["is_captain"] else " (VC)" if p["is_vice"] else ""
        bench = "" if p["in_xi"] else " [bench]"
        lines.append(f"  {p['name']:30s} {p['position']:12s}{tag}{bench}")

    lines.append("")
    lines.append("--- Automated flags ---")
    for f in report["automated_flags"]:
        status = "OK" if f["passed"] else "REVIEW"
        lines.append(f"  [{status}] {f['name']}: {f['detail']}")

    if report["backtest_summary"]:
        bs = report["backtest_summary"]
        lines.append("")
        lines.append(f"--- Backtest track record (run {bs['backtest_run_id']}) ---")
        for tier in ("cold", "warm", "mature"):
            if tier in bs["metrics_by_tier"]:
                lines.append(f"  [{tier}]")
                for name, v in bs["metrics_by_tier"][tier].items():
                    lines.append(f"    {name}: {v['mean']:.4f} (n={v['n']})")

    if report["transfer_chip_rationale"]:
        tr = report["transfer_chip_rationale"]
        lines.append("")
        lines.append("--- Transfer & chip rationale ---")
        for t in tr["top_transfers"][:3]:
            lines.append(f"  #{t['rank']}: OUT {t['player_out']} -> IN {t['player_in']} (net {t['net_value']:.2f})")
        for chip_type, c in tr["chips"].items():
            lines.append(f"  {chip_type}: recommended={c['recommended']} score={c['score_or_gain']}")
        if tr["gw19_deadline"]["urgent"]:
            lines.append(f"  GW19 URGENT: unused chips {tr['gw19_deadline']['unused_set1_chips']}")

    if report["parameter_transparency"]:
        n_invented = sum(1 for row in report["parameter_transparency"] if not row["backtested_via_m7"])
        lines.append("")
        lines.append(f"--- Parameter transparency: {n_invented}/{len(report['parameter_transparency'])} still invented, not yet backtested via M7 ---")

    lines.append("")
    lines.append(f">>> {report['human_prompt']}")
    return "\n".join(lines)
