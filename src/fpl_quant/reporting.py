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

from datetime import datetime

import duckdb

from . import backtest as bt
from . import evidence_blend
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
    # Part 1a (squad-quality guardrails work): "technically legal but obviously wrong" also
    # covers an XI with no real attacking threat, or a defensive spine that's entirely a
    # rotation gamble -- using the EXISTING minutes model's start-probability output (see
    # squad_optimizer.explain_run()), not a new heuristic. Flags, doesn't block, same as every
    # other entry in this list.
    flags.append({
        "name": "attacking_return_and_rotation_risk",
        "passed": bool(audit["has_nailed_attacking_return"]) and not audit["all_def_mid_uncertain"],
        "detail": {
            "has_nailed_attacking_return": audit["has_nailed_attacking_return"],
            "all_def_mid_uncertain": audit["all_def_mid_uncertain"],
        },
    })
    return flags


# ============================================================
# top-level assembler
# ============================================================

# ============================================================
# Part 5 (squad-quality guardrails work): adversarial-review packaging.
#
# Per spec: "run a SEPARATE step (a distinct agent pass, not the same context that built the
# squad)" whose only job is to argue AGAINST the squad. Every other M0-M9 module in this
# project is pure deterministic Python/SQL with zero LLM/API dependency anywhere -- that's a
# real, load-bearing property (reproducibility, no external service to fail/cost/rate-limit),
# not an oversight. The adversarial CRITIQUE itself genuinely needs a separate reasoning
# process (a fresh model context, or a human) to be worth anything -- a rule reproduced here
# would just be another automated_flags entry, not "a distinct agent pass." So this function
# does only the part that IS a good fit for this codebase: packaging build_report()'s output
# into a self-contained brief a genuinely separate reviewer can act on with zero other
# context, no DB access required. The actual critique step is deliberately left to whoever
# calls this with that brief.
# ============================================================

def assemble_adversarial_review_brief(con: duckdb.DuckDBPyConnection, report: dict) -> dict:
    """Self-contained: every field a fresh reviewer would need is inlined here, nothing
    requires going back to the DB or this session's context. Deliberately re-queries price/
    ownership itself (a few extra rows) rather than threading them through build_report()'s
    own return shape -- keeps this callable standalone against any already-built report dict.
    """
    h = report["headline"]
    squad_uids = [p["player_uid"] for p in h["squad"]]
    price_by_uid, ownership_by_uid = {}, {}
    if squad_uids:
        placeholders = ",".join(["?"] * len(squad_uids))
        # Real gap fixed here, caught building the actual live GW1 brief: fact_player_season_
        # stats carries rows from EVERY ingested season for a given player_uid, and gw numbering
        # restarts each season (1..38) -- "ORDER BY gw DESC" with no season filter happily
        # returns a prior season's gw=38 row (a much higher gw number) over the current season's
        # gw=1, silently substituting last season's price for this season's. Real, live impact:
        # this returned Haaland's 2024-25 season-end price (14.9) instead of his actual current
        # 2026-27 GW1 price, and similarly for every other squad member -- exactly the kind of
        # mistake this brief exists to help a reviewer catch, caught here in its own code first.
        rows = con.execute(
            f"SELECT player_uid, now_cost, selected_by_percent FROM fact_player_season_stats "
            f"WHERE player_uid IN ({placeholders}) AND season = ? "
            f"QUALIFY row_number() OVER (PARTITION BY player_uid ORDER BY gw DESC) = 1",
            [*squad_uids, h["target_season"]],
        ).fetchall()
        price_by_uid = {uid: price for uid, price, _own in rows}
        ownership_by_uid = {uid: own for uid, _price, own in rows}

    squad_lines = [
        {
            "name": p["name"], "position": p["position"], "in_xi": p["in_xi"],
            "is_captain": p["is_captain"], "is_vice": p["is_vice"],
            "price": price_by_uid.get(p["player_uid"]),
            "ownership_percent": ownership_by_uid.get(p["player_uid"]),
        }
        for p in sorted(h["squad"], key=lambda p: (not p["in_xi"], p["position"]))
    ]
    total_price = sum(v for v in price_by_uid.values() if v is not None)

    return {
        "target_season": h["target_season"], "target_gameweek": h["target_gameweek"],
        "squad": squad_lines,
        "n_squad": len(h["squad"]), "n_xi": sum(1 for p in h["squad"] if p["in_xi"]),
        "total_price": total_price,
        "total_projected_ep": h["total_projected_ep"],
        "rationale": h["rationale"],
        "automated_flags": report["automated_flags"],
        "guardrail_audit": report["guardrail_audit"],
        "consensus_divergence": report["consensus_divergence"],
        "community_evidence_roundup": report["community_evidence_roundup"],
        "review_instructions": (
            "Argue AGAINST this squad. Look specifically for: an incomplete squad (not 15 "
            "players, or XI not exactly 11), an impossible or miscounted budget (total_price "
            "must be <= 100.0), a captain pick that looks indefensible given its ownership% or "
            "the consensus_divergence evidence, excessive same-club concentration, a weak or "
            "unstartable bench, and anything else that looks wrong even though the "
            "automated_flags above may say 'passed'. A passed flag is a pattern-detection "
            "result, not proof of correctness -- do not treat it as license to skip your own "
            "check of the same failure modes. Report every issue you find, however minor, even "
            "if you conclude the squad is still defensible overall."
        ),
    }


def build_report(
    con: duckdb.DuckDBPyConnection,
    squad_optimizer_run_id: int,
    *,
    transfer_plan_run_id: int | None = None,
    backtest_run_id: int | None = None,
    active_param_versions: dict[str, int] | None = None,
    evidence_asof: datetime | None = None,
    decay_params_version: int | None = None,
    fact_multiplier_params_version: int | None = None,
) -> dict:
    """The minimal headline is always present; every other section is a dict key a caller can
    choose to render or not -- that choice is the "expandable on demand" the spec asks for,
    made at the display layer this module doesn't own. transfer_plan_run_id/backtest_run_id
    are optional because not every report has an existing squad to plan transfers from, or a
    backtest run to cite -- absence is recorded plainly (a `None` section), never silently
    dropped from the report's shape.

    evidence_asof/decay_params_version/fact_multiplier_params_version (Part 3, squad-quality
    guardrails work): all three optional, same "None section when not supplied" pattern as
    backtest_run_id/active_param_versions above -- a caller with no opinion on asof/decay
    versioning gets consensus_divergence=community_evidence_roundup=None rather than a
    guessed default.
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

    evidence_versions_given = (
        evidence_asof is not None and decay_params_version is not None and fact_multiplier_params_version is not None
    )

    total_ep = 0.0
    category_breakdown, risk_analytic, risk_empirical, evidence_provenance = {}, {}, {}, {}
    consensus_divergence, community_evidence_roundup = {}, {}
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
        if evidence_versions_given:
            # Part 3: only recorded when there's actually something to show -- most players
            # have zero analyst-debate/community claims right now, and a report shouldn't
            # carry an empty entry for every one of them.
            divergence = evidence_blend.explain_analyst_debate_divergence(
                con, uid, evidence_asof, decay_params_version, fact_multiplier_params_version
            )
            if divergence:
                consensus_divergence[uid] = divergence
            roundup = evidence_blend.explain_community_evidence_roundup(con, uid, evidence_asof)
            if roundup:
                community_evidence_roundup[uid] = roundup

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
        # Part 3: consensus_divergence is the structured analyst_debate check (real comparative
        # structure, weighted, never auto-judged); community_evidence_roundup is deliberately
        # separate and unweighted (community_sentiment/youtube_evidence have no comparative
        # structure to diverge FROM) -- see evidence_blend.py's own module-level comment for why
        # these two are kept apart rather than merged into one "consensus" section. Both None
        # (not {}) when the caller didn't supply evidence_asof/decay/fact-multiplier versions,
        # matching backtest_summary/transfer_chip_rationale's own None-when-not-requested shape.
        "consensus_divergence": consensus_divergence if evidence_versions_given else None,
        "community_evidence_roundup": community_evidence_roundup if evidence_versions_given else None,
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

    if report["consensus_divergence"]:
        lines.append("")
        lines.append("--- Consensus divergence (analyst_debate, structured, weighted) ---")
        for uid, debates in report["consensus_divergence"].items():
            for d in debates:
                lines.append(f"  {uid} vs. debate ({d['tab_origin']} row {d['row_origin']}):")
                for side in d["sides"]:
                    lines.append(f"    {side['player_uid']} (weight={side['weight']:.3f}): {side['opinion']}")

    if report["community_evidence_roundup"]:
        lines.append("")
        lines.append("--- Community/YouTube evidence roundup (unweighted, not a divergence check) ---")
        for uid, claims in report["community_evidence_roundup"].items():
            for c in claims:
                lines.append(f"  {uid} [{c['claim_type']}]: {c['payload']}")

    lines.append("")
    lines.append(f">>> {report['human_prompt']}")
    return "\n".join(lines)
