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

from . import adversarial_check as ac
from . import backtest as bt
from . import consensus_check as cc
from . import expected_points as ep
from . import minutes_model as mm
from . import monte_carlo
from . import params as params_mod
from . import squad_optimizer
from . import transfer_planner as tp
from . import uncertainty as un

HUMAN_PROMPT = "Does this squad look defensible to you?"


def seed_v1_params(con: duckdb.DuckDBPyConnection) -> None:
    # Priority 2's own exact spec values ("no clearly-nailed (p_start_final >= 0.75)... every
    # DEF/MID below a rotation-risk threshold (0.55)") -- not invented literals here, pinned
    # verbatim from the spec the same way M5's seed_v1_params pins lambda_value=0.15.
    params_mod.write_param(con, "sanity_check_params", 1, "2026-08-10", "nailed_p_start_threshold", value_numeric=0.75)
    params_mod.write_param(con, "sanity_check_params", 1, "2026-08-10", "rotation_risk_p_start_threshold", value_numeric=0.55)
    # Priority 2 -- consensus-divergence: invented v1 defaults (same invented-default status
    # as every other unpinned constant in this project) -- +/-GBP0.5m is a real, motivating
    # "same price bracket" a manager would actually compare players within (matches the real
    # incident this MVP is scoped to catch); 0.2 (a 20% higher blended-evidence-weight bar)
    # is sized to require a REAL gap, not just noise between two similarly-covered players.
    params_mod.write_param(con, "consensus_check_params", 1, "2026-08-10", "price_band", value_numeric=0.5)
    params_mod.write_param(con, "consensus_check_params", 1, "2026-08-10", "divergence_ratio_threshold", value_numeric=0.2)


# ============================================================
# automated sanity-check flags -- pattern-detection, not self-certification
# ============================================================

def compute_automated_flags(
    con: duckdb.DuckDBPyConnection, squad_optimizer_run_id: int, sanity_check_params_version: int | None = None,
) -> list[dict]:
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

    # Priority 2 -- solve-quality transparency, surfaced as its own flag alongside the others
    # rather than only living in explain_run()'s raw audit dict -- a time/gap-limited solve
    # is exactly the kind of thing this section exists to make visible to a human, not just
    # technically retrievable.
    flags.append({
        "name": "solve_proved_optimal", "passed": bool(audit["solve_proved_optimal"]),
        "detail": f"solver_status={audit['solver_status']} mip_gap={audit['mip_gap']}",
    })

    # Priority 2 -- sanity-check flag: no clearly-nailed MID/FWD attacking return, or every
    # DEF/MID a rotation risk. Scoped to the XI (not the full squad) -- bench rotation risk is
    # already handled at squad level by solve()'s own bench-quality floor; what matters for
    # THIS flag is whether the team that actually plays this week has a real attacking core
    # and isn't entirely made up of players who might not start. Opt-in via
    # sanity_check_params_version (None skips both -- same additive, backward-compatible
    # convention as every other Priority 1/2 feature this session).
    if sanity_check_params_version is not None:
        nailed_threshold, _ = params_mod.resolve_param(con, "sanity_check_params", "nailed_p_start_threshold", sanity_check_params_version)
        rotation_threshold, _ = params_mod.resolve_param(con, "sanity_check_params", "rotation_risk_p_start_threshold", sanity_check_params_version)

        xi_rows = con.execute(
            "SELECT s.player_uid, dp.position FROM squad_optimizer_selections s "
            "JOIN dim_player dp ON dp.player_uid = s.player_uid WHERE s.run_id = ? AND s.in_xi",
            [squad_optimizer_run_id],
        ).fetchall()
        uv_row = con.execute(
            "SELECT uncertainty_model_version FROM squad_optimizer_runs WHERE run_id = ?", [squad_optimizer_run_id]
        ).fetchone()
        nailed_attacking_return_note = "no p_start data available"
        rotation_risk_note = "no p_start data available"
        nailed_ok, rotation_ok = True, True
        if uv_row is not None:
            mm_version_row = con.execute(
                "SELECT minutes_model_version FROM uncertainty_model_versions WHERE model_version = ?", [uv_row[0]]
            ).fetchone()
            if mm_version_row is not None:
                p_start_by_uid = mm.p_start_final_by_player(con, mm_version_row[0], [uid for uid, _pos in xi_rows])
                attacking = [p_start_by_uid[uid] for uid, pos in xi_rows if pos in ("Midfielder", "Forward") and uid in p_start_by_uid]
                def_mid = [p_start_by_uid[uid] for uid, pos in xi_rows if pos in ("Defender", "Midfielder") and uid in p_start_by_uid]
                if attacking:
                    nailed_ok = any(p >= nailed_threshold for p in attacking)
                    nailed_attacking_return_note = f"XI MID/FWD p_start_final values: {sorted(attacking, reverse=True)}"
                if def_mid:
                    rotation_ok = not all(p < rotation_threshold for p in def_mid)
                    rotation_risk_note = f"XI DEF/MID p_start_final values: {sorted(def_mid)}"
        flags.append({"name": "nailed_attacking_return", "passed": nailed_ok, "detail": nailed_attacking_return_note})
        flags.append({"name": "rotation_risk_def_mid", "passed": rotation_ok, "detail": rotation_risk_note})

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
    ownership_params_version: int | None = None,
    sanity_check_params_version: int | None = None,
    consensus_check_params_version: int | None = None,
    evidence_decay_params_version: int | None = None,
    evidence_fact_multiplier_params_version: int | None = None,
    bench_quality_params_version: int | None = None,
    report_asof: datetime | None = None,
) -> dict:
    """The minimal headline is always present; every other section is a dict key a caller can
    choose to render or not -- that choice is the "expandable on demand" the spec asks for,
    made at the display layer this module doesn't own. transfer_plan_run_id/backtest_run_id/
    ownership_params_version are optional because not every report has an existing squad to
    plan transfers from, a backtest run to cite, or an ownership-params version pinned for the
    EO computation -- absence is recorded plainly (a `None` section), never silently dropped
    from the report's shape.

    consensus_divergence needs consensus_check_params_version, evidence_decay_params_version,
    evidence_fact_multiplier_params_version, and report_asof ALL set together (an evidence-
    weight computation is meaningless without an as-of date to decay claims against); absent
    any one of them, the whole section is None rather than guessing a default asof.
    adversarial_review additionally needs bench_quality_params_version, reusing the SAME
    min_bench_p_start_probability threshold solve()'s own bench-quality floor already
    resolves (one tunable "is this player a rotation risk" number, not two independently-
    drifting ones for two different consumers of the identical concept)."""
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
    automated_flags = compute_automated_flags(con, squad_optimizer_run_id, sanity_check_params_version)

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
        # Priority 1 -- EO-adjusted captain-risk, its own explicit decision distinct from
        # squad selection itself (per Priority 1's own framing) -- never changes which player
        # is captained, only reports on the rank-risk profile of the choice already made.
        "captain_risk_eo": (
            squad_optimizer.explain_captain_risk_eo(con, squad_optimizer_run_id, ownership_params_version)
            if ownership_params_version is not None else None
        ),
        "consensus_divergence": _consensus_divergence_section(
            con, squad_optimizer_run_id, consensus_check_params_version,
            evidence_decay_params_version, evidence_fact_multiplier_params_version, report_asof,
        ),
        "adversarial_review": _adversarial_review_section(
            con, squad_optimizer_run_id, consensus_check_params_version,
            evidence_decay_params_version, evidence_fact_multiplier_params_version, report_asof,
            bench_quality_params_version,
        ),
        "human_prompt": HUMAN_PROMPT,
    }


def _consensus_divergence_section(
    con, squad_optimizer_run_id, consensus_check_params_version,
    evidence_decay_params_version, evidence_fact_multiplier_params_version, report_asof,
):
    if None in (consensus_check_params_version, evidence_decay_params_version, evidence_fact_multiplier_params_version, report_asof):
        return None
    price_band, _ = params_mod.resolve_param(con, "consensus_check_params", "price_band", consensus_check_params_version)
    ratio, _ = params_mod.resolve_param(con, "consensus_check_params", "divergence_ratio_threshold", consensus_check_params_version)
    return cc.flag_consensus_divergent_picks(
        con, squad_optimizer_run_id, report_asof, evidence_decay_params_version,
        evidence_fact_multiplier_params_version, price_band, ratio,
    )


def _adversarial_review_section(
    con, squad_optimizer_run_id, consensus_check_params_version,
    evidence_decay_params_version, evidence_fact_multiplier_params_version, report_asof,
    bench_quality_params_version,
):
    if None in (
        consensus_check_params_version, evidence_decay_params_version, evidence_fact_multiplier_params_version,
        report_asof, bench_quality_params_version,
    ):
        return None
    price_band, _ = params_mod.resolve_param(con, "consensus_check_params", "price_band", consensus_check_params_version)
    ratio, _ = params_mod.resolve_param(con, "consensus_check_params", "divergence_ratio_threshold", consensus_check_params_version)
    bench_threshold, _ = params_mod.resolve_param(
        con, "bench_quality_params", "min_bench_p_start_probability", bench_quality_params_version
    )
    return ac.adversarial_review(
        con, squad_optimizer_run_id, report_asof, evidence_decay_params_version,
        evidence_fact_multiplier_params_version, price_band, ratio, bench_threshold,
    )


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

    if report["captain_risk_eo"]:
        cr = report["captain_risk_eo"]
        lines.append("")
        eo_str = f"{cr['captain_eo']:.1f}%" if cr["captain_eo"] is not None else "unknown"
        lines.append(f"--- Captain rank-risk (EO-adjusted): {cr['posture_label']} (captain EO {eo_str}) ---")

    if report["consensus_divergence"]:
        lines.append("")
        lines.append("--- Consensus-divergence flags ---")
        for f in report["consensus_divergence"]:
            lines.append(
                f"  {f['selected_player_name']} ({f['position']}) -- consider "
                f"{f['alternative_player_name']} (evidence weight {f['alternative_evidence_weight']:.2f} "
                f"vs {f['selected_evidence_weight']:.2f})"
            )

    if report["adversarial_review"]:
        lines.append("")
        lines.append("--- Adversarial self-check ---")
        for f in report["adversarial_review"]:
            status = "FLAGGED" if f["triggered"] else "clear"
            lines.append(f"  [{status}] {f['check']}: {f['detail']}")

    if report["parameter_transparency"]:
        n_invented = sum(1 for row in report["parameter_transparency"] if not row["backtested_via_m7"])
        lines.append("")
        lines.append(f"--- Parameter transparency: {n_invented}/{len(report['parameter_transparency'])} still invented, not yet backtested via M7 ---")

    lines.append("")
    lines.append(f">>> {report['human_prompt']}")
    return "\n".join(lines)
