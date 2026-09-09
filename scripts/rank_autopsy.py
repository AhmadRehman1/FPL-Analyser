"""Priority 10 Phase B: the weekly rank autopsy.

For every completed 2026-27 gameweek that has a rival sample (fact_rival_squad_sample), score
three subjects against that field and decompose each one's rank gap:
  - model_optimal  -- the from-scratch squad_optimizer pick for that gameweek (the headline:
                      how the pure model would have ranked);
  - account:7139944, account:1305242 -- the two tracked real managers, whose real settled
                      overall rank is fetched too, as a calibration point for the estimate.

Writes data/dashboard/rank_autopsy.json (per-gameweek rows + a season-to-date attribution
roll-up + plain findings) and persists each score to fact_squad_rank_score.

Forward-only -- see src/fpl_quant/field_rank.py and docs/priority10_field_simulator_design.md.
This measures and attributes; it does not change the optimiser (that is Phase C). Best-effort
per subject: a failed account fetch never blocks the model_optimal scoring, which is the point.

Usage (from repo root, after scripts/run_rival_sample_ingestion.py):
    PYTHONPATH=src python scripts/rank_autopsy.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import app_export as ax, db, field_rank  # noqa: E402
from fpl_quant import ingest_fpl_entry_picks as ifp, ingest_workbook as iw  # noqa: E402

TARGET_SEASON = "2026-2027"
DASHBOARD_DIR = REPO_ROOT / "data" / "dashboard"
TRACKED_ACCOUNTS = {7139944: "ChatGPT template team", 1305242: "Matippy toes"}

# The signed attribution numbers that are meaningfully additive across gameweeks (points the
# choice cost, negative, or won, positive) -> the season-to-date roll-up.
_CUMULATIVE_KEYS = {
    "captaincy_vs_best_in_xi": ("captaincy", "vs_best_in_xi"),
    "captaincy_vs_field_captain": ("captaincy", "vs_field_captain"),
    "template_eo_weighted_points_bled": ("template_coverage", "eo_weighted_points_bled"),
    "differentials_net_surprise": ("differentials", "net_surprise"),
    "bench_points_left": ("bench", "points_left"),
}

_LIMITATIONS = [
    "Forward-only: FPL's API serves only current-season picks, so 2024-25/2025-26 cannot be rank-scored.",
    "estimated_rank is a sample projection (percentile x total_players), not an official FPL figure.",
    "Auto-substitution is not modelled -- bench_points_left is an upper bound on bench cost.",
    "Double/blank-gameweek rival squads are scored with realized points as-is.",
    "The rival sample's deep-rank reach depends on FPL's standings pagination -- see sample_shape_by_gameweek.",
]


# ============================================================
# per-subject scoring -- takes a resolved squad dict, so it is DB-testable without any
# squad_optimizer_runs / FPL-fetch scaffolding
# ============================================================

def _persist(con, event, subject, score, attribution, sample_bands, real_overall_rank):
    con.execute(
        "DELETE FROM fact_squad_rank_score WHERE season = ? AND event = ? AND subject = ? AND subject_id = ?",
        [TARGET_SEASON, event, subject, score["subject_id"]],
    )
    con.execute(
        "INSERT INTO fact_squad_rank_score (season, event, subject, subject_id, realized_points, "
        "n_rivals, n_beaten, n_tied, percentile, estimated_rank, total_players, real_overall_rank, "
        "sample_bands, attribution) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            TARGET_SEASON, event, subject, score["subject_id"], score["realized_points"],
            score["n_rivals"], score["n_beaten"], score["n_tied"], score["percentile"],
            score["estimated_rank"], score["total_players"], real_overall_rank,
            json.dumps(sample_bands), json.dumps(attribution),
        ],
    )


def score_subject(con, event, subject_key, squad, real_overall_rank, total_players):
    """Score one resolved squad for one gameweek against that gameweek's rival sample, persist
    it, and return its by_gameweek row -- or None when there is no settled sample to score
    against (n_rivals == 0). subject_key is "model_optimal" or "account:<entry_id>"; the
    stored `subject` is the part before the colon (the schema's CHECK), subject_id the rest."""
    score = field_rank.score_squad_rank(
        con, TARGET_SEASON, event, squad["xi"], squad["captain"], total_players,
    )
    if score["n_rivals"] == 0 or score["percentile"] is None:
        return None
    score["subject_id"] = squad["subject_id"]
    attribution = field_rank.attribute_rank_gap(
        con, TARGET_SEASON, event, squad["xi"], squad["captain"], squad["bench"],
        ep_model_version=squad.get("ep_model_version"),
    )
    shape = field_rank.sample_shape(con, TARGET_SEASON, event)
    _persist(con, event, subject_key.split(":", 1)[0], score, attribution, shape, real_overall_rank)
    return {
        "gameweek": event, "realized_points": score["realized_points"],
        "percentile": score["percentile"], "estimated_rank": score["estimated_rank"],
        "n_rivals": score["n_rivals"], "n_beaten": score["n_beaten"],
        "real_overall_rank": real_overall_rank, "attribution": attribution,
    }


def roll_up(by_gameweek):
    """Season-to-date sum of the additive attribution buckets for one subject."""
    cumulative = {key: 0.0 for key in _CUMULATIVE_KEYS}
    for row in by_gameweek:
        for key, (bucket, field) in _CUMULATIVE_KEYS.items():
            cumulative[key] += row["attribution"].get(bucket, {}).get(field, 0.0) or 0.0
    return {k: round(v, 2) for k, v in cumulative.items()}


def finding(subject_label, event, row):
    """One plain-language line: where this subject lost the most rank this gameweek."""
    attribution = row["attribution"]
    contributions = {
        key: attribution.get(bucket, {}).get(field, 0.0) or 0.0
        for key, (bucket, field) in _CUMULATIVE_KEYS.items()
        if key != "captaincy_vs_field_captain"  # vs_best_in_xi is the sharper of the two
    }
    worst_key = min(contributions, key=lambda k: contributions[k])
    worst = contributions[worst_key]
    rank_txt = f"~{row['estimated_rank']:,}" if row["estimated_rank"] else "n/a"
    head = (f"GW{event} ({subject_label}): {row['percentile']:.0f}th pctile of the sample, "
            f"projected rank {rank_txt} ({row['realized_points']} pts, beat "
            f"{row['n_beaten']}/{row['n_rivals']} rivals).")
    cap = attribution.get("captaincy", {})
    tmpl = attribution.get("template_coverage", {})
    if worst_key.startswith("captaincy") and worst < -1:
        detail = (f" Biggest leak: captaincy -- {cap.get('captain')} ({cap.get('captain_points')} pts) "
                  f"vs best-in-XI {cap.get('best_in_xi')}, {worst:+.0f} (armband doubles it).")
    elif worst_key.startswith("template") and worst < -1:
        missed = tmpl.get("missed", [])
        top = max(missed, key=lambda m: m["eo_weighted_points"], default=None) if missed else None
        detail = (f" Biggest leak: template -- owned {tmpl.get('template_owned')}/{tmpl.get('template_size')}"
                  + (f", missed {top['name']} ({top['eo']:.0f}% EO, {top['points']} pts)." if top else "."))
    elif worst_key == "bench_points_left" and worst < -3:
        detail = f" Biggest leak: {worst:+.0f} left on the bench (auto-subs not modelled)."
    elif worst_key == "differentials_net_surprise" and worst < -1:
        detail = f" Biggest leak: differentials underperformed model EP by {worst:+.1f}."
    else:
        detail = " No single dominant leak this gameweek."
    return head + detail


def assemble_payload(subjects, sample_shape_by_gameweek, total_players):
    """Finalise each subject (sort, roll up, latest) and build the JSON payload. Pure."""
    findings = []
    for entry in subjects.values():
        entry["by_gameweek"].sort(key=lambda r: r["gameweek"])
        entry["cumulative_attribution"] = roll_up(entry["by_gameweek"])
        last = entry["by_gameweek"][-1] if entry["by_gameweek"] else {}
        entry["latest_percentile"] = last.get("percentile")
        entry["latest_estimated_rank"] = last.get("estimated_rank")
        for row in entry["by_gameweek"]:
            findings.append(finding(entry["label"], row["gameweek"], row))
    return {
        "season": TARGET_SEASON,
        "generated_at": datetime.now().isoformat(),
        "goal": "overall rank top 100k",
        "total_players": total_players,
        "gameweeks_scored": sorted({r["gameweek"] for e in subjects.values() for r in e["by_gameweek"]}),
        "sample_shape_by_gameweek": sample_shape_by_gameweek,
        "subjects": subjects,
        "findings": findings,
        "limitations": _LIMITATIONS,
    }


# ============================================================
# squad resolution -- the model's optimal squad from the DB, the tracked accounts from the API
# ============================================================

def model_optimal_squad(con, event):
    row = con.execute(
        "SELECT run_id, ep_model_version FROM squad_optimizer_runs "
        "WHERE target_season = ? AND target_gameweek = ? AND is_manager_snapshot = FALSE "
        "ORDER BY run_id DESC LIMIT 1",
        [TARGET_SEASON, event],
    ).fetchone()
    if not row:
        return None
    run_id, ep_mv = row
    sel = con.execute(
        "SELECT player_uid, in_xi, is_captain FROM squad_optimizer_selections WHERE run_id = ? AND in_squad",
        [run_id],
    ).fetchall()
    if not sel:
        return None
    return {
        "subject_id": str(run_id),
        "xi": [u for u, in_xi, _ in sel if in_xi],
        "bench": [u for u, in_xi, _ in sel if not in_xi],
        "captain": next((u for u, _, is_cap in sel if is_cap), None),
        "ep_model_version": ep_mv,
    }


def account_squad(con, entry_id, event, element_names):
    picks = ifp.fetch_entry_picks(entry_id, event)
    if not picks:
        return None
    xi, bench, captain = [], [], None
    for p in picks:
        uid = iw._resolve_player(con, element_names.get(p["element"]), TARGET_SEASON)
        if not uid:
            continue
        (xi if p["position"] <= 11 else bench).append(uid)
        if p.get("is_captain"):
            captain = uid
    if not xi:
        return None
    return {"subject_id": str(entry_id), "xi": xi, "bench": bench, "captain": captain, "ep_model_version": None}


def account_overall_rank(entry_id, event):
    try:
        history = ax.fetch_entry_history(entry_id)
    except Exception as exc:  # noqa: BLE001 -- a calibration extra, never worth failing the autopsy
        print(f"::warning::rank_autopsy: could not fetch entry {entry_id} history ({exc})")
        return None
    for row in history.get("current", []):
        if row.get("event") == event:
            return row.get("overall_rank")
    return None


# ============================================================
# orchestration
# ============================================================

def run_autopsy(con, *, total_players, element_names, account_squad_fn, account_rank_fn):
    events = [
        r[0] for r in con.execute(
            "SELECT DISTINCT event FROM fact_rival_squad_sample WHERE season = ? ORDER BY event", [TARGET_SEASON],
        ).fetchall()
    ]
    subjects: dict[str, dict] = {}
    sample_shape_by_gameweek: dict[str, dict] = {}

    for event in events:
        if not field_rank.gameweek_is_settled(con, TARGET_SEASON, event):
            continue
        sample_shape_by_gameweek[str(event)] = field_rank.sample_shape(con, TARGET_SEASON, event)

        plan = [("model_optimal", "model optimal", model_optimal_squad(con, event), None)]
        for entry_id, label in TRACKED_ACCOUNTS.items():
            squad = account_squad_fn(entry_id, event)
            plan.append((f"account:{entry_id}", label, squad, account_rank_fn(entry_id, event) if squad else None))

        for subject_key, label, squad, real_rank in plan:
            if not squad:
                continue
            row = score_subject(con, event, subject_key, squad, real_rank, total_players)
            if row is None:
                continue
            subjects.setdefault(subject_key, {"label": label, "by_gameweek": []})["by_gameweek"].append(row)

    return assemble_payload(subjects, sample_shape_by_gameweek, total_players)


def main() -> None:
    con = db.connect()
    if not con.execute("SELECT 1 FROM fact_rival_squad_sample WHERE season = ? LIMIT 1", [TARGET_SEASON]).fetchone():
        print("rank_autopsy: no rival sample yet -- run scripts/run_rival_sample_ingestion.py first")
        con.close()
        return

    try:
        bootstrap = ax.fetch_bootstrap_static()
        total_players = bootstrap.get("total_players")
        element_names = ifp.fetch_bootstrap_elements(payload=bootstrap)
    except Exception as exc:  # noqa: BLE001
        print(f"::warning::rank_autopsy: bootstrap-static fetch failed ({exc}) -- accounts skipped, no rank projection")
        total_players, element_names = None, {}

    payload = run_autopsy(
        con,
        total_players=total_players,
        element_names=element_names,
        account_squad_fn=(lambda eid, ev: account_squad(con, eid, ev, element_names)) if element_names else (lambda eid, ev: None),
        account_rank_fn=account_overall_rank,
    )

    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DASHBOARD_DIR / "rank_autopsy.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[rank_autopsy] wrote {out_path} -- {len(payload['gameweeks_scored'])} gameweek(s), "
          f"{len(payload['subjects'])} subject(s)")
    for line in payload["findings"]:
        print(f"  {line}")

    con.close()


if __name__ == "__main__":
    main()
