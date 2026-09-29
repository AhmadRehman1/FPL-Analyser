"""Priority 2 addition: MVP consensus-divergence flagging.

analyst_debate/community_sentiment/youtube_evidence are free-text prose with zero live
consumers before this module (verified -- grepped ingest_workbook.py's own call sites: all
three always pass claim_value_numeric=None, so evidence_blend.blend_numeric() can never
return anything for them -- see evidence_blend.aggregate_evidence_weight's own docstring).
Full extraction of WHAT these claims actually say (positive/negative, how strongly) is
explicitly out of scope for this MVP -- this instead uses aggregate_evidence_weight() (a
coarse "how much genuine, reliability/decay-weighted evidence exists" signal) as a
volume/strength proxy, not a sentiment-direction-aware one. Real motivating case this is
scoped to catch: the model picked a GBP5.5m defender while a same-price, better-regarded
alternative existed and had to be caught manually.
"""

from datetime import datetime, timezone

import duckdb

from . import evidence_blend as eb
from . import squad_optimizer as so_mod

CONSENSUS_EVIDENCE_CLAIM_TYPES = ["community_sentiment", "analyst_debate", "youtube_evidence"]


def flag_consensus_divergent_picks(
    con: duckdb.DuckDBPyConnection,
    squad_optimizer_run_id: int,
    asof: datetime,
    decay_params_version: int,
    fact_multiplier_params_version: int,
    price_band: float,
    divergence_ratio_threshold: float,
) -> list[dict]:
    """For every selected (squad) player, checks whether a same-position, same-price-band
    (+/- price_band) alternative from the SAME candidate pool this squad was solved against
    has meaningfully higher blended structured evidence -- "meaningfully higher" meaning the
    alternative's aggregate_evidence_weight exceeds the selected player's by at least
    divergence_ratio_threshold (a ratio, not an absolute difference, since evidence volume
    varies hugely by how heavily-discussed a player is -- a flat absolute cutoff would either
    flag every heavily-discussed star or never fire for a barely-covered squad player).

    A selected player with ZERO aggregate evidence weight of their own is compared against
    ANY alternative with nonzero weight (a ratio is undefined at zero -- treated as "any real
    evidence beats none," not silently skipped). Among all divergent alternatives for a given
    selected player, only the single best-evidenced one is flagged (the point is "here's a
    named alternative worth a look," not an exhaustive dump of every candidate in the band).
    """
    run_row = con.execute(
        "SELECT ep_model_version, uncertainty_model_version, target_season FROM squad_optimizer_runs WHERE run_id = ?",
        [squad_optimizer_run_id],
    ).fetchone()
    if not run_row:
        raise ValueError(f"no squad_optimizer_runs row for run_id={squad_optimizer_run_id}")
    ep_model_version, uncertainty_model_version, target_season = run_row

    squad_uids = {
        r[0] for r in con.execute(
            "SELECT player_uid FROM squad_optimizer_selections WHERE run_id = ? AND in_squad", [squad_optimizer_run_id]
        ).fetchall()
    }
    if not squad_uids:
        raise ValueError(f"run_id={squad_optimizer_run_id} has no in_squad players")

    candidates = so_mod.fetch_candidate_pool(con, ep_model_version, uncertainty_model_version, target_season)
    by_uid = {c["player_uid"]: c for c in candidates}

    weight_cache: dict[str, float] = {}

    def _weight(uid: str) -> float:
        if uid not in weight_cache:
            weight_cache[uid] = eb.aggregate_evidence_weight(
                con, "player", uid, CONSENSUS_EVIDENCE_CLAIM_TYPES, asof,
                decay_params_version, fact_multiplier_params_version,
            )
        return weight_cache[uid]

    flags = []
    for uid in sorted(squad_uids):
        selected = by_uid.get(uid)
        if selected is None:
            continue  # not in the solved candidate pool -- shouldn't happen for a real run, never crash a report over it
        selected_weight = _weight(uid)
        alternatives = [
            c for c in candidates
            if c["player_uid"] != uid and c["position"] == selected["position"]
            and abs(c["price"] - selected["price"]) <= price_band
        ]

        best_alt, best_alt_weight = None, selected_weight
        for alt in sorted(alternatives, key=lambda c: c["player_uid"]):
            alt_weight = _weight(alt["player_uid"])
            is_divergent = (
                (selected_weight == 0 and alt_weight > 0)
                or (selected_weight > 0 and alt_weight >= selected_weight * (1 + divergence_ratio_threshold))
            )
            if is_divergent and alt_weight > best_alt_weight:
                best_alt, best_alt_weight = alt, alt_weight

        if best_alt is not None:
            flags.append({
                "selected_player_uid": uid, "selected_player_name": selected["name"],
                "selected_evidence_weight": selected_weight,
                "alternative_player_uid": best_alt["player_uid"], "alternative_player_name": best_alt["name"],
                "alternative_evidence_weight": best_alt_weight,
                "position": selected["position"], "price_band": price_band,
            })
    return flags


# ============================================================
# Model vs expert consensus (Fix G) -- a check, not a model input.
#
# Compares the model's captain / squad / transfers with the numeric expert-consensus claims
# (ingest_expert_consensus.py), lists strong disagreements with the model's own reason, and
# scores each one after the gameweek, so there's an honest "model vs experts" record before
# the consensus is ever allowed into EP or the captain ranking.
# ============================================================

# Placeholders, not fitted: what counts as a "strong" expert view.
EXPERT_CAPTAIN_SHARE_MIN = 0.5
EXPERT_BUY_NET_MIN = 0.5      # expert_transfer_net is net buys / sources in the file (-1..1)
EXPERT_SELL_NET_MAX = -0.5
EXPERT_AVOID_MIN = 0.5        # |expert_avoid|


def expert_disagreements(
    model: dict, consensus: dict[str, dict[str, float]], model_ep: dict[str, float] | None = None,
) -> list[dict]:
    """model: {"captain": name, "squad": [names], "transfers_in": [names], "transfers_out": [names]}.
    consensus: {claim_type: {player_name: numeric}} for the expert_* claim types.
    model_ep: {player_name: projected points} for the "model's reason" line.
    Returns [{"kind", "model_pick", "expert_pick", "detail", "model_reason"}]."""
    model_ep = model_ep or {}
    squad = set(model.get("squad") or [])
    out = []

    def ep(name):
        v = model_ep.get(name)
        return f"{v:.2f}" if v is not None else "n/a"

    caps = consensus.get("expert_captain_share", {})
    if caps:
        top, share = max(caps.items(), key=lambda kv: kv[1])
        if share >= EXPERT_CAPTAIN_SHARE_MIN and model.get("captain") and model["captain"] != top:
            out.append({
                "kind": "captain", "model_pick": model["captain"], "expert_pick": top,
                "detail": f"{share:.0%} of experts captain {top}",
                "model_reason": f"model projects {model['captain']} {ep(model['captain'])} vs {top} {ep(top)}",
            })

    net = consensus.get("expert_transfer_net", {})
    avoid = consensus.get("expert_avoid", {})
    for name in model.get("transfers_in") or []:
        if net.get(name, 0) <= EXPERT_SELL_NET_MAX or -avoid.get(name, 0) >= EXPERT_AVOID_MIN:
            out.append({
                "kind": "transfer_in", "model_pick": name, "expert_pick": None,
                "detail": f"experts are selling/avoiding {name}",
                "model_reason": f"model projects {name} {ep(name)}",
            })
    for name in model.get("transfers_out") or []:
        if net.get(name, 0) >= EXPERT_BUY_NET_MIN:
            out.append({
                "kind": "transfer_out", "model_pick": name, "expert_pick": name,
                "detail": f"experts are buying {name}",
                "model_reason": f"model projects {name} {ep(name)}",
            })
    for name, v in sorted(net.items()):
        already_flagged = (model.get("transfers_in") or []) + (model.get("transfers_out") or [])
        if v >= EXPERT_BUY_NET_MIN and name not in squad and name not in already_flagged:
            out.append({
                "kind": "squad_missing", "model_pick": None, "expert_pick": name,
                "detail": f"expert buy target {name} not in the model's squad",
                "model_reason": f"model projects {name} {ep(name)}",
            })
    for name in sorted(squad):
        if -avoid.get(name, 0) >= EXPERT_AVOID_MIN:
            out.append({
                "kind": "squad_avoid", "model_pick": name, "expert_pick": None,
                "detail": f"experts say avoid {name}",
                "model_reason": f"model projects {name} {ep(name)}",
            })
    return out


def score_disagreement(d: dict, actual_points: dict[str, float]) -> dict:
    """After the gameweek: who was right. For a captain call, compare the two players' points;
    for one-sided calls, the model is right when its player scored at least `par` (4, a
    placeholder for 'returned') and the experts' side didn't."""
    par = 4.0
    m, e = d.get("model_pick"), d.get("expert_pick")
    mp = actual_points.get(m) if m else None
    xp = actual_points.get(e) if e else None
    if d["kind"] == "captain":
        winner = None if mp is None or xp is None else ("model" if mp > xp else "experts" if xp > mp else "tie")
    elif d["kind"] in ("transfer_in", "squad_avoid"):
        winner = None if mp is None else ("model" if mp >= par else "experts")
    elif d["kind"] in ("transfer_out", "squad_missing"):
        winner = None if xp is None else ("experts" if xp >= par else "model")
    else:
        winner = None
    return {**d, "model_points": mp, "expert_points": xp, "winner": winner}


def append_disagreement_log(path, gameweek: int, consensus_file: str, disagreements: list[dict]) -> int:
    """One JSON line per disagreement, keyed by gameweek, so it can be scored after the GW."""
    from pathlib import Path
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logged_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as f:
        for d in disagreements:
            f.write(json.dumps({"gameweek": gameweek, "consensus_file": consensus_file, "logged_at": logged_at, **d}) + "\n")
    return len(disagreements)


def model_vs_experts_record(scored: list[dict]) -> dict:
    """Running tally over scored disagreements."""
    tally = {"model": 0, "experts": 0, "tie": 0, "unscored": 0}
    for d in scored:
        tally[d.get("winner") or "unscored"] += 1
    decided = tally["model"] + tally["experts"]
    tally["model_win_rate"] = round(tally["model"] / decided, 3) if decided else None
    return tally
