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

from datetime import datetime

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
