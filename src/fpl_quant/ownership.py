"""Priority 1 addition: effective-ownership (EO) estimation and EO-adjusted captain-risk.

FPL is a rank tournament against millions of other managers, not a points-forecasting
contest in isolation -- selected_by_percent has been ingested into fact_player_season_stats
since M0 but had zero downstream consumers before this module. Effective ownership (EO) is
the standard refinement of raw ownership%: since a captained player's points count twice,
being captained by some fraction of a player's owners adds that same fraction's worth of
EXTRA exposure on top of their raw ownership% -- EO% = ownership% + captaincy_rate%.

captaincy_rate% (what fraction of a player's owners captain them) is not itself an ingested
field anywhere in this project (grepped the whole schema/ingestion pipeline -- confirmed, the
same conclusion squad_optimizer.captain_choice_with_differential's own caveat already reached
for rival-manager data generally). Modeled here as an explicit, versioned proxy instead of
silently omitted: real-world captaincy overwhelmingly concentrates on the single highest
projected-points "template" pick within a position, and essentially never lands on a
low-projection player regardless of how many people happen to own them (nobody captains a
bench option). estimate_captaincy_rate() below encodes exactly that -- a geometric decay in
the gap between a player's own EP and the highest EP in their position group, scaled by their
real ownership% -- named plainly as a modeled estimate throughout, never oversold as measured
rival-manager behavior.
"""


def estimate_captaincy_rate(
    selected_by_percent: float | None, ep_total: float, top_ep_total: float, captaincy_concentration: float,
) -> float:
    """captaincy_concentration in (0, 1): how sharply captaincy concentrates on the single
    highest-EP player in a position group. At ep_total == top_ep_total (the top player
    itself, or an exact tie), the decay factor is 1.0 -- among owners who'd captain within
    this position group, essentially all of them captain the top pick. For every EP point
    below the top, the factor shrinks geometrically (captaincy_concentration ** gap). The
    result can never exceed selected_by_percent itself -- you can only captain a player you
    own, so a captaincy RATE can never exceed the underlying ownership RATE it's a fraction of.
    """
    if not (0.0 < captaincy_concentration < 1.0):
        raise ValueError(f"captaincy_concentration must be in (0, 1), got {captaincy_concentration}")
    if selected_by_percent is None or selected_by_percent <= 0:
        return 0.0
    ep_gap = max(0.0, top_ep_total - ep_total)
    decay = captaincy_concentration ** ep_gap
    return selected_by_percent * decay


def effective_ownership(selected_by_percent: float | None, captaincy_rate: float) -> float | None:
    """EO% = ownership% + captaincy_rate% (see module docstring). None (not 0.0) when
    ownership itself is unknown for this player -- this project's established "missing != 0"
    discipline, matching captain_choice_with_differential's own ownership handling."""
    if selected_by_percent is None:
        return None
    return selected_by_percent + captaincy_rate


def compute_eo_for_pool(candidates: list[dict], captaincy_concentration: float) -> dict[str, float | None]:
    """EO for every candidate at once, grouped by position -- captaincy realistically only
    ever competes within "who's my best attacking/defensive option this week," never across
    positions, so top_ep_total is computed per position group. Mirrors
    squad_optimizer.fetch_sigma_pairs's own "compute once for the whole pool" shape, so
    solve() can consume a plain {player_uid: eo} dict as a precomputed input rather than
    re-deriving EO per candidate inline.

    Candidates with selected_by_percent is None get eo=None and are excluded from their
    group's top_ep_total computation too -- an unknown-ownership player can't be a real
    captaincy comparison point (their real popularity, and therefore their real pull on the
    field's captaincy choices, is exactly the unknown quantity here)."""
    by_position: dict[str, list[dict]] = {}
    for c in candidates:
        by_position.setdefault(c["position"], []).append(c)

    eo_by_uid: dict[str, float | None] = {}
    for group in by_position.values():
        owned = [c for c in group if c.get("selected_by_percent") is not None]
        top_ep_total = max((c["mu"] for c in owned), default=None)
        for c in group:
            uid = c["player_uid"]
            if c.get("selected_by_percent") is None or top_ep_total is None:
                eo_by_uid[uid] = None
                continue
            captaincy_rate = estimate_captaincy_rate(c["selected_by_percent"], c["mu"], top_ep_total, captaincy_concentration)
            eo_by_uid[uid] = effective_ownership(c["selected_by_percent"], captaincy_rate)
    return eo_by_uid


# ============================================================
# EO-adjusted captain-risk -- an explicit decision distinct from squad selection itself.
# Read-only: never changes which player is captained, only reports on the rank-risk profile
# of the choice solve() already made (same "diagnostics separate from the frozen source of
# truth" pattern as squad_optimizer.captain_choice_with_differential).
# ============================================================

def captain_risk_report(xi_candidates: list[dict], eo_by_uid: dict[str, float | None], captain_uid: str) -> dict:
    """Is the chosen captain a template (high-EO) pick, where a big haul barely moves your
    RANK relative to the field (most managers have the same captain already), or a
    differential (low-EO) pick, where the same haul swings rank sharply -- and a blank costs
    rank sharply too? This is a genuinely separate axis from "is this captain a good pick"
    (that's what solve()'s own risk-adjusted objective already decides) -- two managers can
    make the identical EP/risk-optimal captain choice and be taking very different amounts of
    RANK risk by doing so, depending on how many rivals also made it.
    """
    captain_eo = eo_by_uid.get(captain_uid)
    xi_eos = [eo for c in xi_candidates if (eo := eo_by_uid.get(c["player_uid"])) is not None]
    xi_avg_eo = sum(xi_eos) / len(xi_eos) if xi_eos else None

    if captain_eo is None or xi_avg_eo is None:
        posture_label = "unknown"
    elif captain_eo >= xi_avg_eo:
        posture_label = "template"
    else:
        posture_label = "differential"

    return {
        "captain_uid": captain_uid,
        "captain_eo": captain_eo,
        "xi_avg_eo": xi_avg_eo,
        "posture_label": posture_label,
        "caveat": (
            "EO is a modeled estimate (real ownership% plus an estimated captaincy-concentration "
            "proxy), not measured rival-manager captaincy data -- no such data is ingested anywhere "
            "in this project."
        ),
    }
