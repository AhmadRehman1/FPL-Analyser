"""Per-team, evidence-driven Wildcard / Bench Boost / Free Hit timing for a real held squad.

`forward_season_sim.run_forward_season_sim()` answers "given the squad I hold now, when does
the model want to Wildcard?" -- but only through the model's own greedy, 5-gameweek-visible
horizon (`_is_best_gameweek_in_visible_horizon()`). At gameweek G it genuinely cannot know
whether G+10 is a structurally better rebuild week than G+2, because G+10 is outside the
window it can see when it decides at G. That blind spot is real, and a single `model_choice`
walk is not enough evidence to pick a Wildcard week on its own.

This module closes it by running the forward sim once per *forced* Wildcard gameweek across a
full candidate window and comparing every arm on equal footing:

  * `force_wildcard_at=k` for every k in the sweep window -> "what does full-horizon hindsight
    say the best week is" (the arm with the highest total projected points over the shared
    evaluation window),
  * one `model_choice` arm -> "what does the greedy model do with only its own visible
    horizon at each step",
  * one `hold_wildcard` arm -> "what does never rebuilding cost", and the per-gameweek
    `evaluate_wildcard()` gain trajectory against the *evolved* squad.

`compare_wildcard_timing()` is a pure function over those arm results -- the sweep-and-compare
orchestration, unit-tested directly rather than re-testing `run_forward_season_sim()`. The
heavy walking stays in `forward_season_sim`; nothing here re-implements it.

`robustness_check()` (Step 4) does NOT trust `evaluate_wildcard()`'s single MIQP solve for the
chosen week: it re-solves `squad_optimizer.solve()` under a spread of `lambda_value`,
`rho_residual` and jittered-EP perturbations and reports which players survive every solve
("core") vs only some ("fragile"). `plan_perturbations()` / `classify_core_fragile()` are the
pure pieces.

Every number this module produces is projected expected points, never a realised outcome, and
inherits `forward_season_sim`'s disclosed early-season limitation: with only ~2 gameweeks
played, a GW6 and a GW16 projection differ by fixtures and minutes, not by form.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import duckdb

from . import params as params_mod, squad_optimizer, transfer_planner, uncertainty

# Chip set 1 (Wildcard/Free Hit/Bench Boost/Triple Captain) is forfeited if unused by this
# gameweek; a gameweek at or past it draws from the second set. Mirrors
# transfer_planner.GW19_DEADLINE_GAMEWEEK / backtest._decide_gameweek_action's own is_set1 test.
SET1_DEADLINE_GAMEWEEK = transfer_planner.GW19_DEADLINE_GAMEWEEK


class ChipAlreadyUsedError(RuntimeError):
    """Raised when a sweep would force a chip the tracked entry has already spent for the set
    covering those gameweeks -- the analysis must fail loudly, not quietly recommend re-using
    a chip that is gone."""


class MissingSquadDataError(RuntimeError):
    """Raised when a tracked entry's real squad cannot be read from the FPL API (no picks for
    the requested event). A manager's real squad is exactly 15 real players; substituting a
    placeholder would produce a confident-looking recommendation for a squad nobody holds."""


def assert_wildcard_available(
    *, chips_used_set1: list[str], chips_used_set2: list[str], sweep_gameweeks: list[int],
) -> None:
    """Guard for Step 7: refuse to run a Wildcard sweep over gameweeks whose chip set already
    has the Wildcard marked used. A set-1 Wildcard already spent does NOT block a sweep that
    lies entirely in set 2 (GW>=19) -- that is a genuinely separate chip."""
    set1 = set(chips_used_set1 or [])
    set2 = set(chips_used_set2 or [])
    touches_set1 = any(gw < SET1_DEADLINE_GAMEWEEK for gw in sweep_gameweeks)
    touches_set2 = any(gw >= SET1_DEADLINE_GAMEWEEK for gw in sweep_gameweeks)
    if touches_set1 and "wildcard" in set1:
        raise ChipAlreadyUsedError(
            f"entry has already used its set-1 Wildcard (chips_used_set1={sorted(set1)}); "
            f"cannot sweep forced Wildcard over set-1 gameweeks {[g for g in sweep_gameweeks if g < SET1_DEADLINE_GAMEWEEK]}"
        )
    if touches_set2 and "wildcard" in set2:
        raise ChipAlreadyUsedError(
            f"entry has already used its set-2 Wildcard (chips_used_set2={sorted(set2)}); "
            f"cannot sweep forced Wildcard over set-2 gameweeks {[g for g in sweep_gameweeks if g >= SET1_DEADLINE_GAMEWEEK]}"
        )


def build_bootstrap_squad(*, entry_id: int, picks: list[dict] | None, element_names: dict[int, str]) -> list[dict]:
    """Map an FPL `entry/{id}/event/{event}/picks/` payload's `picks` list to the
    `{"player_name", "in_xi", "is_captain", "is_vice"}` shape `bootstrap_from_real_squad()`
    expects. Positions 1-11 are the XI, 12-15 the bench (FPL's own convention). Raises
    MissingSquadDataError rather than returning a short or placeholder squad."""
    if not picks:
        raise MissingSquadDataError(f"entry {entry_id}: FPL API returned no picks for the requested event")
    squad = []
    for p in picks:
        name = element_names.get(p["element"])
        if not name:
            raise MissingSquadDataError(
                f"entry {entry_id}: pick element {p['element']} has no name in bootstrap-static -- "
                f"cannot resolve the real squad"
            )
        squad.append({
            "player_name": name,
            "in_xi": p["position"] <= 11,
            "is_captain": bool(p.get("is_captain")),
            "is_vice": bool(p.get("is_vice_captain")),
        })
    if len(squad) != 15:
        raise MissingSquadDataError(f"entry {entry_id}: got {len(squad)} picks, expected exactly 15")
    return squad


# ============================================================
# Step 0.2 -- evidence-workbook freshness for the held squad
# ============================================================

# The claim types that actually move a minutes / EP projection for a chip-timing decision
# (minutes_model.py's own adjustment families) -- as opposed to preseason_involvement /
# fpl_price_note / community_sentiment / analyst_debate, which are context, not a live
# availability signal.
_AVAILABILITY_CLAIM_TYPES = ("injury_status", "predicted_xi", "transfer_likelihood", "set_piece_order_override")


def evidence_freshness_flags(
    con: duckdb.DuckDBPyConnection,
    *,
    held_player_uids: list[str],
    as_of_date,
    stale_days: int = 14,
    claim_types: tuple[str, ...] = _AVAILABILITY_CLAIM_TYPES,
) -> list[dict]:
    """Step 0.2 -- for every player currently held, the age of the most recent
    availability-relevant `evidence_claims` observation as of the decision date. A held player
    whose newest such claim is older than `stale_days` (or who has none at all) is flagged:
    the model's injury / rotation signal for that player is only as fresh as the workbook, and
    a chip-timing call weeks out is only as trustworthy as that signal. Never blocks anything
    -- it is a disclosure the owner weighs, exactly as the prompt asks."""
    if not held_player_uids:
        return []
    ph_players = ",".join("?" * len(held_player_uids))
    ph_types = ",".join("?" * len(claim_types))
    rows = con.execute(
        f"""
        SELECT c.subject_entity_id, coalesce(dp.canonical_name, c.subject_entity_id) AS name,
               max(c.observed_date) AS latest_observed, count(*) AS n_claims
        FROM evidence_claims c
        LEFT JOIN dim_player dp ON dp.player_uid = c.subject_entity_id
        WHERE c.subject_entity_id IN ({ph_players}) AND c.claim_type IN ({ph_types})
        GROUP BY 1, 2
        """,
        [*held_player_uids, *claim_types],
    ).fetchall()
    by_uid = {r[0]: r for r in rows}

    flags = []
    for uid in held_player_uids:
        row = by_uid.get(uid)
        if row is None or row[2] is None:
            flags.append({"player_uid": uid, "name": row[1] if row else uid,
                          "latest_observed": None, "age_days": None, "status": "no_availability_claims"})
            continue
        age = (as_of_date - row[2]).days
        if age > stale_days:
            flags.append({"player_uid": uid, "name": row[1], "latest_observed": row[2].isoformat(),
                          "age_days": age, "status": "stale"})
    return flags


# ============================================================
# Step 1 -- Wildcard sweep-and-compare (pure logic over forward-sim arm results)
# ============================================================

@dataclass
class WildcardArm:
    """One `run_forward_season_sim()` arm reduced to what the sweep comparison needs. Built
    from `ForwardSimResult.to_dict()` so an arm can be produced in a separate process (a
    matrix job) and reloaded here from JSON without a live DB."""

    mode: str                                   # "hold_wildcard" | "model_choice" | "force_wildcard_gwN"
    forced_gameweek: int | None
    total_projected_points: float
    total_band_low: float
    total_band_high: float
    wildcard_played_at: int | None              # the gameweek the arm actually played the Wildcard (None if never)
    wildcard_recommendation: dict | None        # ForwardSimResult.wildcard_recommendation (hold arm carries the trajectory read-off)
    gameweeks: list[dict]                       # each GameweekResult.to_dict()

    @classmethod
    def from_forward_sim_dict(cls, d: dict) -> WildcardArm:
        gws = d.get("gameweeks", [])
        forced = None
        mode = d["mode"]
        if mode.startswith("force_wildcard_gw"):
            forced = int(mode.removeprefix("force_wildcard_gw"))
        played_at = next((g["gameweek"] for g in gws if g.get("action") == "wildcard"), None)
        return cls(
            mode=mode,
            forced_gameweek=forced,
            total_projected_points=float(d["total_projected_points"]),
            total_band_low=float(d["total_band"][0]),
            total_band_high=float(d["total_band"][1]),
            wildcard_played_at=played_at,
            wildcard_recommendation=d.get("wildcard_recommendation"),
            gameweeks=gws,
        )


@dataclass
class WildcardTimingComparison:
    entry_label: str
    start_gameweek: int
    eval_end_gameweek: int
    sweep_gameweeks: list[int]
    hold_total_points: float
    hold_band: list[float]
    greedy_gameweek: int | None                 # model_choice's played-at gameweek (None if it held)
    greedy_total_points: float | None
    swept_best_gameweek: int | None             # argmax total_projected_points among the forced arms
    swept_best_points: float | None
    swept_table: list[dict]                     # [{gameweek, total_projected_points, band_low, band_high, delta_vs_hold}]
    trajectory: list[dict]                      # per-gw {gameweek, wildcard_gain, wildcard_recommended, current_squad_horizon_value}
    hold_recommendation: dict | None            # the hold arm's own threshold-based wildcard_recommendation
    greedy_swept_agree: bool | None
    disagreement_note: str

    def to_dict(self) -> dict:
        return {
            "entry_label": self.entry_label,
            "start_gameweek": self.start_gameweek,
            "eval_end_gameweek": self.eval_end_gameweek,
            "sweep_gameweeks": self.sweep_gameweeks,
            "hold_total_points": round(self.hold_total_points, 1),
            "hold_band": [round(x, 1) for x in self.hold_band],
            "greedy_gameweek": self.greedy_gameweek,
            "greedy_total_points": None if self.greedy_total_points is None else round(self.greedy_total_points, 1),
            "swept_best_gameweek": self.swept_best_gameweek,
            "swept_best_points": None if self.swept_best_points is None else round(self.swept_best_points, 1),
            "swept_table": self.swept_table,
            "trajectory": self.trajectory,
            "hold_recommendation": self.hold_recommendation,
            "greedy_swept_agree": self.greedy_swept_agree,
            "disagreement_note": self.disagreement_note,
        }


def _trajectory_from_hold_arm(hold_arm: WildcardArm) -> list[dict]:
    return [
        {
            "gameweek": g["gameweek"],
            "wildcard_gain": g.get("wildcard_gain"),
            "wildcard_recommended": g.get("wildcard_recommended", False),
            "current_squad_horizon_value": g.get("current_squad_horizon_value"),
        }
        for g in hold_arm.gameweeks
    ]


def _explain(greedy_gw, swept_gw, trajectory, horizon_gameweeks, swept_table, hold_total):
    if swept_gw is None:
        return (
            "No forced-Wildcard gameweek in the sweep window beats the hold-Wildcard baseline on "
            "total projected points -- the sweep says hold the chip, not that any particular week is best."
        )
    swept_delta = next((r["delta_vs_hold"] for r in swept_table if r["gameweek"] == swept_gw), None)
    if greedy_gw is None:
        return (
            f"The greedy model_choice walk never played the Wildcard in-window; the full-horizon sweep "
            f"prefers GW{swept_gw} (+{swept_delta:.1f} projected pts vs holding). The greedy walk only ever "
            f"sees {horizon_gameweeks} gameweeks ahead when it decides, so a rebuild whose payoff is "
            f"concentrated later in the window never clears its 'is now the best visible week' check."
        )
    if greedy_gw == swept_gw:
        return (
            f"Greedy and swept agree on GW{swept_gw}. The model's own {horizon_gameweeks}-gameweek-visible "
            f"timing check happened to land on the same week full-horizon hindsight prefers."
        )
    gtraj = {r["gameweek"]: r for r in trajectory}
    g_gain = gtraj.get(greedy_gw, {}).get("wildcard_gain")
    s_gain = gtraj.get(swept_gw, {}).get("wildcard_gain")
    return (
        f"Greedy plays GW{greedy_gw}; full-horizon sweep prefers GW{swept_gw} "
        f"(+{swept_delta:.1f} projected pts vs holding). At GW{greedy_gw} the model only sees "
        f"GW{greedy_gw}..GW{greedy_gw + horizon_gameweeks - 1}, so GW{swept_gw} is outside its visible "
        f"horizon and cannot be compared against GW{greedy_gw} at decision time. Per-GW evaluate_wildcard "
        f"gain: GW{greedy_gw}~{g_gain if g_gain is None else round(g_gain, 1)}, "
        f"GW{swept_gw}~{s_gain if s_gain is None else round(s_gain, 1)}."
    )


def compare_wildcard_timing(
    *,
    entry_label: str,
    start_gameweek: int,
    eval_end_gameweek: int,
    hold_arm: WildcardArm,
    model_choice_arm: WildcardArm | None,
    forced_arms: list[WildcardArm],
    horizon_gameweeks: int = 5,
) -> WildcardTimingComparison:
    """Step 1.4 -- compare the forced sweep, the greedy walk and the hold baseline on equal
    footing over the shared evaluation window, and say where greedy and full-horizon hindsight
    disagree and why.

    `forced_arms` may be given in any order and may be partial (a matrix job that failed drops
    out); each must carry a real `forced_gameweek`. The swept winner is the forced arm with the
    highest `total_projected_points` -- but only if it also beats the hold baseline, otherwise
    the sweep's answer is "hold the chip" (swept_best_gameweek=None)."""
    forced_by_gw = {a.forced_gameweek: a for a in forced_arms if a.forced_gameweek is not None}
    sweep_gameweeks = sorted(forced_by_gw)

    swept_table = [
        {
            "gameweek": gw,
            "total_projected_points": round(forced_by_gw[gw].total_projected_points, 1),
            "band_low": round(forced_by_gw[gw].total_band_low, 1),
            "band_high": round(forced_by_gw[gw].total_band_high, 1),
            "delta_vs_hold": round(forced_by_gw[gw].total_projected_points - hold_arm.total_projected_points, 1),
        }
        for gw in sweep_gameweeks
    ]

    swept_best_gameweek = swept_best_points = None
    if forced_by_gw:
        best_gw = max(forced_by_gw, key=lambda gw: forced_by_gw[gw].total_projected_points)
        if forced_by_gw[best_gw].total_projected_points > hold_arm.total_projected_points:
            swept_best_gameweek = best_gw
            swept_best_points = forced_by_gw[best_gw].total_projected_points

    greedy_gameweek = model_choice_arm.wildcard_played_at if model_choice_arm else None
    greedy_total_points = model_choice_arm.total_projected_points if model_choice_arm else None

    greedy_swept_agree = None
    if swept_best_gameweek is not None and model_choice_arm is not None:
        greedy_swept_agree = greedy_gameweek == swept_best_gameweek

    trajectory = _trajectory_from_hold_arm(hold_arm)
    note = _explain(
        greedy_gameweek, swept_best_gameweek, trajectory, horizon_gameweeks, swept_table,
        hold_arm.total_projected_points,
    )

    return WildcardTimingComparison(
        entry_label=entry_label,
        start_gameweek=start_gameweek,
        eval_end_gameweek=eval_end_gameweek,
        sweep_gameweeks=sweep_gameweeks,
        hold_total_points=hold_arm.total_projected_points,
        hold_band=[hold_arm.total_band_low, hold_arm.total_band_high],
        greedy_gameweek=greedy_gameweek,
        greedy_total_points=greedy_total_points,
        swept_best_gameweek=swept_best_gameweek,
        swept_best_points=swept_best_points,
        swept_table=swept_table,
        trajectory=trajectory,
        hold_recommendation=hold_arm.wildcard_recommendation,
        greedy_swept_agree=greedy_swept_agree,
        disagreement_note=note,
    )


# ============================================================
# Step 4 -- robustness check on the chosen Wildcard week's MIQP squad
# ============================================================

@dataclass
class Perturbation:
    label: str
    lambda_value: float
    rho_residual_params_version: int
    ep_jitter_sigmas: float                     # multiples of each player's own sqrt(var_total) added as N(0, .) noise

    def to_dict(self) -> dict:
        return {
            "label": self.label, "lambda_value": self.lambda_value,
            "rho_residual_params_version": self.rho_residual_params_version,
            "ep_jitter_sigmas": self.ep_jitter_sigmas,
        }


def plan_perturbations(
    *,
    base_lambda_value: float,
    base_rho_residual_params_version: int,
    lambda_values: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.30),
    rho_residual_params_versions: tuple[int, ...] = (1, 2, 4),
    ep_jitter_sigmas: tuple[float, ...] = (0.0, 0.25, 0.5),
    n_ep_jitter_draws: int = 3,
) -> list[Perturbation]:
    """A deliberately one-axis-at-a-time spread (not a full cross product -- that explodes and
    over-weights joint extremes nobody is actually proposing): the unperturbed base point, then
    each lambda alone, each rho_residual version alone, and a few EP-jitter draws alone. Every
    perturbation stays a single, defensible "what if this one uncertain input were different"
    question."""
    seen: set[tuple] = set()
    out: list[Perturbation] = []

    def add(label: str, lam: float, rho_v: int, jitter: float) -> None:
        key = (round(lam, 4), rho_v, round(jitter, 4))
        if key in seen:
            return
        seen.add(key)
        out.append(Perturbation(label=label, lambda_value=lam, rho_residual_params_version=rho_v, ep_jitter_sigmas=jitter))

    add("base", base_lambda_value, base_rho_residual_params_version, 0.0)
    for lam in lambda_values:
        add(f"lambda={lam}", lam, base_rho_residual_params_version, 0.0)
    for rho_v in rho_residual_params_versions:
        add(f"rho_residual_v{rho_v}", base_lambda_value, rho_v, 0.0)
    for sigma in ep_jitter_sigmas:
        if sigma == 0.0:
            continue
        for _ in range(n_ep_jitter_draws):
            add(f"ep_jitter={sigma}", base_lambda_value, base_rho_residual_params_version, sigma)
    return out


def classify_core_fragile(squads_by_label: dict[str, list[str]]) -> dict:
    """Given {perturbation_label: [player_uid, ...]} for every solve, split the union into
    'core' (in every solve) and 'fragile' (in at least one but not all). An empty input is a
    real caller error (no solves ran), not a silently-empty result."""
    if not squads_by_label:
        raise ValueError("classify_core_fragile: no solve results to classify")
    squad_sets = [set(uids) for uids in squads_by_label.values()]
    union = set().union(*squad_sets)
    core = sorted(u for u in union if all(u in s for s in squad_sets))
    fragile = sorted(union - set(core))
    n_solves = len(squad_sets)
    appearances = {u: sum(u in s for s in squad_sets) for u in union}
    return {
        "n_solves": n_solves,
        "core_players": core,
        "fragile_players": fragile,
        "core_count": len(core),
        "fragile_count": len(fragile),
        "core_fraction_of_15": round(len(core) / 15, 3),
        "appearances": {u: appearances[u] for u in sorted(union)},
        "verdict": "stable" if len(fragile) <= 2 else "fragile",
    }


def _jitter_candidates(candidates: list[dict], sigma_multiple: float, rng: random.Random) -> list[dict]:
    if sigma_multiple <= 0.0:
        return candidates
    out = []
    for c in candidates:
        std = (c.get("var") or 0.0) ** 0.5
        noise = rng.gauss(0.0, sigma_multiple * std)
        out.append({**c, "mu": max(0.0, (c["mu"] or 0.0) + noise)})
    return out


@dataclass
class RobustnessReport:
    entry_label: str
    gameweek: int
    perturbations: list[dict]
    squads_by_label: dict[str, list[str]]
    summary: dict
    fresh_run_id_squad: list[str]

    def to_dict(self) -> dict:
        return {
            "entry_label": self.entry_label,
            "gameweek": self.gameweek,
            "perturbations": self.perturbations,
            "squads_by_label": self.squads_by_label,
            "summary": self.summary,
            "fresh_run_id_squad": self.fresh_run_id_squad,
        }


def robustness_check(
    con: duckdb.DuckDBPyConnection,
    *,
    entry_label: str,
    calibration_asof_date,
    target_season: str,
    target_gameweek: int,
    ep_model_version: int,
    uncertainty_model_version: int,
    ts_model_version: int,
    mm_model_version: int,
    scoring_params_version: int,
    bps_params_version: int,
    tau_params_version: int,
    corr_params_version: int,
    guardrail_params_version: int,
    base_lambda_value: float,
    base_rho_residual_params_version: int,
    perturbations: list[Perturbation] | None = None,
    fresh_run_id: int | None = None,
    rng_seed: int = 7,
) -> RobustnessReport:
    """Re-solve `squad_optimizer.solve()` (raw solve -- no divergence check, no DB writes, the
    fast path) once per perturbation and classify core vs fragile players. `rho_residual`
    perturbations re-run `uncertainty.run()` at that version to get a genuinely different
    covariance structure; EP jitter adds N(0, k*sqrt(var)) noise to each candidate's mu with a
    seeded RNG. The base `ep_model_version` / `uncertainty_model_version` are the ones
    `evaluate_wildcard()` itself used for this gameweek."""
    if perturbations is None:
        perturbations = plan_perturbations(
            base_lambda_value=base_lambda_value,
            base_rho_residual_params_version=base_rho_residual_params_version,
        )

    guardrail_cap, _ = params_mod.resolve_param(
        con, "squad_optimizer_guardrail_params", "xi_club_concentration_cap", guardrail_params_version
    )
    rng = random.Random(rng_seed)

    # cache uncertainty_model_version per rho_residual version so we only rebuild covariance once each
    un_mv_by_rho: dict[int, int] = {base_rho_residual_params_version: uncertainty_model_version}

    def un_mv_for(rho_v: int) -> int:
        if rho_v not in un_mv_by_rho:
            un_mv_by_rho[rho_v] = uncertainty.run(
                con, calibration_asof_date, ep_model_version, mm_model_version, ts_model_version,
                scoring_params_version, bps_params_version, tau_params_version, rho_v, corr_params_version,
            )
        return un_mv_by_rho[rho_v]

    squads_by_label: dict[str, list[str]] = {}
    for p in perturbations:
        un_mv = un_mv_for(p.rho_residual_params_version)
        candidates = squad_optimizer.fetch_candidate_pool(con, ep_model_version, un_mv, target_season)
        if len(candidates) < 15:
            raise ValueError(
                f"robustness_check: candidate pool for {target_season} GW{target_gameweek} has "
                f"only {len(candidates)} priced players"
            )
        candidates = _jitter_candidates(candidates, p.ep_jitter_sigmas, rng)
        sigma_pairs = squad_optimizer.fetch_sigma_pairs(con, un_mv, {c["player_uid"] for c in candidates})
        result = squad_optimizer.solve(candidates, sigma_pairs, p.lambda_value, guardrail_cap)
        squads_by_label[p.label] = sorted(result["squad"])

    summary = classify_core_fragile(squads_by_label)

    fresh_squad: list[str] = []
    if fresh_run_id is not None:
        fresh_squad = sorted(
            r[0] for r in con.execute(
                "SELECT player_uid FROM squad_optimizer_selections WHERE run_id = ? AND in_squad", [fresh_run_id]
            ).fetchall()
        )

    return RobustnessReport(
        entry_label=entry_label,
        gameweek=target_gameweek,
        perturbations=[p.to_dict() for p in perturbations],
        squads_by_label=squads_by_label,
        summary=summary,
        fresh_run_id_squad=fresh_squad,
    )


# ============================================================
# Step 2 -- Bench Boost sequenced onto the chosen Wildcard week
# ============================================================

def bench_boost_window(
    con: duckdb.DuckDBPyConnection,
    *,
    wildcard_gameweek: int,
    wildcard_result: dict,
    current_squad_uids: set[str],
    current_xi_uids: set[str],
    horizon_ep_versions: dict[int, tuple[int, int]],
    target_season: str,
    ts_model_version: int | None = None,
    lookahead_gameweeks: int = 3,
) -> list[dict]:
    """Step 2 -- for the WC week and the next `lookahead_gameweeks`, evaluate
    `evaluate_wildcard_bench_boost_combo()` against the *same* fresh post-Wildcard squad
    (`wildcard_result["fresh_run_id"]`), so the question is genuinely "which of these weeks is
    the fresh squad's bench a better Bench-Boost target" rather than re-solving a new squad
    each week. `bench_boost_result` is the CURRENT squad's own Bench Boost value at that week
    (`evaluate_bench_boost()`), the naive-independent-sum baseline the combo is measured
    against."""
    rows = []
    for gw in range(wildcard_gameweek, wildcard_gameweek + lookahead_gameweeks + 1):
        if gw not in horizon_ep_versions:
            rows.append({"gameweek": gw, "available": False, "reason": "no horizon EP for this gameweek"})
            continue
        bb = transfer_planner.evaluate_bench_boost(
            con, {gw: horizon_ep_versions[gw]}, set(current_squad_uids), set(current_xi_uids),
            target_season=target_season, ts_model_version=ts_model_version,
        )
        combo = transfer_planner.evaluate_wildcard_bench_boost_combo(
            con, wildcard_result, bb, target_season, gw, horizon_ep_versions,
        )
        rows.append({
            "gameweek": gw,
            "available": "reason" not in combo,
            "synergy_gain": combo.get("synergy_gain"),
            "combo_value": combo.get("combo_value"),
            "naive_independent_sum": combo.get("naive_independent_sum"),
            "fresh_squad_bench_ep": combo.get("fresh_squad_bench_ep_at_target_gw"),
            "current_squad_bench_boost_value": bb.get("bench_ep_sum") if bb.get("recommended") else 0.0,
            "recommended_combo": combo.get("recommended_combo", False),
            "reason": combo.get("reason"),
        })
    return rows


def wildcard_followups(
    con: duckdb.DuckDBPyConnection,
    *,
    entry_label: str,
    target_season: str,
    wildcard_context: dict,
    bench_boost_lookahead: int = 3,
    robustness_perturbations: list[Perturbation] | None = None,
) -> dict:
    """Runs Step 2 (Bench Boost combo window) and Step 4 (squad robustness) for the gameweek a
    `forward_season_sim` walk actually played the Wildcard, using the live-DB `wildcard_context`
    that walk captured -- no re-walk. Must be called against the SAME connection the walk ran
    on (fresh_run_id / ts_model_version / mm_model_version are DB-local). Re-enters
    `backtest.asof_scope()` at the Wildcard gameweek so the horizon EP is built the same
    asof-safe way as everything else.

    `wildcard_context` shape is exactly what `ForwardSimResult.wildcard_context` carries:
    gameweek, asof_date, fresh_run_id, wildcard_result, holdings_before_uids, xi_before_uids,
    ts_model_version, mm_model_version, versions."""
    from . import backtest as bt

    ctx = wildcard_context
    gw = ctx["gameweek"]
    v = ctx["versions"]
    horizon_gameweeks = int(params_mod.resolve_param(
        con, "planning_horizon_params", "horizon_gameweeks", v["horizon_params_version"])[0])
    window = max(horizon_gameweeks, bench_boost_lookahead + 1)

    with bt.asof_scope(con, target_season, gw, schedule_horizon_gameweeks=horizon_gameweeks):
        asof = ctx["asof_date"]
        horizon_ep_versions = transfer_planner.compute_horizon_ep(
            con, asof, target_season, gw, ctx["ts_model_version"], ctx["mm_model_version"], window,
            v["scoring_params_version"], v["bps_params_version"], v["tau_params_version"],
            v["rho_residual_params_version"], v["corr_params_version"],
        )
        bb_window = bench_boost_window(
            con,
            wildcard_gameweek=gw,
            wildcard_result=ctx["wildcard_result"],
            current_squad_uids=set(ctx["holdings_before_uids"]),
            current_xi_uids=set(ctx["xi_before_uids"]),
            horizon_ep_versions=horizon_ep_versions,
            target_season=target_season,
            ts_model_version=ctx["ts_model_version"],
            lookahead_gameweeks=bench_boost_lookahead,
        )
        ep_mv_gw, un_mv_gw = horizon_ep_versions.get(gw, (None, None))
        robustness = None
        if ep_mv_gw is not None and un_mv_gw is not None:
            base_lambda_value, _ = params_mod.resolve_param(
                con, "risk_aversion_params", "lambda_value", v["lambda_params_version"])
            robustness = robustness_check(
                con,
                entry_label=entry_label,
                calibration_asof_date=asof,
                target_season=target_season,
                target_gameweek=gw,
                ep_model_version=ep_mv_gw,
                uncertainty_model_version=un_mv_gw,
                ts_model_version=ctx["ts_model_version"],
                mm_model_version=ctx["mm_model_version"],
                scoring_params_version=v["scoring_params_version"],
                bps_params_version=v["bps_params_version"],
                tau_params_version=v["tau_params_version"],
                corr_params_version=v["corr_params_version"],
                guardrail_params_version=v["guardrail_params_version"],
                base_lambda_value=base_lambda_value,
                base_rho_residual_params_version=v["rho_residual_params_version"],
                perturbations=robustness_perturbations,
                fresh_run_id=ctx["fresh_run_id"],
            ).to_dict()

    return {
        "wildcard_gameweek": gw,
        "wildcard_gain": ctx["wildcard_result"].get("gain"),
        "wildcard_recommended": bool(ctx["wildcard_result"].get("recommended", False)),
        "bench_boost_window": bb_window,
        "robustness": robustness,
    }


# ============================================================
# Step 3 -- Free Hit scan across the window (independent of WC/BB)
# ============================================================

def free_hit_scan_from_hold_arm(hold_arm: WildcardArm, threshold_min_horizon_gain: float = 1.5) -> list[dict]:
    """Step 3 -- the hold-Wildcard arm already calls the planner every gameweek, and the
    planner always evaluates Free Hit (`chip_evaluations` chip_type='free_hit'). When the arm
    is run with Free Hit capture on, each `GameweekResult` carries `free_hit_gain` /
    `free_hit_recommended` -- so the scan is a read, not a re-solve. Returns every gameweek
    plus a flag for the ones that clear `free_hit_gain_threshold_params`."""
    rows = []
    for g in hold_arm.gameweeks:
        gain = g.get("free_hit_gain")
        rows.append({
            "gameweek": g["gameweek"],
            "free_hit_gain": gain,
            "recommended": bool(g.get("free_hit_recommended", False)),
            "clears_threshold": gain is not None and gain > threshold_min_horizon_gain,
        })
    return rows


# ============================================================
# Step 6 -- per-team comparison output
# ============================================================

def gameweek_csv_rows(hold_arm: WildcardArm) -> list[dict]:
    """Gameweek-by-gameweek table straight from the hold arm's `GameweekResult.to_dict()`s
    (Step 6, first bullet). The hold arm is the reference walk -- it holds the chip so every
    gameweek's numbers are the "continue weekly transfers" counterfactual the sweep deltas are
    measured against."""
    out = []
    for g in hold_arm.gameweeks:
        out.append({
            "gameweek": g["gameweek"],
            "projected_points": g["projected_points"],
            "band_low": g["band_low"],
            "band_high": g["band_high"],
            "action": g["action"],
            "action_detail": g.get("action_detail", ""),
            "wildcard_gain": g.get("wildcard_gain"),
            "wildcard_recommended": g.get("wildcard_recommended", False),
            "free_hit_gain": g.get("free_hit_gain"),
            "free_hit_recommended": g.get("free_hit_recommended", False),
            "chips_used": "|".join(g.get("chips_used", [])),
        })
    return out


def _csv_text(rows: list[dict]) -> str:
    if not rows:
        return ""
    import csv
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


@dataclass
class TeamChipTimingReport:
    entry_id: int
    entry_label: str
    generated_at: str
    active_param_bundle: str
    comparison: dict                                  # WildcardTimingComparison.to_dict()
    bench_boost_window: list[dict]
    free_hit_scan: list[dict]
    robustness: dict | None
    sensitivity: dict | None                          # {bundle_label: {swept_best_gameweek, ...}} or a stability note
    evidence_freshness_flags: list[dict]
    data_flags: list[str]

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "entry_label": self.entry_label,
            "generated_at": self.generated_at,
            "active_param_bundle": self.active_param_bundle,
            "comparison": self.comparison,
            "bench_boost_window": self.bench_boost_window,
            "free_hit_scan": self.free_hit_scan,
            "robustness": self.robustness,
            "sensitivity": self.sensitivity,
            "evidence_freshness_flags": self.evidence_freshness_flags,
            "data_flags": self.data_flags,
        }

    def gameweek_csv(self, hold_arm: WildcardArm) -> str:
        return _csv_text(gameweek_csv_rows(hold_arm))

    def sweep_csv(self) -> str:
        return _csv_text(self.comparison["swept_table"])


def render_team_summary(report: TeamChipTimingReport, hold_arm: WildcardArm) -> str:
    c = report.comparison
    lines: list[str] = []
    lines.append(f"## {report.entry_label} (entry {report.entry_id})")
    lines.append("")
    lines.append(f"_Param bundle: {report.active_param_bundle}. Projected EP over "
                 f"GW{c['start_gameweek']}-{c['eval_end_gameweek']}. Not realised points._")
    lines.append("")

    if report.data_flags:
        lines.append("**Data flags:**")
        for f in report.data_flags:
            lines.append(f"- [!] {f}")
        lines.append("")

    # Wildcard verdict
    swept = c["swept_best_gameweek"]
    if swept is None:
        lines.append("**Wildcard:** the sweep finds no forced week that beats holding the chip over the "
                     "evaluation window -- hold it and re-run as the season fills in.")
    else:
        delta = next((r["delta_vs_hold"] for r in c["swept_table"] if r["gameweek"] == swept), 0.0)
        lines.append(f"**Wildcard -- full-horizon sweep says GW{swept}** "
                     f"(+{delta:.1f} projected pts vs holding, {c['swept_best_points']:.1f} total).")
    lines.append("- Greedy model_choice walk: "
                 + (f"plays GW{c['greedy_gameweek']}" if c["greedy_gameweek"] else "never plays it in-window"))
    lines.append("- Hold arm's own threshold read-off: "
                 + (f"GW{c['hold_recommendation']['gameweek']} "
                    f"(+{c['hold_recommendation']['projected_gain']:.1f})" if c["hold_recommendation"] else "hold"))
    lines.append(f"- {c['disagreement_note']}")
    lines.append("")

    # sweep table
    lines.append("| forced WC GW | total proj pts | 80% band | vs hold |")
    lines.append("|----|----|----|----|")
    for r in c["swept_table"]:
        lines.append(f"| {r['gameweek']} | {r['total_projected_points']:.1f} | "
                     f"{r['band_low']:.0f}-{r['band_high']:.0f} | {r['delta_vs_hold']:+.1f} |")
    lines.append("")

    # bench boost
    lines.append("**Bench Boost combo** (fresh post-WC squad's bench, WC week + next 3):")
    for r in report.bench_boost_window:
        if not r.get("available"):
            lines.append(f"- GW{r['gameweek']}: n/a ({r.get('reason', 'unavailable')})")
        else:
            lines.append(f"- GW{r['gameweek']}: synergy {r['synergy_gain']:+.2f} "
                         f"(combo {r['combo_value']:.1f} vs naive {r['naive_independent_sum']:.1f}) "
                         f"{'[combo recommended]' if r['recommended_combo'] else ''}")
    lines.append("")

    # free hit
    fh_hits = [r for r in report.free_hit_scan if r["clears_threshold"]]
    lines.append("**Free Hit:** "
                 + (", ".join(f"GW{r['gameweek']} (+{r['free_hit_gain']:.1f})" for r in fh_hits)
                    if fh_hits else "no gameweek in the window clears the Free Hit threshold"))
    lines.append("")

    # robustness
    if report.robustness:
        s = report.robustness["summary"]
        lines.append(f"**Wildcard squad robustness (GW{report.robustness['gameweek']}, "
                     f"{s['n_solves']} perturbed solves):** {s['verdict']} -- "
                     f"{s['core_count']}/15 core, {s['fragile_count']} fragile.")
        if s["fragile_players"]:
            lines.append(f"- fragile: {', '.join(s['fragile_players'])}")
        lines.append("")

    # sensitivity
    if report.sensitivity:
        lines.append(f"**Recalibration sensitivity:** {report.sensitivity.get('note', '')}")
        lines.append("")

    return "\n".join(lines) + "\n"
