"""M5: Squad Optimizer.

The module the original v1 rebuild exists because of -- the documented lambda=0 back-five
failure happened here. Its historical failure mode (a stub/zero/identity covariance matrix,
or a solver silently falling back to a linear-only solve) is treated as load-bearing input
to this implementation, not background color: real Sigma from M4 (not a stub), a genuine
MIQP-capable solver (SCIP, not a MIP-only solver that would silently drop the quadratic
term), and the lambda=0-vs-lambda=0.15 divergence check run as the first thing this module
does, before anything else is trusted.

SCIP requires a linear objective -- the quadratic risk term w'Sigma*w is moved into a
constraint via the standard epigraph reformulation: introduce a continuous variable t,
constrain t >= w'Sigma*w (convex since Sigma is PSD, so this is a valid convex quadratic
constraint), and put -lambda*t linearly into the objective instead of -lambda*w'Sigma*w
directly.
"""

from datetime import date, datetime, timezone

import duckdb
import pyscipopt as scip

from . import params as params_mod

POSITIONS = ["Goalkeeper", "Defender", "Midfielder", "Forward"]
POSITION_QUOTA = {"Goalkeeper": 2, "Defender": 5, "Midfielder": 5, "Forward": 3}
XI_POSITION_MIN = {"Defender": 3, "Midfielder": 2, "Forward": 1}
XI_POSITION_MAX = {"Defender": 5, "Midfielder": 5, "Forward": 3}
BUDGET = 100.0


class DivergenceCheckFailedError(Exception):
    """The lambda=0 vs lambda=0.15 sanity check failed: both solves produced the identical
    squad, proving the quadratic risk term is not affecting the solve at all. Per M5's own
    frozen spec this is a hard stop, not a warning -- implementation must investigate
    before trusting any output from this optimizer."""


def seed_v1_params(con: duckdb.DuckDBPyConnection) -> None:
    # Exact frozen spec table: lambda_value=0.15, params_version=1, effective_date
    # 2026-08-10 -- not an invented default here, this one is pinned verbatim in M5's spec.
    params_mod.write_param(con, "risk_aversion_params", 1, "2026-08-10", "lambda_value", value_numeric=0.15)
    # N=3 for the XI-level club concentration cap -- spec's own invented default (chosen to
    # match the squad-level cap for consistency, not independently derived), flagged for M7
    # review alongside lambda itself.
    params_mod.write_param(
        con, "squad_optimizer_guardrail_params", 1, "2026-08-10", "xi_club_concentration_cap", value_numeric=3
    )
    # Invented v1 default for the optional captain-differential tie-break (see
    # captain_choice_with_differential()): small enough that it can only ever break a genuine
    # near-tie in the SAME risk-adjusted objective solve() itself optimizes, never override a
    # real EP/risk-driven captain choice. Same invented-default status as every other unpinned
    # constant in this project, flagged for M7 recalibration once real backtest evidence exists
    # for how much objective value a real differential swing is actually worth.
    params_mod.write_param(con, "captain_differential_params", 1, "2026-08-10", "tiebreak_epsilon", value_numeric=0.05)
    # Invented v1 default for the rank-relative objective term (see solve()'s field_cov_by_uid
    # param) -- magnitude chosen small relative to lambda_value's own risk penalty so that even
    # at kappa_rank's full pinned value, the rank-relative term can shift a genuine near-tie,
    # not override a real EP/variance-driven pick. Flagged for M7 recalibration like every
    # other unpinned constant here; risk_posture itself (protect/neutral/chase) is a per-run
    # user choice, not a pinned param -- neutral is the only one seeded as a "default" in the
    # sense of matching pre-existing behavior exactly (kappa_rank's sign multiplies out to 0).
    params_mod.write_param(con, "rank_posture_params", 1, "2026-08-10", "kappa_rank", value_numeric=0.02)
    # Real gap this fixed, caught sanity-checking a live GW1 run: solve()'s captain choice used
    # to inherit the FULL squad-level lambda (at 3-4x weight, per Var(2X)=4*Var(X) below) with
    # no separate cap of its own -- on the real GW1 candidate pool this made a 0.4%-owned,
    # 2.42-EP defensive midfielder the "risk-optimal" captain over a 69%-owned, 4.51-EP striker
    # (objective_gap=-2.80, nowhere near a near-tie the differential mechanism could catch).
    # Squad-level risk aversion legitimately protects against a genuinely disastrous gameweek
    # across a diversified 15-player squad; applying that SAME magnitude to a single
    # concentrated captaincy bet is a different, much stronger risk decision than lambda was
    # ever calibrated for, and it contradicts how captaincy is actually played (maximize EP on
    # your captain; only deviate for a genuine near-tie or an explicit differential/chase
    # decision -- both of which already have their own dedicated, narrowly-scoped mechanisms:
    # captain_choice_with_differential()'s epsilon band and the risk_posture/kappa_rank term).
    # kappa_captain is deliberately an order of magnitude below lambda_value: small enough to
    # still break a genuine exact-mu tie toward the lower-variance player (preserving
    # test_captain_choice_is_risk_aware_not_just_risk_aware_for_squad_membership's original
    # intent when a caller opts into that magnitude explicitly), nowhere near enough to
    # overturn a real multi-point EP gap like the one this default now protects against.
    # Invented v1 default, same status as every other unpinned constant here, flagged for M7
    # recalibration once real backtest evidence exists for what a captaincy bet's risk penalty
    # should actually be.
    params_mod.write_param(con, "captain_risk_aversion_params", 1, "2026-08-10", "kappa_captain", value_numeric=0.01)


# ============================================================
# effective ownership -- FPL is a rank tournament against the field, not a points-forecasting
# contest in isolation. EO_i approximates a player's real exposure across the field: ownership%
# alone understates it since captaincy doubles a captained player's point contribution for
# whoever holds him as captain. EO_i = ownership_i + captaincy_rate_i x 1 is the textbook
# formula (the x1 reflects captaincy's own point-doubling, already "spent" once in ownership_i
# itself, added again for the doubled share). No captaincy-rate data exists anywhere in the
# ingested sources -- checked fact_player_season_stats' full column list and every raw
# FPL-Core-Insights CSV column, neither carries a per-player captaincy percentage -- so this
# falls back to ownership% alone, a real, disclosed understatement for heavily-captained
# players (Haaland's true field exposure is higher than his raw ownership% suggests), not a
# derived number. Flagged here rather than silently assumed complete.
# ============================================================

def effective_ownership(candidate: dict) -> float:
    """Fraction in [0, 1], not a raw percent -- used as a portfolio weight below, where a
    percent-scaled value would inflate the synthetic field portfolio ~100x. Missing
    selected_by_percent (never priced/no ownership row) is treated as 0.0, the conservative,
    non-inflating choice for a genuine absence of ownership data."""
    ownership_percent = candidate.get("selected_by_percent")
    return (ownership_percent or 0.0) / 100.0


# ============================================================
# candidate pool
# ============================================================

def fetch_candidate_pool(
    con: duckdb.DuckDBPyConnection, ep_model_version: int, uncertainty_model_version: int, target_season: str,
) -> list[dict]:
    # player_alias has multiple rows per (player, season) -- full name and web name both
    # get their own alias row (see reconcile.build_dim_player) -- so this dedupes to one
    # team_code per player first, rather than joining directly and fanning out to one row
    # per alias (a real bug this hit: candidate pool came back at ~2x the true player count).
    rows = con.execute(
        """
        WITH team_of AS (
            SELECT DISTINCT player_uid, team_code FROM player_alias WHERE season = ?
        )
        SELECT o.player_uid, dp.position, dp.canonical_name, o.ep_total, u.var_total, t.team_code
        FROM ep_outputs o
        JOIN dim_player dp ON dp.player_uid = o.player_uid
        JOIN uncertainty_outputs u
            ON u.player_uid = o.player_uid AND u.fixture_match_id = o.fixture_match_id
            AND u.model_version = ?
        JOIN team_of t ON t.player_uid = o.player_uid
        WHERE o.model_version = ?
        """,
        [target_season, uncertainty_model_version, ep_model_version],
    ).fetchall()

    prices = dict(con.execute(
        """
        SELECT player_uid, now_cost FROM fact_player_season_stats
        WHERE season = ? AND now_cost IS NOT NULL
        QUALIFY row_number() OVER (PARTITION BY player_uid ORDER BY gw DESC) = 1
        """,
        [target_season],
    ).fetchall())
    # ownership, for the optional captain-differential tie-break (see
    # captain_choice_with_differential()) -- unlike price, missing ownership doesn't exclude a
    # candidate (it's a secondary tie-break signal, not a hard budget requirement); None means
    # "no ownership data for this player," handled as neutral by the tie-break, not silently 0.
    ownership = dict(con.execute(
        """
        SELECT player_uid, selected_by_percent FROM fact_player_season_stats
        WHERE season = ? AND selected_by_percent IS NOT NULL
        QUALIFY row_number() OVER (PARTITION BY player_uid ORDER BY gw DESC) = 1
        """,
        [target_season],
    ).fetchall())
    # Real gap fixed here, caught sanity-checking a live GW1 run: a player with an official,
    # unambiguous chance_of_playing_next_round=0 (FPL's own "will not play" signal, already
    # reconciled into fact_player_season_stats -- not a qualitative claim) was still showing up
    # as an XI-eligible candidate, because nothing downstream of minutes_model_outputs ever
    # checked this field -- the model relied entirely on the SOFT evidence-claims adjustment
    # mechanism to suppress his start probability, and for a source with real decay/reliability
    # suppression applied, that signal can end up too weak to matter (confirmed: a real "Out,
    # groin surgery, expected to miss start of season" claim contributed only -0.11 logit units
    # after decay+reliability weighting, nowhere near enough to counter a historically
    # near-ever-present player's baseline rate). chance_of_playing_next_round=0 is a hard,
    # official exclusion here -- not a down-weight -- since it means the player genuinely
    # cannot score regardless of what the qualitative evidence mechanism computed. A NULL value
    # (the vast majority of players -- no doubt flag at all) is never excluded; only an explicit
    # confirmed zero is.
    unavailable = {
        player_uid for player_uid, chance in con.execute(
            """
            SELECT player_uid, chance_of_playing_next_round FROM fact_player_season_stats
            WHERE season = ? AND chance_of_playing_next_round IS NOT NULL
            QUALIFY row_number() OVER (PARTITION BY player_uid ORDER BY gw DESC) = 1
            """,
            [target_season],
        ).fetchall()
        if chance == 0
    }

    candidates = []
    for player_uid, position, name, mu, var, team_code in rows:
        price = prices.get(player_uid)
        if price is None or position not in POSITIONS or player_uid in unavailable:
            continue
        candidates.append({
            "player_uid": player_uid, "position": position, "name": name,
            "mu": mu, "var": var, "club": team_code, "price": price,
            "selected_by_percent": ownership.get(player_uid),
        })
    return candidates


def fetch_sigma_pairs(con: duckdb.DuckDBPyConnection, uncertainty_model_version: int, player_uids: set[str]) -> dict:
    rows = con.execute(
        "SELECT player_uid_a, player_uid_b, covariance FROM cross_player_covariance WHERE model_version = ?",
        [uncertainty_model_version],
    ).fetchall()
    return {(a, b): cov for a, b, cov in rows if a in player_uids and b in player_uids}


# ============================================================
# MIQP solve (one lambda value)
# ============================================================

def solve(
    candidates: list[dict], sigma_pairs: dict, lam: float, guardrail_cap: float,
    forced_squad_uids: frozenset[str] = frozenset(), forced_xi_uids: frozenset[str] = frozenset(),
    field_cov_by_uid: dict[str, float] | None = None, kappa_rank: float = 0.0, risk_posture: str = "neutral",
    kappa_captain: float = 0.0,
) -> dict:
    """forced_squad_uids/forced_xi_uids: a manager's own hard lock-ins (e.g. "I'm keeping this
    player regardless of what the model ranks him" / "...and starting him"), applied as real
    constraints on the SAME MIQP -- still risk/budget/quota-optimal *given* those picks, not a
    hand-assembled squad pasted in afterward. forced_xi_uids implies squad membership too (the
    existing xi<=squad constraint below already guarantees this; forced_squad_uids need not
    repeat forced_xi_uids). Every forced uid must actually be in the candidate pool (a typo'd
    or unpriced uid fails loudly here, not silently ignored); infeasibility (e.g. forcing more
    than a position's XI quota, or a combined price over budget) surfaces as a normal
    non-optimal solver status, handled the same way run() already handles any other
    non-optimal result.

    field_cov_by_uid/kappa_rank/risk_posture: the rank-relative objective term. FPL rank
    depends on Excess = MyPoints - FieldPoints, not MyPoints alone; Var[Excess] = Var[My] +
    Var[Field] - 2*Cov[My,Field], and Var[Field] is a constant (independent of my squad) that
    drops out. What's left is a REAL Cov(player_i, field_portfolio) per player -- computed
    once, outside this function, from paired Monte Carlo draws sharing the same scenario space
    as the field's own synthetic EO-weighted portfolio (see monte_carlo.compute_field_covariance)
    -- not a naive EO% proxy, which can't distinguish two equally-owned players with different
    variance or team-correlation exposure. Added to the objective as
    posture_sign * kappa_rank * sum_i w_i * field_cov_i, where w_i = xi_i + captain_i (the
    SAME effective weight linear_ep/risk_expr already use two sections down -- a captained
    player's correlation with the field counts double too, matching their doubled point
    contribution). This term is LINEAR in the binary variables (field_cov_i is a precomputed
    constant per player), so it adds zero quadratic complexity to the solve.

    risk_posture="protect" (posture_sign=+1) REWARDS high field-covariance: preferentially
    holds whatever the field's variance is concentrated in, so when those players swing, this
    squad swings WITH the field instead of alone -- reduces Var[Excess]. "chase"
    (posture_sign=-1) penalizes it: pushes toward players genuinely uncorrelated (or
    negatively correlated) with the field, deliberately taking on Var[Excess] to close a rank
    gap. "neutral" (posture_sign=0, the default) makes this whole term vanish regardless of
    kappa_rank or field_cov_by_uid -- an EXACT reduction to the pre-existing objective, not an
    approximate one, so every caller that doesn't opt in sees byte-identical behavior to
    before this parameter existed."""
    posture_sign = {"protect": 1.0, "neutral": 0.0, "chase": -1.0}.get(risk_posture)
    if posture_sign is None:
        raise ValueError(f"risk_posture must be 'protect', 'neutral', or 'chase', got {risk_posture!r}")
    if forced_squad_uids or forced_xi_uids:
        missing = (forced_squad_uids | forced_xi_uids) - {c["player_uid"] for c in candidates}
        if missing:
            raise ValueError(f"forced uids not found in candidate pool: {sorted(missing)}")

    m = scip.Model()
    m.hideOutput()
    m.setParam("limits/time", 300)

    squad = {c["player_uid"]: m.addVar(vtype="B", name=f"squad_{i}") for i, c in enumerate(candidates)}
    xi = {c["player_uid"]: m.addVar(vtype="B", name=f"xi_{i}") for i, c in enumerate(candidates)}
    captain = {c["player_uid"]: m.addVar(vtype="B", name=f"cap_{i}") for i, c in enumerate(candidates)}
    vice = {c["player_uid"]: m.addVar(vtype="B", name=f"vice_{i}") for i, c in enumerate(candidates)}

    for uid in forced_squad_uids:
        m.addCons(squad[uid] == 1)
    for uid in forced_xi_uids:
        m.addCons(xi[uid] == 1)
        m.addCons(squad[uid] == 1)
    m.addCons(scip.quicksum(squad.values()) == 15)
    for pos, quota in POSITION_QUOTA.items():
        m.addCons(scip.quicksum(squad[c["player_uid"]] for c in candidates if c["position"] == pos) == quota)
    m.addCons(scip.quicksum(c["price"] * squad[c["player_uid"]] for c in candidates) <= BUDGET)

    clubs = {c["club"] for c in candidates}
    for club in clubs:
        m.addCons(scip.quicksum(squad[c["player_uid"]] for c in candidates if c["club"] == club) <= 3)

    for c in candidates:
        m.addCons(xi[c["player_uid"]] <= squad[c["player_uid"]])
    m.addCons(scip.quicksum(xi.values()) == 11)
    m.addCons(scip.quicksum(xi[c["player_uid"]] for c in candidates if c["position"] == "Goalkeeper") == 1)
    for pos in ("Defender", "Midfielder", "Forward"):
        total = scip.quicksum(xi[c["player_uid"]] for c in candidates if c["position"] == pos)
        m.addCons(total >= XI_POSITION_MIN[pos])
        m.addCons(total <= XI_POSITION_MAX[pos])

    for c in candidates:
        m.addCons(captain[c["player_uid"]] <= xi[c["player_uid"]])
        m.addCons(vice[c["player_uid"]] <= xi[c["player_uid"]])
    m.addCons(scip.quicksum(captain.values()) == 1)
    m.addCons(scip.quicksum(vice.values()) == 1)
    for c in candidates:
        m.addCons(captain[c["player_uid"]] + vice[c["player_uid"]] <= 1)

    # Guardrail 1: captain cannot be a goalkeeper
    for c in candidates:
        if c["position"] == "Goalkeeper":
            m.addCons(captain[c["player_uid"]] == 0)

    # Guardrail 2: per-club XI concentration cap (alongside, not instead of, the
    # lambda-driven mean-variance mechanism -- duplicates what the squad-level cap already
    # implies, kept as its own constraint per spec so the protection holds even if the
    # squad-level cap is ever loosened)
    for club in clubs:
        m.addCons(scip.quicksum(xi[c["player_uid"]] for c in candidates if c["club"] == club) <= guardrail_cap)

    linear_ep = scip.quicksum(xi[c["player_uid"]] * c["mu"] for c in candidates)
    linear_ep += scip.quicksum(captain[c["player_uid"]] * c["mu"] for c in candidates)

    # Effective scoring weight per player is w_i = xi_i + captain_i (1 for a normal XI player, 2
    # for the captain, since captain[uid] <= xi[uid] is already enforced above) -- exactly
    # mirrors how linear_ep itself is built two lines up. Var(total) = sum_i w_i^2 * var_i +
    # 2 * sum_{i<j} w_i*w_j*cov_ij, using the SAME weights linear_ep uses (Var(2X)=4*Var(X), not
    # Var(X), for a captained player -- captaincy doubling a player's EP contribution but never
    # his variance/covariance contribution was a real, since-fixed bug). Binary algebra
    # (captain_i <= xi_i, at most one captain): w_i^2 = xi_i + 3*captain_i, and for i != j,
    # w_i*w_j = xi_i*xi_j + xi_i*captain_j + captain_i*xi_j (captain_i*captain_j is 0 for i != j
    # since exactly one captain is chosen).
    #
    # Split into two independently-scaled pieces rather than one combined term scaled by a
    # single lambda -- a second real gap this fixed, caught sanity-checking a live GW1 run: the
    # combined-lambda version applied squad-level risk aversion (calibrated for diversifying a
    # 15-player squad) to a single concentrated captaincy bet at 3-4x weight, with no separate
    # cap of its own, which on real data made a 0.4%-owned defensive midfielder the "risk-
    # optimal" captain over a 69%-owned, nearly-double-EP striker -- a real multi-point EP gap
    # overturned by a risk term that was never calibrated for a decision this concentrated.
    # risk_xi_base (scaled by lam, squad-level risk aversion, exactly as before this split for
    # non-captaincy decisions -- flat xi_i weight, unaffected by who captains) governs WHICH 15/
    # 11 players are selected; risk_captain_extra (scaled by kappa_captain, deliberately much
    # smaller -- see seed_v1_params()) governs the marginal risk added by WHO captains, on top
    # of whichever base squad/XI risk_xi_base already selected.
    objective_expr = linear_ep
    if lam > 0:
        risk_xi_base = scip.quicksum(xi[c["player_uid"]] * c["var"] for c in candidates)
        for (a, b), cov in sigma_pairs.items():
            risk_xi_base += 2 * cov * xi[a] * xi[b]
        t_xi = m.addVar(vtype="C", lb=0, name="risk_xi_base")
        m.addCons(t_xi >= risk_xi_base)
        objective_expr = objective_expr - lam * t_xi
    if kappa_captain > 0:
        risk_captain_extra = scip.quicksum(3 * captain[c["player_uid"]] * c["var"] for c in candidates)
        for (a, b), cov in sigma_pairs.items():
            risk_captain_extra += 2 * cov * (xi[a] * captain[b] + captain[a] * xi[b])
        t_captain = m.addVar(vtype="C", lb=0, name="risk_captain_extra")
        m.addCons(t_captain >= risk_captain_extra)
        objective_expr = objective_expr - kappa_captain * t_captain

    if posture_sign != 0.0 and field_cov_by_uid:
        field_overlap = scip.quicksum(
            (xi[c["player_uid"]] + captain[c["player_uid"]]) * field_cov_by_uid.get(c["player_uid"], 0.0)
            for c in candidates
        )
        objective_expr = objective_expr + posture_sign * kappa_rank * field_overlap

    m.setObjective(objective_expr, sense="maximize")

    m.optimize()
    status = m.getStatus()
    if status not in ("optimal", "timelimit") or m.getNSols() == 0:
        return {"status": status, "squad": frozenset(), "xi": frozenset(), "captain": None, "vice": None, "objective": None}

    squad_set = frozenset(uid for uid, v in squad.items() if m.getVal(v) > 0.5)
    xi_set = frozenset(uid for uid, v in xi.items() if m.getVal(v) > 0.5)
    captain_uid = next((uid for uid, v in captain.items() if m.getVal(v) > 0.5), None)
    # Real gap fixed here, caught alongside the captain-risk fix above: vice[uid] never appears
    # in objective_expr anywhere in this function -- the `vice` variables exist purely to
    # satisfy "someone is vice, distinct from captain" constraints, so the solver's own choice
    # of who is entirely arbitrary (whatever feasible assignment it lands on), never EP-driven.
    # Vice-captain points only matter as a fallback (doubled only if the actual captain doesn't
    # play at all), so standard practice -- next-highest-mu XI player after the captain -- is
    # applied here as a deterministic post-processing step rather than by trusting the solver's
    # unconstrained choice of the `vice` decision variable.
    mu_by_uid = {c["player_uid"]: c["mu"] for c in candidates}
    vice_uid = max((uid for uid in xi_set if uid != captain_uid), key=lambda uid: mu_by_uid[uid], default=None)
    return {"status": status, "squad": squad_set, "xi": xi_set, "captain": captain_uid, "vice": vice_uid, "objective": m.getObjVal()}


# ============================================================
# orchestrator -- divergence check runs FIRST, before anything else is trusted
# ============================================================

def run(
    con: duckdb.DuckDBPyConnection,
    calibration_asof_date: date,
    target_season: str,
    target_gameweek: int,
    ep_model_version: int,
    uncertainty_model_version: int,
    lambda_params_version: int,
    guardrail_params_version: int,
    forced_squad_uids: frozenset[str] = frozenset(),
    forced_xi_uids: frozenset[str] = frozenset(),
    risk_posture: str = "neutral",
    rank_posture_params_version: int | None = None,
    field_cov_by_uid: dict[str, float] | None = None,
    captain_lambda_params_version: int = 1,
) -> int:
    lam, _ = params_mod.resolve_param(con, "risk_aversion_params", "lambda_value", lambda_params_version)
    guardrail_cap, _ = params_mod.resolve_param(
        con, "squad_optimizer_guardrail_params", "xi_club_concentration_cap", guardrail_params_version
    )
    # Real bug fix, not an opt-in feature -- see seed_v1_params()'s captain_risk_aversion_params
    # comment. Defaults to v1 (not None/off) so every existing caller picks up the fix without
    # needing to change its call site; a caller can still pass a different version explicitly.
    kappa_captain, _ = params_mod.resolve_param(
        con, "captain_risk_aversion_params", "kappa_captain", captain_lambda_params_version
    )
    # kappa_rank is only ever resolved (and field_cov_by_uid only ever required) when the
    # caller actually opts into a non-neutral posture -- callers that never pass risk_posture
    # keep working unmodified, with the rank-relative term fully absent, not merely zeroed.
    kappa_rank = 0.0
    if risk_posture != "neutral":
        if rank_posture_params_version is None:
            raise ValueError("risk_posture other than 'neutral' requires rank_posture_params_version")
        if not field_cov_by_uid:
            raise ValueError(
                "risk_posture other than 'neutral' requires field_cov_by_uid -- see "
                "monte_carlo.compute_field_covariance() to produce it"
            )
        kappa_rank, _ = params_mod.resolve_param(con, "rank_posture_params", "kappa_rank", rank_posture_params_version)

    candidates = fetch_candidate_pool(con, ep_model_version, uncertainty_model_version, target_season)
    if len(candidates) < 15:
        raise ValueError(f"candidate pool has only {len(candidates)} priced players -- cannot fill a 15-player squad")
    player_uids = {c["player_uid"] for c in candidates}
    sigma_pairs = fetch_sigma_pairs(con, uncertainty_model_version, player_uids)

    # REQUIRED FIRST, before anything else from this optimizer is trusted: solve the same
    # candidate pool once at lambda=0 and once at the frozen lambda. If the two squads are
    # identical, the quadratic risk term is provably not affecting the solve -- carried
    # forward verbatim from LESSONS_LEARNED.md rather than left as unwritten folklore.
    # risk_posture/kappa_rank/field_cov_by_uid/kappa_captain are held IDENTICAL across both
    # solves, same as forced_squad_uids/forced_xi_uids above -- the whole point of this check is
    # to isolate lambda's own effect, so every other input must be held fixed between the two
    # solves. (kappa_captain no longer scales with lambda, so it no longer contributes to
    # "captain differs" between these two solves the way it used to -- see seed_v1_params()'s
    # captain_risk_aversion_params comment. The check remains a real, live guarantee via "XI
    # differs", which lambda still governs on its own.)
    result_zero = solve(
        candidates, sigma_pairs, lam=0.0, guardrail_cap=guardrail_cap,
        forced_squad_uids=forced_squad_uids, forced_xi_uids=forced_xi_uids,
        field_cov_by_uid=field_cov_by_uid, kappa_rank=kappa_rank, risk_posture=risk_posture,
        kappa_captain=kappa_captain,
    )
    result_real = solve(
        candidates, sigma_pairs, lam=lam, guardrail_cap=guardrail_cap,
        forced_squad_uids=forced_squad_uids, forced_xi_uids=forced_xi_uids,
        field_cov_by_uid=field_cov_by_uid, kappa_rank=kappa_rank, risk_posture=risk_posture,
        kappa_captain=kappa_captain,
    )

    # The baseline (lambda=0) solve must itself have actually reached optimality -- a
    # "timelimit" status there is silently accepted identically to "optimal" further down,
    # so a slow/degenerate baseline solve could return an arbitrary incumbent squad and still
    # spuriously "pass" the check below just by differing from result_real. Real bug fixed
    # here: only a genuinely optimal baseline is trustworthy as the comparison point.
    baseline_reliable = result_zero["status"] == "optimal" and bool(result_zero["squad"])

    # Squad-membership alone is too weak a check: a single bench player swapping (while the
    # XI, captain, and vice are all identical) would satisfy `squad != squad` and spuriously
    # "pass," even though nothing that actually affects points changed and the risk term
    # could still be doing essentially nothing on the decisions that matter. Real gap fixed
    # here: require the XI or the captain to actually differ, not just any of the 15 slots.
    meaningfully_different = (
        result_zero["xi"] != result_real["xi"] or result_zero["captain"] != result_real["captain"]
    )

    divergence_passed = baseline_reliable and meaningfully_different
    note = None
    if not divergence_passed:
        if not baseline_reliable:
            note = (
                f"lambda=0 baseline solve did not reach a reliable optimum "
                f"(status={result_zero['status']}) -- cannot trust the divergence comparison."
            )
        else:
            note = (
                f"lambda=0 and lambda={lam} produced the same XI and captain "
                f"({len(result_real['squad'])}-player squad) -- the quadratic risk term is "
                f"provably not affecting the decisions that matter. Per M5's frozen spec this "
                f"is a hard stop, not a warning."
            )

    run_id = con.execute(
        """
        INSERT INTO squad_optimizer_runs
            (calibration_asof_date, target_season, target_gameweek, ep_model_version,
             uncertainty_model_version, lambda_params_version, lambda_value, guardrail_params_version,
             divergence_check_passed, divergence_check_note, solver_status, objective_value)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        RETURNING run_id
        """,
        [calibration_asof_date, target_season, target_gameweek, ep_model_version, uncertainty_model_version,
         lambda_params_version, lam, guardrail_params_version, divergence_passed, note,
         result_real["status"], result_real["objective"]],
    ).fetchone()[0]

    if not divergence_passed:
        # the run itself is logged (so the failure is auditable), but no squad selection is
        # stored or should ever be trusted -- raising here is deliberate, not a formality.
        raise DivergenceCheckFailedError(note)

    if result_real["status"] != "optimal":
        raise RuntimeError(f"solver did not reach optimality at the frozen lambda: status={result_real['status']}")

    for c in candidates:
        uid = c["player_uid"]
        con.execute(
            "INSERT INTO squad_optimizer_selections (run_id, player_uid, in_squad, in_xi, is_captain, is_vice) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [run_id, uid, uid in result_real["squad"], uid in result_real["xi"],
             uid == result_real["captain"], uid == result_real["vice"]],
        )

    return run_id


# ============================================================
# M9 adapter -- guardrail/audit trail, captain-position sanity check
# ============================================================

def explain_run(con: duckdb.DuckDBPyConnection, run_id: int) -> dict:
    """M9's guardrail/audit-trail section: "which M5 guardrails bound for this specific run...
    and the pass/fail result of the lambda=0-vs-lambda=0.15 divergence check." The divergence
    result is already stored; which club(s) actually sit at the concentration cap is not
    computed anywhere else in this project -- this is the first real audit of a *specific*
    stored solution against the two caps (squad-level, hardcoded literal 3 at solve()'s own
    MIQP constraint; XI-level, the resolved xi_club_concentration_cap param), not just
    confirming the constraints existed at solve time.
    """
    run_row = con.execute(
        "SELECT target_season, divergence_check_passed, divergence_check_note, "
        "guardrail_params_version, lambda_value, is_manager_snapshot "
        "FROM squad_optimizer_runs WHERE run_id = ?", [run_id],
    ).fetchone()
    if not run_row:
        raise ValueError(f"no squad_optimizer_runs row for run_id={run_id}")
    target_season, divergence_passed, divergence_note, guardrail_params_version, lambda_value, is_snapshot = run_row

    xi_cap = None
    if guardrail_params_version:
        xi_cap, _ = params_mod.resolve_param(
            con, "squad_optimizer_guardrail_params", "xi_club_concentration_cap", guardrail_params_version
        )

    # Captain-position detection is deliberately its own direct join, not folded into the
    # club-count query below -- a real bug this caught building the test for it: coupling the
    # two meant the captained-GK check (cheap, and arguably the more important of the two)
    # silently no-opped whenever the club-resolution join found nothing, instead of only the
    # club audit degrading gracefully.
    captain_row = con.execute(
        "SELECT s.player_uid, dp.position FROM squad_optimizer_selections s "
        "JOIN dim_player dp ON dp.player_uid = s.player_uid WHERE s.run_id = ? AND s.is_captain",
        [run_id],
    ).fetchone()
    captain_uid, captain_position = captain_row if captain_row else (None, None)

    from . import reconcile as reconcile_mod
    found = reconcile_mod._season_root_table(con, target_season, "teams.csv")
    club_counts_squad, club_counts_xi = {}, {}
    if found:
        rows = con.execute(
            """
            WITH team_of AS (
                SELECT DISTINCT pa.player_uid, ta.team_uid
                FROM player_alias pa
                JOIN "{}" t ON t.code = pa.team_code
                JOIN team_alias ta ON ta.alias_name = t.name AND ta.season = pa.season
                WHERE pa.season = ?
            )
            SELECT s.player_uid, s.in_squad, s.in_xi, dt.canonical_name
            FROM squad_optimizer_selections s
            JOIN team_of tm ON tm.player_uid = s.player_uid
            JOIN dim_team dt ON dt.team_uid = tm.team_uid
            WHERE s.run_id = ?
            """.format(found[1]),
            [target_season, run_id],
        ).fetchall()
        for player_uid, in_squad, in_xi, club_name in rows:
            if in_squad:
                club_counts_squad[club_name] = club_counts_squad.get(club_name, 0) + 1
            if in_xi:
                club_counts_xi[club_name] = club_counts_xi.get(club_name, 0) + 1

    squad_cap = 3  # solve()'s own hardcoded squad-level literal, not a versioned param
    clubs_at_squad_cap = sorted(c for c, n in club_counts_squad.items() if n >= squad_cap)
    clubs_at_xi_cap = sorted(c for c, n in club_counts_xi.items() if xi_cap is not None and n >= xi_cap)

    return {
        "run_id": run_id, "is_manager_snapshot": bool(is_snapshot), "lambda_value": lambda_value,
        "divergence_check_passed": divergence_passed, "divergence_check_note": divergence_note,
        "club_counts_squad": club_counts_squad, "club_counts_xi": club_counts_xi,
        "squad_cap": squad_cap, "xi_cap": xi_cap,
        "clubs_at_squad_cap": clubs_at_squad_cap, "clubs_at_xi_cap": clubs_at_xi_cap,
        "captain_uid": captain_uid, "captain_position": captain_position,
        "captain_is_goalkeeper": captain_position == "Goalkeeper",
    }


# ============================================================
# optional captain-differential tie-break -- read-only advisory, never mutates the stored
# EP/risk-optimal squad_optimizer_selections row (same "diagnostics separate from the frozen
# source of truth" pattern M9's own automated-flags mechanism already uses).
#
# NOT a rank projection: there is no real rival-manager overall-rank data anywhere in the
# ingested sources (verified -- grepped the whole schema/pipeline, nothing tracks other
# managers' squads or the overall leaderboard). selected_by_percent lower-is-more-differentiated
# is a proxy for "how differentiated is my own squad," named plainly here and in every result
# this returns, not oversold as an actual rank estimate.
# ============================================================

def _captain_objective_component(
    xi_uids: list[str], var_by_uid: dict[str, float], mu_by_uid: dict[str, float],
    cov_by_xi_pair: dict[tuple[str, str], float], captain_uid: str, lam: float,
) -> float:
    """The exact slice of solve()'s own objective (linear_ep - lam*risk) that varies with
    captain choice, holding a FIXED XI (xi_i=1 for every uid in xi_uids by construction) --
    closed-form, not a re-solve, since with the XI already fixed only one binary choice (which
    of the 11 is captain) remains free, and solve()'s own w_i=xi_i+captain_i risk weighting
    (see its own docstring) reduces to a simple sum over 11 candidates rather than a new MIQP.
    """
    linear_ep = sum(mu_by_uid.values()) + mu_by_uid[captain_uid]
    if lam <= 0:
        return linear_ep
    risk = sum((1 + (3 if uid == captain_uid else 0)) * var_by_uid[uid] for uid in xi_uids)
    for (a, b), cov in cov_by_xi_pair.items():
        cross = 1 + (1 if b == captain_uid else 0) + (1 if a == captain_uid else 0)
        risk += 2 * cov * cross
    return linear_ep - lam * risk


def captain_choice_with_differential(
    xi_candidates: list[dict], sigma_pairs: dict, lam: float, base_captain_uid: str, tiebreak_epsilon: float,
) -> dict:
    """Given the real XI solve() already chose and the risk-optimal captain it picked, checks
    whether captaining a DIFFERENT XI player instead would cost at most tiebreak_epsilon in the
    SAME risk-adjusted objective solve() itself optimizes -- and if so, among every such
    near-optimal choice (base_captain_uid included), prefers whichever has the lowest
    selected_by_percent. A hard constraint (near_optimal membership), not a blended epsilon
    folded into the objective -- this can only ever break a genuine near-tie, never talk the
    optimizer out of a real EP/risk-driven pick: a candidate outside the epsilon band is never
    eligible regardless of how much lower its ownership is.

    xi_candidates: the solved XI's own candidate dicts (mu/var/selected_by_percent, from
    fetch_candidate_pool()). Goalkeepers are never proposed (captaining a GK is already a
    separate, unconditional guardrail in solve() itself, not repeated here)."""
    xi_uids = [c["player_uid"] for c in xi_candidates if c["position"] != "Goalkeeper"]
    var_by_uid = {c["player_uid"]: c["var"] for c in xi_candidates}
    mu_by_uid = {c["player_uid"]: c["mu"] for c in xi_candidates}
    ownership_by_uid = {c["player_uid"]: c["selected_by_percent"] for c in xi_candidates}
    xi_uid_set = set(c["player_uid"] for c in xi_candidates)
    cov_by_xi_pair = {(a, b): cov for (a, b), cov in sigma_pairs.items() if a in xi_uid_set and b in xi_uid_set}

    scores = {
        uid: _captain_objective_component(xi_uids, var_by_uid, mu_by_uid, cov_by_xi_pair, uid, lam)
        for uid in xi_uids
    }
    base_score = scores[base_captain_uid]
    near_optimal = [uid for uid in xi_uids if scores[uid] >= base_score - tiebreak_epsilon]
    # missing ownership data is neutral (never artificially preferred over a real, lower, known
    # value) -- treated as the least-differentiated case among the near-optimal set, not 0.0
    # (which would wrongly make "no data" look like the most differentiated option available).
    recommended = min(near_optimal, key=lambda uid: ownership_by_uid[uid] if ownership_by_uid[uid] is not None else float("inf"))

    # Captain EO-risk: a distinct, explicit report on TOP OF the tie-break above, not folded
    # into how recommended is picked -- a manager choosing between "nailed high-EO captain" and
    # "differential captain" needs to see the real point-swing between those two SPECIFIC
    # options, not just which one wins the epsilon-band tie-break. field_typical_captain_uid is
    # the highest-EO player in the XI eligible to captain -- the natural proxy for "who does the
    # template captain" (no rival-manager captaincy-rate data exists to do better, same gap
    # effective_ownership() itself already discloses).
    xi_candidates_by_eo = {c["player_uid"]: effective_ownership(c) for c in xi_candidates if c["player_uid"] in xi_uid_set and c["position"] != "Goalkeeper"}
    field_typical_captain_uid = max(xi_candidates_by_eo, key=xi_candidates_by_eo.get)
    captain_eo_risk = {
        "field_typical_captain_uid": field_typical_captain_uid,
        "field_typical_captain_eo": xi_candidates_by_eo[field_typical_captain_uid],
        "recommended_captain_eo": xi_candidates_by_eo[recommended],
        "eo_gap": xi_candidates_by_eo[field_typical_captain_uid] - xi_candidates_by_eo[recommended],
        "objective_gap": scores[field_typical_captain_uid] - scores[recommended],
        "caveat": (
            "EO(ownership%) proxy for 'who the template captains' -- no rival-manager "
            "captaincy-rate data exists to compute this more directly."
        ),
    }

    return {
        "base_captain_uid": base_captain_uid, "recommended_captain_uid": recommended,
        "changed": recommended != base_captain_uid, "near_optimal_candidates": sorted(near_optimal),
        "tiebreak_epsilon": tiebreak_epsilon,
        "captain_eo_risk": captain_eo_risk,
        "caveat": (
            "selected_by_percent-based differentiation proxy, not a real rival-manager "
            "overall-rank projection -- no rival-manager or overall-rank data exists in the "
            "ingested sources."
        ),
    }


def recommend_captain_with_differential(
    con: duckdb.DuckDBPyConnection, run_id: int, differential_tiebreak_params_version: int,
) -> dict:
    """M9-adapter-style read-only wrapper: reads a real squad_optimizer_runs row's own stored
    XI/captain and re-derives the candidate pool it was solved against, then applies
    captain_choice_with_differential(). Never mutates squad_optimizer_selections -- the stored
    captain remains the pure EP/risk-optimal one solve() actually chose; this is an advisory
    overlay for a human (or M9's report) to weigh, not a silent second opinion that overrides
    the audited decision."""
    run_row = con.execute(
        "SELECT ep_model_version, uncertainty_model_version, lambda_value, target_season "
        "FROM squad_optimizer_runs WHERE run_id = ?", [run_id],
    ).fetchone()
    if not run_row:
        raise ValueError(f"no squad_optimizer_runs row for run_id={run_id}")
    ep_model_version, uncertainty_model_version, lam, target_season = run_row

    xi_uids = {
        r[0] for r in con.execute(
            "SELECT player_uid FROM squad_optimizer_selections WHERE run_id = ? AND in_xi", [run_id]
        ).fetchall()
    }
    captain_row = con.execute(
        "SELECT player_uid FROM squad_optimizer_selections WHERE run_id = ? AND is_captain", [run_id]
    ).fetchone()
    if not xi_uids or not captain_row:
        raise ValueError(f"run_id={run_id} has no stored XI/captain -- was the divergence check ever passed for it?")
    base_captain_uid = captain_row[0]

    candidates = fetch_candidate_pool(con, ep_model_version, uncertainty_model_version, target_season)
    xi_candidates = [c for c in candidates if c["player_uid"] in xi_uids]
    sigma_pairs = fetch_sigma_pairs(con, uncertainty_model_version, xi_uids)

    epsilon, _ = params_mod.resolve_param(
        con, "captain_differential_params", "tiebreak_epsilon", differential_tiebreak_params_version
    )
    return {"run_id": run_id, **captain_choice_with_differential(xi_candidates, sigma_pairs, lam, base_captain_uid, epsilon)}
