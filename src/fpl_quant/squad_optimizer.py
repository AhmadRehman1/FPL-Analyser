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

from . import field_covariance as field_covariance_mod
from . import ownership as ownership_mod
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

    # Priority 1 -- ownership/EO. captaincy_concentration is an invented v1 default (same
    # invented-default status as every other unpinned constant in this project): how sharply
    # captaincy is modeled as concentrating on the single highest-EP player in a position
    # group (see ownership.estimate_captaincy_rate's own docstring for the reasoning), not a
    # measured quantity -- flagged for M7 recalibration once real backtest evidence exists.
    params_mod.write_param(con, "ownership_params", 1, "2026-08-10", "captaincy_concentration", value_numeric=0.3)
    # posture defaults to "protect" (favor stability over differentiation) as the more
    # conservative starting posture; eo_weight_kappa=0.02 is sized so the EO term can only
    # ever tip a genuinely close EP contest (typical EO values run 0-100+, so the max per-
    # player nudge is ~2 EP-equivalent points), never override a real, larger EP edge --
    # same invented-magnitude status as risk_aversion_params' own lambda before it was pinned
    # by spec, flagged the same way for M7.
    params_mod.write_param(con, "risk_posture_params", 1, "2026-08-10", "posture", value_text="protect")
    params_mod.write_param(con, "risk_posture_params", 1, "2026-08-10", "eo_weight_kappa", value_numeric=0.02)
    # field_covariance_params.kappa: Cov(player, field) values are on a much larger absolute
    # scale than EO (tens to low hundreds in early testing, vs. EO's 0-100 range) since they're
    # products of point totals, not percentages -- kappa is scaled down accordingly so this
    # term plays the same "gentle nudge" role eo_weight_kappa does, not a claim that this ratio
    # is itself calibrated against real data yet.
    params_mod.write_param(con, "field_covariance_params", 1, "2026-08-10", "kappa", value_numeric=0.001)
    # Priority 2 -- bench-quality floor: 0.25 is the spec's own agreed default ("conservative
    # for thin early-season data"), not an invented literal here.
    params_mod.write_param(con, "bench_quality_params", 1, "2026-08-10", "min_bench_p_start_probability", value_numeric=0.25)
    # Priority 2 -- team-concentration risk: 0.0 is the spec's own agreed default (exact
    # no-op) -- the <=3-per-club legality cap stays the only hard constraint at v1; this lever
    # exists for a future, more risk-averse posture without touching the legality cap itself.
    params_mod.write_param(con, "concentration_risk_params", 1, "2026-08-10", "kappa", value_numeric=0.0)


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
        ORDER BY o.player_uid
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

    # p_start_final, for Priority 2's bench-quality floor (see solve()'s own
    # min_bench_p_start_probability handling) -- derived from uncertainty_model_version the
    # same way reporting.build_report already derives it, so this candidate pool can never
    # disagree with the rest of the pipeline about which minutes_model_version is "the" one
    # for this run. A run whose uncertainty_model_version has no matching minutes model (only
    # possible with a malformed/test fixture) gets an empty p_start map -- every candidate's
    # p_start_final is then None, which solve() already treats as "no bench-floor information,
    # don't gate this player" rather than a crash.
    mm_version_row = con.execute(
        "SELECT minutes_model_version FROM uncertainty_model_versions WHERE model_version = ?",
        [uncertainty_model_version],
    ).fetchone()
    p_start = {}
    if mm_version_row is not None:
        p_start = dict(con.execute(
            "SELECT player_uid, p_start_final FROM minutes_model_outputs WHERE model_version = ?",
            [mm_version_row[0]],
        ).fetchall())

    candidates = []
    for player_uid, position, name, mu, var, team_code in rows:
        price = prices.get(player_uid)
        if price is None or position not in POSITIONS:
            continue
        candidates.append({
            "player_uid": player_uid, "position": position, "name": name,
            "mu": mu, "var": var, "club": team_code, "price": price,
            "selected_by_percent": ownership.get(player_uid),
            "p_start_final": p_start.get(player_uid),
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
    candidates: list[dict],
    sigma_pairs: dict,
    lam: float,
    guardrail_cap: float,
    *,
    eo_by_uid: dict[str, float | None] | None = None,
    posture: str | None = None,
    eo_weight_kappa: float = 0.0,
    field_cov_by_uid: dict[str, float] | None = None,
    field_cov_kappa: float = 0.0,
    p_start_by_uid: dict[str, float | None] | None = None,
    min_bench_p_start_probability: float | None = None,
    concentration_kappa: float = 0.0,
) -> dict:
    """Every keyword-only argument here defaults to an exact no-op (None / 0.0) -- an
    existing caller that only ever passes the original four positional arguments gets
    byte-identical behavior to before Priority 1/2 existed. Each feature below is documented
    at its own point of use, not here, matching this module's existing per-guardrail comment
    style rather than a single upfront summary that would drift out of sync with the code."""
    # Root-caused determinism bug: a squad recommendation once
    # flip-flopped between formations/captains across repeated runs against IDENTICAL
    # underlying data. Neither of the two initially-suspected causes held up -- the budget
    # constraint below already sums price*squad[] over every one of the 15 squad slots (not
    # just the XI), and run() already raises loudly whenever the real-lambda solve doesn't
    # reach "optimal" (a timelimit/gap-limited return can never ship silently). The real cause:
    # whenever the true optimum admits more than one exactly-tied solution (an expected,
    # unavoidable occurrence -- e.g. two near-identical players at the bench cut line, or two
    # captain candidates with equal projected points), WHICH tied optimum SCIP's branch-and-
    # bound returns depends on the order variables/constraints were built in. Candidate order
    # previously came straight from fetch_candidate_pool's ORDER-BY-less SQL join (not
    # guaranteed stable across runs when DuckDB executes it across multiple threads), so
    # solve() was not a pure function of its logical inputs. Sorting here makes it one --
    # verified empirically that this, not the two originally-suspected causes, was what made
    # the captain/XI choice order-sensitive.
    candidates = sorted(candidates, key=lambda c: c["player_uid"])
    m = scip.Model()
    m.hideOutput()
    m.setParam("limits/time", 300)
    # Belt-and-braces alongside the candidate sort above: SCIP's own tie-breaking is itself
    # deterministic given a fixed random seed, but leaving it unset relies on pyscipopt's
    # default rather than pinning it explicitly -- pin it so a future pyscipopt/SCIP upgrade
    # changing that default can't silently reintroduce order-sensitivity.
    m.setParam("randomization/randomseedshift", 0)

    squad = {c["player_uid"]: m.addVar(vtype="B", name=f"squad_{i}") for i, c in enumerate(candidates)}
    xi = {c["player_uid"]: m.addVar(vtype="B", name=f"xi_{i}") for i, c in enumerate(candidates)}
    captain = {c["player_uid"]: m.addVar(vtype="B", name=f"cap_{i}") for i, c in enumerate(candidates)}
    vice = {c["player_uid"]: m.addVar(vtype="B", name=f"vice_{i}") for i, c in enumerate(candidates)}

    m.addCons(scip.quicksum(squad.values()) == 15)
    for pos, quota in POSITION_QUOTA.items():
        m.addCons(scip.quicksum(squad[c["player_uid"]] for c in candidates if c["position"] == pos) == quota)
    m.addCons(scip.quicksum(c["price"] * squad[c["player_uid"]] for c in candidates) <= BUDGET)

    # sorted(), not a raw set iteration: Python's set iteration order for strings is
    # hash-randomized per-process by default (PYTHONHASHSEED), so two separate runs of this
    # module against identical data could add the per-club constraints below in a different
    # order -- another input to the same order-sensitive tie-breaking documented on solve()'s
    # own candidate sort above. Belt-and-braces given that sort already fixes variable order;
    # this fixes constraint order too.
    clubs = sorted({c["club"] for c in candidates})
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

    # Priority 2 -- bench-quality floor: a candidate whose real p_start_final sits below
    # min_bench_p_start_probability is thin, early-season-data-driven rotation risk that
    # nobody should be carrying purely as a bench option (their whole value proposition on
    # the bench -- covering a blank/doubtful starter -- assumes they're likely to actually
    # play if called on). squad[uid] - xi[uid] <= 0, combined with xi[uid] <= squad[uid]
    # already enforced above, forces squad[uid] == xi[uid]: this candidate is either not
    # selected at all, or selected AND started -- never benched. Missing p_start data (None)
    # never gates a player -- absence of minutes-model coverage isn't evidence of rotation
    # risk, and this constraint set must stay exactly the same shape (no-op) when the caller
    # doesn't opt in (min_bench_p_start_probability is None).
    if min_bench_p_start_probability is not None:
        p_start_map = p_start_by_uid or {}
        for c in candidates:
            uid = c["player_uid"]
            p_start = p_start_map.get(uid)
            if p_start is not None and p_start < min_bench_p_start_probability:
                m.addCons(squad[uid] - xi[uid] <= 0)

    linear_ep = scip.quicksum(xi[c["player_uid"]] * c["mu"] for c in candidates)
    linear_ep += scip.quicksum(captain[c["player_uid"]] * c["mu"] for c in candidates)

    if lam > 0:
        # Effective scoring weight per player is w_i = xi_i + captain_i (1 for a normal XI
        # player, 2 for the captain, since captain[uid] <= xi[uid] is already enforced above)
        # -- exactly mirrors how linear_ep itself is built two lines up. The risk term must
        # use the SAME weights: Var(total) = sum_i w_i^2 * var_i + 2 * sum_{i<j} w_i*w_j*cov_ij.
        # Previously this used a flat xi_i weight even in the risk term while linear_ep used
        # w_i -- captaincy doubled a player's EP contribution but never doubled (let alone
        # quadrupled, since Var(2X)=4*Var(X)) their variance/covariance contribution, so the
        # optimizer's risk aversion was structurally blind to the single largest variance
        # decision in the whole squad (who to captain). Fixed by using w_i everywhere below.
        # Binary algebra (captain_i <= xi_i, at most one captain): w_i^2 = xi_i + 3*captain_i,
        # and for i != j, w_i*w_j = xi_i*xi_j + xi_i*captain_j + captain_i*xi_j (captain_i*
        # captain_j is 0 for i != j since exactly one captain is chosen).
        risk_expr = scip.quicksum(
            (xi[c["player_uid"]] + 3 * captain[c["player_uid"]]) * c["var"] for c in candidates
        )
        for (a, b), cov in sigma_pairs.items():
            risk_expr += 2 * cov * (xi[a] * xi[b] + xi[a] * captain[b] + captain[a] * xi[b])
        t = m.addVar(vtype="C", lb=0, name="risk")
        m.addCons(t >= risk_expr)
        objective_expr = linear_ep - lam * t
    else:
        objective_expr = linear_ep

    # Priority 1 -- risk-posture (EO) and field-covariance terms. Both are governed by the
    # SAME posture sign: "protect" rewards higher EO / higher Cov(player, field) (players who
    # move WITH the field protect rank -- a captained haul the field also has barely moves
    # your relative position, but so does a blank), "chase" rewards the opposite
    # (differentiation -- a low-EO, low-field-correlation pick swings rank sharply either
    # way, which is exactly the upside a manager trying to climb the table wants). This is
    # the standard rank-relative-variance argument: Var(you - field) = Var(you) + Var(field)
    # - 2*Cov(you, field), so maximizing Cov(you, field) minimizes rank-relative variance
    # (protect) and minimizing it maximizes rank-relative variance (chase) -- not an
    # arbitrarily-chosen sign convention. Each term's own kappa controls how much it can ever
    # move a decision; a real, larger EP edge between two candidates still wins regardless of
    # kappa, exactly as lam already works for the risk term above -- these are additional
    # linear terms in the SAME objective, not a separate epsilon-window mechanism.
    if posture is not None and posture not in ("protect", "chase"):
        raise ValueError(f"posture must be 'protect' or 'chase', got {posture!r}")
    posture_sign = {"protect": 1.0, "chase": -1.0}.get(posture, 0.0)

    if eo_weight_kappa > 0:
        if posture is None:
            raise ValueError("eo_weight_kappa > 0 requires posture to be set ('protect' or 'chase')")
        if eo_by_uid is None:
            raise ValueError("eo_weight_kappa > 0 requires eo_by_uid to be provided")
        eo_terms = [
            xi[c["player_uid"]] * eo_by_uid[c["player_uid"]]
            for c in candidates if eo_by_uid.get(c["player_uid"]) is not None
        ]
        if eo_terms:
            objective_expr += posture_sign * eo_weight_kappa * scip.quicksum(eo_terms)

    if field_cov_kappa > 0:
        if posture is None:
            raise ValueError("field_cov_kappa > 0 requires posture to be set ('protect' or 'chase')")
        if field_cov_by_uid is None:
            raise ValueError("field_cov_kappa > 0 requires field_cov_by_uid to be provided")
        field_cov_terms = [
            xi[c["player_uid"]] * field_cov_by_uid[c["player_uid"]]
            for c in candidates if c["player_uid"] in field_cov_by_uid
        ]
        if field_cov_terms:
            objective_expr += posture_sign * field_cov_kappa * scip.quicksum(field_cov_terms)

    # Priority 2 -- team-concentration risk: reuses sigma_pairs (real Sigma from M4/M6, not an
    # invented "concentration score"), restricted to same-club pairs -- a genuine principal
    # submatrix of the real, PSD Sigma restricted to one club's players is itself PSD, and the
    # union across clubs (cross-club terms zeroed) is block-diagonal-by-club, hence still PSD
    # -- so t_conc >= concentration_expr below is a valid convex epigraph constraint, the same
    # trick the main risk term above already relies on. Weighted by squad membership (not
    # just xi) per spec, so bench stacking of same-club players is priced too, not just XI
    # stacking (the guardrail cap above only ever bounds XI-level club concentration).
    if concentration_kappa > 0:
        club_by_uid = {c["player_uid"]: c["club"] for c in candidates}
        same_club_pairs = {
            (a, b): cov for (a, b), cov in sigma_pairs.items()
            if club_by_uid.get(a) is not None and club_by_uid.get(a) == club_by_uid.get(b)
        }
        if same_club_pairs:
            concentration_expr = scip.quicksum(2 * cov * squad[a] * squad[b] for (a, b), cov in same_club_pairs.items())
            t_conc = m.addVar(vtype="C", lb=0, name="concentration_risk")
            m.addCons(t_conc >= concentration_expr)
            objective_expr -= concentration_kappa * t_conc

    m.setObjective(objective_expr, sense="maximize")

    m.optimize()
    status = m.getStatus()
    if status not in ("optimal", "timelimit") or m.getNSols() == 0:
        return {
            "status": status, "squad": frozenset(), "xi": frozenset(), "captain": None, "vice": None,
            "objective": None, "mip_gap": None,
        }

    squad_set = frozenset(uid for uid, v in squad.items() if m.getVal(v) > 0.5)
    xi_set = frozenset(uid for uid, v in xi.items() if m.getVal(v) > 0.5)
    captain_uid = next((uid for uid, v in captain.items() if m.getVal(v) > 0.5), None)
    vice_uid = next((uid for uid, v in vice.items() if m.getVal(v) > 0.5), None)
    # Priority 2 -- solve-quality transparency: SCIP's own proven relative gap at termination.
    # 0.0 iff the returned solution is proven globally optimal (matching status == "optimal");
    # a nonzero value on a "timelimit" status shows exactly how far from proven optimal the
    # returned incumbent actually is, rather than leaving that invisible.
    return {
        "status": status, "squad": squad_set, "xi": xi_set, "captain": captain_uid, "vice": vice_uid,
        "objective": m.getObjVal(), "mip_gap": m.getGap(),
    }


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
    *,
    ownership_params_version: int | None = None,
    risk_posture_params_version: int | None = None,
    field_covariance_params_version: int | None = None,
    bench_quality_params_version: int | None = None,
    concentration_risk_params_version: int | None = None,
) -> int:
    """The five new keyword-only *_params_version arguments are all independently optional
    and all default to an exact no-op (matching solve()'s own default-off convention) -- an
    existing caller passing only the original eight positional arguments gets byte-identical
    behavior to before Priority 1/2 existed. ownership_params_version and
    risk_posture_params_version must be provided TOGETHER (EO computation and the posture
    term that consumes it are two independently-versioned dimensions, but activating one
    without the other is always a caller mistake, not a valid partial configuration) --
    resolve_param's own "fail loud on a missing lookup" discipline extends here rather than
    silently defaulting one and not the other. field_covariance_params_version additionally
    requires both of those (the field-covariance term needs real EO weights to build its
    synthetic field portfolio -- see field_covariance.compute_field_covariance)."""
    lam, _ = params_mod.resolve_param(con, "risk_aversion_params", "lambda_value", lambda_params_version)
    guardrail_cap, _ = params_mod.resolve_param(
        con, "squad_optimizer_guardrail_params", "xi_club_concentration_cap", guardrail_params_version
    )

    candidates = fetch_candidate_pool(con, ep_model_version, uncertainty_model_version, target_season)
    if len(candidates) < 15:
        raise ValueError(f"candidate pool has only {len(candidates)} priced players -- cannot fill a 15-player squad")
    player_uids = {c["player_uid"] for c in candidates}
    sigma_pairs = fetch_sigma_pairs(con, uncertainty_model_version, player_uids)

    if (ownership_params_version is None) != (risk_posture_params_version is None):
        raise ValueError(
            "ownership_params_version and risk_posture_params_version must both be set together, "
            "or both left unset -- EO computation and the posture term that consumes it are not "
            "independently activatable"
        )

    eo_by_uid, posture, eo_weight_kappa = None, None, 0.0
    if ownership_params_version is not None:
        captaincy_concentration, _ = params_mod.resolve_param(
            con, "ownership_params", "captaincy_concentration", ownership_params_version
        )
        eo_by_uid = ownership_mod.compute_eo_for_pool(candidates, captaincy_concentration)
        _, posture = params_mod.resolve_param(con, "risk_posture_params", "posture", risk_posture_params_version)
        eo_weight_kappa, _ = params_mod.resolve_param(con, "risk_posture_params", "eo_weight_kappa", risk_posture_params_version)

    field_cov_by_uid, field_cov_kappa = None, 0.0
    if field_covariance_params_version is not None:
        if eo_by_uid is None:
            raise ValueError(
                "field_covariance_params_version requires ownership_params_version/"
                "risk_posture_params_version to also be set -- the field-covariance term needs "
                "real EO weights to build its synthetic field portfolio"
            )
        field_cov_kappa, _ = params_mod.resolve_param(con, "field_covariance_params", "kappa", field_covariance_params_version)
        scoring_params_version = con.execute(
            "SELECT scoring_matrix_params_version FROM ep_model_versions WHERE model_version = ?", [ep_model_version]
        ).fetchone()[0]
        rho_residual_params_version = con.execute(
            "SELECT rho_residual_params_version FROM uncertainty_model_versions WHERE model_version = ?",
            [uncertainty_model_version],
        ).fetchone()[0]
        field_cov_by_uid = field_covariance_mod.compute_field_covariance(
            con, candidates, target_season, target_gameweek, ep_model_version, scoring_params_version,
            rho_residual_params_version, eo_by_uid, calibration_asof_date,
        )

    p_start_by_uid, min_bench_p_start_probability = None, None
    if bench_quality_params_version is not None:
        min_bench_p_start_probability, _ = params_mod.resolve_param(
            con, "bench_quality_params", "min_bench_p_start_probability", bench_quality_params_version
        )
        p_start_by_uid = {c["player_uid"]: c["p_start_final"] for c in candidates}

    concentration_kappa = 0.0
    if concentration_risk_params_version is not None:
        concentration_kappa, _ = params_mod.resolve_param(con, "concentration_risk_params", "kappa", concentration_risk_params_version)

    solve_kwargs = dict(
        eo_by_uid=eo_by_uid, posture=posture, eo_weight_kappa=eo_weight_kappa,
        field_cov_by_uid=field_cov_by_uid, field_cov_kappa=field_cov_kappa,
        p_start_by_uid=p_start_by_uid, min_bench_p_start_probability=min_bench_p_start_probability,
        concentration_kappa=concentration_kappa,
    )

    # REQUIRED FIRST, before anything else from this optimizer is trusted: solve the same
    # candidate pool once at lambda=0 and once at the frozen lambda. If the two squads are
    # identical, the quadratic risk term is provably not affecting the solve -- carried
    # forward verbatim from LESSONS_LEARNED.md rather than left as unwritten folklore. Any
    # Priority 1/2 terms above are held IDENTICAL across both solves (same solve_kwargs), so
    # this remains a clean measurement of the risk (Sigma) term's own marginal effect,
    # uncontaminated by whatever else additionally sits in the objective.
    result_zero = solve(candidates, sigma_pairs, lam=0.0, guardrail_cap=guardrail_cap, **solve_kwargs)
    result_real = solve(candidates, sigma_pairs, lam=lam, guardrail_cap=guardrail_cap, **solve_kwargs)

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
             divergence_check_passed, divergence_check_note, solver_status, objective_value,
             ownership_params_version, risk_posture_params_version, field_covariance_params_version,
             bench_quality_params_version, concentration_risk_params_version, mip_gap)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        RETURNING run_id
        """,
        [calibration_asof_date, target_season, target_gameweek, ep_model_version, uncertainty_model_version,
         lambda_params_version, lam, guardrail_params_version, divergence_passed, note,
         result_real["status"], result_real["objective"],
         ownership_params_version, risk_posture_params_version, field_covariance_params_version,
         bench_quality_params_version, concentration_risk_params_version, result_real["mip_gap"]],
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
        "guardrail_params_version, lambda_value, is_manager_snapshot, solver_status, mip_gap "
        "FROM squad_optimizer_runs WHERE run_id = ?", [run_id],
    ).fetchone()
    if not run_row:
        raise ValueError(f"no squad_optimizer_runs row for run_id={run_id}")
    (target_season, divergence_passed, divergence_note, guardrail_params_version, lambda_value,
     is_snapshot, solver_status, mip_gap) = run_row

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
        # Priority 2 -- solve-quality transparency: proved_optimal is derived from
        # solver_status (SCIP only ever reports "optimal" for a proven 0%-gap solution) rather
        # than re-deriving it from mip_gap's own value, since mip_gap can be NULL for a run
        # stored before this column existed -- solver_status is the one field that has always
        # been recorded, so it's the more reliable source of truth for this flag.
        "solver_status": solver_status, "mip_gap": mip_gap, "solve_proved_optimal": solver_status == "optimal",
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

    return {
        "base_captain_uid": base_captain_uid, "recommended_captain_uid": recommended,
        "changed": recommended != base_captain_uid, "near_optimal_candidates": sorted(near_optimal),
        "tiebreak_epsilon": tiebreak_epsilon,
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
