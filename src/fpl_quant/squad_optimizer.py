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

    candidates = []
    for player_uid, position, name, mu, var, team_code in rows:
        price = prices.get(player_uid)
        if price is None or position not in POSITIONS:
            continue
        candidates.append({
            "player_uid": player_uid, "position": position, "name": name,
            "mu": mu, "var": var, "club": team_code, "price": price,
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

def solve(candidates: list[dict], sigma_pairs: dict, lam: float, guardrail_cap: float) -> dict:
    m = scip.Model()
    m.hideOutput()
    m.setParam("limits/time", 300)

    squad = {c["player_uid"]: m.addVar(vtype="B", name=f"squad_{i}") for i, c in enumerate(candidates)}
    xi = {c["player_uid"]: m.addVar(vtype="B", name=f"xi_{i}") for i, c in enumerate(candidates)}
    captain = {c["player_uid"]: m.addVar(vtype="B", name=f"cap_{i}") for i, c in enumerate(candidates)}
    vice = {c["player_uid"]: m.addVar(vtype="B", name=f"vice_{i}") for i, c in enumerate(candidates)}

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

    if lam > 0:
        risk_expr = scip.quicksum(xi[c["player_uid"]] * c["var"] for c in candidates)
        for (a, b), cov in sigma_pairs.items():
            risk_expr += 2 * cov * xi[a] * xi[b]
        t = m.addVar(vtype="C", lb=0, name="risk")
        m.addCons(t >= risk_expr)
        m.setObjective(linear_ep - lam * t, sense="maximize")
    else:
        m.setObjective(linear_ep, sense="maximize")

    m.optimize()
    status = m.getStatus()
    if status not in ("optimal", "timelimit") or m.getNSols() == 0:
        return {"status": status, "squad": frozenset(), "xi": frozenset(), "captain": None, "vice": None, "objective": None}

    squad_set = frozenset(uid for uid, v in squad.items() if m.getVal(v) > 0.5)
    xi_set = frozenset(uid for uid, v in xi.items() if m.getVal(v) > 0.5)
    captain_uid = next((uid for uid, v in captain.items() if m.getVal(v) > 0.5), None)
    vice_uid = next((uid for uid, v in vice.items() if m.getVal(v) > 0.5), None)
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
) -> int:
    lam, _ = params_mod.resolve_param(con, "risk_aversion_params", "lambda_value", lambda_params_version)
    guardrail_cap, _ = params_mod.resolve_param(
        con, "squad_optimizer_guardrail_params", "xi_club_concentration_cap", guardrail_params_version
    )

    candidates = fetch_candidate_pool(con, ep_model_version, uncertainty_model_version, target_season)
    if len(candidates) < 15:
        raise ValueError(f"candidate pool has only {len(candidates)} priced players -- cannot fill a 15-player squad")
    player_uids = {c["player_uid"] for c in candidates}
    sigma_pairs = fetch_sigma_pairs(con, uncertainty_model_version, player_uids)

    # REQUIRED FIRST, before anything else from this optimizer is trusted: solve the same
    # candidate pool once at lambda=0 and once at the frozen lambda. If the two squads are
    # identical, the quadratic risk term is provably not affecting the solve -- carried
    # forward verbatim from LESSONS_LEARNED.md rather than left as unwritten folklore.
    result_zero = solve(candidates, sigma_pairs, lam=0.0, guardrail_cap=guardrail_cap)
    result_real = solve(candidates, sigma_pairs, lam=lam, guardrail_cap=guardrail_cap)

    divergence_passed = bool(result_zero["squad"]) and result_zero["squad"] != result_real["squad"]
    note = None
    if not divergence_passed:
        note = (
            f"lambda=0 and lambda={lam} produced the identical squad "
            f"({len(result_real['squad'])} players) -- the quadratic risk term is provably "
            f"not affecting the solve. Per M5's frozen spec this is a hard stop, not a warning."
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
