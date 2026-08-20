"""M8: Transfer & Chip Strategy Planner.

Operates on an EXISTING squad -- distinct from M5's from-scratch problem. Nothing in M0-M7
tracks "what the manager actually owns" (squad_optimizer's tables are one-off recommendation
logs, never a persisted holding); manager_state_versions/manager_squad_holdings is genuinely
new state here, bootstrapped once from a real squad_optimizer_runs selection and evolved
forward by M8's own accepted recommendations -- never silently re-derived from a fresh M5
call, since a real manager's holdings legitimately diverge from what M5 would pick fresh (the
actual reason transfer planning is a distinct problem, not just "re-run M5 every week").

No multi-gameweek EP horizon exists anywhere upstream: expected_points.run() takes one
target_gameweek per call. team_strength.calibrate()/minutes_model.run() are gameweek-agnostic
snapshots though (no target_gameweek parameter at all), so a 5-gameweek horizon only costs 5
extra ep.run()/uncertainty.run() calls reusing the same ts_model_version/mm_model_version --
not 5x the full M1-M4 chain. compute_horizon_ep() is that one new piece of orchestration.

Triple Captain needs zero new simulation: captaincy has no effect on monte_carlo.py's
mechanics (doesn't touch any player's minutes state, Plackett-Luce bonus-rank strength, or
Z_fixture draw -- grepped, confirmed), so marginal_value_i = 2*X_i - X_i = X_i exactly per
realization. monte_carlo_player_summary.mean_total/sqrt(var_total) already ARE
E[marginal_value_i]/StdDev[marginal_value_i] -- evaluate_triple_captain() is a read against
M6's existing output, not a new run.

Honest gap, not silently patched: the spec describes transfer valuation using "M4's variance
naturally widening for further-out gameweeks" -- confirmed false of the current
implementation (team_strength.calibrate() produces one horizon-agnostic snapshot reused
unchanged for every target gameweek; uncertainty.run() has no calendar-distance-to-target
term). No new uncertainty-inflation formula is invented here to compensate -- that would be
scope beyond what the locked spec actually specifies as a requirement (it's presented as a
description of an assumed mechanism, not a numbered requirement), so this module inherits
whatever variance M4 actually produces per horizon gameweek, undecorated.
"""

import json
from datetime import date

import duckdb

from . import expected_points as ep
from . import monte_carlo
from . import params as params_mod
from . import squad_optimizer
from . import uncertainty as un_mod

PL = "Premier League"


def _write_manager_snapshot_as_optimizer_run(
    con: duckdb.DuckDBPyConnection, state_version: int, calibration_asof_date: date, target_season: str,
    target_gameweek: int, ep_model_version: int, uncertainty_model_version: int,
) -> int:
    """monte_carlo.run() only knows how to simulate a squad that exists as a real
    squad_optimizer_runs row -- but the manager's actual current holdings were never a real M5
    solve (that's the whole point of M8). Two other approaches were tried and confirmed not to
    work before this one, both by direct testing against the real schema, not assumption:
    (1) making monte_carlo_run_versions.squad_optimizer_run_id nullable -- DuckDB refuses to
    ALTER a table that other tables have foreign keys into (monte_carlo_player_totals/summary/
    empirical_covariance all FK into it); (2) a connection-scoped TEMP TABLE shadow of
    squad_optimizer_runs (M7's asof_scope() pattern) -- satisfies monte_carlo.run()'s own
    SELECT queries fine, but DuckDB validates FK constraints against the real catalog table
    regardless of any TEMP TABLE shadowing a query would otherwise resolve through, so the
    later INSERT into monte_carlo_run_versions still fails FK validation.

    The only path that actually works: a real, permanent row in squad_optimizer_runs, flagged
    is_manager_snapshot=TRUE to stay clearly distinguishable from an actual divergence-checked
    M5 solve. It can't be deleted afterward either (same "can't modify a row with FK
    dependents" limitation, once monte_carlo_run_versions references it) -- a one-way door,
    disclosed here and in the README, not hidden.
    """
    holdings = _read_holdings(con, state_version)
    run_id = con.execute(
        """
        INSERT INTO squad_optimizer_runs
            (run_id, calibration_asof_date, target_season, target_gameweek, ep_model_version,
             uncertainty_model_version, lambda_params_version, lambda_value, guardrail_params_version,
             divergence_check_passed, divergence_check_note, solver_status, objective_value, is_manager_snapshot)
        VALUES (nextval('seq_squad_optimizer_run'), ?, ?, ?, ?, ?, 0, 0.0, 0, TRUE,
                'M8 manager-holdings snapshot, not a real solve', 'manager_snapshot', NULL, TRUE)
        RETURNING run_id
        """,
        [calibration_asof_date, target_season, target_gameweek, ep_model_version, uncertainty_model_version],
    ).fetchone()[0]
    for h in holdings:
        con.execute(
            "INSERT INTO squad_optimizer_selections (run_id, player_uid, in_squad, in_xi, is_captain, is_vice) "
            "VALUES (?, ?, TRUE, ?, ?, ?)",
            [run_id, h["player_uid"], h["in_xi"], h["is_captain"], h["is_vice"]],
        )
    return run_id


def seed_v1_params(con: duckdb.DuckDBPyConnection) -> None:
    # H=5: pinned verbatim by the spec, not invented.
    params_mod.write_param(con, "planning_horizon_params", 1, "2026-08-12", "horizon_gameweeks", value_numeric=5)
    # -4 points per extra transfer: verified via live web search against the Premier League's
    # own site and Fantasy Football Scout (matching M3's verification-gate source pattern),
    # confirmed unchanged for 2026-27 -- not assumed from convention, per the spec's own
    # explicit implementation-time verification requirement.
    params_mod.write_param(con, "transfer_cost_params", 1, "2026-08-12", "points_per_hit", value_numeric=4)
    # kappa_tc: invented v1 default, same status as lambda/xi/rho -- pinned to match M5's
    # lambda_value for lack of any other anchor (same reasoning as M5's own N=3 XI-concentration
    # cap being "chosen to match the existing squad-level cap for consistency, not independently
    # derived"), flagged here for a future M7 recalibration extension, not hidden.
    params_mod.write_param(con, "tc_risk_aversion_params", 1, "2026-08-12", "kappa_tc", value_numeric=0.15)
    # Invented v1 default: the spec requires *a* versioned Wildcard gain threshold but doesn't
    # pin a number. A small positive floor -- big enough to filter out noise-level "gains",
    # small enough not to suppress a genuine, real rebuild opportunity.
    params_mod.write_param(con, "wildcard_gain_threshold_params", 1, "2026-08-12", "min_horizon_gain", value_numeric=8.0)
    # Real bug fixed here: evaluate_free_hit() used to reuse wildcard_gain_threshold_params
    # directly, but Wildcard's gain is summed over the WHOLE squad (~15 players) across the
    # WHOLE horizon (H=5 gameweeks), while Free Hit's gain (see evaluate_free_hit()) is only
    # the starting XI (~11 players) for ONE gameweek -- categorically smaller in typical
    # magnitude, so the shared 8.0 bar almost certainly made Free Hit implausibly hard to ever
    # trigger. Own family, own invented default: scaled down from Wildcard's 8.0 by the same
    # (players x gameweeks) ratio the two gains are actually computed over --
    # 8.0 * (11 players x 1 gameweek) / (15 players x 5 gameweeks) = 8.0 * 11/75 ~= 1.17,
    # rounded to a clean 1.5 -- same invented-default status as every other unpinned threshold
    # in this project (not derived from data), flagged for its own future M7 recalibration,
    # independent of Wildcard's.
    params_mod.write_param(con, "free_hit_gain_threshold_params", 1, "2026-08-12", "min_horizon_gain", value_numeric=1.5)


# ============================================================
# manager state: bootstrap + read helpers
# ============================================================

def _compute_bank_for_squad(con: duckdb.DuckDBPyConnection, target_season: str, player_uids: list[str]) -> float:
    """Leftover budget against M5's BUDGET for a given set of held players -- known exactly
    only when every held player has a resolvable current price (the same price source
    squad_optimizer.fetch_candidate_pool() itself uses); otherwise conservatively 0.0 rather
    than guessing, since overstating bank would let evaluate_transfers() legalize a transfer
    the manager can't actually afford. Shared by bootstrap_from_squad_optimizer_run() (the
    initial squad) and apply_recommendation()'s Wildcard-accept path (a fresh squad replacing
    the whole existing one) -- both are "what's left over from a real M5 BUDGET-constrained
    squad" questions, not two different calculations."""
    prices = dict(con.execute(
        "SELECT player_uid, now_cost FROM fact_player_season_stats WHERE season = ? AND now_cost IS NOT NULL "
        "QUALIFY row_number() OVER (PARTITION BY player_uid ORDER BY gw DESC) = 1",
        [target_season],
    ).fetchall())
    if all(uid in prices for uid in player_uids):
        return max(0.0, squad_optimizer.BUDGET - sum(prices[uid] for uid in player_uids))
    return 0.0


def bootstrap_from_squad_optimizer_run(con: duckdb.DuckDBPyConnection, squad_optimizer_run_id: int) -> int:
    """One-time seed: reads a real squad_optimizer_selections row and writes the first
    manager_state_versions/manager_squad_holdings rows. free_transfers_available starts at 1
    (a fresh account's real starting allocation); chip usage starts empty. bank starts at
    whatever the source squad left unspent against M5's BUDGET -- known exactly only when every
    held player has a resolvable current price (the same price source squad_optimizer.
    fetch_candidate_pool() itself uses); otherwise conservatively starts at 0.0 rather than
    guessing, since overstating bank would let evaluate_transfers() legalize a transfer the
    manager can't actually afford."""
    run_row = con.execute(
        "SELECT target_season, target_gameweek FROM squad_optimizer_runs WHERE run_id = ?", [squad_optimizer_run_id]
    ).fetchone()
    if not run_row:
        raise ValueError(f"no squad_optimizer_runs row for run_id={squad_optimizer_run_id}")
    target_season, target_gameweek = run_row

    holdings = con.execute(
        "SELECT player_uid, in_xi, is_captain, is_vice FROM squad_optimizer_selections "
        "WHERE run_id = ? AND in_squad", [squad_optimizer_run_id],
    ).fetchall()
    if not holdings:
        raise ValueError(f"squad_optimizer_run_id={squad_optimizer_run_id} has no in_squad players -- cannot bootstrap")

    held_uids = [uid for uid, *_ in holdings]
    bank = _compute_bank_for_squad(con, target_season, held_uids)

    state_version = con.execute(
        "INSERT INTO manager_state_versions (season, as_of_gameweek, free_transfers_available, "
        "chips_used_set1, chips_used_set2, derived_from_state_version, bank) "
        "VALUES (?, ?, 1, '[]', '[]', NULL, ?) RETURNING state_version",
        [target_season, target_gameweek, bank],
    ).fetchone()[0]

    for player_uid, in_xi, is_captain, is_vice in holdings:
        con.execute(
            "INSERT INTO manager_squad_holdings (state_version, player_uid, in_xi, is_captain, is_vice) "
            "VALUES (?, ?, ?, ?, ?)",
            [state_version, player_uid, in_xi, is_captain, is_vice],
        )
    return state_version


def _read_holdings(con: duckdb.DuckDBPyConnection, state_version: int) -> list[dict]:
    rows = con.execute(
        "SELECT player_uid, in_xi, is_captain, is_vice FROM manager_squad_holdings WHERE state_version = ?",
        [state_version],
    ).fetchall()
    return [{"player_uid": r[0], "in_xi": r[1], "is_captain": r[2], "is_vice": r[3]} for r in rows]


# ============================================================
# multi-gameweek EP horizon
# ============================================================

def compute_horizon_ep(
    con: duckdb.DuckDBPyConnection,
    calibration_asof_date: date,
    target_season: str,
    start_gameweek: int,
    ts_model_version: int,
    mm_model_version: int,
    horizon_gameweeks: int,
    scoring_params_version: int,
    bps_params_version: int,
    tau_params_version: int,
    rho_residual_params_version: int,
    corr_params_version: int,
) -> dict[int, tuple[int, int]]:
    """One ep.run() + uncertainty.run() pair per gameweek in [start_gameweek,
    start_gameweek+horizon_gameweeks), reusing the same ts_model_version/mm_model_version
    throughout (both are gameweek-agnostic snapshots -- confirmed no target_gameweek
    dependency in either). Returns {gameweek: (ep_model_version, uncertainty_model_version)}.
    A gameweek with no fixtures for either side (ep.run() raises on zero fixtures) is skipped,
    not fatal -- a real blank gameweek is a legitimate, if currently unscheduled, outcome.
    """
    out = {}
    for gw in range(start_gameweek, start_gameweek + horizon_gameweeks):
        try:
            ep_mv = ep.run(
                con, calibration_asof_date, target_season, gw, ts_model_version, mm_model_version,
                scoring_params_version, bps_params_version, tau_params_version,
            )
        except ValueError:
            continue
        un_mv = un_mod.run(
            con, calibration_asof_date, ep_mv, mm_model_version, ts_model_version,
            scoring_params_version, bps_params_version, tau_params_version,
            rho_residual_params_version, corr_params_version,
        )
        out[gw] = (ep_mv, un_mv)
    return out


def _horizon_ep_by_player(con: duckdb.DuckDBPyConnection, target_season: str, horizon_ep_versions: dict[int, tuple[int, int]]) -> dict:
    """{player_uid: {"total_ep": float, "position": str, "club": str, "price": float,
    "per_gw": {gw: mu}}} -- pre-fetched once so evaluate_transfers() does pure in-memory
    arithmetic afterward, no per-candidate-pair DB query."""
    by_player: dict = {}
    for gw, (ep_mv, un_mv) in horizon_ep_versions.items():
        for c in squad_optimizer.fetch_candidate_pool(con, ep_mv, un_mv, target_season):
            row = by_player.setdefault(
                c["player_uid"],
                {"total_ep": 0.0, "position": c["position"], "club": c["club"], "price": c["price"], "per_gw": {}},
            )
            row["total_ep"] += c["mu"]
            row["per_gw"][gw] = c["mu"]
    return by_player


def price_momentum_by_player(
    con: duckdb.DuckDBPyConnection, target_season: str, as_of_gameweek: int, lookback_gameweeks: int = 3,
) -> dict[str, dict]:
    """{player_uid: {"price_delta": float | None, "ownership_delta": float | None}} --
    now_cost/selected_by_percent change over the last lookback_gameweeks gameweeks, both real,
    already-reconciled per-gameweek "live" columns in fact_player_season_stats (see
    reconcile.py's own column-semantics tagging), with each distinct gw preserved as its own
    row -- build_fact_player_season_stats()'s QUALIFY/dedup only ever collapses MULTIPLE
    ingestion batches of the SAME (player, gw), never different gws into each other, confirmed
    by reading that query directly (see reconcile.py's PARTITION BY player_uid, gw). That is a
    structural guarantee, not a claim about the real values: whether now_cost/selected_by_percent
    genuinely move week to week in the real ingested data (vs. the source CSVs happening to
    repeat one current snapshot across every historical gw row) was never checked against real
    data in this session -- data/external/ is gitignored and wasn't present in the environment
    this was built in (see README). If it turns out flat, this degrades safely, not silently
    wrongly: a flat series computes a real price_delta/ownership_delta of 0.0 (a true "no
    movement observed", not a lie), and this signal is already, deliberately, informational-only
    everywhere it's used (see evaluate_transfers() below) -- it never feeds net_value or the
    ranking sort, so a flat/uninformative real signal would be inert, not actively misleading.
    Worth confirming against the real DB before leaning on it operationally -- see README's
    design notes for a one-line check to run.

    A real, secondary signal on its own terms if the data supports it -- a player rising in
    price/ownership is trending toward a rise the transfer window will close, one falling is
    trending toward a fall that frees future budget -- but deliberately NOT folded into
    horizon_value_gain/net_value or the ranking sort anywhere in this module: price movement is
    about budget timing, not about a player's own expected points, and conflating the two would
    corrupt the EP-driven ranking for no good reason. None means no price/ownership row exists
    that far back (a genuinely new player, or early enough in a season that lookback_gameweeks
    of history doesn't exist yet) -- not silently treated as zero movement.
    """
    rows = con.execute(
        "SELECT player_uid, gw, now_cost, selected_by_percent FROM fact_player_season_stats "
        "WHERE season = ? AND gw <= ? AND gw >= ?",
        [target_season, as_of_gameweek, max(1, as_of_gameweek - lookback_gameweeks)],
    ).fetchall()
    by_player: dict[str, dict[int, tuple]] = {}
    for player_uid, gw, now_cost, selected_by_percent in rows:
        by_player.setdefault(player_uid, {})[gw] = (now_cost, selected_by_percent)

    out = {}
    for player_uid, by_gw in by_player.items():
        gws = sorted(by_gw)
        latest_gw, earliest_gw = gws[-1], gws[0]
        latest_price, latest_ownership = by_gw[latest_gw]
        earliest_price, earliest_ownership = by_gw[earliest_gw]
        out[player_uid] = {
            "price_delta": (latest_price - earliest_price) if (latest_price is not None and earliest_price is not None and latest_gw != earliest_gw) else None,
            "ownership_delta": (
                latest_ownership - earliest_ownership
                if (latest_ownership is not None and earliest_ownership is not None and latest_gw != earliest_gw) else None
            ),
        }
    return out


# ============================================================
# transfer evaluation
# ============================================================

def evaluate_transfers(
    con: duckdb.DuckDBPyConnection,
    current_holdings: list[dict],
    target_season: str,
    horizon_ep_versions: dict[int, tuple[int, int]],
    free_transfers_available: int,
    points_per_hit: float,
    max_club_count: int = 3,
    bank: float = 0.0,
    target_gameweek: int | None = None,
    momentum_lookback_gameweeks: int = 3,
) -> list[dict]:
    """Exhaustive single-transfer search: every current squad player x every other real
    candidate, ranked by net value over the horizon. Single-best-transfer-per-gameweek scope
    for v1, not combinatorial multi-transfer search -- the spec's own self-critique leaves the
    solving algorithm as an implementation choice; multi-transfer combos are a natural,
    separately-scoped extension, not silently dropped.

    Enforces the three constraints a transfer must satisfy to be real, not optional
    embellishments: same position (can't swap a defender for a forward), price_in <=
    price_out + bank (a real upgrade can legitimately cost more than the player leaving, funded
    by money already saved up from prior transfers -- see apply_recommendation()'s bank
    bookkeeping; bank defaults to 0.0 for a caller that hasn't tracked it), and the post-swap
    club count staying <= max_club_count (M5's own guardrail, must hold for the resulting squad
    too).

    target_gameweek (optional, default None -- prior behavior, no momentum keys on results):
    when given, each result also carries price_momentum_in/out and ownership_momentum_in/out
    (see price_momentum_by_player()) -- purely informational, never part of net_value or the
    ranking sort, which stays exactly the same EP/risk-driven order regardless of this flag.
    """
    horizon_ep = _horizon_ep_by_player(con, target_season, horizon_ep_versions)
    momentum = price_momentum_by_player(con, target_season, target_gameweek, momentum_lookback_gameweeks) if target_gameweek is not None else {}
    current_uids = {h["player_uid"] for h in current_holdings}
    club_counts: dict[str, int] = {}
    for h in current_holdings:
        info = horizon_ep.get(h["player_uid"])
        if info:
            club_counts[info["club"]] = club_counts.get(info["club"], 0) + 1

    results = []
    for out_uid in current_uids:
        out_info = horizon_ep.get(out_uid)
        if out_info is None:
            continue
        for in_uid, in_info in horizon_ep.items():
            if in_uid in current_uids:
                continue
            if in_info["position"] != out_info["position"]:
                continue
            if in_info["price"] > out_info["price"] + bank:
                continue
            new_club_count = club_counts.get(in_info["club"], 0) + (1 if in_info["club"] != out_info["club"] else 0)
            if in_info["club"] != out_info["club"] and new_club_count > max_club_count:
                continue
            horizon_value_gain = in_info["total_ep"] - out_info["total_ep"]
            transfer_cost = 0.0 if free_transfers_available >= 1 else points_per_hit
            result = {
                "player_out": out_uid, "player_in": in_uid,
                "price_out": out_info["price"], "price_in": in_info["price"],
                "horizon_value_gain": horizon_value_gain, "transfer_cost": transfer_cost,
                "net_value": horizon_value_gain - transfer_cost,
            }
            if target_gameweek is not None:
                in_momentum = momentum.get(in_uid, {"price_delta": None, "ownership_delta": None})
                out_momentum = momentum.get(out_uid, {"price_delta": None, "ownership_delta": None})
                result["price_momentum_in"] = in_momentum["price_delta"]
                result["price_momentum_out"] = out_momentum["price_delta"]
                result["ownership_momentum_in"] = in_momentum["ownership_delta"]
                result["ownership_momentum_out"] = out_momentum["ownership_delta"]
            results.append(result)

    results.sort(key=lambda r: r["net_value"], reverse=True)
    for rank, r in enumerate(results, start=1):
        r["rank"] = rank
    return results


# ============================================================
# chip evaluation
# ============================================================

def ensure_squad_simulation(
    con: duckdb.DuckDBPyConnection, state_version: int, calibration_asof_date: date, target_season: str,
    target_gameweek: int, ep_model_version: int, mm_model_version: int, ts_model_version: int,
    uncertainty_model_version: int, scoring_params_version: int, tau_params_version: int,
    rho_residual_params_version: int, n_antithetic_pairs: int = 5000,
) -> int:
    """Runs M6's real monte_carlo.run() against the manager's *actual* current holdings for
    one gameweek, via a real manager-snapshot squad_optimizer_runs row (see
    _write_manager_snapshot_as_optimizer_run()) -- needed for Triple Captain (Finding 2:
    captaincy has no simulation-mechanics effect, so TC's E[marginal_value]/
    StdDev[marginal_value] terms are exactly mean_total/sqrt(var_total) from this run's own
    monte_carlo_player_summary)."""
    snapshot_run_id = _write_manager_snapshot_as_optimizer_run(
        con, state_version, calibration_asof_date, target_season, target_gameweek, ep_model_version, uncertainty_model_version,
    )
    return monte_carlo.run(
        con, calibration_asof_date, snapshot_run_id, ep_model_version, mm_model_version, ts_model_version,
        uncertainty_model_version, scoring_params_version, tau_params_version, rho_residual_params_version,
        n_antithetic_pairs=n_antithetic_pairs,
    )


def evaluate_triple_captain(
    con: duckdb.DuckDBPyConnection, mc_model_version: int, xi_uids: set[str], kappa_tc_params_version: int,
    horizon_ep_map: dict | None = None,
) -> dict:
    """TC_score_i = E[marginal_value_i] - kappa_tc * StdDev[marginal_value_i], where
    marginal_value_i is exactly player i's own simulated total_points (captaincy has no other
    effect on simulation mechanics -- see module docstring / Finding 2). Direct read against
    M6's existing monte_carlo_player_summary, filtered to the current XI (captaincy is only
    ever assigned to a starting player in real FPL).

    horizon_ep_map (optional -- already computed once by run() for the other chip evaluators,
    reused here, not recomputed) buys the same kind of zero-extra-run "is now the best time"
    signal evaluate_wildcard()/evaluate_free_hit() carry: the winning candidate's own mu
    trajectory across the visible horizon (captain_value_per_gw). A cheap EP proxy, not a
    real Monte Carlo re-simulation at every horizon gameweek -- re-running M6's simulation
    (antithetic-variate, thousands of draws) at each of up to 5 horizon gameweeks, every
    simulated gameweek of a season, was rejected as a real cost blow-up for a comparison
    signal, not a squad-selection decision. mu is the dominant term in tc_score anyway (see
    the formula above), so a per-gw mu trajectory for the SAME candidate is an honest, cheap
    stand-in for "would this player's fixture swing look better later," disclosed as a proxy,
    not asserted as MC-equivalent precision."""
    kappa_tc, _ = params_mod.resolve_param(con, "tc_risk_aversion_params", "kappa_tc", kappa_tc_params_version)
    rows = con.execute(
        "SELECT player_uid, mean_total, var_total FROM monte_carlo_player_summary WHERE model_version = ?",
        [mc_model_version],
    ).fetchall()
    scored = [
        {"player_uid": uid, "tc_score": mean_total - kappa_tc * (var_total ** 0.5), "mean_total": mean_total, "var_total": var_total}
        for uid, mean_total, var_total in rows if uid in xi_uids
    ]
    if not scored:
        return {"recommended": False, "reason": "no simulated XI players found for this model_version"}
    scored.sort(key=lambda r: r["tc_score"], reverse=True)
    best = scored[0]
    captain_value_per_gw = (horizon_ep_map or {}).get(best["player_uid"], {}).get("per_gw", {})
    return {
        "recommended": True, "captain_candidate": best["player_uid"], "tc_score": best["tc_score"],
        "all_candidates": scored, "captain_value_per_gw": captain_value_per_gw,
    }


def evaluate_bench_boost(con: duckdb.DuckDBPyConnection, horizon_ep_versions: dict[int, tuple[int, int]], squad_uids: set[str], xi_uids: set[str]) -> dict:
    """Compares projected bench EP sum (M3's ep_total, not a simulation) across horizon
    gameweeks, recommends the gameweek maximizing it. Bench = squad minus XI."""
    bench_uids = squad_uids - xi_uids
    if not bench_uids:
        return {"recommended": False, "reason": "no bench players (XI equals full squad?)"}
    placeholders = ",".join("?" * len(bench_uids))
    by_gw = {}
    for gw, (ep_mv, _un_mv) in horizon_ep_versions.items():
        total = con.execute(
            f"SELECT coalesce(sum(ep_total), 0) FROM ep_outputs WHERE model_version = ? AND player_uid IN ({placeholders})",
            [ep_mv, *bench_uids],
        ).fetchone()[0]
        by_gw[gw] = total
    if not by_gw:
        return {"recommended": False, "reason": "no horizon gameweeks with fixtures"}
    best_gw = max(by_gw, key=by_gw.get)
    return {"recommended": True, "target_gameweek": best_gw, "bench_ep_sum": by_gw[best_gw], "all_gameweeks": by_gw}


def evaluate_wildcard(
    con: duckdb.DuckDBPyConnection, calibration_asof_date: date, target_season: str, target_gameweek: int,
    current_squad_horizon_value: float, best_transfer_net_value: float, horizon_ep_versions: dict[int, tuple[int, int]],
    lambda_params_version: int, guardrail_params_version: int, threshold_params_version: int,
    current_holdings: list[dict] | None = None,
) -> dict:
    """Calls M5 fresh (a real, logged, divergence-checked squad_optimizer.run() -- not raw
    solve(), so a Wildcard recommendation carries the same audit trail as any other real M5
    solve) at target_gameweek, compares its projected horizon value against the *best
    available alternative* -- current squad plus whatever single transfer would otherwise be
    made -- not a do-nothing baseline, since that's the real choice a manager faces. Recommends
    Wildcard when the gain clears wildcard_gain_threshold_params.

    current_holdings (optional -- only the run()/season-simulation caller needs it) also buys
    a real, zero-extra-solve "is now actually the best time" signal: current_squad_value_per_gw,
    the CURRENT squad's own already-computed per-gw mu trajectory across the same visible
    horizon (no new solve, horizon_ep_map is already being built for fresh_squad_horizon_value
    above). Whoever weighs "play now vs. hold" (see backtest._decide_gameweek_action()) can
    read this to check whether target_gameweek is genuinely the worst point in the CURRENT
    squad's own visible trajectory -- the real situation Wildcard exists to fix -- rather than
    just clearing the threshold in isolation. Deliberately NOT folded into `recommended` here:
    a live one-off planner run has no "later week" to hold out for in the same sense a season
    simulation does, so this stays additive, informational detail, not a changed gate."""
    if target_gameweek not in horizon_ep_versions:
        return {"recommended": False, "reason": "no fixtures this gameweek -- cannot rebuild"}
    ep_mv, un_mv = horizon_ep_versions[target_gameweek]
    fresh_run_id = squad_optimizer.run(
        con, calibration_asof_date, target_season, target_gameweek, ep_mv, un_mv,
        lambda_params_version, guardrail_params_version,
    )
    fresh_uids = {
        r[0] for r in con.execute(
            "SELECT player_uid FROM squad_optimizer_selections WHERE run_id = ? AND in_squad", [fresh_run_id]
        ).fetchall()
    }
    horizon_ep_map = _horizon_ep_by_player(con, target_season, horizon_ep_versions)
    fresh_squad_horizon_value = sum(horizon_ep_map.get(uid, {}).get("total_ep", 0.0) for uid in fresh_uids)

    baseline_value = current_squad_horizon_value + best_transfer_net_value
    gain = fresh_squad_horizon_value - baseline_value
    threshold, _ = params_mod.resolve_param(con, "wildcard_gain_threshold_params", "min_horizon_gain", threshold_params_version)
    current_squad_value_per_gw = {}
    if current_holdings is not None:
        current_squad_uids = {h["player_uid"] for h in current_holdings}
        current_squad_value_per_gw = {
            gw: sum(horizon_ep_map.get(uid, {}).get("per_gw", {}).get(gw, 0.0) for uid in current_squad_uids)
            for gw in horizon_ep_versions
        }
    return {
        "recommended": gain > threshold, "fresh_run_id": fresh_run_id,
        "fresh_squad_horizon_value": fresh_squad_horizon_value, "baseline_value": baseline_value, "gain": gain,
        "current_squad_value_per_gw": current_squad_value_per_gw,
    }


def evaluate_free_hit(
    con: duckdb.DuckDBPyConnection, calibration_asof_date: date, target_season: str, target_gameweek: int,
    current_holdings: list[dict], horizon_ep_versions: dict[int, tuple[int, int]],
    lambda_params_version: int, guardrail_params_version: int, threshold_params_version: int,
) -> dict:
    """One-gameweek-only rebuild (unlike Wildcard, only target_gameweek's own EP, not the
    whole horizon -- the squad reverts after this single gameweek). Given the confirmed
    absence of scheduled doubles/blanks in 2026-27 (M8's own spec finding), FH's real use case
    here is a single gameweek with an unusually poor fixture swing for the current squad, not
    DGW exploitation -- re-evaluated against live fixture data every call, never assumed fixed.
    Uses its own free_hit_gain_threshold_params family, not Wildcard's -- a real bug this used
    to have: reusing wildcard_gain_threshold_params (calibrated for a ~15-player x 5-gameweek
    sum) as the bar for a ~11-player x 1-gameweek gain structurally miscalibrated one of the
    two chips. threshold_params_version here resolves against free_hit_gain_threshold_params."""
    if target_gameweek not in horizon_ep_versions:
        return {"recommended": False, "reason": "no fixtures this gameweek -- cannot rebuild"}
    ep_mv, un_mv = horizon_ep_versions[target_gameweek]
    fresh_run_id = squad_optimizer.run(
        con, calibration_asof_date, target_season, target_gameweek, ep_mv, un_mv,
        lambda_params_version, guardrail_params_version,
    )
    fresh_xi_uids = {
        r[0] for r in con.execute(
            "SELECT player_uid FROM squad_optimizer_selections WHERE run_id = ? AND in_xi", [fresh_run_id]
        ).fetchall()
    }
    mu_by_uid = {c["player_uid"]: c["mu"] for c in squad_optimizer.fetch_candidate_pool(con, ep_mv, un_mv, target_season)}
    fresh_gw_value = sum(mu_by_uid.get(uid, 0.0) for uid in fresh_xi_uids)
    current_xi_uids = {h["player_uid"] for h in current_holdings if h["in_xi"]}
    current_gw_value = sum(mu_by_uid.get(uid, 0.0) for uid in current_xi_uids)

    gain = fresh_gw_value - current_gw_value
    threshold, _ = params_mod.resolve_param(con, "free_hit_gain_threshold_params", "min_horizon_gain", threshold_params_version)
    # Same zero-extra-solve "is now really the best time" signal evaluate_wildcard() computes
    # (see its own docstring) -- the current XI's own mu trajectory across the visible horizon,
    # reusing the same _horizon_ep_by_player() helper. Free Hit's economics are XI-only and
    # single-gameweek (unlike Wildcard's whole-squad/whole-horizon gain), so the trajectory is
    # XI-only too, matching current_gw_value/fresh_gw_value above -- one extra DB read (already
    # paid by evaluate_bench_boost()'s own by-gameweek loop pattern), no new solves.
    horizon_ep_map = _horizon_ep_by_player(con, target_season, horizon_ep_versions)
    current_xi_value_per_gw = {
        gw: sum(horizon_ep_map.get(uid, {}).get("per_gw", {}).get(gw, 0.0) for uid in current_xi_uids)
        for gw in horizon_ep_versions
    }
    return {
        "recommended": gain > threshold, "fresh_run_id": fresh_run_id,
        "fresh_gw_value": fresh_gw_value, "current_gw_value": current_gw_value, "gain": gain,
        "current_xi_value_per_gw": current_xi_value_per_gw,
    }


# ============================================================
# GW19 hard deadline
# ============================================================

ALL_CHIP_TYPES = frozenset({"wildcard", "free_hit", "triple_captain", "bench_boost"})
GW19_DEADLINE_GAMEWEEK = 19


def check_gw19_deadline(target_gameweek: int, chips_used_set1: list[str], warning_window: int = 3) -> dict:
    """Chip set 1 is forfeited entirely, not softly discounted, if unused by the GW19
    deadline -- modeled here as an explicit use-it-or-lose-it flag, not a preference that can
    silently lapse (per the spec's own explicit requirement).

    Real bug fixed here: `urgent`'s lower bound was `0 <= gameweeks_remaining`, so at
    target_gameweek == GW19_DEADLINE_GAMEWEEK itself (gameweeks_remaining == 0) both `urgent`
    and `forfeited_now` came out True simultaneously -- a self-contradictory "hurry, use it
    now" plus "it's already gone" pair, written straight into chip_evaluations.gw19_urgent_flag
    for M9 to display. `urgent` now requires at least 1 gameweek still remaining; GW19 itself is
    exclusively `forfeited_now`."""
    unused = ALL_CHIP_TYPES - set(chips_used_set1)
    gameweeks_remaining = GW19_DEADLINE_GAMEWEEK - target_gameweek
    urgent = 1 <= gameweeks_remaining <= warning_window and bool(unused)
    forfeited_now = target_gameweek >= GW19_DEADLINE_GAMEWEEK and bool(unused)
    return {
        "unused_set1_chips": sorted(unused), "gameweeks_until_gw19": gameweeks_remaining,
        "urgent": urgent, "forfeited_now": forfeited_now,
    }


# ============================================================
# orchestrator
# ============================================================

def run(
    con: duckdb.DuckDBPyConnection,
    calibration_asof_date: date,
    target_season: str,
    target_gameweek: int,
    input_state_version: int,
    ts_model_version: int,
    mm_model_version: int,
    horizon_params_version: int,
    scoring_params_version: int,
    bps_params_version: int,
    tau_params_version: int,
    rho_residual_params_version: int,
    corr_params_version: int,
    transfer_cost_params_version: int,
    lambda_params_version: int,
    guardrail_params_version: int,
    wildcard_threshold_params_version: int,
    free_hit_threshold_params_version: int,
    kappa_tc_params_version: int,
) -> int:
    """One planning invocation: computes the horizon EP, evaluates transfers and all four
    chips against the manager's actual current holdings (input_state_version), writes
    transfer_plan_runs + transfer_recommendations + chip_evaluations, returns run_id. Proposes
    only -- does not advance manager_state_versions (see apply_recommendation()), mirroring
    M7's propose-then-confirm gate."""
    state_row = con.execute(
        "SELECT season, free_transfers_available, chips_used_set1, bank FROM manager_state_versions WHERE state_version = ?",
        [input_state_version],
    ).fetchone()
    if not state_row:
        raise ValueError(f"no manager_state_versions row for state_version={input_state_version}")
    _season, free_transfers_available, chips_used_set1_json, bank = state_row
    chips_used_set1 = json.loads(chips_used_set1_json)
    current_holdings = _read_holdings(con, input_state_version)
    if not current_holdings:
        raise ValueError(f"manager_state_version={input_state_version} has no holdings -- cannot plan")

    horizon_gameweeks, _ = params_mod.resolve_param(con, "planning_horizon_params", "horizon_gameweeks", horizon_params_version)
    horizon_ep_versions = compute_horizon_ep(
        con, calibration_asof_date, target_season, target_gameweek, ts_model_version, mm_model_version,
        int(horizon_gameweeks), scoring_params_version, bps_params_version, tau_params_version,
        rho_residual_params_version, corr_params_version,
    )

    points_per_hit, _ = params_mod.resolve_param(con, "transfer_cost_params", "points_per_hit", transfer_cost_params_version)
    transfer_results = evaluate_transfers(
        con, current_holdings, target_season, horizon_ep_versions, free_transfers_available, points_per_hit,
        bank=bank or 0.0, target_gameweek=target_gameweek,
    )

    horizon_ep_map = _horizon_ep_by_player(con, target_season, horizon_ep_versions)
    current_squad_horizon_value = sum(horizon_ep_map.get(h["player_uid"], {}).get("total_ep", 0.0) for h in current_holdings)
    best_transfer_net_value = transfer_results[0]["net_value"] if transfer_results else 0.0

    wildcard_result = evaluate_wildcard(
        con, calibration_asof_date, target_season, target_gameweek, current_squad_horizon_value,
        best_transfer_net_value, horizon_ep_versions, lambda_params_version, guardrail_params_version,
        wildcard_threshold_params_version, current_holdings=current_holdings,
    )
    free_hit_result = evaluate_free_hit(
        con, calibration_asof_date, target_season, target_gameweek, current_holdings, horizon_ep_versions,
        lambda_params_version, guardrail_params_version, free_hit_threshold_params_version,
    )

    xi_uids = {h["player_uid"] for h in current_holdings if h["in_xi"]}
    squad_uids = {h["player_uid"] for h in current_holdings}
    if target_gameweek in horizon_ep_versions:
        ep_mv, un_mv = horizon_ep_versions[target_gameweek]
        mc_model_version = ensure_squad_simulation(
            con, input_state_version, calibration_asof_date, target_season, target_gameweek, ep_mv,
            mm_model_version, ts_model_version, un_mv, scoring_params_version, tau_params_version,
            rho_residual_params_version,
        )
        tc_result = evaluate_triple_captain(con, mc_model_version, xi_uids, kappa_tc_params_version, horizon_ep_map=horizon_ep_map)
    else:
        tc_result = {"recommended": False, "reason": "no fixtures this gameweek"}
    bb_result = evaluate_bench_boost(con, horizon_ep_versions, squad_uids, xi_uids)

    gw19 = check_gw19_deadline(target_gameweek, chips_used_set1)

    run_id = con.execute(
        """
        INSERT INTO transfer_plan_runs
            (calibration_asof_date, target_season, target_gameweek, input_state_version,
             horizon_params_version, transfer_cost_params_version, ep_model_versions, uncertainty_model_versions)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        RETURNING run_id
        """,
        [calibration_asof_date, target_season, target_gameweek, input_state_version, horizon_params_version,
         transfer_cost_params_version, json.dumps({str(gw): v[0] for gw, v in horizon_ep_versions.items()}),
         json.dumps({str(gw): v[1] for gw, v in horizon_ep_versions.items()})],
    ).fetchone()[0]

    for r in transfer_results[:10]:  # top 10 stored -- not all ~8k candidate pairs, this is a report, not an audit of the whole search
        con.execute(
            "INSERT INTO transfer_recommendations "
            "(run_id, rank, player_out, player_in, price_out, price_in, horizon_value_gain, transfer_cost, net_value) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [run_id, r["rank"], r["player_out"], r["player_in"], r["price_out"], r["price_in"],
             r["horizon_value_gain"], r["transfer_cost"], r["net_value"]],
        )

    for chip_type, result in (
        ("wildcard", wildcard_result), ("free_hit", free_hit_result),
        ("triple_captain", tc_result), ("bench_boost", bb_result),
    ):
        score = result.get("gain", result.get("tc_score", result.get("bench_ep_sum")))
        con.execute(
            "INSERT INTO chip_evaluations (run_id, chip_type, recommended, score_or_gain, detail, gw19_urgent_flag) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [run_id, chip_type, bool(result.get("recommended", False)), score, json.dumps(result, default=str),
             gw19["urgent"] and chip_type in gw19["unused_set1_chips"]],
        )

    return run_id


def _read_fresh_chip_squad(con: duckdb.DuckDBPyConnection, run_id: int, chip_type: str) -> list[dict]:
    """Reads the fresh M5 squad a Wildcard/Free Hit recommendation evaluated, via the
    fresh_run_id captured in chip_evaluations.detail (see evaluate_wildcard()'s/
    evaluate_free_hit()'s own return dicts, both of which include "fresh_run_id"). Shared by
    apply_recommendation()'s Wildcard-accept squad rebuild and by a season simulation's
    one-off Free Hit scoring (see backtest.run_season_simulation()) -- one real read against
    the actual solved squad, not two places independently re-deriving the same thing."""
    row = con.execute(
        "SELECT detail FROM chip_evaluations WHERE run_id = ? AND chip_type = ?", [run_id, chip_type]
    ).fetchone()
    if not row or not row[0]:
        raise ValueError(f"no chip_evaluations row for run_id={run_id} chip_type={chip_type!r}")
    fresh_run_id = json.loads(row[0]).get("fresh_run_id")
    if fresh_run_id is None:
        raise ValueError(f"chip_evaluations detail for run_id={run_id} chip_type={chip_type!r} has no fresh_run_id")
    rows = con.execute(
        "SELECT player_uid, in_xi, is_captain, is_vice FROM squad_optimizer_selections WHERE run_id = ? AND in_squad",
        [fresh_run_id],
    ).fetchall()
    if not rows:
        raise ValueError(f"fresh_run_id={fresh_run_id} (from run_id={run_id} chip_type={chip_type!r}) has no in_squad players")
    return [{"player_uid": uid, "in_xi": in_xi, "is_captain": is_captain, "is_vice": is_vice} for uid, in_xi, is_captain, is_vice in rows]


def apply_recommendation(
    con: duckdb.DuckDBPyConnection, run_id: int, *, accept_transfer_rank: int | None = None, accept_chip: str | None = None,
) -> int:
    """Writes a new manager_state_versions row reflecting an accepted recommendation, with
    produced_by_run_id set at INSERT time (not an UPDATE to transfer_plan_runs afterward --
    DuckDB refuses to UPDATE a row with FK-referencing children in another table, confirmed by
    testing; see schema/0009's comment on that column) -- mirrors M7's propose-then-confirm
    gate (run() proposes and logs; nothing is applied automatically). accept_transfer_rank=None
    means "make no transfer this gameweek" (free_transfers_available increments, capped at 5,
    per the real banking rule).

    accept_chip="wildcard": real bug fixed here -- this used to only record the chip as used
    and never actually touch holdings_by_uid, so accepting a Wildcard was a complete no-op on
    the squad. Wildcard genuinely replaces the whole squad (unlike Free Hit, which reverts
    after one gameweek and is deliberately NOT handled here at all -- leaving holdings
    unchanged on accept_chip="free_hit" is correct, not an oversight: nothing should persist
    forward from a one-week-only rebuild. Scoring that one gameweek off the fresh Free Hit
    squad, then continuing next week from the pre-Free-Hit holdings, is the caller's job --
    see backtest.run_season_simulation() and _read_fresh_chip_squad() above.) accept_chip
    and accept_transfer_rank are mutually exclusive when the chip is "wildcard": a real M5
    solve already replaces every holding, so layering a single-player transfer on top of it
    isn't a coherent action, not a case to silently pick one of two winners for."""
    if accept_chip == "wildcard" and accept_transfer_rank is not None:
        raise ValueError("cannot accept both a transfer and Wildcard in the same call -- Wildcard already replaces the whole squad")

    run_row = con.execute(
        "SELECT input_state_version, target_season, target_gameweek FROM transfer_plan_runs WHERE run_id = ?", [run_id]
    ).fetchone()
    if not run_row:
        raise ValueError(f"no transfer_plan_runs row for run_id={run_id}")
    input_state_version, target_season, target_gameweek = run_row

    state_row = con.execute(
        "SELECT free_transfers_available, chips_used_set1, chips_used_set2, bank FROM manager_state_versions WHERE state_version = ?",
        [input_state_version],
    ).fetchone()
    free_transfers_available, chips_used_set1_json, chips_used_set2_json, bank = state_row
    chips_used_set1 = set(json.loads(chips_used_set1_json))
    chips_used_set2 = set(json.loads(chips_used_set2_json))
    holdings_by_uid = {h["player_uid"]: h for h in _read_holdings(con, input_state_version)}
    new_bank = bank or 0.0

    if accept_chip == "wildcard":
        # The actual fix: rebuild holdings from the fresh M5 squad Wildcard evaluated (used to
        # be a complete no-op on the squad -- see this function's own docstring), and recompute
        # bank the same way bootstrap_from_squad_optimizer_run() does for any real M5-solved
        # squad (leftover against BUDGET), since the old bank figure belonged to a squad that
        # no longer exists after a full rebuild.
        fresh_holdings = _read_fresh_chip_squad(con, run_id, "wildcard")
        holdings_by_uid = {h["player_uid"]: h for h in fresh_holdings}
        new_bank = _compute_bank_for_squad(con, target_season, list(holdings_by_uid.keys()))
        new_free_transfers = min(5, free_transfers_available + 1)  # Wildcard doesn't consume/grant transfers beyond the normal weekly allocation
    elif accept_transfer_rank is not None:
        rec = con.execute(
            "SELECT player_out, player_in, price_out, price_in, transfer_cost FROM transfer_recommendations "
            "WHERE run_id = ? AND rank = ?",
            [run_id, accept_transfer_rank],
        ).fetchone()
        if not rec:
            raise ValueError(f"no transfer_recommendations row for run_id={run_id} rank={accept_transfer_rank}")
        player_out, player_in, price_out, price_in, transfer_cost = rec
        outgoing = holdings_by_uid.pop(player_out)
        holdings_by_uid[player_in] = {
            "player_uid": player_in, "in_xi": outgoing["in_xi"], "is_captain": False, "is_vice": False,
        }
        # bank moves by exactly what the swap frees up or costs -- selling a pricier player
        # than the one bought in grows bank, buying up spends it down. This is what actually
        # closes the "can't afford Haaland" gap: evaluate_transfers() only allows price_in >
        # price_out up to this same bank, so a transfer that draws it down here is exactly one
        # evaluate_transfers() already verified the manager could afford.
        new_bank = new_bank + (price_out or 0.0) - (price_in or 0.0)
        # Real FPL rule: a new free transfer is granted every gameweek regardless of whether
        # a transfer was made this week -- this must apply on the accepted-transfer path too,
        # not just the "no transfer" path below. Previously this line only decremented (or
        # held flat on a paid hit) with no +1, so free_transfers_available silently drained
        # toward 0 every time a free transfer was actually used, and never grew back --
        # a real bug: it made the planner progressively undercount how many free transfers
        # it had available, biasing it toward pricing genuinely-free transfers as -4 hits.
        new_free_transfers = min(5, max(0, free_transfers_available - (1 if transfer_cost == 0.0 else 0)) + 1)
    else:
        new_free_transfers = min(5, free_transfers_available + 1)  # banked, unused this gameweek

    if accept_chip is not None:
        if target_gameweek < GW19_DEADLINE_GAMEWEEK:
            chips_used_set1 = chips_used_set1 | {accept_chip}
        else:
            chips_used_set2 = chips_used_set2 | {accept_chip}

    new_state_version = con.execute(
        "INSERT INTO manager_state_versions "
        "(season, as_of_gameweek, free_transfers_available, chips_used_set1, chips_used_set2, "
        "derived_from_state_version, produced_by_run_id, bank) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?) RETURNING state_version",
        [target_season, target_gameweek + 1, new_free_transfers, json.dumps(sorted(chips_used_set1)),
         json.dumps(sorted(chips_used_set2)), input_state_version, run_id, new_bank],
    ).fetchone()[0]
    for h in holdings_by_uid.values():
        con.execute(
            "INSERT INTO manager_squad_holdings (state_version, player_uid, in_xi, is_captain, is_vice) VALUES (?, ?, ?, ?, ?)",
            [new_state_version, h["player_uid"], h["in_xi"], h["is_captain"], h["is_vice"]],
        )
    return new_state_version


# ============================================================
# M9 adapter -- transfer/chip rationale
# ============================================================

def explain_plan(con: duckdb.DuckDBPyConnection, run_id: int, top_n: int = 5) -> dict:
    """M9's transfer/chip-rationale section, including GW19 urgency flagging. Pure assembly --
    transfer_recommendations and chip_evaluations.detail already carry everything a rationale
    needs (see this module's own run(), which writes fully-populated detail JSON per chip), no
    new computation."""
    run_row = con.execute(
        "SELECT target_season, target_gameweek FROM transfer_plan_runs WHERE run_id = ?", [run_id]
    ).fetchone()
    if not run_row:
        raise ValueError(f"no transfer_plan_runs row for run_id={run_id}")
    target_season, target_gameweek = run_row

    recs = con.execute(
        "SELECT rank, player_out, player_in, horizon_value_gain, transfer_cost, net_value "
        "FROM transfer_recommendations WHERE run_id = ? ORDER BY rank LIMIT ?", [run_id, top_n],
    ).fetchall()
    top_transfers = [
        {"rank": r, "player_out": out_uid, "player_in": in_uid, "horizon_value_gain": gain,
         "transfer_cost": cost, "net_value": net}
        for r, out_uid, in_uid, gain, cost, net in recs
    ]

    chip_rows = con.execute(
        "SELECT chip_type, recommended, score_or_gain, gw19_urgent_flag, detail FROM chip_evaluations WHERE run_id = ?",
        [run_id],
    ).fetchall()
    chips = {
        chip_type: {"recommended": bool(recommended), "score_or_gain": score, "gw19_urgent": bool(urgent), "detail": json.loads(detail)}
        for chip_type, recommended, score, urgent, detail in chip_rows
    }

    state_row = con.execute(
        "SELECT chips_used_set1 FROM manager_state_versions WHERE state_version = "
        "(SELECT input_state_version FROM transfer_plan_runs WHERE run_id = ?)", [run_id],
    ).fetchone()
    chips_used_set1 = json.loads(state_row[0]) if state_row else []
    gw19 = check_gw19_deadline(target_gameweek, chips_used_set1)

    return {
        "run_id": run_id, "target_season": target_season, "target_gameweek": target_gameweek,
        "top_transfers": top_transfers, "chips": chips, "gw19_deadline": gw19,
    }
