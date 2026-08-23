"""Roadmap Feature 2: "Rate My Team" -- a real manager's squad graded against the
mathematically-optimal squad SCIP finds, backed by squad_optimizer's real MIQP solve (not a
heuristic). Built entirely on transfer_planner's own established machinery
(squad_optimizer.run() + transfer_planner._horizon_ep_by_player(), the same pair
evaluate_wildcard() already uses to compare a fresh optimal squad against a manager's
current one) -- this module adds no new solve mode, only the grading/swap-finding layer on
top.

Real, disclosed simplification: grades against the FIXED 100.0 budget squad_optimizer.solve()
always uses (squad_optimizer.BUDGET), not the manager's own current total squad value, which
can differ slightly from 100.0 due to price rises/falls since the season started -- the same
budget basis Wildcard/Free Hit evaluation elsewhere in this project already grades against.
"""

from dataclasses import dataclass
from datetime import date

import duckdb

from . import squad_optimizer as so_mod
from . import transfer_planner as tp

# points_gap < bound -> that grade; points_gap >= the last bound -> "D". Tuned to a plausible
# real-world spread (a few points is a strong squad, 10+ is a real, visible underperformance)
# -- an invented v1 default like every other unpinned magnitude in this project, flagged for
# the same eventual recalibration-against-real-backtest-distribution treatment.
GRADE_BANDS = [(2.0, "A"), (5.0, "B"), (10.0, "C")]


class SquadGradeInvariantError(Exception):
    """points_gap came out negative -- the user's squad scored MORE than the proven-optimal
    solve at the same fixed budget, which should be structurally impossible unless the
    user's real squad value exceeds that budget (price rises since the season started).
    Raised rather than silently clipped to 0, so this gets investigated, not hidden."""


@dataclass(frozen=True)
class Swap:
    out_player_uid: str
    in_player_uid: str
    delta_ep: float
    reason: str


@dataclass(frozen=True)
class Provenance:
    model_version: str
    data_asof: str


@dataclass(frozen=True)
class SquadGrade:
    entry_id: int
    gw: int
    grade: str
    points_gap: float
    optimal_ep: float
    user_squad_ep: float
    top_swaps: list[Swap]
    provenance: Provenance


def _letter_grade(points_gap: float) -> str:
    for bound, letter in GRADE_BANDS:
        if points_gap < bound:
            return letter
    return "D"


def _find_top_swaps(user_uids: set[str], optimal_uids: set[str], horizon_ep_map: dict, n: int = 3) -> list[Swap]:
    """Position-for-position swaps only (a real FPL transfer can't change a squad's position
    quotas) -- greedily picks the n highest-delta_ep swaps with no player reused across more
    than one suggested swap. reason is a real, computed comparison (never a fixture-swing/
    ownership causal claim this module has no data to actually back): 'price-density' when
    the incoming player is cheaper AND scores more, 'higher projected points' otherwise."""
    only_user = user_uids - optimal_uids
    only_optimal = optimal_uids - user_uids
    empty = {"total_ep": 0.0, "position": None, "price": None}

    candidates = []
    for out_uid in only_user:
        out_info = horizon_ep_map.get(out_uid, empty)
        for in_uid in only_optimal:
            in_info = horizon_ep_map.get(in_uid, empty)
            if in_info["position"] != out_info["position"] or in_info["position"] is None:
                continue
            delta = in_info["total_ep"] - out_info["total_ep"]
            if delta <= 0:
                continue
            reason = "price-density" if (in_info["price"] or 0.0) < (out_info["price"] or 0.0) else "higher projected points"
            candidates.append(Swap(out_player_uid=out_uid, in_player_uid=in_uid, delta_ep=delta, reason=reason))
    candidates.sort(key=lambda s: -s.delta_ep)

    top: list[Swap] = []
    used_out: set[str] = set()
    used_in: set[str] = set()
    for s in candidates:
        if s.out_player_uid in used_out or s.in_player_uid in used_in:
            continue
        top.append(s)
        used_out.add(s.out_player_uid)
        used_in.add(s.in_player_uid)
        if len(top) >= n:
            break
    return top


def grade_squad(
    con: duckdb.DuckDBPyConnection,
    entry_id: int,
    calibration_asof_date: date,
    target_season: str,
    target_gameweek: int,
    current_holdings: list[dict],
    horizon_ep_versions: dict[int, tuple[int, int]],
    *,
    lambda_params_version: int,
    guardrail_params_version: int,
) -> SquadGrade:
    """current_holdings: [{"player_uid": ...}, ...] -- a real manager's holdings (e.g. from
    transfer_planner._read_holdings(state_version), the same shape backtest.py's own
    walk-forward loop already reads). horizon_ep_versions: {gw: (ep_model_version,
    uncertainty_model_version)}, e.g. from transfer_planner.compute_horizon_ep() -- reused
    unchanged, never recomputed here.
    """
    if target_gameweek not in horizon_ep_versions:
        raise ValueError(f"no fixtures for {target_season} GW{target_gameweek} -- cannot grade")
    ep_mv, un_mv = horizon_ep_versions[target_gameweek]

    optimal_run_id = so_mod.run(
        con, calibration_asof_date, target_season, target_gameweek, ep_mv, un_mv,
        lambda_params_version, guardrail_params_version,
    )
    optimal_uids = {
        r[0] for r in con.execute(
            "SELECT player_uid FROM squad_optimizer_selections WHERE run_id = ? AND in_squad", [optimal_run_id]
        ).fetchall()
    }

    horizon_ep_map = tp._horizon_ep_by_player(con, target_season, horizon_ep_versions)
    user_uids = {h["player_uid"] for h in current_holdings}
    optimal_ep = sum(horizon_ep_map.get(uid, {}).get("total_ep", 0.0) for uid in optimal_uids)
    user_squad_ep = sum(horizon_ep_map.get(uid, {}).get("total_ep", 0.0) for uid in user_uids)

    points_gap = optimal_ep - user_squad_ep
    if points_gap < 0:
        raise SquadGradeInvariantError(
            f"points_gap={points_gap:.2f} < 0 for entry_id={entry_id}: the user's squad scored "
            f"more than the proven-optimal solve at the standard {so_mod.BUDGET} budget -- their "
            f"real squad value most likely exceeds that budget (price rises since the season started)."
        )

    top_swaps = _find_top_swaps(user_uids, optimal_uids, horizon_ep_map, n=3)

    return SquadGrade(
        entry_id=entry_id, gw=target_gameweek, grade=_letter_grade(points_gap),
        points_gap=points_gap, optimal_ep=optimal_ep, user_squad_ep=user_squad_ep,
        top_swaps=top_swaps,
        provenance=Provenance(
            model_version=f"ep_v{ep_mv}/un_v{un_mv}/miqp_run{optimal_run_id}",
            data_asof=calibration_asof_date.isoformat(),
        ),
    )
