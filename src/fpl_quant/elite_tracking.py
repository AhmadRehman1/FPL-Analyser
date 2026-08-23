"""Roadmap Feature 9: elite-manager tracking, done transparently. For each configured elite
manager's latest real transfer, shows the model's own recommendation for that manager's
squad vs what they actually did, plus a divergence reason computed from real, available
signals -- never presented as a recommendation by default (correlation with what a top-10k
manager did is not reasoning; showing the divergence and a real reason lets the user decide).

Reuses decision_engine.recommend_best_move() for "what would the model have done" (the same
engine Feature 3 already uses for a live manager, not a second, parallel mechanism) and
transfer_planner.price_momentum_by_player() for the divergence reason -- both already real,
tested signals elsewhere in this project, not invented here. Player identity is resolved via
ingest_workbook._resolve_player() -- the SAME normalized-name matching every other real-name
source in this project already uses, not a separately-invented rule -- so an elite manager's
actual transfer is comparable to the model's own player_uid-keyed recommendation at all.
"""

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb

from . import decision_engine as de
from . import ingest_workbook as iw
from . import transfer_planner as tp
from .errors import MissingModelVersionError


@dataclass(frozen=True)
class ActualMove:
    players_out: list[int]
    players_in: list[int]


def load_elite_managers(path: Path) -> list[dict]:
    """[{"entry_id": int, "name": str}, ...] from a committed, configurable JSON file --
    never a hardcoded list in code. Missing file -> [] (not an error): a project that hasn't
    configured any elite managers yet has nothing to track, not a broken one."""
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return data.get("managers", [])


def compute_actual_move(previous_picks: list[dict], current_picks: list[dict]) -> ActualMove:
    """picks: FPL API's own raw pick dicts (must carry "element", the numeric FPL element id --
    NOT this project's own player_uid). The set difference between two consecutive gameweeks'
    full 15-man squads -- empty lists (no transfers made) is a real, valid result, not an
    error. Kept as pure element-id set logic (no DB access) so it's directly unit-testable;
    resolving element ids to this project's own player_uid is the caller's job (build_elite_
    divergence() below), since it needs a DB connection and a season to do so."""
    prev_uids = {p["element"] for p in previous_picks}
    curr_uids = {p["element"] for p in current_picks}
    return ActualMove(players_out=sorted(prev_uids - curr_uids), players_in=sorted(curr_uids - prev_uids))


def _format_move(players_out: list[str], players_in: list[str]) -> str:
    if not players_out and not players_in:
        return "roll"
    return f"transfer_in:{','.join(players_out)}->{','.join(players_in)}"


def _divergence_reason(
    con: duckdb.DuckDBPyConnection, target_season: str, target_gameweek: int,
    actual_in_uid: str | None, model_in_uid: str | None,
) -> str | None:
    """A real, computed reason from available signals -- never a fabricated psychological
    claim ("a hunch", "a different risk posture") this project has no data to back. None when
    no real signal explains the divergence from what's actually available."""
    if actual_in_uid is None or actual_in_uid == model_in_uid:
        return None
    momentum = tp.price_momentum_by_player(con, target_season, target_gameweek, lookback_gameweeks=3)
    actual_momentum = momentum.get(actual_in_uid, {})
    if (actual_momentum.get("price_delta") or 0) > 0:
        return "chased a recent price rise"
    if (actual_momentum.get("ownership_delta") or 0) > 0:
        return "followed rising ownership (template pick)"
    return "diverged from the model's own ranking -- no price/ownership signal explains it from available data"


def build_elite_divergence(
    con: duckdb.DuckDBPyConnection,
    elite_managers: list[dict],
    calibration_asof_date: date,
    target_season: str,
    current_event: int,
    fetch_picks_fn,
    element_names: dict[int, str],
    run_kwargs: dict,
) -> list[dict]:
    """fetch_picks_fn(entry_id, event) -> list[dict] | None -- injected (not imported
    directly) so this stays testable offline, matching this project's established fetch-
    isolation convention. element_names: FPL element id -> full name, e.g. from
    ingest_fpl_entry_picks.fetch_bootstrap_elements() -- the SAME bootstrap-static snapshot
    for every manager at a point in time, so the caller fetches it once and passes it down
    rather than this function re-fetching it per manager. run_kwargs: every
    *_params_version/ts_model_version/mm_model_version decision_engine.recommend_best_move()
    needs, minus entry_id/target_gameweek/input_state_version (supplied per-manager here).

    An unknown/unreachable entry_id, or one whose picks reference a player name that fails to
    resolve, is skipped with a printed warning, never crashes the whole run -- one elite
    manager's bad data shouldn't blank the whole tracker. Raises MissingModelVersionError up
    front (not per-manager) if current_event itself has no fixtures -- every manager would
    fail identically, so that's a real, shared precondition failure, not a per-manager skip.
    """
    ts_mv, mm_mv = run_kwargs["ts_model_version"], run_kwargs["mm_model_version"]
    horizon_versions = tp.compute_horizon_ep(
        con, calibration_asof_date, target_season, current_event, ts_mv, mm_mv, 1,
        run_kwargs["scoring_params_version"], run_kwargs["bps_params_version"], run_kwargs["tau_params_version"],
        run_kwargs["rho_residual_params_version"], run_kwargs["corr_params_version"],
    )
    if current_event not in horizon_versions:
        raise MissingModelVersionError(f"no fixtures for {target_season} GW{current_event} -- cannot track elite divergence")
    ep_mv, un_mv = horizon_versions[current_event]

    de_kwargs = {k: v for k, v in run_kwargs.items() if k not in ("ts_model_version", "mm_model_version")}

    out = []
    for manager in elite_managers:
        entry_id, name = manager["entry_id"], manager.get("name", str(manager["entry_id"]))
        try:
            previous_picks = fetch_picks_fn(entry_id, current_event - 1)
            current_picks = fetch_picks_fn(entry_id, current_event)
            if previous_picks is None or current_picks is None:
                print(f"::warning::elite_tracking: entry_id={entry_id} ({name}) has no picks for "
                      f"GW{current_event - 1} or GW{current_event} -- skipped.")
                continue

            move = compute_actual_move(previous_picks, current_picks)

            def _resolve(element_id: int) -> str | None:
                return iw._resolve_player(con, element_names.get(element_id), target_season)

            actual_out_uids = {u for u in (_resolve(e) for e in move.players_out) if u is not None}
            actual_in_uids = {u for u in (_resolve(e) for e in move.players_in) if u is not None}
            actual_move_str = _format_move(sorted(actual_out_uids), sorted(actual_in_uids))

            squad = [
                {"player_name": element_names.get(p["element"]), "in_xi": p["position"] <= 11,
                 "is_captain": bool(p.get("is_captain")), "is_vice": bool(p.get("is_vice_captain"))}
                for p in previous_picks
            ]
            state_version = tp.bootstrap_from_real_squad(
                con, calibration_asof_date, target_season, current_event - 1, ep_mv, un_mv, squad,
            )
            decision = de.recommend_best_move(
                con, entry_id, calibration_asof_date, target_season, current_event, state_version,
                ts_mv, mm_mv, include_sensitivity=False, **de_kwargs,
            )

            model_in_uids = {s.in_player_uid for s in decision.swaps}
            diverged = actual_in_uids != model_in_uids
            actual_in_uid = next(iter(actual_in_uids)) if len(actual_in_uids) == 1 else None
            model_in_uid = decision.swaps[0].in_player_uid if decision.swaps else None
            reason = _divergence_reason(con, target_season, current_event, actual_in_uid, model_in_uid) if diverged else None

            out.append({
                "entry_id": entry_id, "name": name, "actual_move": actual_move_str,
                "model_move": decision.action, "diverged": diverged, "divergence_reason": reason,
                "provenance": {
                    "model_version": decision.provenance.model_version, "data_asof": decision.provenance.data_asof,
                },
            })
        except Exception as e:  # noqa: BLE001 -- one bad elite manager must never blank the whole run
            print(f"::warning::elite_tracking: entry_id={entry_id} ({name}) failed ({e}) -- skipped.")
            continue
    return out
