"""A model-managed FPL team -- tracked live and scored against the field. The public proof
that the quant model works, for the app's Track Record page.

Unlike the two real tracked accounts (advice on a squad someone else built and may not
follow -- and which keep getting an un-scoreable "wildcard" recommendation), this is the
model's OWN team: its from-scratch GW1 optimal squad, then every gameweek its real
`transfer_planner.run()` / `backtest._decide_gameweek_action()` decision applied and the
realised points recorded. No human in the loop.

State lives in `data/model_team/state.json` -- a small committed JSON, same "the repo is the
cross-run memory" convention as `data/decision_log/`. Stateful and incremental: each pipeline
run advances exactly one gameweek from the stored squad (O(1), not a whole-season re-sim).

GW1-2 are backfilled by simulation on the first run -- each gameweek's decision is built
inside `backtest.asof_scope()` so nothing looks ahead, and the ledger row is flagged
`simulated: true`. Every gameweek from the first live run onward is `simulated: false`.

Scoring: realised XI points (`backtest._realized_xi_points`, real `event_points` ground
truth) once a gameweek is ingested, vs the FPL overall average that gameweek
(`bootstrap-static events[].average_entry_score`) and a frozen "never transfer / never chip"
baseline squad.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from . import backtest as bt
from . import forward_season_sim as fss
from . import transfer_planner

STATE_FILENAME = "state.json"
SEASON = "2026-2027"


# ------------------------------------------------------------------ state I/O

def load_state(state_dir: Path | str) -> dict | None:
    path = Path(state_dir) / STATE_FILENAME
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_state(state_dir: Path | str, state: dict) -> Path:
    d = Path(state_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / STATE_FILENAME
    path.write_text(json.dumps(state, indent=2))
    return path


# ------------------------------------------------------------------ squad helpers

def _names(con: duckdb.DuckDBPyConnection, uids: list[str]) -> dict[str, str]:
    if not uids:
        return {}
    ph = ",".join("?" * len(uids))
    return dict(con.execute(f"SELECT player_uid, canonical_name FROM dim_player WHERE player_uid IN ({ph})", uids).fetchall())


def seed_squad(con: duckdb.DuckDBPyConnection, season: str = SEASON) -> list[dict]:
    """The model's own from-scratch GW1 optimal 15, from the real GW1 squad_optimizer solve
    `run_ingestion.py` already produces (is_manager_snapshot=FALSE, target_gameweek=1). In
    `bootstrap_from_real_squad()`'s expected shape."""
    run = con.execute(
        "SELECT run_id FROM squad_optimizer_runs WHERE target_season = ? AND is_manager_snapshot = FALSE "
        "AND target_gameweek = 1 ORDER BY run_id DESC LIMIT 1",
        [season],
    ).fetchone()
    if not run:
        raise ValueError(f"no from-scratch GW1 squad_optimizer_runs row for {season} -- run scripts/run_ingestion.py first")
    rows = con.execute(
        "SELECT s.player_uid, s.in_xi, s.is_captain, s.is_vice, dp.canonical_name "
        "FROM squad_optimizer_selections s JOIN dim_player dp ON dp.player_uid = s.player_uid "
        "WHERE s.run_id = ? AND s.in_squad",
        [run[0]],
    ).fetchall()
    if len(rows) != 15:
        raise ValueError(f"GW1 solve run_id={run[0]} has {len(rows)} in_squad players, expected 15")
    return [
        {"player_name": name, "in_xi": bool(in_xi), "is_captain": bool(is_cap), "is_vice": bool(is_vice)}
        for _uid, in_xi, is_cap, is_vice, name in rows
    ]


def _resolve(con: duckdb.DuckDBPyConnection, names: list[str]) -> dict[str, str]:
    """canonical_name -> player_uid for the seed squad (names come straight from dim_player)."""
    if not names:
        return {}
    ph = ",".join("?" * len(names))
    return dict(con.execute(f"SELECT canonical_name, player_uid FROM dim_player WHERE canonical_name IN ({ph})", names).fetchall())


def _gw1_ledger_row(con: duckdb.DuckDBPyConnection, seed: list[dict], season: str, current_event: int) -> dict:
    by_name = _resolve(con, [p["player_name"] for p in seed])
    squad_uids, xi_uids, captain_uid = [], [], None
    for p in seed:
        uid = by_name.get(p["player_name"])
        if not uid:
            continue
        squad_uids.append(uid)
        if p["in_xi"]:
            xi_uids.append(uid)
        if p["is_captain"]:
            captain_uid = uid
    played = con.execute(
        "SELECT count(*) FROM fact_player_season_stats WHERE season = ? AND gw = 1 AND event_points IS NOT NULL",
        [season],
    ).fetchone()[0]
    realized = (
        round(bt._realized_xi_points(con, season, 1, frozenset(xi_uids), captain_uid), 1) if played else None
    )
    return {
        "gameweek": 1, "entry_label": "FPL Quant Model Team", "simulated": 1 < current_event,
        "action": "seed", "action_detail": "from-scratch GW1 optimal squad", "projected_points": None,
        "band_low": None, "band_high": None, "wildcard_gain": None, "wildcard_recommended": False,
        "free_hit_gain": None, "free_hit_recommended": False, "current_squad_horizon_value": None,
        "chips_used": [], "squad_uids": sorted(squad_uids), "xi_uids": sorted(xi_uids),
        "captain_uid": captain_uid, "realized_points": realized,
    }


def _carryforward_fields(row: dict) -> tuple[list[str], set[str], str | None]:
    """(squad_uids, xi_uids, captain_uid) of the squad this ledger row CARRIES FORWARD -- not
    necessarily the one it scored. Prefers the row's `carryforward_*` fields (the real persisted
    post-decision holdings that forward_season_sim now records every gameweek); on a Free Hit
    week those are the pre-chip 15, while `squad_uids`/`xi_uids`/`captain_uid` record the one-off
    fresh XI that was scored. Falls back to `squad_uids`/`xi_uids`/`captain_uid` for pre-
    carryforward ledger rows (identical on every non-Free-Hit row anyway)."""
    squad = row.get("carryforward_squad_uids") or row["squad_uids"]
    xi = set(row.get("carryforward_xi_uids") or row["xi_uids"])
    cap = row.get("carryforward_captain_uid") or row.get("captain_uid")
    return list(squad), xi, cap


def _carryforward_row(ledger: list[dict]) -> dict:
    """The ledger row whose squad the next `advance()` must bootstrap from, and that
    `build_summary()` shows as the current squad. Normally the last row. Exception: a legacy
    Free Hit row written before `carryforward_*` existed stored the one-off Free Hit 15 in
    `squad_uids` -- a real bug, since the squad must revert after a Free Hit week. For those,
    fall back to the row before it, which still holds the real pre-chip squad (Free Hit changes
    no holdings)."""
    ordered = sorted(ledger, key=lambda x: x["gameweek"])
    last = ordered[-1]
    legacy_free_hit = last.get("action") == "free_hit" and not last.get("carryforward_squad_uids")
    if legacy_free_hit and len(ordered) >= 2:
        return ordered[-2]
    return last


def _squad_from_ledger_row(con: duckdb.DuckDBPyConnection, row: dict) -> list[dict]:
    """Reconstruct the bootstrap-shaped squad this ledger row carries forward (see
    `_carryforward_fields`)."""
    squad_uids, xi, cap = _carryforward_fields(row)
    names = _names(con, squad_uids)
    # vice: the highest-order non-captain XI player isn't stored; pick any XI non-captain as
    # vice (only matters if the captain is auto-subbed -- realised scoring below uses FPL's own
    # entry_history for the real teams, and for the model team a missing-vice edge is rare and
    # never silently wrong: _realized_xi_points just doubles the captain's real points).
    vice = next((u for u in sorted(xi) if u != cap), None)
    return [
        {"player_name": names.get(u, u), "in_xi": u in xi, "is_captain": u == cap, "is_vice": u == vice}
        for u in squad_uids
    ]


# ------------------------------------------------------------------ the weekly advance

def advance(
    con: duckdb.DuckDBPyConnection,
    *,
    current_event: int,
    state_dir: Path | str,
    active_versions: dict,
    season: str = SEASON,
) -> dict:
    """Advance the model team's ledger to `current_event`. On the first call, seeds the GW1
    squad, scores GW1 directly, and walks GW2..current_event. On later calls, walks exactly
    the new gameweek(s) from the stored squad. Returns the updated state (also saved to disk).

    GW1 is scored directly rather than walked: a walk starting at GW1 would re-solve the
    candidate pool inside `asof_scope(season, 1)`, whose price cut-off (before any 2026-27
    match) leaves `fetch_candidate_pool` empty. The GW1 squad is the model's real from-scratch
    solve from ingestion; there is no transfer decision to make for the opening squad anyway."""
    state = load_state(state_dir)
    if state is None:
        state = {"season": season, "current_gameweek": 0, "ledger": [], "chips_used_set1": [], "chips_used_set2": []}

    if not state["ledger"]:
        seed = seed_squad(con, season)
        state["ledger"].append(_gw1_ledger_row(con, seed, season, current_event))
        state["current_gameweek"] = 1

    from_gw = max(state["current_gameweek"] + 1, 2)
    if from_gw > current_event:
        state["current_gameweek"] = current_event
        save_state(state_dir, state)
        return state

    bootstrap = _squad_from_ledger_row(con, _carryforward_row(state["ledger"]))

    result = fss.run_forward_season_sim(
        con,
        entry_label="FPL Quant Model Team",
        target_season=season,
        start_gameweek=from_gw,
        end_gameweek=current_event,
        bootstrap_squad=bootstrap,
        active_versions=active_versions,
        real_chips_used_set1=state["chips_used_set1"],
        real_chips_used_set2=state["chips_used_set2"],
        score_realized=True,
    )

    for r in result.rows:
        row = r.to_dict()
        row["entry_label"] = "FPL Quant Model Team"
        # A row walked for a gameweek that had already been played (we were catching up) is a
        # simulated backfill; a row walked at the gameweek that just became current is a live,
        # pre-deadline decision.
        row["simulated"] = r.gameweek < current_event
        state["ledger"].append(row)

    for r in result.rows:
        set1_deadline = transfer_planner.GW19_DEADLINE_GAMEWEEK
        for chip in r.chips_used:
            bucket = "chips_used_set1" if r.gameweek < set1_deadline else "chips_used_set2"
            if chip not in state[bucket]:
                state[bucket] = sorted([*state[bucket], chip])
    state["current_gameweek"] = current_event
    save_state(state_dir, state)
    return state


# ------------------------------------------------------------------ realize past gameweeks

def realize(
    con: duckdb.DuckDBPyConnection, state_dir: Path | str, season: str = SEASON,
    finished_gameweeks: set[int] | None = None,
) -> dict:
    """Score any ledger row whose gameweek has been ingested (`event_points` populated).

    `finished_gameweeks` is the set of gameweeks the FPL API reports as `finished`. When given,
    a row whose gameweek is still in progress is scored to a *provisional* figure and left
    unlocked, so a mid-match partial score (e.g. only the Friday game played) is corrected on
    the next run rather than frozen forever; a row is locked (`realized_final`) only once its
    gameweek is finished. `None` (the default, and every existing caller/test) keeps the old
    behaviour -- score once on first ingestion and lock immediately."""
    state = load_state(state_dir)
    if state is None:
        return {"realized": 0}
    n = 0
    dirty = False
    for row in state["ledger"]:
        # `realized_final` locks a row once its gameweek has finished -- re-score everything else
        # (a row never scored, or scored to a provisional figure off an in-progress gameweek).
        if not row.get("xi_uids") or row.get("realized_final"):
            continue
        gw = row["gameweek"]
        is_final = finished_gameweeks is None or gw in finished_gameweeks
        played = con.execute(
            "SELECT count(*) FROM fact_player_season_stats WHERE season = ? AND gw = ? AND event_points IS NOT NULL",
            [season, gw],
        ).fetchone()[0]
        if not played:
            continue
        mult = 3 if row.get("action") == "triple_captain" else 2
        new_points = round(
            bt._realized_xi_points(con, season, gw, frozenset(row["xi_uids"]), row.get("captain_uid"),
                                   captain_multiplier=mult),
            1,
        )
        if new_points != row.get("realized_points"):
            n += 1
            dirty = True
        if bool(row.get("realized_final")) != is_final:
            dirty = True
        row["realized_points"] = new_points
        row["realized_final"] = is_final
    if dirty:
        save_state(state_dir, state)
    return {"realized": n}


# ------------------------------------------------------------------ public summary

def build_summary(con: duckdb.DuckDBPyConnection, state_dir: Path | str, field_average_by_gw: dict[int, float]) -> dict:
    """`data/dashboard/app_model_team.json` -- the Track Record page's headline panel."""
    state = load_state(state_dir)
    if state is None or not state["ledger"]:
        return {"ready": False, "reason": "model team not seeded yet"}

    ordered = sorted(state["ledger"], key=lambda x: x["gameweek"])

    def _is_final(r: dict, fa: float | None) -> bool:
        """A gameweek is 'final' (safe to fold into the cumulative headline) once its FPL
        overall average is known -- run_model_team.py only puts a gameweek in
        `field_average_by_gw` when bootstrap-static reports it `finished`. A backfilled
        GW1-2 row is always final. A row scored off a still-in-progress gameweek (realised
        points present, no field average yet) is `provisional`: shown in its own week row,
        but kept out of totals and the vs-field headline so a half-played gameweek can't read
        as a swing against the field."""
        return r.get("simulated", False) or bool(r.get("realized_final")) or fa is not None

    weeks = []
    cum_model = cum_field = 0.0
    n_final_scored = 0
    for r in ordered:
        rp = r.get("realized_points")
        fa = field_average_by_gw.get(r["gameweek"])
        final = _is_final(r, fa)
        provisional = rp is not None and not final
        if rp is not None and final:
            n_final_scored += 1
            cum_model += rp
            if fa is not None:
                cum_field += fa
        weeks.append({
            "gameweek": r["gameweek"], "simulated": r.get("simulated", False),
            "action": r["action"], "action_detail": r.get("action_detail", ""),
            "projected_points": r["projected_points"], "realized_points": rp,
            "provisional": provisional,
            "field_average": fa,
            "delta_vs_field": None if (rp is None or fa is None) else round(rp - fa, 1),
            "cumulative_points": round(cum_model, 1) if (rp is not None and final) else None,
            "cumulative_vs_field": round(cum_model - cum_field, 1) if (rp is not None and final and cum_field) else None,
        })

    carry = _carryforward_row(ordered)
    cf_squad, cf_xi, cf_cap = _carryforward_fields(carry)
    names = _names(con, cf_squad)
    latest = ordered[-1]
    return {
        "ready": True,
        "season": state["season"],
        "current_gameweek": state["current_gameweek"],
        "n_gameweeks_scored": n_final_scored,
        "n_gameweeks_simulated": sum(1 for r in state["ledger"] if r.get("simulated")),
        "total_realized_points": round(cum_model, 1),
        "total_vs_field": round(cum_model - cum_field, 1) if cum_field else None,
        "chips_used": sorted(set(state["chips_used_set1"]) | set(state["chips_used_set2"])),
        "weeks": weeks,
        "free_hit_audit": _free_hit_audit(con, ordered, state["season"], field_average_by_gw),
        "current_squad": [
            {"name": names.get(u, u), "in_xi": u in cf_xi, "is_captain": u == cf_cap}
            for u in cf_squad
        ],
        "next_decision": {
            "gameweek": latest["gameweek"], "action": latest["action"], "detail": latest.get("action_detail", ""),
        },
        "disclaimer": (
            "GW1-2 backfilled by simulation with a per-gameweek data cutoff (nothing looks "
            "ahead); every gameweek since is a live, pre-deadline decision committed to the "
            "repo before kickoff. Realised FPL points; the FPL overall average is the field "
            "benchmark. A gameweek still in progress is shown as provisional and kept out of "
            "the cumulative totals until it is final."
        ),
    }


def _free_hit_audit(
    con: duckdb.DuckDBPyConnection, ordered_ledger: list[dict], season: str, field_average_by_gw: dict[int, float],
) -> dict | None:
    """A transparent read of the most recent Free Hit decision: the projected gain the model
    saw, the threshold it had to clear, the projected score of the one-off Free Hit XI, its
    realised score, and -- the counterfactual that actually answers "was the chip worth it" --
    what the pre-Free-Hit long-term XI would have scored the same gameweek if the chip had been
    held. None when the model team has never played Free Hit."""
    fh_rows = [r for r in ordered_ledger if r.get("action") == "free_hit"]
    if not fh_rows:
        return None
    fhr = fh_rows[-1]
    gw = fhr["gameweek"]
    prev = next((r for r in reversed(ordered_ledger) if r["gameweek"] < gw), None)
    hold_xi = fhr.get("carryforward_xi_uids") or (prev["xi_uids"] if prev else None)
    hold_cap = fhr.get("carryforward_captain_uid") or (prev.get("captain_uid") if prev else None)

    counterfactual_hold = None
    if hold_xi:
        played = con.execute(
            "SELECT count(*) FROM fact_player_season_stats WHERE season = ? AND gw = ? AND event_points IS NOT NULL",
            [season, gw],
        ).fetchone()[0]
        if played:
            counterfactual_hold = round(
                bt._realized_xi_points(con, season, gw, frozenset(hold_xi), hold_cap), 1
            )

    realized = fhr.get("realized_points")
    final = field_average_by_gw.get(gw) is not None
    return {
        "gameweek": gw,
        "projected_gain_vs_current_xi": fhr.get("free_hit_gain"),
        "gain_threshold": fhr.get("free_hit_threshold"),
        "recommended": fhr.get("free_hit_recommended"),
        "projected_points": fhr.get("projected_points"),
        "realized_points": realized,
        "counterfactual_hold_realized_points": counterfactual_hold,
        "realized_minus_hold": (
            None if (realized is None or counterfactual_hold is None)
            else round(realized - counterfactual_hold, 1)
        ),
        "gameweek_final": final,
        "note": (
            "gameweek final" if final
            else "gameweek still in progress -- realised and counterfactual figures are provisional"
        ),
    }
