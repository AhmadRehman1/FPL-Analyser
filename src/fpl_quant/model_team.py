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


def _squad_from_ledger_row(con: duckdb.DuckDBPyConnection, row: dict) -> list[dict]:
    """Reconstruct the bootstrap-shaped squad from a ledger row's stored uids."""
    names = _names(con, row["squad_uids"])
    xi, cap = set(row["xi_uids"]), row.get("captain_uid")
    # vice: the highest-order non-captain XI player isn't stored; pick any XI non-captain as
    # vice (only matters if the captain is auto-subbed -- realised scoring below uses FPL's own
    # entry_history for the real teams, and for the model team a missing-vice edge is rare and
    # never silently wrong: _realized_xi_points just doubles the captain's real points).
    vice = next((u for u in row["xi_uids"] if u != cap), None)
    return [
        {"player_name": names.get(u, u), "in_xi": u in xi, "is_captain": u == cap, "is_vice": u == vice}
        for u in row["squad_uids"]
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
    squad and backfills GW1..current_event by simulation. On later calls, walks exactly the
    new gameweek(s) from the stored squad. Returns the updated state (also saved to disk)."""
    state = load_state(state_dir)
    if state is None:
        state = {"season": season, "current_gameweek": 0, "ledger": [], "chips_used_set1": [], "chips_used_set2": []}

    from_gw = state["current_gameweek"] + 1
    if from_gw > current_event:
        return state  # already current

    bootstrap = seed_squad(con, season) if not state["ledger"] else _squad_from_ledger_row(con, state["ledger"][-1])

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

def realize(con: duckdb.DuckDBPyConnection, state_dir: Path | str, season: str = SEASON) -> dict:
    """Re-score any ledger row whose gameweek was NOT yet ingested when it was walked
    (realized_points is None) but now is. Idempotent."""
    state = load_state(state_dir)
    if state is None:
        return {"realized": 0}
    n = 0
    for row in state["ledger"]:
        if row.get("realized_points") is not None or not row.get("xi_uids"):
            continue
        played = con.execute(
            "SELECT count(*) FROM fact_player_season_stats WHERE season = ? AND gw = ? AND event_points IS NOT NULL",
            [season, row["gameweek"]],
        ).fetchone()[0]
        if not played:
            continue
        mult = 3 if row.get("action") == "triple_captain" else 2
        row["realized_points"] = round(
            bt._realized_xi_points(con, season, row["gameweek"], frozenset(row["xi_uids"]), row.get("captain_uid"),
                                   captain_multiplier=mult),
            1,
        )
        n += 1
    if n:
        save_state(state_dir, state)
    return {"realized": n}


# ------------------------------------------------------------------ public summary

def build_summary(con: duckdb.DuckDBPyConnection, state_dir: Path | str, field_average_by_gw: dict[int, float]) -> dict:
    """`data/dashboard/app_model_team.json` -- the Track Record page's headline panel."""
    state = load_state(state_dir)
    if state is None or not state["ledger"]:
        return {"ready": False, "reason": "model team not seeded yet"}

    scored = [r for r in state["ledger"] if r.get("realized_points") is not None]
    weeks = []
    cum_model = cum_field = 0.0
    for r in sorted(state["ledger"], key=lambda x: x["gameweek"]):
        rp = r.get("realized_points")
        fa = field_average_by_gw.get(r["gameweek"])
        if rp is not None:
            cum_model += rp
            if fa is not None:
                cum_field += fa
        weeks.append({
            "gameweek": r["gameweek"], "simulated": r.get("simulated", False),
            "action": r["action"], "action_detail": r.get("action_detail", ""),
            "projected_points": r["projected_points"], "realized_points": rp,
            "field_average": fa,
            "delta_vs_field": None if (rp is None or fa is None) else round(rp - fa, 1),
            "cumulative_points": round(cum_model, 1) if rp is not None else None,
            "cumulative_vs_field": round(cum_model - cum_field, 1) if (rp is not None and cum_field) else None,
        })

    latest = sorted(state["ledger"], key=lambda x: x["gameweek"])[-1]
    names = _names(con, latest["squad_uids"])
    return {
        "ready": True,
        "season": state["season"],
        "current_gameweek": state["current_gameweek"],
        "n_gameweeks_scored": len(scored),
        "n_gameweeks_simulated": sum(1 for r in state["ledger"] if r.get("simulated")),
        "total_realized_points": round(cum_model, 1),
        "total_vs_field": round(cum_model - cum_field, 1) if cum_field else None,
        "chips_used": sorted(set(state["chips_used_set1"]) | set(state["chips_used_set2"])),
        "weeks": weeks,
        "current_squad": [
            {"name": names.get(u, u), "in_xi": u in set(latest["xi_uids"]), "is_captain": u == latest.get("captain_uid")}
            for u in latest["squad_uids"]
        ],
        "next_decision": {
            "gameweek": latest["gameweek"], "action": latest["action"], "detail": latest.get("action_detail", ""),
        },
        "disclaimer": (
            "GW1-2 backfilled by simulation with a per-gameweek data cutoff (nothing looks "
            "ahead); every gameweek since is a live, pre-deadline decision committed to the "
            "repo before kickoff. Realised FPL points; the FPL overall average is the field "
            "benchmark."
        ),
    }
