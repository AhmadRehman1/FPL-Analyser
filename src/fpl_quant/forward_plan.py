"""The model's forward plan for one squad -- its whole thinking from the next unplayed
gameweek through a horizon end (GW18 by default), reshaped for the app.

`forward_season_sim.run_forward_season_sim()` in `model_choice` mode already walks the real
production planner one decision per gameweek (the same greedy arm `chip_timing_analysis`
runs). This module wraps that walk and turns its `ForwardSimResult` into a display payload:
per-week action / transfer(s) / captain / projected points, the evolving 15, and -- pulled
out on its own -- the Wildcard squad at the week the model plays it.

Player rows use the dashboard's `preview_squad` shape (`player_name` / `club`, not
`player_uid`) so the frontend resolves them exactly as it already does for the chip preview
squads -- see `run_transfer_planner_for_real_squad._build_chip_preview_squad`. Nothing new is
modelled here; this is a read + reshape of a planner walk.
"""

from __future__ import annotations

from datetime import datetime, timezone

import duckdb

from . import fixture_swing
from . import forward_season_sim as fss

HORIZON_END_GAMEWEEK = 18


def _name_and_club_maps(
    con: duckdb.DuckDBPyConnection, target_season: str, uids: set[str]
) -> tuple[dict[str, str], dict[str, str | None]]:
    """(player_uid -> canonical_name, player_uid -> club canonical_name) for `uids`."""
    if not uids:
        return {}, {}
    uid_list = sorted(uids)
    ph = ",".join("?" * len(uid_list))
    name_by_uid = dict(
        con.execute(
            f"SELECT player_uid, canonical_name FROM dim_player WHERE player_uid IN ({ph})", uid_list
        ).fetchall()
    )
    team_uid_by_player = fixture_swing.team_uid_by_player(con, target_season)
    team_names = {
        r[0]: r[1] for r in con.execute("SELECT team_uid, canonical_name FROM dim_team").fetchall()
    }
    club_by_uid = {u: team_names.get(team_uid_by_player.get(u)) for u in uid_list}
    return name_by_uid, club_by_uid


def _squad_rows(
    squad_uids: list[str],
    xi_uids: list[str],
    captain_uid: str | None,
    name_by_uid: dict[str, str],
    club_by_uid: dict[str, str | None],
) -> list[dict]:
    """The `preview_squad` shape: {player_name, club, in_xi, is_captain, is_vice}. Vice is the
    first XI non-captain -- only cosmetic here (the plan never auto-subs)."""
    xi = set(xi_uids)
    vice = next((u for u in xi_uids if u != captain_uid), None)
    return [
        {
            "player_name": name_by_uid.get(u, u),
            "club": club_by_uid.get(u),
            "in_xi": u in xi,
            "is_captain": u == captain_uid,
            "is_vice": u == vice,
        }
        for u in squad_uids
    ]


def _transfer_rows(transfers: list[dict], name_by_uid: dict[str, str]) -> list[dict]:
    return [
        {
            "out": name_by_uid.get(t["out_uid"], t["out_uid"]),
            "in": name_by_uid.get(t["in_uid"], t["in_uid"]),
            "net": t.get("net"),
        }
        for t in transfers
    ]


def _week_summary(action: str, transfer_rows: list[dict]) -> str:
    if action == "wildcard":
        return "Wildcard — full rebuild"
    if action == "free_hit":
        return "Free Hit — one-week squad"
    if action == "bench_boost":
        return "Bench Boost"
    if action == "triple_captain":
        return "Triple Captain"
    if action == "transfer" and transfer_rows:
        return "  ·  ".join(f"{t['out']} → {t['in']}" for t in transfer_rows)
    return "Hold — no transfer"


def build_forward_plan(
    con: duckdb.DuckDBPyConnection,
    *,
    entity_key: str,
    entry_label: str,
    entry_id: int | None,
    target_season: str,
    bootstrap_squad: list[dict],
    active_versions: dict,
    start_gameweek: int,
    end_gameweek: int = HORIZON_END_GAMEWEEK,
    real_chips_used_set1: list[str] | None = None,
    real_chips_used_set2: list[str] | None = None,
    base_gameweek: int | None = None,
) -> dict:
    """Walk `start_gameweek..end_gameweek` in model_choice mode and return the display payload.

    `base_gameweek` is the last already-played gameweek (for the app's "as of" label); it
    defaults to `start_gameweek - 1`.
    """
    result = fss.run_forward_season_sim(
        con,
        entry_label=entry_label,
        target_season=target_season,
        start_gameweek=start_gameweek,
        end_gameweek=end_gameweek,
        bootstrap_squad=bootstrap_squad,
        active_versions=active_versions,
        real_chips_used_set1=real_chips_used_set1,
        real_chips_used_set2=real_chips_used_set2,
        score_realized=True,
    )

    uids: set[str] = set()
    for r in result.rows:
        uids.update(r.squad_uids)
        uids.update(r.formation_xi_uids or r.xi_uids)
        if r.captain_uid:
            uids.add(r.captain_uid)
        for t in r.transfers:
            uids.add(t["out_uid"])
            uids.add(t["in_uid"])
    name_by_uid, club_by_uid = _name_and_club_maps(con, target_season, uids)

    weeks: list[dict] = []
    chips_planned: list[dict] = []
    wildcard_block: dict | None = None
    free_hit_block: dict | None = None
    for r in result.rows:
        transfer_rows = _transfer_rows(r.transfers, name_by_uid)
        squad_rows = _squad_rows(
            r.squad_uids, r.formation_xi_uids or r.xi_uids, r.captain_uid, name_by_uid, club_by_uid
        )
        week = {
            "gameweek": r.gameweek,
            "action": r.action,
            "summary": _week_summary(r.action, transfer_rows),
            "transfers": transfer_rows,
            "chip": r.action if r.action in ("wildcard", "free_hit", "bench_boost", "triple_captain") else None,
            "captain": name_by_uid.get(r.captain_uid, r.captain_uid) if r.captain_uid else None,
            "projected_points": round(r.projected_points, 1),
            "band": [round(r.band_low, 1), round(r.band_high, 1)],
            "wildcard_gain": None if r.wildcard_gain is None else round(r.wildcard_gain, 1),
            "wildcard_recommended": r.wildcard_recommended,
            "squad": squad_rows,
        }
        weeks.append(week)

        if week["chip"]:
            chips_planned.append({"chip": week["chip"], "gameweek": r.gameweek})
        if r.action == "wildcard" and wildcard_block is None:
            wildcard_block = {
                "gameweek": r.gameweek,
                "projected_gain": None if r.wildcard_gain is None else round(r.wildcard_gain, 1),
                "projected_points": round(r.projected_points, 1),
                "captain": week["captain"],
                "squad": squad_rows,
            }
        if r.action == "free_hit" and free_hit_block is None:
            free_hit_block = {
                "gameweek": r.gameweek,
                "projected_gain": None if r.free_hit_gain is None else round(r.free_hit_gain, 1),
                "squad": squad_rows,
            }

    # If the model never plays the Wildcard in-window, still surface the best week it *would*
    # (its own recommendation) so the app can show "holding until GWn" -- but with no squad,
    # because a not-played Wildcard was never solved to a 15 in this walk.
    held_wildcard = None
    if wildcard_block is None:
        reco = result.wildcard_recommendation
        if reco is not None:
            held_wildcard = {"gameweek": reco["gameweek"], "projected_gain": reco["projected_gain"]}

    return {
        "entity_key": entity_key,
        "entry_id": entry_id,
        "label": entry_label,
        "target_season": target_season,
        "base_gameweek": base_gameweek if base_gameweek is not None else start_gameweek - 1,
        "start_gameweek": start_gameweek,
        "end_gameweek": end_gameweek,
        "total_projected_points": round(result.total_projected_points, 1),
        "band": [round(result.total_band_low, 1), round(result.total_band_high, 1)],
        "chips_planned": chips_planned,
        "wildcard": wildcard_block,
        "wildcard_held_until": held_wildcard,
        "free_hit": free_hit_block,
        "weeks": weeks,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
