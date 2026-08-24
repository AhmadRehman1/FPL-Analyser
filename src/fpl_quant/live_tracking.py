"""Final-third roadmap item 1: real, in-match tracking -- provisional bonus points and a live
rank estimate while fixtures are actually being played. Same real-live-API-only ethos as
app_export.py (see its own module docstring): everything here is either a direct read of FPL's
live data or a faithful reimplementation of FPL's own PUBLISHED bonus-points rule, never a
guess. The one genuinely estimated number (live rank) is always returned with its sample size
attached and is never to be presented as an official FPL figure.

Deliberately a separate module from app_export.py: this one only matters during a live match
window (scripts/export_live_data.py exits immediately when nothing's live), while app_export.py
runs on every scheduled pipeline tick regardless.
"""


# ============================================================
# provisional bonus -- FPL's own published bonus-points rule, computed live from BPS
# ============================================================

def is_any_fixture_live(fixtures: list[dict]) -> bool:
    """True if any real fixture has kicked off and FPL hasn't yet confirmed its bonus points
    (fixture['finished'] flips true only once bonus is official, per FPL's own fixture schema).
    scripts/export_live_data.py uses this as its fast no-op exit outside match windows."""
    return any(f.get("started") and not f.get("finished") for f in fixtures)


def _bonus_from_bps(bps_by_player: dict[int, int]) -> dict[int, int]:
    """One fixture's real bonus award from real BPS values, FPL's own published rule: the
    highest BPS gets 3, next distinct BPS gets 2, next distinct gets 1. A tie at a rank shares
    that rank's points AND consumes the ranking positions below it -- e.g. two players tied for
    1st both get 3 and nobody gets 2, but the next distinct BPS still gets the 3rd-place point;
    three players tied for 1st all get 3 and nobody gets 2 or 1 at all."""
    distinct_bps_desc = sorted(set(bps_by_player.values()), reverse=True)
    points_by_rank_slot = [3, 2, 1]
    bonus: dict[int, int] = {}
    slot = 0
    for bps_value in distinct_bps_desc:
        if slot >= len(points_by_rank_slot):
            break
        tied_players = [pid for pid, bps in bps_by_player.items() if bps == bps_value]
        awarded = points_by_rank_slot[slot]
        for pid in tied_players:
            bonus[pid] = awarded
        slot += len(tied_players)
    return bonus


def compute_provisional_bonus(bootstrap: dict, fixtures: list[dict], live: dict) -> dict[int, int]:
    """player_id -> provisional bonus for every fixture that's kicked off but whose bonus FPL
    hasn't confirmed yet. Once a fixture is truly finished (bonus official),
    live['elements'][i]['stats']['bonus'] IS the real number and that fixture's players are
    deliberately left out of this dict -- the caller reads the real stat for them instead of an
    estimate that would just duplicate (or, worse, ever disagree with) FPL's own official figure.
    Players with zero recorded minutes are skipped -- BPS is 0 by default for them and including
    them would incorrectly count as ties among everyone who hasn't played."""
    team_by_player = {e["id"]: e["team"] for e in bootstrap.get("elements", [])}
    stats_by_player = {el["id"]: el.get("stats", {}) for el in live.get("elements", [])}

    result: dict[int, int] = {}
    for fixture in fixtures:
        if not fixture.get("started") or fixture.get("finished"):
            continue
        fixture_teams = {fixture.get("team_h"), fixture.get("team_a")}
        fixture_bps = {
            pid: stats_by_player[pid].get("bps", 0)
            for pid, team in team_by_player.items()
            if team in fixture_teams and pid in stats_by_player and stats_by_player[pid].get("minutes", 0) > 0
        }
        if fixture_bps:
            result.update(_bonus_from_bps(fixture_bps))
    return result


def build_live_squad_rows(picks: list[dict], live: dict) -> list[dict]:
    """Minimal squad rows (player_id/in_xi/multiplier/event_points) straight from a real picks
    payload + the live stats payload -- enough for compute_live_squad_total without the full
    player directory. Used for the rival-sample totals behind estimate_live_rank, where only the
    numeric total matters, not names; app_export.build_team_snapshot() builds the fuller,
    name-enriched version used for the two real accounts' own Team screen."""
    stats_by_id = {el["id"]: el.get("stats", {}) for el in live.get("elements", [])}
    return [
        {
            "player_id": p["element"], "in_xi": p["position"] <= 11,
            "multiplier": p.get("multiplier", 1),
            "event_points": stats_by_id.get(p["element"], {}).get("total_points", 0),
        }
        for p in picks
    ]


# ============================================================
# live squad total -- captain-multiplier-aware, provisional bonus included
# ============================================================

def compute_live_squad_total(squad: list[dict], provisional_bonus_by_id: dict[int, int]) -> dict:
    """Real live total for the starting XI, captain/triple-captain multiplier applied. FPL's own
    live stats.total_points (the source of each squad row's event_points, from
    app_export.build_team_snapshot) does NOT include bonus until a fixture's bonus is officially
    confirmed -- without adding provisional_bonus_by_id on top, a live score understates anyone
    who's about to get a bonus. Players whose fixture already finished aren't in
    provisional_bonus_by_id (see compute_provisional_bonus), so they correctly get +0 here --
    their event_points already includes the real, official bonus."""
    per_player = []
    total = 0
    for p in squad:
        if not p.get("in_xi"):
            continue
        bonus = provisional_bonus_by_id.get(p["player_id"], 0)
        live_points = p["event_points"] + bonus
        contribution = live_points * p.get("multiplier", 1)
        total += contribution
        per_player.append({
            "player_id": p["player_id"], "web_name": p.get("web_name"),
            "live_points": live_points, "provisional_bonus": bonus, "contribution": contribution,
        })
    return {"total": total, "players": per_player}


# ============================================================
# live rank estimate -- sample-based, always disclosed as an estimate
# ============================================================

def estimate_live_rank(your_points: int, sample_points: list[int], total_players: int | None) -> dict:
    """A real, sample-based estimate of live overall rank -- explicitly NOT FPL's own official
    number (FPL doesn't expose overall rank live; it settles once a gameweek is fully
    confirmed). Computed the same way community live-rank tools do: what fraction of a real
    sample of managers' live scores you're currently beating, projected onto the real total
    player count. sample_size is always returned alongside the estimate so nothing downstream
    can present this as more certain than it is."""
    n = len(sample_points)
    if n == 0:
        return {"sample_size": 0, "percentile": None, "estimated_rank": None}
    n_beaten = sum(1 for s in sample_points if your_points > s)
    n_tied = sum(1 for s in sample_points if your_points == s)
    percentile = (n_beaten + 0.5 * n_tied) / n
    # FPL overall rank is 1-indexed -- rank 0 doesn't exist, even for a manager who beats
    # (or ties) everyone in the sample (percentile=1.0, which would otherwise round to 0).
    # Real bug this produced: export_live_data.py's caller checks `if rank_estimate[...]`,
    # so the falsy 0 silently skipped writing a rank snapshot in exactly the one case -- a
    # manager topping their sample -- where a live-rank feature is most worth showing.
    estimated_rank = max(1, round((1 - percentile) * total_players)) if total_players else None
    return {"sample_size": n, "percentile": round(percentile * 100, 1), "estimated_rank": estimated_rank}


# ============================================================
# live match center -- fixture scores + goal/assist events, straight from FPL's own
# live payload. Accumulated for the gameweek (FPL's live elements[].stats are GW-totals,
# not per-minute), so the frontend shows "scorers this GW" and detects "just now" via
# per-poll contribution deltas rather than trusting an event timestamp FPL never publishes.
# ============================================================

def build_live_fixture_rows(fixtures: list[dict], bootstrap: dict) -> list[dict]:
    """One row per fixture that has kicked off in the current gameweek, with the live score
    from FPL's own team_h_score/team_a_score. Pure read of fixture + bootstrap-static team
    fields -- started/finished/minute are FPL's own, never inferred."""
    teams_by_id = {t.get("id"): t for t in bootstrap.get("teams", [])}
    rows: list[dict] = []
    for f in fixtures:
        if not f.get("started"):
            continue
        home = teams_by_id.get(f.get("team_h"), {})
        away = teams_by_id.get(f.get("team_a"), {})
        rows.append({
            "id": f.get("id"),
            "kickoff": f.get("kickoff_time"),
            "minute": f.get("minute"),
            "started": bool(f.get("started")),
            "finished": bool(f.get("finished")),
            "home": {"short_name": home.get("short_name"), "name": home.get("name"), "score": f.get("team_h_score")},
            "away": {"short_name": away.get("short_name"), "name": away.get("name"), "score": f.get("team_a_score")},
        })
    return rows


# FPL live elements[].stats field -> the event label we surface. All real FPL stat keys.
_LIVE_EVENT_FIELDS = [
    ("goals_scored", "goal"),
    ("assists", "assist"),
    ("own_goals", "own_goal"),
    ("penalties_saved", "pen_saved"),
    ("penalties_missed", "pen_missed"),
]


def build_live_event_rows(bootstrap: dict, live: dict, fixtures: list[dict]) -> list[dict]:
    """Goal/assist/penalty events for the current gameweek, from FPL's own live elements stats.
    Only players with recorded minutes are considered (a 0-minute player with goals_scored=0
    would otherwise be dead weight in the feed). Each event carries the fixture_id of the
    started fixture its team is playing in, where that mapping resolves -- FPL doesn't publish
    a per-event fixture link in the live payload, so the team->fixture match is the faithful
    approximation (same convention compute_provisional_bonus already uses)."""
    player_by_id = {e.get("id"): e for e in bootstrap.get("elements", [])}
    team_by_player = {e.get("id"): e.get("team") for e in bootstrap.get("elements", [])}
    # A team -> the single started fixture it's in. Ambiguous in a double gameweek (a team playing
    # two live fixtures at once can't be attributed to one), so those teams resolve to None rather
    # than silently picking the last fixture. The frontend still shows the scorer; only the
    # fixture link is dropped until fixture-level stats are used.
    fixtures_by_team: dict[int, set[int]] = {}
    for f in fixtures:
        if not f.get("started"):
            continue
        fid = f.get("id")
        if not isinstance(fid, int):
            continue
        for side in (f.get("team_h"), f.get("team_a")):
            if isinstance(side, int):
                fixtures_by_team.setdefault(side, set()).add(fid)

    def _fixture_for(team_id) -> int | None:
        fids = fixtures_by_team.get(team_id)
        if not fids or len(fids) != 1:
            return None
        return next(iter(fids))

    rows: list[dict] = []
    for el in live.get("elements", []):
        pid = el.get("id")
        stats = el.get("stats", {}) or {}
        if stats.get("minutes", 0) <= 0:
            continue
        for field, label in _LIVE_EVENT_FIELDS:
            count = stats.get(field, 0)
            if not count:
                continue
            player = player_by_id.get(pid, {})
            rows.append({
                "player_id": pid,
                "web_name": player.get("web_name", ""),
                "team": player.get("team"),
                "event": label,
                "count": count,
                "fixture_id": _fixture_for(team_by_player.get(pid)) if isinstance(team_by_player.get(pid), int) else None,
            })
    return rows
