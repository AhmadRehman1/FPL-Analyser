"""Priority 10 Phase B (partial): rank instrumentation.

Places a squad in a real, stratified sample of rival squads (fact_rival_squad_sample,
schema/0017) for a COMPLETED gameweek and decomposes the rank gap into captaincy /
template-coverage / differential / bench. See schema/0018 and
docs/priority10_field_simulator_design.md.

Forward-only by construction -- FPL's API serves only current-season picks, so this starts
accumulating from 2026-27 GW1. Pure DB reads, no network: the rival sample and the realized
outcomes are both already ingested by the time anything here runs.

The percentile -> projected-rank arithmetic is live_tracking.estimate_live_rank(), reused
verbatim -- the only difference is the sample here is a settled gameweek's real points, not a
live in-match estimate. Realized XI scoring mirrors backtest._realized_xi_points (a separate
small copy, per this project's one-helper-per-module convention); the vice-captain armband-
transfer nuance that function carries is out of scope for step 1 and would only ever move a
score when a captain records zero minutes.

NOT in this module: the joint Monte Carlo across the rival sample (Phase B proper) and any
squad_optimizer integration (Phase C). This measures and attributes; it does not simulate or
optimise. Double-gameweek / blank-gameweek rival squads are scored with realized points as-is
(missing player rows treated as 0) -- a disclosed simplification, not silently patched.
"""

import duckdb

from . import live_tracking

# "The template" for the template-coverage attribution: the N most effectively-owned players
# in the sample. 15 = a full squad's worth -- a manager who owns all 15 is maximally
# template; each one missed is rank ceded when that player returns.
TEMPLATE_TOP_N = 15

# A squad player owned by fewer than this percent of the sample is a "differential" -- the
# holdings whose realized-vs-expected surprise actually moves rank against the field.
DIFFERENTIAL_OWNERSHIP_CEILING_PCT = 10.0


def gameweek_is_settled(con: duckdb.DuckDBPyConnection, season: str, event: int) -> bool:
    """Every real fixture for (season, event) has finished. The rank autopsy only scores
    settled gameweeks -- a half-played gameweek would rank everyone against a partial field."""
    row = con.execute(
        "SELECT count(*), count(*) FILTER (WHERE finished) FROM fact_match WHERE season = ? AND gameweek = ?",
        [season, event],
    ).fetchone()
    return bool(row and row[0] and row[0] == row[1])


def _player_names(con: duckdb.DuckDBPyConnection, uids: list[str]) -> dict[str, str]:
    if not uids:
        return {}
    placeholders = ",".join("?" * len(uids))
    return dict(con.execute(
        f"SELECT player_uid, canonical_name FROM dim_player WHERE player_uid IN ({placeholders})", list(uids),
    ).fetchall())


def _realized_points(con: duckdb.DuckDBPyConnection, season: str, event: int, uids: list[str]) -> dict[str, int]:
    """player_uid -> realized event_points that gameweek. A missing row or a NULL is 0 (a
    benched / non-featuring player), never treated as "unknown" -- the caller has already
    checked the gameweek is settled."""
    if not uids:
        return {}
    placeholders = ",".join("?" * len(uids))
    rows = con.execute(
        f"SELECT player_uid, coalesce(event_points, 0) FROM fact_player_season_stats "
        f"WHERE season = ? AND gw = ? AND player_uid IN ({placeholders})",
        [season, event, *uids],
    ).fetchall()
    got = dict(rows)
    return {u: int(got.get(u, 0)) for u in uids}


def realized_xi_points(
    con: duckdb.DuckDBPyConnection, season: str, event: int,
    xi_uids: list[str] | frozenset, captain_uid: str | None, *, captain_multiplier: int = 2,
) -> int:
    """Real FPL scoring: the starting XI's event_points, with the captain's counted
    `captain_multiplier` times (2 normally, 3 for Triple Captain)."""
    pts = _realized_points(con, season, event, list(xi_uids))
    total = sum(pts.values())
    if captain_uid is not None and captain_uid in pts:
        total += pts[captain_uid] * (captain_multiplier - 1)
    return int(total)


def settled_rival_totals(con: duckdb.DuckDBPyConnection, season: str, event: int) -> list[dict]:
    """Each sampled rival entry's realized gameweek score -- sum(event_points * multiplier)
    over its picks (multiplier already encodes bench=0, captain=2, TC=3). Returns [] when no
    rival sample exists for (season, event). [{entry_id, league_rank, points}]."""
    rows = con.execute(
        """
        SELECT s.entry_id,
               min(s.league_rank) AS league_rank,
               sum(coalesce(fps.event_points, 0) * s.multiplier) AS points
        FROM fact_rival_squad_sample s
        LEFT JOIN fact_player_season_stats fps
          ON fps.player_uid = s.player_uid AND fps.season = s.season AND fps.gw = s.event
        WHERE s.season = ? AND s.event = ? AND s.multiplier > 0
        GROUP BY s.entry_id
        """,
        [season, event],
    ).fetchall()
    return [{"entry_id": e, "league_rank": lr, "points": int(p or 0)} for e, lr, p in rows]


def field_ownership(con: duckdb.DuckDBPyConnection, season: str, event: int) -> dict[str, dict]:
    """player_uid -> {name, eo, owned_pct, captain_pct} from the real sample. `eo` is
    effective ownership (started, multiplier>0) as a percent of sampled entries; `owned_pct`
    counts bench slots too; `captain_pct` is the real captaincy rate (a real sample gets this
    directly, unlike ownership.compute_eo_for_pool's estimate). {} when the sample is empty."""
    n_entries = con.execute(
        "SELECT count(DISTINCT entry_id) FROM fact_rival_squad_sample WHERE season = ? AND event = ?",
        [season, event],
    ).fetchone()[0]
    if not n_entries:
        return {}
    rows = con.execute(
        """
        SELECT s.player_uid, dp.canonical_name,
               count(DISTINCT s.entry_id) FILTER (WHERE s.multiplier > 0) AS n_starting,
               count(DISTINCT s.entry_id) AS n_owning,
               count(DISTINCT s.entry_id) FILTER (WHERE s.is_captain) AS n_captains
        FROM fact_rival_squad_sample s JOIN dim_player dp ON dp.player_uid = s.player_uid
        WHERE s.season = ? AND s.event = ?
        GROUP BY s.player_uid, dp.canonical_name
        """,
        [season, event],
    ).fetchall()
    return {
        uid: {
            "name": name,
            "eo": round(100.0 * n_start / n_entries, 1),
            "owned_pct": round(100.0 * n_own / n_entries, 1),
            "captain_pct": round(100.0 * n_cap / n_entries, 1),
        }
        for uid, name, n_start, n_own, n_cap in rows
    }


def sample_shape(con: duckdb.DuckDBPyConnection, season: str, event: int) -> dict:
    """The overall-rank bands the stored rival sample for (season, event) actually covers --
    read off the league_rank column, so it reflects what fetch_entries_in_rank_bands really
    reached (the standings API's deep-pagination frontier is not knowable in advance), not
    what was requested. {} when there is no sample."""
    rows = con.execute(
        "SELECT league_rank FROM (SELECT DISTINCT entry_id, league_rank FROM fact_rival_squad_sample "
        "WHERE season = ? AND event = ? AND league_rank IS NOT NULL)",
        [season, event],
    ).fetchall()
    ranks = sorted(r[0] for r in rows)
    if not ranks:
        return {}
    edges = [(10_000, "top_10k"), (60_000, "10k_60k"), (140_000, "60k_140k"),
             (400_000, "140k_400k"), (600_000, "400k_600k")]
    buckets: dict[str, int] = {}
    for r in ranks:
        key = next((label for edge, label in edges if r <= edge), "600k_plus")
        buckets[key] = buckets.get(key, 0) + 1
    return {
        "n_entries": len(ranks), "rank_min": ranks[0], "rank_max": ranks[-1],
        "rank_p50": ranks[len(ranks) // 2], "buckets": buckets,
    }


def score_squad_rank(
    con: duckdb.DuckDBPyConnection, season: str, event: int,
    xi_uids: list[str] | frozenset, captain_uid: str | None, total_players: int | None,
    *, captain_multiplier: int = 2,
) -> dict:
    """Where the subject squad's realized XI score places it against the settled rival sample.
    percentile / estimated_rank come straight from live_tracking.estimate_live_rank. Returns
    n_rivals=0 (percentile/estimated_rank None) when there is no settled sample -- the caller
    must not persist a score in that case."""
    your_points = realized_xi_points(con, season, event, xi_uids, captain_uid, captain_multiplier=captain_multiplier)
    sample_points = [r["points"] for r in settled_rival_totals(con, season, event)]
    est = live_tracking.estimate_live_rank(your_points, sample_points, total_players)
    return {
        "realized_points": your_points,
        "n_rivals": len(sample_points),
        "n_beaten": sum(1 for s in sample_points if your_points > s),
        "n_tied": sum(1 for s in sample_points if your_points == s),
        "percentile": est["percentile"],
        "estimated_rank": est["estimated_rank"],
        "total_players": total_players,
    }


def attribute_rank_gap(
    con: duckdb.DuckDBPyConnection, season: str, event: int,
    xi_uids: list[str], captain_uid: str | None, bench_uids: list[str],
    *, ep_model_version: int | None = None,
) -> dict:
    """Decompose the subject squad's rank gap vs the field into four buckets. Every points
    figure is signed so a reader (or the cumulative roll-up) can add them: negative = points
    the choice cost against the field, positive = points it won.
    """
    own = field_ownership(con, season, event)
    xi = list(xi_uids)
    bench = list(bench_uids)
    squad = xi + bench
    realized = _realized_points(con, season, event, squad)
    xi_realized = {u: realized.get(u, 0) for u in xi}
    names = _player_names(con, squad + ([captain_uid] if captain_uid else []))

    def name_of(uid: str | None) -> str | None:
        if uid is None:
            return None
        return (own.get(uid) or {}).get("name") or names.get(uid) or uid

    # -- captaincy: the marginal (single) captain point vs the alternatives; the armband adds
    #    the same delta again on top, so the rank cost of a wrong armband is ~2x these numbers.
    captain_pts = realized.get(captain_uid, 0) if captain_uid else 0
    field_cap_uid = max(
        (uid for uid, m in own.items() if m["captain_pct"] > 0),
        key=lambda uid: own[uid]["captain_pct"], default=None,
    )
    field_cap_pts = _realized_points(con, season, event, [field_cap_uid]).get(field_cap_uid, 0) if field_cap_uid else 0
    best_xi_uid = max(xi_realized, key=lambda u: xi_realized[u], default=None)
    best_xi_pts = xi_realized[best_xi_uid] if best_xi_uid is not None else 0
    captaincy = {
        "captain": name_of(captain_uid),
        "captain_points": captain_pts,
        "field_captain": name_of(field_cap_uid),
        "field_captain_points": field_cap_pts,
        "field_captain_pct": (own.get(field_cap_uid) or {}).get("captain_pct") if field_cap_uid else None,
        "vs_field_captain": captain_pts - field_cap_pts,
        "best_in_xi": name_of(best_xi_uid),
        "vs_best_in_xi": captain_pts - best_xi_pts,
    }

    # -- template coverage: of the N most effectively-owned sampled players, how many the
    #    squad holds; EO-weighted realized points of the ones it missed = rank ceded.
    template = sorted(own.items(), key=lambda kv: kv[1]["eo"], reverse=True)[:TEMPLATE_TOP_N]
    owned = set(squad)
    missed = [(uid, m) for uid, m in template if uid not in owned]
    missed_pts = _realized_points(con, season, event, [uid for uid, _ in missed])
    template_coverage = {
        "template_owned": len(template) - len(missed),
        "template_size": len(template),  # < TEMPLATE_TOP_N only when the sample has fewer distinct players
        "missed": [
            {
                "name": m["name"], "eo": m["eo"], "points": missed_pts.get(uid, 0),
                "eo_weighted_points": round(m["eo"] / 100.0 * missed_pts.get(uid, 0), 2),
            }
            for uid, m in missed
        ],
        "eo_weighted_points_bled": -round(sum(m["eo"] / 100.0 * missed_pts.get(uid, 0) for uid, m in missed), 2),
    }

    # -- differentials: sub-ceiling-owned holdings, realized minus model-expected.
    diffs = [u for u in squad if (own.get(u) or {}).get("owned_pct", 0.0) < DIFFERENTIAL_OWNERSHIP_CEILING_PCT]
    ep_of: dict[str, float] = {}
    if ep_model_version is not None and diffs:
        placeholders = ",".join("?" * len(diffs))
        ep_of = dict(con.execute(
            f"SELECT player_uid, ep_total FROM ep_outputs WHERE model_version = ? AND player_uid IN ({placeholders})",
            [ep_model_version, *diffs],
        ).fetchall())
    differentials = {
        "n": len(diffs),
        "players": [
            {
                "name": name_of(u), "owned_pct": (own.get(u) or {}).get("owned_pct", 0.0),
                "in_xi": u in set(xi), "points": realized.get(u, 0),
                "ep": round(ep_of[u], 2) if u in ep_of else None,
                "surprise": round(realized.get(u, 0) - ep_of[u], 2) if u in ep_of else None,
            }
            for u in diffs
        ],
        "net_surprise": round(sum(realized.get(u, 0) - ep_of[u] for u in diffs if u in ep_of), 2),
    }

    bench_points = sum(realized.get(u, 0) for u in bench)
    bench_bucket = {
        "points_left": -bench_points,
        "note": "auto-substitution not modelled -- an upper bound on bench cost",
    }

    return {
        "captaincy": captaincy,
        "template_coverage": template_coverage,
        "differentials": differentials,
        "bench": bench_bucket,
    }
