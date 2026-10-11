"""Premium and in-form returns, Phase 1 (docs/plans/2026-10_premium_and_form_returns.md, RQ1 and
RQ2b's alignment check). Diagnosis only: it writes nothing to the DB.

The walk-forward under-predicts £9.0m+ assists and "other" (bonus etc.) and over-predicts their
goals. Before any lever is tried, this separates "the measurement is off" from "the model is off":
  assist_sources  FPL's season assists against the match feed the backtest's component split
                  counts (the split puts any assist FPL awards and the feed lacks into "other")
  conversion      goals per xG and assists per xA by price band (FPL totals from 2025-26; the
                  match feed and per-90 xG x match minutes for 2024-25, whose totals are NULL)
  bonus_columns   any raw table column with "bonus" in its name (no reconciled table has one)
  form_groups     in-form / cooling player-steps per season, declared and widened, counted the
                  way the walk-forward judge would see them, and which version judging uses
  live_form       the live season's in-form and cooling players at the next gameweek
  assist_alignment  FPL's per-gameweek assists (differences of the cumulative season count)
                  against the match feed, gameweek by gameweek -- the check RQ2b needs before
                  FPL's count can replace the feed's

Usage (from repo root, against the cached DB):
    PYTHONPATH=src python scripts/diagnose_premium_returns.py > diagnose_premium_returns.json
"""

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from fpl_quant import backtest, breakout, db, form_groups  # noqa: E402

import compute_ml_shadow as cms  # noqa: E402

BANDS = ("<5.0", "5.0-7.0", "7.0-9.0", "9.0+")
MIN_CONVERSION_MINUTES = 900
# H3 is killed when FPL and the feed agree within this share at 9.0m+ (plan, Approach).
ASSIST_AGREEMENT = 0.03
# RQ2b applies only if FPL's per-gameweek assists match the feed on this share of 9.0m+
# player-gameweeks the player played in.
ALIGNMENT_THRESHOLD = 0.95


def _league_seasons(con) -> list[str]:
    return [r[0] for r in con.execute(
        "SELECT DISTINCT season FROM fact_match WHERE competition = ? ORDER BY season", [backtest.PL],
    ).fetchall()]


def _has_fpl_totals(con, season: str) -> bool:
    return bool(con.execute(
        "SELECT count(*) FROM fact_player_season_stats WHERE season = ? AND goals_scored IS NOT NULL", [season],
    ).fetchone()[0])


def _last_rows(con, season: str) -> dict:
    """{player_uid: (position, now_cost, minutes, goals, assists, xg, xa, xg90, xa90)} from each
    player's last row of the season."""
    rows = con.execute(
        "SELECT f.player_uid, dp.position, f.now_cost, f.minutes, f.goals_scored, f.assists, f.expected_goals, "
        "f.expected_assists, f.expected_goals_per_90, f.expected_assists_per_90 "
        "FROM fact_player_season_stats f JOIN dim_player dp ON dp.player_uid = f.player_uid "
        "WHERE f.season = ? QUALIFY row_number() OVER (PARTITION BY f.player_uid ORDER BY f.gw DESC) = 1",
        [season],
    ).fetchall()
    return {r[0]: r[1:] for r in rows}


def _feed_totals(con, season: str) -> dict:
    """{player_uid: (league minutes, goals, assists)} from the match feed."""
    rows = con.execute(
        "SELECT s.player_uid, sum(coalesce(s.minutes_played, 0)), sum(coalesce(s.goals, 0)), sum(coalesce(s.assists, 0)) "
        "FROM fact_player_match_stats s JOIN fact_match m ON m.match_id = s.match_id "
        "WHERE m.season = ? AND m.competition = ? GROUP BY s.player_uid",
        [season, backtest.PL],
    ).fetchall()
    return {r[0]: (float(r[1]), float(r[2]), float(r[3])) for r in rows}


def _ratio(num: float, den: float):
    return round(num / den, 4) if den else None


def assist_sources(con, season: str) -> dict:
    """Per price band: FPL season assists against the match feed's league assists, over players
    with league minutes in either."""
    last, feed = _last_rows(con, season), _feed_totals(con, season)
    out = {b: {"n_players": 0, "fpl_assists": 0.0, "feed_assists": 0.0, "n_differ": 0} for b in BANDS}
    for uid, (_pos, cost, minutes, _g, assists, *_rest) in last.items():
        if assists is None:
            continue
        band = backtest._price_band(cost)
        if band not in out:
            continue
        feed_minutes, _fg, feed_assists = feed.get(uid, (0.0, 0.0, 0.0))
        if not (minutes or feed_minutes):
            continue
        block = out[band]
        block["n_players"] += 1
        block["fpl_assists"] += assists
        block["feed_assists"] += feed_assists
        block["n_differ"] += int(assists != feed_assists)
    for block in out.values():
        block["fpl_per_feed"] = _ratio(block["fpl_assists"], block["feed_assists"])
    return out


def conversion(con, season: str) -> dict:
    """Per price band (and position within it), players with at least MIN_CONVERSION_MINUTES
    league minutes: pooled goals / xG and assists / xA."""
    fpl = _has_fpl_totals(con, season)
    last, feed = _last_rows(con, season), _feed_totals(con, season)
    acc: dict = {}
    for uid, (pos, cost, minutes, goals, assists, xg, xa, xg90, xa90) in last.items():
        band = backtest._price_band(cost)
        if band not in BANDS:
            continue
        if fpl:
            if not minutes or xg is None:
                continue
            mins, g, a, exg, exa = float(minutes), float(goals or 0), float(assists or 0), float(xg or 0), float(xa or 0)
        else:
            mins, g, a = feed.get(uid, (0.0, 0.0, 0.0))
            if xg90 is None and xa90 is None:
                continue
            exg, exa = (xg90 or 0.0) * mins / 90.0, (xa90 or 0.0) * mins / 90.0
        if mins < MIN_CONVERSION_MINUTES:
            continue
        for key in (band, f"{band}|{pos}"):
            block = acc.setdefault(key, {"n_players": 0, "minutes": 0.0, "goals": 0.0, "xg": 0.0, "assists": 0.0, "xa": 0.0})
            block["n_players"] += 1
            block["minutes"] += mins
            block["goals"] += g
            block["xg"] += exg
            block["assists"] += a
            block["xa"] += exa
    out: dict = {"source": "fpl" if fpl else "match feed", "bands": {}}
    for key, block in sorted(acc.items()):
        block = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in block.items()}
        block["goals_per_xg"] = _ratio(block["goals"], block["xg"])
        block["assists_per_xa"] = _ratio(block["assists"], block["xa"])
        band, _, pos = key.partition("|")
        target = out["bands"].setdefault(band, {})
        if pos:
            target.setdefault("by_position", {})[pos] = block
        else:
            target.update(block)
    return out


def bonus_columns(con) -> dict:
    """Raw tables (listed in fact_raw_ingestion_log) with a column whose name contains "bonus",
    collapsed by source path pattern (GW numbers replaced by *)."""
    tables = con.execute(
        "SELECT DISTINCT raw_table_name, season, source_relpath FROM fact_raw_ingestion_log ORDER BY 1",
    ).fetchall()
    names = [t[0] for t in tables]
    found: dict = {}
    if names:
        cols = con.execute(
            "SELECT table_name, column_name FROM information_schema.columns "
            "WHERE table_name IN (SELECT unnest(?::VARCHAR[])) AND lower(column_name) LIKE '%bonus%' ORDER BY 1, 2",
            [names],
        ).fetchall()
        by_table: dict = {}
        for table, col in cols:
            by_table.setdefault(table, []).append(col)
        for table, season, relpath in tables:
            if table not in by_table:
                continue
            pattern = re.sub(r"GW\d+", "GW*", relpath)
            entry = found.setdefault(f"{season}:{pattern}", {"columns": by_table[table], "n_tables": 0})
            entry["n_tables"] += 1
    return {"tables_checked": len(names), "with_bonus": found}


def _visible(con, season: str, gameweek: int, uids: list[str]) -> set[str]:
    """The players the walk-forward judge sees at this step: a non-null event_points row for the
    gameweek and a league fixture for his club in it."""
    if not uids:
        return set()
    rows = con.execute(
        "SELECT DISTINCT f.player_uid FROM fact_player_season_stats f "
        "JOIN _player_season_team t ON t.player_uid = f.player_uid AND t.season = f.season "
        "AND t.first_gw <= f.gw AND t.last_gw >= f.gw "
        "JOIN fact_match m ON m.season = f.season AND m.gameweek = f.gw AND m.competition = ? "
        "AND t.team_uid IN (m.home_team_uid, m.away_team_uid) "
        "WHERE f.season = ? AND f.gw = ? AND f.event_points IS NOT NULL AND f.player_uid IN (SELECT unnest(?::VARCHAR[]))",
        [backtest.PL, season, gameweek, uids],
    ).fetchall()
    return {r[0] for r in rows}


def form_group_sizes(con, season: str) -> dict | None:
    """Player-steps per label over the season's played gameweeks, declared and widened, and
    the version judging uses: declared if its in-form group reaches MIN_GROUP_SIZE, else the
    widened one if it does, else none (that half of the plan stops)."""
    gameweeks = [r[0] for r in con.execute(
        "SELECT DISTINCT gameweek FROM fact_match WHERE season = ? AND competition = ? AND home_score IS NOT NULL "
        "AND gameweek IS NOT NULL ORDER BY 1", [season, backtest.PL],
    ).fetchall()]
    counts = {f"{label}{suffix}": 0 for label in (form_groups.IN_FORM, form_groups.COOLING) for suffix in ("", "_widened")}
    memo: dict = {}
    for gw in gameweeks:
        deadline = backtest.gameweek_deadline(con, season, gw)
        for suffix, rule in (("", {}), ("_widened", form_groups.WIDENED)):
            groups = form_groups.classify(con, season, gw, deadline, memo=memo, **rule)
            if groups is None:
                return None
            seen = _visible(con, season, gw, list(groups))
            for uid, label in groups.items():
                if uid in seen:
                    counts[label + suffix] += 1
    if counts[form_groups.IN_FORM] >= form_groups.MIN_GROUP_SIZE:
        judged = "declared"
    elif counts[form_groups.IN_FORM + "_widened"] >= form_groups.MIN_GROUP_SIZE:
        judged = "widened"
    else:
        judged = "too_small"
    return {"gameweeks": len(gameweeks), "player_steps": counts, "judged_on": judged,
            "min_group_size": form_groups.MIN_GROUP_SIZE}


def live_form(con, season: str, gameweek: int | None) -> dict:
    if gameweek is None:
        return {"status": "no_next_gameweek"}
    deadline = backtest.gameweek_deadline(con, season, gameweek)
    out: dict = {"season": season, "gameweek": gameweek}
    for suffix, rule in (("", {}), ("_widened", form_groups.WIDENED)):
        details: dict = {}
        groups = form_groups.classify(con, season, gameweek, deadline, details=details, **rule)
        if groups is None:
            return {"status": "no_earlier_season"}
        rows = []
        for uid, label in groups.items():
            name, pos = con.execute("SELECT canonical_name, position FROM dim_player WHERE player_uid = ?", [uid]).fetchone()
            cost = con.execute(
                "SELECT now_cost FROM fact_player_season_stats WHERE player_uid = ? AND season = ? AND gw < ? "
                "ORDER BY gw DESC LIMIT 1", [uid, season, gameweek],
            ).fetchone()
            rows.append({"player_uid": uid, "name": name, "position": pos, "price": cost[0] if cost else None,
                         "label": label, **details[uid]})
        out["members" + suffix] = sorted(rows, key=lambda r: -r["ratio"])
    return out


def assist_alignment(con, season: str) -> dict:
    """FPL's per-gameweek assists, as differences of the cumulative season count between a
    player's consecutive rows, against the match feed's league assists in that gameweek, over the
    player-gameweeks he played in. Also the same comparison shifted a gameweek either way, so an
    off-by-one in what "row G" covers shows up as a better shifted score."""
    rows = con.execute(
        "SELECT player_uid, gw, assists, now_cost FROM fact_player_season_stats "
        "WHERE season = ? ORDER BY player_uid, gw", [season],
    ).fetchall()
    feed = {(r[0], r[1]): (float(r[2]), float(r[3])) for r in con.execute(
        "SELECT s.player_uid, m.gameweek, sum(coalesce(s.assists, 0)), sum(coalesce(s.minutes_played, 0)) "
        "FROM fact_player_match_stats s JOIN fact_match m ON m.match_id = s.match_id "
        "WHERE m.season = ? AND m.competition = ? AND m.gameweek IS NOT NULL GROUP BY 1, 2",
        [season, backtest.PL],
    ).fetchall()}
    fpl: dict = {}
    band_of: dict = {}
    negatives = 0
    prev_uid, prev_cum = None, 0.0
    last_total: dict = {}
    summed: dict = {}
    for uid, gw, cum, cost in rows:
        if uid != prev_uid:
            prev_uid, prev_cum = uid, 0.0
        if cum is None:
            continue
        diff = float(cum) - prev_cum
        negatives += int(diff < 0)
        fpl[(uid, gw)] = diff
        band_of[(uid, gw)] = backtest._price_band(cost)
        summed[uid] = summed.get(uid, 0.0) + diff
        prev_cum = last_total[uid] = float(cum)
    sums_off = sum(abs(summed[uid] - total) > 1e-9 for uid, total in last_total.items())

    def agreement(shift: int, band: str | None) -> dict:
        n = agree = 0
        for (uid, gw), (feed_assists, minutes) in feed.items():
            if minutes <= 0 or (band is not None and band_of.get((uid, gw)) != band):
                continue
            n += 1
            agree += int(fpl.get((uid, gw + shift)) == feed_assists)
        return {"player_gameweeks": n, "agree": agree, "share": _ratio(agree, n)}

    premium = agreement(0, "9.0+")
    return {
        "players": len(last_total), "players_whose_differences_miss_the_total": sums_off,
        "negative_differences": negatives,
        "9.0+": premium, "9.0+_fpl_one_gw_later": agreement(1, "9.0+"), "9.0+_fpl_one_gw_earlier": agreement(-1, "9.0+"),
        "all": agreement(0, None),
        "threshold": ALIGNMENT_THRESHOLD,
        "passes": sums_off == 0 and premium["share"] is not None and premium["share"] >= ALIGNMENT_THRESHOLD,
    }


def diagnose(con) -> dict:
    seasons = _league_seasons(con)
    breakout.build_club_spells(con, tuple(seasons))
    fpl_seasons = [s for s in seasons if _has_fpl_totals(con, s)]
    payload: dict = {"seasons": seasons, "seasons_with_fpl_totals": fpl_seasons}
    payload["assist_sources"] = {s: assist_sources(con, s) for s in fpl_seasons}
    payload["conversion"] = {s: conversion(con, s) for s in seasons}
    payload["bonus_columns"] = bonus_columns(con)
    payload["form_groups"] = {s: form_group_sizes(con, s) for s in seasons}
    payload["assist_alignment"] = {s: assist_alignment(con, s) for s in fpl_seasons}
    cutoff = cms._data_cutoff(con)
    live_season = cms.TARGET_SEASON
    payload["live_form"] = (
        live_form(con, live_season, cms._target_gameweek_from_db(con, cutoff))
        if live_season in seasons else {"status": "live_season_not_loaded"}
    )
    payload["verdicts"] = verdicts(payload["assist_sources"], payload["assist_alignment"])
    return payload


def verdicts(sources: dict, alignment: dict) -> dict:
    """Per season with FPL totals: H3 is killed when FPL and the feed agree within
    ASSIST_AGREEMENT at 9.0m+; RQ2b applies only if H3 survives and the alignment check passes."""
    out = {}
    for season, bands in sources.items():
        ratio = bands["9.0+"]["fpl_per_feed"]
        h3_killed = ratio is not None and abs(ratio - 1.0) <= ASSIST_AGREEMENT
        out[season] = {"9.0+_fpl_per_feed": ratio, "h3_killed": h3_killed,
                       "rq2b_applies": (not h3_killed) and alignment[season]["passes"]}
    return out


def main() -> None:
    con = db.connect()
    print(json.dumps(diagnose(con), indent=2, default=str))
    con.close()


if __name__ == "__main__":
    main()
