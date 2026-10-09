"""Where does the live model lose this season's breakout players? (docs/plans/
2026-10_breakout_players.md, Phase 1 / R1). Diagnosis only: it writes nothing to the DB.

For the next gameweek's live EP (the version compute_ml_shadow.py would read), it splits each
named player's prediction into minutes, scoring rates and club strength, and sets them against
what he is doing this season. Scoring rates are compared with his xG/xA per 90, not his goals
and assists, so a few matches of finishing luck don't pass for a rate problem.

Usage (from repo root, against the cached live DB):
    PYTHONPATH=src python scripts/diagnose_breakouts.py [player_uid ...] > diagnose_breakouts.json
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from fpl_quant import backtest, breakout, db  # noqa: E402

import compute_ml_shadow as cms  # noqa: E402

TARGET_SEASON = cms.TARGET_SEASON
PROJECTIONS_PATH = REPO_ROOT / "data" / "dashboard" / "projections_latest.json"
NAMED = ("player_pascal_gross", "player_antonin_kinsky", "player_charalampos_kostoulas", "player_maxim_de_cuyper")
REFERENCE = ("player_bruno_borges_fernandes", "player_bryan_mbeumo")
# base_scoring_matrix v1 (expected_points.seed_v1_params): points per goal by position, per assist.
GOAL_POINTS = {"Goalkeeper": 10.0, "Defender": 6.0, "Midfielder": 5.0, "Forward": 4.0}
ASSIST_POINTS = 3.0
# Rough minutes for a 1-59 minute appearance, only to turn p_60plus/p_1_59 into expected minutes.
SUB_MINUTES = 35.0


def _projection_ep(gameweek: int) -> dict[str, float]:
    if not PROJECTIONS_PATH.exists():
        return {}
    data = json.loads(PROJECTIONS_PATH.read_text())
    return {
        p["player_uid"]: e["ep"]
        for p in data.get("players", []) for e in p.get("ep_per_gw", []) if e.get("gw") == gameweek
    }


def _one(con, uid, ep_mv, mm_mv, ts_mv, gameweek, deadline, groups, projected) -> dict:
    row = {"player_uid": uid}
    name_pos = con.execute("SELECT canonical_name, position FROM dim_player WHERE player_uid = ?", [uid]).fetchone()
    if name_pos is None:
        return {**row, "status": "unknown_player"}
    row["name"], row["position"] = name_pos

    mins = con.execute(
        "SELECT p_start_final, p_60plus_min, p_1_59min, weight_own FROM minutes_model_outputs "
        "WHERE model_version = ? AND player_uid = ?", [mm_mv, uid],
    ).fetchone()
    row["minutes_model"] = None if mins is None else dict(zip(("p_start_final", "p_60plus_min", "p_1_59min", "weight_own"), mins))

    # the match-level record the minutes model's current-season blends read (every competition
    # in fact_player_match_stats), which can differ from FPL's league totals in this_season
    matches = con.execute(
        "SELECT m.competition, m.gameweek, s.start_min, s.minutes_played FROM fact_player_match_stats s "
        "JOIN fact_match m ON m.match_id = s.match_id WHERE s.player_uid = ? AND s.season = ? "
        "ORDER BY m.kickoff_time", [uid, TARGET_SEASON],
    ).fetchall()
    row["match_stats_this_season"] = {
        "starts": sum(1 for _, _, start, _ in matches if start == 0),
        "starts_60plus": sum(1 for _, _, start, played in matches if start == 0 and (played or 0) >= 60),
        "matches": [{"competition": c, "gw": g, "start_min": start, "minutes": played} for c, g, start, played in matches],
    }

    ep = con.execute("SELECT * FROM ep_gameweek_outputs WHERE model_version = ? AND player_uid = ?", [ep_mv, uid]).fetchdf()
    row["ep"] = None if ep.empty else {k: (float(v) if k.startswith(("ep_", "expected_")) else v)
                                       for k, v in ep.iloc[0].to_dict().items() if k not in ("model_version", "player_uid")}

    fixture = con.execute(
        "SELECT m.home_team_uid, m.away_team_uid FROM ep_outputs o JOIN fact_match m ON m.match_id = o.fixture_match_id "
        "WHERE o.model_version = ? AND o.player_uid = ? LIMIT 1", [ep_mv, uid],
    ).fetchone()
    club = con.execute(
        "SELECT team_uid FROM _player_season_team WHERE player_uid = ? AND season = ? AND first_gw <= ? AND last_gw >= ? "
        "ORDER BY first_gw DESC LIMIT 1", [uid, TARGET_SEASON, gameweek, gameweek],
    ).fetchone()
    club = club[0] if club else None
    opponent = None
    if fixture and club in fixture:
        opponent = fixture[1] if fixture[0] == club else fixture[0]

    def strength(team):
        r = con.execute(
            "SELECT final_attack, final_defence FROM team_strength_snapshots WHERE model_version = ? AND team_uid = ?",
            [ts_mv, team],
        ).fetchone()
        return None if r is None else {"attack": r[0], "defence": r[1]}

    league = con.execute(
        "SELECT median(final_attack), median(final_defence) FROM team_strength_snapshots WHERE model_version = ?", [ts_mv],
    ).fetchone()
    row["club"] = {"team_uid": club, **(strength(club) or {})} if club else None
    row["opponent"] = {"team_uid": opponent, **(strength(opponent) or {})} if opponent else None
    row["league_median"] = {"attack": league[0], "defence": league[1]}

    stats = con.execute(
        "SELECT gw, minutes, goals_scored, assists, expected_goals, expected_assists, expected_goals_per_90, "
        "expected_assists_per_90, total_points, now_cost FROM fact_player_season_stats "
        "WHERE player_uid = ? AND season = ? AND gw < ? AND minutes IS NOT NULL ORDER BY gw DESC LIMIT 1",
        [uid, TARGET_SEASON, gameweek],
    ).fetchone()
    keys = ("as_of_gw", "minutes", "goals", "assists", "xg", "xa", "xg_per_90", "xa_per_90", "total_points", "price")
    season = dict(zip(keys, stats)) if stats else {}
    starts = con.execute(
        """
        SELECT count(*) FILTER (WHERE pms.start_min = 0), count(pms.match_id)
        FROM fact_player_match_stats pms JOIN fact_match m ON m.match_id = pms.match_id
        WHERE pms.player_uid = ? AND m.season = ? AND m.competition = ? AND m.kickoff_time < ?
        """,
        [uid, TARGET_SEASON, backtest.PL, deadline],
    ).fetchone()
    club_played = con.execute(
        "SELECT count(*) FROM fact_match WHERE season = ? AND competition = ? AND kickoff_time < ? "
        "AND home_score IS NOT NULL AND ? IN (home_team_uid, away_team_uid)",
        [TARGET_SEASON, backtest.PL, deadline, club],
    ).fetchone()[0] if club else 0
    season.update({"starts": starts[0], "appearances": starts[1], "club_matches_played": club_played})
    if season.get("minutes"):
        season["points_per_90"] = 90.0 * (season.get("total_points") or 0) / season["minutes"]
    row["this_season"] = season
    row["in_breakout_group"] = None if groups is None else groups.get(uid)
    row["projections_latest_ep"] = projected.get(uid)
    if row["ep"] is not None and row["projections_latest_ep"] is not None:
        row["matches_projections_within_0_05"] = abs(row["ep"]["ep_total"] - row["projections_latest_ep"]) <= 0.05
    row["gap"] = _attribute(row)
    return row


def _attribute(row: dict) -> dict:
    """Rough points-per-gameweek each cause would add if the model matched this season. Crude by
    design (linear in expected minutes); it ranks causes, it doesn't forecast."""
    mm, ep, season = row.get("minutes_model"), row.get("ep"), row.get("this_season") or {}
    if not mm or not ep:
        return {"status": "missing_model_rows"}
    xmins = 90.0 * mm["p_60plus_min"] + SUB_MINUTES * mm["p_1_59min"]
    played = season.get("club_matches_played") or 0
    start_share = season.get("starts", 0) / played if played else None
    out = {"expected_minutes": xmins, "start_share_this_season": start_share}
    causes = {}
    if start_share is not None and mm["p_start_final"] > 0:
        causes["minutes"] = ep["ep_total"] * (start_share / mm["p_start_final"] - 1.0)
    gp = GOAL_POINTS.get(row.get("position"), 5.0)
    if xmins > 0:
        implied_g90 = ep["ep_goals"] / gp / (xmins / 90.0)
        implied_a90 = ep["ep_assists"] / ASSIST_POINTS / (xmins / 90.0)
        out.update({"model_goals_per_90": implied_g90, "model_assists_per_90": implied_a90})
        if season.get("xg_per_90") is not None and season.get("xa_per_90") is not None:
            causes["scoring_rates"] = (
                (season["xg_per_90"] - implied_g90) * gp + (season["xa_per_90"] - implied_a90) * ASSIST_POINTS
            ) * xmins / 90.0
    club, league = row.get("club") or {}, row.get("league_median") or {}
    if club.get("attack") is not None and league.get("attack") is not None:
        out["club_attack_vs_median"] = club["attack"] - league["attack"]
        out["club_defence_vs_median"] = club["defence"] - league["defence"]
    out["causes_points_per_gw"] = causes
    out["main_cause"] = max(causes, key=causes.get) if causes and max(causes.values()) > 0 else "none_found"
    return out


def diagnose(con, player_uids: tuple[str, ...] = NAMED + REFERENCE) -> dict:
    cutoff = cms._data_cutoff(con)
    gameweek = cms._target_gameweek_from_db(con, cutoff)
    payload = {"target_season": TARGET_SEASON, "data_cutoff": None if cutoff is None else cutoff.isoformat(),
               "target_gameweek": gameweek, "status": "ok", "players": []}
    if gameweek is None:
        payload["status"] = "no_next_gameweek"
        return payload
    versions = cms._ep_versions_for_gameweek(con, gameweek)
    if versions is None:
        payload["status"] = "no_ep_outputs_for_target_gameweek"
        return payload
    ep_mv, mm_mv = versions
    ts_mv = con.execute("SELECT team_strength_model_version FROM ep_model_versions WHERE model_version = ?", [ep_mv]).fetchone()[0]
    deadline = backtest.gameweek_deadline(con, TARGET_SEASON, gameweek)
    seasons = tuple(r[0] for r in con.execute(
        "SELECT DISTINCT season FROM fact_match WHERE competition = ? AND season <= ? ORDER BY season", [backtest.PL, TARGET_SEASON],
    ).fetchall())
    breakout.build_club_spells(con, seasons)
    groups = breakout.classify(con, TARGET_SEASON, gameweek, deadline)
    payload.update({"ep_model_version": ep_mv, "minutes_model_version": mm_mv, "team_strength_model_version": ts_mv,
                    "deadline": deadline.isoformat(),
                    "breakout_group_size": None if groups is None else sum(v == breakout.BREAKOUT for v in groups.values())})
    projected = _projection_ep(gameweek)
    payload["players"] = [_one(con, uid, ep_mv, mm_mv, ts_mv, gameweek, deadline, groups, projected) for uid in player_uids]
    return payload


def main() -> None:
    con = db.connect()
    uids = tuple(sys.argv[1:]) or NAMED + REFERENCE
    print(json.dumps(diagnose(con, uids), indent=2, default=str))
    con.close()


if __name__ == "__main__":
    main()
