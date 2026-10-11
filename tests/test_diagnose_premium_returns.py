"""scripts/diagnose_premium_returns.py: FPL vs match-feed assists, goals/xG and assists/xA by price
band, raw bonus columns, in-form group sizes and the per-gameweek assist alignment check."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import diagnose_premium_returns as diag  # noqa: E402
import test_form_groups as tfg  # noqa: E402

NOW = datetime(2026, 1, 1)
PRIOR, SEASON = "2024-2025", "2025-2026"


def _match(con, mid, season, gw, kickoff):
    con.execute(
        "INSERT INTO fact_match (match_id, season, gameweek, kickoff_time, home_team_uid, away_team_uid, "
        "home_score, away_score, finished, competition, _ingested_at) "
        "VALUES (?, ?, ?, ?, 'team_a', 'team_b', 1, 0, TRUE, 'Premier League', ?)",
        [mid, season, gw, kickoff, NOW],
    )


def _played(con, uid, mid, season, *, goals=0, assists=0):
    con.execute(
        "INSERT INTO fact_player_match_stats (player_uid, match_id, season, start_min, minutes_played, goals, assists, "
        "_ingested_at) VALUES (?, ?, ?, 0, 90, ?, ?, ?)",
        [uid, mid, season, goals, assists, NOW],
    )


def _row(con, uid, season, gw, cost, **cols):
    names = ["player_uid", "season", "gw", "now_cost", "_ingested_at", *cols]
    con.execute(
        f"INSERT INTO fact_player_season_stats ({', '.join(names)}) VALUES ({', '.join('?' * len(names))})",
        [uid, season, gw, cost, NOW, *cols.values()],
    )


def _seed(con):
    """prem (MID, £10.0m) and cheap (DEF, £4.5m), both 90 minutes every match.
    2024-25, 20 matches, per-90-only season rows (the real shape): prem scores in GW1-10 and assists
    in GW1-5 (xG 0.4, xA 0.2 per 90); cheap assists once (xG 0.05, xA 0.05 per 90).
    2025-26, 12 gameweeks, cumulative FPL rows: prem's feed shows assists in GW2/4/6/8/10, FPL also
    awards one in GW11 the feed lacks (6 against 5); goals 6 on xG 5.0, xA 3.0."""
    for team in ("team_a", "team_b"):
        con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES (?, ?)", [team, team])
    for uid, pos in (("prem", "Midfielder"), ("cheap", "Defender")):
        con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, ?)", [uid, uid, pos])
    start = datetime(2024, 8, 17)
    for gw in range(1, 21):
        mid = f"p{gw}"
        _match(con, mid, PRIOR, gw, start + timedelta(days=7 * gw))
        _played(con, "prem", mid, PRIOR, goals=int(gw <= 10), assists=int(gw <= 5))
        _played(con, "cheap", mid, PRIOR, assists=int(gw == 1))
    _row(con, "prem", PRIOR, 20, 10.0, expected_goals_per_90=0.4, expected_assists_per_90=0.2)
    _row(con, "cheap", PRIOR, 20, 4.5, expected_goals_per_90=0.05, expected_assists_per_90=0.05)
    start = datetime(2025, 8, 16)
    fpl_assist_gws = {2, 4, 6, 8, 10, 11}
    cum_assists = 0
    for gw in range(1, 13):
        mid = f"s{gw}"
        _match(con, mid, SEASON, gw, start + timedelta(days=7 * gw))
        feed_assist = int(gw in fpl_assist_gws and gw != 11)
        _played(con, "prem", mid, SEASON, goals=int(gw <= 6), assists=feed_assist)
        _played(con, "cheap", mid, SEASON)
        cum_assists += int(gw in fpl_assist_gws)
        _row(con, "prem", SEASON, gw, 10.0, minutes=90 * gw, goals_scored=min(gw, 6), assists=cum_assists,
             expected_goals=5.0 * gw / 12, expected_assists=3.0 * gw / 12)
        _row(con, "cheap", SEASON, gw, 4.5, minutes=90 * gw, goals_scored=0, assists=0,
             expected_goals=0.1 * gw / 12, expected_assists=0.1 * gw / 12)


def test_assist_sources_counts_fpl_against_the_feed(con):
    _seed(con)
    out = diag.assist_sources(con, SEASON)
    assert out["9.0+"] == {"n_players": 1, "fpl_assists": 6.0, "feed_assists": 5.0, "n_differ": 1, "fpl_per_feed": 1.2}
    assert out["<5.0"]["n_players"] == 1 and out["<5.0"]["fpl_per_feed"] is None


def test_conversion_reads_fpl_totals_and_rebuilds_2024_25_from_the_feed(con):
    _seed(con)
    now = diag.conversion(con, SEASON)
    assert now["source"] == "fpl"
    prem = now["bands"]["9.0+"]
    assert (prem["goals_per_xg"], prem["assists_per_xa"]) == (1.2, 2.0)
    assert prem["by_position"]["Midfielder"]["n_players"] == 1
    # 2024-25: NULL totals; goals and assists from the feed, xG/xA = per-90 x 1800 feed minutes
    before = diag.conversion(con, PRIOR)
    assert before["source"] == "match feed"
    prem = before["bands"]["9.0+"]
    assert (prem["goals"], prem["xg"], prem["goals_per_xg"]) == (10.0, 8.0, 1.25)
    assert (prem["assists"], prem["xa"], prem["assists_per_xa"]) == (5.0, 4.0, 1.25)


def test_assist_alignment_scores_each_played_gameweek(con):
    _seed(con)
    out = diag.assist_alignment(con, SEASON)
    assert out["players_whose_differences_miss_the_total"] == 0 and out["negative_differences"] == 0
    # 12 played gameweeks at 9.0m+, GW11's FPL-only assist the one disagreement
    assert out["9.0+"] == {"player_gameweeks": 12, "agree": 11, "share": 0.9167}
    assert out["9.0+_fpl_one_gw_later"]["agree"] < 11
    assert out["passes"] is False


def test_verdicts(con):
    sources = {SEASON: {"9.0+": {"fpl_per_feed": 1.2}}, "2026-2027": {"9.0+": {"fpl_per_feed": 1.02}}}
    alignment = {SEASON: {"passes": True}, "2026-2027": {"passes": True}}
    out = diag.verdicts(sources, alignment)
    assert out[SEASON] == {"9.0+_fpl_per_feed": 1.2, "h3_killed": False, "rq2b_applies": True}
    assert out["2026-2027"]["h3_killed"] is True and out["2026-2027"]["rq2b_applies"] is False
    alignment[SEASON]["passes"] = False
    assert diag.verdicts(sources, alignment)[SEASON]["rq2b_applies"] is False


def test_bonus_columns_found_in_raw_tables(con):
    for gw in (1, 2):
        table = f"raw_2025_2026_by_gameweek_gw{gw}_playerstats_csv"
        con.execute(f'CREATE TABLE "{table}" (id VARCHAR, bonus VARCHAR, bps VARCHAR)')
        con.execute(
            "INSERT INTO fact_raw_ingestion_log (raw_table_name, season, source_relpath, source_file_hash, row_count) "
            "VALUES (?, ?, ?, ?, 0)", [table, SEASON, f"By Gameweek/GW{gw}/playerstats.csv", f"h{gw}"],
        )
    con.execute('CREATE TABLE "raw_2025_2026_teams_csv" (id VARCHAR)')
    con.execute(
        "INSERT INTO fact_raw_ingestion_log (raw_table_name, season, source_relpath, source_file_hash, row_count) "
        "VALUES ('raw_2025_2026_teams_csv', ?, 'teams.csv', 'h3', 0)", [SEASON],
    )
    out = diag.bonus_columns(con)
    assert out == {"tables_checked": 3, "with_bonus": {
        f"{SEASON}:By Gameweek/GW*/playerstats.csv": {"columns": ["bonus"], "n_tables": 2},
    }}


def test_form_group_sizes_count_the_steps_the_judge_sees(con):
    tfg._seed(con)
    con.execute("UPDATE fact_player_season_stats SET event_points = 2 WHERE season = ?", [SEASON])
    out = diag.form_group_sizes(con, SEASON)
    # only GW5 qualifies (GW4's row is the first with 360 minutes); warm joins in the widening
    assert out["player_steps"] == {"in_form": 1, "in_form_widened": 2, "cooling": 1, "cooling_widened": 1}
    assert out["judged_on"] == "too_small" and out["gameweeks"] == 5
    # a missing event_points row hides the step, as it does from the walk-forward judge
    con.execute("UPDATE fact_player_season_stats SET event_points = NULL WHERE player_uid = 'hot' AND gw = 5")
    assert diag.form_group_sizes(con, SEASON)["player_steps"]["in_form"] == 0
    assert diag.form_group_sizes(con, PRIOR) is None
