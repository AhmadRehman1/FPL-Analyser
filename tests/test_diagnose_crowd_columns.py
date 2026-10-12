"""scripts/diagnose_crowd_columns.py: the plan's A3 coverage and A2 knowability checks
(docs/plans/2026-10_beat_the_template.md)."""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import diagnose_crowd_columns as diag  # noqa: E402

SEASON = "2026-2027"
HEADER = "id,gw,status,selected_by_percent,transfers_in_event,transfers_out_event,now_cost\n"


def _commit(repo: Path, text: str, when: str) -> None:
    path = repo / "data" / SEASON / "playerstats.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER + text)
    env = {**os.environ, "GIT_AUTHOR_DATE": when, "GIT_COMMITTER_DATE": when,
           "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, env=env)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", when], check=True, env=env)


def _seed_matches(con, kickoffs):
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('a', 'A'), ('b', 'B')")
    for gw, kickoff in kickoffs.items():
        con.execute(
            "INSERT INTO fact_match (match_id, season, gameweek, home_team_uid, away_team_uid, finished, competition, "
            "kickoff_time, _ingested_at) VALUES (?, ?, ?, 'a', 'b', TRUE, 'Premier League', ?, current_timestamp)",
            [f"m{gw}", SEASON, gw, kickoff],
        )


def test_a2_holds_when_row_g_minus_1_is_settled_before_the_deadline(con, tmp_path):
    repo = tmp_path / "src"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    # GW2 deadline 2026-08-22 13:30 (kickoff 15:00), GW3 deadline 2026-08-29 13:30
    _commit(repo, "1,1,a,10.0,100,5,5.0\n2,1,i,20.0,0,50,7.0\n", "2026-08-20T10:00:00+00:00")
    _commit(repo, "1,1,a,10.0,100,5,5.0\n2,1,i,20.0,0,50,7.0\n1,2,a,12.0,30,1,5.0\n2,2,s,18.0,0,9,7.0\n",
            "2026-08-27T10:00:00+00:00")
    # after GW3's deadline, row 2 changes: a late status flip would be a leak
    _commit(repo, "1,1,a,10.0,100,5,5.0\n2,1,i,20.0,0,50,7.0\n1,2,a,12.0,30,1,5.0\n2,2,a,18.0,0,9,7.0\n",
            "2026-08-29T14:00:00+00:00")
    _seed_matches(con, {2: datetime(2026, 8, 22, 15), 3: datetime(2026, 8, 29, 15)})
    out = diag.knowability(con, repo, SEASON)
    by_gw = {g["gw"]: g for g in out["gameweeks"]}
    assert by_gw[2]["match_share"]["status"] == 1.0
    assert by_gw[3]["match_share"]["status"] == 0.5  # player 2: 's' before the deadline, 'a' now
    assert by_gw[3]["match_share"]["selected_by_percent"] == 1.0
    assert out["n_judged"] == 2 and out["verdict"] == "inconclusive"


def test_a2_needs_two_gameweeks_with_a_snapshot(con, tmp_path):
    repo = tmp_path / "src"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    _commit(repo, "1,1,a,10.0,100,5,5.0\n1,2,a,12.0,30,1,5.0\n", "2026-08-27T10:00:00+00:00")
    _seed_matches(con, {2: datetime(2026, 8, 22, 15), 3: datetime(2026, 8, 29, 15)})
    out = diag.knowability(con, repo, SEASON)
    assert [g["commit"] is None for g in out["gameweeks"]] == [True, False]
    assert out["n_judged"] == 1 and out["verdict"] == "inconclusive"
    _commit(repo, "1,1,a,10.0,100,5,5.0\n1,2,a,12.0,30,1,5.0\n1,3,a,1.0,0,0,5.0\n", "2026-09-04T10:00:00+00:00")
    con.execute("INSERT INTO fact_match (match_id, season, gameweek, home_team_uid, away_team_uid, finished, competition, "
                "kickoff_time, _ingested_at) VALUES ('m4', ?, 4, 'a', 'b', TRUE, 'Premier League', ?, current_timestamp)",
                [SEASON, datetime(2026, 9, 5, 15)])
    assert diag.knowability(con, repo, SEASON)["verdict"] == "holds"


def test_compare_snapshots_treats_missing_as_equal_and_absent_players_as_mismatches():
    final = {"1": {"status": "a", "selected_by_percent": "5.0"}, "2": {"status": "", "selected_by_percent": "1"}}
    before = {"2": {"status": None, "selected_by_percent": "1.0"}}
    out = diag.compare_snapshots(before, final)
    assert out["n_players"] == 2 and out["n_in_snapshot"] == 1
    assert out["match_share"]["status"] == 0.5 and out["match_share"]["selected_by_percent"] == 0.5


def test_coverage_counts_each_column_by_season(con):
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p', 'P', 'Forward')")
    for gw, status, sbp in ((1, "a", 3.0), (2, None, 4.0)):
        con.execute("INSERT INTO fact_player_season_stats (player_uid, season, gw, status, selected_by_percent, "
                    "_ingested_at) VALUES ('p', '2024-2025', ?, ?, ?, current_timestamp)", [gw, status, sbp])
    out = diag.coverage(con)["2024-2025"]
    assert out["rows"] == 2
    assert out["columns"]["status"] == {"non_null": 1, "share": 0.5, "gameweeks": 1}
    assert out["columns"]["transfers_in_event"]["non_null"] == 0
    assert out["status_codes"] == {"(null)": 1, "a": 1}
