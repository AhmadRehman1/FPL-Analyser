"""Regression: FPL-Core-Insights periodically regenerates a *historical* season's root
players.csv (and its later per-gameweek copies, and the playermatchstats.csv match
attribution) from a CURRENT bootstrap, so a since-transferred player is retroactively written
onto their new club for a season they never played there -- 2025-2026's root lists Isak at
Liverpool though he was at Newcastle all season. reconcile.suspect_transfer_player_seasons()
detects that (root club != earliest-gameweek club) and minutes_model excludes the affected
(player, season) from the historical start-rate fit rather than measuring it against the
wrong club's fixtures.
"""

from pathlib import Path

from fpl_quant import ingest_csv, minutes_model, reconcile

PLAYERS_HEADER = "player_code,player_id,first_name,second_name,web_name,team_code,position"


def _write(path: Path, header: str, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")


def _ingest_players(con, tmp_path, season, relpath, rows):
    p = tmp_path / season / relpath
    _write(p, PLAYERS_HEADER, rows)
    ingest_csv.ingest_csv_file(con, season, relpath, p)


def test_detects_a_player_whose_prior_season_club_was_rewritten(con, tmp_path):
    # 2025-2026: GW1 snapshot (pre-rewrite) has Isak at 4 (Newcastle); the season-root file
    # (regenerated after his 2026 move) has him at 14 (Liverpool). Ordinary teammate Bruno
    # is at 4 in both.
    _ingest_players(con, tmp_path, "2025-2026", "By Gameweek/GW1/players.csv", [
        "p_isak,9,Alexander,Isak,Isak,4,Forward",
        "p_bruno,10,Bruno,Guimaraes,Bruno G.,4,Midfielder",
    ])
    _ingest_players(con, tmp_path, "2025-2026", "players.csv", [
        "p_isak,9,Alexander,Isak,Isak,14,Forward",
        "p_bruno,10,Bruno,Guimaraes,Bruno G.,4,Midfielder",
    ])
    _ingest_players(con, tmp_path, "2026-2027", "players.csv", [
        "p_isak,9,Alexander,Isak,Isak,14,Forward",
        "p_bruno,10,Bruno,Guimaraes,Bruno G.,4,Midfielder",
    ])
    reconcile.build_dim_player(con)

    suspect = reconcile.suspect_transfer_player_seasons(con, target_season="2026-2027")

    isak_uid = reconcile.er.player_uid_for("Alexander Isak")
    bruno_uid = reconcile.er.player_uid_for("Bruno Guimaraes")
    assert (isak_uid, "2025-2026") in suspect
    assert (bruno_uid, "2025-2026") not in suspect


def test_target_season_is_never_flagged(con, tmp_path):
    # Even if 2026-2027's own root disagrees with its GW1 snapshot, the root is the freshest
    # correct roster for the current season -- flagging it would drop live data.
    _ingest_players(con, tmp_path, "2026-2027", "By Gameweek/GW1/players.csv", [
        "p_x,7,Some,Player,Player,4,Midfielder",
    ])
    _ingest_players(con, tmp_path, "2026-2027", "players.csv", [
        "p_x,7,Some,Player,Player,14,Midfielder",
    ])
    reconcile.build_dim_player(con)

    assert reconcile.suspect_transfer_player_seasons(con, target_season="2026-2027") == set()


def test_season_without_per_gameweek_rosters_is_skipped_not_crashed(con, tmp_path):
    # 2024-2025's source layout has no By Gameweek/ roster files -- nothing to cross-check.
    _ingest_players(con, tmp_path, "2024-2025", "players/players.csv", [
        "p_a,1,A,Player,A. Player,3,Defender",
    ])
    reconcile.build_dim_player(con)

    assert reconcile.suspect_transfer_player_seasons(con, target_season="2026-2027") == set()


def test_compute_historical_components_drops_excluded_player_seasons(con):
    # Team A plays one PL match per season; p1 and p2 both start it in both seasons.
    # Excluding p1's 2025-2026 leaves only its 2024-2025 half in the fit; p2 keeps both.
    from datetime import date, datetime, timezone

    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('team_a','A'), ('team_b','B')")
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES "
                "('p1','P One','Forward'), ('p2','P Two','Forward')")
    now = datetime.now(timezone.utc)
    for i, season in enumerate(("2024-2025", "2025-2026")):
        table = f"raw_{season.replace('-', '_')}_teams"
        con.execute(f'CREATE TABLE "{table}" (code VARCHAR, name VARCHAR)')
        con.execute(f'INSERT INTO "{table}" VALUES (\'1\', \'A\'), (\'2\', \'B\')')
        con.execute("INSERT INTO fact_raw_ingestion_log (raw_table_name, season, source_relpath, "
                    "source_file_hash, row_count) VALUES (?, ?, 'teams.csv', ?, 2)", [table, season, table])
        con.execute("INSERT INTO team_alias (alias_name, season, team_uid, alias_source) VALUES "
                    "('A', ?, 'team_a', 't'), ('B', ?, 'team_b', 't')", [season, season])
        for uid in ("p1", "p2"):
            con.execute("INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, "
                        "player_uid) VALUES (?, ?, '1', ?, ?)", [uid, uid, season, uid])
        mid = f"m{i}"
        con.execute("INSERT INTO fact_match (match_id, season, home_team_uid, away_team_uid, finished, "
                    "competition, kickoff_time, _ingested_at) VALUES (?, ?, 'team_a', 'team_b', TRUE, "
                    "'Premier League', ?, ?)", [mid, season, datetime(2025 + i, 1, 1), now])
        for uid in ("p1", "p2"):
            con.execute("INSERT INTO fact_player_match_stats (player_uid, match_id, season, start_min, "
                        "finish_min, minutes_played, _ingested_at) VALUES (?, ?, ?, 0, 90, 90, ?)",
                        [uid, mid, season, now])

    base = minutes_model.compute_player_historical_components(
        con, ("2024-2025", "2025-2026"), date(2026, 8, 1), 0.0,
    ).set_index("player_uid")
    assert base.loc["p1", "raw_team_matches"] == 2
    assert base.loc["p2", "raw_team_matches"] == 2

    dropped = minutes_model.compute_player_historical_components(
        con, ("2024-2025", "2025-2026"), date(2026, 8, 1), 0.0,
        exclude_player_seasons={("p1", "2025-2026")},
    ).set_index("player_uid")
    assert dropped.loc["p1", "raw_team_matches"] == 1  # 2025-2026 half removed
    assert dropped.loc["p2", "raw_team_matches"] == 2  # untouched
