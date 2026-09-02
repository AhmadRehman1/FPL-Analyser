"""B1 + B2 fitness handling in the minutes model.

B1: p_start_historical_own is a "starts-when-available" role signal -- a team match the
player sat out injured/suspended (fact_player_season_stats.status in ('i','s'), 0 minutes) is
dropped from the fit, not counted as a non-start. Before, a nailed starter who missed a
stretch injured (Chris Wood ~0.43) was permanently down-rated.

B2: the live FPL chance_of_playing_next_round / status flag is applied in run() as a
forward-looking availability multiplier -- before this the minutes model had no automatic
injury signal at all.
"""

from datetime import date, datetime, timezone

from fpl_quant import minutes_model as mm
from fpl_quant import params

NOW = datetime(2026, 8, 1, tzinfo=timezone.utc)


def _seed_team_and_player(con, uid, name, position, team_code="1"):
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, ?)", [uid, name, position])
    for season in ("2024-2025", "2025-2026"):
        con.execute(
            "INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) VALUES (?, ?, ?, ?, ?)",
            [name, name.lower(), team_code, season, uid],
        )


def _seed_league_with_gameweeks(con):
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('team_a','A'), ('team_b','B')")
    for season in ("2024-2025", "2025-2026"):
        table = f"raw_{season.replace('-', '_')}_teams"
        con.execute(f'CREATE TABLE "{table}" (code VARCHAR, name VARCHAR)')
        con.execute(f'INSERT INTO "{table}" VALUES (\'1\', \'A\'), (\'2\', \'B\')')
        con.execute("INSERT INTO fact_raw_ingestion_log (raw_table_name, season, source_relpath, source_file_hash, row_count) "
                    "VALUES (?, ?, 'teams.csv', ?, 2)", [table, season, table])
        con.execute("INSERT INTO team_alias (alias_name, season, team_uid, alias_source) VALUES "
                    "('A', ?, 'team_a', 't'), ('B', ?, 'team_b', 't')", [season, season])
        for gw in range(1, 11):
            con.execute(
                "INSERT INTO fact_match (match_id, season, gameweek, home_team_uid, away_team_uid, finished, "
                "competition, kickoff_time, _ingested_at) VALUES (?, ?, ?, 'team_a', 'team_b', TRUE, "
                "'Premier League', ?, ?)",
                [f"{season}-m{gw}", season, gw, datetime(2025 if season == "2024-2025" else 2026, 1, gw), NOW],
            )


def _played(con, uid, season, gw, minutes=90, start_min=0):
    con.execute(
        "INSERT INTO fact_player_match_stats (player_uid, match_id, season, start_min, finish_min, minutes_played, _ingested_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [uid, f"{season}-m{gw}", season, start_min, start_min + minutes, minutes, NOW],
    )


def _season_stat(con, uid, season, gw, status="a", chance=None):
    con.execute(
        "INSERT INTO fact_player_season_stats (player_uid, season, gw, status, chance_of_playing_next_round, _ingested_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [uid, season, gw, status, chance, NOW],
    )


# ---- B1: injured matches drop out of the historical fit -----------------------

def test_injured_stretch_does_not_depress_start_rate(con):
    _seed_league_with_gameweeks(con)
    _seed_team_and_player(con, "wood", "Wood", "Forward")
    # 2024-25: started all 10. 2025-26: started GW1-4, then injured GW5-10 (0 min, status 'i').
    for gw in range(1, 11):
        _played(con, "wood", "2024-2025", gw)
        _season_stat(con, "wood", "2024-2025", gw, status="a")
    for gw in range(1, 5):
        _played(con, "wood", "2025-2026", gw)
        _season_stat(con, "wood", "2025-2026", gw, status="a")
    for gw in range(5, 11):
        _season_stat(con, "wood", "2025-2026", gw, status="i")  # no _played row

    comp = mm.compute_player_historical_components(con, ("2024-2025", "2025-2026"), date(2026, 8, 1), 0.0).set_index("player_uid")
    # 14 real availabilities (10 + 4), all starts -> ~1.0, not 14/20 = 0.7
    assert comp.loc["wood", "raw_team_matches"] == 14
    assert comp.loc["wood", "weighted_starts"] / comp.loc["wood", "weighted_total"] > 0.99


def test_benched_while_fit_still_counts_as_a_non_start(con):
    _seed_league_with_gameweeks(con)
    _seed_team_and_player(con, "sub", "Sub", "Midfielder")
    # fit ('a') every gameweek but only started half -- a genuine rotation player, unchanged.
    for season in ("2024-2025", "2025-2026"):
        for gw in range(1, 11):
            _season_stat(con, "sub", season, gw, status="a")
            if gw <= 5:
                _played(con, "sub", season, gw)

    comp = mm.compute_player_historical_components(con, ("2024-2025", "2025-2026"), date(2026, 8, 1), 0.0).set_index("player_uid")
    assert comp.loc["sub", "raw_team_matches"] == 20
    assert abs(comp.loc["sub", "weighted_starts"] / comp.loc["sub", "weighted_total"] - 0.5) < 1e-6


# ---- B2: live availability gate ----------------------------------------------

def test_live_availability_maps_chance_and_status(con):
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES "
                "('out','O','F'),('doubt','D','F'),('pct','P','F'),('fit','A','F'),('unknown','U','F')")
    _season_stat(con, "out", "2026-2027", 2, status="i", chance=None)
    _season_stat(con, "doubt", "2026-2027", 2, status="d", chance=None)
    _season_stat(con, "pct", "2026-2027", 2, status="d", chance=75)
    _season_stat(con, "fit", "2026-2027", 2, status="a", chance=100)
    _season_stat(con, "unknown", "2026-2027", 2, status="u", chance=None)
    # a later gameweek must win the per-player pick
    _season_stat(con, "out", "2026-2027", 1, status="a", chance=100)

    avail = mm.live_availability_by_player(con, "2026-2027")
    assert avail["out"] == 0.0
    assert avail["doubt"] == 0.5
    assert avail["pct"] == 0.75
    assert "fit" not in avail       # chance 100 -> no gate
    assert "unknown" not in avail   # status 'u' with null chance -> no guess


def test_run_gates_a_flagged_players_p_start_final(con):
    _seed_league_with_gameweeks(con)
    _seed_team_and_player(con, "star", "Star", "Forward")
    con.execute("INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) "
                "VALUES ('Star','star','1','2026-2027','star')")
    for season in ("2024-2025", "2025-2026"):
        for gw in range(1, 11):
            _played(con, "star", season, gw)
            _season_stat(con, "star", season, gw, status="a")
    # current-season flag: 25% chance for the upcoming round
    _season_stat(con, "star", "2026-2027", 1, status="d", chance=25)

    params.write_param(con, "minutes_model_decay_params", 1, "2026-08-10", "xi", value_numeric=0.0)
    params.write_param(con, "minutes_model_shrinkage_params", 1, "2026-08-10", "competitive_matches_threshold", value_numeric=10)
    params.write_param(con, "minutes_adjustment_params", 1, "2026-08-10", "cap", value_numeric=6.0, dimensions={"scope": "global"})

    mv = mm.run(con, date(2026, 8, 10), "2026-2027", decay_params_version=1, adjustment_params_version=1,
                shrinkage_params_version=1, fact_multiplier_params_version=1)
    row = con.execute("SELECT p_start_historical_final, p_start_final, p_0min FROM minutes_model_outputs "
                      "WHERE model_version = ? AND player_uid = 'star'", [mv]).fetchone()
    hist_final, p_start_final, p_0 = row
    assert hist_final > 0.95                    # clean role signal: nailed
    assert abs(p_start_final - 0.25 * hist_final) < 0.02   # gated to ~25%
    assert p_0 > 0.7                            # mostly won't feature
