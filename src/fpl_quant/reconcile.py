"""Builds fact_reconciled from fact_raw: entity resolution onto stable UIDs, match_id
dedup, and column-semantics tagging. Reads only from fact_raw_* tables (never re-reads
the original CSVs directly) so the three-layer boundary is real, not just documented.

Known, verified doc-vs-data discrepancy: the source README states matches.csv's
home_team/away_team link to teams.id, but real data shows they carry teams.code
(confirmed: Arsenal code=3/id=1 appears as home_team='3.0' in a fixture where Arsenal
is listed as the home side). This module joins on code, not id, per the verified data.
"""

import re
from datetime import date, datetime, timezone

import duckdb
import openpyxl

from . import entity_resolution as er

SEASONS = ["2024-2025", "2025-2026", "2026-2027"]


def _tables_matching(con: duckdb.DuckDBPyConnection, season: str, like_pattern: str):
    return con.execute(
        "SELECT DISTINCT source_relpath, raw_table_name FROM fact_raw_ingestion_log "
        "WHERE season = ? AND source_relpath LIKE ? ORDER BY 1",
        [season, like_pattern],
    ).fetchall()


def _season_root_table(con: duckdb.DuckDBPyConnection, season: str, filename: str):
    """Finds a season-level master file (teams.csv, players.csv, playerstats.csv).

    2025-2026 and 2026-2027 keep these at the season root (relpath == filename).
    2024-2025 nests them one level down under a same-named directory instead
    (relpath == 'teams/teams.csv', 'players/players.csv', ...) -- a real layout
    difference in the source repo, not a typo. An exact-match-only lookup silently
    returns nothing for 2024-2025 and drops that entire season; try both.
    """
    for relpath in (filename, f"{filename[:-4]}/{filename}"):
        rows = _tables_matching(con, season, relpath)
        if rows:
            return rows[0]
    return None


def _ensure_id_macro(con: duckdb.DuckDBPyConnection) -> None:
    # Raw numeric-ish ID columns sometimes arrive float-formatted ('3.0') from the CSV
    # pipeline; normalize before joining so a format quirk never silently drops a join.
    con.execute(
        """
        CREATE OR REPLACE MACRO norm_id(x) AS
            CASE WHEN x IS NULL OR trim(x) = '' THEN NULL
                 ELSE CAST(CAST(TRY_CAST(x AS DOUBLE) AS BIGINT) AS VARCHAR)
            END
        """
    )


# ---------------------------------------------------------------- teams ----

def build_dim_team(con: duckdb.DuckDBPyConnection) -> None:
    _ensure_id_macro(con)
    con.execute(
        "CREATE OR REPLACE TABLE _team_code_map (season VARCHAR, code VARCHAR, team_uid VARCHAR)"
    )
    con.execute(
        "CREATE OR REPLACE TABLE _team_id_map (season VARCHAR, team_id_local VARCHAR, team_uid VARCHAR)"
    )

    for season in SEASONS:
        found = _season_root_table(con, season, "teams.csv")
        if not found:
            continue
        _relpath, table = found
        data = con.execute(f'SELECT code, id, name, short_name FROM "{table}"').fetchall()
        for code, local_id, name, short_name in data:
            uid = er.team_uid_for(name)
            # ON CONFLICT targets team_uid (the actual PK), not canonical_name: two literally
            # different name spellings can normalize to the same uid (see dim_player below for
            # a real example of this happening), and that's the collision that must be caught.
            con.execute(
                "INSERT INTO dim_team (team_uid, canonical_name, short_name, prior_division) "
                "VALUES (?, ?, ?, NULL) ON CONFLICT (team_uid) DO NOTHING",
                [uid, name, short_name],
            )
            con.execute(
                "INSERT INTO team_alias (alias_name, season, team_uid, alias_source) "
                "VALUES (?, ?, ?, ?) ON CONFLICT (alias_name, season) DO NOTHING",
                [name, season, uid, f"{season}:teams.csv"],
            )
            con.execute(
                "INSERT INTO _team_code_map VALUES (?, ?, ?)", [season, str(code), uid]
            )
            con.execute(
                "INSERT INTO _team_id_map VALUES (?, ?, ?)", [season, str(local_id), uid]
            )


def apply_club_name_map(con: duckdb.DuckDBPyConnection, xlsx_path: str) -> int:
    """26_Club Name Map: alias rows for club-name variants across workbook tabs, applied
    across every season we have team data for (the map itself isn't season-specific)."""
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb["26_Club Name Map"]
    added = 0
    for row in ws.iter_rows(min_row=2, values_only=True):
        alias_name, official_name = row[0], row[1]
        if not alias_name or not official_name:
            continue
        match = con.execute(
            "SELECT team_uid FROM dim_team WHERE canonical_name = ?", [official_name]
        ).fetchone()
        if not match:
            continue
        uid = match[0]
        for season in SEASONS:
            con.execute(
                "INSERT INTO team_alias (alias_name, season, team_uid, alias_source) "
                "VALUES (?, ?, ?, ?) ON CONFLICT (alias_name, season) DO NOTHING",
                [alias_name, season, uid, "26_Club Name Map"],
            )
            added += 1
    return added


# -------------------------------------------------------------- players ----

def build_dim_player(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        "CREATE OR REPLACE TABLE _player_id_map (season VARCHAR, player_id_local VARCHAR, player_uid VARCHAR)"
    )
    for season in SEASONS:
        found = _season_root_table(con, season, "players.csv")
        if not found:
            continue
        _relpath, table = found
        data = con.execute(
            f'SELECT player_code, player_id, first_name, second_name, web_name, team_code, position FROM "{table}"'
        ).fetchall()
        for player_code, player_id, first_name, second_name, web_name, team_code, position in data:
            full_name = f"{first_name} {second_name}".strip()
            if not full_name:
                full_name = web_name
            uid = er.player_uid_for(full_name)
            # ON CONFLICT targets player_uid, not canonical_name: real example found in this
            # data -- "Aaron Anselmino" (2024-25, 2026-27) vs "Aarón Anselmino" (2025-26,
            # accented) are two different literal strings that correctly normalize to the same
            # uid. That's the intended alias-variant case (M0's "manual alias table for
            # renames/accented-name variants"), not an error -- the dim_player row is written
            # once under whichever spelling is seen first, and every season's alias row still
            # gets recorded below regardless of which spelling produced the canonical row.
            con.execute(
                "INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, ?) "
                "ON CONFLICT (player_uid) DO NOTHING",
                [uid, full_name, position],
            )
            for alias_name in {full_name, web_name}:
                if not alias_name:
                    continue
                con.execute(
                    "INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, "
                    "player_uid, source_player_id) VALUES (?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT (alias_name, team_code, season) DO NOTHING",
                    [alias_name, er.normalize_name(alias_name), str(team_code), season, uid, str(player_code)],
                )
            con.execute(
                "INSERT INTO _player_id_map VALUES (?, ?, ?)", [season, str(player_id), uid]
            )


# --------------------------------------------------------------- matches ----

_COMPETITION_RE = re.compile(r"By Tournament/([^/]+)/GW\d+/")


def _competition_from_relpath(season: str, relpath: str) -> str:
    m = _COMPETITION_RE.search(relpath)
    if m:
        return m.group(1)
    return "Premier League"  # 2024-2025's flat matches/GW{n}/ structure is PL-only


def build_fact_match(con: duckdb.DuckDBPyConnection) -> int:
    _ensure_id_macro(con)
    parts = []
    for season in SEASONS:
        if season == "2024-2025":
            file_sets = [("matches.csv", 1, "matches/GW%/matches.csv")]
        else:
            file_sets = [
                ("matches.csv", 1, "By Tournament/%/GW%/matches.csv"),
                ("fixtures.csv", 2, "By Tournament/%/GW%/fixtures.csv"),
            ]
        for _fname, priority, pattern in file_sets:
            for relpath, table in _tables_matching(con, season, pattern):
                competition = _competition_from_relpath(season, relpath)
                comp_sql = competition.replace("'", "''")
                season_sql = season.replace("'", "''")
                parts.append(
                    f"""
                    SELECT
                        match_id,
                        '{season_sql}' AS season,
                        TRY_CAST(gameweek AS INTEGER) AS gameweek,
                        TRY_CAST(kickoff_time AS TIMESTAMP) AS kickoff_time,
                        norm_id(home_team) AS home_code,
                        norm_id(away_team) AS away_code,
                        TRY_CAST(NULLIF(home_score, '') AS INTEGER) AS home_score,
                        TRY_CAST(NULLIF(away_score, '') AS INTEGER) AS away_score,
                        TRY_CAST(NULLIF(home_team_elo, '') AS DOUBLE) AS home_team_elo,
                        TRY_CAST(NULLIF(away_team_elo, '') AS DOUBLE) AS away_team_elo,
                        CASE WHEN lower(finished) IN ('true', '1') THEN TRUE
                             WHEN lower(finished) IN ('false', '0') THEN FALSE
                             ELSE NULL END AS finished,
                        '{comp_sql}' AS competition,
                        {priority} AS _priority
                    FROM "{table}"
                    """
                )
    if not parts:
        return 0

    union_sql = "\nUNION ALL\n".join(parts)
    con.execute(f'CREATE OR REPLACE TEMP TABLE _match_union AS {union_sql}')
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE _match_dedup AS
        SELECT * FROM (
            SELECT *, row_number() OVER (PARTITION BY match_id ORDER BY _priority) AS rn
            FROM _match_union
        ) WHERE rn = 1
        """
    )
    now = datetime.now(timezone.utc)
    con.execute(
        f"""
        INSERT INTO fact_match
            (match_id, season, gameweek, kickoff_time, home_team_uid, away_team_uid,
             home_score, away_score, home_team_elo, away_team_elo, finished, competition, _ingested_at)
        SELECT
            d.match_id, d.season, d.gameweek, d.kickoff_time,
            hc.team_uid, ac.team_uid,
            d.home_score, d.away_score, d.home_team_elo, d.away_team_elo, d.finished, d.competition,
            TIMESTAMP '{now.strftime("%Y-%m-%d %H:%M:%S.%f")}'
        FROM _match_dedup d
        JOIN _team_code_map hc ON hc.season = d.season AND hc.code = d.home_code
        JOIN _team_code_map ac ON ac.season = d.season AND ac.code = d.away_code
        ON CONFLICT (match_id) DO NOTHING
        """
    )
    return con.execute("SELECT count(*) FROM fact_match").fetchone()[0]


# ------------------------------------------------------- player-match stats ----

def build_fact_player_match_stats(con: duckdb.DuckDBPyConnection) -> int:
    _ensure_id_macro(con)
    now = datetime.now(timezone.utc)
    now_sql = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    total = 0
    for season in SEASONS:
        pattern = (
            "playermatchstats/GW%/playermatchstats.csv"
            if season == "2024-2025"
            else "By Tournament/%/GW%/playermatchstats.csv"
        )
        for _relpath, table in _tables_matching(con, season, pattern):
            season_sql = season.replace("'", "''")
            con.execute(
                f"""
                INSERT INTO fact_player_match_stats
                    (player_uid, match_id, season, start_min, finish_min, minutes_played,
                     goals, assists, saves, goals_conceded, team_goals_conceded,
                     tackles, clearances, interceptions, recoveries, blocks, _ingested_at)
                SELECT
                    pm.player_uid, r.match_id, '{season_sql}',
                    TRY_CAST(NULLIF(r.start_min, '') AS INTEGER),
                    TRY_CAST(NULLIF(r.finish_min, '') AS INTEGER),
                    TRY_CAST(NULLIF(r.minutes_played, '') AS INTEGER),
                    TRY_CAST(NULLIF(r.goals, '') AS INTEGER),
                    TRY_CAST(NULLIF(r.assists, '') AS INTEGER),
                    TRY_CAST(NULLIF(r.saves, '') AS INTEGER),
                    TRY_CAST(NULLIF(r.goals_conceded, '') AS INTEGER),
                    TRY_CAST(NULLIF(r.team_goals_conceded, '') AS INTEGER),
                    TRY_CAST(NULLIF(r.tackles, '') AS INTEGER),
                    TRY_CAST(NULLIF(r.clearances, '') AS INTEGER),
                    TRY_CAST(NULLIF(r.interceptions, '') AS INTEGER),
                    TRY_CAST(NULLIF(r.recoveries, '') AS INTEGER),
                    TRY_CAST(NULLIF(r.blocks, '') AS INTEGER),
                    TIMESTAMP '{now_sql}'
                FROM "{table}" r
                JOIN _player_id_map pm ON pm.season = '{season_sql}' AND pm.player_id_local = norm_id(r.player_id)
                WHERE r.match_id IN (SELECT match_id FROM fact_match)
                ON CONFLICT (player_uid, match_id) DO NOTHING
                """
            )
    total = con.execute("SELECT count(*) FROM fact_player_match_stats").fetchone()[0]
    return total


# ------------------------------------------------------- player-season stats ----

_SEASON_STATS_NUMERIC_COLS = {
    "now_cost": "DOUBLE", "selected_by_percent": "DOUBLE", "ep_next": "DOUBLE",
    "chance_of_playing_next_round": "DOUBLE", "minutes": "INTEGER", "goals_scored": "INTEGER",
    "assists": "INTEGER", "bps": "INTEGER", "expected_goals": "DOUBLE", "expected_assists": "DOUBLE",
    "expected_goals_per_90": "DOUBLE", "expected_assists_per_90": "DOUBLE",
    "defensive_contribution": "DOUBLE", "defensive_contribution_per_90": "DOUBLE",
    "saves_per_90": "DOUBLE", "total_points": "INTEGER", "event_points": "INTEGER",
}


def _existing_columns(con: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    return {
        r[0] for r in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = ?", [table]
        ).fetchall()
    }


def build_fact_player_season_stats(con: duckdb.DuckDBPyConnection) -> int:
    _ensure_id_macro(con)
    now = datetime.now(timezone.utc)
    now_sql = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    for season in SEASONS:
        found = _season_root_table(con, season, "playerstats.csv")
        if not found:
            continue
        _relpath, table = found
        season_sql = season.replace("'", "''")

        # 2024-2025's playerstats.csv genuinely predates several columns 2025-2026+ has
        # (minutes, goals_scored, defensive_contribution, ... -- the source repo added
        # CBIT/DefCon tracking starting 2025-26). A column missing here isn't a NULLIF-able
        # empty string, the column doesn't exist in the table at all -- select NULL for it
        # rather than hardcoding one column list across seasons with genuinely different schemas.
        available = _existing_columns(con, table)
        select_exprs = []
        for col, sql_type in _SEASON_STATS_NUMERIC_COLS.items():
            if col in available:
                select_exprs.append(f"TRY_CAST(NULLIF(r.{col}, '') AS {sql_type})")
            else:
                select_exprs.append(f"CAST(NULL AS {sql_type})")
        status_expr = "r.status" if "status" in available else "CAST(NULL AS VARCHAR)"

        con.execute(
            f"""
            INSERT INTO fact_player_season_stats
                (player_uid, season, gw, now_cost, selected_by_percent, ep_next,
                 chance_of_playing_next_round, status, minutes, goals_scored, assists, bps,
                 expected_goals, expected_assists, expected_goals_per_90, expected_assists_per_90,
                 defensive_contribution, defensive_contribution_per_90, saves_per_90,
                 total_points, event_points, _ingested_at)
            SELECT
                pm.player_uid, '{season_sql}', TRY_CAST(r.gw AS INTEGER),
                -- unlike the standard FPL API convention (tenths, e.g. 155 => 15.5m),
                -- this dataset's now_cost is already decimal pounds-millions (verified:
                -- Haaland's raw now_cost is the literal string '15.5') -- no /10 here.
                {select_exprs[0]}, {select_exprs[1]}, {select_exprs[2]}, {select_exprs[3]},
                {status_expr},
                {select_exprs[4]}, {select_exprs[5]}, {select_exprs[6]}, {select_exprs[7]},
                {select_exprs[8]}, {select_exprs[9]}, {select_exprs[10]}, {select_exprs[11]},
                {select_exprs[12]}, {select_exprs[13]}, {select_exprs[14]},
                {select_exprs[15]}, {select_exprs[16]},
                TIMESTAMP '{now_sql}'
            FROM "{table}" r
            JOIN _player_id_map pm ON pm.season = '{season_sql}' AND pm.player_id_local = norm_id(r.id)
            WHERE TRY_CAST(r.gw AS INTEGER) IS NOT NULL
            ON CONFLICT (player_uid, season, gw) DO NOTHING
            """
        )
    return con.execute("SELECT count(*) FROM fact_player_season_stats").fetchone()[0]


# ------------------------------------------------------------- semantics ----

_COLUMN_SEMANTICS = [
    ("fact_player_season_stats", "now_cost", "live", "current price, not a running total"),
    ("fact_player_season_stats", "selected_by_percent", "live", None),
    ("fact_player_season_stats", "ep_next", "live", None),
    ("fact_player_season_stats", "chance_of_playing_next_round", "live", None),
    ("fact_player_season_stats", "status", "live", None),
    ("fact_player_season_stats", "minutes", "cumulative_to_date", "zero, not null, pre-season"),
    ("fact_player_season_stats", "goals_scored", "cumulative_to_date", None),
    ("fact_player_season_stats", "assists", "cumulative_to_date", None),
    ("fact_player_season_stats", "bps", "cumulative_to_date", None),
    ("fact_player_season_stats", "expected_goals", "cumulative_to_date", None),
    ("fact_player_season_stats", "expected_assists", "cumulative_to_date", None),
    ("fact_player_season_stats", "expected_goals_per_90", "cumulative_to_date", "0/0 pre-season"),
    ("fact_player_season_stats", "expected_assists_per_90", "cumulative_to_date", "0/0 pre-season"),
    ("fact_player_season_stats", "defensive_contribution", "cumulative_to_date", None),
    ("fact_player_season_stats", "defensive_contribution_per_90", "cumulative_to_date", None),
    ("fact_player_season_stats", "saves_per_90", "cumulative_to_date", None),
    ("fact_player_season_stats", "total_points", "cumulative_to_date", None),
    (
        "fact_player_season_stats",
        "event_points",
        "live",
        "closest available tag; actually a single-gameweek delta, not a running total nor a current-state snapshot",
    ),
]


def seed_column_semantics(con: duckdb.DuckDBPyConnection) -> None:
    for table_name, column_name, semantics, notes in _COLUMN_SEMANTICS:
        con.execute(
            "INSERT INTO fact_reconciled_column_semantics (table_name, column_name, semantics, notes) "
            "VALUES (?, ?, ?, ?) ON CONFLICT (table_name, column_name) DO NOTHING",
            [table_name, column_name, semantics, notes],
        )


# ------------------------------------------------------------- orchestrator ----

def reconcile_all(con: duckdb.DuckDBPyConnection, xlsx_path: str) -> dict:
    build_dim_team(con)
    club_aliases = apply_club_name_map(con, xlsx_path)
    build_dim_player(con)
    n_matches = build_fact_match(con)
    n_pms = build_fact_player_match_stats(con)
    n_pss = build_fact_player_season_stats(con)
    seed_column_semantics(con)
    return {
        "teams": con.execute("SELECT count(*) FROM dim_team").fetchone()[0],
        "team_aliases": con.execute("SELECT count(*) FROM team_alias").fetchone()[0],
        "club_name_map_aliases_applied": club_aliases,
        "players": con.execute("SELECT count(*) FROM dim_player").fetchone()[0],
        "player_aliases": con.execute("SELECT count(*) FROM player_alias").fetchone()[0],
        "matches": n_matches,
        "player_match_stats": n_pms,
        "player_season_stats": n_pss,
    }
