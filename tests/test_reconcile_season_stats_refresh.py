"""Regression test for a real bug in reconcile.build_fact_player_season_stats(): the source
playerstats.csv refreshes twice daily and every refresh is appended as a brand-new batch into
the same append-only raw table (see ingest_csv.py's own module docstring), so several columns
explicitly tagged "live" (now_cost, status, chance_of_playing_next_round, selected_by_percent,
ep_next -- see reconcile._COLUMN_SEMANTICS) can differ across batches for the same
(player, gw). The previous version of the reconcile query had no ORDER BY/dedup and used
`ON CONFLICT (player_uid, season, gw) DO NOTHING`, so whichever batch happened to be scanned
first for a key won permanently -- a later, more current re-ingestion (e.g. a status flip to
"Injured/Out", or a price change) was silently discarded forever, even on a subsequent
reconcile_all() run. Fixed by deduping to the latest batch per key (QUALIFY ROW_NUMBER() ...
ORDER BY _ingested_at DESC) and using ON CONFLICT ... DO UPDATE so live columns actually
refresh.
"""

from pathlib import Path

from fpl_quant import ingest_csv, reconcile

PLAYERS_HEADER = "player_code,player_id,first_name,second_name,web_name,team_code,position"
STATS_HEADER = "id,gw,now_cost,status,minutes,goals_scored,assists"


def _write_csv(path: Path, header: str, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")


def _ingest_and_reconcile(con, tmp_path, stats_rows: list[str]):
    season = "2026-2027"
    players_path = tmp_path / season / "players.csv"
    stats_path = tmp_path / season / "playerstats.csv"
    _write_csv(players_path, PLAYERS_HEADER, ["101,101,Erling,Haaland,Haaland,43,Forward"])
    _write_csv(stats_path, STATS_HEADER, stats_rows)

    ingest_csv.ingest_csv_file(con, season, "players.csv", players_path)
    ingest_csv.ingest_csv_file(con, season, "playerstats.csv", stats_path)

    reconcile.build_dim_player(con)
    reconcile.build_fact_player_season_stats(con)

    return con.execute(
        "SELECT now_cost, status, minutes FROM fact_player_season_stats "
        "WHERE season = '2026-2027' AND gw = 1"
    ).fetchone()


def test_second_ingestion_batch_refreshes_live_stats(con, tmp_path):
    # First reconcile pass: player is fit, priced at 15.0
    _ingest_and_reconcile(con, tmp_path, ["101,1,15.0,a,90,1,0"])
    now_cost, status, minutes = con.execute(
        "SELECT now_cost, status, minutes FROM fact_player_season_stats WHERE season = '2026-2027' AND gw = 1"
    ).fetchone()
    assert (now_cost, status) == (15.0, "a")

    # A later same-day refresh: price rises and the player picks up an injury flag for the
    # same gameweek row -- this must overwrite the stale snapshot, not be silently discarded.
    now_cost2, status2, minutes2 = _ingest_and_reconcile(con, tmp_path, ["101,1,15.5,i,90,1,0"])
    assert now_cost2 == 15.5, "a later refresh's price change must not be silently discarded"
    assert status2 == "i", "a later refresh's injury-status flip must not be silently discarded"

    # exactly one row for this (player, season, gw) -- refreshed in place, not duplicated
    count = con.execute(
        "SELECT count(*) FROM fact_player_season_stats WHERE season = '2026-2027' AND gw = 1"
    ).fetchone()[0]
    assert count == 1


def test_distinct_gameweeks_are_preserved_as_distinct_rows_not_collapsed(con, tmp_path):
    """The dedup above only ever collapses multiple ingestion BATCHES of the SAME (player, gw)
    -- confirmed here that it never collapses DIFFERENT gws into each other. This is the real
    structural premise price_momentum_by_player() (M8) depends on: whatever price/ownership
    trajectory the real source CSVs genuinely carry across gameweeks is preserved intact, not
    an artifact of this reconcile step itself. Two real, DIFFERENT prices at two different
    gameweeks for the same player -- both must survive, each at its own value."""
    _ingest_and_reconcile(con, tmp_path, ["101,1,15.0,a,90,1,0", "101,2,15.5,a,90,0,1"])
    rows = con.execute(
        "SELECT gw, now_cost FROM fact_player_season_stats WHERE season = '2026-2027' ORDER BY gw"
    ).fetchall()
    assert rows == [(1, 15.0), (2, 15.5)]


# ============================================================
# Priority 4 -- transfers_in_event / transfers_out_event promotion (the forward-looking
# price-change-timing signal's own real input; see transfer_planner.price_change_risk_by_player())
# ============================================================

TRANSFERS_HEADER = "id,gw,now_cost,status,minutes,goals_scored,assists,transfers_in_event,transfers_out_event"


def test_transfers_in_out_event_columns_promoted_when_present(con, tmp_path):
    season = "2026-2027"
    players_path = tmp_path / season / "players.csv"
    stats_path = tmp_path / season / "playerstats.csv"
    _write_csv(players_path, PLAYERS_HEADER, ["101,101,Erling,Haaland,Haaland,43,Forward"])
    _write_csv(stats_path, TRANSFERS_HEADER, ["101,1,15.0,a,90,1,0,62000,1500"])

    ingest_csv.ingest_csv_file(con, season, "players.csv", players_path)
    ingest_csv.ingest_csv_file(con, season, "playerstats.csv", stats_path)
    reconcile.build_dim_player(con)
    reconcile.build_fact_player_season_stats(con)

    row = con.execute(
        "SELECT transfers_in_event, transfers_out_event FROM fact_player_season_stats "
        "WHERE season = '2026-2027' AND gw = 1"
    ).fetchone()
    assert row == (62000.0, 1500.0)


def test_transfers_in_out_event_columns_null_when_source_lacks_them(con, tmp_path):
    """Same graceful-degrade-across-seasons handling every other _SEASON_STATS_NUMERIC_COLS
    entry already gets (see build_fact_player_season_stats()'s own docstring on 2024-2025
    genuinely predating several columns) -- a season's playerstats.csv without these two
    columns at all must reconcile cleanly to NULL, never raise or silently fabricate a 0."""
    # STATS_HEADER (module-level, above) has no transfers_in_event/transfers_out_event at all.
    row = _ingest_and_reconcile(con, tmp_path, ["101,1,15.0,a,90,1,0"])
    full_row = con.execute(
        "SELECT transfers_in_event, transfers_out_event FROM fact_player_season_stats "
        "WHERE season = '2026-2027' AND gw = 1"
    ).fetchone()
    assert full_row == (None, None)


def test_transfers_in_out_event_columns_refresh_on_a_later_batch(con, tmp_path):
    """Same live-refresh discipline as now_cost/status above -- a later same-day ingestion
    batch's updated transfer counts must overwrite the stale snapshot, not be discarded."""
    season = "2026-2027"
    players_path = tmp_path / season / "players.csv"
    stats_path = tmp_path / season / "playerstats.csv"
    _write_csv(players_path, PLAYERS_HEADER, ["101,101,Erling,Haaland,Haaland,43,Forward"])

    _write_csv(stats_path, TRANSFERS_HEADER, ["101,1,15.0,a,90,1,0,10000,500"])
    ingest_csv.ingest_csv_file(con, season, "players.csv", players_path)
    ingest_csv.ingest_csv_file(con, season, "playerstats.csv", stats_path)
    reconcile.build_dim_player(con)
    reconcile.build_fact_player_season_stats(con)

    _write_csv(stats_path, TRANSFERS_HEADER, ["101,1,15.0,a,90,1,0,55000,900"])
    ingest_csv.ingest_csv_file(con, season, "playerstats.csv", stats_path)
    reconcile.build_fact_player_season_stats(con)

    row = con.execute(
        "SELECT transfers_in_event, transfers_out_event FROM fact_player_season_stats "
        "WHERE season = '2026-2027' AND gw = 1"
    ).fetchone()
    assert row == (55000.0, 900.0)
