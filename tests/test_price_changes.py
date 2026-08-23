from datetime import date

from fpl_quant import price_changes as pc


def _seed_player(con, uid, transfers_in, transfers_out, season="2026-2027", gw=5):
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, 'Forward') ON CONFLICT DO NOTHING", [uid, uid])
    con.execute(
        "INSERT INTO fact_player_season_stats (player_uid, season, gw, transfers_in_event, transfers_out_event, _ingested_at) "
        "VALUES (?, ?, ?, ?, ?, current_timestamp)",
        [uid, season, gw, transfers_in, transfers_out],
    )


def test_forecast_price_changes_rise_direction(con):
    _seed_player(con, "riser", transfers_in=100_000, transfers_out=1_000)
    result = pc.forecast_price_changes(
        con, target_season="2026-2027", as_of_gameweek=5, rise_threshold=50_000, fall_threshold=50_000, data_asof=date(2026, 8, 24),
    )
    assert len(result) == 1
    f = result[0]
    assert f.player_uid == "riser"
    assert f.direction == "rise"
    assert f.delta_pence == 10
    assert f.estimated_date == "2026-08-25"
    assert 0.0 < f.confidence <= 1.0


def test_forecast_price_changes_fall_direction(con):
    _seed_player(con, "faller", transfers_in=1_000, transfers_out=100_000)
    result = pc.forecast_price_changes(
        con, target_season="2026-2027", as_of_gameweek=5, rise_threshold=50_000, fall_threshold=50_000, data_asof=date(2026, 8, 24),
    )
    f = result[0]
    assert f.direction == "fall"
    assert f.delta_pence == -10
    assert f.estimated_date == "2026-08-25"


def test_forecast_price_changes_stable_direction(con):
    _seed_player(con, "steady", transfers_in=1_000, transfers_out=900)
    result = pc.forecast_price_changes(
        con, target_season="2026-2027", as_of_gameweek=5, rise_threshold=50_000, fall_threshold=50_000, data_asof=date(2026, 8, 24),
    )
    f = result[0]
    assert f.direction == "stable"
    assert f.delta_pence == 0
    assert f.estimated_date is None
    assert f.confidence == 0.0


def test_forecast_price_changes_omits_players_with_no_transfer_data(con):
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('nodata', 'nodata', 'Forward')")
    con.execute(
        "INSERT INTO fact_player_season_stats (player_uid, season, gw, _ingested_at) VALUES ('nodata', '2026-2027', 5, current_timestamp)"
    )
    result = pc.forecast_price_changes(
        con, target_season="2026-2027", as_of_gameweek=5, rise_threshold=50_000, fall_threshold=50_000, data_asof=date(2026, 8, 24),
    )
    assert result == []


def test_forecast_price_changes_delta_pence_always_a_multiple_of_10(con):
    _seed_player(con, "riser", transfers_in=200_000, transfers_out=0)
    _seed_player(con, "faller", transfers_in=0, transfers_out=200_000)
    _seed_player(con, "steady", transfers_in=10, transfers_out=10)
    result = pc.forecast_price_changes(
        con, target_season="2026-2027", as_of_gameweek=5, rise_threshold=50_000, fall_threshold=50_000, data_asof=date(2026, 8, 24),
    )
    assert len(result) == 3
    for f in result:
        assert f.delta_pence % 10 == 0


def test_forecast_price_changes_never_touches_ranking_tables(con):
    """This module must be pure read + reshape -- never writes anything that could feed a
    ranking (or anything else)."""
    _seed_player(con, "riser", transfers_in=100_000, transfers_out=1_000)
    before_tables = con.execute("SELECT table_name FROM information_schema.tables ORDER BY table_name").fetchall()
    pc.forecast_price_changes(
        con, target_season="2026-2027", as_of_gameweek=5, rise_threshold=50_000, fall_threshold=50_000, data_asof=date(2026, 8, 24),
    )
    after_tables = con.execute("SELECT table_name FROM information_schema.tables ORDER BY table_name").fetchall()
    assert before_tables == after_tables
    # and the seeded row itself is untouched
    row = con.execute(
        "SELECT transfers_in_event, transfers_out_event FROM fact_player_season_stats WHERE player_uid = 'riser'"
    ).fetchone()
    assert row == (100_000, 1_000)
