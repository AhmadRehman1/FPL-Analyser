"""fpl_quant.residual_levers: the crowd features and the residual export behind the residual-fitted
levers (docs/plans/2026-10_beat_the_template.md, Req 11 and Req 12)."""

import csv
import math

import pytest

from fpl_quant import backtest as bt
from fpl_quant import residual_levers as rl
from tests.test_backtest import _WALK_KWARGS, _seed_season_simulation_league
from tests.test_template_baseline import _seed_template_league

SEASON = "2025-2026"


def _set_transfers(con, gw, net_by_uid):
    for uid, net in net_by_uid.items():
        con.execute(
            "UPDATE fact_player_season_stats SET transfers_in_event = ?, transfers_out_event = 0 "
            "WHERE player_uid = ? AND season = ? AND gw = ?", [net, uid, SEASON, gw],
        )


def test_feature_rows_rank_ownership_within_position_and_transfers_overall(con):
    _seed_template_league(con)
    _set_transfers(con, 2, {"m1": 500.0, "f1": -300.0, "g1": 100.0})
    f = rl.feature_rows(con, SEASON, 3)
    assert len(f) == 18
    # goalkeepers: g1 30% above g2 10%
    assert f["g1"]["own_pct"] == 1.0 and f["g2"]["own_pct"] == 0.0
    # midfielders m1 60% top of six, m6 3% bottom
    assert f["m1"]["own_pct"] == 1.0 and f["m6"]["own_pct"] == 0.0
    assert f["m1"]["log_sbp"] == pytest.approx(math.log(61.0))
    # three players with transfer figures: f1 -300, g1 100, m1 500
    assert (f["f1"]["xfer_pct"], f["g1"]["xfer_pct"], f["m1"]["xfer_pct"]) == (0.0, 0.5, 1.0)
    assert f["d1"]["xfer_pct"] is None


def test_xfer_pct_ranks_every_player_with_transfer_figures_owned_or_not(con):
    _seed_template_league(con)
    _set_transfers(con, 2, {"m1": 500.0, "f1": -300.0, "g1": 100.0})
    con.execute("UPDATE fact_player_season_stats SET selected_by_percent = NULL WHERE player_uid = 'f1' AND gw = 2")
    f = rl.feature_rows(con, SEASON, 3)
    assert "f1" not in f  # no ownership, no feature row
    assert (f["g1"]["xfer_pct"], f["m1"]["xfer_pct"]) == (0.5, 1.0)  # f1's -300 still ranks below them


def test_feature_rows_never_read_row_g(con):
    # the leak test: changing row G's ownership doesn't change step G's features
    _seed_template_league(con)
    before = rl.feature_rows(con, SEASON, 3)
    con.execute("UPDATE fact_player_season_stats SET selected_by_percent = 99.0 - selected_by_percent WHERE gw = 3")
    _set_transfers(con, 3, {"m6": 10_000.0})
    assert rl.feature_rows(con, SEASON, 3) == before
    with bt.asof_scope(con, SEASON, 3):
        assert rl.feature_rows(con, SEASON, 3) == before


def test_feature_rows_missing_data(con):
    _seed_template_league(con)
    con.execute("UPDATE fact_player_season_stats SET selected_by_percent = NULL WHERE player_uid = 'd6' AND gw = 2")
    con.execute("DELETE FROM fact_player_season_stats WHERE player_uid = 'f4' AND gw = 2")
    f = rl.feature_rows(con, SEASON, 3)
    assert "d6" not in f and "f4" not in f
    assert rl.feature_rows(con, SEASON, 1) == {}  # no row 0
    assert rl.feature_rows(con, SEASON, 3, lag=2) and rl.feature_rows(con, SEASON, 3, lag=2)["m1"]["own_pct"] == 1.0


def test_price_band_boundaries_match_the_backtest():
    for price in (3.9, 4.99, 5.0, 6.5, 7.0, 8.9, 9.0, 14.0):
        assert rl.price_band(price) == bt._price_band(price)
    assert rl.price_band(None) is None


def test_residual_export_writes_one_row_per_projected_and_scored_player_gameweek(tmp_path):
    from fpl_quant import db

    con = db.connect(tmp_path / "w.duckdb")
    _seed_season_simulation_league(con)
    con.execute("UPDATE fact_player_season_stats SET selected_by_percent = 5.0 + now_cost, "
                "transfers_in_event = 10, transfers_out_event = 5")
    run_id = bt.run(con, n_antithetic_pairs=200, run_monte_carlo=False, seasons=(SEASON,), **_WALK_KWARGS)
    path = tmp_path / "residual_features.csv"
    n = rl.write_residual_csv(con, run_id, path)
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    assert n == len(rows) > 0
    assert list(rows[0]) == list(rl.RESIDUAL_COLUMNS)
    steps = {(r["season"], int(r["gw"])) for r in rows}
    scored = set(con.execute(
        "SELECT season, gameweek FROM backtest_gameweek_steps WHERE backtest_run_id = ? AND ep_model_version IS NOT NULL",
        [run_id],
    ).fetchall())
    assert steps == scored
    # GW1 has no earlier price row, so no band; later steps have both
    assert all(r["price_band"] == "" for r in rows if r["gw"] == "1")
    row = next(r for r in rows if int(r["gw"]) == 3)
    # the projection is the step's own gameweek EP, the realized score that week's event_points
    season, gw, uid = row["season"], int(row["gw"]), row["player_uid"]
    ep_mv = con.execute("SELECT ep_model_version FROM backtest_gameweek_steps WHERE backtest_run_id = ? AND season = ? "
                        "AND gameweek = ?", [run_id, season, gw]).fetchone()[0]
    ep_total = con.execute("SELECT ep_total FROM ep_gameweek_outputs WHERE model_version = ? AND player_uid = ?",
                           [ep_mv, uid]).fetchone()[0]
    realized = con.execute("SELECT event_points FROM fact_player_season_stats WHERE player_uid = ? AND season = ? "
                           "AND gw = ?", [uid, season, gw]).fetchone()[0]
    assert float(row["ep_pre_adjustment"]) == pytest.approx(ep_total)
    assert float(row["realized_points"]) == realized
    assert row["price_band"] in ("<5.0", "5.0-7.0", "7.0-9.0", "9.0+")
    assert row["xfer_pct"] != ""  # every player has transfer figures in this fixture
    con.close()
