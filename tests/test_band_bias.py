"""Lever L2, the band bias (docs/plans/2026-10_beat_the_template.md, Phase 6): each (position,
price band) cell's mean residual shrunk by n / (n + pseudo_rows), the fold each season is scored
with, the shared apply inside expected_points.run() (stacked with L1), and the Req 6 / Req 11
checks (flags, control identity, leak, missing data, fold selection, one adjustment in every
harness)."""

import csv
import json
import sys
from pathlib import Path

import pytest

from fpl_quant import backtest as bt
from fpl_quant import expected_points as ep
from fpl_quant import residual_levers as rl
from fpl_quant import transfer_planner as tp
from tests.test_backtest import _SEASON_SIM_VERSIONS
from tests.test_crowd_prior import FOLDS as CROWD_FOLDS
from tests.test_crowd_prior import _crowd_league, _snapshots
from tests.test_crowd_prior import _version as _crowd_version

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import fit_residual_lever as fit_script  # noqa: E402
import run_season_sim_arm as arm  # noqa: E402
import run_walkforward as rw  # noqa: E402

SEASON = "2025-2026"
COEFF_DATE = 20260201
WALK = {k: v for k, v in _SEASON_SIM_VERSIONS.items()
        if k not in ("horizon_params_version", "transfer_cost_params_version", "wildcard_threshold_params_version",
                     "free_hit_threshold_params_version", "kappa_tc_params_version")}


def _cells(adj_by_key: dict[str, float]) -> dict:
    return {key: {"n": 1000, "mean_resid": adj, "adj": adj} for key, adj in adj_by_key.items()}


# the season-simulation league's cells: nailed players cost 7.0 (8.5 for forwards), fringe 4.0 (5.5)
CELLS = {
    "Goalkeeper|<5.0": 0.10, "Goalkeeper|7.0-9.0": -0.20, "Defender|<5.0": 0.05, "Defender|7.0-9.0": 0.30,
    "Midfielder|<5.0": -0.40, "Midfielder|7.0-9.0": 0.50, "Forward|5.0-7.0": 0.20, "Forward|7.0-9.0": -0.60,
}
FOLDS = {"2024-2025": _cells(CELLS), "2025-2026": _cells({k: 2 * v for k, v in CELLS.items()}),
         "both": _cells({k: 3 * v for k, v in CELLS.items()})}


def _write_coeffs(tmp_path, monkeypatch, folds, coeff_date=COEFF_DATE, pseudo_rows=500.0):
    monkeypatch.setattr(rl, "COEFF_ROOT", tmp_path)
    rl.coeff_path(coeff_date, "band_bias").write_text(json.dumps({"pseudo_rows": pseudo_rows, "folds": folds}))


def _version(con, tmp_path, monkeypatch, folds=FOLDS, **kwargs):
    _write_coeffs(tmp_path, monkeypatch, folds, **kwargs)
    return rl.band_bias_version(con, kwargs.get("coeff_date", COEFF_DATE), rw.EXPERIMENT_EFFECTIVE_DATE)


def _expected_cell(con, season, deadline, uid, folds=FOLDS):
    position, price = con.execute(
        "SELECT dp.position, s.now_cost FROM fact_player_season_stats s JOIN dim_player dp USING (player_uid) "
        "WHERE s.player_uid = ? AND s.season = ? AND s.gw < ? AND s.now_cost IS NOT NULL ORDER BY s.gw DESC LIMIT 1",
        [uid, season, deadline],
    ).fetchone()
    cell = folds[rl.fold_for(season)].get(rl.band_key(position, rl.price_band(price)))
    return None if cell is None else cell["adj"]


def _ep_rows(con, mv):
    return {uid: (total, crowd, band) for uid, total, crowd, band in con.execute(
        "SELECT player_uid, ep_total, ep_crowd_adj, ep_band_adj FROM ep_gameweek_outputs WHERE model_version = ?", [mv],
    ).fetchall()}


def _ep(con, season, gameweek, asof, ts, mm, band=None, crowd=None):
    with bt.asof_scope(con, season, gameweek):
        mv = ep.run(con, asof, season, gameweek, ts, mm, 1, 1, 1, band_bias_params_version=band,
                    crowd_prior_params_version=crowd, lookback_seasons=bt.lookback_seasons_for(season))
    return _ep_rows(con, mv), mv


# ------------------------------------------------------------------ the fit


def _row(season, position, band, realized, ep_pre):
    return {"season": season, "position": position, "price_band": band, "realized_points": realized,
            "ep_pre_adjustment": ep_pre}


def test_fit_band_cells_shrinks_each_cell_mean_by_n_over_n_plus_k():
    rows = [_row(SEASON, "Midfielder", "9.0+", 3.0, 2.0)] * 500 + [_row(SEASON, "Midfielder", "9.0+", 1.0, 2.0)] * 250
    rows += [_row(SEASON, "Defender", "<5.0", 0.0, 0.5)] * 100
    rows += [_row(SEASON, "Defender", "", 9.0, 0.0), _row(SEASON, "Defender", "<5.0", "", 0.0)]  # incomplete: skipped
    cells = rl.fit_band_cells(rows, pseudo_rows=500)
    mid = cells["Midfielder|9.0+"]
    assert mid["n"] == 750
    assert mid["mean_resid"] == pytest.approx((500 * 1.0 - 250 * 1.0) / 750)
    assert mid["adj"] == pytest.approx((250 / 750) * 750 / 1250, abs=1e-6)
    dfn = cells["Defender|<5.0"]
    assert (dfn["n"], dfn["mean_resid"]) == (100, -0.5) and dfn["adj"] == pytest.approx(-0.5 * 100 / 600, abs=1e-6)
    assert set(cells) == {"Midfielder|9.0+", "Defender|<5.0"}


def test_fit_band_bias_fits_each_season_and_both():
    rows = [_row("2024-2025", "Forward", "9.0+", 2.0, 1.0)] * 100 + [_row("2025-2026", "Forward", "9.0+", 0.0, 1.0)] * 300
    folds = rl.fit_band_bias(rows, pseudo_rows=500)
    assert folds["2024-2025"]["Forward|9.0+"]["mean_resid"] == 1.0
    assert folds["2025-2026"]["Forward|9.0+"]["mean_resid"] == -1.0
    assert folds["both"]["Forward|9.0+"] == {"n": 400, "mean_resid": -0.5, "adj": round(-0.5 * 400 / 900, 6)}


def test_fit_script_writes_band_folds_and_a_cross_fold_check(tmp_path):
    rows = ([_row("2024-2025", "Forward", "9.0+", 3.0, 1.0)] * 400 + [_row("2025-2026", "Forward", "9.0+", 3.0, 1.0)] * 400
            + [_row("2025-2026", "Defender", "<5.0", 0.0, 0.0)] * 100)
    path = tmp_path / "residual_features.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    out = tmp_path / "band.json"
    assert fit_script.main(["--residuals", str(path), "--date", "20260201", "--lever", "band_bias", "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["pseudo_rows"] == 500.0 and data["fold_for"]["2025-2026"] == "2024-2025"
    assert data["folds"]["2024-2025"]["Forward|9.0+"]["adj"] == pytest.approx(2.0 * 400 / 900, abs=1e-6)
    check = data["cross_fold_check"]["2025-2026"]
    assert check["fitted_fold"] == "2024-2025" and check["mae_after"] < check["mae_before"]
    # the 2025-26 defenders have no 2024-25 cell: no adjustment
    assert check["n_adjusted"] == 400
    assert fit_script.main(["--residuals", str(path), "--date", "20260201", "--lever", "band_bias", "--out", str(out)]) == 2
    with pytest.raises(SystemExit):  # the crowd prior still needs its lag
        fit_script.main(["--residuals", str(path), "--date", "20260201", "--out", str(tmp_path / "c.json")])


# ------------------------------------------------------------------ the apply


def test_band_adjustments_pick_the_seasons_fold_and_the_latest_price_before_the_deadline(con):
    _crowd_league(con)
    for season in ("2024-2025", "2025-2026"):
        adj = rl.band_adjustments(con, season, 4, {"folds": FOLDS})
        assert adj and all(adj[uid] == pytest.approx(_expected_cell(con, season, 4, uid)) for uid in adj)
    # Req 11's leak test: row G's price is not read for step G; row G-1's is
    before = rl.band_adjustments(con, SEASON, 4, {"folds": FOLDS})
    con.execute("UPDATE fact_player_season_stats SET now_cost = 12.0 WHERE season = ? AND gw = 4", [SEASON])
    assert rl.band_adjustments(con, SEASON, 4, {"folds": FOLDS}) == before
    con.execute("UPDATE fact_player_season_stats SET now_cost = 4.5 WHERE season = ? AND gw = 3 AND player_uid = 'p9'",
                [SEASON])  # p9 (a 7.0 midfielder) drops into the <5.0 band
    moved = rl.band_adjustments(con, SEASON, 4, {"folds": FOLDS})
    assert before["p9"] == pytest.approx(FOLDS["2024-2025"]["Midfielder|7.0-9.0"]["adj"])
    assert moved["p9"] == pytest.approx(FOLDS["2024-2025"]["Midfielder|<5.0"]["adj"])
    with pytest.raises(KeyError):
        rl.band_adjustments(con, SEASON, 4, {"folds": {"both": FOLDS["both"]}})


def test_a_player_without_a_price_or_a_fitted_cell_keeps_his_ep(con, tmp_path, monkeypatch):
    _crowd_league(con)
    con.execute("UPDATE fact_player_season_stats SET now_cost = NULL WHERE player_uid = 'p3' AND season = ?", [SEASON])
    folds = {k: {c: v for c, v in cells.items() if not c.startswith("Goalkeeper")} for k, cells in FOLDS.items()}
    v = _version(con, tmp_path, monkeypatch, folds)
    asof, ts, mm = _snapshots(con, SEASON, 4)
    control, _ = _ep(con, SEASON, 4, asof, ts, mm)
    lever, _ = _ep(con, SEASON, 4, asof, ts, mm, band=v)
    goalkeepers = [uid for (uid,) in con.execute("SELECT player_uid FROM dim_player WHERE position = 'Goalkeeper'").fetchall()]
    for uid in ["p3", *goalkeepers]:
        assert lever[uid] == control[uid] and lever[uid][2] == 0.0
    assert any(band != 0.0 for _, _, band in lever.values())


def test_band_bias_adds_the_cell_adjustment_floored_at_zero_ep(con, tmp_path, monkeypatch):
    _crowd_league(con)
    # cheap midfielders get a cut far bigger than any projection: floored at zero, never below
    folds = {k: {**cells, "Midfielder|<5.0": {"n": 1000, "mean_resid": -20.0, "adj": -20.0}} for k, cells in FOLDS.items()}
    v = _version(con, tmp_path, monkeypatch, folds)
    asof, ts, mm = _snapshots(con, SEASON, 4)
    control, _ = _ep(con, SEASON, 4, asof, ts, mm)
    lever, mv = _ep(con, SEASON, 4, asof, ts, mm, band=v)
    floored = 0
    for uid, (total, crowd, band) in lever.items():
        cell = _expected_cell(con, SEASON, 4, uid, folds)
        pre = control[uid][0]
        expected = 0.0 if cell is None else max(cell, -max(pre, 0.0))  # one fixture each
        floored += cell is not None and expected != cell
        assert band == pytest.approx(expected) and crowd == 0.0
        assert total == pytest.approx(pre + expected)
    assert floored  # the cheap midfielders, now at exactly zero
    assert con.execute("SELECT band_bias_params_version FROM ep_model_versions WHERE model_version = ?",
                       [mv]).fetchone()[0] == v


def test_without_the_flag_ep_is_identical_and_zero_cells_change_nothing(con, tmp_path, monkeypatch):
    _crowd_league(con)
    zero = _version(con, tmp_path, monkeypatch, {k: _cells({c: 0.0 for c in CELLS}) for k in FOLDS})
    asof, ts, mm = _snapshots(con, SEASON, 4)
    control, mv_control = _ep(con, SEASON, 4, asof, ts, mm)
    flat, _ = _ep(con, SEASON, 4, asof, ts, mm, band=zero)
    assert flat == control
    assert con.execute("SELECT count(*) FROM ep_outputs WHERE model_version = ? AND ep_band_adj != 0.0",
                       [mv_control]).fetchone()[0] == 0


def test_l2_stacks_on_l1_and_the_export_takes_both_out(tmp_path, monkeypatch):
    from fpl_quant import db

    con = db.connect(tmp_path / "w.duckdb")
    _crowd_league(con)
    crowd = _crowd_version(con, tmp_path, monkeypatch, CROWD_FOLDS)
    band = _version(con, tmp_path, monkeypatch)
    run_id = con.execute("INSERT INTO backtest_runs (warm_up_gameweeks) VALUES (0) RETURNING backtest_run_id").fetchone()[0]
    bt.run_gameweek_step(con, run_id, SEASON, 4, n_antithetic_pairs=200, crowd_prior_params_version=crowd,
                         band_bias_params_version=band, **WALK)
    mv = con.execute("SELECT ep_model_version FROM backtest_gameweek_steps WHERE backtest_run_id = ?", [run_id]).fetchone()[0]
    ep_rows = _ep_rows(con, mv)
    assert any(c != 0 for _, c, _ in ep_rows.values()) and any(b != 0 for _, _, b in ep_rows.values())
    rows = rl.residual_rows(con, run_id, lag=2)
    assert rows
    for r in rows:
        total, c, b = ep_rows[r["player_uid"]]
        assert (r["crowd_adj"], r["band_adj"]) == (pytest.approx(c), pytest.approx(b))
        assert r["ep_pre_adjustment"] == pytest.approx(total - c - b)
    con.close()


def test_monte_carlo_summary_carries_both_levers(con, tmp_path, monkeypatch):
    from fpl_quant import monte_carlo

    _crowd_league(con)
    crowd = _crowd_version(con, tmp_path, monkeypatch, CROWD_FOLDS)
    band = _version(con, tmp_path, monkeypatch)
    run_id = con.execute("INSERT INTO backtest_runs (warm_up_gameweeks) VALUES (0) RETURNING backtest_run_id").fetchone()[0]
    bt.run_gameweek_step(con, run_id, SEASON, 4, n_antithetic_pairs=200, **WALK)
    ts, mm, ep_control, un, so = con.execute(
        "SELECT ts_model_version, mm_model_version, ep_model_version, un_model_version, so_run_id "
        "FROM backtest_gameweek_steps WHERE backtest_run_id = ?", [run_id]).fetchone()
    with bt.asof_scope(con, SEASON, 4) as deadline:
        asof = deadline.date()
        ep_lever = ep.run(con, asof, SEASON, 4, ts, mm, 1, 1, 1, crowd_prior_params_version=crowd,
                          band_bias_params_version=band, lookback_seasons=bt.lookback_seasons_for(SEASON))
        mcs = [monte_carlo.run(con, asof, so, mv, mm, ts, un, 1, 1, 1, n_antithetic_pairs=200,
                               season_priority=bt.lookback_seasons_for(SEASON)) for mv in (ep_control, ep_lever)]
    summary = [dict((uid, (mean, var)) for uid, mean, var in con.execute(
        "SELECT player_uid, mean_total, var_total FROM monte_carlo_player_summary WHERE model_version = ?", [mc]).fetchall())
        for mc in mcs]
    lever = _ep_rows(con, ep_lever)
    assert summary[0] and any(lever[uid][2] != 0.0 for uid in summary[0])
    for uid, (mean, var) in summary[0].items():
        _, c, b = lever[uid]
        assert summary[1][uid][0] == pytest.approx(mean + c + b) and summary[1][uid][1] == pytest.approx(var)


def test_walk_forward_and_the_planner_apply_the_same_band_adjustment(con, tmp_path, monkeypatch):
    # Req 11's identical-adjustment check for L2: the walk-forward step and the planner's horizon
    # (GW3 and GW4 from GW3's deadline, as the season sim and live call it) all carry GW3's cell
    # adjustment, floored at each projection's own zero
    _crowd_league(con)
    v = _version(con, tmp_path, monkeypatch)
    expected = rl.band_adjustments(con, SEASON, 3, {"folds": FOLDS})

    def check(mv):
        rows = _ep_rows(con, mv)
        assert set(rows) >= set(expected)
        for uid, (total, _, band) in rows.items():
            assert band == pytest.approx(max(expected.get(uid, 0.0), -max(total - band, 0.0)))
        assert any(band != 0 for _, _, band in rows.values())

    run_id = con.execute("INSERT INTO backtest_runs (warm_up_gameweeks) VALUES (0) RETURNING backtest_run_id").fetchone()[0]
    bt.run_gameweek_step(con, run_id, SEASON, 3, n_antithetic_pairs=200, band_bias_params_version=v, **WALK)
    check(con.execute("SELECT ep_model_version FROM backtest_gameweek_steps WHERE backtest_run_id = ?", [run_id]).fetchone()[0])
    asof, ts, mm = _snapshots(con, SEASON, 3, schedule_horizon_gameweeks=2)
    with bt.asof_scope(con, SEASON, 3, schedule_horizon_gameweeks=2):
        horizon = tp.compute_horizon_ep(con, asof, SEASON, 3, ts, mm, 2, 1, 1, 1, 1, 1, band_bias_params_version=v)
    assert set(horizon) == {3, 4}
    for ep_mv, _ in horizon.values():
        check(ep_mv)
    # the season sim threads it into its EP runs
    before = con.execute("SELECT coalesce(max(model_version), 0) FROM ep_model_versions").fetchone()[0]
    bt.run_season_simulation(con, SEASON, start_gameweek=2, end_gameweek=3, n_antithetic_pairs=200,
                             band_bias_params_version=v, **_SEASON_SIM_VERSIONS)
    assert con.execute("SELECT count(*) FROM ep_model_versions WHERE model_version > ? AND band_bias_params_version = ?",
                       [before, v]).fetchone()[0] >= 2


# ------------------------------------------------------------------ flags (Req 6)


def test_band_bias_flag_mints_and_reuses_a_version_in_both_scripts(con, tmp_path, monkeypatch):
    _write_coeffs(tmp_path, monkeypatch, FOLDS)
    assert "band_bias_params_version" not in rw._experiment_versions(con, rw._parse_args([]))
    out = rw._experiment_versions(con, rw._parse_args(["--band-bias", str(COEFF_DATE)]))
    assert set(out) == {"band_bias_params_version"}
    assert rl.resolve_band_bias(con, out["band_bias_params_version"])["coeffs"]["folds"] == FOLDS
    assert rw._experiment_versions(con, rw._parse_args(["--band-bias", str(COEFF_DATE)])) == out
    versions, changed = arm.arm_versions(con, arm._parse_args(["--label", "x", "--band-bias", str(COEFF_DATE)]))
    assert changed == out and versions["band_bias_params_version"] == out["band_bias_params_version"]
    control, _ = arm.arm_versions(con, arm._parse_args(["--label", "control"]))
    assert "band_bias_params_version" not in control
    # a changed coefficient file is refused
    path = rl.coeff_path(COEFF_DATE, "band_bias")
    path.write_text(path.read_text().replace("500.0", "400.0"))
    with pytest.raises(ValueError, match="changed"):
        rl.resolve_band_bias(con, out["band_bias_params_version"])
