"""Lever L1, the crowd prior (docs/plans/2026-10_beat_the_template.md, Phase 5): the ridge fit and
its min_rows gate, the fold each season is scored with, the shared apply inside
expected_points.run(), and the Req 6 / Req 11 acceptance tests (flags, control identity, leak,
missing data, fold selection, one adjustment across the walk-forward, season sim and live paths)."""

import json
import random
import sys
from pathlib import Path

import pytest

from fpl_quant import backtest as bt
from fpl_quant import expected_points as ep
from fpl_quant import minutes_model, team_strength
from fpl_quant import residual_levers as rl
from fpl_quant import transfer_planner as tp
from tests.test_backtest import _SEASON_SIM_VERSIONS, _seed_season_simulation_league
from tests.test_template_baseline import _seed_template_league

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import fit_residual_lever as fit_script  # noqa: E402
import run_season_sim_arm as arm  # noqa: E402
import run_walkforward as rw  # noqa: E402

SEASON = "2025-2026"
COEFF_DATE = 20260101
MEANS, SDS = [0.5, 1.5, 0.5], [0.3, 1.0, 0.3]


def _fit(beta):
    return {"n": 5000, "means": list(MEANS), "sds": list(SDS), "beta": list(beta), "resid_mean": 0.0}


def _write_coeffs(tmp_path, monkeypatch, folds, clip=1.5, lag=2, coeff_date=COEFF_DATE):
    """A coefficient file in a temporary data/recalibration, as fit_residual_lever.py writes it."""
    monkeypatch.setattr(rl, "COEFF_ROOT", tmp_path)
    payload = {"features": list(rl.FEATURES), "lag": lag, "alpha": 10.0, "min_rows": 2000, "clip": clip, "folds": folds}
    rl.coeff_path(coeff_date).write_text(json.dumps(payload))
    return payload


def _version(con, tmp_path, monkeypatch, folds, **kwargs):
    _write_coeffs(tmp_path, monkeypatch, folds, **kwargs)
    return rl.crowd_prior_version(con, kwargs.get("coeff_date", COEFF_DATE), rw.EXPERIMENT_EFFECTIVE_DATE)


def _expected(values, beta, clip=1.5):
    total = sum(b * (values[f] - m) / s for f, b, m, s in zip(rl.FEATURES, beta, MEANS, SDS))
    return max(-clip, min(clip, total))


# ------------------------------------------------------------------ the fit


def _rows(n, season=SEASON, signal=0.4, seed=3):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        own = rng.random()
        rows.append({"season": season, "own_pct": own, "log_sbp": rng.random() * 3, "xfer_pct": rng.random(),
                     "realized_points": 2.0 + signal * (own - 0.5) / 0.2887 + rng.gauss(0, 1.0),
                     "ep_pre_adjustment": 2.0})
    return rows


def test_fit_ridge_recovers_a_linear_signal_on_the_standardized_feature():
    fit = rl.fit_ridge(_rows(4000))
    assert fit["n"] == 4000
    assert fit["beta"][0] == pytest.approx(0.4, abs=0.05)
    assert abs(fit["beta"][1]) < 0.05 and abs(fit["beta"][2]) < 0.05
    # standardized: each feature's fold mean and sd are kept for the apply
    assert fit["means"][0] == pytest.approx(0.5, abs=0.02) and fit["sds"][0] == pytest.approx(0.2887, abs=0.01)


def test_fit_ridge_min_rows_gate_counts_complete_rows_only():
    rows = _rows(2000)
    assert rl.fit_ridge(rows) is not None
    rows[0]["xfer_pct"] = ""  # one incomplete row leaves 1,999
    assert rl.fit_ridge(rows) is None
    assert rl.fit_ridge(rows, min_rows=1999) is not None


def test_fit_crowd_prior_fits_each_season_and_both():
    rows = _rows(2500, "2024-2025", signal=0.2) + _rows(2500, "2025-2026", signal=0.6, seed=4)
    folds = rl.fit_crowd_prior(rows)
    assert set(folds) == {"2024-2025", "2025-2026", "both"}
    assert folds["2024-2025"]["n"] == folds["2025-2026"]["n"] == 2500 and folds["both"]["n"] == 5000
    assert folds["2024-2025"]["beta"][0] < folds["both"]["beta"][0] < folds["2025-2026"]["beta"][0]
    # a season below min_rows is not fitted; the pooled fold still is
    thin = rl.fit_crowd_prior(_rows(2500, "2024-2025") + _rows(100, "2025-2026"))
    assert thin["2025-2026"] is None and thin["2024-2025"] is not None and thin["both"]["n"] == 2600


def test_adjustment_is_clipped_and_needs_every_feature():
    fit = _fit([10.0, 0.0, 0.0])
    assert rl.adjustment(fit, {"own_pct": 1.0, "log_sbp": 1.5, "xfer_pct": 0.5}, rl.FEATURES, 1.5) == 1.5
    assert rl.adjustment(fit, {"own_pct": 0.0, "log_sbp": 1.5, "xfer_pct": 0.5}, rl.FEATURES, 1.5) == -1.5
    assert rl.adjustment(fit, {"own_pct": 0.5, "log_sbp": 1.5, "xfer_pct": None}, rl.FEATURES, 1.5) is None


def test_fit_script_writes_folds_and_a_cross_fold_check(tmp_path):
    rows = _rows(2500, "2024-2025", signal=0.5) + _rows(2500, "2025-2026", signal=0.5, seed=5)
    path = tmp_path / "residual_features.csv"
    with open(path, "w", newline="") as fh:
        import csv

        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    out = tmp_path / "coeffs.json"
    assert fit_script.main(["--residuals", str(path), "--date", "20260101", "--lag", "2", "--source-run", "1",
                            "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert (data["lag"], data["alpha"], data["min_rows"], data["clip"]) == (2, 10.0, 2000, 1.5)
    assert data["fold_for"] == {"2024-2025": "2025-2026", "2025-2026": "2024-2025", "2026-2027": "both"}
    assert data["source"]["run_id"] == "1" and len(data["source"]["sha256"]) == 64
    check = data["cross_fold_check"]
    for season in ("2024-2025", "2025-2026"):
        # a real signal fitted on the other season cuts this season's error
        assert check[season]["fitted_fold"] == rl.fold_for(season)
        assert check[season]["mae_after"] < check[season]["mae_before"]


# ------------------------------------------------------------------ folds and features


def test_fold_for_scores_each_season_with_a_fit_that_never_saw_it():
    assert rl.fold_for("2024-2025") == "2025-2026"
    assert rl.fold_for("2025-2026") == "2024-2025"
    assert rl.fold_for("2026-2027") == "both"


def _copy_season(con, to_season, from_season=SEASON):
    con.execute(
        "INSERT INTO fact_player_season_stats (player_uid, season, gw, now_cost, selected_by_percent, status, "
        "event_points, _ingested_at) SELECT player_uid, ?, gw, now_cost, selected_by_percent, status, event_points, "
        "_ingested_at FROM fact_player_season_stats WHERE season = ?", [to_season, from_season],
    )


def test_crowd_adjustments_pick_the_fold_of_the_season_scored(con):
    # Req 11's fold-selection test: the same feature rows in three seasons, three folds with
    # different coefficients -- each season is adjusted by the fold fold_for() names
    _seed_template_league(con)
    _copy_season(con, "2024-2025")
    _copy_season(con, "2026-2027")
    con.execute("UPDATE fact_player_season_stats SET transfers_in_event = length(player_uid) * selected_by_percent, "
                "transfers_out_event = 0")
    betas = {"2024-2025": [1.0, 0.0, 0.0], "2025-2026": [2.0, 0.0, 0.0], "both": [3.0, 0.0, 0.0]}
    coeffs = {"features": list(rl.FEATURES), "folds": {k: _fit(b) for k, b in betas.items()}}
    for season in ("2024-2025", "2025-2026", "2026-2027"):
        adj = rl.crowd_adjustments(con, season, 3, coeffs, clip=100.0, lag=2)
        values = rl.feature_rows(con, season, 3, lag=2)
        beta = betas[rl.fold_for(season)]
        assert adj["m1"] == pytest.approx(_expected(values["m1"], beta, clip=100.0))
        assert adj["m6"] == pytest.approx(_expected(values["m6"], beta, clip=100.0))


def test_crowd_adjustments_are_empty_without_a_fitted_fold(con):
    _seed_template_league(con)
    coeffs = {"features": list(rl.FEATURES), "folds": {"2024-2025": None, "2025-2026": _fit([1, 0, 0]), "both": None}}
    assert rl.crowd_adjustments(con, SEASON, 3, coeffs, clip=1.5, lag=2) == {}


# ------------------------------------------------------------------ the apply inside expected_points.run()


def _crowd_league(con):
    """The season-simulation league with ownership and transfer figures that differ by player and
    gameweek, so every feature varies."""
    _seed_season_simulation_league(con)
    con.execute(
        "UPDATE fact_player_season_stats SET selected_by_percent = 1.0 + 3.0 * CAST(substr(player_uid, 2) AS INTEGER) + gw, "
        "transfers_in_event = 100 * ((CAST(substr(player_uid, 2) AS INTEGER) * 7 + gw) % 11), transfers_out_event = 300"
    )


def _snapshots(con, season, gameweek, schedule_horizon_gameweeks=1):
    with bt.asof_scope(con, season, gameweek, schedule_horizon_gameweeks=schedule_horizon_gameweeks) as deadline:
        asof = deadline.date()
        ts = team_strength.calibrate(con, asof, 1, 1, target_season=season, fit_seasons=bt.fit_seasons_for(season))
        mm = minutes_model.run(con, asof, season, 1, 1, 1, 1)
    return asof, ts, mm


def _ep(con, season, gameweek, asof, ts, mm, version):
    with bt.asof_scope(con, season, gameweek):
        mv = ep.run(con, asof, season, gameweek, ts, mm, 1, 1, 1, crowd_prior_params_version=version,
                    lookback_seasons=bt.lookback_seasons_for(season))
    return {uid: (total, adj) for uid, total, adj in con.execute(
        "SELECT player_uid, ep_total, ep_crowd_adj FROM ep_gameweek_outputs WHERE model_version = ?", [mv],
    ).fetchall()}, mv


FOLDS = {"2024-2025": _fit([0.6, -0.2, 0.4]), "2025-2026": _fit([0.1, 0.1, 0.1]), "both": _fit([0.3, 0.0, 0.2])}


def test_crowd_prior_adds_each_players_adjustment_to_ep_total(con, tmp_path, monkeypatch):
    _crowd_league(con)
    v = _version(con, tmp_path, monkeypatch, FOLDS)
    asof, ts, mm = _snapshots(con, SEASON, 4)
    control, _ = _ep(con, SEASON, 4, asof, ts, mm, None)
    lever, mv = _ep(con, SEASON, 4, asof, ts, mm, v)
    features = rl.feature_rows(con, SEASON, 4, lag=2)
    assert set(lever) == set(control) and features
    for uid, (total, adj) in lever.items():
        expected = _expected(features[uid], FOLDS["2024-2025"]["beta"]) if uid in features else 0.0
        expected = max(expected, -max(control[uid][0], 0.0))  # one fixture each: floored at zero EP
        assert adj == pytest.approx(expected)
        assert total == pytest.approx(control[uid][0] + expected)
        assert control[uid][1] == 0.0
    assert con.execute("SELECT crowd_prior_params_version FROM ep_model_versions WHERE model_version = ?",
                       [mv]).fetchone()[0] == v


def test_a_player_with_no_feature_row_keeps_his_ep(con, tmp_path, monkeypatch):
    # Req 11's missing-data test: no ownership in row G-2, or no row at all, means no adjustment
    _crowd_league(con)
    con.execute("UPDATE fact_player_season_stats SET selected_by_percent = NULL WHERE player_uid = 'p3' AND season = ? AND gw = 2", [SEASON])
    con.execute("DELETE FROM fact_player_season_stats WHERE player_uid = 'p9' AND season = ? AND gw = 2", [SEASON])
    v = _version(con, tmp_path, monkeypatch, FOLDS)
    asof, ts, mm = _snapshots(con, SEASON, 4)
    control, _ = _ep(con, SEASON, 4, asof, ts, mm, None)
    lever, _ = _ep(con, SEASON, 4, asof, ts, mm, v)
    for uid in ("p3", "p9"):
        assert lever[uid] == (control[uid][0], 0.0)
    assert any(adj != 0.0 for _, adj in lever.values())


def test_changing_row_g_or_g_minus_1_does_not_change_step_gs_ep(con, tmp_path, monkeypatch):
    # Req 11's leak test (A2 failed, so features read row G-2): rewriting rows G and G-1's
    # ownership and transfers leaves step G's EP as it was; rewriting row G-2 moves it
    _crowd_league(con)
    v = _version(con, tmp_path, monkeypatch, FOLDS)
    asof, ts, mm = _snapshots(con, SEASON, 4)
    before, _ = _ep(con, SEASON, 4, asof, ts, mm, v)
    for gw in (3, 4):
        con.execute(
            "UPDATE fact_player_season_stats SET selected_by_percent = 99.0 - selected_by_percent, "
            "transfers_in_event = 50000 - transfers_in_event WHERE season = ? AND gw = ?", [SEASON, gw],
        )
    after, _ = _ep(con, SEASON, 4, asof, ts, mm, v)
    assert after == before
    con.execute("UPDATE fact_player_season_stats SET selected_by_percent = 99.0 - selected_by_percent "
                "WHERE season = ? AND gw = 2", [SEASON])
    moved, _ = _ep(con, SEASON, 4, asof, ts, mm, v)
    assert moved != before


def test_a_double_gameweek_splits_the_adjustment_by_fixture_ep(con, tmp_path, monkeypatch):
    _crowd_league(con)
    # GW4's round is B-C, D-E, F-A; give A and B a second fixture that week
    con.execute(
        "INSERT INTO fact_match (match_id, season, gameweek, home_team_uid, away_team_uid, home_score, away_score, "
        "finished, competition, kickoff_time, _ingested_at) SELECT 'm_dgw', season, gameweek, 'team_a', 'team_b', 1, 1, "
        "TRUE, competition, kickoff_time + INTERVAL 2 DAY, _ingested_at FROM fact_match WHERE match_id = ?",
        [f"m_{SEASON}_4_0"],
    )
    v = _version(con, tmp_path, monkeypatch, {k: _fit([3.0, 0.0, 0.0]) for k in FOLDS})
    asof, ts, mm = _snapshots(con, SEASON, 4)
    lever, mv = _ep(con, SEASON, 4, asof, ts, mm, v)
    features = rl.feature_rows(con, SEASON, 4, lag=2)
    rows = con.execute(
        "SELECT player_uid, ep_total - ep_crowd_adj, ep_crowd_adj FROM ep_outputs WHERE model_version = ? ORDER BY player_uid",
        [mv],
    ).fetchall()
    by_player: dict[str, list] = {}
    for uid, pre, adj in rows:
        by_player.setdefault(uid, []).append((pre, adj))
    doubles = {uid: fx for uid, fx in by_player.items() if len(fx) == 2}
    assert doubles  # A's and B's players
    unfloored = 0
    for uid, fixtures in doubles.items():
        target = _expected(features[uid], [3.0, 0.0, 0.0])
        total_pre = sum(max(pre, 0.0) for pre, _ in fixtures)
        for pre, adj in fixtures:
            assert adj == pytest.approx(max(target * max(pre, 0.0) / total_pre, -max(pre, 0.0)))
        assert lever[uid][1] == pytest.approx(sum(adj for _, adj in fixtures))
        if target >= -total_pre:  # the floor doesn't bind: the shares add up to the adjustment
            unfloored += 1
            assert sum(adj for _, adj in fixtures) == pytest.approx(target)
    assert unfloored


def test_an_adjustment_never_takes_ep_below_zero(con, tmp_path, monkeypatch):
    _crowd_league(con)
    # the full -1.5 cut (the clip) for every player with features (own_pct is never near a mean of
    # 2): a projection below 1.5 ends at zero, never below it; a bigger one loses 1.5
    cut_fit = {"n": 5000, "means": [2.0, 1.5, 0.5], "sds": [1e-6, 1.0, 1.0], "beta": [1.0, 0.0, 0.0], "resid_mean": 0.0}
    v = _version(con, tmp_path, monkeypatch, {k: dict(cut_fit) for k in FOLDS})
    asof, ts, mm = _snapshots(con, SEASON, 4)
    control, _ = _ep(con, SEASON, 4, asof, ts, mm, None)
    lever, _ = _ep(con, SEASON, 4, asof, ts, mm, v)
    features = rl.feature_rows(con, SEASON, 4, lag=2)
    cut = list(features)
    assert any(0.0 <= control[uid][0] < 1.5 for uid in cut) and any(control[uid][0] > 1.5 for uid in cut)
    for uid in cut:
        pre = control[uid][0]
        assert lever[uid][1] == pytest.approx(max(-1.5, -max(pre, 0.0)))
        assert lever[uid][0] == pytest.approx(pre - 1.5 if pre > 1.5 else min(pre, 0.0), abs=1e-12)
        assert lever[uid][0] >= min(pre, 0.0)


def test_without_the_flag_ep_is_identical_to_the_control(con, tmp_path, monkeypatch):
    # Req 6's control identity: no version leaves every EP row untouched (ep_crowd_adj 0), and a
    # version whose coefficients are all zero reproduces the control exactly
    _crowd_league(con)
    zero = _version(con, tmp_path, monkeypatch, {k: _fit([0.0, 0.0, 0.0]) for k in FOLDS})
    asof, ts, mm = _snapshots(con, SEASON, 4)
    control, mv_control = _ep(con, SEASON, 4, asof, ts, mm, None)
    flat, mv_flat = _ep(con, SEASON, 4, asof, ts, mm, zero)
    assert flat == control
    cols = [c for (c,) in con.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'ep_outputs' "
        "AND column_name NOT IN ('model_version', 'ep_crowd_adj', 'created_at', '_computed_at')"
    ).fetchall()]
    sel = ", ".join(cols)
    a = con.execute(f"SELECT {sel} FROM ep_outputs WHERE model_version = ? ORDER BY player_uid, fixture_match_id", [mv_control]).fetchall()
    b = con.execute(f"SELECT {sel} FROM ep_outputs WHERE model_version = ? ORDER BY player_uid, fixture_match_id", [mv_flat]).fetchall()
    assert a == b
    assert con.execute("SELECT count(*) FROM ep_outputs WHERE model_version = ? AND ep_crowd_adj != 0.0",
                       [mv_control]).fetchone()[0] == 0


# ------------------------------------------------------------------ one adjustment in every harness


def test_walk_forward_season_sim_and_live_apply_the_same_adjustment(con, tmp_path, monkeypatch):
    """Req 11's identical-adjustment test. For the deadline of GW3: the walk-forward step's EP,
    the season sim's planner horizon (GW3 and GW4) and the live planner horizon (compute_horizon_ep,
    as scripts/run_transfer_planner_for_real_squad.py calls it) all carry the same per-player
    adjustment, through residual_levers.crowd_adjustments() at deadline 3."""
    _crowd_league(con)
    v = _version(con, tmp_path, monkeypatch, FOLDS)
    calls = []
    real = rl.crowd_adjustments

    def spy(con_, season, deadline, coeffs, clip, lag):
        out = real(con_, season, deadline, coeffs, clip, lag)
        calls.append((season, deadline, lag, out))
        return out

    monkeypatch.setattr(rl, "crowd_adjustments", spy)
    expected = real(con, SEASON, 3, rl.resolve_crowd_prior(con, v)["coeffs"], 1.5, 2)
    assert expected

    def adj_of(mv):
        return {uid: (adj, total - adj) for uid, adj, total in con.execute(
            "SELECT player_uid, ep_crowd_adj, ep_total FROM ep_gameweek_outputs WHERE model_version = ?", [mv]).fetchall()}

    def gw_of(mv):
        return con.execute("SELECT DISTINCT m.gameweek FROM ep_outputs o JOIN fact_match m ON m.match_id = o.fixture_match_id "
                           "WHERE o.model_version = ?", [mv]).fetchone()[0]

    def check(adj):
        # the same adjustment everywhere, floored at each projection's own zero
        assert set(adj) >= set(expected)
        for uid, (value, pre) in adj.items():
            assert value == pytest.approx(max(expected.get(uid, 0.0), -max(pre, 0.0)))
        assert any(value != 0.0 for value, _ in adj.values())

    # walk-forward
    run_id = con.execute("INSERT INTO backtest_runs (warm_up_gameweeks) VALUES (0) RETURNING backtest_run_id").fetchone()[0]
    walk = {k: val for k, val in _SEASON_SIM_VERSIONS.items()
            if k not in ("horizon_params_version", "transfer_cost_params_version", "wildcard_threshold_params_version",
                         "free_hit_threshold_params_version", "kappa_tc_params_version")}
    bt.run_gameweek_step(con, run_id, SEASON, 3, n_antithetic_pairs=200, crowd_prior_params_version=v, **walk)
    step_mv = con.execute("SELECT ep_model_version FROM backtest_gameweek_steps WHERE backtest_run_id = ? AND gameweek = 3",
                          [run_id]).fetchone()[0]
    check(adj_of(step_mv))

    # live: the planner's horizon from the deadline of GW3
    asof, ts, mm = _snapshots(con, SEASON, 3, schedule_horizon_gameweeks=2)
    with bt.asof_scope(con, SEASON, 3, schedule_horizon_gameweeks=2):
        horizon = tp.compute_horizon_ep(con, asof, SEASON, 3, ts, mm, 2, 1, 1, 1, 1, 1, crowd_prior_params_version=v)
    assert set(horizon) == {3, 4}
    for ep_mv, _ in horizon.values():
        check(adj_of(ep_mv))

    # season sim: GW2 bootstrap, then the GW3 decision's planner horizon
    before = con.execute("SELECT coalesce(max(model_version), 0) FROM ep_model_versions").fetchone()[0]
    bt.run_season_simulation(con, SEASON, start_gameweek=2, end_gameweek=3, n_antithetic_pairs=200,
                             crowd_prior_params_version=v, **_SEASON_SIM_VERSIONS)
    sim_mvs = [mv for (mv,) in con.execute(
        "SELECT model_version FROM ep_model_versions WHERE model_version > ? AND crowd_prior_params_version = ?", [before, v],
    ).fetchall()]
    planner_mvs = [mv for mv in sim_mvs if mv != min(sim_mvs)]  # the first is GW2's bootstrap
    assert {gw_of(mv) for mv in planner_mvs} >= {3, 4}
    for mv in planner_mvs:
        check(adj_of(mv))

    # every call at a GW3 deadline read row 1 (lag 2) and returned the same adjustments
    at3 = [c for c in calls if c[1] == 3]
    assert at3 and all(c[0] == SEASON and c[2] == 2 and c[3] == expected for c in at3)
    assert {c[1] for c in calls} == {2, 3}  # the bootstrap's own deadline, then GW3's


# ------------------------------------------------------------------ flags (Req 6) and the export


def test_crowd_prior_flag_mints_and_reuses_a_version_in_both_scripts(con, tmp_path, monkeypatch):
    _write_coeffs(tmp_path, monkeypatch, FOLDS)
    assert "crowd_prior_params_version" not in rw._experiment_versions(con, rw._parse_args([]))
    out = rw._experiment_versions(con, rw._parse_args(["--crowd-prior", str(COEFF_DATE)]))
    assert set(out) == {"crowd_prior_params_version"}
    prior = rl.resolve_crowd_prior(con, out["crowd_prior_params_version"])
    assert (prior["clip"], prior["lag"]) == (1.5, 2) and prior["coeffs"]["folds"]["both"] == FOLDS["both"]
    assert rw._experiment_versions(con, rw._parse_args(["--crowd-prior", str(COEFF_DATE)])) == out
    versions, changed = arm.arm_versions(con, arm._parse_args(["--label", "x", "--crowd-prior", str(COEFF_DATE)]))
    assert changed == out and versions["crowd_prior_params_version"] == out["crowd_prior_params_version"]
    control, _ = arm.arm_versions(con, arm._parse_args(["--label", "control"]))
    assert "crowd_prior_params_version" not in control
    # a different coefficient file is a different version
    _write_coeffs(tmp_path, monkeypatch, FOLDS, clip=1.0, coeff_date=20260102)
    other = rw._experiment_versions(con, rw._parse_args(["--crowd-prior", "20260102"]))
    assert other["crowd_prior_params_version"] != out["crowd_prior_params_version"]


def test_residual_export_takes_the_crowd_adjustment_out_of_the_projection(tmp_path, monkeypatch):
    from fpl_quant import db

    con = db.connect(tmp_path / "w.duckdb")
    _crowd_league(con)
    v = _version(con, tmp_path, monkeypatch, FOLDS)
    run_id = con.execute("INSERT INTO backtest_runs (warm_up_gameweeks) VALUES (0) RETURNING backtest_run_id").fetchone()[0]
    walk = {k: val for k, val in _SEASON_SIM_VERSIONS.items()
            if k not in ("horizon_params_version", "transfer_cost_params_version", "wildcard_threshold_params_version",
                         "free_hit_threshold_params_version", "kappa_tc_params_version")}
    bt.run_gameweek_step(con, run_id, SEASON, 3, n_antithetic_pairs=200, crowd_prior_params_version=v, **walk)
    rows = rl.residual_rows(con, run_id, lag=2)
    step_mv = con.execute("SELECT ep_model_version FROM backtest_gameweek_steps WHERE backtest_run_id = ?", [run_id]).fetchone()[0]
    ep_rows = {uid: (total, adj) for uid, total, adj in con.execute(
        "SELECT player_uid, ep_total, ep_crowd_adj FROM ep_gameweek_outputs WHERE model_version = ?", [step_mv]).fetchall()}
    assert rows and any(r["crowd_adj"] != 0.0 for r in rows)
    for r in rows:
        total, adj = ep_rows[r["player_uid"]]
        assert r["crowd_adj"] == pytest.approx(adj)
        assert r["ep_pre_adjustment"] == pytest.approx(total - adj)
    con.close()
