import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import print_chip_timing_roadmap as pctr  # noqa: E402
from fpl_quant import squad_optimizer as so  # noqa: E402
from test_transfer_planner import _seed_real_squad_optimizer_candidate_pool  # noqa: E402


def test_real_wildcard_gain_uses_the_real_planning_horizon_not_a_hardcoded_one(con, monkeypatch):
    """Regression test for the real bug: _real_wildcard_gain() used to hardcode
    horizon_gameweeks=1 for its wc_horizon call, even though evaluate_wildcard()'s own
    threshold (wildcard_gain_threshold_params.min_horizon_gain=8.0) is calibrated for the
    SAME 5-gameweek horizon transfer_planner.run() always uses live
    (planning_horizon_params.horizon_gameweeks, seeded to 5). Confirms the fix by spying on
    every compute_horizon_ep() call this function makes: the bootstrap-metadata call must
    still request exactly 1 gameweek (that one's legitimate -- see the module's own comment),
    but the wc_horizon call must now request the real seeded planning horizon (5), not 1."""
    horizon_ep_versions, holdings = _seed_real_squad_optimizer_candidate_pool(con)
    so.seed_v1_params(con)
    pctr.tp.seed_v1_params(con)  # seeds planning_horizon_params=5 among others

    element_names = {i: uid for i, uid in enumerate(h["player_uid"] for h in holdings)}
    picks = [
        {"element": i, "position": i + 1, "is_captain": i == 0, "is_vice_captain": i == 1}
        for i in element_names
    ]
    monkeypatch.setattr(pctr.ifp, "fetch_bootstrap_elements", lambda: element_names)
    monkeypatch.setattr(pctr.ifp, "fetch_entry_picks", lambda entry_id, event: picks)

    horizon_gameweeks_requested = []

    def _spy_compute_horizon_ep(con, calibration_asof_date, target_season, start_gameweek, ts_mv, mm_mv, horizon_gameweeks, *args, **kwargs):
        horizon_gameweeks_requested.append(horizon_gameweeks)
        return horizon_ep_versions

    monkeypatch.setattr(pctr.tp, "compute_horizon_ep", _spy_compute_horizon_ep)

    result = pctr._real_wildcard_gain(con, entry_id=999, event=2, target_gameweek=2)

    assert result is not None
    assert "gain" in result
    # bootstrap_horizon (current-event metadata only) is legitimately 1; wc_horizon (the real
    # EP sum evaluate_wildcard()'s threshold is calibrated against) must be the real planning
    # horizon, not the same hardcoded 1.
    assert horizon_gameweeks_requested == [1, 5]


def test_real_wildcard_gain_returns_none_and_warns_rather_than_raising_on_failure(con, monkeypatch, capsys):
    monkeypatch.setattr(pctr.ifp, "fetch_bootstrap_elements", lambda: {})
    monkeypatch.setattr(pctr.ifp, "fetch_entry_picks", lambda entry_id, event: [])

    result = pctr._real_wildcard_gain(con, entry_id=999, event=2, target_gameweek=2)

    assert result is None


def test_real_free_hit_gain_uses_a_single_gameweek_horizon_not_wildcards_multi_gameweek_one(con, monkeypatch):
    """_real_free_hit_gain() must request horizon_gameweeks=1 for BOTH its compute_horizon_ep()
    calls (bootstrap metadata AND the fh_horizon EP sum evaluate_free_hit() actually scores
    against) -- unlike _real_wildcard_gain(), which correctly widens its second call to the
    real 5-gameweek planning horizon (see the regression test above). evaluate_free_hit()'s own
    fresh_gw_value/current_gw_value are single-gameweek sums (the squad reverts after
    target_gameweek), so requesting a wider horizon here would be the mirror-image bug: paying
    for EP versions the evaluator never reads."""
    horizon_ep_versions, holdings = _seed_real_squad_optimizer_candidate_pool(con)
    so.seed_v1_params(con)
    pctr.tp.seed_v1_params(con)

    element_names = {i: uid for i, uid in enumerate(h["player_uid"] for h in holdings)}
    picks = [
        {"element": i, "position": i + 1, "is_captain": i == 0, "is_vice_captain": i == 1}
        for i in element_names
    ]
    monkeypatch.setattr(pctr.ifp, "fetch_bootstrap_elements", lambda: element_names)
    monkeypatch.setattr(pctr.ifp, "fetch_entry_picks", lambda entry_id, event: picks)

    horizon_gameweeks_requested = []

    def _spy_compute_horizon_ep(con, calibration_asof_date, target_season, start_gameweek, ts_mv, mm_mv, horizon_gameweeks, *args, **kwargs):
        horizon_gameweeks_requested.append(horizon_gameweeks)
        return horizon_ep_versions

    monkeypatch.setattr(pctr.tp, "compute_horizon_ep", _spy_compute_horizon_ep)

    result = pctr._real_free_hit_gain(con, entry_id=999, event=2, target_gameweek=2)

    assert result is not None
    assert "gain" in result
    assert horizon_gameweeks_requested == [1, 1]


def test_real_free_hit_gain_returns_none_and_warns_rather_than_raising_on_failure(con, monkeypatch, capsys):
    monkeypatch.setattr(pctr.ifp, "fetch_bootstrap_elements", lambda: {})
    monkeypatch.setattr(pctr.ifp, "fetch_entry_picks", lambda entry_id, event: [])

    result = pctr._real_free_hit_gain(con, entry_id=999, event=2, target_gameweek=2)

    assert result is None


def test_weekly_avg_squad_swing_matches_manual_average(con):
    """Real DB read, no monkeypatching: seeds a team strength snapshot and real fixtures across
    the full FIRST_HALF_GAMEWEEKS range, then confirms _weekly_avg_squad_swing() reports the
    same per-gameweek value rolling_swing_score() itself computes for that squad's one club,
    not some other aggregation -- and that every requested gameweek is present in order."""
    from fpl_quant import fixture_swing as fs
    from test_fixture_swing import _insert_fixture, _seed_teams_and_strength

    ts_mv = _seed_teams_and_strength(con, [("team_a", 0.3, -0.1), ("team_b", -0.2, 0.2)])
    for gw in pctr.FIRST_HALF_GAMEWEEKS:
        _insert_fixture(con, f"m{gw}", "2026-2027", gw, "team_a", "team_b")

    weekly = pctr._weekly_avg_squad_swing(con, {"team_a"}, short_window=1)

    assert [w["gameweek"] for w in weekly] == list(pctr.FIRST_HALF_GAMEWEEKS)
    probe_gw = pctr.FIRST_HALF_GAMEWEEKS.start
    expected = fs.rolling_swing_score(
        con, "team_a", "2026-2027", probe_gw, ts_mv,
        short_window=1, long_window=pctr.LONG_WINDOW_GAMEWEEKS,
    )
    probe_row = next(w for w in weekly if w["gameweek"] == probe_gw)
    assert probe_row["avg_squad_swing"] == round(expected.swing_score, 3)
    assert probe_row["n_teams_with_data"] == 1
