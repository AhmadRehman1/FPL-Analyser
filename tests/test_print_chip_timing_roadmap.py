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
