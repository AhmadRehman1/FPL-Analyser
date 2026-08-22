import pytest

from fpl_quant import ownership as own


def test_estimate_captaincy_rate_full_at_top_ep():
    """The single highest-EP player in a group gets the full captaincy_concentration**0 = 1.0
    decay factor -- captaincy rate equals raw ownership."""
    rate = own.estimate_captaincy_rate(selected_by_percent=40.0, ep_total=8.0, top_ep_total=8.0, captaincy_concentration=0.5)
    assert rate == pytest.approx(40.0)


def test_estimate_captaincy_rate_decays_with_ep_gap():
    """Two players with identical ownership but a real EP gap from the top: the one closer
    to the top pick's EP gets a higher estimated captaincy rate."""
    near = own.estimate_captaincy_rate(20.0, ep_total=7.5, top_ep_total=8.0, captaincy_concentration=0.5)
    far = own.estimate_captaincy_rate(20.0, ep_total=5.0, top_ep_total=8.0, captaincy_concentration=0.5)
    assert 0.0 < far < near < 20.0


def test_estimate_captaincy_rate_never_exceeds_ownership():
    """A captaincy RATE can never exceed the underlying ownership RATE -- you can only
    captain a player you own."""
    rate = own.estimate_captaincy_rate(15.0, ep_total=8.0, top_ep_total=8.0, captaincy_concentration=0.9)
    assert rate <= 15.0


def test_estimate_captaincy_rate_zero_for_unowned_or_missing_ownership():
    assert own.estimate_captaincy_rate(0.0, ep_total=8.0, top_ep_total=8.0, captaincy_concentration=0.5) == 0.0
    assert own.estimate_captaincy_rate(None, ep_total=8.0, top_ep_total=8.0, captaincy_concentration=0.5) == 0.0


def test_estimate_captaincy_rate_rejects_out_of_range_concentration():
    with pytest.raises(ValueError):
        own.estimate_captaincy_rate(10.0, 8.0, 8.0, captaincy_concentration=0.0)
    with pytest.raises(ValueError):
        own.estimate_captaincy_rate(10.0, 8.0, 8.0, captaincy_concentration=1.0)


def test_effective_ownership_adds_captaincy_rate_to_raw_ownership():
    assert own.effective_ownership(30.0, captaincy_rate=10.0) == pytest.approx(40.0)


def test_effective_ownership_none_when_ownership_unknown():
    """Missing ownership must never be silently coerced to 0 -- it's a genuinely different
    claim ("we don't know") from "this player is owned by nobody."""
    assert own.effective_ownership(None, captaincy_rate=0.0) is None


def _candidate(uid, position, mu, selected_by_percent):
    return {"player_uid": uid, "position": position, "mu": mu, "selected_by_percent": selected_by_percent}


def test_compute_eo_for_pool_groups_by_position():
    """top_ep_total is computed PER POSITION -- a midfielder's EP should never be compared
    against a forward's when estimating captaincy concentration."""
    pool = [
        _candidate("mid_top", "Midfielder", mu=8.0, selected_by_percent=40.0),
        _candidate("mid_low", "Midfielder", mu=4.0, selected_by_percent=40.0),
        _candidate("fwd_top", "Forward", mu=3.0, selected_by_percent=40.0),  # lower mu than mid_top, but top of its own group
    ]
    eo = own.compute_eo_for_pool(pool, captaincy_concentration=0.5)
    # fwd_top is the top of its OWN group (Forward), so it gets the full captaincy-rate boost
    # despite having lower absolute mu than mid_top -- proves position-scoping, not pool-wide ranking.
    assert eo["fwd_top"] == pytest.approx(40.0 + 40.0)
    assert eo["mid_top"] == pytest.approx(40.0 + 40.0)
    assert eo["mid_low"] < eo["mid_top"]


def test_compute_eo_for_pool_none_for_missing_ownership():
    pool = [_candidate("a", "Midfielder", mu=5.0, selected_by_percent=None)]
    eo = own.compute_eo_for_pool(pool, captaincy_concentration=0.5)
    assert eo["a"] is None


def test_compute_eo_for_pool_all_missing_ownership_in_group_is_still_none():
    """No owned players at all in a position group -- top_ep_total is None for that group,
    every candidate in it gets eo=None (never accidentally 0)."""
    pool = [_candidate("a", "Goalkeeper", mu=4.0, selected_by_percent=None)]
    eo = own.compute_eo_for_pool(pool, captaincy_concentration=0.5)
    assert eo["a"] is None


# ============================================================
# captain_risk_report
# ============================================================

def test_captain_risk_report_labels_template_when_above_xi_average():
    xi = [
        {"player_uid": "cap", "position": "Midfielder"},
        {"player_uid": "b", "position": "Defender"},
        {"player_uid": "c", "position": "Forward"},
    ]
    eo_by_uid = {"cap": 60.0, "b": 20.0, "c": 10.0}
    result = own.captain_risk_report(xi, eo_by_uid, captain_uid="cap")
    assert result["posture_label"] == "template"
    assert result["captain_eo"] == 60.0
    assert result["xi_avg_eo"] == pytest.approx((60.0 + 20.0 + 10.0) / 3)


def test_captain_risk_report_labels_differential_when_below_xi_average():
    xi = [
        {"player_uid": "cap", "position": "Midfielder"},
        {"player_uid": "b", "position": "Defender"},
        {"player_uid": "c", "position": "Forward"},
    ]
    eo_by_uid = {"cap": 2.0, "b": 50.0, "c": 50.0}
    result = own.captain_risk_report(xi, eo_by_uid, captain_uid="cap")
    assert result["posture_label"] == "differential"


def test_captain_risk_report_unknown_when_captain_eo_missing():
    xi = [{"player_uid": "cap", "position": "Midfielder"}, {"player_uid": "b", "position": "Defender"}]
    eo_by_uid = {"cap": None, "b": 50.0}
    result = own.captain_risk_report(xi, eo_by_uid, captain_uid="cap")
    assert result["posture_label"] == "unknown"
    assert result["captain_eo"] is None


def test_captain_risk_report_caveat_is_always_present_and_honest():
    xi = [{"player_uid": "cap", "position": "Midfielder"}]
    result = own.captain_risk_report(xi, {"cap": 10.0}, captain_uid="cap")
    assert "modeled estimate" in result["caveat"]
    assert "not measured rival-manager" in result["caveat"]
