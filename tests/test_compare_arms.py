"""scripts/compare_arms.py: the breakout promotion rule (docs/plans/2026-10_breakout_players.md, R4), and rules R5, confirm and R6."""

import copy
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import compare_arms as ca  # noqa: E402

S = "2025-2026"


def _run(beats_real, mae=1.08, resid=0.60, n=300, widened_n=400, key="duckdb-1", log=-0.53):
    return {
        "progress": {"complete": True},
        "db_cache_key": key,
        "headline_by_season": {S: {"ep_total_calibration_mae": mae, "log_score_minutes_mean": log}},
        "per_gameweek": [{"season": S, "gw": gw, "beats_real": v} for gw, v in enumerate(beats_real, start=1)]
        + [{"season": "2024-2025", "gw": 5, "beats_real": None}],
        "breakout": {S: {
            "breakout": {"n_player_steps": n, "mean_resid": resid, "mae": 2.0},
            "breakout_widened": {"n_player_steps": widened_n, "mean_resid": resid, "mae": 2.0},
        }, "2024-2025": None},
    }


CONTROL = [10.0, 12.0, 8.0, 11.0]


def test_pass_when_points_level_mae_flat_and_breakout_cut_by_a_third():
    out = ca.compare(_run(CONTROL), _run([11.0, 12.0, 9.0, 11.0], mae=1.081, resid=0.39))
    assert out["verdict"] == "PASS"
    assert out["checks"]["points"]["mean"] == pytest.approx(0.5)
    assert out["checks"]["breakout"]["limit"] == pytest.approx(0.4)


@pytest.mark.parametrize("arm, failing", [
    (_run([9.0, 11.0, 7.0, 10.0], resid=0.2), "points"),         # m = -1.0
    (_run([9.9, 11.9, 7.9, 10.9], resid=0.2), "points"),         # m = -0.1 every week: m + SE < 0
    (_run(CONTROL, mae=1.0825, resid=0.2), "mae"),               # +0.0025
    (_run(CONTROL, resid=0.41), "breakout"),                     # above two thirds of 0.60
])
def test_each_failing_reason(arm, failing):
    out = ca.compare(_run(CONTROL), arm)
    assert out["verdict"] == "FAIL"
    assert [k for k, c in out["checks"].items() if not c["pass"]] == [failing]


def test_m_plus_se_rule_lets_a_noisy_small_loss_through_but_not_a_large_one():
    noisy = _run([10.5, 11.0, 8.6, 11.5], resid=0.2)  # diffs +0.5 -1.0 +0.6 +0.5: m = 0.15
    assert ca.compare(_run(CONTROL), noisy)["checks"]["points"]["pass"]
    assert not ca.compare(_run(CONTROL), _run([9.6, 11.6, 7.6, 10.6], resid=0.2))["checks"]["points"]["pass"]  # m = -0.4


def test_control_not_under_predicting_the_group_stops():
    out = ca.compare(_run(CONTROL, resid=-0.1), _run(CONTROL, resid=-0.3))
    assert out["verdict"] == "FAIL"
    assert "stop and report" in out["checks"]["breakout"]["reason"]


def test_small_group_falls_back_to_the_declared_widening_then_stops():
    widened = ca.compare(_run(CONTROL, n=150), _run(CONTROL, n=150, resid=0.3))
    assert widened["checks"]["breakout"]["group"].startswith("breakout_widened")
    both_small = ca.compare(_run(CONTROL, n=150, widened_n=180), _run(CONTROL, resid=0.3))
    assert "too small" in both_small["checks"]["breakout"]["reason"]


@pytest.mark.parametrize("mutate, message", [
    (lambda r: r["progress"].update(complete=False), "incomplete"),
    (lambda r: r.update(db_cache_key=None), "no db_cache_key"),
    (lambda r: r.update(db_cache_key="duckdb-2"), "different cached DBs"),
])
def test_refuses_runs_that_cannot_be_compared(mutate, message):
    arm = copy.deepcopy(_run(CONTROL))
    mutate(arm)
    with pytest.raises(ca.Refused, match=message):
        ca.compare(_run(CONTROL), arm)


def test_cli_exit_codes(tmp_path, capsys):
    import json
    control, arm = tmp_path / "c.json", tmp_path / "a.json"
    control.write_text(json.dumps(_run(CONTROL)))
    arm.write_text(json.dumps(_run([11.0, 12.0, 9.0, 11.0], resid=0.3)))
    assert ca.main([str(control), str(arm)]) == 0
    assert capsys.readouterr().out.startswith("PASS")
    arm.write_text(json.dumps(_run(CONTROL, key="duckdb-9")))
    assert ca.main([str(control), str(arm)]) == 2
    # a crashed scoreboard leaves an empty file: refused, not a FAIL
    arm.write_text("")
    assert ca.main([str(control), str(arm)]) == 2
    arm.write_text(json.dumps({**_run(CONTROL), "headline_by_season": {}}))
    assert ca.main([str(control), str(arm)]) == 2


# the follow-up rule: control is the live model (breakout fixed), baseline the model before the fix
BEFORE_FIX = _run(CONTROL, resid=0.30)


def test_follow_up_passes_when_it_keeps_the_fix_and_improves_the_log_score():
    out = ca.compare(_run(CONTROL, resid=0.10), _run(CONTROL, resid=0.15, log=-0.52), BEFORE_FIX)
    assert out["verdict"] == "PASS"
    assert out["checks"]["breakout"]["limit"] == pytest.approx(0.2)
    assert out["checks"]["log_score"]["delta"] == pytest.approx(0.01)


@pytest.mark.parametrize("arm, failing", [
    (_run(CONTROL, resid=0.25, log=-0.52), "breakout"),  # gives back more than the rule allows
    (_run(CONTROL, resid=0.15, log=-0.53), "log_score"),  # no better than control
    (_run([9.0, 11.0, 7.0, 10.0], resid=0.15, log=-0.52), "points"),
])
def test_follow_up_failing_reasons(arm, failing):
    out = ca.compare(_run(CONTROL, resid=0.10), arm, BEFORE_FIX)
    assert [k for k, c in out["checks"].items() if not c["pass"]] == [failing]


def test_follow_up_refuses_a_baseline_on_another_db():
    with pytest.raises(ca.Refused, match="different cached DBs"):
        ca.compare(_run(CONTROL), _run(CONTROL), _run(CONTROL, key="duckdb-2"))


def test_cli_baseline_flag(tmp_path, capsys):
    import json
    files = {}
    for name, run in (("c", _run(CONTROL, resid=0.10)), ("a", _run(CONTROL, resid=0.15, log=-0.52)), ("b", BEFORE_FIX)):
        files[name] = tmp_path / f"{name}.json"
        files[name].write_text(json.dumps(run))
    assert ca.main([str(files["c"]), str(files["a"]), "--baseline", str(files["b"])]) == 0
    out = capsys.readouterr().out
    assert out.startswith("PASS") and "before fix" in out and "(iv)" in out
    assert ca.main([str(files["c"]), str(files["a"]), "--baseline"]) == 2


# ---- rule R5 (docs/plans/2026-10_premium_and_form_returns.md): the target group's component sum ----

def _r5(beats_real, *, prem=None, form=None, form_n=300, form_widened_n=400, breakout_resid=0.10, **kw):
    run = _run(beats_real, resid=breakout_resid, **kw)
    prem = prem or {"appearance": 0.0, "goals": -0.27, "assists": 0.23, "cleansheet": -0.01, "other": 0.16}
    run["price_band"] = {"9.0+": {**prem, "total": sum(prem.values())}}
    form = form or {"appearance": 0.05, "goals": 0.20, "assists": 0.10, "cleansheet": 0.0, "other": 0.05}
    run["form_groups"] = {S: {
        "in_form": {"n_player_steps": form_n, "components": form,
                    "component_abs_sum": round(sum(abs(v) for v in form.values()), 4)},
        "in_form_widened": {"n_player_steps": form_widened_n, "components": form,
                            "component_abs_sum": round(sum(abs(v) for v in form.values()), 4)},
    }, "2024-2025": None}
    return run


def test_r5_passes_on_the_component_sum_not_the_total():
    # premium errors that cancel: control's sum is 0.67; the arm fixes goals and assists
    fixed = {"appearance": 0.0, "goals": -0.05, "assists": 0.05, "cleansheet": -0.01, "other": 0.16}
    out = ca.compare(_r5(CONTROL), _r5(CONTROL, prem=fixed), rule="r5", target="price_band=9.0+")
    assert out["verdict"] == "PASS"
    assert out["checks"]["target"]["control"] == pytest.approx(0.67)
    assert out["checks"]["target"]["limit"] == pytest.approx(0.67 * 2 / 3)
    # moving one component until the total looks fixed is not enough (bps-tau 7 in batch B1)
    bonus_only = {"appearance": 0.0, "goals": -0.27, "assists": 0.23, "cleansheet": -0.01, "other": 0.03}
    assert not ca.compare(_r5(CONTROL), _r5(CONTROL, prem=bonus_only), rule="r5", target="price_band=9.0+")["checks"]["target"]["pass"]


@pytest.mark.parametrize("arm, failing", [
    (_r5([9.0, 11.0, 7.0, 10.0], form={"goals": 0.05}), "points"),
    (_r5(CONTROL, mae=1.0825, form={"appearance": 0.0, "goals": 0.05, "assists": 0.0, "cleansheet": 0.0, "other": 0.0}), "mae"),
    (_r5(CONTROL), "target"),  # in_form sum unchanged
    (_r5(CONTROL, breakout_resid=0.19, form={"appearance": 0.0, "goals": 0.05, "assists": 0.0, "cleansheet": 0.0, "other": 0.0}),
     "breakout_limit"),
])
def test_r5_each_failing_reason(arm, failing):
    out = ca.compare(_r5(CONTROL), arm, rule="r5", target="in_form")
    assert out["verdict"] == "FAIL"
    assert [k for k, c in out["checks"].items() if not c["pass"]] == [failing]


def test_r5_in_form_uses_the_widening_when_the_group_is_small_then_stops():
    small = _r5(CONTROL, form_n=100)
    out = ca.compare(small, _r5(CONTROL, form_n=100), rule="r5", target="in_form")
    assert out["checks"]["target"]["group"] == "in_form_widened (n=400)"
    tiny = _r5(CONTROL, form_n=100, form_widened_n=120)
    out = ca.compare(tiny, tiny, rule="r5", target="in_form")
    assert not out["checks"]["target"]["pass"] and "too small" in out["checks"]["target"]["reason"]
    # a summary written before the form_groups block existed fails with a reason, not a crash
    old = _r5(CONTROL)
    del old["form_groups"]
    assert "missing" in ca.compare(old, old, rule="r5", target="in_form")["checks"]["target"]["reason"]


def test_r5_reads_a_split_block_without_its_sum():
    """B1's summaries predate component_abs_sum; the sum comes from the components."""
    assert ca.component_sum({"appearance": -0.0036, "goals": -0.2671, "assists": 0.2267, "cleansheet": -0.0066,
                             "other": 0.1576, "total": 0.107}) == pytest.approx(0.6616)
    assert ca.component_sum({"goals": 0.1}) is None



def test_r5_cheap_band_target():
    """Round 3 (docs/reports/2026-10_premium_and_form_returns.md): the <5.0m band, whose error is
    mostly appearance points."""
    cheap = {"appearance": -0.11, "goals": -0.02, "assists": 0.01, "cleansheet": -0.03, "other": -0.01}
    fixed = {**cheap, "appearance": -0.03}
    control, arm = _r5(CONTROL), _r5(CONTROL)
    control["price_band"]["<5.0"] = {**cheap, "total": sum(cheap.values())}
    arm["price_band"]["<5.0"] = {**fixed, "total": sum(fixed.values())}
    out = ca.compare(control, arm, rule="r5", target="price_band=<5.0")
    assert out["checks"]["target"]["control"] == pytest.approx(0.18)
    assert out["checks"]["target"]["arm"] == pytest.approx(0.10) and out["verdict"] == "PASS"
    # the 9.0m+ block is untouched, so judged on it the same arm fails (iii)
    assert ca.compare(control, arm, rule="r5", target="price_band=9.0+")["verdict"] == "FAIL"

def test_r5_needs_a_target_and_no_baseline():
    with pytest.raises(ca.Refused):
        ca.compare(_r5(CONTROL), _r5(CONTROL), rule="r5", target="9.0+")
    with pytest.raises(ca.Refused):
        ca.compare(_r5(CONTROL), _r5(CONTROL), _r5(CONTROL), rule="r5", target="in_form")


def test_cli_rule_r5(tmp_path, capsys):
    import json

    fixed = {"appearance": 0.0, "goals": -0.05, "assists": 0.05, "cleansheet": -0.01, "other": 0.16}
    c, a = tmp_path / "c.json", tmp_path / "a.json"
    c.write_text(json.dumps(_r5(CONTROL)))
    a.write_text(json.dumps(_r5(CONTROL, prem=fixed)))
    assert ca.main([str(c), str(a), "--rule", "r5", "--target", "price_band=9.0+"]) == 0
    assert "(iii) target  pass" in capsys.readouterr().out
    assert ca.main([str(c), str(a), "--rule", "r5", "--target", "in_form"]) == 1
    assert ca.main([str(c), str(a), "--rule", "r5"]) == 2
    assert ca.main([str(c), str(a), "--rule", "r9", "--target", "in_form"]) == 2


def test_r5_in_form_pass_and_refusals():
    better = {"appearance": 0.05, "goals": 0.05, "assists": 0.05, "cleansheet": 0.0, "other": 0.05}
    out = ca.compare(_r5(CONTROL), _r5(CONTROL, form=better), rule="r5", target="in_form")
    assert out["verdict"] == "PASS" and out["checks"]["target"]["group"] == "in_form (n=300)"
    with pytest.raises(ca.Refused, match="different cached DBs"):
        ca.compare(_r5(CONTROL), _r5(CONTROL, form=better, key="duckdb-2"), rule="r5", target="in_form")
    # a target without rule r5 would quietly be judged under R4
    with pytest.raises(ca.Refused, match="--rule r5"):
        ca.compare(_r5(CONTROL), _r5(CONTROL), target="in_form")


def test_r5_price_band_needs_a_2025_26_only_run():
    both = _r5(CONTROL)
    both["headline_by_season"]["2024-2025"] = both["headline_by_season"][S]
    with pytest.raises(ca.Refused, match="pools seasons"):
        ca.compare(_r5(CONTROL), both, rule="r5", target="price_band=9.0+")
    # the in-form block is per season already
    assert ca.compare(_r5(CONTROL), both, rule="r5", target="in_form")["verdict"] == "FAIL"


def test_r5_breakout_limit_is_on_the_declared_group():
    out = ca.compare(_r5(CONTROL, n=100), _r5(CONTROL, n=100), rule="r5", target="in_form")
    assert not out["checks"]["breakout_limit"]["pass"]
    assert "declared breakout group too small" in out["checks"]["breakout_limit"]["reason"]


# ---- rule confirm (round 3's confirmation): both seasons, pooled points, each season's MAE ----

P = "2024-2025"


def _two_seasons(prior_points, points, mae_prior=1.10, mae=1.07, resid=0.10, key="duckdb-1"):
    """2025-26 rows carry beats_real (squad points less a real average of 50); 2024-25 rows have
    no real average manager, so squad_points alone, as in the data."""
    run = _run(points, mae=mae, resid=resid, key=key)
    run["headline_by_season"][P] = {"ep_total_calibration_mae": mae_prior, "log_score_minutes_mean": -0.6}
    run["per_gameweek"] = [
        {**r, "squad_points": r["beats_real"] + 50.0} for r in run["per_gameweek"] if r["season"] == S
    ] + [{"season": P, "gw": gw, "beats_real": None, "squad_points": v} for gw, v in enumerate(prior_points, start=1)]
    return run


def test_confirm_pools_both_seasons_points():
    # 2025-26 alone: -0.5 a week; with 2024-25's +1.0 a week the pooled mean is +0.25
    control = _two_seasons([5.0, 6.0, 7.0, 8.0], CONTROL)
    arm = _two_seasons([6.0, 7.0, 8.0, 9.0], [9.5, 11.5, 7.5, 10.5], mae_prior=1.09, mae=1.05)
    out = ca.compare(control, arm, rule="confirm")
    assert out["checks"]["points"]["n"] == 8 and out["checks"]["points"]["mean"] == pytest.approx(0.25)
    assert out["verdict"] == "PASS"
    assert ca.compare(control, arm, rule="r4")["checks"]["points"]["mean"] == pytest.approx(-0.5)


def test_confirm_needs_mae_no_worse_in_each_season():
    control = _two_seasons([5.0, 6.0, 7.0, 8.0], CONTROL)
    worse_prior = _two_seasons([5.0, 6.0, 7.0, 8.0], CONTROL, mae_prior=1.1001, mae=1.0)
    out = ca.compare(control, worse_prior, rule="confirm")
    assert out["verdict"] == "FAIL" and [k for k, c in out["checks"].items() if not c["pass"]] == ["mae_by_season"]
    assert out["checks"]["mae_by_season"]["seasons"][S]["delta"] == pytest.approx(-0.07)
    high = _two_seasons([5.0, 6.0, 7.0, 8.0], CONTROL, resid=0.19)
    assert [k for k, c in ca.compare(control, high, rule="confirm")["checks"].items() if not c["pass"]] == ["breakout_limit"]


def test_confirm_refusals():
    two = _two_seasons([5.0, 6.0, 7.0, 8.0], CONTROL)
    with pytest.raises(ca.Refused, match="same seasons"):
        ca.compare(_run(CONTROL), _run(CONTROL), rule="confirm")
    with pytest.raises(ca.Refused, match="same seasons"):
        ca.compare(two, _run(CONTROL), rule="confirm")
    with pytest.raises(ca.Refused, match="no --baseline"):
        ca.compare(two, two, two, rule="confirm")
    with pytest.raises(ca.Refused, match="different cached DBs"):
        ca.compare(two, _two_seasons([5.0, 6.0, 7.0, 8.0], CONTROL, key="duckdb-2"), rule="confirm")
    with pytest.raises(ca.Refused, match="--rule r5"):
        ca.compare(two, two, rule="confirm", target="in_form")
    # a summary written before per-gameweek squad_points: 2024-25 can't be paired, so refuse
    # rather than judge 2025-26 alone
    old = _two_seasons([5.0, 6.0, 7.0, 8.0], CONTROL)
    for r in old["per_gameweek"]:
        r.pop("squad_points")
    with pytest.raises(ca.Refused, match="2024-2025: 0 gameweeks"):
        ca.compare(old, old, rule="confirm")


def test_cli_rule_confirm(tmp_path, capsys):
    import json

    c, a = tmp_path / "c.json", tmp_path / "a.json"
    c.write_text(json.dumps(_two_seasons([5.0, 6.0, 7.0, 8.0], CONTROL)))
    a.write_text(json.dumps(_two_seasons([6.0, 7.0, 8.0, 9.0], [9.5, 11.5, 7.5, 10.5], mae_prior=1.09, mae=1.05)))
    assert ca.main([str(c), str(a), "--rule", "confirm"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("PASS") and "EP MAE by season pass" in out and "2024-2025 1.1000 -> 1.0900" in out


# ============================================================
# rule r6 (docs/plans/2026-10_beat_the_template.md, Req 5 and Req 9)
# ============================================================

P = "2024-2025"


def _r6_run(prior, points, *, mae_prior=1.10, mae=1.07, overlap=6.0, non_template=20.0, key="duckdb-1",
            template=None):
    """A two-season summary: per-gameweek squad points, overlap and non-template points."""
    rows = [{"season": P, "gw": gw, "squad_points": v, "template_xi_overlap": overlap, "non_template_points": non_template}
            for gw, v in enumerate(prior, start=2)]
    rows += [{"season": S, "gw": gw, "squad_points": v, "template_xi_overlap": overlap, "non_template_points": non_template}
             for gw, v in enumerate(points, start=1)]
    return {
        "progress": {"complete": True}, "db_cache_key": key,
        "headline_by_season": {P: {"ep_total_calibration_mae": mae_prior}, S: {"ep_total_calibration_mae": mae}},
        "per_gameweek": rows,
        "template": {"pooled": template or {"n": len(rows), "gap_vs_gated": {"mean": -1.0, "se": 0.5, "n": len(rows)},
                                            "gap_vs_ungated": {"mean": 0.5, "se": 0.5, "n": len(rows)},
                                            "xi_overlap_mean": overlap, "non_template_points_per_gw": non_template}},
    }


C_PRIOR, C_POINTS = [50.0, 60.0, 55.0], [40.0, 70.0, 52.0]


def test_r6_passes_when_m_minus_se_clears_zero_and_mae_is_within_tolerance():
    arm = _r6_run([52.0, 62.0, 57.0], [42.0, 73.0, 54.0], mae_prior=1.105, mae=1.079)  # diffs 2 2 2 2 3 2
    out = ca.compare(_r6_run(C_PRIOR, C_POINTS), arm, rule="r6")
    assert out["verdict"] == "PASS"
    pts = out["checks"]["points"]
    assert pts["n"] == 6 and pts["changed"] == 6
    assert pts["mean"] == pytest.approx(13 / 6)
    assert pts["m_minus_se"] > 0.1
    assert out["checks"]["mae_by_season"]["seasons"][S]["delta"] == pytest.approx(0.009)


def test_r6_fails_when_m_minus_se_is_below_zero():
    arm = _r6_run([52.0, 58.0, 57.0], [42.0, 66.0, 54.0])  # diffs 2 -2 2 2 -4 2: m = 1/3, SE ~ 1.0
    out = ca.compare(_r6_run(C_PRIOR, C_POINTS), arm, rule="r6")
    assert out["verdict"] == "FAIL" and not out["checks"]["points"]["pass"]


def test_r6_fails_on_mae_in_either_season():
    arm = _r6_run([52.0, 62.0, 57.0], [42.0, 73.0, 54.0], mae_prior=1.111)  # 2024-25 +0.011
    out = ca.compare(_r6_run(C_PRIOR, C_POINTS), arm, rule="r6")
    assert out["verdict"] == "FAIL"
    assert not out["checks"]["mae_by_season"]["pass"] and out["checks"]["points"]["pass"]


def test_r6_asks_for_a_replicate_near_the_threshold_then_judges_the_mean():
    near = _r6_run([50.5, 60.4, 55.6], [40.5, 70.4, 52.6])  # diffs .5 .4 .6 .5 .4 .6: m .5, SE ~ .037
    c = _r6_run(C_PRIOR, C_POINTS)
    out = ca.compare(c, _r6_run([50.1, 60.0, 55.1], [40.0, 70.1, 52.0]), rule="r6")  # m - SE within 0.1 of 0
    assert out["verdict"] == "REPLICATE" and out["replicate_required"]
    rep = ca.compare(c, _r6_run([50.1, 60.0, 55.1], [40.0, 70.1, 52.0]), rule="r6", replicate=(c, near))
    assert rep["replicated"] and not rep["replicate_required"]
    assert rep["checks"]["points"]["mean"] == pytest.approx((0.3 + 0.2 + 0.35 + 0.25 + 0.25 + 0.3) / 6)
    assert rep["verdict"] == "PASS"


def test_r6_replicate_uses_the_mean_of_both_runs_maes():
    c = _r6_run(C_PRIOR, C_POINTS)
    arm = _r6_run([52.0, 62.0, 57.0], [42.0, 73.0, 54.0], mae=1.082)       # +0.012 alone
    arm_b = _r6_run([52.0, 62.0, 57.0], [42.0, 73.0, 54.0], mae=1.076)     # mean +0.009
    assert ca.compare(c, arm, rule="r6")["verdict"] == "FAIL"
    assert ca.compare(c, arm, rule="r6", replicate=(c, arm_b))["verdict"] == "PASS"


def test_r6_req9_blocks_a_collapse_into_the_template():
    arm = _r6_run([52.0, 62.0, 57.0], [42.0, 73.0, 54.0], overlap=7.2, non_template=19.0)
    out = ca.compare(_r6_run(C_PRIOR, C_POINTS), arm, rule="r6")
    assert out["checks"]["points"]["pass"]
    assert out["req9"]["blocks_promotion"] and out["verdict"] == "BLOCKED"
    assert out["req9"]["overlap_rise"] == pytest.approx(1.2)


@pytest.mark.parametrize("overlap, non_template, blocked", [
    (7.2, 20.5, False),  # overlap up a player, but the points from outside the template rose too
    (6.9, 19.0, False),  # less than a player more overlap
    (7.0, 20.0, True),   # exactly one more, no gain outside the template
])
def test_r6_req9_branches(overlap, non_template, blocked):
    arm = _r6_run([52.0, 62.0, 57.0], [42.0, 73.0, 54.0], overlap=overlap, non_template=non_template)
    assert ca.compare(_r6_run(C_PRIOR, C_POINTS), arm, rule="r6")["req9"]["blocks_promotion"] is blocked


def test_r6_req9_pauses_the_programme_at_nine_of_eleven():
    arm = _r6_run([52.0, 62.0, 57.0], [42.0, 73.0, 54.0], overlap=9.0, non_template=8.0)
    c = _r6_run(C_PRIOR, C_POINTS, overlap=8.5, non_template=7.0)
    out = ca.compare(c, arm, rule="r6")
    assert out["req9"]["pause_for_owner"] and not out["req9"]["blocks_promotion"]
    assert "PAUSE FOR OWNER REVIEW" in ca._fmt(out)


def test_r6_without_template_rows_judges_points_and_says_req9_was_not_measured():
    c, arm = _r6_run(C_PRIOR, C_POINTS), _r6_run([52.0, 62.0, 57.0], [42.0, 73.0, 54.0])
    for run in (c, arm):
        for r in run["per_gameweek"]:
            del r["template_xi_overlap"], r["non_template_points"]
        del run["template"]
    out = ca.compare(c, arm, rule="r6")
    assert out["verdict"] == "PASS" and not out["req9"]["measured"]
    assert "Req 9 not measured" in ca._fmt(out)


def _drop_season_rows(run, season, keep):
    rows = [r for r in run["per_gameweek"] if r["season"] == season][:keep]
    run["per_gameweek"] = [r for r in run["per_gameweek"] if r["season"] != season] + rows


@pytest.mark.parametrize("mutate, message", [
    (lambda c, a: a.update(db_cache_key="duckdb-2"), "different cached DBs"),
    (lambda c, a: a.update(progress={"complete": False}), "incomplete"),
    (lambda c, a: a["headline_by_season"].pop(P), "different seasons"),
    (lambda c, a: [r["headline_by_season"].pop(P) for r in (c, a)], "every season"),
    (lambda c, a: _drop_season_rows(a, P, 1), "2024-2025: 1 gameweeks"),
])
def test_r6_refusals(mutate, message):
    c, arm = _r6_run(C_PRIOR, C_POINTS), _r6_run([52.0, 62.0, 57.0], [42.0, 73.0, 54.0])
    mutate(c, arm)
    with pytest.raises(ca.Refused, match=message):
        ca.compare(c, arm, rule="r6")


def test_r6_refuses_a_replicate_on_another_db_and_replicate_without_r6():
    c, arm = _r6_run(C_PRIOR, C_POINTS), _r6_run([52.0, 62.0, 57.0], [42.0, 73.0, 54.0])
    with pytest.raises(ca.Refused, match="different cached DBs"):
        ca.compare(c, arm, rule="r6", replicate=(c, _r6_run(C_PRIOR, C_POINTS, key="duckdb-9")))
    with pytest.raises(ca.Refused, match="rule r6's"):
        ca.compare(c, arm, rule="confirm", replicate=(c, arm))


def test_cli_rule_r6(tmp_path, capsys):
    import json

    c, arm = tmp_path / "c.json", tmp_path / "a.json"
    c.write_text(json.dumps(_r6_run(C_PRIOR, C_POINTS)))
    arm.write_text(json.dumps(_r6_run([52.0, 62.0, 57.0], [42.0, 73.0, 54.0])))
    assert ca.main([str(c), str(arm), "--rule", "r6"]) == 0
    printed = capsys.readouterr().out
    assert printed.startswith("PASS") and "vs gated template -1.00 ± 0.50" in printed and "6 changed" in printed
    assert ca.main([str(c), str(arm), "--rule", "r6", "--replicate", str(c), str(arm)]) == 0
    near = tmp_path / "near.json"
    near.write_text(json.dumps(_r6_run([50.1, 60.0, 55.1], [40.0, 70.1, 52.0])))
    assert ca.main([str(c), str(near), "--rule", "r6"]) == 3
    assert "replicate required" in capsys.readouterr().out
    assert ca.main([str(c), str(arm), "--rule", "r6", "--replicate", str(c)]) == 2
