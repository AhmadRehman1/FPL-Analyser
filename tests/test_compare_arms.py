"""scripts/compare_arms.py: the breakout promotion rule (docs/plans/2026-10_breakout_players.md, R4)."""

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
