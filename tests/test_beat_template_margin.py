"""scripts/beat_template_margin.py: Phase 3's chain table and G2's margin M
(docs/plans/2026-10_beat_the_template.md)."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import beat_template_margin as btm  # noqa: E402


def _control(gap_mean=1.0, gap_se=0.5):
    rows = []
    for gw in (2, 3):
        rows.append({"season": "2024-2025", "gw": gw, "squad_points": 60.0, "beats_real": None, "template_points": 55.0})
    for gw in (2, 3, 4):
        rows.append({"season": "2025-2026", "gw": gw, "squad_points": 62.0 + gw, "beats_real": 10.0,
                     "template_points": 58.0})
    return {
        "per_gameweek": rows,
        "template": {
            "pooled": {"n": 5, "gap_vs_gated": {"mean": gap_mean, "se": gap_se, "n": 5}},
            "by_season": {"2024-2025": {"n": 2, "template_points_per_gw": 55.0},
                          "2025-2026": {"n": 3, "template_points_per_gw": 58.0}},
        },
    }


SIM = {"gameweeks": [2, 3, 4], "weekly_points": [60.0, 58.0, 70.0], "weekly_hits": [0.0, 4.0, 0.0],
       "weekly_real_avg": [50.0, 55.0, 60.0]}
CUTOFFS = {"seasons": {"2024-2025": {"pace_per_gw": 64.0}, "2025-2026": {"pace_per_gw": 63.0}},
           "current_season": {"by_gameweek": [{"gameweek": 1, "total_points": 80.0},
                                              {"gameweek": 2, "total_points": 140.0},
                                              {"gameweek": 3, "total_points": 210.0},
                                              {"gameweek": 4, "total_points": 300.0}]}}


def test_evolving_loss_pairs_the_oracle_with_the_sims_net_points():
    e = btm.evolving_loss(_control(), SIM)
    # oracle 64, 65, 66 vs net 60, 54, 70 -> 4, 11, -4
    assert e["n"] == 3 and e["mean"] == pytest.approx(11 / 3, abs=1e-3)


def test_o_plus_counts_only_a_clear_optimism():
    assert btm.o_plus({"mean": 3.0, "se": 1.0}) == 3.0
    assert btm.o_plus({"mean": 1.5, "se": 1.0}) == 0.0
    assert btm.o_plus({"mean": -4.0, "se": 1.0}) == 0.0
    assert btm.o_plus({"mean": None, "se": None}) == 0.0


def test_margin_follows_the_declared_formula_with_its_floor():
    out = btm.margin(_control(), SIM, CUTOFFS, {"mean": 3.0, "se": 1.0})
    e = out["E"]["mean"]
    m24 = max(2.0, 64.0 + e + 3.0 - 55.0)
    m25 = max(2.0, 63.0 + e + 3.0 - 58.0)
    assert out["seasons"]["2024-2025"]["M"] == pytest.approx(m24, abs=1e-3)
    assert out["seasons"]["2025-2026"]["M"] == pytest.approx(m25, abs=1e-3)
    assert out["M_bar"] == pytest.approx((2 * m24 + 3 * m25) / 5, abs=1e-3)
    assert out["G2_met_by_control"] is False
    # a cutoff below the template plus the losses: the floor of 2.0 holds
    low = {"seasons": {"2024-2025": {"pace_per_gw": 40.0}, "2025-2026": {"pace_per_gw": 40.0}}}
    floor = btm.margin(_control(gap_mean=2.5, gap_se=0.5), SIM, low, {"mean": None, "se": None})
    assert floor["M_bar"] == 2.0 and floor["G2_met_by_control"] is True


def test_margin_is_none_until_every_input_exists():
    out = btm.margin(_control(), None, None, {"mean": None, "se": None})
    assert out["M_bar"] is None and out["G2_met_by_control"] is None


def test_live_optimism_pairs_the_sims_net_with_the_settled_live_weeks():
    leaderboard = {"status": "ok", "rows": [
        {"name": "FPL-Analyser (model)", "is_baseline": False, "weekly_points": {"2": 70.0, "3": 60.0, "4": 50.0},
         "weekly_hits": {"3": 4.0}},
    ]}
    team = {"weeks": [
        {"gameweek": 2, "realized_points": 62.0, "hit_cost": 0.0, "provisional": False},
        {"gameweek": 3, "realized_points": 67.0, "hit_cost": 0.0, "provisional": False},
        {"gameweek": 4, "realized_points": 73.0, "hit_cost": 0.0, "provisional": True},
    ]}
    o = btm.live_optimism(leaderboard, team)
    assert o["n"] == 2 and o["mean"] == pytest.approx((8.0 + (56.0 - 67.0)) / 2)
    old = {"status": "ok", "rows": [{"name": "m", "is_baseline": False, "total_points": 1.0}]}
    assert btm.live_optimism(old, team)["mean"] is None


def test_chain_lists_each_harness_with_its_cutoff():
    leaderboard = {"status": "ok", "rows": [
        {"name": "FPL-Analyser (model)", "is_baseline": False, "weekly_points": {"2": 70.0}, "weekly_hits": {}},
        {"name": "model (oracle)", "kind": "model_oracle", "is_baseline": True, "weekly_points": {"2": 80.0}},
    ]}
    team = {"weeks": [{"gameweek": 2, "realized_points": 62.0, "hit_cost": 0.0, "provisional": False,
                       "field_average": 81}]}
    rows = btm.chain(_control(), SIM, SIM, leaderboard, team, CUTOFFS)
    by = {(r["row"], r["season"]): r for r in rows}
    assert by[("oracle model", "2025-2026")]["vs_fpl_average"] == pytest.approx(10.0)
    assert by[("oracle model", "2024-2025")]["vs_fpl_average"] is None
    assert by[("gated template", "2025-2026")]["vs_cutoff"] == pytest.approx(58.0 - 63.0)
    assert by[("season sim control", "2025-2026")]["pts_per_gw"] == pytest.approx(184 / 3, abs=1e-3)
    assert by[("model (oracle)", "2026-2027")]["vs_fpl_average"] == pytest.approx(-1.0)
    # each 2026-27 row is set against the cutoff's pace over its own gameweeks: GW2 alone is
    # 140 - 80 = 60, not GW1-2's 70 or GW1-4's 75
    assert by[("live team", "2026-2027")]["cutoff_pace"] == pytest.approx(60.0)
    assert by[("leaderboard sim", "2026-2027")]["vs_cutoff"] == pytest.approx(10.0)
    team["weeks"] = [{"gameweek": gw, "realized_points": 60.0, "hit_cost": 0.0, "provisional": False,
                      "field_average": 50} for gw in (1, 2, 3)]
    rows = btm.chain(_control(), SIM, SIM, leaderboard, team, CUTOFFS)
    assert next(r for r in rows if r["row"] == "live team")["cutoff_pace"] == pytest.approx(70.0)  # GW1-3: 210 / 3
    # a row that skips GW2 is set against GW1 and GW3's gains: (80 + 70) / 2
    team["weeks"] = [w for w in team["weeks"] if w["gameweek"] != 2]
    rows = btm.chain(_control(), SIM, SIM, leaderboard, team, CUTOFFS)
    assert next(r for r in rows if r["row"] == "live team")["cutoff_pace"] == pytest.approx(75.0)
    assert by[("leaderboard sim", "2026-2027")]["pts_per_gw"] == 70.0
