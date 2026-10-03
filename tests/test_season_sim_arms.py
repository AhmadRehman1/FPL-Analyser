import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import aggregate_season_sim_arms as agg  # noqa: E402
import run_season_sim_arm as arm  # noqa: E402

from fpl_quant import backtest as bt  # noqa: E402
from fpl_quant import forward_season_sim as fss  # noqa: E402


def test_control_is_the_live_model_team_config(con):
    versions, changed = arm.arm_versions(con, arm._parse_args(["--label", "control"]))
    assert changed == {}
    assert versions == fss._resolve_versions(con, bt.active_recalibratable_versions(arm.RECALIBRATION_SEED_DIR))


def test_flags_change_one_setting_each(con):
    versions, changed = arm.arm_versions(con, arm._parse_args(["--label", "x", "--no-assists", "--chip-wait", "--transfer-threshold", "1.5"]))
    assert versions["assist_calibration_params_version"] is None
    assert versions["triple_captain_timing_params_version"] == versions["bench_boost_timing_params_version"] == 1
    assert versions["chip_wait_params_version"] == 1
    assert versions["accept_transfer_if_net_value_above"] == 1.5
    assert set(changed) == {
        "assist_calibration_params_version", "triple_captain_timing_params_version",
        "bench_boost_timing_params_version", "chip_wait_params_version", "accept_transfer_if_net_value_above",
    }


def _arm(label, points, hits, real=None):
    gws = list(range(2, 2 + len(points)))
    real = real or [50.0] * len(points)
    return {
        "label": label, "season": "2025-2026", "changed": {}, "gameweeks": gws, "weekly_points": points,
        "weekly_hits": hits, "weekly_real_avg": real, "n_transfers": 0, "chips_played": [],
        "real_benchmark": bt.season_real_benchmark(points, hits, real),
    }


def test_paired_difference_is_on_net_points():
    control = _arm("control", [50.0, 60.0, 55.0], [0.0, 0.0, 0.0])
    other = _arm("x", [58.0, 60.0, 59.0], [4.0, 0.0, 0.0])   # net 54, 60, 59 -> diffs +4, 0, +4
    d = agg.paired_difference(other, control)
    assert d["n"] == 3
    assert abs(d["mean"] - 8 / 3) < 1e-9
    assert d["se"] is not None


def test_summary_lists_every_arm_against_control():
    text = agg.summarize([_arm("control", [50.0, 60.0], [0.0, 0.0]), _arm("x", [52.0, 61.0], [0.0, 0.0])])
    assert "| control | live settings |" in text
    assert "+1.50" in text


def _chunk(label, start, points, complete=True):
    c = _arm(label, points, [0.0] * len(points))
    c["gameweeks"] = list(range(start, start + len(points)))
    c.update(start_gameweek=start, end_gameweek=start + len(points) - 1, complete=complete, n_transfers=1,
             chips_played=[(start, "bench_boost")])
    return c


def test_chunks_of_one_arm_are_stitched_in_gameweek_order():
    late, early = _chunk("control", 19, [60.0, 40.0]), _chunk("control", 2, [50.0])
    (arm,) = agg.merge_chunks([late, early])
    assert arm["gameweeks"] == [2, 19, 20]
    assert arm["weekly_points"] == [50.0, 60.0, 40.0]
    assert arm["n_transfers"] == 2
    assert arm["chips_played"] == [(2, "bench_boost"), (19, "bench_boost")]
    assert arm["real_benchmark"] == bt.season_real_benchmark([50.0, 60.0, 40.0], [0.0] * 3, [50.0] * 3)
    assert arm["complete"]


def test_a_cut_off_chunk_counts_what_it_scored_and_is_flagged():
    arms = agg.merge_chunks([_chunk("control", 2, [50.0]), _chunk("x", 2, [55.0], complete=False)])
    assert [a["complete"] for a in arms] == [True, False]
    text = agg.summarize(arms)
    assert "incomplete: GW2-2 stopped after 1" in text
    assert "+5.00" in text


def test_multi_transfers_flag_switches_on_two_transfer_moves(con):
    versions, changed = arm.arm_versions(con, arm._parse_args(["--label", "x", "--multi-transfers"]))
    assert versions["multi_transfer_params_version"] == changed["multi_transfer_params_version"] == 1
    control, _ = arm.arm_versions(con, arm._parse_args(["--label", "control"]))
    assert control["multi_transfer_params_version"] is None


def test_chip_option_value_flag_uses_chip_wait_v2_with_the_wide_window(con):
    versions, changed = arm.arm_versions(con, arm._parse_args(["--label", "x", "--chip-option-value"]))
    assert versions["chip_wait_params_version"] == 2
    assert versions["triple_captain_timing_params_version"] == versions["bench_boost_timing_params_version"] == 1
    assert set(changed) == {
        "chip_wait_params_version", "triple_captain_timing_params_version", "bench_boost_timing_params_version",
    }
