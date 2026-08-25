import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_report import _would_regress_track_record  # noqa: E402


def test_no_regression_guard_needed_when_this_run_has_a_real_backtest():
    # This run's own DB has a real backtest_run_id -- always write it, whatever was there before.
    new = {"backtest_run_id": 7, "n_gameweek_steps": 71}
    existing = {"backtest_run_id": 3, "n_gameweek_steps": 40}
    assert _would_regress_track_record(new, existing) is False


def test_blocks_overwriting_a_real_committed_backtest_with_an_empty_one():
    # The exact real scenario: a weekly scripts/run_backtest.py run committed a real track
    # record, then this script's own next twice-daily run -- against a fresh, backtest-less
    # DB -- would otherwise silently wipe it back to "no backtest yet".
    new = {"backtest_run_id": None, "n_gameweek_steps": None}
    existing = {"backtest_run_id": 3, "n_gameweek_steps": 40}
    assert _would_regress_track_record(new, existing) is True


def test_no_regression_guard_needed_when_nothing_committed_yet():
    new = {"backtest_run_id": None, "n_gameweek_steps": None}
    assert _would_regress_track_record(new, existing_track_record=None) is False


def test_no_regression_guard_needed_when_existing_file_also_has_no_backtest():
    # Nothing real to lose -- both this run and the last committed file are placeholders.
    new = {"backtest_run_id": None, "n_gameweek_steps": None}
    existing = {"backtest_run_id": None, "n_gameweek_steps": None}
    assert _would_regress_track_record(new, existing) is False
