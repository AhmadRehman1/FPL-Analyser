import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from realize_decision_log_outcomes import (  # noqa: E402
    _actual_action_taken,
    _build_name_to_element_id,
    _normalize_decision_row_names,
)


def test_actual_action_taken_reports_the_active_chip_first():
    assert _actual_action_taken({"event_transfers": 2}, "bboost") == "bboost"


def test_actual_action_taken_reports_transfer_made_when_transfers_happened():
    assert _actual_action_taken({"event_transfers": 1}, None) == "transfer_made"


def test_actual_action_taken_reports_hold_when_nothing_changed():
    assert _actual_action_taken({"event_transfers": 0}, None) == "hold"


def test_actual_action_taken_missing_event_transfers_defaults_to_hold():
    assert _actual_action_taken({}, None) == "hold"


# ============================================================
# _build_name_to_element_id -- normalized, ambiguity-safe name -> element id map
# ============================================================

def test_build_name_to_element_id_normalizes_accents_and_case():
    # Same underlying problem entity_resolution.normalize_name()'s own docstring names --
    # dim_player.canonical_name (frozen at first ingestion) can be spelled differently than
    # whatever bootstrap-static reports right now.
    mapping = _build_name_to_element_id({101: "Aarón Anselmino"})
    assert mapping == {"aaron anselmino": 101}


def test_build_name_to_element_id_drops_ambiguous_duplicate_names():
    # Two different elements normalizing to the same name -- real FPL history has genuine
    # duplicate full names -- must never silently resolve to either one.
    mapping = _build_name_to_element_id({101: "Ben Davies", 202: "Ben Davies"})
    assert mapping == {}


def test_build_name_to_element_id_keeps_unambiguous_names():
    mapping = _build_name_to_element_id({101: "Ben Davies", 202: "Someone Else"})
    assert mapping == {"ben davies": 101, "someone else": 202}


# ============================================================
# _normalize_decision_row_names -- matches _build_name_to_element_id's key format
# ============================================================

def test_normalize_decision_row_names_normalizes_only_the_name_fields():
    row = {
        "recommended_action": "transfer_now",
        "recommended_transfer_out": "Aarón Anselmino",
        "recommended_transfer_in": "Someone Else",
        "recommended_captain": None,
        "recommended_chip": None,
    }
    normalized = _normalize_decision_row_names(row)
    assert normalized["recommended_transfer_out"] == "aaron anselmino"
    assert normalized["recommended_transfer_in"] == "someone else"
    assert normalized["recommended_captain"] is None
    assert normalized["recommended_action"] == "transfer_now"  # non-name fields untouched


def test_normalize_decision_row_names_does_not_mutate_the_original():
    row = {"recommended_transfer_out": "Aarón Anselmino"}
    _normalize_decision_row_names(row)
    assert row["recommended_transfer_out"] == "Aarón Anselmino"


# ============================================================
# Catch-up sweep: realize EVERY finished logged gameweek before current_event, not just
# current_event - 1 (FPL's is_current lags a full cycle -- see the module docstring).
# ============================================================

import json  # noqa: E402

import realize_decision_log_outcomes as rdlo  # noqa: E402


def _write_log(dir_, entry_id, gw, realized=False):
    (dir_ / f"{entry_id}_2026-2027_gw{gw}.json").write_text(json.dumps({
        "entry_id": entry_id, "target_season": "2026-2027", "target_gameweek": gw,
        "recommended_action": "hold", "recommended_chip": None,
        "realized_points_actual": 55 if realized else None,
    }))


def test_logged_gameweeks_enumerates_committed_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(rdlo, "DECISION_LOG_DIR", tmp_path)
    _write_log(tmp_path, 7139944, 2)
    _write_log(tmp_path, 1305242, 2)
    _write_log(tmp_path, 7139944, 3)
    _write_log(tmp_path, 7139944, 4)
    assert rdlo._logged_gameweeks("2026-2027") == [2, 3, 4]


def test_main_realizes_the_just_finished_gameweek_while_is_current_still_lags(tmp_path, monkeypatch):
    # The real "now": GW2 finished + data_checked, but FPL's is_current is still 2 (GW3's
    # deadline hasn't passed) -> current_event == 2. The old "current_event - 1" logic targeted
    # GW1 (nothing logged) and GW2 stayed unrealized for a week. The sweep realizes GW2 now
    # because bootstrap says it's final; GW3 (== current_event, in progress) is skipped.
    monkeypatch.setattr(rdlo, "DECISION_LOG_DIR", tmp_path)
    _write_log(tmp_path, 7139944, 2)
    _write_log(tmp_path, 7139944, 3)
    monkeypatch.setattr(rdlo.ax, "fetch_bootstrap_static", lambda: {"events": [
        {"id": 2, "finished": True, "data_checked": True},
        {"id": 3, "finished": False, "data_checked": False},  # in progress
    ]})
    realized = []
    monkeypatch.setattr(rdlo, "realize_gameweek", lambda eid, gw: realized.append((eid, gw)))
    monkeypatch.setattr(rdlo, "TRACKED_ACCOUNTS", [{"entry_id": 7139944, "label": "x"}])
    monkeypatch.setattr(sys, "argv", ["realize_decision_log_outcomes.py", "2"])
    rdlo.main()
    assert realized == [(7139944, 2)]


def test_main_catches_up_multiple_stragglers_in_one_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(rdlo, "DECISION_LOG_DIR", tmp_path)
    for gw in (2, 3, 4):
        _write_log(tmp_path, 7139944, gw)
    monkeypatch.setattr(rdlo.ax, "fetch_bootstrap_static", lambda: {"events": [
        {"id": 2, "finished": True, "data_checked": True},
        {"id": 3, "finished": True, "data_checked": True},
        {"id": 4, "finished": True, "data_checked": True},
    ]})
    realized = []
    monkeypatch.setattr(rdlo, "realize_gameweek", lambda eid, gw: realized.append((eid, gw)))
    monkeypatch.setattr(rdlo, "TRACKED_ACCOUNTS", [{"entry_id": 7139944, "label": "x"}])
    monkeypatch.setattr(sys, "argv", ["realize_decision_log_outcomes.py", "4"])
    rdlo.main()
    assert realized == [(7139944, 2), (7139944, 3), (7139944, 4)]


def test_main_skips_a_logged_gameweek_that_is_not_yet_data_checked(tmp_path, monkeypatch):
    monkeypatch.setattr(rdlo, "DECISION_LOG_DIR", tmp_path)
    _write_log(tmp_path, 7139944, 2)
    _write_log(tmp_path, 7139944, 3)
    monkeypatch.setattr(rdlo.ax, "fetch_bootstrap_static", lambda: {"events": [
        {"id": 2, "finished": True, "data_checked": True},
        {"id": 3, "finished": True, "data_checked": False},  # bonus not final
    ]})
    realized = []
    monkeypatch.setattr(rdlo, "realize_gameweek", lambda eid, gw: realized.append((eid, gw)))
    monkeypatch.setattr(rdlo, "TRACKED_ACCOUNTS", [{"entry_id": 7139944, "label": "x"}])
    monkeypatch.setattr(sys, "argv", ["realize_decision_log_outcomes.py", "4"])
    rdlo.main()
    assert realized == [(7139944, 2)]


def test_main_no_op_when_nothing_logged_at_or_before_current_event(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rdlo, "DECISION_LOG_DIR", tmp_path)
    _write_log(tmp_path, 7139944, 5)  # logged, but > current_event
    monkeypatch.setattr(sys, "argv", ["realize_decision_log_outcomes.py", "3"])
    rdlo.main()
    assert "no logged gameweek at or before GW3" in capsys.readouterr().out
