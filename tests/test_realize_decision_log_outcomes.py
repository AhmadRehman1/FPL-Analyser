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
