import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import run_scenarios as rs  # noqa: E402


# ============================================================
# bench_player_uids
# ============================================================

def test_bench_player_uids_returns_only_non_starters():
    holdings = [
        {"player_uid": "gk0", "in_xi": True},
        {"player_uid": "def0", "in_xi": True},
        {"player_uid": "gk1", "in_xi": False},
        {"player_uid": "def5", "in_xi": False},
    ]
    assert rs.bench_player_uids(holdings) == ["gk1", "def5"]


def test_bench_player_uids_empty_when_every_holding_starts():
    holdings = [{"player_uid": "gk0", "in_xi": True}, {"player_uid": "def0", "in_xi": True}]
    assert rs.bench_player_uids(holdings) == []


# ============================================================
# armband_uids
# ============================================================

def test_armband_uids_finds_captain_and_vice():
    holdings = [
        {"player_uid": "mid3", "is_captain": True, "is_vice": False},
        {"player_uid": "fwd1", "is_captain": False, "is_vice": True},
        {"player_uid": "def0", "is_captain": False, "is_vice": False},
    ]
    assert rs.armband_uids(holdings) == {"captain": "mid3", "vice_captain": "fwd1"}


def test_armband_uids_omits_a_missing_role_rather_than_defaulting():
    holdings = [{"player_uid": "mid3", "is_captain": True, "is_vice": False}]
    assert rs.armband_uids(holdings) == {"captain": "mid3"}


def test_armband_uids_empty_when_neither_role_present():
    holdings = [{"player_uid": "def0", "is_captain": False, "is_vice": False}]
    assert rs.armband_uids(holdings) == {}


# ============================================================
# scenario_result_row
# ============================================================

@dataclass(frozen=True)
class _FakeDecision:
    action: str


@dataclass(frozen=True)
class _FakeScenarioResult:
    baseline_decision: _FakeDecision
    perturbed_decision: _FakeDecision
    delta_ep: float
    flipped: bool


def test_scenario_result_row_shapes_the_fields_pwa_needs():
    result = _FakeScenarioResult(
        baseline_decision=_FakeDecision(action="hold"),
        perturbed_decision=_FakeDecision(action="transfer"),
        delta_ep=2.5,
        flipped=True,
    )
    row = rs.scenario_result_row("mid3", result)
    assert row == {
        "player_uid": "mid3", "delta_ep": 2.5, "flipped": True,
        "baseline_action": "hold", "perturbed_action": "transfer",
    }


def test_scenario_result_row_reflects_a_non_flip_case():
    result = _FakeScenarioResult(
        baseline_decision=_FakeDecision(action="hold"),
        perturbed_decision=_FakeDecision(action="hold"),
        delta_ep=-0.3,
        flipped=False,
    )
    row = rs.scenario_result_row("fwd1", result)
    assert row["flipped"] is False
    assert row["baseline_action"] == row["perturbed_action"] == "hold"
