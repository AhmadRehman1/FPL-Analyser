"""Phase C-2 (Track C, docs/plans/2026-08_roadmap_plan.md) -- once a gameweek's real results
are known, back-fills the planner_decision_log entry Phase C-1 logged for it with what actually
happened (realized_points_actual, actual_action_taken) and what would have happened had the
manager instead followed the logged recommendation (realized_points_if_recommendation_followed).

Usage (from repo root, same job as scripts/run_ingestion.py, right after it):
    PYTHONPATH=src python scripts/realize_decision_log_outcomes.py <event>

<event> is the SAME current-gameweek value scheduled_pipeline.yml's own "Determine current
gameweek" step already computes and threads through to every other real-squad step.

Realizes EVERY logged gameweek that is finished + data_checked per bootstrap-static and not
already realized -- not just <event> - 1. FPL's events[].is_current lags a full gameweek
cycle (it stays on GW N for the whole GW N -> GW N+1 window, only flipping to N+1 at N+1's
deadline), so a "prior gameweek only" pass left GW2's real outcome unrecorded from the moment
GW2 finished until GW3's deadline a week later -- the track-record accuracy panel showing a
gameweek behind reality the whole time. realize_gameweek() is idempotent (skips once
realized_points_actual is set), so a catch-up sweep over the committed decision_log/ files is
safe. Not yet finished / not data_checked / nothing logged -> skipped, not an error.

Same network-blocked-in-sandbox caveat as every other real-squad script in this project.
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import app_export as ax  # noqa: E402
from fpl_quant import entity_resolution as er  # noqa: E402
from fpl_quant import ingest_fpl_entry_picks as ifp, reporting  # noqa: E402

TARGET_SEASON = "2026-2027"
DECISION_LOG_DIR = REPO_ROOT / "data" / "decision_log"

# Same two recurring, tracked accounts Phase C-1 logs for (run_transfer_planner_for_real_squad.py's
# is_tracked_account guard) -- an ad-hoc dispatch account never gets a decision_log entry to
# realize in the first place, so there's nothing to special-case here.
TRACKED_ACCOUNTS = [
    {"entry_id": 7139944, "label": "ChatGPT template team"},
    {"entry_id": 1305242, "label": "Main account"},
]


def _build_name_to_element_id(elements: dict[int, str]) -> dict[str, int]:
    """Maps FPL's live bootstrap-element names to element ids, keyed by
    entity_resolution.normalize_name() rather than the literal string -- decision_log entries
    store dim_player.canonical_name (reconcile.py freezes it at first ingestion, `ON CONFLICT
    (player_uid) DO NOTHING`), which can be spelled differently (accents, punctuation) than
    whatever bootstrap-static reports as this element's name right now. A normalized name shared
    by more than one element (a real, if rare, FPL occurrence) is dropped entirely rather than
    guessed -- the same "only resolve if unambiguous" rule ingest_workbook._resolve_player() uses
    for the identical underlying problem.
    """
    by_normalized: dict[str, set[int]] = {}
    for element_id, name in elements.items():
        by_normalized.setdefault(er.normalize_name(name), set()).add(element_id)
    return {name: ids.pop() for name, ids in by_normalized.items() if len(ids) == 1}


def _normalize_decision_row_names(decision_row: dict) -> dict:
    """Copy of decision_row with its player-name fields run through the same normalize_name()
    used to build _build_name_to_element_id()'s keys, so the two sides of the lookup actually
    match (see that function's own docstring). Never mutates the caller's decision_row -- that
    dict is still written back to disk with its original, human-readable names intact."""
    normalized = dict(decision_row)
    for key in ("recommended_transfer_out", "recommended_transfer_in", "recommended_captain"):
        if normalized.get(key):
            normalized[key] = er.normalize_name(normalized[key])
    return normalized


def _actual_action_taken(entry_history: dict, active_chip: str | None) -> str:
    """A real, computed label for what the manager actually did that gameweek -- never
    guessed. entry_history['event_transfers'] and active_chip both come straight from FPL's own
    picks payload."""
    if active_chip:
        return active_chip
    if entry_history.get("event_transfers", 0) > 0:
        return "transfer_made"
    return "hold"


def realize_gameweek(entry_id: int, gameweek: int) -> dict | None:
    """Returns the updated decision_log row, or None if there was nothing to realize (no entry
    logged, or it was already realized)."""
    decision_row = reporting.load_decision_log_entry(entry_id, TARGET_SEASON, gameweek, DECISION_LOG_DIR)
    if decision_row is None:
        print(f"[realize] entry_id={entry_id} GW{gameweek}: no logged recommendation -- nothing to realize")
        return None
    if decision_row.get("realized_points_actual") is not None:
        print(f"[realize] entry_id={entry_id} GW{gameweek}: already realized -- skipping")
        return None

    actual_payload = ax.fetch_entry_picks(entry_id, gameweek)
    if actual_payload is None:
        print(f"[realize] entry_id={entry_id} GW{gameweek}: no real picks found (404) -- skipping")
        return None
    baseline_payload = ax.fetch_entry_picks(entry_id, gameweek - 1)
    if baseline_payload is None:
        print(f"[realize] entry_id={entry_id} GW{gameweek}: no baseline (GW{gameweek - 1}) picks found -- skipping")
        return None

    entry_history = actual_payload.get("entry_history") or {}
    live_points_by_element = {
        el["id"]: el.get("stats", {}).get("total_points", 0) for el in ax.fetch_event_live(gameweek).get("elements", [])
    }
    name_to_element_id = _build_name_to_element_id(ifp.fetch_bootstrap_elements())

    decision_row["realized_points_actual"] = entry_history.get("points")
    decision_row["actual_action_taken"] = _actual_action_taken(entry_history, actual_payload.get("active_chip"))
    decision_row["realized_points_if_recommendation_followed"] = reporting.compute_counterfactual_points(
        baseline_payload.get("picks", []), live_points_by_element, name_to_element_id,
        _normalize_decision_row_names(decision_row),
    )
    reporting.save_decision_log_entry(entry_id, TARGET_SEASON, gameweek, decision_row, DECISION_LOG_DIR)
    print(
        f"[realize] entry_id={entry_id} GW{gameweek}: actual={decision_row['realized_points_actual']} "
        f"if_followed={decision_row['realized_points_if_recommendation_followed']} "
        f"(recommended {decision_row['recommended_action']}, actually {decision_row['actual_action_taken']})"
    )
    return decision_row


_LOG_GW_RE = re.compile(r"_(\d{4}-\d{4})_gw(\d+)\.json$")


def _logged_gameweeks(season: str) -> list[int]:
    """Every gameweek that has at least one committed decision_log entry for this season,
    ascending. The catch-up sweep realizes any of these that are finished + not yet realized."""
    gws = set()
    for path in DECISION_LOG_DIR.glob(f"*_{season}_gw*.json"):
        m = _LOG_GW_RE.search(path.name)
        if m and m.group(1) == season:
            gws.add(int(m.group(2)))
    return sorted(gws)


def _gameweek_is_scorable(bootstrap: dict, gameweek: int) -> bool:
    event_row = next((ev for ev in bootstrap.get("events", []) if ev.get("id") == gameweek), None)
    if event_row is None or not event_row.get("finished"):
        print(f"[realize] GW{gameweek} not yet finished per bootstrap-static -- skipping until it is")
        return False
    # `finished` alone can flip true before bonus points (BPS) are fully confirmed --
    # `data_checked` is FPL's own stricter flag for that, and this step is idempotent-on-success
    # only (realize_gameweek() skips forever once realized_points_actual is set), so realizing on
    # a not-yet-data_checked gameweek would lock in a wrong number permanently, not just briefly.
    if not event_row.get("data_checked"):
        print(f"[realize] GW{gameweek} finished but not yet data_checked (bonus points not final) -- skipping until it is")
        return False
    return True


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <event>")
    current_event = int(sys.argv[1])

    # Every logged gameweek up to and including the current one is a realization candidate;
    # bootstrap-static's own finished + data_checked flags (checked in _gameweek_is_scorable)
    # are the real "results are final" gate. `current_event - 1` alone would leave the
    # just-finished gameweek unrealized for up to a week, because FPL's is_current stays on
    # GW N for the whole GW N -> GW N+1 window (see the module docstring). `<= current_event`
    # bounds the search safely -- a gameweek past the current one cannot be finished.
    candidates = [gw for gw in _logged_gameweeks(TARGET_SEASON) if gw <= current_event]
    if not candidates:
        print(f"[realize] no logged gameweek at or before GW{current_event} to realize yet this season")
        return

    bootstrap = ax.fetch_bootstrap_static()
    for gameweek in candidates:
        if not _gameweek_is_scorable(bootstrap, gameweek):
            continue
        for account in TRACKED_ACCOUNTS:
            realize_gameweek(account["entry_id"], gameweek)


if __name__ == "__main__":
    main()
