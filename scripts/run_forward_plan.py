"""Build one entity's forward plan (the model's decisions from the next unplayed gameweek
through GW18) and write it to data/forward_plan/<slug>.json.

Entities:
    model_team   the autonomous model-managed team (bootstrap = its committed current squad)
    7139944      real entry "Ahmad sucks"   (bootstrap = live FPL picks)
    1305242      real entry "Matippy toes"  (bootstrap = live FPL picks)

One model_choice forward walk over ~16 gameweeks is ~1.5-2h of MIQP solving on the CI runner
(same cost class as scripts/run_chip_timing_arm.py). forward_plan.yml fans the three entities
out as a matrix. Same live-FPL-API + real-DB constraints as run_chip_timing_arm.py -- built
for the CI runner, not the network-blocked sandbox.

Usage (from repo root):
    PYTHONPATH=src python scripts/run_forward_plan.py <entity> [end_gameweek]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import app_export as ax  # noqa: E402
from fpl_quant import backtest as bt  # noqa: E402
from fpl_quant import chip_timing_analysis as cta  # noqa: E402
from fpl_quant import db  # noqa: E402
from fpl_quant import forward_plan as fp  # noqa: E402
from fpl_quant import ingest_fpl_entry_picks as ifp  # noqa: E402
from fpl_quant import model_team as mt  # noqa: E402

TARGET_SEASON = "2026-2027"
OUT_DIR = REPO_ROOT / "data" / "forward_plan"
STATE_DIR = REPO_ROOT / "data" / "model_team"
RECALIBRATION_SEED_DIR = REPO_ROOT / "data" / "recalibration"

REAL_ENTRIES = {7139944: "Ahmad sucks", 1305242: "Matippy toes"}
SLUGS = {"model_team": "model_team", "7139944": "7139944", "1305242": "1305242"}
_CHIP_NAME_MAP = {"wildcard": "wildcard", "bboost": "bench_boost", "3xc": "triple_captain", "freehit": "free_hit"}


def _real_chip_state(entry_id: int) -> tuple[list[str], list[str]]:
    """The chips this entry has actually spent, split set-1 (before GW19) / set-2, normalised
    to this project's vocabulary. Mirrors run_chip_timing_arm.real_chip_state."""
    history = ax.fetch_entry_history(entry_id)
    set1, set2 = [], []
    for c in history.get("chips", []):
        chip = _CHIP_NAME_MAP.get(c.get("name"), c.get("name"))
        (set1 if int(c.get("event", 0)) < cta.SET1_DEADLINE_GAMEWEEK else set2).append(chip)
    return sorted(set(set1)), sorted(set(set2))


def _model_team_bootstrap(con) -> tuple[list[dict], int, list[str], list[str]]:
    """(squad in bootstrap shape, base_gameweek, chips_set1, chips_set2) from the committed
    model-team ledger. Raises if the model team has not been seeded yet."""
    state = mt.load_state(STATE_DIR)
    if state is None or not state["ledger"]:
        raise SystemExit(
            "model_team not seeded -- scripts/run_model_team.py must run at least once first"
        )
    latest = sorted(state["ledger"], key=lambda r: r["gameweek"])[-1]
    squad = mt._squad_from_ledger_row(con, latest)
    return squad, int(latest["gameweek"]), state.get("chips_used_set1", []), state.get("chips_used_set2", [])


def _real_entry_bootstrap(con, entry_id: int, current_event: int, bootstrap_payload: dict):
    """(squad, base_gameweek, chips_set1, chips_set2) for a real FPL entry from the live API."""
    element_names = ifp.fetch_bootstrap_elements(payload=bootstrap_payload)
    picks_payload = ax.fetch_entry_picks(entry_id, current_event)
    picks = picks_payload.get("picks") if picks_payload else None
    squad = cta.build_bootstrap_squad(entry_id=entry_id, picks=picks, element_names=element_names)
    set1, set2 = _real_chip_state(entry_id)
    return squad, current_event, set1, set2


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in SLUGS:
        raise SystemExit(f"usage: run_forward_plan.py <{' | '.join(SLUGS)}> [end_gameweek]")
    entity_key = sys.argv[1]
    end_gameweek = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else fp.HORIZON_END_GAMEWEEK

    con = db.connect()
    active = bt.active_recalibratable_versions(RECALIBRATION_SEED_DIR)

    bootstrap_payload = ax.fetch_bootstrap_static()
    current_event = ax.current_event(bootstrap_payload)
    if current_event is None:
        raise SystemExit("bootstrap-static reports no current gameweek")

    if entity_key == "model_team":
        squad, base_gw, chips1, chips2 = _model_team_bootstrap(con)
        entry_id: int | None = None
        label = "FPL Quant Model Team"
    else:
        entry_id = int(entity_key)
        squad, base_gw, chips1, chips2 = _real_entry_bootstrap(con, entry_id, current_event, bootstrap_payload)
        label = REAL_ENTRIES[entry_id]

    start_gw = base_gw + 1
    if start_gw > end_gameweek:
        raise SystemExit(f"nothing to plan: base GW{base_gw}, horizon end GW{end_gameweek}")

    print(
        f"[{label}] forward plan GW{start_gw}..{end_gameweek}  "
        f"(current_event={current_event}, real chips set1={chips1} set2={chips2})",
        flush=True,
    )
    t0 = time.time()
    plan = fp.build_forward_plan(
        con,
        entity_key=entity_key,
        entry_label=label,
        entry_id=entry_id,
        target_season=TARGET_SEASON,
        bootstrap_squad=squad,
        active_versions=active,
        start_gameweek=start_gw,
        end_gameweek=end_gameweek,
        real_chips_used_set1=chips1 or None,
        real_chips_used_set2=chips2 or None,
        base_gameweek=base_gw,
    )
    plan["current_event"] = current_event
    plan["elapsed_seconds"] = round(time.time() - t0, 1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{SLUGS[entity_key]}.json"
    out_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    wc = plan.get("wildcard") or plan.get("wildcard_held_until")
    wc_note = f"wildcard GW{wc['gameweek']}" if wc else "no wildcard in window"
    print(
        f"[{label}] {plan['elapsed_seconds']:.0f}s; {len(plan['weeks'])} weeks; "
        f"{plan['total_projected_points']:.0f} projected pts; {wc_note}; wrote {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
