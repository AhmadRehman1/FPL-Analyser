"""One arm of the two-team Wildcard-timing sweep (chip_timing_analysis Step 1).

Each invocation runs exactly one `forward_season_sim.run_forward_season_sim()` walk for one
tracked entry -- either the `hold_wildcard` baseline, the greedy `model_choice` walk, or a
single `force_wildcard_at=<gw>` arm -- and writes its `ForwardSimResult.to_dict()` to a JSON
artifact. `aggregate_chip_timing.py` reads every arm's artifact and does Steps 2-6.

Split this way because one forward walk over a ~17-gameweek window is ~1h of MIQP solving on
the CI runner (measured: forward_season_sim.yml run 33432100674). A GW4-19 sweep is 16 forced
walks + hold + model_choice per team = 36 arms; run as a GitHub Actions matrix they finish in
a couple of waves instead of ~36h in series. Same live-FPL-API + real-DB constraints as
scripts/run_forward_season_sim.py -- built for the CI runner, not the network-blocked sandbox.

Config via env vars (the workflow sets them from the matrix):
    CTA_ENTRY_ID     7139944 | 1305242            (required)
    CTA_MODE         hold | model_choice | force  (required)
    CTA_FORCE_GW     int                          (required iff CTA_MODE=force)
    CTA_END_GW       eval-window end gameweek     (default 22)
    CTA_PARAM_BUNDLE active | confirmed_pending | all_v1   (default active)
    CTA_OUT          output path                  (default data/chip_timing/arms/<name>.json)
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import app_export as ax  # noqa: E402
from fpl_quant import backtest as bt  # noqa: E402
from fpl_quant import chip_timing_analysis as cta  # noqa: E402
from fpl_quant import db  # noqa: E402
from fpl_quant import forward_season_sim as fss  # noqa: E402
from fpl_quant import ingest_fpl_entry_picks as ifp  # noqa: E402

TARGET_SEASON = "2026-2027"
OUT_DIR = REPO_ROOT / "data" / "chip_timing" / "arms"
RECALIBRATION_SEED_DIR = REPO_ROOT / "data" / "recalibration"

ENTRY_LABELS = {7139944: "Ahmad sucks", 1305242: "Matippy toes"}

# The three parameter bundles Step 5's sensitivity check compares. `active` is what
# backtest.active_recalibratable_versions() resolves against the committed seed file today
# (xi v2, rho_residual v2 = 0.0, everything else v1). `confirmed_pending` layers on the three
# recalibration_proposals that are status='confirmed' in the DB but not yet activated (their
# seed file is parked in scratchpad/ pending owner review): rho_residual v4 (= 0.0168),
# lambda_value v2 (= 0.05), kappa_tc v2 (= 0.5). `all_v1` is a pure no-recalibration baseline.
# NB: the prompt's "xi 0.0018->0.005 / rho_residual 0.15->0.0 / lambda 0.15->0.0 pending"
# framing does not match the DB -- xi->0.005 and rho_residual->0.0 are ALREADY active, there is
# no lambda->0.0 proposal, and only competitive_matches_threshold 10->20 is genuinely pending
# (held). See README's chip-timing Design note.
_CONFIRMED_PENDING_OVERRIDES = {
    "rho_residual_params_version": 4,
    "lambda_params_version": 2,
    "kappa_tc_params_version": 2,
}


def resolve_param_bundle(name: str) -> dict:
    if name == "all_v1":
        return {}
    active = bt.active_recalibratable_versions(RECALIBRATION_SEED_DIR)
    if name == "active":
        return active
    if name == "confirmed_pending":
        return {**active, **_CONFIRMED_PENDING_OVERRIDES}
    raise SystemExit(f"unknown CTA_PARAM_BUNDLE={name!r} (active | confirmed_pending | all_v1)")


def real_chip_state(entry_id: int) -> tuple[list[str], list[str]]:
    """The chips the entry has actually spent, split into set 1 (played before GW19) and set 2
    (GW19+), normalised to this project's chip vocabulary (FPL calls the Wildcard 'wildcard',
    Bench Boost 'bboost', Triple Captain '3xc', Free Hit 'freehit')."""
    name_map = {"wildcard": "wildcard", "bboost": "bench_boost", "3xc": "triple_captain", "freehit": "free_hit"}
    history = ax.fetch_entry_history(entry_id)
    set1, set2 = [], []
    for c in history.get("chips", []):
        chip = name_map.get(c.get("name"), c.get("name"))
        (set1 if int(c.get("event", 0)) < cta.SET1_DEADLINE_GAMEWEEK else set2).append(chip)
    return sorted(set(set1)), sorted(set(set2))


def main() -> None:
    entry_id = int(os.environ["CTA_ENTRY_ID"])
    mode = os.environ["CTA_MODE"]
    end_gw = int(os.environ.get("CTA_END_GW", "22"))
    bundle_name = os.environ.get("CTA_PARAM_BUNDLE", "active")
    force_gw = int(os.environ["CTA_FORCE_GW"]) if mode == "force" else None
    label = ENTRY_LABELS.get(entry_id, f"entry {entry_id}")

    if mode not in ("hold", "model_choice", "force"):
        raise SystemExit(f"CTA_MODE must be hold | model_choice | force, got {mode!r}")

    tag = mode if mode != "force" else f"force_gw{force_gw}"
    out_path = Path(os.environ["CTA_OUT"]) if os.environ.get("CTA_OUT") else (
        OUT_DIR / f"arm_{entry_id}_{tag}_{bundle_name}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    con = db.connect()
    versions = resolve_param_bundle(bundle_name)
    print(f"[{label}] mode={mode} force_gw={force_gw} bundle={bundle_name} -> {versions or 'all v1'}", flush=True)

    bootstrap = ax.fetch_bootstrap_static()
    current_event = ax.current_event(bootstrap)
    if current_event is None:
        raise SystemExit("bootstrap-static reports no current gameweek")
    start_gw = current_event + 1

    element_names = ifp.fetch_bootstrap_elements(payload=bootstrap)
    picks_payload = ax.fetch_entry_picks(entry_id, current_event)
    picks = picks_payload.get("picks") if picks_payload else None
    squad = cta.build_bootstrap_squad(entry_id=entry_id, picks=picks, element_names=element_names)

    chips_set1, chips_set2 = real_chip_state(entry_id)
    print(f"[{label}] GW{start_gw}..{end_gw}; real chips used: set1={chips_set1} set2={chips_set2}", flush=True)

    if mode == "force":
        cta.assert_wildcard_available(
            chips_used_set1=chips_set1, chips_used_set2=chips_set2, sweep_gameweeks=[force_gw],
        )

    kwargs = dict(
        entry_label=label, target_season=TARGET_SEASON, start_gameweek=start_gw, end_gameweek=end_gw,
        bootstrap_squad=squad, active_versions=versions,
        real_chips_used_set1=chips_set1, real_chips_used_set2=chips_set2,
    )
    if mode == "hold":
        kwargs["hold_wildcard"] = True
    elif mode == "force":
        kwargs["force_wildcard_at"] = force_gw

    t0 = time.time()
    result = fss.run_forward_season_sim(con, **kwargs)
    elapsed = time.time() - t0

    # Steps 2 + 4 for the gameweek this arm actually played the Wildcard -- computed inline
    # against the same connection (fresh_run_id / model versions are DB-local), no re-walk.
    # aggregate_chip_timing.py surfaces the winning arm's followups; running them on every
    # forced arm also shows whether the pick is fragile everywhere or only at the winner.
    followups = None
    if result.wildcard_context is not None:
        try:
            followups = cta.wildcard_followups(
                con, entry_label=label, target_season=TARGET_SEASON,
                wildcard_context=result.wildcard_context,
            )
        except Exception as exc:  # a followup failure must not lose the sweep arm itself
            print(f"::warning::wildcard_followups failed for {label} {tag}: {exc}", flush=True)
            followups = {"error": str(exc)}

    payload = {
        "entry_id": entry_id,
        "label": label,
        "arm": tag,
        "param_bundle": bundle_name,
        "param_versions": versions,
        "real_chips_used_set1": chips_set1,
        "real_chips_used_set2": chips_set2,
        "current_event": current_event,
        "start_gameweek": start_gw,
        "eval_end_gameweek": end_gw,
        "elapsed_seconds": round(elapsed, 1),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "forward_sim": result.to_dict(),
        "wildcard_followups": followups,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    wc_played = next((r.gameweek for r in result.rows if r.action == "wildcard"), None)
    print(f"[{label}] {tag}: {elapsed:.0f}s; projected {result.total_projected_points:.0f} pts; "
          f"WC played at {wc_played}; wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
