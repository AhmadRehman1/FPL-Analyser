"""Shadow ML view: ep_total_ml (Huber δ=4 residual model) alongside the quant ep_total, for
the upcoming gameweek. Writes data/dashboard/ml_shadow.json for a side-by-side "does the ML
view agree?" panel -- it feeds NO recommendation. Promoting it to a real decision input is a
separate, human-gated step (research/ml/forward_test/FROZEN_CONFIG.md, REPORT.md §10b).

Needs both: a live ingestion (scripts/run_ingestion.py -> current ep_outputs) AND a
walk-forward backtest already in the same DB (scripts/run_walkforward.py -> backtest_gameweek_
steps to train on). nightly_backtest.yml runs both, so this slots in right after.

Usage (from repo root):
    PYTHONPATH=src python scripts/compute_ml_shadow.py [target_gameweek]

The target gameweek is derived from the DB by default -- the first gameweek whose deadline is
after the DB's data cutoff -- and the shadow reads the ep_outputs version built for exactly that
gameweek (so it always matches the DB it reads); pass it explicitly only to override.
"""

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))  # so `from research.ml import forward` resolves

from fpl_quant import db, projections as proj  # noqa: E402

TARGET_SEASON = "2026-2027"
DASHBOARD_DIR = REPO_ROOT / "data" / "dashboard"


def _data_cutoff(con):
    """When this DB's data was pulled: the latest fact_match ingest time. Read from the DB, not a
    live bootstrap fetch or today's clock -- the nightly restores a cached ingestion whose
    ep_outputs may lag a live deadline, and the live next gameweek is the one after this moment."""
    return con.execute("SELECT max(_ingested_at) FROM fact_match").fetchone()[0]


def _target_gameweek_from_db(con, cutoff) -> int | None:
    """The live next gameweek as of the data cutoff: the first TARGET_SEASON gameweek whose
    deadline (its first kickoff, backtest.gameweek_deadline's stand-in) is after it. None when no
    gameweek is."""
    if cutoff is None:
        return None
    row = con.execute(
        """
        SELECT gameweek
        FROM fact_match
        WHERE season = ? AND competition = 'Premier League' AND gameweek IS NOT NULL
        GROUP BY gameweek
        HAVING min(kickoff_time) > ?
        ORDER BY min(kickoff_time), gameweek
        LIMIT 1
        """,
        [TARGET_SEASON, cutoff],
    ).fetchone()
    return row[0] if row else None


def _ep_versions_for_gameweek(con, gameweek: int) -> tuple[int, int] | None:
    """(ep_model_version, minutes_model_version) of the newest ep_outputs built for exactly this
    TARGET_SEASON gameweek -- every fixture it points at is one of that gameweek's -- paired with
    the minutes model it was built from, as each walk-forward training step is. Not the newest
    version overall: the nightly's own walk-forward writes one per historical step (2025-26 GW38
    last), and planner horizons / forward plans write later gameweeks. None when there is none."""
    row = con.execute(
        """
        SELECT v.model_version, v.minutes_model_version
        FROM ep_model_versions v
        JOIN ep_outputs o ON o.model_version = v.model_version
        JOIN fact_match m ON m.match_id = o.fixture_match_id
        WHERE v.target_season = ?
        GROUP BY v.model_version, v.minutes_model_version
        HAVING bool_and(coalesce(m.season = ? AND m.gameweek = ? AND m.competition = 'Premier League', FALSE))
        ORDER BY v.model_version DESC
        LIMIT 1
        """,
        [TARGET_SEASON, TARGET_SEASON, gameweek],
    ).fetchone()
    return (row[0], row[1]) if row else None


def _fetch_element_names() -> dict[int, str]:
    try:
        from fpl_quant import ingest_fpl_entry_picks as ifp
        return ifp.fetch_bootstrap_elements()
    except Exception as e:  # noqa: BLE001
        print(f"::warning::compute_ml_shadow: bootstrap-static fetch failed ({e}) -- fpl_element_id omitted")
        return {}


def build_ml_shadow_payload(con, target_gameweek: int | None = None, element_names: dict[int, str] | None = None) -> dict:
    """The full ml_shadow.json body. `status` is 'ok' only when a real prediction was made;
    every other value ('no_ep_outputs_for_target_gameweek', ...) is an honest "couldn't run"
    placeholder, never fabricated numbers. target_gameweek None -> the next gameweek after the
    DB's data cutoff; either way only an ep_outputs version built for that gameweek is read."""
    payload: dict = {
        "data_asof": date.today().isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target_season": TARGET_SEASON,
        "target_gameweek": target_gameweek,
        "model": "quant_lightgbm_huber_delta4",
        "disclaimer": "Shadow only -- the ML residual model's view of the same gameweek, never "
                      "a recommendation. ep_ml = ep_quant + a learned residual correction.",
        "status": "ok",
        "players": [],
    }

    if target_gameweek is None:
        cutoff = _data_cutoff(con)
        payload["data_cutoff"] = cutoff.isoformat() if cutoff is not None else None
        target_gameweek = _target_gameweek_from_db(con, cutoff)
        payload["target_gameweek"] = target_gameweek
        if target_gameweek is None:
            payload["status"] = "no_next_gameweek"
            return payload
    versions = _ep_versions_for_gameweek(con, target_gameweek)
    if versions is None:
        payload["status"] = "no_ep_outputs_for_target_gameweek"
        return payload
    ep_mv, mm_mv = versions
    payload["ep_model_version"] = ep_mv

    try:
        from research.ml import forward
    except Exception as e:  # noqa: BLE001
        payload["status"] = f"research_ml_unavailable: {e}"
        return payload

    out = forward.predict_forward(con, TARGET_SEASON, target_gameweek, ep_mv, mm_mv)
    if out is None:
        payload["status"] = "no_walk_forward_history_or_lightgbm"
        return payload

    element_by_uid = proj.resolve_element_ids(con, TARGET_SEASON, element_names or {})
    name_by_uid = dict(con.execute("SELECT player_uid, canonical_name FROM dim_player").fetchall())
    rows = [
        {
            "player_uid": r.player_uid,
            "name": name_by_uid.get(r.player_uid, r.player_uid),
            "fpl_element_id": element_by_uid.get(r.player_uid),
            "ep_quant": round(float(r.ep_quant), 3),
            "ep_ml": round(float(r.ep_total_ml), 3),
            "ml_residual": round(float(r.predicted_residual), 3),
        }
        for r in out.itertuples(index=False)
    ]
    rows.sort(key=lambda x: x["ep_ml"], reverse=True)
    ranked = sorted(rows, key=lambda x: abs(x["ml_residual"]), reverse=True)
    payload.update({
        "players": rows,
        "n_train_rows": out.attrs.get("n_train_rows"),
        "train_seasons": out.attrs.get("train_seasons"),
        "ml_boosts": [r for r in ranked if r["ml_residual"] > 0][:5],
        "ml_fades": [r for r in ranked if r["ml_residual"] < 0][:5],
    })
    return payload


def main() -> None:
    con = db.connect()
    override_gw = int(sys.argv[1]) if len(sys.argv) > 1 else None
    payload = build_ml_shadow_payload(con, override_gw, _fetch_element_names())

    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    (DASHBOARD_DIR / "ml_shadow.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] == "ok":
        print(f"[compute_ml_shadow] GW{payload['target_gameweek']}: {len(payload['players'])} players, "
              f"trained on {payload['n_train_rows']} rows ({payload['train_seasons']})")
    else:
        print(f"[compute_ml_shadow] status={payload['status']} -- wrote a placeholder ml_shadow.json")
    con.close()


if __name__ == "__main__":
    main()
