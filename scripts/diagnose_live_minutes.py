"""Attribute a live projection change to the minutes model's switches (read-only diagnostic).

On the most recent cached live DB, re-runs minutes_model.run() and expected_points.run() for one
gameweek twice: with every live switch (active_recalibratable_versions()) and with the own
substitute rate off (sub_rate_params_version=None). Prints, for each named player:
- both runs' minutes states and EP;
- the player's own league match rows this season (start_min, minutes);
- the latest live minutes_model_outputs row.

The new model versions it writes land in a DB copy that is never saved back, so nothing persists.

Usage (from repo root, in Actions on the cached DB):
    PYTHONPATH=src python scripts/diagnose_live_minutes.py 7 haaland salah palmer
"""

import json
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from fpl_quant import backtest, db, minutes_model  # noqa: E402
from fpl_quant import expected_points as ep  # noqa: E402
from run_backtest import RECALIBRATION_SEED_DIR  # noqa: E402

SEASON = "2026-2027"
MINUTES_COLS = ("p_start_final", "p_used_as_sub_given_not_started", "p_0min", "p_1_59min", "p_60plus_min")


def _uids(con, needles: list[str]) -> dict[str, str]:
    out = {}
    for needle in needles:
        for uid, name in con.execute(
            "SELECT player_uid, canonical_name FROM dim_player WHERE lower(canonical_name) LIKE ? LIMIT 3",
            [f"%{needle.lower()}%"],
        ).fetchall():
            out[uid] = name
    return out


def _minutes(con, mv: int, uid: str) -> dict:
    row = con.execute(
        f"SELECT {', '.join(MINUTES_COLS)} FROM minutes_model_outputs WHERE model_version = ? AND player_uid = ?",
        [mv, uid],
    ).fetchone()
    return {c: (None if v is None else round(v, 4)) for c, v in zip(MINUTES_COLS, row)} if row else {}


def _ep(con, mv: int, uid: str) -> dict:
    row = con.execute(
        "SELECT sum(ep_appearance), sum(ep_goals), sum(ep_assists), sum(ep_bonus), sum(ep_total), count(*) "
        "FROM ep_outputs WHERE model_version = ? AND player_uid = ?",
        [mv, uid],
    ).fetchone()
    keys = ("appearance", "goals", "assists", "bonus", "total", "n_fixtures")
    return {k: (None if v is None else round(v, 3)) for k, v in zip(keys, row)} if row else {}


def main(argv: list[str]) -> None:
    gameweek, needles = int(argv[0]), argv[1:]
    con = db.connect()
    backtest.materialize_confirmed_seeds(con, RECALIBRATION_SEED_DIR)
    active = backtest.active_recalibratable_versions(RECALIBRATION_SEED_DIR)
    asof = date.today()
    # the pipeline's latest live fit (the table has no season column)
    ts_mv = con.execute("SELECT max(model_version) FROM team_strength_model_versions").fetchone()[0]
    latest_mm = con.execute(
        "SELECT max(model_version) FROM minutes_model_versions WHERE target_season = ?", [SEASON]
    ).fetchone()[0]
    uids = _uids(con, needles)

    def run(sub_rate):
        mm = minutes_model.run(
            con, asof, SEASON, decay_params_version=1,
            adjustment_params_version=active["adjustment_params_version"],
            shrinkage_params_version=active["shrinkage_params_version"],
            fact_multiplier_params_version=active["fact_multiplier_params_version"],
            minutes_bounds_params_version=active["minutes_bounds_params_version"],
            start_prior_params_version=active["minutes_start_prior_params_version"],
            current_season_role_params_version=active["current_season_role_params_version"],
            current_season_minutes_params_version=active["current_season_minutes_params_version"],
            sub_rate_params_version=sub_rate,
        )
        epv = ep.run(
            con, asof, SEASON, gameweek, ts_model_version=ts_mv, mm_model_version=mm,
            scoring_params_version=1, bps_params_version=1, tau_params_version=1,
            rate_shrinkage_params_version=active["rate_shrinkage_params_version"],
            rate_prior_params_version=active["rate_prior_params_version"],
        )
        return mm, epv

    live_mm, live_ep = run(active["sub_rate_params_version"])
    off_mm, off_ep = run(None)
    out = {"asof": str(asof), "gameweek": gameweek, "ts_model_version": ts_mv, "latest_live_mm": latest_mm,
           "sub_rate_params_version": active["sub_rate_params_version"], "players": {}}
    for uid, name in uids.items():
        matches = con.execute(
            "SELECT m.gameweek, s.start_min, s.minutes_played, m.competition FROM fact_player_match_stats s "
            "JOIN fact_match m ON m.match_id = s.match_id WHERE s.player_uid = ? AND m.season = ? "
            "ORDER BY m.kickoff_time",
            [uid, SEASON],
        ).fetchall()
        out["players"][name] = {
            "latest_live_minutes": _minutes(con, latest_mm, uid),
            "minutes_switch_on": _minutes(con, live_mm, uid),
            "minutes_switch_off": _minutes(con, off_mm, uid),
            "ep_switch_on": _ep(con, live_ep, uid),
            "ep_switch_off": _ep(con, off_ep, uid),
            "season_matches": [list(r) for r in matches],
        }
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main(sys.argv[1:])
