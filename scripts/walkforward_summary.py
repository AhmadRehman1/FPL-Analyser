"""Prints the scoreboard numbers for a finished walk-forward run as JSON.

Headline beats-avg-manager, price-band / position ep_total residuals, minutes log score, and
captain stats (the walk-forward XI's own captain: points per GW before doubling, how often
that captain was a defender or goalkeeper, and how often it was the XI's top-EP player), and
captain counterfactuals: what the same XI's captain would have scored under other picking rules
(highest P95, mean + half a standard deviation) and with hindsight.

Usage (from repo root, after scripts/run_walkforward.py):
    PYTHONPATH=src python scripts/walkforward_summary.py [backtest_run_id]
"""

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import db  # noqa: E402


def _mean_metric(con, run_id, name, tier=None):
    sql = "SELECT avg(metric_value), count(*) FROM backtest_metrics WHERE backtest_run_id = ? AND metric_name = ?"
    args = [run_id, name]
    if tier:
        sql += " AND tier = ?"
        args.append(tier)
    avg, n = con.execute(sql, args).fetchone()
    return (round(avg, 4) if avg is not None else None), n


def captain_stats(con, run_id) -> dict:
    rows = con.execute(
        """
        SELECT s.season, s.gameweek, sel.player_uid, dp.position, f.event_points,
               eo.ep_total,
               (SELECT max(eo2.ep_total) FROM squad_optimizer_selections x
                  JOIN ep_gameweek_outputs eo2 ON eo2.player_uid = x.player_uid AND eo2.model_version = s.ep_model_version
                 WHERE x.run_id = s.so_run_id AND x.in_xi) AS best_xi_ep
        FROM backtest_gameweek_steps s
        JOIN squad_optimizer_selections sel ON sel.run_id = s.so_run_id AND sel.is_captain
        JOIN dim_player dp ON dp.player_uid = sel.player_uid
        LEFT JOIN fact_player_season_stats f ON f.player_uid = sel.player_uid AND f.season = s.season AND f.gw = s.gameweek
        LEFT JOIN ep_gameweek_outputs eo ON eo.player_uid = sel.player_uid AND eo.model_version = s.ep_model_version
        WHERE s.backtest_run_id = ? AND s.so_run_id IS NOT NULL
        """,
        [run_id],
    ).fetchall()
    if not rows:
        return {"n": 0}
    pts = [r[4] for r in rows if r[4] is not None]
    by_pos: dict[str, int] = {}
    for r in rows:
        by_pos[r[3]] = by_pos.get(r[3], 0) + 1
    top_ep = sum(1 for r in rows if r[5] is not None and r[6] is not None and r[5] >= r[6] - 1e-9)
    return {
        "n": len(rows),
        "captain_points_per_gw": round(sum(pts) / len(pts), 3) if pts else None,
        "captain_share_by_position": {k: round(v / len(rows), 3) for k, v in sorted(by_pos.items())},
        "captain_defender_or_gk_rate": round((by_pos.get("Defender", 0) + by_pos.get("Goalkeeper", 0)) / len(rows), 3),
        "captain_is_top_ep_in_xi_rate": round(top_ep / len(rows), 3),
    }


# Captain rules scored on the same XI. Each takes one XI player's (ep, var, p95) and returns the
# value the rule maximises.
CAPTAIN_RULES = {
    "top_ep": lambda ep, var, p95: ep,
    "top_p95": lambda ep, var, p95: p95,
    "ep_plus_half_sd": lambda ep, var, p95: ep + 0.5 * math.sqrt(max(var, 0.0)),
}


def captain_rule_points(steps: list[list[tuple]]) -> dict:
    """steps: one list per scored gameweek of (ep, var, p95, realized) per XI player. Returns,
    per rule, the captain's mean realized points per gameweek and the paired difference from
    top_ep (mean, standard error), plus the hindsight ceiling (best realized in the XI). The
    captain's points count twice, so +1 here is +1 squad point per gameweek."""
    usable = [xi for xi in steps if xi and all(r[0] is not None and r[3] is not None for r in xi)]
    if not usable:
        return {"n": 0}
    picks: dict[str, list[float]] = {name: [] for name in CAPTAIN_RULES}
    hindsight = []
    for xi in usable:
        for name, rule in CAPTAIN_RULES.items():
            best = max(xi, key=lambda r: rule(r[0], r[1] or 0.0, r[2] if r[2] is not None else r[0]))
            picks[name].append(float(best[3]))
        hindsight.append(float(max(r[3] for r in xi)))

    def paired(a: list[float], b: list[float]) -> dict:
        d = [x - y for x, y in zip(a, b)]
        m = sum(d) / len(d)
        se = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d)) if len(d) > 1 else None
        return {"mean": round(m, 3), "se": None if se is None else round(se, 3)}

    base = picks["top_ep"]
    return {
        "n": len(usable),
        "points_per_gw": {name: round(sum(v) / len(v), 3) for name, v in picks.items()},
        "vs_top_ep": {name: paired(v, base) for name, v in picks.items() if name != "top_ep"},
        "hindsight_points_per_gw": round(sum(hindsight) / len(hindsight), 3),
        "hindsight_vs_top_ep": paired(hindsight, base),
    }


def captain_counterfactuals(con, run_id) -> dict:
    rows = con.execute(
        """
        SELECT s.season, s.gameweek, eo.ep_total, uo.var_total, uo.quantile_95, f.event_points
        FROM backtest_gameweek_steps s
        JOIN squad_optimizer_selections sel ON sel.run_id = s.so_run_id AND sel.in_xi
        LEFT JOIN ep_gameweek_outputs eo ON eo.player_uid = sel.player_uid AND eo.model_version = s.ep_model_version
        LEFT JOIN uncertainty_gameweek_outputs uo ON uo.player_uid = sel.player_uid AND uo.model_version = s.un_model_version
        LEFT JOIN fact_player_season_stats f ON f.player_uid = sel.player_uid AND f.season = s.season AND f.gw = s.gameweek
        WHERE s.backtest_run_id = ? AND s.so_run_id IS NOT NULL
        ORDER BY s.season, s.gameweek
        """,
        [run_id],
    ).fetchall()
    steps: dict[tuple, list[tuple]] = {}
    for season, gw, ep, var, p95, pts in rows:
        steps.setdefault((season, gw), []).append((ep, var, p95, pts))
    return captain_rule_points(list(steps.values()))


def summarize(con, run_id: int) -> dict:
    out = {"backtest_run_id": run_id, "headline": {}, "price_band": {}, "position": {}}
    for name in ("beats_crowd_points_delta", "beats_real_avg_points_delta", "real_avg_manager_points",
                 "model_squad_realized_points", "avg_manager_benchmark_points",
                 "log_score_minutes_mean", "brier_minutes_mean", "ep_total_calibration_mean_resid",
                 "ep_total_calibration_mae"):
        out["headline"][name], out["headline"][f"n_{name}"] = _mean_metric(con, run_id, name)
    for band in ("<5.0", "5.0-7.0", "7.0-9.0", "9.0+"):
        out["price_band"][band] = {
            comp: _mean_metric(con, run_id, f"ep_{comp}_calibration_mean_resid:price_band={band}")[0]
            for comp in ("total", "appearance", "goals", "assists", "cleansheet", "other")
        }
    for pos in ("Goalkeeper", "Defender", "Midfielder", "Forward"):
        out["position"][pos] = _mean_metric(con, run_id, f"ep_total_calibration_mean_resid:position={pos}")[0]
    out["captain"] = captain_stats(con, run_id)
    out["captain_counterfactuals"] = captain_counterfactuals(con, run_id)
    # per-gameweek rows so two arms can be compared on the SAME scored steps (an arm can lose
    # steps, e.g. to the optimizer's divergence check at very low lambda)
    out["per_gameweek"] = [
        {"season": s, "gw": g, "beats_crowd": round(v, 3), "beats_real": None if r is None else round(r, 3)}
        for s, g, v, r in con.execute(
            "SELECT c.season, c.gameweek, c.metric_value, r.metric_value FROM backtest_metrics c "
            "LEFT JOIN backtest_metrics r ON r.backtest_run_id = c.backtest_run_id AND r.season = c.season "
            "AND r.gameweek = c.gameweek AND r.metric_name = 'beats_real_avg_points_delta' "
            "WHERE c.backtest_run_id = ? AND c.metric_name = 'beats_crowd_points_delta' "
            "ORDER BY c.season, c.gameweek", [run_id],
        ).fetchall()
    ]
    return out


def main() -> None:
    con = db.connect(read_only=True)
    run_id = int(sys.argv[1]) if len(sys.argv) > 1 else con.execute("SELECT max(backtest_run_id) FROM backtest_runs").fetchone()[0]
    print(json.dumps(summarize(con, run_id), indent=2))
    con.close()


if __name__ == "__main__":
    main()
