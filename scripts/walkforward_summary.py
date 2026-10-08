"""Prints the scoreboard numbers for a finished walk-forward run as JSON.

Headline beats-avg-manager, price-band / position ep_total residuals, minutes log score, and
captain stats (the walk-forward XI's own captain: points per GW before doubling, how often
that captain was a defender or goalkeeper, and how often it was the XI's top-EP player), and
captain counterfactuals: what the same XI's captain would have scored under other picking rules
(highest P95, mean + half a standard deviation) and with hindsight. The breakout group
(fpl_quant.breakout, docs/plans/2026-10_breakout_players.md) gets its own residual per season,
and db_cache_key (env DB_CACHE_KEY, the restored cache) lets scripts/compare_arms.py refuse to
compare runs made on different DBs.

Usage (from repo root, after scripts/run_walkforward.py):
    PYTHONPATH=src python scripts/walkforward_summary.py [backtest_run_id]
"""

import json
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import backtest, breakout, db  # noqa: E402


HEADLINE_METRICS = (
    "beats_crowd_points_delta", "beats_real_avg_points_delta", "real_avg_manager_points",
    "model_squad_realized_points", "avg_manager_benchmark_points",
    "log_score_minutes_mean", "brier_minutes_mean", "ep_total_calibration_mean_resid",
    "ep_total_calibration_mae", "match_score_log_lik_mean",
)

# Promoted clubs' players and the players who face them: where a promoted club's strength
# reaches the points (docs/reports/2026-10_promoted_club_strength.md). "Promoted" needs the
# previous season in the data, so 2024-25 has none.
PROMOTED_SEGMENTS = ("promoted_team", "vs_promoted_team")
PROMOTED_METRICS = (
    "ep_total_calibration_mean_resid", "ep_total_calibration_mae",
    "log_score_clean_sheet_mean", "brier_clean_sheet_mean", "log_score_goals_mean",
)


def _mean_metric(con, run_id, name, tier=None, season=None):
    sql = "SELECT avg(metric_value), count(*) FROM backtest_metrics WHERE backtest_run_id = ? AND metric_name = ?"
    args = [run_id, name]
    if tier:
        sql += " AND tier = ?"
        args.append(tier)
    if season:
        sql += " AND season = ?"
        args.append(season)
    avg, n = con.execute(sql, args).fetchone()
    return (round(avg, 4) if avg is not None else None), n


def minutes_prior_by_price_band(con, run_id) -> dict:
    """{season: {tier: {band: {...}}}}: the minutes model's mean start prior
    (p_start_historical_final, before evidence and fitness), its final p_start, and the share
    that actually started, for every rostered player priced in that band that week. Shows
    whether a start prior sits where it should, e.g. premiums at the 2024-25 cold start
    (docs/reports/2026-10_open_issues.md, issue 4: Salah at 0.57)."""
    rows = con.execute(
        """
        SELECT s.season, s.tier,
               CASE WHEN f.now_cost < 5.0 THEN '<5.0' WHEN f.now_cost < 7.0 THEN '5.0-7.0'
                    WHEN f.now_cost < 9.0 THEN '7.0-9.0' ELSE '9.0+' END AS band,
               avg(m.p_start_historical_final), avg(m.p_start_final),
               avg(CASE WHEN EXISTS (
                   SELECT 1 FROM fact_player_match_stats pm JOIN fact_match fm ON fm.match_id = pm.match_id
                   WHERE pm.player_uid = m.player_uid AND fm.season = s.season AND fm.gameweek = s.gameweek
                     AND pm.start_min = 0
               ) THEN 1.0 ELSE 0.0 END),
               count(*)
        FROM backtest_gameweek_steps s
        JOIN minutes_model_outputs m ON m.model_version = s.mm_model_version
        JOIN fact_player_season_stats f ON f.player_uid = m.player_uid AND f.season = s.season AND f.gw = s.gameweek
        WHERE s.backtest_run_id = ? AND f.now_cost IS NOT NULL
        GROUP BY 1, 2, 3 ORDER BY 1, 2, 3
        """,
        [run_id],
    ).fetchall()
    out: dict = {}
    for season, tier, band, prior, final, started, n in rows:
        out.setdefault(season, {}).setdefault(tier, {})[band] = {
            "p_start_prior": round(prior, 3), "p_start_final": round(final, 3),
            "started_share": round(started, 3), "n": n,
        }
    return out


def promoted_club_cuts(con, run_id) -> dict:
    """{season: {"promoted_match": mean match-score log-likelihood of matches with a promoted
    club, segment: {metric: mean}}} for the PROMOTED_SEGMENTS, seasons that have them only."""
    out: dict = {}
    seasons = [r[0] for r in con.execute(
        "SELECT DISTINCT season FROM backtest_metrics WHERE backtest_run_id = ? ORDER BY season", [run_id],
    ).fetchall()]
    for season in seasons:
        by_season: dict = {}
        value, n = _mean_metric(con, run_id, "match_score_log_lik_mean:promoted_match", season=season)
        if n:
            by_season["promoted_match_score_log_lik_mean"] = value
        for segment in PROMOTED_SEGMENTS:
            cut = {}
            for name in PROMOTED_METRICS:
                value, n = _mean_metric(con, run_id, f"{name}:{segment}", season=season)
                if n:
                    cut[name] = value
            if cut:
                by_season[segment] = cut
        if by_season:
            out[season] = by_season
    return out


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


BREAKOUT_SAMPLE_SIZE = 10


def _residual_block(resid: list[float]) -> dict:
    n = len(resid)
    return {
        "n_player_steps": n,
        "mean_resid": round(sum(resid) / n, 4) if n else None,
        "mae": round(sum(abs(r) for r in resid) / n, 4) if n else None,
    }


def breakout_cuts(con, run_id) -> dict:
    """{season: None | {label: {n_player_steps, mean_resid, mae}, "sample": [...]}} -- realized
    event_points minus predicted ep_total (the step's own ep_gameweek_outputs, so a double
    gameweek is summed and a blank one drops out) for the breakout group at each step. Labels:
    breakout / breakout_promoted (the rule as declared) and the same with _widened (the one
    declared fallback, 2 of the last 3). None for a season with no earlier loaded season."""
    steps = con.execute(
        "SELECT season, gameweek, ep_model_version FROM backtest_gameweek_steps "
        "WHERE backtest_run_id = ? AND ep_model_version IS NOT NULL ORDER BY season, gameweek", [run_id],
    ).fetchall()
    if not steps:
        return {}
    loaded = tuple(r[0] for r in con.execute(
        "SELECT DISTINCT season FROM fact_match WHERE competition = ? ORDER BY season", [backtest.PL],
    ).fetchall())
    breakout.build_club_spells(con, loaded)
    memo: dict = {}
    resid: dict = {}
    samples: dict = {}
    for season, gw, ep_mv in steps:
        deadline = backtest.gameweek_deadline(con, season, gw)
        for suffix, rule in (("", {}), ("_widened", breakout.WIDENED)):
            groups = breakout.classify(con, season, gw, deadline, promoted_memo=memo, **rule)
            if groups is None:  # depends on the season alone: no earlier loaded season
                resid[season] = None
                continue
            by_season = resid.setdefault(season, {})
            for label in (breakout.BREAKOUT, breakout.PROMOTED):
                by_season.setdefault(label + suffix, [])
            if not groups:
                continue
            rows = con.execute(
                "SELECT e.player_uid, dp.canonical_name, e.ep_total, f.event_points "
                "FROM ep_gameweek_outputs e "
                "JOIN fact_player_season_stats f ON f.player_uid = e.player_uid AND f.season = ? AND f.gw = ? "
                "JOIN dim_player dp ON dp.player_uid = e.player_uid "
                "WHERE e.model_version = ? AND f.event_points IS NOT NULL "
                "AND e.player_uid IN (SELECT unnest(?::VARCHAR[])) ORDER BY e.player_uid",
                [season, gw, ep_mv, list(groups)],
            ).fetchall()
            for uid, name, predicted, realized in rows:
                by_season[groups[uid] + suffix].append(realized - predicted)
                if groups[uid] == breakout.BREAKOUT:
                    samples.setdefault((season, suffix), []).append(
                        {"gw": gw, "player": name, "predicted": round(predicted, 3), "realized": realized}
                    )
    out: dict = {}
    for season, by_label in resid.items():
        if by_label is None:
            out[season] = None
            continue
        out[season] = {label: _residual_block(values) for label, values in sorted(by_label.items())}
        for suffix in ("", "_widened"):
            out[season]["sample" + suffix] = _spread_sample(samples.get((season, suffix), []))
    return out


def _spread_sample(rows: list[dict]) -> list[dict]:
    """Up to BREAKOUT_SAMPLE_SIZE player-steps for a sanity read: one per player (his first
    step in the group), evenly spaced across the season rather than the first gameweek's."""
    first: dict = {}
    for row in sorted(rows, key=lambda r: (r["gw"], r["player"])):
        first.setdefault(row["player"], row)
    picks = sorted(first.values(), key=lambda r: (r["gw"], r["player"]))
    if len(picks) <= BREAKOUT_SAMPLE_SIZE:
        return picks
    step = len(picks) / BREAKOUT_SAMPLE_SIZE
    return [picks[int(i * step)] for i in range(BREAKOUT_SAMPLE_SIZE)]


def summarize(con, run_id: int) -> dict:
    # steps planned vs scored: a run stopped by its time budget averages fewer gameweeks
    out = {"backtest_run_id": run_id, "progress": backtest.walk_forward_progress(con, run_id),
           "headline": {}, "headline_by_season": {}, "price_band": {}, "position": {}}
    for name in HEADLINE_METRICS:
        out["headline"][name], out["headline"][f"n_{name}"] = _mean_metric(con, run_id, name)
    # each season on its own: 2025-26 decides an arm, 2024-25 (no earlier season in the data) is
    # the cold-start stress test
    seasons = [r[0] for r in con.execute(
        "SELECT DISTINCT season FROM backtest_metrics WHERE backtest_run_id = ? ORDER BY season", [run_id],
    ).fetchall()]
    for season in seasons:
        by_season = out["headline_by_season"][season] = {}
        for name in HEADLINE_METRICS:
            by_season[name], by_season[f"n_{name}"] = _mean_metric(con, run_id, name, season=season)
    for band in ("<5.0", "5.0-7.0", "7.0-9.0", "9.0+"):
        out["price_band"][band] = {
            comp: _mean_metric(con, run_id, f"ep_{comp}_calibration_mean_resid:price_band={band}")[0]
            for comp in ("total", "appearance", "goals", "assists", "cleansheet", "other")
        }
        out["price_band"][band]["brier_minutes"] = _mean_metric(con, run_id, f"brier_minutes_mean:price_band={band}")[0]
    out["minutes_prior_by_price_band"] = minutes_prior_by_price_band(con, run_id)
    out["promoted_clubs"] = promoted_club_cuts(con, run_id)
    for pos in ("Goalkeeper", "Defender", "Midfielder", "Forward"):
        out["position"][pos] = _mean_metric(con, run_id, f"ep_total_calibration_mean_resid:position={pos}")[0]
    out["captain"] = captain_stats(con, run_id)
    out["captain_counterfactuals"] = captain_counterfactuals(con, run_id)
    out["breakout"] = breakout_cuts(con, run_id)
    out["db_cache_key"] = os.environ.get("DB_CACHE_KEY") or None
    # per-gameweek rows so two arms can be compared on the SAME scored steps (an arm can lose
    # steps, e.g. to the optimizer's divergence check at very low lambda)
    out["per_gameweek"] = [
        {"season": s, "gw": g, "beats_crowd": round(v, 3), "beats_real": None if r is None else round(r, 3),
         "match_log_lik": None if ll is None else round(ll, 4)}
        for s, g, v, r, ll in con.execute(
            "SELECT c.season, c.gameweek, c.metric_value, r.metric_value, ll.metric_value FROM backtest_metrics c "
            "LEFT JOIN backtest_metrics r ON r.backtest_run_id = c.backtest_run_id AND r.season = c.season "
            "AND r.gameweek = c.gameweek AND r.metric_name = 'beats_real_avg_points_delta' "
            "LEFT JOIN backtest_metrics ll ON ll.backtest_run_id = c.backtest_run_id AND ll.season = c.season "
            "AND ll.gameweek = c.gameweek AND ll.metric_name = 'match_score_log_lik_mean' "
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
