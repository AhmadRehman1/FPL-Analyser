"""Prints the scoreboard numbers for a finished walk-forward run as JSON.

Headline beats-avg-manager, price-band / position ep_total residuals, minutes log score, and
captain stats (the walk-forward XI's own captain: points per GW before doubling, how often
that captain was a defender or goalkeeper, and how often it was the XI's top-EP player), and
captain counterfactuals: what the same XI's captain would have scored under other picking rules
(highest P95, mean + half a standard deviation) and with hindsight. The breakout group
(fpl_quant.breakout, docs/plans/2026-10_breakout_players.md) gets its own residual per season,
as do the in-form and cooling groups (fpl_quant.form_groups, docs/plans/
2026-10_premium_and_form_returns.md) split by component, and every split block carries a
component_abs_sum (|appearance| + |goals| + |assists| + |clean sheet| + |other|, the sum rule R5
judges in scripts/compare_arms.py). minutes_states_by_price_band sets the minutes model's states
(start, substitute appearance, 1-59, 60+) against what happened, by price band and start
probability. db_cache_key (env DB_CACHE_KEY, the restored cache) lets scripts/compare_arms.py refuse to
compare runs made on different DBs. template is the availability-gated ownership template's
yardstick (docs/plans/2026-10_beat_the_template.md): the model's paired gap to it, XI overlap,
points from outside the template XI, the captains, spend by price band and realized - projected
for both XIs, pooled and by season; per_gameweek carries the same rows.

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

from fpl_quant import backtest, breakout, db, form_groups  # noqa: E402


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


_BAND_SQL = (
    "CASE WHEN f.now_cost < 5.0 THEN '<5.0' WHEN f.now_cost < 7.0 THEN '5.0-7.0' "
    "WHEN f.now_cost < 9.0 THEN '7.0-9.0' ELSE '9.0+' END"
)


def minutes_prior_by_price_band(con, run_id) -> dict:
    """{season: {tier: {band: {...}}}}: the minutes model's mean start prior
    (p_start_historical_final, before evidence and fitness), its final p_start, and the share
    that actually started, for every rostered player priced in that band that week. Shows
    whether a start prior sits where it should, e.g. premiums at the 2024-25 cold start
    (docs/reports/2026-10_open_issues.md, issue 4: Salah at 0.57)."""
    rows = con.execute(
        f"""
        SELECT s.season, s.tier,
               {_BAND_SQL} AS band,
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


# p_start_final buckets for minutes_states_by_price_band: squad players who rarely start, rotation
# players, regular starters.
START_BUCKETS = (("<0.1", 0.0, 0.1), ("0.1-0.5", 0.1, 0.5), ("0.5+", 0.5, 1.01))


def minutes_states_by_price_band(con, run_id) -> dict:
    """{season: {band: {bucket: {...}}}}: the minutes model's states against what happened, one row
    per player-fixture the step predicted (ep_outputs), by price band and by p_start_final bucket
    (START_BUCKETS, plus "all"). EP's appearance points are 1 x p_1_59 + 2 x p_60plus, so
    appearance_resid (realized minus predicted appearance points) says which state carries a
    band's appearance error. p_sub is p_used_as_sub_given_not_started over the fixtures the
    player did not start, against the share of those he came on in. A fixture with no match row
    is 0 minutes (an unused or absent player), as the backtest scores it."""
    rows = con.execute(
        f"""
        WITH fx AS (
            SELECT s.season, {_BAND_SQL} AS band, m.p_start_final AS p_start,
                   m.p_used_as_sub_given_not_started AS p_sub, m.p_1_59min AS p1, m.p_60plus_min AS p2,
                   coalesce(pm.start_min = 0, FALSE) AS started,
                   coalesce(pm.minutes_played, 0) AS mins
            FROM backtest_gameweek_steps s
            JOIN ep_outputs o ON o.model_version = s.ep_model_version
            JOIN minutes_model_outputs m ON m.model_version = s.mm_model_version AND m.player_uid = o.player_uid
            JOIN fact_player_season_stats f ON f.player_uid = o.player_uid AND f.season = s.season AND f.gw = s.gameweek
            LEFT JOIN fact_player_match_stats pm ON pm.player_uid = o.player_uid AND pm.match_id = o.fixture_match_id
            WHERE s.backtest_run_id = ? AND f.now_cost IS NOT NULL
        )
        SELECT season, band, p_start, p_sub, p1, p2, started, mins FROM fx
        """,
        [run_id],
    ).fetchall()
    sums: dict = {}
    for season, band, p_start, p_sub, p1, p2, started, mins in rows:
        bucket = next(name for name, lo, hi in START_BUCKETS if lo <= p_start < hi)
        realized_1_59 = 0 < mins < 60
        realized_60 = mins >= 60
        for key in (bucket, "all"):
            acc = sums.setdefault(season, {}).setdefault(band, {}).setdefault(key, {
                "n": 0, "p_start": 0.0, "started": 0, "p_1_59": 0.0, "share_1_59": 0, "p_60plus": 0.0,
                "share_60plus": 0, "appearance_resid": 0.0, "n_not_started": 0, "p_sub": 0.0, "sub_share": 0,
            })
            acc["n"] += 1
            acc["p_start"] += p_start
            acc["started"] += started
            acc["p_1_59"] += p1
            acc["share_1_59"] += realized_1_59
            acc["p_60plus"] += p2
            acc["share_60plus"] += realized_60
            acc["appearance_resid"] += (realized_1_59 + 2 * realized_60) - (p1 + 2 * p2)
            if not started:
                acc["n_not_started"] += 1
                acc["p_sub"] += p_sub
                acc["sub_share"] += mins > 0
    out: dict = {}
    for season, bands in sums.items():
        for band, buckets in bands.items():
            for key, acc in buckets.items():
                n, n_ns = acc["n"], acc["n_not_started"]
                out.setdefault(season, {}).setdefault(band, {})[key] = {
                    "n": n,
                    "p_start": round(acc["p_start"] / n, 3), "started_share": round(acc["started"] / n, 3),
                    "p_1_59": round(acc["p_1_59"] / n, 3), "share_1_59": round(acc["share_1_59"] / n, 3),
                    "p_60plus": round(acc["p_60plus"] / n, 3), "share_60plus": round(acc["share_60plus"] / n, 3),
                    "appearance_resid": round(acc["appearance_resid"] / n, 4),
                    "n_not_started": n_ns,
                    "p_sub": round(acc["p_sub"] / n_ns, 3) if n_ns else None,
                    "sub_share": round(acc["sub_share"] / n_ns, 3) if n_ns else None,
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


def component_abs_sum(components: dict) -> float | None:
    """|appearance| + |goals| + |assists| + |clean sheet| + |other|: errors that cancel in the
    total still count. None if any component is missing."""
    values = [components.get(c) for c in backtest.COMPONENTS]
    if any(v is None for v in values):
        return None
    return round(sum(abs(v) for v in values), 4)


FORM_LABELS = tuple(f"{label}{suffix}" for label in (form_groups.IN_FORM, form_groups.COOLING) for suffix in ("", "_widened"))


def form_group_cuts(con, run_id) -> dict:
    """{season: None | {label: {n_player_steps, mean_resid, components, component_abs_sum}}} for
    the in-form and cooling groups, declared and widened (labels FORM_LABELS). Realized minus
    predicted per component, recomputed at each step with backtest.realized_components() for the
    step's own EP version (only per-gameweek segment means are stored). Each figure is the mean
    over gameweeks of that gameweek's group mean, the way price_band's metrics are aggregated;
    mean_resid is the components' sum. None for a season with no earlier loaded season."""
    steps = con.execute(
        "SELECT s.season, s.gameweek, s.ep_model_version, v.scoring_matrix_params_version "
        "FROM backtest_gameweek_steps s JOIN ep_model_versions v ON v.model_version = s.ep_model_version "
        "WHERE s.backtest_run_id = ? ORDER BY s.season, s.gameweek", [run_id],
    ).fetchall()
    if not steps:
        return {}
    loaded = tuple(r[0] for r in con.execute(
        "SELECT DISTINCT season FROM fact_match WHERE competition = ? ORDER BY season", [backtest.PL],
    ).fetchall())
    breakout.build_club_spells(con, loaded)
    memo: dict = {}
    per_gw: dict = {}
    for season, gw, ep_mv, scoring_version in steps:
        deadline = backtest.gameweek_deadline(con, season, gw)
        groups = {}
        for suffix, rule in (("", {}), ("_widened", form_groups.WIDENED)):
            found = form_groups.classify(con, season, gw, deadline, memo=memo, **rule)
            if found is None:  # depends on the season alone: no earlier loaded season
                per_gw[season] = None
                break
            groups[suffix] = found
        if per_gw.get(season, {}) is None:
            continue
        by_label = per_gw.setdefault(season, {label: [] for label in FORM_LABELS})
        members = set().union(*groups.values())
        if not members:
            continue
        split = backtest.realized_components(con, season, gw, ep_mv, scoring_version, player_uids=members)
        for suffix, found in groups.items():
            for label in (form_groups.IN_FORM, form_groups.COOLING):
                rows = [split[uid] for uid, lab in found.items() if lab == label and uid in split]
                if rows:
                    by_label[label + suffix].append(
                        (len(rows), {c: sum(r[c] for r in rows) / len(rows) for c in backtest.COMPONENTS})
                    )
    out: dict = {}
    for season, by_label in per_gw.items():
        if by_label is None:
            out[season] = None
            continue
        out[season] = {}
        for label, gameweeks in by_label.items():
            if not gameweeks:
                out[season][label] = {"n_player_steps": 0, "mean_resid": None, "components": None, "component_abs_sum": None}
                continue
            components = {c: round(sum(g[1][c] for g in gameweeks) / len(gameweeks), 4) for c in backtest.COMPONENTS}
            out[season][label] = {
                "n_player_steps": sum(g[0] for g in gameweeks),
                "mean_resid": round(sum(components.values()), 4),
                "components": components,
                "component_abs_sum": component_abs_sum(components),
            }
    return out


# Per-gameweek template rows (fpl_quant.backtest._score_template_baseline,
# docs/plans/2026-10_beat_the_template.md Req 1 and Req 2): summary key -> metric name.
TEMPLATE_GW_METRICS = {
    "template_points": "template_squad_realized_points",
    "template_points_ungated": "template_squad_realized_points_ungated",
    "template_xi_overlap": "template_xi_overlap",
    "non_template_points": "non_template_points",
    "model_captain_points": "model_captain_points",
    "template_captain_points": "template_captain_points",
    "captain_ep_plus_eo_points": "captain_ep_plus_eo_points",
    "model_xi_resid": "model_xi_resid",
    "template_xi_resid": "template_xi_resid",
}
PRICE_BANDS = ("<5.0", "5.0-7.0", "7.0-9.0", "9.0+")


def template_rows(con, run_id) -> dict:
    """{(season, gw): {summary key: value}} for the template rows, including
    model_xi_spend / template_xi_spend as {band: pounds}."""
    out: dict = {}
    by_metric = {metric: key for key, metric in TEMPLATE_GW_METRICS.items()}
    for season, gw, name, value in con.execute(
        "SELECT season, gameweek, metric_name, metric_value FROM backtest_metrics WHERE backtest_run_id = ? "
        "AND (metric_name IN (SELECT unnest(?::VARCHAR[])) OR metric_name LIKE 'model\\_xi\\_spend:%' ESCAPE '\\' "
        "OR metric_name LIKE 'template\\_xi\\_spend:%' ESCAPE '\\')",
        [run_id, list(by_metric)],
    ).fetchall():
        row = out.setdefault((season, gw), {})
        if ":price_band=" in name:
            family, band = name.split(":price_band=", 1)
            row.setdefault(family, {})[band] = round(value, 2)
        else:
            row[by_metric[name]] = round(value, 3)
    return out


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def _paired(rows: list[dict], a: str, b: str) -> dict:
    """Mean and SE (sample sd / sqrt n) of row[a] - row[b] over the rows that carry both."""
    d = [r[a] - r[b] for r in rows if r.get(a) is not None and r.get(b) is not None]
    n = len(d)
    if n < 2:
        return {"n": n, "mean": round(d[0], 3) if d else None, "se": None}
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    return {"n": n, "mean": round(m, 3), "se": round(sd / math.sqrt(n), 3)}


def template_block(per_gameweek: list[dict]) -> dict:
    """The template yardstick over a set of per_gameweek rows: the model's gap to the gated and
    ungated templates (paired, pts/GW), the template's own mean (T_s in the plan's G2), XI
    overlap, points from outside the template XI, the captains, spend by price band and
    realized - projected for both XIs."""
    rows = [r for r in per_gameweek if r.get("template_points") is not None]
    if not rows:
        return {"n": 0}

    def spend(family):
        return {band: _mean([r[family][band] for r in rows if band in (r.get(family) or {})]) for band in PRICE_BANDS}

    return {
        "n": len(rows),
        "model_points_per_gw": _mean([r["squad_points"] for r in rows if r.get("squad_points") is not None]),
        "template_points_per_gw": _mean([r["template_points"] for r in rows]),
        "template_points_ungated_per_gw": _mean([r["template_points_ungated"] for r in rows
                                                 if r.get("template_points_ungated") is not None]),
        "gap_vs_gated": _paired(rows, "squad_points", "template_points"),
        "gap_vs_ungated": _paired(rows, "squad_points", "template_points_ungated"),
        "xi_overlap_mean": _mean([r["template_xi_overlap"] for r in rows if r.get("template_xi_overlap") is not None]),
        "non_template_points_per_gw": _mean([r["non_template_points"] for r in rows
                                             if r.get("non_template_points") is not None]),
        "captain": {
            "model_points_per_gw": _mean([r["model_captain_points"] for r in rows if r.get("model_captain_points") is not None]),
            "template_points_per_gw": _mean([r["template_captain_points"] for r in rows
                                             if r.get("template_captain_points") is not None]),
            "ep_plus_eo_points_per_gw": _mean([r["captain_ep_plus_eo_points"] for r in rows
                                               if r.get("captain_ep_plus_eo_points") is not None]),
            # before doubling: +1 here is +1 squad point a gameweek
            "ep_plus_eo_vs_model": _paired(rows, "captain_ep_plus_eo_points", "model_captain_points"),
            "model_vs_template": _paired(rows, "model_captain_points", "template_captain_points"),
        },
        "model_xi_spend": spend("model_xi_spend"),
        "template_xi_spend": spend("template_xi_spend"),
        "model_xi_resid": _mean([r["model_xi_resid"] for r in rows if r.get("model_xi_resid") is not None]),
        "template_xi_resid": _mean([r["template_xi_resid"] for r in rows if r.get("template_xi_resid") is not None]),
    }


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
        out["price_band"][band]["component_abs_sum"] = component_abs_sum(out["price_band"][band])
    out["minutes_prior_by_price_band"] = minutes_prior_by_price_band(con, run_id)
    out["minutes_states_by_price_band"] = minutes_states_by_price_band(con, run_id)
    out["promoted_clubs"] = promoted_club_cuts(con, run_id)
    for pos in ("Goalkeeper", "Defender", "Midfielder", "Forward"):
        out["position"][pos] = _mean_metric(con, run_id, f"ep_total_calibration_mean_resid:position={pos}")[0]
    out["captain"] = captain_stats(con, run_id)
    out["captain_counterfactuals"] = captain_counterfactuals(con, run_id)
    out["breakout"] = breakout_cuts(con, run_id)
    out["form_groups"] = form_group_cuts(con, run_id)
    out["db_cache_key"] = os.environ.get("DB_CACHE_KEY") or None
    # per-gameweek rows so two arms can be compared on the SAME scored steps (an arm can lose
    # steps, e.g. to the optimizer's divergence check at very low lambda). squad_points is the
    # model squad's own score: beats_real needs the real average manager, which 2024-25 lacks,
    # so a two-season comparison pairs on squad_points (the same difference where both exist).
    out["per_gameweek"] = [
        {"season": s, "gw": g, "beats_crowd": round(v, 3), "beats_real": None if r is None else round(r, 3),
         "squad_points": None if sp is None else round(sp, 3), "match_log_lik": None if ll is None else round(ll, 4)}
        for s, g, v, r, sp, ll in con.execute(
            "SELECT c.season, c.gameweek, c.metric_value, r.metric_value, sp.metric_value, ll.metric_value "
            "FROM backtest_metrics c "
            "LEFT JOIN backtest_metrics r ON r.backtest_run_id = c.backtest_run_id AND r.season = c.season "
            "AND r.gameweek = c.gameweek AND r.metric_name = 'beats_real_avg_points_delta' "
            "LEFT JOIN backtest_metrics sp ON sp.backtest_run_id = c.backtest_run_id AND sp.season = c.season "
            "AND sp.gameweek = c.gameweek AND sp.metric_name = 'model_squad_realized_points' "
            "LEFT JOIN backtest_metrics ll ON ll.backtest_run_id = c.backtest_run_id AND ll.season = c.season "
            "AND ll.gameweek = c.gameweek AND ll.metric_name = 'match_score_log_lik_mean' "
            "WHERE c.backtest_run_id = ? AND c.metric_name = 'beats_crowd_points_delta' "
            "ORDER BY c.season, c.gameweek", [run_id],
        ).fetchall()
    ]
    # the template yardstick (docs/plans/2026-10_beat_the_template.md): per gameweek on the same
    # rows, then by season and pooled
    templates = template_rows(con, run_id)
    for row in out["per_gameweek"]:
        row.update(templates.get((row["season"], row["gw"]), {}))
    out["template"] = {"pooled": template_block(out["per_gameweek"]), "by_season": {
        season: template_block([r for r in out["per_gameweek"] if r["season"] == season]) for season in seasons
    }}
    seconds = con.execute(
        "SELECT sum(metric_value) FROM backtest_metrics WHERE backtest_run_id = ? AND metric_name = 'template_solve_seconds'",
        [run_id],
    ).fetchone()[0]
    out["template"]["solve_seconds_total"] = None if seconds is None else round(seconds, 1)
    return out


def main() -> None:
    con = db.connect(read_only=True)
    run_id = int(sys.argv[1]) if len(sys.argv) > 1 else con.execute("SELECT max(backtest_run_id) FROM backtest_runs").fetchone()[0]
    print(json.dumps(summarize(con, run_id), indent=2))
    con.close()


if __name__ == "__main__":
    main()
