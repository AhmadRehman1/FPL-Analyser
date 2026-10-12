"""Phase 3 of docs/plans/2026-10_beat_the_template.md: the chain table (oracle model -> season sim ->
live) and G2's margin M, computed exactly as the plan declares it. Prints JSON.

Per season s in {2024-25, 2025-26}, in pts/GW:
    M_s = max(2.0, C_s + E + O+ - T_s)
  C_s  the season's real top-100k cutoff pace (rank_autopsy.json top_100k_cutoff, Req 4);
  T_s  the gated template's mean squad points over the season's walk-forward gameweeks (the
       control walk-forward summary's template block);
  E    the evolving loss: the oracle model's squad points minus the season-sim control's net
       points, paired over the 2025-26 gameweeks both scored (used for both seasons);
  O+   the live optimism O = the 2026-27 leaderboard simulation's net pace minus the live
       team's net pace over the same gameweeks, kept only when O > 2 SE(O), else 0.
M-bar is the mean of M_2024-25 and M_2025-26 weighted by each season's template gameweeks, the
way the walk-forward's pooled gap m weighs them. G2 holds for a run when its pooled gap to the
gated template m >= M-bar and m - SE > 0.

Usage (from repo root):
    python scripts/beat_template_margin.py --control walkforward_summary.json \\
        --season-sim control.json [--season-sim-b control_b.json] \\
        [--leaderboard data/dashboard/leaderboard_latest.json] [--model-team data/dashboard/app_model_team.json] \\
        [--autopsy data/dashboard/rank_autopsy.json]
"""

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = REPO_ROOT / "data" / "dashboard"
SEASONS = ("2024-2025", "2025-2026")
SIM_SEASON = "2025-2026"
LIVE_SEASON = "2026-2027"
M_FLOOR = 2.0


def paired(diffs: list[float]) -> dict:
    n = len(diffs)
    if n == 0:
        return {"n": 0, "mean": None, "se": None}
    m = sum(diffs) / n
    se = math.sqrt(sum((d - m) ** 2 for d in diffs) / (n - 1)) / math.sqrt(n) if n > 1 else None
    return {"n": n, "mean": round(m, 3), "se": None if se is None else round(se, 3)}


def oracle_by_gw(control: dict, season: str) -> dict[int, float]:
    return {r["gw"]: r["squad_points"] for r in control["per_gameweek"]
            if r["season"] == season and r.get("squad_points") is not None}


def season_sim_net(sim: dict) -> dict[int, float]:
    """{gw: net points} from a season_sim_arms arm file (weekly points minus hits)."""
    hits = sim.get("weekly_hits") or [0.0] * len(sim["weekly_points"])
    return {gw: pts - hit for gw, pts, hit in zip(sim["gameweeks"], sim["weekly_points"], hits)}


def evolving_loss(control: dict, sim: dict) -> dict:
    """E: oracle model minus season-sim control, paired per 2025-26 gameweek."""
    oracle, net = oracle_by_gw(control, SIM_SEASON), season_sim_net(sim)
    return paired([oracle[gw] - net[gw] for gw in sorted(set(oracle) & set(net))])


def live_optimism(leaderboard: dict | None, model_team: dict | None) -> dict:
    """O: the 2026-27 leaderboard simulation's net points minus the live team's, per gameweek
    both have (the live team's settled, non-provisional weeks)."""
    if not leaderboard or not model_team or leaderboard.get("status") != "ok":
        return {"n": 0, "mean": None, "se": None, "reason": "no leaderboard or live team yet"}
    model_row = next((r for r in leaderboard["rows"] if not r.get("is_baseline")), None)
    if not model_row or "weekly_points" not in model_row:
        return {"n": 0, "mean": None, "se": None, "reason": "leaderboard written before per-gameweek rows"}
    hits = model_row.get("weekly_hits") or {}
    sim = {int(gw): pts - hits.get(gw, 0.0) for gw, pts in model_row["weekly_points"].items()}
    live = {w["gameweek"]: w["realized_points"] - (w.get("hit_cost") or 0.0) for w in model_team.get("weeks", [])
            if not w.get("provisional") and w.get("realized_points") is not None}
    return paired([sim[gw] - live[gw] for gw in sorted(set(sim) & set(live))])


def o_plus(o: dict) -> float:
    if o.get("mean") is None or o.get("se") is None:
        return 0.0
    return o["mean"] if o["mean"] > 2 * o["se"] else 0.0


def margin(control: dict, sim: dict | None, cutoffs: dict | None, o: dict) -> dict:
    """C_s, T_s, E, O+, M_s and M-bar (None where an input is missing), and the G2 decision for
    the control itself."""
    block = (control.get("template") or {}).get("by_season") or {}
    e = evolving_loss(control, sim) if sim else {"n": 0, "mean": None, "se": None}
    op = o_plus(o)
    seasons = {}
    for season in SEASONS:
        t = (block.get(season) or {}).get("template_points_per_gw")
        n = (block.get(season) or {}).get("n") or 0
        c = ((cutoffs or {}).get("seasons") or {}).get(season) or {}
        c_pace = c.get("pace_per_gw")
        m_s = None
        if None not in (t, c_pace, e["mean"]):
            m_s = round(max(M_FLOOR, c_pace + e["mean"] + op - t), 3)
        seasons[season] = {"C": c_pace, "T": t, "n_gameweeks": n, "M": m_s}
    weights = [(v["M"], v["n_gameweeks"]) for v in seasons.values()]
    m_bar = None
    if all(m is not None and w for m, w in weights):
        m_bar = round(sum(m * w for m, w in weights) / sum(w for _, w in weights), 3)
    gap = ((control.get("template") or {}).get("pooled") or {}).get("gap_vs_gated") or {}
    g2 = None
    if m_bar is not None and gap.get("mean") is not None and gap.get("se") is not None:
        g2 = gap["mean"] >= m_bar and gap["mean"] - gap["se"] > 0
    return {"E": e, "O": o, "O_plus": op, "seasons": seasons, "M_bar": m_bar,
            "control_gap_vs_gated": gap, "G2_met_by_control": g2}


def chain(control: dict, sim: dict | None, sim_b: dict | None, leaderboard: dict | None,
          model_team: dict | None, cutoffs: dict | None) -> list[dict]:
    """The chain table: each row's net pts/GW, FPL's real average over the same gameweeks where
    known, and that season's cutoff pace."""
    def pace(season):
        return (((cutoffs or {}).get("seasons") or {}).get(season) or {}).get("pace_per_gw")

    def row(name, season, points: dict, real: dict | None = None, cutoff=None):
        pts = list(points.values())
        mean = round(sum(pts) / len(pts), 3) if pts else None
        real_vals = [real[gw] for gw in points if real and real.get(gw) is not None]
        vs_real = round(mean - sum(real_vals) / len(real_vals), 3) if real_vals and len(real_vals) == len(pts) else None
        cut = pace(season) if cutoff is None else cutoff
        return {"row": name, "season": season, "n_gameweeks": len(pts), "pts_per_gw": mean,
                "vs_fpl_average": vs_real, "cutoff_pace": cut,
                "vs_cutoff": None if mean is None or cut is None else round(mean - cut, 3)}

    rows = []
    for season in SEASONS:
        real = {r["gw"]: r["squad_points"] - r["beats_real"] for r in control["per_gameweek"]
                if r["season"] == season and r.get("beats_real") is not None and r.get("squad_points") is not None}
        rows.append(row("oracle model", season, oracle_by_gw(control, season), real))
        template = {r["gw"]: r["template_points"] for r in control["per_gameweek"]
                    if r["season"] == season and r.get("template_points") is not None}
        rows.append(row("gated template", season, template, real))
    for name, s in (("season sim control", sim), ("season sim control_b", sim_b)):
        if s:
            real = dict(zip(s["gameweeks"], s.get("weekly_real_avg") or []))
            rows.append(row(name, SIM_SEASON, season_sim_net(s), real))
    if leaderboard and leaderboard.get("status") == "ok":
        live_real = {w["gameweek"]: w.get("field_average") for w in (model_team or {}).get("weeks", [])}
        settled = [w["gameweek"] for w in (model_team or {}).get("weeks", [])
                   if not w.get("provisional") and w.get("realized_points") is not None]
        by_gw = (((cutoffs or {}).get("current_season") or {}).get("by_gameweek")) or []
        # the cutoff's pace up to the live team's last settled gameweek (a later, unfinished one is provisional)
        usable = [g for g in by_gw if not settled or g["gameweek"] <= max(settled)]
        live_cut = round(usable[-1]["total_points"] / usable[-1]["gameweek"], 3) if usable else None
        for r in leaderboard["rows"]:
            if r.get("kind") == "model_oracle" or not r.get("is_baseline"):
                weekly = {int(gw): pts for gw, pts in (r.get("weekly_points") or {}).items()}
                if not r.get("is_baseline"):
                    hits = r.get("weekly_hits") or {}
                    weekly = {gw: pts - hits.get(str(gw), 0.0) for gw, pts in weekly.items()}
                name = "model (oracle)" if r.get("is_baseline") else "leaderboard sim"
                rows.append(row(name, LIVE_SEASON, weekly, live_real, live_cut))
        live = {w["gameweek"]: w["realized_points"] - (w.get("hit_cost") or 0.0) for w in (model_team or {}).get("weeks", [])
                if not w.get("provisional") and w.get("realized_points") is not None}
        rows.append(row("live team", LIVE_SEASON, live, live_real, live_cut))
    return rows


def _load(path):
    return json.loads(Path(path).read_text()) if path and Path(path).exists() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--control", required=True, help="the control walk-forward's walkforward_summary.json")
    parser.add_argument("--season-sim", help="the season-sim control's arm file (2025-26, GW2-38)")
    parser.add_argument("--season-sim-b", help="control_b's arm file, same job")
    parser.add_argument("--leaderboard", default=str(DASHBOARD / "leaderboard_latest.json"))
    parser.add_argument("--model-team", default=str(DASHBOARD / "app_model_team.json"))
    parser.add_argument("--autopsy", default=str(DASHBOARD / "rank_autopsy.json"))
    args = parser.parse_args(argv)
    control = _load(args.control)
    sim, sim_b = _load(args.season_sim), _load(args.season_sim_b)
    leaderboard, model_team = _load(args.leaderboard), _load(args.model_team)
    cutoffs = (_load(args.autopsy) or {}).get("top_100k_cutoff")
    o = live_optimism(leaderboard, model_team)
    out = {"chain": chain(control, sim, sim_b, leaderboard, model_team, cutoffs),
           "margin": margin(control, sim, cutoffs, o)}
    if sim and sim_b:
        a, b = season_sim_net(sim), season_sim_net(sim_b)
        out["season_sim_noise_floor"] = paired([b[gw] - a[gw] for gw in sorted(set(a) & set(b))])
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
