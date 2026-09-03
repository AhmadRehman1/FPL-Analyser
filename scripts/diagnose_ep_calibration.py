"""Where is the EP model biased? Prints predicted ep_total vs realized event_points, broken
down by position and price band, from a walk-forward backtest run.

This is the diagnostic instrument for the "cheap defenders over premium attackers" tilt the
README's M5/M6 notes describe, the retrospective validation measured (6.9th percentile vs
2,000 real managers), and the live model-managed team is reproducing (-38 pts vs field in 2
GWs). It reads what backtest.score_gameweek(compute_segments=True) records -- so it needs a
walk-forward run that carried the position / price_band segments (nightly_backtest.yml after
PR #124), OR it falls back to computing the residuals itself from ep_outputs +
fact_player_season_stats for whatever gameweeks a run scored.

Usage:
    PYTHONPATH=src python scripts/diagnose_ep_calibration.py [backtest_run_id]

With no argument, uses max(backtest_run_id). Read-only -- writes nothing.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import db  # noqa: E402


def _fmt(v, spec="{:+.3f}"):
    return spec.format(v) if isinstance(v, (int, float)) else "  -  "


def main() -> None:
    con = db.connect()
    run_id = int(sys.argv[1]) if len(sys.argv) > 1 else con.execute(
        "SELECT max(backtest_run_id) FROM backtest_runs"
    ).fetchone()[0]
    if run_id is None:
        print("no backtest_runs row (run scripts/run_walkforward.py, or nightly_backtest.yml).")
        print("Falling back to the live ep_outputs predicted-EP DISTRIBUTION -- not calibration,")
        print("but it still shows whether cheap defenders cluster near the premiums.\n")
        _live_distribution(con)
        con.close()
        return
    print(f"=== EP calibration diagnosis, backtest_run_id={run_id} ===\n")

    n_steps, seasons = con.execute(
        "SELECT count(*), string_agg(DISTINCT season, ',') FROM backtest_gameweek_steps WHERE backtest_run_id = ?",
        [run_id],
    ).fetchone()
    print(f"{n_steps} scored gameweek-steps over {seasons}\n")

    # --- 1. whole-population EP bias, by tier ---
    print("-- ep_total calibration (signed: realized event_points - predicted ep_total; >0 = model UNDER-predicts) --")
    rows = con.execute(
        "SELECT tier, metric_name, avg(metric_value), count(*) FROM backtest_metrics "
        "WHERE backtest_run_id = ? AND metric_name IN ('ep_total_calibration_mean_resid','ep_total_calibration_mae') "
        "GROUP BY tier, metric_name ORDER BY metric_name, tier",
        [run_id],
    ).fetchall()
    if rows:
        for tier, name, mean, n in rows:
            print(f"  [{tier:6}] {name}: {_fmt(mean)}  (n={n})")
    else:
        print("  (no ep_total_calibration_* metrics on this run -- pre-#124 walk-forward; using the fallback below)")

    # --- 2. by position / price band (needs PR #124's segments) ---
    seg = con.execute(
        "SELECT metric_name, avg(metric_value), count(*) FROM backtest_metrics "
        "WHERE backtest_run_id = ? AND tier = 'mature' "
        "AND metric_name LIKE 'ep_total_calibration_mean_resid:%' "
        "GROUP BY metric_name ORDER BY metric_name",
        [run_id],
    ).fetchall()
    if seg:
        print("\n-- ep_total bias by segment (mature tier) --")
        for name, mean, n in seg:
            print(f"  {name.split(':', 1)[1]:22} {_fmt(mean)}  (n={n})")
        print("\n  Reading: a large +resid on position=Forward with a -resid on the cheap DEF/GKP")
        print("  price bands is the 'premiums under-valued, cheap defenders over-valued' tilt.")
    else:
        print("\n-- no segment rows on this run -- computing residuals directly from ep_outputs --")
        _fallback_direct(con, run_id)

    # --- 3. per-category log scores by position (also #124) ---
    catseg = con.execute(
        "SELECT metric_name, avg(metric_value), count(*) FROM backtest_metrics "
        "WHERE backtest_run_id = ? AND tier = 'mature' "
        "AND (metric_name LIKE 'log_score_clean_sheet_mean:position=%' "
        "     OR metric_name LIKE 'log_score_goals_mean:position=%' "
        "     OR metric_name LIKE 'log_score_assists_mean:position=%') "
        "GROUP BY metric_name ORDER BY metric_name",
        [run_id],
    ).fetchall()
    if catseg:
        print("\n-- per-category log score by position (mature; less negative = better calibrated) --")
        for name, mean, n in catseg:
            print(f"  {name.split(':', 1)[1]:22} {name.split(':')[0]:26} {_fmt(mean, '{:.4f}')}  (n={n})")

    con.close()


def _live_distribution(con) -> None:
    ep_mv = con.execute("SELECT max(model_version) FROM ep_outputs").fetchone()[0]
    if ep_mv is None:
        raise SystemExit("no ep_outputs either -- run scripts/run_ingestion.py")
    rows = con.execute(
        """
        SELECT dp.position,
               CASE WHEN fps.now_cost < 5.0 THEN '<5.0' WHEN fps.now_cost < 7.0 THEN '5.0-7.0'
                    WHEN fps.now_cost < 9.0 THEN '7.0-9.0' ELSE '9.0+' END AS band,
               o.ep_total, o.ep_clean_sheet + o.ep_defcon + o.ep_goals_conceded AS ep_defensive
        FROM ep_outputs o
        JOIN dim_player dp ON dp.player_uid = o.player_uid
        JOIN fact_player_season_stats fps ON fps.player_uid = o.player_uid
        WHERE o.model_version = ? AND fps.now_cost IS NOT NULL
        QUALIFY row_number() OVER (PARTITION BY o.player_uid ORDER BY fps.gw DESC) = 1
        """,
        [ep_mv],
    ).fetchall()
    by: dict = {}
    for pos, band, ep_total, ep_def in rows:
        by.setdefault(("position", pos), []).append((ep_total, ep_def))
        by.setdefault(("price_band", band), []).append((ep_total, ep_def))
    print(f"{'segment':22} {'n':>4} {'mean ep_total':>13} {'top-10 mean':>12} {'mean ep_defensive':>18}")
    for (dim, key), vals in sorted(by.items()):
        tot = sorted((t for t, _ in vals), reverse=True)
        top10 = sum(tot[:10]) / max(len(tot[:10]), 1)
        mean_def = sum(d for _, d in vals) / len(vals)
        print(f"{dim + '=' + str(key):22} {len(vals):>4} {sum(tot)/len(tot):>13.2f} {top10:>12.2f} {mean_def:>18.2f}")
    print("\nIf DEF/GKP 'top-10 mean' sits within ~0.5 of FWD/MID, the model is compressing the")
    print("premium ceiling and/or inflating cheap defensive floors -- confirm against realized")
    print("points with a walk-forward run (this is only the predicted distribution).")


def _fallback_direct(con, run_id: int) -> None:
    """Compute realized event_points - ep_total by position/price for whatever (season, gw) this
    run's steps cover, straight from ep_outputs joined to fact_player_season_stats -- so the
    diagnosis works even against a pre-#124 walk-forward (or the live ep_outputs)."""
    steps = con.execute(
        "SELECT DISTINCT season, gameweek FROM backtest_gameweek_steps WHERE backtest_run_id = ?", [run_id],
    ).fetchall()
    if not steps:
        print("  (no gameweek steps to fall back on)")
        return
    step_ep = dict(con.execute(
        "SELECT (season || ':' || gameweek), ep_model_version FROM backtest_gameweek_steps "
        "WHERE backtest_run_id = ? AND ep_model_version IS NOT NULL", [run_id],
    ).fetchall())
    agg: dict = {}
    for season, gw in steps:
        ep_mv = step_ep.get(f"{season}:{gw}")
        if ep_mv is None:
            continue
        for pos, band, resid in con.execute(
            """
            SELECT dp.position,
                   CASE WHEN fps.now_cost < 5.0 THEN '<5.0' WHEN fps.now_cost < 7.0 THEN '5.0-7.0'
                        WHEN fps.now_cost < 9.0 THEN '7.0-9.0' ELSE '9.0+' END,
                   fps.event_points - o.ep_total
            FROM ep_outputs o
            JOIN dim_player dp ON dp.player_uid = o.player_uid
            JOIN fact_player_season_stats fps ON fps.player_uid = o.player_uid AND fps.season = ? AND fps.gw = ?
            WHERE o.model_version = ? AND fps.event_points IS NOT NULL AND fps.now_cost IS NOT NULL
            """,
            [season, gw, ep_mv],
        ).fetchall():
            agg.setdefault(("position", pos), []).append(resid)
            agg.setdefault(("price_band", band), []).append(resid)
    if not agg:
        print("  (no overlapping ep_outputs / realized rows)")
        return
    for (dim, key), vals in sorted(agg.items()):
        mean = sum(vals) / len(vals)
        mae = sum(abs(v) for v in vals) / len(vals)
        print(f"  {dim}={key:10} mean_resid={mean:+.3f}  mae={mae:.3f}  (n={len(vals)})")


if __name__ == "__main__":
    main()
