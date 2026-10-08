"""M7 walk-forward backtest ONLY -- the `backtest.run()` half of scripts/run_backtest.py,
without the recalibration tail.

Why this exists: `research/ml/experiment.py` reads `backtest_gameweek_steps` (the walk-forward's
per-gameweek asof predictions) and nothing else -- it never touches recalibration proposals. But
`scripts/run_backtest.py` runs `backtest.run()` AND `backtest.recalibrate()`, and the recalibration
tail (the lambda-grid SCIP re-solves) alone exceeds a GitHub Actions job's 6-hour hard cap (a real
`weekly_backtest.yml` dispatch on 2026-08-25 was cancelled at 5h having not finished). So the ML
experiment could never be provisioned with a walk-forward on a cloud runner.

`backtest.run()` on its own is the ~1-2 hour part (README's own estimate) and fits comfortably in
one Actions job. This script runs exactly that and prints the same run-summary block
`run_backtest.py` does, so `.github/workflows/ml_experiment.yml` can cache the resulting DB (now
carrying `backtest_gameweek_steps`) and hand it to `python -m research.ml.experiment`.

Recalibration is a separate concern and stays in `scripts/run_backtest.py` -- run that locally, or
on a runner without the 6h limit, when you actually want new `recalibration_proposals`.

Usage (from repo root):
    PYTHONPATH=src python scripts/run_walkforward.py
    PYTHONPATH=src python scripts/run_walkforward.py --lambda 0.10    # an experiment arm

The experiment flags (--lambda, --role-matches-threshold, --role-minutes-blend, --no-current-season-blend,
--assist-prior-xa, --finishing-prior-xg, --minutes-price-prior, --minutes-start-prior, --no-minutes-start-prior)
each swap one setting for a fresh or existing param version, so an arm runs from master via
branch_walkforward.yml's `args` input instead of a bt/** branch. Nothing is activated.

Bonus model (docs/reports/2026-10_open_issues.md, issue 6: premiums get more bonus than their
estimated BPS implies):
    --bps-calibration-k 450    add each player's BPS the estimate's terms miss, from his real
                               season BPS, shrunk toward his position with this k (minutes)
    --bps-tau 7                the Plackett-Luce BPS dispersion (live: 10); smaller gives the
                               top BPS in a match more of the bonus

Minutes start prior (docs/reports/2026-10_open_issues.md, issue 4): live since 2026-10-04 --
a thin history shrinks toward the player's own record this season, else in earlier seasons,
else a start rate rising with price, each level counting the next as 5 matches.
    --minutes-start-prior 3    the same with each level counting the next as 3 matches
    --no-minutes-start-prior   the old prior: the position average
Judge either on 2025-26 (--seasons 2025-2026); 2024-25 is the cold-start stress test.
--minutes-price-prior replaces the live start prior (the two are alternatives).

Team strength (docs/reports/2026-10_promoted_club_strength.md: promoted clubs at league average
live, a club that hasn't scored fitted at attack -13, an end-of-season Elo in every backtest):
    --team-strength honest        as off, with each club's Elo as known at the deadline
    --team-strength live-like     prior seasons only and no Elo for a promoted club, as live was
    --team-strength fix           the guard with a promoted club's own-match Elo as its prior
    --team-strength fix-withheld  the recommended guard (team_strength.GUARD_RECOMMENDED, v1,
                                  live since 2026-10-06): a promoted club's Elo withheld
    --team-strength off           the model before 2026-10-06: no guard, end-of-season Elo

Scoring rates (expected_points' rate prior block, docs/reports/2026-10_rate_prior.md): goal and
assist rates shrink toward a per-position rate rising with price, live since 2026-10-07
(rate_prior_params v1). The --rate-* flags change that live bundle; each one alone is an arm:
    --k-minutes 450                    the shrinkage k (live: 900 since 2026-09-09)
    --no-rate-price-anchor             shrink toward the position average, as before 2026-10-07
    --rate-current-season-weight 2     this season's minutes and returns count double
    --rate-season-decay 0.5            each earlier season counts half the one after it

Long runs (docs/reports/2026-10_open_issues.md: an arm hit the job's 330-minute limit and left
no summary):
    --max-minutes 300          stop before a step that would run past this; the summary marks
                               the run incomplete
    --resume RUN_ID            carry on that run in this DB, walking only its unscored steps
    --seasons 2025-2026        walk only these seasons (comma-separated)
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import backtest, db  # noqa: E402
from fpl_quant import expected_points as ep  # noqa: E402
from fpl_quant import params as params_mod  # noqa: E402
from fpl_quant import team_strength  # noqa: E402

# Reuse run_backtest.py's own version-resolution verbatim -- the walk-forward must measure the
# model against the same git-committed confirmed-seed versions every other script uses, not a
# hardcoded literal (see run_backtest.py's own comment on _param_versions).
from run_backtest import _param_versions, RECALIBRATION_SEED_DIR  # noqa: E402


EXPERIMENT_EFFECTIVE_DATE = "2026-09-30"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lambda", dest="lambda_value", type=float, default=None,
                        help="risk_aversion_params lambda_value for this arm (default: the live version)")
    parser.add_argument("--role-matches-threshold", type=float, default=None,
                        help="the current-season role blend's matches threshold for this arm (live: 4)")
    parser.add_argument("--assist-prior-xa", type=float, default=None,
                        help="turn on the FPL/xA assist calibration with this prior pseudo-count")
    parser.add_argument("--finishing-prior-xg", type=float, default=None,
                        help="turn on per-player finishing skill: (goals + p) / (xG + p), same for assists/xA")
    parser.add_argument("--minutes-price-prior", type=float, default=None, metavar="MIN_BAND_WEIGHT",
                        help="shrink thin minutes histories toward the (position, price band) start rate")
    parser.add_argument("--minutes-start-prior", type=float, default=None, metavar="PSEUDO_MATCHES",
                        help="shrink thin minutes histories toward the player's own record (this season, "
                             "then earlier ones), then a start rate rising with price; each level counts "
                             "the next as this many matches (live: 5)")
    parser.add_argument("--role-minutes-blend", action="store_true",
                        help="blend this season's own P(60+ | started) in at weight min(1, starts/4) "
                             "(live since 2026-10-08)")
    parser.add_argument("--no-current-season-blend", action="store_true",
                        help="the minutes model without this season's own start rate and P(60+ | started) "
                             "blends, as before 2026-10-08")
    parser.add_argument("--no-minutes-start-prior", action="store_true",
                        help="shrink thin minutes histories toward the position average, as before 2026-10-04")
    parser.add_argument("--team-strength", choices=[*sorted(team_strength.GUARD_ARMS), "off"], default=None,
                        help="a team-strength arm (team_strength.GUARD_ARMS), or off for the model before "
                             "2026-10-06; default: the live guard")
    parser.add_argument("--backtest-evidence", action="store_true",
                        help="let each step see evidence claims observed before its deadline (default: none, as before)")
    parser.add_argument("--bps-calibration-k", type=float, default=None, metavar="K_MINUTES",
                        help="turn on the per-player BPS calibration with this shrinkage k")
    parser.add_argument("--bps-tau", type=float, default=None,
                        help="bps_dispersion_params tau for this arm (live: 10)")
    parser.add_argument("--k-minutes", type=float, default=None,
                        help="rate_shrinkage_params k_minutes for this arm (live: 900)")
    parser.add_argument("--no-rate-price-anchor", action="store_true",
                        help="shrink goal/assist rates toward the position average, as before 2026-10-07")
    parser.add_argument("--rate-current-season-weight", type=float, default=None, metavar="WEIGHT",
                        help="weight on the newest season's minutes and returns in a player's rate pool")
    parser.add_argument("--rate-season-decay", type=float, default=None, metavar="DECAY",
                        help="weight of each earlier season relative to the one after it")
    parser.add_argument("--max-minutes", type=float, default=None,
                        help="stop before a step that would likely run past this many minutes")
    parser.add_argument("--resume", type=int, default=None, metavar="RUN_ID",
                        help="carry on this backtest_run_id, walking only its steps not yet scored")
    parser.add_argument("--seasons", default=None,
                        help="comma-separated seasons to walk (default: both)")
    return parser.parse_args(argv)


def _experiment_versions(con, args: argparse.Namespace) -> dict:
    """backtest.run() kwargs for the experiment flags that were given. Each maps to an
    immutable param version (get_or_create_version), never an activation."""
    out: dict = {}
    if args.lambda_value is not None:
        out["lambda_params_version"] = params_mod.get_or_create_version(
            con, "risk_aversion_params", "lambda_value", EXPERIMENT_EFFECTIVE_DATE, value_numeric=args.lambda_value,
        )
    if args.role_matches_threshold is not None:
        out["current_season_role_params_version"] = params_mod.get_or_create_version(
            con, "current_season_role_params", "current_season_matches_threshold", EXPERIMENT_EFFECTIVE_DATE,
            value_numeric=args.role_matches_threshold,
        )
    if args.assist_prior_xa is not None:
        out["assist_calibration_params_version"] = params_mod.get_or_create_version(
            con, "fpl_assist_calibration_params", "prior_xa", EXPERIMENT_EFFECTIVE_DATE, value_numeric=args.assist_prior_xa,
        )
    if getattr(args, "finishing_prior_xg", None) is not None:
        out["finishing_skill_params_version"] = params_mod.get_or_create_version(
            con, "finishing_skill_params", "prior_xg", EXPERIMENT_EFFECTIVE_DATE, value_numeric=args.finishing_prior_xg,
        )
    if getattr(args, "role_minutes_blend", False):
        out["current_season_minutes_params_version"] = params_mod.get_or_create_version(
            con, "current_season_minutes_params", "starts_threshold", EXPERIMENT_EFFECTIVE_DATE, value_numeric=4.0,
        )
    if getattr(args, "no_current_season_blend", False):
        out["current_season_role_params_version"] = None
        out["current_season_minutes_params_version"] = None
    if getattr(args, "minutes_price_prior", None) is not None:
        out["minutes_price_prior_params_version"] = params_mod.get_or_create_version(
            con, "minutes_price_prior_params", "min_band_weight", EXPERIMENT_EFFECTIVE_DATE,
            value_numeric=args.minutes_price_prior,
        )
        # an alternative to the live start prior, not an addition to it
        out["minutes_start_prior_params_version"] = None
    if getattr(args, "no_minutes_start_prior", False):
        out["minutes_start_prior_params_version"] = None
    if getattr(args, "minutes_start_prior", None) is not None:
        out["minutes_start_prior_params_version"] = params_mod.get_or_create_version(
            con, "minutes_start_prior_params", "pseudo_matches", EXPERIMENT_EFFECTIVE_DATE,
            value_numeric=args.minutes_start_prior,
        )
    if getattr(args, "team_strength", None) == "off":
        out["team_strength_guard_params_version"] = None
    elif getattr(args, "team_strength", None):
        out["team_strength_guard_params_version"] = params_mod.get_or_create_bundle_version(
            con, team_strength.GUARD_FAMILY, team_strength.GUARD_ARMS[args.team_strength], EXPERIMENT_EFFECTIVE_DATE,
        )
    if getattr(args, "backtest_evidence", False):
        out["backtest_evidence"] = True
    if getattr(args, "bps_calibration_k", None) is not None:
        out["bps_calibration_params_version"] = params_mod.get_or_create_version(
            con, "bps_calibration_params", "k_minutes", EXPERIMENT_EFFECTIVE_DATE, value_numeric=args.bps_calibration_k,
        )
    if getattr(args, "k_minutes", None) is not None:
        out["rate_shrinkage_params_version"] = params_mod.get_or_create_version(
            con, "rate_shrinkage_params", "k_minutes", EXPERIMENT_EFFECTIVE_DATE, value_numeric=args.k_minutes,
        )
    weight, decay = getattr(args, "rate_current_season_weight", None), getattr(args, "rate_season_decay", None)
    no_anchor = getattr(args, "no_rate_price_anchor", False)
    if weight is not None or decay is not None:
        # the live bundle with these knobs changed
        bundle = dict(ep.LIVE_RATE_PRIOR)
        if no_anchor:
            bundle["price_anchor"] = 0.0
        if weight is not None:
            bundle["current_season_weight"] = weight
        if decay is not None:
            bundle["season_decay"] = decay
        out["rate_prior_params_version"] = params_mod.get_or_create_bundle_version(
            con, "rate_prior_params", bundle, EXPERIMENT_EFFECTIVE_DATE,
        )
    elif no_anchor:
        out["rate_prior_params_version"] = None
    if getattr(args, "bps_tau", None) is not None:
        out["tau_params_version"] = params_mod.get_or_create_version(
            con, "bps_dispersion_params", "tau", EXPERIMENT_EFFECTIVE_DATE, value_numeric=args.bps_tau,
        )
    return out


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    con = db.connect()
    # The cached DB can predate a newly committed seed; make sure every active version exists.
    backtest.materialize_confirmed_seeds(con, RECALIBRATION_SEED_DIR)
    active = backtest.active_recalibratable_versions(RECALIBRATION_SEED_DIR)
    param_versions = _param_versions(active)
    experiment = _experiment_versions(con, args)
    param_versions.update({k: v for k, v in experiment.items() if k in param_versions})
    extra = {k: v for k, v in experiment.items() if k not in param_versions}
    if experiment:
        print(f"[experiment] {vars(args)} -> {experiment}")

    t0 = time.time()
    seasons = tuple(s.strip() for s in args.seasons.split(",") if s.strip()) if args.seasons else None
    backtest_run_id = backtest.run(
        con, **param_versions, n_antithetic_pairs=5000, run_monte_carlo=True,
        seasons=seasons,
        stop_after_seconds=args.max_minutes * 60 if args.max_minutes is not None else None,
        resume_backtest_run_id=args.resume,
        # compute_segments: the position / price_band / promoted_team / new_signing /
        # set_piece_taker breakdowns of every scored metric -- the diagnostic axis for "where is
        # the EP model biased" (nightly_backtest.yml -> app_track_record.json's segment_calibration).
        # ownership_params_version: makes beats_crowd_points_delta ("does the model's own weekly
        # XI beat the ownership-weighted average manager") measurable from the walk-forward
        # itself -- previously only the run_season_simulation() leaderboard path computed it, so
        # the Track Record headline (BUSINESS_PLAN.md P0) sat null between weekly runs.
        compute_segments=True,
        ownership_params_version=1,
        **extra,
        # Fix D (captain weight 0) and Fix F (minutes floor 0.005) come in via _param_versions().
        notes="M7 walk-forward (ml_experiment.yml provisioning -- no recalibration)",
    )
    print(f"[backtest.run] {time.time() - t0:.1f}s -> backtest_run_id={backtest_run_id}")
    progress = backtest.walk_forward_progress(con, backtest_run_id)
    print(f"[walk-forward] {progress['steps_scored']}/{progress['steps_planned']} steps scored"
          + ("" if progress["complete"] else f" -- INCOMPLETE, resume with --resume {backtest_run_id}"))

    steps = con.execute(
        "SELECT tier, count(*), sum(CASE WHEN divergence_check_passed THEN 1 ELSE 0 END) "
        "FROM backtest_gameweek_steps WHERE backtest_run_id = ? GROUP BY tier ORDER BY tier",
        [backtest_run_id],
    ).fetchall()
    for tier, n, n_passed in steps:
        print(f"  tier={tier}: {n} steps, {n_passed} divergence-check passes")

    metrics = con.execute(
        "SELECT tier, metric_name, count(*), avg(metric_value) FROM backtest_metrics "
        "WHERE backtest_run_id = ? AND metric_name NOT LIKE 'realized%' AND metric_name NOT LIKE '%:%' "
        "GROUP BY tier, metric_name ORDER BY metric_name, tier",
        [backtest_run_id],
    ).fetchall()
    for tier, name, n, avg in metrics:
        print(f"  [{tier}] {name}: n={n} mean={avg:.4f}")

    # ep_total calibration by position / price band (mature tier) -- the "where is the EP model
    # biased" read, printed here so a cloud run's log carries it without a follow-up SQL query.
    seg = con.execute(
        "SELECT metric_name, count(*), avg(metric_value) FROM backtest_metrics "
        "WHERE backtest_run_id = ? AND tier = 'mature' "
        "AND (metric_name LIKE 'ep_total_calibration_mean_resid:position=%' "
        "     OR metric_name LIKE 'ep_total_calibration_mean_resid:price_band=%') "
        "GROUP BY metric_name ORDER BY metric_name",
        [backtest_run_id],
    ).fetchall()
    if seg:
        print("[ep_total calibration -- mature tier, signed resid = realized - predicted]")
        for name, n, avg in seg:
            print(f"  {name.split(':', 1)[1]}: n={n} mean_resid={avg:+.3f}")

    n_pred = con.execute(
        "SELECT count(*) FROM backtest_gameweek_steps WHERE backtest_run_id = ? AND ep_model_version IS NOT NULL",
        [backtest_run_id],
    ).fetchone()[0]
    print(f"[walk-forward] {n_pred} gameweek steps carry an ep_model_version -- research.ml.experiment can now run")

    con.close()


if __name__ == "__main__":
    main()
