"""First-half-of-season chip-timing roadmap for both real FPL accounts this project tracks,
computed from fixture_swing.py's real rolling swing scores (Dixon-Coles fixture-difficulty
deltas from THIS run's real ts_model_version, not invented dates).

For each gameweek in FIRST_HALF_GAMEWEEKS, averages each account's own squad's teams' swing
scores (negative = an easier-than-average near-term run for that team, per fixture_swing.py's
own sign convention) into one number for that gameweek -- the single easiest window across the
first half is flagged as a bench_boost/triple_captain candidate (several of the squad's teams
enjoying good fixtures at once); the toughest is flagged as a wildcard-consideration window
(several teams hitting a difficult patch simultaneously). A real, computed signal from today's
actual squad composition, not a locked-in plan -- transfers made along the way will change
which teams are actually in the squad by the time any of these gameweeks arrive.

LONG_WINDOW=8 (not fixture_swing.py's own short_window=3/long_window=6 defaults): real FPL
chip-timing guidance frames a wildcard's payoff as "does this create a clearly stronger squad
for the next five to eight gameweeks," a materially wider forward window than a bare 6-GW
default -- see this module's own git history/PR notes for the actual research this was checked
against.

Wildcard candidate windows also get ONE real evaluate_wildcard() check (the same M5/M8
mechanism the live per-gameweek decision path uses, not a second independently-invented
metric) -- the fixture-swing signal alone is a real, computed transition detector, but it's
purely qualitative (which window looks toughest), not a magnitude (how many points a wildcard
there is actually projected to gain, or whether that clears the real recommendation
threshold). Bounded to the ALREADY-CHOSEN candidate gameweek per account (not a sweep across
every gameweek in the range) -- a real MIQP solve plus a multi-gameweek EP horizon is
materially more expensive than the swing computation, so this only pays that cost once the
swing signal has already narrowed it to one gameweek.

Also reports how many real Premier League matches have actually been played so far this
season (a plain fact, not a derived confidence score) -- real FPL strategy guidance is explicit
that fixture/form signals "only settle after about six gameweeks," so an early-season
candidate window (few matches played yet) is disclosed as resting on a thinner sample, not
silently presented as equally solid as a later one.

Needs the same freshly-ingested database run_ingestion.py just built (same real-network-access
caveat as run_transfer_planner_for_real_squad.py's own module docstring: only runs somewhere
with open internet, not this project's own dev sandbox).

Usage (from repo root, same job as run_ingestion.py):
    PYTHONPATH=src python scripts/print_chip_timing_roadmap.py
"""

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import app_export as ax, db, fixture_swing as fs, ingest_fpl_entry_picks as ifp, ingest_workbook as iw  # noqa: E402
from fpl_quant import params as params_mod, squad_optimizer as so_mod, transfer_planner as tp  # noqa: E402

TARGET_SEASON = "2026-2027"
TS_MODEL_VERSION = 1
# GW1-2 already have their own real, computed transfer plan (run_transfer_planner_for_real_squad.py);
# this roadmap covers the rest of the first half. 19 is a reasonable midpoint for a 38-gameweek
# season -- not FPL's own real confirmed winter-break gameweek for 2026-27, which this project
# doesn't have visibility into beyond what's already in its ingested fixture data.
FIRST_HALF_GAMEWEEKS = range(3, 20)
SHORT_WINDOW_GAMEWEEKS = 3
LONG_WINDOW_GAMEWEEKS = 8
# Free Hit reverts after ONE gameweek (see evaluate_free_hit()'s own docstring: "a single
# gameweek with an unusually poor fixture swing... not DGW exploitation"), so its candidate
# window is the single worst gameweek relative to the surrounding baseline, not a multi-
# gameweek transition the way Wildcard's own SHORT_WINDOW_GAMEWEEKS=3 signal is -- short_window
# of exactly 1 reuses rolling_swing_score()'s own short-vs-long delta mechanism unchanged, just
# parameterized for a single-week chip instead of inventing a second metric.
FREE_HIT_SHORT_WINDOW_GAMEWEEKS = 1
DASHBOARD_DIR = REPO_ROOT / "data" / "dashboard"

# Every one of these is currently seeded at version=1 by scripts/run_ingestion.py -- same
# real-solve param set explain_my_move.py/grade_squad.py already use for a real account.
PARAM_VERSIONS = dict(
    scoring_params_version=1, bps_params_version=1, tau_params_version=1,
    rho_residual_params_version=2, corr_params_version=1,
)

ACCOUNTS = [
    {"entry_id": 7139944, "label": "ChatGPT template team"},
    {"entry_id": 1305242, "label": "Main account"},
]


def _account_team_uids(con, entry_id: int, event: int, team_by_player: dict[str, str]) -> set[str]:
    element_names = ifp.fetch_bootstrap_elements()
    picks = ifp.fetch_entry_picks(entry_id, event)
    if not picks:
        return set()
    uids = set()
    for p in picks:
        name = element_names.get(p["element"])
        player_uid = iw._resolve_player(con, name, TARGET_SEASON)
        if player_uid and player_uid in team_by_player:
            uids.add(team_by_player[player_uid])
    return uids


def _real_wildcard_gain(con, entry_id: int, event: int, target_gameweek: int) -> dict | None:
    """One real evaluate_wildcard() check at target_gameweek -- see module docstring for why
    this is bounded to a single already-chosen candidate rather than a sweep. Best-effort: any
    failure (a real fetch/solve issue for this one account) is caught and disclosed via a
    printed warning, returning None -- one account's real-EP check failing must never blank
    the whole roadmap, which still has the real fixture-swing signal either way."""
    try:
        element_names = ifp.fetch_bootstrap_elements()
        picks = ifp.fetch_entry_picks(entry_id, event)
        if not picks:
            return None
        squad = [
            {"player_name": element_names.get(p["element"]), "in_xi": p["position"] <= 11,
             "is_captain": bool(p.get("is_captain")), "is_vice": bool(p.get("is_vice_captain"))}
            for p in picks
        ]
        calibration_asof_date = date.today()
        ts_mv = con.execute("SELECT max(model_version) FROM team_strength_model_versions").fetchone()[0]
        mm_mv = con.execute("SELECT max(model_version) FROM minutes_model_versions").fetchone()[0]
        if ts_mv is None or mm_mv is None:
            return None

        # Bootstrap needs a real ep_model_version/uncertainty_model_version for the CURRENT
        # event first (same two-step "bootstrap, then evaluate elsewhere" pattern
        # explain_my_move.py/grade_squad.py already establish).
        bootstrap_horizon = tp.compute_horizon_ep(
            con, calibration_asof_date, TARGET_SEASON, event, ts_mv, mm_mv, 1, **PARAM_VERSIONS,
        )
        if event not in bootstrap_horizon:
            return None
        ep_mv, un_mv = bootstrap_horizon[event]
        state_version = tp.bootstrap_from_real_squad(con, calibration_asof_date, TARGET_SEASON, event, ep_mv, un_mv, squad)
        holdings = tp._read_holdings(con, state_version)

        # evaluate_wildcard()'s own horizon starts at target_gameweek, not the current event --
        # a separate real horizon, not the bootstrap one above.
        #
        # Real bug fixed here: this used to hardcode horizon_gameweeks=1, a single-gameweek EP
        # sum, even though evaluate_wildcard()'s own threshold (wildcard_gain_threshold_params.
        # min_horizon_gain=8.0) is calibrated for the SAME 5-gameweek horizon
        # transfer_planner.run() always uses live (planning_horizon_params.horizon_gameweeks,
        # seeded to 5 -- see seed_v1_params()). A 1-gameweek EP sum can essentially never clear
        # an 8.0-point threshold sized for a 5-gameweek sum, so this silently made the wildcard-
        # gain check (the one real EP-magnitude check this module's own docstring says it
        # exists to provide -- "a materially wider forward window than a bare 6-GW default")
        # report "does NOT clear the threshold" almost unconditionally, even for a genuinely
        # strong wildcard window. Resolved the same live horizon transfer_planner.run() uses,
        # not a second, independently-invented number, per this module's own stated principle.
        horizon_gameweeks, _ = params_mod.resolve_param(con, "planning_horizon_params", "horizon_gameweeks", 1)
        wc_horizon = tp.compute_horizon_ep(
            con, calibration_asof_date, TARGET_SEASON, target_gameweek, ts_mv, mm_mv, int(horizon_gameweeks), **PARAM_VERSIONS,
        )
        if target_gameweek not in wc_horizon:
            return None
        horizon_ep_map = tp._horizon_ep_by_player(con, TARGET_SEASON, wc_horizon)
        current_squad_horizon_value = sum(
            horizon_ep_map.get(h["player_uid"], {}).get("total_ep", 0.0) for h in holdings
        )

        so_mod.seed_v1_params(con)
        tp.seed_v1_params(con)
        result = tp.evaluate_wildcard(
            con, calibration_asof_date, TARGET_SEASON, target_gameweek,
            current_squad_horizon_value,
            # Disclosed simplification: the roadmap's own baseline is the current squad
            # UNCHANGED (best_transfer_net_value=0.0), not "current squad plus the single best
            # available transfer" the live per-gameweek decision path credits -- finding that
            # real best-single-transfer alternative needs its own full transfer_planner.run(),
            # a second expensive solve this forward-looking roadmap doesn't also pay for. This
            # makes the reported gain a slight overstatement of wildcarding's real edge over
            # the best realistic alternative, never an understatement.
            best_transfer_net_value=0.0,
            horizon_ep_versions=wc_horizon,
            lambda_params_version=1, guardrail_params_version=1, threshold_params_version=1,
            current_holdings=holdings,
        )
        return {"gain": round(result["gain"], 2), "recommended": result["recommended"]}
    except Exception as e:  # noqa: BLE001 -- best-effort, see this function's own docstring
        print(f"::warning::print_chip_timing_roadmap: real wildcard-gain check failed for entry_id={entry_id} at GW{target_gameweek} ({e}) -- omitted.")
        return None


def _real_free_hit_gain(con, entry_id: int, event: int, target_gameweek: int) -> dict | None:
    """One real evaluate_free_hit() check at target_gameweek -- same best-effort contract as
    _real_wildcard_gain() above (any failure is caught, disclosed, and returns None; the
    roadmap's fixture-swing signal still stands on its own either way).

    Deliberately a 1-gameweek horizon, not _real_wildcard_gain()'s 5-gameweek
    planning_horizon_params: evaluate_free_hit()'s own fresh_gw_value/current_gw_value are both
    single-gameweek sums (the squad reverts after target_gameweek), so a wider horizon would
    just waste a compute_horizon_ep() call building EP versions this evaluator never reads."""
    try:
        element_names = ifp.fetch_bootstrap_elements()
        picks = ifp.fetch_entry_picks(entry_id, event)
        if not picks:
            return None
        squad = [
            {"player_name": element_names.get(p["element"]), "in_xi": p["position"] <= 11,
             "is_captain": bool(p.get("is_captain")), "is_vice": bool(p.get("is_vice_captain"))}
            for p in picks
        ]
        calibration_asof_date = date.today()
        ts_mv = con.execute("SELECT max(model_version) FROM team_strength_model_versions").fetchone()[0]
        mm_mv = con.execute("SELECT max(model_version) FROM minutes_model_versions").fetchone()[0]
        if ts_mv is None or mm_mv is None:
            return None

        bootstrap_horizon = tp.compute_horizon_ep(
            con, calibration_asof_date, TARGET_SEASON, event, ts_mv, mm_mv, 1, **PARAM_VERSIONS,
        )
        if event not in bootstrap_horizon:
            return None
        ep_mv, un_mv = bootstrap_horizon[event]
        state_version = tp.bootstrap_from_real_squad(con, calibration_asof_date, TARGET_SEASON, event, ep_mv, un_mv, squad)
        holdings = tp._read_holdings(con, state_version)

        fh_horizon = tp.compute_horizon_ep(
            con, calibration_asof_date, TARGET_SEASON, target_gameweek, ts_mv, mm_mv, 1, **PARAM_VERSIONS,
        )
        if target_gameweek not in fh_horizon:
            return None

        so_mod.seed_v1_params(con)
        tp.seed_v1_params(con)
        result = tp.evaluate_free_hit(
            con, calibration_asof_date, TARGET_SEASON, target_gameweek, holdings, fh_horizon,
            lambda_params_version=1, guardrail_params_version=1, threshold_params_version=1,
        )
        return {"gain": round(result["gain"], 2), "recommended": result["recommended"]}
    except Exception as e:  # noqa: BLE001 -- best-effort, see this function's own docstring
        print(f"::warning::print_chip_timing_roadmap: real free-hit-gain check failed for entry_id={entry_id} at GW{target_gameweek} ({e}) -- omitted.")
        return None


def _weekly_avg_squad_swing(con, squad_team_uids: set[str], short_window: int) -> list[dict]:
    """rolling_swing_score() (see its own module docstring: short_window_avg_difficulty minus
    long_window_avg_difficulty) averaged across a squad's clubs, for every gameweek in
    FIRST_HALF_GAMEWEEKS. Shared by the Wildcard/Bench-Boost/Triple-Captain series
    (short_window=SHORT_WINDOW_GAMEWEEKS, a multi-gameweek transition) and the Free Hit series
    (short_window=FREE_HIT_SHORT_WINDOW_GAMEWEEKS=1, a single gameweek's own difficulty) --
    same mechanism, different window, not two independently-invented signals."""
    weekly = []
    for gw in FIRST_HALF_GAMEWEEKS:
        scores = fs.swing_scores_by_team(
            con, TARGET_SEASON, gw, TS_MODEL_VERSION,
            short_window=short_window, long_window=LONG_WINDOW_GAMEWEEKS,
        )
        squad_swings = [
            scores[t].swing_score for t in squad_team_uids
            if t in scores and scores[t].swing_score is not None
        ]
        avg_swing = sum(squad_swings) / len(squad_swings) if squad_swings else None
        weekly.append({
            "gameweek": gw,
            "avg_squad_swing": round(avg_swing, 3) if avg_swing is not None else None,
            "n_teams_with_data": len(squad_swings),
        })
    return weekly


def main() -> None:
    # Both accounts' current-squad read-from gameweek -- the same bootstrap-static-derived
    # app_export.current_event() scheduled_pipeline.yml's own "Determine current gameweek" step
    # resolves for the transfer-planner/app-export steps this roadmap runs alongside, so all
    # three can never independently drift the way the previous hardcoded-and-hand-bumped values
    # did (three days stale after GW1's deadline before anyone bumped it to GW2).
    event = ax.current_event(ax.fetch_bootstrap_static())
    if event is None:
        raise SystemExit("bootstrap-static reports no current gameweek right now")

    con = db.connect()
    team_names = {r[0]: r[1] for r in con.execute("SELECT team_uid, canonical_name FROM dim_team").fetchall()}
    team_by_player = fs.team_uid_by_player(con, TARGET_SEASON)
    n_matches_played = con.execute(
        "SELECT count(*) FROM fact_match WHERE season = ? AND finished = TRUE", [TARGET_SEASON],
    ).fetchone()[0]

    roadmap = {
        "target_season": TARGET_SEASON,
        "first_half_gameweeks": [FIRST_HALF_GAMEWEEKS.start, FIRST_HALF_GAMEWEEKS.stop - 1],
        "n_matches_played_this_season": n_matches_played,
        "reliability_note": (
            "Fixture/form signals are noisier the fewer real matches have been played this "
            "season -- a candidate window based on very few results is a real, computed signal "
            "but rests on a thinner sample than one identified later in the season."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "accounts": [],
    }

    for account in ACCOUNTS:
        print(f"[chip_timing] {account['label']} (entry_id={account['entry_id']})...")
        squad_team_uids = _account_team_uids(con, account["entry_id"], event, team_by_player)
        print(f"  {len(squad_team_uids)} distinct clubs in squad")

        weekly = _weekly_avg_squad_swing(con, squad_team_uids, SHORT_WINDOW_GAMEWEEKS)
        for w in weekly:
            print(f"  GW{w['gameweek']}: avg_squad_swing={w['avg_squad_swing']} ({w['n_teams_with_data']} teams)")

        valid = [w for w in weekly if w["avg_squad_swing"] is not None]
        best_window = min(valid, key=lambda w: w["avg_squad_swing"]) if valid else None
        worst_window = max(valid, key=lambda w: w["avg_squad_swing"]) if valid else None

        if worst_window is not None:
            print(f"  checking real wildcard EP-gain at the candidate window (GW{worst_window['gameweek']})...")
            real_check = _real_wildcard_gain(con, account["entry_id"], event, worst_window["gameweek"])
            worst_window = {**worst_window, "real_wildcard_check": real_check}
            if real_check:
                verdict = "clears" if real_check["recommended"] else "does NOT clear"
                print(f"  GW{worst_window['gameweek']}: real wildcard gain={real_check['gain']:+.2f} EP -- {verdict} the model's own recommendation threshold")

        # Free Hit's own series -- a single toughest gameweek, not the multi-gameweek transition
        # `weekly` above looks for (see FREE_HIT_SHORT_WINDOW_GAMEWEEKS' own comment).
        weekly_fh = _weekly_avg_squad_swing(con, squad_team_uids, FREE_HIT_SHORT_WINDOW_GAMEWEEKS)
        valid_fh = [w for w in weekly_fh if w["avg_squad_swing"] is not None]
        free_hit_window = max(valid_fh, key=lambda w: w["avg_squad_swing"]) if valid_fh else None
        if free_hit_window is not None:
            print(f"  checking real free-hit EP-gain at the candidate window (GW{free_hit_window['gameweek']})...")
            real_fh_check = _real_free_hit_gain(con, account["entry_id"], event, free_hit_window["gameweek"])
            free_hit_window = {**free_hit_window, "real_free_hit_check": real_fh_check}
            if real_fh_check:
                verdict = "clears" if real_fh_check["recommended"] else "does NOT clear"
                print(f"  GW{free_hit_window['gameweek']}: real free-hit gain={real_fh_check['gain']:+.2f} EP -- {verdict} the model's own recommendation threshold")

        roadmap["accounts"].append({
            "entry_id": account["entry_id"],
            "label": account["label"],
            "squad_clubs": sorted(team_names.get(t, t) for t in squad_team_uids),
            "weekly_swing": weekly,
            "best_bench_boost_triple_captain_window": best_window,
            "toughest_window_wildcard_candidate": worst_window,
            "free_hit_candidate_window": free_hit_window,
        })

    con.close()

    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DASHBOARD_DIR / "chip_timing_roadmap.json"
    out_path.write_text(json.dumps(roadmap, indent=2))
    print(f"\n[chip_timing] roadmap written to {out_path}")


if __name__ == "__main__":
    main()
