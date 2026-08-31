"""Forward season simulation for a real, currently-held FPL squad -- used to time the Wildcard.

`backtest.run_season_simulation()` walks forward too, but it (a) bootstraps a fresh M5 squad,
not the squad a manager actually holds, and (b) scores each gameweek on realised
`event_points`, so it only works over gameweeks already played. This module fills both gaps for
a live, forward-looking question -- "given the squad I hold right now, when should I Wildcard?":

  * seeds from `transfer_planner.bootstrap_from_real_squad()` (the real 15),
  * walks `start_gameweek..end_gameweek`, making one real `transfer_planner.run()` /
    `_decide_gameweek_action()` decision per gameweek and applying it, exactly as
    `run_season_simulation()` does,
  * scores each gameweek on the *projected* XI EP (`ep_outputs` for that gameweek), with an
    80% band from `uncertainty_outputs.var_total`, since there is no realised outcome yet,
  * records, every gameweek, `evaluate_wildcard()`'s own projected gain against the *evolved*
    squad -- not a squad frozen at today, which is what `print_chip_timing_roadmap.py` compares
    against. The gameweek with the largest gain that also clears the recommendation threshold
    is the Wildcard timing recommendation.

Two modes: `hold_wildcard=True` never plays the Wildcard (a clean gain trajectory across the
window); the default lets `_decide_gameweek_action()` play it whenever the model judges best
(the model's own timing answer). `force_wildcard_at` pins it to one gameweek for an A/B.

Everything here is asof-safe via `backtest.asof_scope()` and reuses production planner code --
no new modelling. Known early-season limitation: with only ~2 gameweeks played, every future
gameweek's EP model is fit on the same data, so GW3 vs GW15 projections differ by fixtures and
minutes, not by form. Re-run weekly as the season fills in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import duckdb

from . import backtest as bt
from . import expected_points, minutes_model, params as params_mod, team_strength, transfer_planner, uncertainty

# 80% projected band: +-1.2816 sigma. A disclosed normal approximation on the XI's summed
# var_total -- not a full Monte Carlo (which every gameweek of every sweep arm would make this
# far too slow), and it ignores cross-player covariance, so the band is a floor on the real
# spread, stated not hidden.
_Z80 = 1.2816


@dataclass
class GameweekResult:
    gameweek: int
    projected_points: float
    band_low: float
    band_high: float
    action: str                       # "transfer" / "hold" / "wildcard" / "free_hit" / "bench_boost" / "triple_captain"
    action_detail: str
    wildcard_gain: float | None       # evaluate_wildcard()'s projected gain vs the evolved squad + best transfer
    wildcard_recommended: bool
    current_squad_horizon_value: float | None
    chips_used: list[str] = field(default_factory=list)
    # evaluate_free_hit()'s own projected gain / recommendation for this gameweek -- the planner
    # evaluates Free Hit every gameweek anyway (it is one of the four chips run() scores), so
    # surfacing it here is a read of chip_evaluations, not an extra solve. Lets a caller scan
    # the whole walked window for a Free-Hit-worthy blank/bad-fixture week (chip_timing_analysis
    # Step 3) off the hold-Wildcard arm alone.
    free_hit_gain: float | None = None
    free_hit_recommended: bool = False

    def to_dict(self) -> dict:
        return {
            "gameweek": self.gameweek,
            "projected_points": round(self.projected_points, 2),
            "band_low": round(self.band_low, 2),
            "band_high": round(self.band_high, 2),
            "action": self.action,
            "action_detail": self.action_detail,
            "wildcard_gain": None if self.wildcard_gain is None else round(self.wildcard_gain, 2),
            "wildcard_recommended": self.wildcard_recommended,
            "current_squad_horizon_value": (
                None if self.current_squad_horizon_value is None else round(self.current_squad_horizon_value, 2)
            ),
            "chips_used": self.chips_used,
            "free_hit_gain": None if self.free_hit_gain is None else round(self.free_hit_gain, 2),
            "free_hit_recommended": self.free_hit_recommended,
        }


def _resolve_versions(con: duckdb.DuckDBPyConnection, active: dict) -> dict:
    """The full keyword set backtest.run_season_simulation() takes, with the recalibratable
    families resolved from `active` (backtest.active_recalibratable_versions()) and everything
    else pinned to the seeded v1 -- exactly what run_transfer_planner_for_real_squad.py uses."""
    return {
        "xi_params_version": active.get("xi_params_version", 1),
        "rho_params_version": active.get("rho_params_version", 1),
        "decay_params_version": 1,
        "adjustment_params_version": active.get("adjustment_params_version", 1),
        "shrinkage_params_version": active.get("shrinkage_params_version", 1),
        "fact_multiplier_params_version": active.get("fact_multiplier_params_version", 1),
        "scoring_params_version": 1,
        "bps_params_version": 1,
        "tau_params_version": 1,
        "rho_residual_params_version": active.get("rho_residual_params_version", 1),
        "corr_params_version": 1,
        "lambda_params_version": active.get("lambda_params_version", 1),
        "guardrail_params_version": 1,
        "horizon_params_version": 1,
        "transfer_cost_params_version": 1,
        "wildcard_threshold_params_version": 1,
        "free_hit_threshold_params_version": 1,
        "kappa_tc_params_version": active.get("kappa_tc_params_version", 1),
    }


def _projected_xi_points(
    con: duckdb.DuckDBPyConnection, season: str, gameweek: int, ep_mv: int, un_mv: int,
    xi_uids: frozenset[str], captain_uid: str | None, captain_multiplier: int,
) -> tuple[float, float]:
    """(mean, std) of the XI's projected points for this gameweek from `ep_outputs` /
    `uncertainty_outputs`. Captain's mean counts `captain_multiplier` times; its variance
    counts `captain_multiplier**2` times."""
    ep_rows = dict(con.execute(
        """
        SELECT o.player_uid, o.ep_total
        FROM ep_outputs o JOIN fact_match m ON m.match_id = o.fixture_match_id
        WHERE o.model_version = ? AND m.season = ? AND m.gameweek = ? AND m.competition = ?
        """,
        [ep_mv, season, gameweek, bt.PL],
    ).fetchall())
    var_rows = dict(con.execute(
        """
        SELECT u.player_uid, u.var_total
        FROM uncertainty_outputs u JOIN fact_match m ON m.match_id = u.fixture_match_id
        WHERE u.model_version = ? AND m.season = ? AND m.gameweek = ? AND m.competition = ?
        """,
        [un_mv, season, gameweek, bt.PL],
    ).fetchall())

    mean = 0.0
    var = 0.0
    for uid in xi_uids:
        mu = ep_rows.get(uid, 0.0) or 0.0
        v = var_rows.get(uid, 0.0) or 0.0
        if uid == captain_uid:
            mean += mu * captain_multiplier
            var += v * (captain_multiplier ** 2)
        else:
            mean += mu
            var += v
    return mean, var ** 0.5


def _read_wildcard_eval(con: duckdb.DuckDBPyConnection, plan_run_id: int) -> dict:
    row = con.execute(
        "SELECT recommended, score_or_gain, detail FROM chip_evaluations WHERE run_id = ? AND chip_type = 'wildcard'",
        [plan_run_id],
    ).fetchone()
    if not row:
        return {}
    recommended, score, detail = row
    d = json.loads(detail or "{}")
    d.setdefault("gain", score)
    d.setdefault("recommended", bool(recommended))
    return d


def _read_free_hit_eval(con: duckdb.DuckDBPyConnection, plan_run_id: int) -> dict:
    row = con.execute(
        "SELECT recommended, score_or_gain, detail FROM chip_evaluations WHERE run_id = ? AND chip_type = 'free_hit'",
        [plan_run_id],
    ).fetchone()
    if not row:
        return {}
    recommended, score, detail = row
    d = json.loads(detail or "{}")
    d.setdefault("gain", score)
    d.setdefault("recommended", bool(recommended))
    return d


def _best_transfer_rank_if_positive(con: duckdb.DuckDBPyConnection, plan_run_id: int) -> int | None:
    row = con.execute(
        "SELECT rank FROM transfer_recommendations WHERE run_id = ? AND net_value > 0 ORDER BY rank LIMIT 1",
        [plan_run_id],
    ).fetchone()
    return int(row[0]) if row else None


@dataclass
class ForwardSimResult:
    entry_label: str
    season: str
    start_gameweek: int
    end_gameweek: int
    mode: str
    rows: list[GameweekResult]
    total_projected_points: float
    total_band_low: float
    total_band_high: float
    # Populated (once) at the gameweek the Wildcard was actually played -- forced or
    # model-chosen -- with the live-DB handles a caller needs to run a Bench-Boost-combo or a
    # squad-robustness check against that fresh Wildcard squad WITHOUT re-walking the season
    # (chip_timing_analysis Steps 2 and 4). Not serialised by to_dict(): fresh_run_id /
    # ts_model_version / mm_model_version are only meaningful in the same DB the walk ran
    # against. None whenever the Wildcard was never played in-window.
    wildcard_context: dict | None = None

    @property
    def wildcard_recommendation(self) -> dict | None:
        """The gameweek with the largest wildcard gain that also clears the model's threshold.
        None if no gameweek in the window recommends it."""
        cands = [
            (r.gameweek, r.wildcard_gain) for r in self.rows
            if r.wildcard_recommended and r.wildcard_gain is not None
        ]
        if not cands:
            return None
        gameweek, gain = max(cands, key=lambda t: t[1])
        return {"gameweek": gameweek, "projected_gain": round(gain, 2)}

    def to_dict(self) -> dict:
        return {
            "entry_label": self.entry_label,
            "season": self.season,
            "start_gameweek": self.start_gameweek,
            "end_gameweek": self.end_gameweek,
            "mode": self.mode,
            "total_projected_points": round(self.total_projected_points, 1),
            "total_band": [round(self.total_band_low, 1), round(self.total_band_high, 1)],
            "wildcard_recommendation": self.wildcard_recommendation,
            "gameweeks": [r.to_dict() for r in self.rows],
        }


def run_forward_season_sim(
    con: duckdb.DuckDBPyConnection,
    *,
    entry_label: str,
    target_season: str,
    start_gameweek: int,
    end_gameweek: int,
    bootstrap_squad: list[dict],
    active_versions: dict,
    hold_wildcard: bool = False,
    force_wildcard_at: int | None = None,
    real_chips_used_set1: list[str] | None = None,
    real_chips_used_set2: list[str] | None = None,
) -> ForwardSimResult:
    """Walk `start_gameweek..end_gameweek` from the real `bootstrap_squad`, one real planner
    decision per gameweek, scoring on projected EP. See module docstring for the two modes.

    real_chips_used_set1 / real_chips_used_set2: the chips the real manager has ALREADY spent
    (from the FPL API's own entry history). transfer_planner.bootstrap_from_real_squad() always
    seeds an empty chip state (its docstring is explicit that it inherits a fresh-account
    assumption); passing the real used-chip lists here writes them onto the bootstrapped
    manager_state_versions row so the walk's own _decide_gameweek_action() will not re-play a
    chip that is gone. None (the default) keeps the fresh-account behavior unchanged -- correct
    for the two currently-tracked entries, which have used zero chips as of GW2 2026-27."""
    versions = _resolve_versions(con, active_versions)
    horizon_gameweeks = int(params_mod.resolve_param(
        con, "planning_horizon_params", "horizon_gameweeks", versions["horizon_params_version"])[0])

    mode = "hold_wildcard" if hold_wildcard else (f"force_wildcard_gw{force_wildcard_at}" if force_wildcard_at else "model_choice")

    if not bt.has_fittable_history(con, target_season, start_gameweek):
        raise ValueError(f"{target_season} GW{start_gameweek} has insufficient prior history to simulate from")

    # ---- bootstrap the real squad ----
    with bt.asof_scope(con, target_season, start_gameweek, schedule_horizon_gameweeks=horizon_gameweeks) as deadline:
        asof = deadline.date()
        ts0 = team_strength.calibrate(con, asof, versions["xi_params_version"], versions["rho_params_version"],
                                      target_season=target_season, fit_seasons=bt.fit_seasons_for(target_season))
        mm0 = minutes_model.run(con, asof, target_season, versions["decay_params_version"],
                                versions["adjustment_params_version"], versions["shrinkage_params_version"],
                                versions["fact_multiplier_params_version"])
        ep0 = expected_points.run(con, asof, target_season, start_gameweek, ts0, mm0,
                                  versions["scoring_params_version"], versions["bps_params_version"],
                                  versions["tau_params_version"])
        un0 = uncertainty.run(con, asof, ep0, mm0, ts0, versions["scoring_params_version"],
                              versions["bps_params_version"], versions["tau_params_version"],
                              versions["rho_residual_params_version"], versions["corr_params_version"])
        state_version = transfer_planner.bootstrap_from_real_squad(
            con, asof, target_season, start_gameweek, ep_model_version=ep0, uncertainty_model_version=un0,
            squad=bootstrap_squad,
        )
        if real_chips_used_set1 is not None or real_chips_used_set2 is not None:
            con.execute(
                "UPDATE manager_state_versions SET chips_used_set1 = ?, chips_used_set2 = ? WHERE state_version = ?",
                [
                    json.dumps(sorted(set(real_chips_used_set1 or []))),
                    json.dumps(sorted(set(real_chips_used_set2 or []))),
                    state_version,
                ],
            )

    rows: list[GameweekResult] = []
    wildcard_context: dict | None = None
    for gw in range(start_gameweek, end_gameweek + 1):
        if bt.has_double_gameweek(con, target_season, gw):
            continue  # v1 scope: DGW planning skipped (same boundary as backtest.run_season_simulation)

        with bt.asof_scope(con, target_season, gw, schedule_horizon_gameweeks=horizon_gameweeks) as deadline:
            asof = deadline.date()
            ts_mv = team_strength.calibrate(con, asof, versions["xi_params_version"], versions["rho_params_version"],
                                            target_season=target_season, fit_seasons=bt.fit_seasons_for(target_season))
            mm_mv = minutes_model.run(con, asof, target_season, versions["decay_params_version"],
                                      versions["adjustment_params_version"], versions["shrinkage_params_version"],
                                      versions["fact_multiplier_params_version"])
            plan_run_id = transfer_planner.run(
                con, asof, target_season, gw, state_version, ts_mv, mm_mv,
                versions["horizon_params_version"], versions["scoring_params_version"], versions["bps_params_version"],
                versions["tau_params_version"], versions["rho_residual_params_version"], versions["corr_params_version"],
                versions["transfer_cost_params_version"], versions["lambda_params_version"],
                versions["guardrail_params_version"], versions["wildcard_threshold_params_version"],
                versions["free_hit_threshold_params_version"], versions["kappa_tc_params_version"],
            )

            state_row = con.execute(
                "SELECT chips_used_set1, chips_used_set2 FROM manager_state_versions WHERE state_version = ?",
                [state_version],
            ).fetchone()
            chips_set1 = set(json.loads(state_row[0])) if state_row else set()
            chips_set2 = set(json.loads(state_row[1])) if state_row else set()

            wc = _read_wildcard_eval(con, plan_run_id)
            wc_gain = wc.get("gain")
            wc_reco = bool(wc.get("recommended", False))
            wc_traj = wc.get("current_squad_value_per_gw") or {}
            cur_horizon_val = sum(wc_traj.values()) if wc_traj else None

            fh = _read_free_hit_eval(con, plan_run_id)
            fh_gain = fh.get("gain")
            fh_reco = bool(fh.get("recommended", False))

            # ---- decide the action ----
            forced = force_wildcard_at == gw and "wildcard" not in (chips_set1 | chips_set2)
            accept_rank: int | None
            accept_chip: str | None
            if forced:
                accept_rank, accept_chip = None, "wildcard"
            else:
                accept_rank, accept_chip = bt._decide_gameweek_action(
                    con, plan_run_id, chips_set1, chips_set2, gw, accept_transfer_if_net_value_above=0.0,
                )
                if hold_wildcard and accept_chip == "wildcard":
                    accept_chip = None
                    accept_rank = _best_transfer_rank_if_positive(con, plan_run_id)

            free_hit_squad = None
            if accept_chip == "free_hit":
                free_hit_squad = transfer_planner.read_fresh_chip_squad(con, plan_run_id, "free_hit")

            holdings_before_wc: list[dict] = (
                transfer_planner._read_holdings(con, state_version) if accept_chip == "wildcard" else []
            )

            state_version = transfer_planner.apply_recommendation(
                con, plan_run_id, accept_transfer_rank=accept_rank, accept_chip=accept_chip,
            )

            if accept_chip == "wildcard" and wildcard_context is None:
                wildcard_context = {
                    "gameweek": gw,
                    "asof_date": asof,
                    "fresh_run_id": wc.get("fresh_run_id"),
                    "wildcard_result": wc,
                    "holdings_before_uids": sorted(h["player_uid"] for h in holdings_before_wc),
                    "xi_before_uids": sorted(h["player_uid"] for h in holdings_before_wc if h["in_xi"]),
                    "ts_model_version": ts_mv,
                    "mm_model_version": mm_mv,
                    "versions": dict(versions),
                }

            # ---- score this gameweek on projected EP ----
            horizon_versions = transfer_planner.compute_horizon_ep(
                con, asof, target_season, gw, ts_mv, mm_mv, 1,
                versions["scoring_params_version"], versions["bps_params_version"], versions["tau_params_version"],
                versions["rho_residual_params_version"], versions["corr_params_version"],
            )
            ep_mv_gw, un_mv_gw = horizon_versions.get(gw, (None, None))
            holdings = transfer_planner._read_holdings(con, state_version)
            if accept_chip == "free_hit" and free_hit_squad is not None:
                xi = frozenset(h["player_uid"] for h in free_hit_squad if h["in_xi"])
                cap = next((h["player_uid"] for h in free_hit_squad if h["is_captain"]), None)
                mult = 2
            elif accept_chip == "bench_boost":
                xi = frozenset(h["player_uid"] for h in holdings)
                cap = next((h["player_uid"] for h in holdings if h["is_captain"]), None)
                mult = 2
            else:
                xi = frozenset(h["player_uid"] for h in holdings if h["in_xi"])
                cap = next((h["player_uid"] for h in holdings if h["is_captain"]), None)
                mult = 3 if accept_chip == "triple_captain" else 2

            if ep_mv_gw is None or un_mv_gw is None:
                mean, std = 0.0, 0.0
            else:
                mean, std = _projected_xi_points(con, target_season, gw, ep_mv_gw, un_mv_gw, xi, cap, mult)

        action = accept_chip or ("transfer" if accept_rank is not None else "hold")
        detail = ""
        if accept_chip == "wildcard":
            detail = "forced" if forced else "model chose wildcard"
        elif accept_rank is not None:
            tr = con.execute(
                "SELECT player_out, player_in, net_value FROM transfer_recommendations WHERE run_id = ? AND rank = ?",
                [plan_run_id, accept_rank],
            ).fetchone()
            if tr:
                detail = f"{tr[0]} -> {tr[1]} (net {tr[2]:+.2f})"

        rows.append(GameweekResult(
            gameweek=gw, projected_points=mean, band_low=mean - _Z80 * std, band_high=mean + _Z80 * std,
            action=action, action_detail=detail,
            wildcard_gain=wc_gain, wildcard_recommended=wc_reco,
            current_squad_horizon_value=cur_horizon_val,
            chips_used=sorted(chips_set1 | chips_set2 | ({accept_chip} if accept_chip else set())),
            free_hit_gain=fh_gain, free_hit_recommended=fh_reco,
        ))

    total = sum(r.projected_points for r in rows)
    lo = sum(r.band_low for r in rows)
    hi = sum(r.band_high for r in rows)
    return ForwardSimResult(
        entry_label=entry_label, season=target_season, start_gameweek=start_gameweek, end_gameweek=end_gameweek,
        mode=mode, rows=rows, total_projected_points=total, total_band_low=lo, total_band_high=hi,
        wildcard_context=wildcard_context,
    )
