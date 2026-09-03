"""Roadmap Feature 1: per-gameweek point-projection table + captaincy ranking with
confidence bands -- the table-stakes FPLReview-style deliverable this project's engine has
never surfaced end-to-end, built entirely on top of EXISTING M3/M4/M8 machinery.

build_projections() calls transfer_planner.compute_horizon_ep() (M8's own multi-gameweek-
horizon mechanism: one ep.run() + uncertainty.run() pair per gameweek, reusing the same
gameweek-agnostic ts_model_version/mm_model_version throughout) rather than re-deriving any
point estimate here -- this module is a read + reshape + provenance-tag layer, not a new
model.

Confidence band, a real and disclosed adaptation of this feature's original spec: M6's Monte
Carlo engine only ever simulates one already-CHOSEN squad's 15 players (see monte_carlo.py's
own module docstring on its deliberate query-scoped boundary), not the full ~577-player pool
a projections table needs to cover. The band shown here is M4's own Cornish-Fisher quantile
(uncertainty_outputs.quantile_05/quantile_95, the SAME analytic approximation
uncertainty.explain_player_risk() already surfaces to users elsewhere, with the same
CORNISH_FISHER_DISPLAY_CAVEAT) -- real, computed uncertainty for every player, rather than
a live MC draw only available for whichever 15 happen to be in one specific optimized squad.
"""

import functools
from dataclasses import dataclass
from datetime import date

import duckdb

from . import ingest_workbook as iw
from . import squad_optimizer as so_mod
from . import transfer_planner as tp
from .errors import MissingModelVersionError

POSITION_LABELS_SHORT = {"Goalkeeper": "GKP", "Defender": "DEF", "Midfielder": "MID", "Forward": "FWD"}


@dataclass(frozen=True)
class GWBand:
    gw: int
    ep: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class Provenance:
    model_version: str
    data_asof: str
    calibrated_params_fraction: float | None


@dataclass(frozen=True)
class ProjectionRow:
    player_uid: str
    name: str
    team: str
    pos: str
    now_cost: float
    ep_per_gw: list[GWBand]
    provenance: Provenance


def build_projections(
    con: duckdb.DuckDBPyConnection,
    *,
    calibration_asof_date: date,
    target_season: str,
    gameweeks: list[int],
    ts_model_version: int,
    mm_model_version: int,
    scoring_params_version: int,
    bps_params_version: int,
    tau_params_version: int,
    rho_residual_params_version: int,
    corr_params_version: int,
    calibrated_params_fraction: float | None = None,
) -> list[ProjectionRow]:
    """Builds one ProjectionRow per player who has a real fixture in at least one of
    `gameweeks`. Players restricted to those present in the FIRST requested gameweek's
    candidate pool (a real, disclosed simplification: a player blank in gameweeks[0] but
    with a fixture later in the horizon is not yet surfaced here -- the same "one gameweek's
    candidate pool" scope squad_optimizer.fetch_candidate_pool() itself already has).

    Raises MissingModelVersionError if ANY requested gameweek has no fixtures to compute an
    ep_model_version/uncertainty_model_version for -- never silently drops a requested
    gameweek from the table.
    """
    if not gameweeks:
        raise ValueError("build_projections() needs at least one gameweek")
    gameweeks = sorted(set(gameweeks))
    start_gw = gameweeks[0]
    horizon = gameweeks[-1] - start_gw + 1

    horizon_versions = tp.compute_horizon_ep(
        con, calibration_asof_date, target_season, start_gw, ts_model_version, mm_model_version, horizon,
        scoring_params_version, bps_params_version, tau_params_version,
        rho_residual_params_version, corr_params_version,
    )
    missing = [gw for gw in gameweeks if gw not in horizon_versions]
    if missing:
        raise MissingModelVersionError(
            f"no ep_model_version/uncertainty_model_version could be computed for "
            f"{target_season} gameweek(s) {missing} (no fixtures scheduled?) -- cannot build projections."
        )

    ep_mv0, un_mv0 = horizon_versions[start_gw]
    candidates = so_mod.fetch_candidate_pool(con, ep_mv0, un_mv0, target_season)

    bands_by_player: dict[str, dict[int, GWBand]] = {}
    for gw in gameweeks:
        ep_mv, un_mv = horizon_versions[gw]
        rows = con.execute(
            "SELECT o.player_uid, o.ep_total, u.quantile_05, u.quantile_95 "
            "FROM ep_outputs o JOIN uncertainty_outputs u "
            "ON u.model_version = ? AND u.player_uid = o.player_uid AND u.fixture_match_id = o.fixture_match_id "
            "WHERE o.model_version = ?",
            [un_mv, ep_mv],
        ).fetchall()
        for player_uid, ep_total, q05, q95 in rows:
            bands_by_player.setdefault(player_uid, {})[gw] = GWBand(gw=gw, ep=ep_total, ci_low=q05, ci_high=q95)

    model_version_tag = ",".join(f"gw{gw}:ep_v{ep_mv}/un_v{un_mv}" for gw, (ep_mv, un_mv) in sorted(horizon_versions.items()) if gw in gameweeks)
    provenance = Provenance(
        model_version=model_version_tag, data_asof=calibration_asof_date.isoformat(),
        calibrated_params_fraction=calibrated_params_fraction,
    )

    out = []
    for c in candidates:
        uid = c["player_uid"]
        player_bands = bands_by_player.get(uid, {})
        ep_per_gw = [player_bands[gw] for gw in gameweeks if gw in player_bands]
        if not ep_per_gw:
            continue  # blank across the whole requested horizon
        out.append(ProjectionRow(
            player_uid=uid, name=c["name"], team=c["club"],
            pos=POSITION_LABELS_SHORT.get(c["position"], c["position"]),
            now_cost=c["price"], ep_per_gw=ep_per_gw, provenance=provenance,
        ))
    return out


def _captain_compare(item_a: tuple, item_b: tuple, tie_epsilon: float) -> int:
    """Pairwise comparator: within tie_epsilon of each other's EP, the HIGHER CEILING
    (ci_high) ranks first -- otherwise higher EP wins. Comparator-based (not a single sort
    key) since "tied" is a tolerance relation, not exact equality.

    Captaincy DOUBLES the score, so its payoff is dominated by the upside: a captain haul
    gains ~10+ points relative to the field, a captain blank costs ~5. When two candidates
    have ~equal EP you want the one most likely to explode, i.e. the wider band / higher
    ceiling -- the OPPOSITE of the original spec here, which preferred the safer narrower
    band and was a real contributor to low-variance defenders out-ranking premium attackers
    for captaincy (backtest: model captained the flatter option and lost hauls)."""
    _ra, ba = item_a
    _rb, bb = item_b
    if abs(ba.ep - bb.ep) <= tie_epsilon:
        if ba.ci_high != bb.ci_high:
            return -1 if ba.ci_high > bb.ci_high else 1
    if ba.ep != bb.ep:
        return -1 if ba.ep > bb.ep else 1
    return 0


def build_captain_ranking(rows: list[ProjectionRow], gw: int, *, tie_epsilon: float = 0.15) -> list[dict]:
    """The EP-per-GW table sorted by ep for gw, with the confidence band shown so a
    higher-ceiling near-tie wins captaincy over a flatter one (see _captain_compare()). Vice
    captain (rank 2) carries a vice_captain_reason explaining why it isn't rank 1; every
    other row's reason is None. Returns [] if no player has a fixture at gw."""
    eligible = [(r, band) for r in rows for band in r.ep_per_gw if band.gw == gw]
    if not eligible:
        return []

    ranked = sorted(eligible, key=functools.cmp_to_key(lambda a, b: _captain_compare(a, b, tie_epsilon)))

    out = []
    for i, (r, band) in enumerate(ranked):
        rank = i + 1
        reason = None
        if rank == 2:
            _top_r, top_band = ranked[0]
            if abs(band.ep - top_band.ep) <= tie_epsilon and band.ci_high < top_band.ci_high:
                reason = (
                    f"projected points are close to the #1 pick ({band.ep:.1f} vs {top_band.ep:.1f}) "
                    f"but a lower ceiling ({band.ci_high:.1f} vs {top_band.ci_high:.1f}) drops them to vice captain"
                )
            else:
                reason = "second-highest projected points this gameweek"
        out.append({
            "rank": rank, "player_uid": r.player_uid, "name": r.name, "team": r.team,
            "ep": band.ep, "ci_low": band.ci_low, "ci_high": band.ci_high,
            "vice_captain_reason": reason,
        })
    return out


def resolve_element_ids(con: duckdb.DuckDBPyConnection, target_season: str, element_names: dict[int, int | str]) -> dict[str, int]:
    """player_uid -> FPL bootstrap-static element id, resolved via the SAME normalized-name
    matching every other real-name source in this project already uses
    (ingest_workbook._resolve_player()) -- not a new, separately-invented join. Exists because
    this module's own player_uid is this project's internal identity, meaningless to the PWA on
    its own, which is built entirely around FPL's numeric element id (see app_export.py's own
    module docstring on why those are two different identity spaces) -- the PWA's planner needs
    a real id to join a projection row against the manager's own app_team_<id>.json squad.

    element_names: {element_id: full_name}, e.g. from
    ingest_fpl_entry_picks.fetch_bootstrap_elements() -- injected (not fetched here) so this
    stays testable offline, matching this project's established fetch-isolation convention. A
    name with no resolvable player_uid is simply absent from the result, never guessed. The
    first element id to resolve to a given player_uid wins ties (a genuinely duplicate name
    across two different elements is not expected in a real bootstrap-static payload)."""
    out: dict[str, int] = {}
    for element_id, name in element_names.items():
        player_uid = iw._resolve_player(con, name, target_season)
        if player_uid and player_uid not in out:
            out[player_uid] = element_id
    return out
