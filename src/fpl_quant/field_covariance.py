"""Priority 1 addition: field-covariance term for squad_optimizer's MIQP objective.

EO alone (see ownership.py) is a static, per-player number answering "how popular is this
player," not "how much does this player's own scoring outcome move together with the field's
outcome". A high-EO player whose scoring only weakly correlates with the rest of the
highly-owned pool (their fixture doesn't overlap with the busiest template fixtures)
protects rank less than raw EO alone suggests; a low-EO differential riding the same busy
fixture as much of the highly-owned pool protects (or costs) rank more than raw EO alone
suggests. Cov(player, field) captures that; this is the "not a naive EP - lambda*ownership%
heuristic" Priority 1 explicitly asks for.

Reuses monte_carlo.py's own Z_fixture generative mechanism (deterministic_seed,
z_fixture_variance, sample_z_fixture, compute_lambda_representative) rather than inventing a
second, independently-drawn random process -- the same shared per-fixture tempo factor M6
already uses to correlate teammates' outcomes is what correlates a candidate's outcome with
the field's here too.

Scope, a genuine simplification flagged explicitly rather than silently picked (matching
monte_carlo.py's own established practice of naming its scope decisions): M6's full
per-player simulation (goals/assists/bonus-rank/DefCon/saves, individually Poisson-resampled
every realization) is deliberately NOT reproduced here for the full ~577-candidate pool --
M6's own docstring already flags that re-simulating every fixture participant at that level of
detail is exactly the computational-scope blowup its own query-level restriction ("one
already-CHOSEN squad," not the full candidate pool) exists to avoid. This module runs BEFORE
a squad is chosen (it feeds solve()'s own objective as a precomputed input), so it cannot
lean on M6's "already chosen squad" scoping to bound its own cost the way M6 does -- it has to
cover the full candidate pool. Each candidate's simulated point total is instead a
mean-preserving linear scaling of their own M3 mu by the fixture's shared Z_fixture draw:
points_i(realization) = mu_i * z_fixture(fixture, realization). This keeps E[points_i] = mu_i
exactly (matching M3's own mean) and reproduces the ESSENTIAL correlation mechanism M6 models
(players in the same fixture share a common tempo factor that scales their outputs together)
without redoing per-category Poisson/bonus-rank resampling for every one of ~577 candidates on
every solve.

The synthetic "field portfolio" is an EO-weighted aggregate across the WHOLE candidate pool,
not a simulated distribution of real individual rival squads -- that full version (genuine
rival-squad-distribution simulation) is Priority 10's own, much larger, explicitly-deferred
scope. A candidate who is themselves part of the field (nonzero EO) naturally contributes to
their own Cov(player, field) figure -- standard practice for a player-vs-index covariance
(the same way a stock's CAPM beta is computed against a market index that includes the stock
itself), not a bug to be corrected out.
"""

from datetime import date

import duckdb
import numpy as np

from . import monte_carlo as mc_mod
from . import params as params_mod


def compute_field_covariance(
    con: duckdb.DuckDBPyConnection,
    candidates: list[dict],
    target_season: str,
    target_gameweek: int,
    ep_model_version: int,
    scoring_params_version: int,
    rho_residual_params_version: int,
    eo_by_uid: dict[str, float | None],
    calibration_asof_date: date,
    n_antithetic_pairs: int = 2000,
) -> dict[str, float]:
    """Cov(player_i, field) per candidate, precomputed ONCE per solve() call (a plain float
    per candidate, fed into the MIQP as a linear coefficient -- not a per-SCIP-node
    computation). field(realization) = sum_i (eo_i/100) * points_i(realization): a candidate
    with eo_i=100 (effectively owned by the whole player base) contributes its full simulated
    points to the field portfolio every realization; eo_i=0 contributes nothing. Candidates
    with unknown EO (None) are excluded from the field portfolio itself (an unknown
    popularity can't be assumed either way) but still get their own Cov(player, field)
    computed against the field the KNOWN-EO candidates make up.

    A candidate with no fixture this gameweek (blank GW) legitimately contributes nothing to
    the field and gets Cov=0 against it -- not an error, matching monte_carlo.run()'s own
    "blanks are a legitimate exclusion" stance on the same situation.
    """
    rho_residual, _ = params_mod.resolve_param(con, "correlation_params", "rho_residual", rho_residual_params_version)
    uid_list = [c["player_uid"] for c in candidates]
    lambda_representative = mc_mod.compute_lambda_representative(con, uid_list, ep_model_version, scoring_params_version)
    sigma_z_sq = mc_mod.z_fixture_variance(rho_residual, lambda_representative)

    fixture_rows = con.execute(
        "SELECT match_id FROM fact_match WHERE season = ? AND gameweek = ?", [target_season, target_gameweek],
    ).fetchall()
    fixture_ids = {m for (m,) in fixture_rows}

    fixture_of: dict[str, str] = {}
    if uid_list:
        placeholders = ",".join("?" for _ in uid_list)
        rows = con.execute(
            f"SELECT player_uid, fixture_match_id FROM ep_outputs "
            f"WHERE model_version = ? AND player_uid IN ({placeholders})",
            [ep_model_version, *uid_list],
        ).fetchall()
        fixture_of = {uid: fid for uid, fid in rows if fid in fixture_ids}

    query_id = f"field_cov_gw{target_gameweek}_n{len(uid_list)}"
    seed = mc_mod.deterministic_seed(0, calibration_asof_date, query_id)
    rng = np.random.default_rng(seed)
    n_real = 2 * n_antithetic_pairs

    mu_by_uid = {c["player_uid"]: c["mu"] for c in candidates}
    eo_weight_by_uid = {uid: (eo_by_uid.get(uid) or 0.0) / 100.0 for uid in uid_list}

    # sorted(): the same order-determinism discipline as squad_optimizer.solve()'s own
    # candidate sort (see its docstring) -- fixture_ids is a Python set, whose iteration
    # order is hash-randomized per process; sorting makes the RNG draw order (hence the
    # resulting field-portfolio realizations) reproducible across separate runs, not just
    # within one process.
    points_by_uid: dict[str, np.ndarray] = {}
    for match_id in sorted(fixture_ids):
        members = [uid for uid in uid_list if fixture_of.get(uid) == match_id]
        if not members:
            continue
        u = rng.random(n_antithetic_pairs)
        u_full = np.concatenate([u, 1.0 - u])
        z = mc_mod.sample_z_fixture(sigma_z_sq, u_full)
        for uid in members:
            points_by_uid[uid] = mu_by_uid[uid] * z

    field_portfolio = np.zeros(n_real)
    for uid in sorted(points_by_uid):
        field_portfolio += eo_weight_by_uid.get(uid, 0.0) * points_by_uid[uid]

    field_var = float(np.var(field_portfolio, ddof=0))
    cov_by_uid: dict[str, float] = {}
    for uid in uid_list:
        points = points_by_uid.get(uid)
        if points is None or field_var == 0.0:
            cov_by_uid[uid] = 0.0
            continue
        cov_by_uid[uid] = float(np.cov(points, field_portfolio, ddof=0)[0, 1])
    return cov_by_uid
