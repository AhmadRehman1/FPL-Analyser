"""M6: Monte Carlo Simulation Engine.

Supersedes (not merely recalibrates) M4's rho_residual=0.15 placeholder with a real
generative mechanism: a shared match-intensity latent factor Z_fixture, drawn once per
simulated fixture, scaling every involved player's goal/assist/BPS-relevant rates together.
Z_fixture ~ Gamma(shape=1/sigma_z^2, scale=sigma_z^2) (mean 1, variance sigma_z^2) -- the
standard Gamma-Poisson mixture construction, chosen because it gives a closed-form
correlation calibration (see z_fixture_variance()) and keeps every downstream rate
nonnegative by construction.

Scope, a genuine judgment call flagged per spec's own instruction to state simplifications
rather than silently pick one: "the candidate pool relevant to the specific query (M5's
squad candidates), not the full 577-player league" is read here as *M5's actual chosen
squad* for a specific squad_optimizer_runs row (15 players), not M5's ~577-player input
pool -- the input pool IS the "full 577-player league" the spec's own Research section
just finished naming as the computational-scope concern to avoid, so reading "candidate
pool" as that same 577 would contradict the sentence's own contrast. The chosen squad is
also the only pool the spec's own Outputs bullets actually need ("M5's chosen squad",
M8's chip-value estimation over that squad+bench). A "query" is therefore one
squad_optimizer_runs.run_id.

Within that scope, real fixtures still involve non-squad players (an FPL match has ~22
first-team-relevant participants; Plackett-Luce bonus ranking needs the whole competing
field to be realistic, not just the squad's own players in that match). Re-simulating every
non-squad participant's own minutes state each realization would reintroduce the exact
computational-scope blowup the query-level restriction exists to avoid, for a class of
players whose own distributional output nothing downstream consumes. Documented
simplification: non-squad fixture participants use their mean-based M3 strength
(exp(expected_bps/tau)*p_played, identical to what M3/M4 already compute) scaled by that
fixture's own per-realization Z_fixture -- they still react to the shared tempo factor, they
just don't get their own re-drawn minutes state. Squad players get the full per-realization
treatment: minutes state, goals, assists, clean sheet, goals conceded, DefCon, saves, and
bonus rank are all genuinely resampled every realization.

DefCon and saves are intentionally NOT scaled by Z_fixture -- the spec's generative-mechanism
paragraph names goals, assists, and "bonus-relevant BPS components" explicitly; defensive
actions and saves are a different mechanism (a busier match doesn't obviously inflate a
single defender's CBI count the way it inflates goal-scoring), and inventing a second
untested scaling channel beyond what spec actually states would be a bigger simplification
than leaving them at their M3 rate, not a smaller one.

Clean sheet and goals-conceded are NOT independently redrawn from lambda_against -- they are
read directly off the same joint (home_goals, away_goals) draw already sampled for the
fixture's Dixon-Coles bivariate Poisson. This is a strictly more correct generative link than
an independent Poisson redraw would be: it makes teammates' clean-sheet outcomes the *same*
underlying draw (not just correlated by construction) and opponents' clean-sheet/goals-
conceded outcomes exact complements, with no extra machinery required.

Review B4 -- per-player GOALS are, likewise, conditioned on that same drawn scoreline, not
independently drawn. Earlier versions of this module sampled each squad player's goal count
as an independent Poisson(Z_fixture * lambda_i), which is coherent with the team-level
clean-sheet/goals-conceded read above but NOT with the team-level goal TOTAL: a simulated
squad's summed goals had no guaranteed relationship to that fixture's own drawn home_goals/
away_goals (two Arsenal squad players could each independently draw 2 goals in a
simulated realization where the shared scoreline draw said Arsenal scored 1). Fixed via
_multinomial_allocate_team_goals(): each side's drawn goal total is allocated across every
fixture participant on that side (squad and non-squad, weighted by each player's own
per-realization goal rate -- non-squad participants use their constant M3 ep_goals, matching
their existing non-resimulated scope above) using the standard multinomial-via-sequential-
binomial decomposition, so a squad's summed simulated goals is now IDENTICAL to the team's
own drawn scoreline every realization, not merely correlated with it via the shared Z_fixture
factor alone. Assists are deliberately left as independent Poisson draws -- they are not
constrained by the match scoreline the way goals are (an assist doesn't require the assisting
player's own team to have scored a DIFFERENT goal than the one already accounted for; the
scoreline total already IS the sum of goals, but no equivalent "assist total" exists to
condition on).
"""

import hashlib
from datetime import date

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import binom, gamma, poisson

from . import expected_points as ep
from . import params as params_mod
from . import team_strength as ts_mod

POSITIONS = ["Goalkeeper", "Defender", "Midfielder", "Forward"]
MAX_GOALS = 10       # truncation for the bivariate Poisson score grid
MAX_COUNT = 12        # truncation for per-player Poisson count draws (goals/assists/defcon/saves)


# ============================================================
# deterministic seeding (spec: seed = hash(model_version, calibration_asof_date, query_id))
# ============================================================

def deterministic_seed(model_version: int, calibration_asof_date: date, query_id: str) -> int:
    """Spec pins the *inputs* to the hash, not a specific hash function -- sha256 is used here
    for a stable, platform-independent digest (Python's builtin hash() is randomized per
    process via PYTHONHASHSEED, which would silently break the reproducibility the spec
    requires). Truncated to fit numpy's default_rng seed range."""
    payload = f"{model_version}|{calibration_asof_date.isoformat()}|{query_id}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:15], 16)  # < 2**60 -- np.random.default_rng accepts arbitrarily large ints, this is just a stable truncation


# ============================================================
# Z_fixture calibration: Gamma-Poisson mixture, closed-form variance solve
# ============================================================

def z_fixture_variance(rho_residual: float, lambda_representative: float) -> float:
    """Solves for sigma_z^2 such that two symmetric Poisson-Gamma-mixed variables sharing
    Z_fixture ~ Gamma(mean=1, var=sigma_z^2) have Corr(X_i, X_j) = rho_residual at
    lambda_i = lambda_j = lambda_representative.

    Derivation: X_i | Z ~ Poisson(Z * lambda_i), Z independent of the Poisson draw given Z.
    Var(X_i) = E[Z]*lambda_i + Var(Z)*lambda_i^2 = lambda_i + sigma_z^2*lambda_i^2.
    Cov(X_i, X_j) = Cov(Z*lambda_i, Z*lambda_j) = sigma_z^2*lambda_i*lambda_j (conditional
    independence given Z kills any other term). At lambda_i=lambda_j=lambda:
        rho = sigma_z^2*lambda^2 / (lambda + sigma_z^2*lambda^2)
    Solving for sigma_z^2:
        sigma_z^2 = rho / (lambda * (1 - rho))
    """
    if lambda_representative <= 0 or rho_residual <= 0:
        return 0.0
    if rho_residual >= 1:
        raise ValueError(f"rho_residual must be < 1, got {rho_residual}")
    return rho_residual / (lambda_representative * (1 - rho_residual))


def compute_lambda_representative(
    con: duckdb.DuckDBPyConnection, squad_uids: list[str], ep_model_version: int, scoring_params_version: int,
) -> float:
    """Mean expected attacking-event COUNT (goals + assists, not points) per squad
    player-fixture -- a real, data-derived "typical" lambda for the calibration above, not an
    invented literal. Units must be counts, not points, since Z_fixture scales a Poisson rate."""
    total, n = 0.0, 0
    for player_uid in squad_uids:
        position_row = con.execute("SELECT position FROM dim_player WHERE player_uid = ?", [player_uid]).fetchone()
        if not position_row:
            continue
        position = position_row[0]
        goal_pts = ep._sm(con, "goal_points", scoring_params_version, position)
        assist_pts = ep._sm(con, "assist_points", scoring_params_version)
        rows = con.execute(
            "SELECT ep_goals, ep_assists FROM ep_outputs WHERE model_version = ? AND player_uid = ?",
            [ep_model_version, player_uid],
        ).fetchall()
        for ep_goals, ep_assists in rows:
            e_goals = ep_goals / goal_pts if goal_pts else 0.0
            e_assists = ep_assists / assist_pts if assist_pts else 0.0
            total += e_goals + e_assists
            n += 1
    return total / n if n else 0.1  # small positive fallback -- avoids a div-by-zero degenerate case


# ============================================================
# vectorized samplers -- inverse-transform, all consuming a uniform array u of shape (n_real,)
# ============================================================

def sample_z_fixture(sigma_z_sq: float, u: np.ndarray) -> np.ndarray:
    if sigma_z_sq <= 0:
        return np.ones_like(u)
    shape = 1.0 / sigma_z_sq
    return gamma.ppf(u, a=shape, scale=sigma_z_sq)


def bivariate_poisson_grid(lam_home: float, lam_away: float, rho: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """Joint pmf grid over (home_goals, away_goals) in [0,max_goals]^2, with Dixon-Coles
    tau(x,y;rho) applied to the four low-score cells and renormalized -- an exact discrete
    joint distribution to sample from, not an independence approximation."""
    x = np.arange(max_goals + 1)
    pmf_home = poisson.pmf(x, lam_home)
    pmf_away = poisson.pmf(x, lam_away)
    grid = np.outer(pmf_home, pmf_away)
    for hx, ay in ((0, 0), (0, 1), (1, 0), (1, 1)):
        grid[hx, ay] *= ts_mod.tau(hx, ay, lam_home, lam_away, rho)
    grid = np.clip(grid, 0.0, None)
    total = grid.sum()
    return grid / total if total > 0 else grid


def sample_from_grid(grid: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = grid.flatten()
    cum = np.cumsum(flat)
    cum[-1] = 1.0  # guard against float round-off leaving cum[-1] fractionally under 1.0
    idx = np.searchsorted(cum, u, side="right")
    idx = np.clip(idx, 0, len(flat) - 1)
    n_cols = grid.shape[1]
    return idx // n_cols, idx % n_cols


def sample_poisson_vec(lam: np.ndarray, u: np.ndarray, max_k: int = MAX_COUNT) -> np.ndarray:
    """Vectorized inverse-CDF Poisson draw, one (possibly distinct) lambda per element."""
    lam = np.asarray(lam, dtype=float)
    cum = np.zeros_like(u)
    result = np.full(u.shape, max_k, dtype=int)
    assigned = np.zeros(u.shape, dtype=bool)
    for k in range(max_k + 1):
        cum = cum + poisson.pmf(k, lam)
        newly = (~assigned) & (u <= cum)
        result[newly] = k
        assigned |= newly
    return result


def sample_binomial_vec(n: np.ndarray, p: np.ndarray, u: np.ndarray, max_k: int) -> np.ndarray:
    """Vectorized inverse-CDF Binomial(n, p) draw, one (possibly distinct) n/p pair per
    element -- same technique as sample_poisson_vec() above, so a caller passing an
    antithetic-paired u (this module's own variance-reduction convention, see
    simulate_fixture()'s _u_pair()) gets that same benefit for binomial draws too. n is
    per-element (not a single shared value) because _multinomial_allocate_team_goals() below
    needs a different remaining-count at every realization."""
    n = np.asarray(n, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    cum = np.zeros_like(u)
    result = np.zeros(u.shape, dtype=np.int64)
    assigned = np.zeros(u.shape, dtype=bool)
    for k in range(max_k + 1):
        cum = cum + binom.pmf(k, n, p)
        newly = (~assigned) & (u <= cum)
        result[newly] = k
        assigned |= newly
    # Float round-off can leave cum fractionally under 1.0 at k=n for a handful of elements;
    # never let the draw exceed the real n for that element (binomial support is [0, n]).
    return np.minimum(result, n.astype(np.int64))


def _multinomial_allocate_team_goals(team_goals: np.ndarray, lambdas_by_player: dict[str, np.ndarray], u_pair_fn) -> dict[str, np.ndarray]:
    """Allocates each realization's already-drawn team goal total (from the Dixon-Coles
    bivariate grid) across every fixture participant on that side, weighted by each player's
    own (per-realization) goal-rate lambda -- so a squad's summed simulated goals is IDENTICAL
    to the team's own drawn scoreline that realization, not merely correlated with it via the
    shared Z_fixture factor (see this module's docstring for why the marginal-only version was
    a real, disclosed coherence gap). Vectorized across all realizations via the standard
    multinomial-via-sequential-binomial decomposition: numpy's own Generator.multinomial only
    accepts one shared probability vector for a whole batch, but per-realization lambdas here
    genuinely differ realization to realization (Z_fixture and each squad player's own drawn
    minutes state both vary per realization) -- sequential conditional binomial draws handle
    that correctly since sample_binomial_vec() takes a full per-element n/p array.

    lambdas_by_player: {player_uid: array} for EVERY fixture participant on this side (squad
    and non-squad) -- a non-squad player's constant per-fixture rate is a legitimate
    contribution to the allocation even though this function never returns their own draw
    (see simulate_fixture()'s own module-docstring scope note on why non-squad participants
    aren't otherwise resimulated). The last player in sorted order absorbs whatever remains
    once every other player's share is drawn -- exact conservation by construction, including
    the disclosed edge case where every remaining player's lambda is 0 (own goals, e.g. --
    explicitly out of this project's modeled scope per README) and the remainder is
    attributed to that one arbitrary-but-deterministic last player rather than lost.
    """
    uids = sorted(lambdas_by_player)
    if not uids:
        return {}

    remaining_n = np.asarray(team_goals, dtype=np.int64).copy()
    remaining_lambda = np.sum([np.asarray(lambdas_by_player[u], dtype=float) for u in uids], axis=0)
    max_k = int(remaining_n.max()) if remaining_n.size else 0

    out: dict[str, np.ndarray] = {}
    for i, uid in enumerate(uids):
        if i == len(uids) - 1:
            out[uid] = remaining_n.copy()
            break
        lam_i = np.asarray(lambdas_by_player[uid], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            p_i = np.where(remaining_lambda > 0, lam_i / remaining_lambda, 0.0)
        draw = sample_binomial_vec(remaining_n, p_i, u_pair_fn(), max_k)
        out[uid] = draw
        remaining_n = remaining_n - draw
        remaining_lambda = remaining_lambda - lam_i
    return out


def sample_minutes_state_vec(p0: float, p1: float, p2: float, u: np.ndarray) -> np.ndarray:
    return np.where(u < p0, "0", np.where(u < p0 + p1, "1_59", "60plus"))


def sample_plackett_luce_ranks_vec(
    strengths: dict[str, np.ndarray], u_rank1: np.ndarray, u_rank2: np.ndarray, u_rank3: np.ndarray,
) -> dict[str, np.ndarray]:
    """Sequential without-replacement categorical sampling, vectorized across realizations.
    strengths: {player_uid: array of shape (n_real,)}. Returns {player_uid: int array of
    rank in {0,1,2,3}, 0 = not top-3} -- the sampling analogue of M3's
    plackett_luce_rank_distribution(), used here to draw one concrete rank per realization
    rather than the full marginal distribution."""
    players = list(strengths.keys())
    n_real = len(u_rank1)
    strength_matrix = np.array([strengths[p] for p in players], dtype=float)
    remaining = np.ones((len(players), n_real), dtype=bool)
    ranks = {p: np.zeros(n_real, dtype=int) for p in players}

    for rank_num, u in ((1, u_rank1), (2, u_rank2), (3, u_rank3)):
        masked = np.where(remaining, strength_matrix, 0.0)
        total = masked.sum(axis=0)
        valid = total > 0
        total_safe = np.where(valid, total, 1.0)
        cumprob = np.cumsum(masked, axis=0) / total_safe
        pick_mask = cumprob >= u[None, :]
        picked_idx = np.argmax(pick_mask, axis=0)
        for i, p in enumerate(players):
            sel = valid & (picked_idx == i)
            ranks[p][sel] = rank_num
            remaining[i, sel] = False
    return ranks


# ============================================================
# per-fixture roster / team-side lookup (same join pattern as uncertainty.run)
# ============================================================

def _team_of_for_fixture(con: duckdb.DuckDBPyConnection, home_uid: str, away_uid: str, season: str) -> dict:
    from . import reconcile as reconcile_mod
    team_of = {}
    for team_uid in (home_uid, away_uid):
        found = reconcile_mod._season_root_table(con, season, "teams.csv")
        roster = con.execute(
            """
            SELECT DISTINCT dp.player_uid
            FROM player_alias pa JOIN dim_player dp ON dp.player_uid = pa.player_uid
            JOIN "{}" t ON t.code = pa.team_code
            JOIN team_alias ta ON ta.alias_name = t.name AND ta.season = pa.season
            WHERE pa.season = ? AND ta.team_uid = ?
            """.format(found[1]),
            [season, team_uid],
        ).fetchall()
        for (pid,) in roster:
            team_of[pid] = team_uid
    return team_of


def _fixture_roster(con: duckdb.DuckDBPyConnection, ep_model_version: int, mm_model_version: int, match_id: str) -> list[dict]:
    rows = con.execute(
        """
        SELECT o.player_uid, dp.position, o.expected_bps, o.ep_goals,
               m.p_0min, m.p_1_59min, m.p_60plus_min
        FROM ep_outputs o
        JOIN dim_player dp ON dp.player_uid = o.player_uid
        JOIN minutes_model_outputs m ON m.player_uid = o.player_uid AND m.model_version = ?
        WHERE o.model_version = ? AND o.fixture_match_id = ?
        """,
        [mm_model_version, ep_model_version, match_id],
    ).fetchall()
    out = []
    for player_uid, position, expected_bps, ep_goals, p0, p1, p2 in rows:
        if position not in POSITIONS:
            continue
        out.append({
            "player_uid": player_uid, "position": position, "expected_bps": expected_bps,
            "ep_goals": ep_goals or 0.0, "p_0": p0, "p_1_59": p1, "p_60plus": p2, "p_played": p1 + p2,
        })
    return out


# ============================================================
# core per-fixture simulation
# ============================================================

def simulate_fixture(
    con: duckdb.DuckDBPyConnection, match_id: str, home_uid: str, away_uid: str, target_season: str,
    season_priority: list[str], squad_uids: set, ep_model_version: int, mm_model_version: int, ts_model_version: int,
    scoring_params_version: int, tau_val: float, sigma_z_sq: float, mean_minutes: dict,
    rng: np.random.Generator, n_pairs: int,
) -> dict:
    """Returns {player_uid: {category: array of shape (2*n_pairs,)}} for every squad player
    present in this fixture (empty dict if none). One call = one fixture's contribution to
    one full-gameweek realization, across all 2*n_pairs realizations at once."""
    roster = _fixture_roster(con, ep_model_version, mm_model_version, match_id)
    squad_in_fixture = [r for r in roster if r["player_uid"] in squad_uids]
    if not squad_in_fixture:
        return {}

    team_of = _team_of_for_fixture(con, home_uid, away_uid, target_season)
    lam_home, lam_away, _is_home = ep._fixture_lambdas(con, home_uid, match_id, ts_model_version)
    ts_rho_params_version = con.execute(
        "SELECT rho_params_version FROM team_strength_model_versions WHERE model_version = ?", [ts_model_version]
    ).fetchone()[0]
    rho_dc, _ = params_mod.resolve_param(con, "model_decay_params", "rho", ts_rho_params_version)

    def _u_pair():
        u = rng.random(n_pairs)
        return np.concatenate([u, 1.0 - u])

    n_real = 2 * n_pairs
    grid = bivariate_poisson_grid(lam_home, lam_away, rho_dc)
    home_goals, away_goals = sample_from_grid(grid, _u_pair())
    z_fixture = sample_z_fixture(sigma_z_sq, _u_pair())

    strengths = {}
    per_player = {}
    # Review B4: per-realization goal-rate weights for EVERY fixture participant on each
    # side (squad and non-squad) -- fed to _multinomial_allocate_team_goals() below once the
    # full roster has been walked, so each squad player's simulated goal count is conditioned
    # on the side's own already-drawn scoreline instead of an independent Poisson draw.
    home_goal_weights: dict[str, np.ndarray] = {}
    away_goal_weights: dict[str, np.ndarray] = {}
    for r in roster:
        player_uid = r["player_uid"]
        team_uid = team_of.get(player_uid)

        if player_uid not in squad_uids:
            # non-squad fixture participant: mean-based strength reacting only to the shared
            # Z_fixture tempo factor -- see module docstring for the scope reasoning. Their
            # real M3 per-fixture ep_goals (constant across realizations, not re-simulated --
            # same documented scope boundary) is still a real, legitimate contribution to
            # their side's goal-allocation pool below: a non-squad player who could plausibly
            # have scored some of the team's goals must not be silently excluded from that
            # pool just because this function doesn't track their own individual draw.
            strengths[player_uid] = np.exp(r["expected_bps"] * z_fixture / tau_val) * r["p_played"]
            if team_uid == home_uid:
                home_goal_weights[player_uid] = np.full(n_real, r["ep_goals"])
            elif team_uid == away_uid:
                away_goal_weights[player_uid] = np.full(n_real, r["ep_goals"])
            continue

        if team_uid is None:
            # couldn't resolve this squad player's team side for this fixture (a
            # reconciliation gap) -- cannot compute their clean-sheet/goals-conceded draw
            # without knowing which side's goals are "against" them, so skip rather than guess.
            continue
        is_home_side = team_uid == home_uid
        own_goals_against = away_goals if is_home_side else home_goals
        side_weights = home_goal_weights if is_home_side else away_goal_weights

        position = r["position"]
        state = sample_minutes_state_vec(r["p_0"], r["p_1_59"], r["p_60plus"], _u_pair())
        played = state != "0"
        mean_min = np.where(state == "1_59", mean_minutes["mean_1_59"], np.where(state == "60plus", mean_minutes["mean_60plus"], 0.0))

        rates = ep.player_rates_shrunk(con, player_uid, position, season_priority)
        def_rates = ep._defensive_action_rates_per_90(con, player_uid, position, season_priority)

        # Goals are NOT independently Poisson-drawn here (Review B4's actual fix, not just
        # disclosure -- see module docstring): lam_goals is this player's per-realization
        # share of their side's goal-allocation pool, multinomial-allocated below against the
        # ALREADY-drawn home_goals/away_goals total, once every roster player's own weight is
        # known -- the same joint scoreline clean_sheet/goals_conceded_floor already read off.
        lam_goals = rates["expected_goals_per_90"] * mean_min / 90.0 * z_fixture
        side_weights[player_uid] = lam_goals
        lam_assists = rates["expected_assists_per_90"] * mean_min / 90.0 * z_fixture
        assists = sample_poisson_vec(lam_assists, _u_pair())

        clean_sheet = (own_goals_against == 0) & (state == "60plus")
        goals_conceded_floor = own_goals_against // 2

        defcon_hit = np.zeros(state.shape, dtype=bool)
        if position != "Goalkeeper":
            defcon_rate90 = def_rates["cbi_per_90"] + def_rates["recoveries_per_90"]
            threshold = ep._sm(con, "defcon_threshold", scoring_params_version, position)
            lam_defcon = defcon_rate90 * mean_min / 90.0
            defcon_count = sample_poisson_vec(lam_defcon, _u_pair())
            defcon_hit = defcon_count >= threshold

        saves_count = np.zeros(state.shape, dtype=int)
        if position == "Goalkeeper":
            lam_saves = rates["saves_per_90"] * mean_min / 90.0
            saves_count = sample_poisson_vec(lam_saves, _u_pair())

        strength = np.exp(r["expected_bps"] * z_fixture / tau_val) * played
        strengths[player_uid] = strength
        per_player[player_uid] = {
            "position": position, "state": state, "assists": assists,
            "clean_sheet": clean_sheet, "goals_conceded_floor": goals_conceded_floor,
            "defcon_hit": defcon_hit, "saves_count": saves_count,
        }

    home_allocated = _multinomial_allocate_team_goals(home_goals, home_goal_weights, _u_pair)
    away_allocated = _multinomial_allocate_team_goals(away_goals, away_goal_weights, _u_pair)
    for player_uid, draws in per_player.items():
        allocated = home_allocated.get(player_uid)
        draws["goals"] = allocated if allocated is not None else away_allocated[player_uid]

    ranks = sample_plackett_luce_ranks_vec(strengths, _u_pair(), _u_pair(), _u_pair())

    out = {}
    for player_uid, draws in per_player.items():
        out[player_uid] = {**draws, "rank": ranks[player_uid]}
    return out


# ============================================================
# points assembly (draws -> FPL points, per squad player)
# ============================================================

def _assemble_points(con, position: str, draws: dict, scoring_params_version: int) -> dict:
    goal_pts = ep._sm(con, "goal_points", scoring_params_version, position)
    assist_pts = ep._sm(con, "assist_points", scoring_params_version)
    cs_pts = ep._sm(con, "clean_sheet_points", scoring_params_version, position)
    saves_per_pt = ep._sm(con, "saves_per_point", scoring_params_version)
    defcon_pts = ep._sm(con, "defcon_points", scoring_params_version) if position != "Goalkeeper" else 0.0
    app_1_59 = ep._sm(con, "appearance_points_1_59", scoring_params_version)
    app_60plus = ep._sm(con, "appearance_points_60plus", scoring_params_version)

    state = draws["state"]
    pts_appearance = np.where(state == "1_59", app_1_59, np.where(state == "60plus", app_60plus, 0.0))
    pts_goals = draws["goals"] * goal_pts
    pts_assists = draws["assists"] * assist_pts
    pts_clean_sheet = np.where(draws["clean_sheet"], cs_pts, 0.0) if cs_pts else np.zeros(state.shape)
    pts_goals_conceded = np.where(
        (position in ("Goalkeeper", "Defender")) & (state == "60plus"), -draws["goals_conceded_floor"].astype(float), 0.0
    )
    pts_defcon = np.where(draws["defcon_hit"], defcon_pts, 0.0)
    # Real FPL rule: 1 point per every `saves_per_pt` saves, integer floor -- not a continuous
    # rate. goals_conceded_floor two lines above already gets this right (own_goals_against //
    # 2); this line used continuous division instead, so e.g. saves_count=2 (saves_per_pt=3.0)
    # awarded 0.667 points instead of the true 0, and saves_count=4 awarded 1.333 instead of 1
    # -- systematically overstating every goalkeeper's simulated mean/variance/quantiles here
    # and in every downstream reader (explain_player_risk_empirical, evaluate_triple_captain).
    pts_saves = draws["saves_count"] // saves_per_pt if saves_per_pt and position == "Goalkeeper" else np.zeros(state.shape)
    pts_bonus = np.select([draws["rank"] == 1, draws["rank"] == 2, draws["rank"] == 3], [3.0, 2.0, 1.0], default=0.0)

    total = pts_appearance + pts_goals + pts_assists + pts_clean_sheet + pts_goals_conceded + pts_defcon + pts_bonus + pts_saves
    return {
        "minutes_state": state, "pts_appearance": pts_appearance, "pts_goals": pts_goals,
        "pts_assists": pts_assists, "pts_clean_sheet": pts_clean_sheet, "pts_goals_conceded": pts_goals_conceded,
        "pts_defcon": pts_defcon, "pts_bonus": pts_bonus, "pts_saves": pts_saves, "total_points": total,
    }


# ============================================================
# orchestrator
# ============================================================

def run(
    con: duckdb.DuckDBPyConnection,
    calibration_asof_date: date,
    squad_optimizer_run_id: int,
    ep_model_version: int,
    mm_model_version: int,
    ts_model_version: int,
    uncertainty_model_version: int,
    scoring_params_version: int,
    tau_params_version: int,
    rho_residual_params_version: int,
    n_antithetic_pairs: int = 5000,
    season_priority: tuple[str, ...] = ("2026-2027", "2025-2026", "2024-2025"),
) -> int:
    squad_run = con.execute(
        "SELECT target_season, target_gameweek FROM squad_optimizer_runs WHERE run_id = ?", [squad_optimizer_run_id]
    ).fetchone()
    if not squad_run:
        raise ValueError(f"no squad_optimizer_runs row for run_id={squad_optimizer_run_id}")
    target_season, target_gameweek = squad_run

    squad_uids = {
        r[0] for r in con.execute(
            "SELECT player_uid FROM squad_optimizer_selections WHERE run_id = ? AND in_squad", [squad_optimizer_run_id]
        ).fetchall()
    }
    if not squad_uids:
        raise ValueError(f"squad_optimizer_run_id={squad_optimizer_run_id} has no in_squad players -- cannot simulate")

    tau_val, _ = params_mod.resolve_param(con, "bps_dispersion_params", "tau", tau_params_version)
    rho_residual, _ = params_mod.resolve_param(con, "correlation_params", "rho_residual", rho_residual_params_version)
    mean_minutes = ep._mean_minutes_by_bucket(con)

    lambda_representative = compute_lambda_representative(con, list(squad_uids), ep_model_version, scoring_params_version)
    sigma_z_sq = z_fixture_variance(rho_residual, lambda_representative)

    model_version = con.execute("SELECT nextval('seq_monte_carlo_model_version')").fetchone()[0]
    query_id = f"squad_run_{squad_optimizer_run_id}_gw{target_gameweek}"
    seed = deterministic_seed(model_version, calibration_asof_date, query_id)

    con.execute(
        """
        INSERT INTO monte_carlo_run_versions
            (model_version, calibration_asof_date, squad_optimizer_run_id, ep_model_version,
             minutes_model_version, team_strength_model_version, uncertainty_model_version,
             rho_residual_params_version, z_fixture_lambda_representative, z_fixture_variance,
             n_antithetic_pairs, query_id, seed)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [model_version, calibration_asof_date, squad_optimizer_run_id, ep_model_version, mm_model_version,
         ts_model_version, uncertainty_model_version, rho_residual_params_version, lambda_representative,
         sigma_z_sq, n_antithetic_pairs, query_id, seed],
    )

    rng = np.random.default_rng(seed)
    n_real = 2 * n_antithetic_pairs
    realization_index = np.arange(n_real)

    fixtures = con.execute(
        "SELECT match_id, home_team_uid, away_team_uid FROM fact_match "
        "WHERE season = ? AND gameweek = ? AND competition = ?",
        [target_season, target_gameweek, ep.PL],
    ).fetchall()

    frames = []
    fixture_of, team_of_squad = {}, {}
    for match_id, home_uid, away_uid in fixtures:
        fixture_result = simulate_fixture(
            con, match_id, home_uid, away_uid, target_season, list(season_priority), squad_uids,
            ep_model_version, mm_model_version, ts_model_version, scoring_params_version,
            tau_val, sigma_z_sq, mean_minutes, rng, n_antithetic_pairs,
        )
        if not fixture_result:
            continue
        team_of = _team_of_for_fixture(con, home_uid, away_uid, target_season)
        for player_uid, draws in fixture_result.items():
            position = draws["position"]
            pts = _assemble_points(con, position, draws, scoring_params_version)
            fixture_of[player_uid] = match_id
            team_of_squad[player_uid] = team_of.get(player_uid)
            frames.append(pd.DataFrame({
                "model_version": model_version, "player_uid": player_uid,
                "realization_index": realization_index, "minutes_state": pts["minutes_state"],
                "pts_appearance": pts["pts_appearance"], "pts_goals": pts["pts_goals"],
                "pts_assists": pts["pts_assists"], "pts_clean_sheet": pts["pts_clean_sheet"],
                "pts_goals_conceded": pts["pts_goals_conceded"], "pts_defcon": pts["pts_defcon"],
                "pts_bonus": pts["pts_bonus"], "pts_saves": pts["pts_saves"], "total_points": pts["total_points"],
            }))

    simulated_uids = set(fixture_of.keys())
    # squad players with no fixture this gameweek (blank gameweek) are legitimately excluded --
    # not an error, but worth being able to see (M8/M9's job to handle blanks explicitly).

    cols = [
        "model_version", "player_uid", "realization_index", "minutes_state", "pts_appearance",
        "pts_goals", "pts_assists", "pts_clean_sheet", "pts_goals_conceded", "pts_defcon",
        "pts_bonus", "pts_saves", "total_points",
    ]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
    con.register("_mc_totals_df", df)
    con.execute("INSERT INTO monte_carlo_player_totals SELECT * FROM _mc_totals_df")
    con.unregister("_mc_totals_df")

    con.execute(
        """
        INSERT INTO monte_carlo_player_summary
        SELECT ?, player_uid, avg(total_points), var_pop(total_points),
               quantile_cont(total_points, 0.05), quantile_cont(total_points, 0.25),
               quantile_cont(total_points, 0.75), quantile_cont(total_points, 0.95),
               min(total_points), max(total_points)
        FROM monte_carlo_player_totals WHERE model_version = ?
        GROUP BY player_uid
        """,
        [model_version, model_version],
    )

    if simulated_uids:
        wide = df.pivot(index="realization_index", columns="player_uid", values="total_points")
        cov_matrix = wide.cov(ddof=0)
        m4_cov = {
            (a, b): cov for a, b, cov in con.execute(
                "SELECT player_uid_a, player_uid_b, covariance FROM cross_player_covariance WHERE model_version = ?",
                [uncertainty_model_version],
            ).fetchall()
        }
        uids_sorted = sorted(simulated_uids)
        for i, a in enumerate(uids_sorted):
            for b in uids_sorted[i + 1:]:
                empirical_cov = float(cov_matrix.loc[a, b])
                if fixture_of[a] != fixture_of[b]:
                    relationship = "independent"
                elif team_of_squad[a] == team_of_squad[b]:
                    relationship = "teammate"
                else:
                    relationship = "opponent"
                lo, hi = sorted([a, b])
                m4_val = m4_cov.get((lo, hi))
                con.execute(
                    "INSERT INTO monte_carlo_empirical_covariance "
                    "(model_version, player_uid_a, player_uid_b, relationship, empirical_covariance, m4_covariance) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    [model_version, lo, hi, relationship, empirical_cov, m4_val],
                )

    return model_version


# ============================================================
# M9 adapter -- empirical risk display, alongside M4's analytic quantiles
# ============================================================

def explain_player_risk_empirical(con: duckdb.DuckDBPyConnection, mc_model_version: int, player_uid: str) -> dict | None:
    """M9's risk-display section (empirical leg): monte_carlo_player_summary is already
    schema-commented as "the M9-facing summary" (schema/0007) -- this is that summary read in
    a display-ready shape, offered alongside M4's Cornish-Fisher quantiles (a real simulated
    distribution vs. an analytic approximation), not instead of. Returns None if this player
    wasn't part of the simulated squad at this model_version."""
    row = con.execute(
        "SELECT mean_total, var_total, quantile_05, quantile_25, quantile_75, quantile_95, "
        "min_total, max_total FROM monte_carlo_player_summary WHERE model_version = ? AND player_uid = ?",
        [mc_model_version, player_uid],
    ).fetchone()
    if row is None:
        return None
    mean_total, var_total, q05, q25, q75, q95, min_total, max_total = row
    return {
        "player_uid": player_uid, "mean": mean_total, "var_total": var_total,
        "floor": q05, "q25": q25, "q75": q75, "ceiling": q95,
        "min": min_total, "max": max_total,
    }
