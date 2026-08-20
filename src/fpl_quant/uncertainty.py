"""M4: Uncertainty & Correlation Layer.

Var[total_points] = Sum_c Var[category_c] + 2*Sum_{c<c'} Cov[category_c, category_c'].
Category variances follow directly from M3's own distributional choices (Poisson:
Var=mean; Bernoulli/threshold: Var=p(1-p); bonus: full categorical over {0,1,2,3} from
the Plackett-Luce ranks) -- no separate empirical-fitting exercise, per spec.

Within-player covariance is the law of total covariance conditioned on M2's three-state
minutes distribution: a shared-minutes-gate term (computed exactly from each category's
conditional mean per state) plus a residual term for correlation *given* the player is
playing, approximated with the spec's own pinned rho_residual=0.15 placeholder.

Scope approximation, stated plainly: the residual term needs Var(c | playing), which would
require each category's within-state second moment, not just its state mean. This
implementation proxies Var(c | playing) with the category's unconditional variance --
reasonable given rho_residual is itself already an invented placeholder explicitly flagged
for full replacement (not mere recalibration) once M6's Monte Carlo engine exists, per
spec. Not a hidden shortcut -- named here and in the README.

Cross-player Sigma is built block-wise by fixture: teammates correlate positively through
the shared lambda_for draw (attacking) and the shared clean-sheet Bernoulli outcome
(defensive); opposing-fixture players correlate negatively on the clean-sheet/goals-
conceded axis; different-fixture players in the same gameweek are a confirmed zero
covariance, per spec -- not stored at all.

Cornish-Fisher quantiles are reporting/explainability output only (feeds M9) -- confirmed
NOT wired into M5's optimization objective, which works directly off Sigma (this module)
and the means (M3).
"""

import math
from datetime import date, datetime, timezone

import duckdb
import numpy as np
from scipy.stats import norm, poisson

from . import expected_points as ep
from . import params as params_mod

_QUANTILE_Z = {"quantile_05": norm.ppf(0.05), "quantile_25": norm.ppf(0.25),
               "quantile_75": norm.ppf(0.75), "quantile_95": norm.ppf(0.95)}


def seed_v1_params(con: duckdb.DuckDBPyConnection) -> None:
    # Invented placeholder (no literature to cite), flagged for full replacement -- not
    # mere recalibration -- once M6's Monte Carlo engine exists to capture this structure
    # directly, per spec.
    params_mod.write_param(con, "correlation_params", 1, "2026-08-10", "rho_residual", value_numeric=0.15)
    # Cross-player block-structure correlation coefficients: invented v1 defaults (same
    # status as rho_residual -- no literature, ordinal reasoning only: teammates' shared
    # clean-sheet/goals-conceded Bernoulli draw is near-deterministic within a match, so
    # pinned high; shared attacking tempo is real but far looser, so pinned low; opponents
    # get a materially smaller version of the same two effects). Flagged for the same M6
    # Monte Carlo replacement as rho_residual.
    for key, value in {
        "teammate_attacking": 0.25, "opponent_attacking": 0.08,
        "teammate_defensive": 0.9, "opponent_defensive": 0.5,
    }.items():
        params_mod.write_param(con, "cross_player_correlation_params", 1, "2026-08-10", key, value_numeric=value)


def _rho_residual(con, params_version):
    v, _ = params_mod.resolve_param(con, "correlation_params", "rho_residual", params_version)
    return v


# ============================================================
# per-category variance (points-space, re-derived from ep_outputs via base_scoring_matrix)
# ============================================================

def category_variances(con, ep_row: dict, position: str, scoring_params_version: int) -> dict:
    goal_pts = ep._sm(con, "goal_points", scoring_params_version, position)
    assist_pts = ep._sm(con, "assist_points", scoring_params_version)
    cs_pts = ep._sm(con, "clean_sheet_points", scoring_params_version, position)
    saves_per_pt = ep._sm(con, "saves_per_point", scoring_params_version)
    defcon_pts = ep._sm(con, "defcon_points", scoring_params_version) if position != "Goalkeeper" else 0.0

    p0, p1, p2 = ep_row["p_0"], ep_row["p_1_59"], ep_row["p_60plus"]
    e_app = 1 * p1 + 2 * p2
    e_app2 = 1 * p1 + 4 * p2
    var_appearance = max(e_app2 - e_app**2, 0.0)

    e_goals_count = ep_row["ep_goals"] / goal_pts if goal_pts else 0.0
    var_goals = goal_pts**2 * e_goals_count
    e_assists_count = ep_row["ep_assists"] / assist_pts if assist_pts else 0.0
    var_assists = assist_pts**2 * e_assists_count

    p_cs = ep_row["ep_clean_sheet"] / cs_pts if cs_pts else 0.0
    var_clean_sheet = cs_pts**2 * p_cs * (1 - p_cs) if cs_pts else 0.0

    var_goals_conceded = 0.0
    if position in ("Goalkeeper", "Defender") and ep_row["lambda_against"] is not None:
        lam = ep_row["lambda_against"]
        e_fh = ep._expected_floor_half(lam)
        e_fh2 = sum(((k // 2) ** 2) * poisson.pmf(k, lam) for k in range(16))
        var_fh_given_played = max(e_fh2 - e_fh**2, 0.0)
        p60 = p2
        # G = -floor(X/2) w.p. p60 (gate open), else 0 -- variance of this mixture:
        var_goals_conceded = p60 * var_fh_given_played + p60 * (1 - p60) * e_fh**2

    p_defcon = ep_row["ep_defcon"] / defcon_pts if defcon_pts else 0.0
    var_defcon = defcon_pts**2 * p_defcon * (1 - p_defcon) if defcon_pts else 0.0

    p_r1, p_r2, p_r3 = ep_row["p_rank1"], ep_row["p_rank2"], ep_row["p_rank3"]
    e_bonus = 3 * p_r1 + 2 * p_r2 + 1 * p_r3
    e_bonus2 = 9 * p_r1 + 4 * p_r2 + 1 * p_r3
    var_bonus = max(e_bonus2 - e_bonus**2, 0.0)

    var_saves = 0.0
    if position == "Goalkeeper" and saves_per_pt:
        e_saves_count = ep_row["ep_saves"] * saves_per_pt
        var_saves = e_saves_count / saves_per_pt**2

    return {
        "var_appearance": var_appearance, "var_goals": var_goals, "var_assists": var_assists,
        "var_clean_sheet": var_clean_sheet, "var_goals_conceded": var_goals_conceded,
        "var_defcon": var_defcon, "var_bonus": var_bonus, "var_saves": var_saves,
    }


# ============================================================
# conditional-on-minutes-state category means (points-space) -- the shared-gate covariance term
# ============================================================

def category_state_means(con, ep_row: dict, position: str, rates: dict, def_rates: dict,
                          mean_minutes: dict, scoring_params_version: int) -> dict:
    """{category: (mean_given_0min, mean_given_1_59, mean_given_60plus)}."""
    goal_pts = ep._sm(con, "goal_points", scoring_params_version, position)
    assist_pts = ep._sm(con, "assist_points", scoring_params_version)
    cs_pts = ep._sm(con, "clean_sheet_points", scoring_params_version, position)
    saves_per_pt = ep._sm(con, "saves_per_point", scoring_params_version)
    defcon_pts = ep._sm(con, "defcon_points", scoring_params_version) if position != "Goalkeeper" else 0.0

    m1, m2 = mean_minutes["mean_1_59"], mean_minutes["mean_60plus"]
    means = {"appearance": (0.0, 1.0, 2.0)}

    means["goals"] = (0.0, goal_pts * rates["expected_goals_per_90"] * m1 / 90, goal_pts * rates["expected_goals_per_90"] * m2 / 90)
    means["assists"] = (0.0, assist_pts * rates["expected_assists_per_90"] * m1 / 90, assist_pts * rates["expected_assists_per_90"] * m2 / 90)

    # clean sheet and goals conceded are gated exactly (resp. approximately, per M3) at 60+
    if cs_pts and ep_row["lambda_against"] is not None:
        p_opp_0 = math.exp(-ep_row["lambda_against"])
        means["clean_sheet"] = (0.0, 0.0, cs_pts * p_opp_0)
    else:
        means["clean_sheet"] = (0.0, 0.0, 0.0)

    if position in ("Goalkeeper", "Defender") and ep_row["lambda_against"] is not None:
        means["goals_conceded"] = (0.0, 0.0, -ep._expected_floor_half(ep_row["lambda_against"]))
    else:
        means["goals_conceded"] = (0.0, 0.0, 0.0)

    if defcon_pts:
        thr = ep._sm(con, "defcon_threshold", scoring_params_version, position)
        rate90 = def_rates["cbi_per_90"] + def_rates["recoveries_per_90"]
        p1_over = 1 - poisson.cdf(thr - 1, max(rate90 * m1 / 90, 1e-9))
        p2_over = 1 - poisson.cdf(thr - 1, max(rate90 * m2 / 90, 1e-9))
        means["defcon"] = (0.0, defcon_pts * p1_over, defcon_pts * p2_over)
    else:
        means["defcon"] = (0.0, 0.0, 0.0)

    # bonus: proportional-to-minutes approximation within the "played" states (BPS rank
    # depends on the whole fixture, not just this player's own minutes bucket in
    # isolation -- exact conditioning is out of scope here, see module docstring)
    e_bonus_overall = ep_row["ep_bonus"]
    p_played = ep_row["p_1_59"] + ep_row["p_60plus"]
    if p_played > 0 and (m1 * ep_row["p_1_59"] + m2 * ep_row["p_60plus"]) > 0:
        e_min_played = (m1 * ep_row["p_1_59"] + m2 * ep_row["p_60plus"]) / p_played
        means["bonus"] = (0.0, e_bonus_overall * m1 / e_min_played, e_bonus_overall * m2 / e_min_played)
    else:
        means["bonus"] = (0.0, 0.0, 0.0)

    if position == "Goalkeeper" and saves_per_pt:
        means["saves"] = (0.0, rates["saves_per_90"] * m1 / 90 / saves_per_pt, rates["saves_per_90"] * m2 / 90 / saves_per_pt)
    else:
        means["saves"] = (0.0, 0.0, 0.0)

    return means


def _shared_gate_covariance(state_means: dict, p0: float, p1: float, p2: float, cat_a: str, cat_b: str) -> float:
    a0, a1, a2 = state_means[cat_a]
    b0, b1, b2 = state_means[cat_b]
    e_a = p0 * a0 + p1 * a1 + p2 * a2
    e_b = p0 * b0 + p1 * b1 + p2 * b2
    e_ab = p0 * a0 * b0 + p1 * a1 * b1 + p2 * a2 * b2
    return e_ab - e_a * e_b


CATEGORIES = ["appearance", "goals", "assists", "clean_sheet", "goals_conceded", "defcon", "bonus", "saves"]


def total_variance(con, ep_row: dict, position: str, rates: dict, def_rates: dict,
                    mean_minutes: dict, scoring_params_version: int, rho_residual: float) -> tuple[dict, float]:
    variances = category_variances(con, ep_row, position, scoring_params_version)
    state_means = category_state_means(con, ep_row, position, rates, def_rates, mean_minutes, scoring_params_version)
    p0, p1, p2 = ep_row["p_0"], ep_row["p_1_59"], ep_row["p_60plus"]

    # rho_residual is a flat, POSITIVE placeholder correlation applied uniformly to every
    # active category pair "given the player is playing." That's a reasonable default for
    # most pairs (e.g. goals/assists/bonus genuinely tend to move together), but it's wrong
    # for a same-player pair that is mechanically complementary rather than reinforcing:
    # clean_sheet and goals_conceded are both driven off the *same* opponent-goals draw for
    # this player's own team in this same match, and a clean sheet (opponent scores 0)
    # structurally excludes goals being conceded. The shared-gate term below already gets
    # this right (it correctly comes out negative for this pair, since clean_sheet's state
    # mean is positive exactly where goals_conceded's is at its least-negative), but adding
    # a flat +rho_residual on top was silently fighting that correct negative signal and
    # understating GK/DEF risk. Real bug, fixed here: this specific pair gets the residual
    # applied with a negative sign instead of the flat positive default.
    NEGATIVELY_LINKED_PAIRS = {frozenset({"clean_sheet", "goals_conceded"})}

    var_total = sum(variances.values())
    active = [c for c in CATEGORIES if variances[f"var_{c}"] > 0]
    for i, ca in enumerate(active):
        for cb in active[i + 1:]:
            shared = _shared_gate_covariance(state_means, p0, p1, p2, ca, cb)
            residual_magnitude = rho_residual * math.sqrt(variances[f"var_{ca}"] * variances[f"var_{cb}"])
            residual_sign = -1.0 if frozenset({ca, cb}) in NEGATIVELY_LINKED_PAIRS else 1.0
            var_total += 2 * (shared + residual_sign * residual_magnitude)

    return variances, max(var_total, 0.0)


# ============================================================
# Cornish-Fisher skew/kurtosis (moment-combination approximation, reporting only)
# ============================================================

def _discrete_moments(values: list[float], probs: list[float]) -> tuple[float, float, float, float]:
    mean = sum(v * p for v, p in zip(values, probs))
    m2 = sum(((v - mean) ** 2) * p for v, p in zip(values, probs))
    m3 = sum(((v - mean) ** 3) * p for v, p in zip(values, probs))
    m4 = sum(((v - mean) ** 4) * p for v, p in zip(values, probs))
    return mean, m2, m3, m4


def category_skew_excess_kurtosis(ep_row: dict, position: str, con, scoring_params_version: int) -> dict:
    """Closed-form skew/excess-kurtosis per category, from each category's own assumed
    distributional form (Poisson / Bernoulli / discrete), per spec."""
    out = {}

    p0, p1, p2 = ep_row["p_0"], ep_row["p_1_59"], ep_row["p_60plus"]
    _, _, m3, m4 = _discrete_moments([0, 1, 2], [p0, p1, p2])
    var = max(1 * p1 + 4 * p2 - (1 * p1 + 2 * p2) ** 2, 1e-12)
    out["appearance"] = (m3 / var**1.5, m4 / var**2 - 3)

    goal_pts = ep._sm(con, "goal_points", scoring_params_version, position)
    lam_goals = ep_row["ep_goals"] / goal_pts if goal_pts else 0.0
    out["goals"] = (1 / math.sqrt(lam_goals), 1 / lam_goals) if lam_goals > 1e-9 else (0.0, 0.0)

    assist_pts = ep._sm(con, "assist_points", scoring_params_version)
    lam_assists = ep_row["ep_assists"] / assist_pts if assist_pts else 0.0
    out["assists"] = (1 / math.sqrt(lam_assists), 1 / lam_assists) if lam_assists > 1e-9 else (0.0, 0.0)

    cs_pts = ep._sm(con, "clean_sheet_points", scoring_params_version, position)
    p_cs = ep_row["ep_clean_sheet"] / cs_pts if cs_pts else 0.0
    out["clean_sheet"] = _bernoulli_skew_kurt(p_cs)

    defcon_pts = ep._sm(con, "defcon_points", scoring_params_version) if position != "Goalkeeper" else 0.0
    p_dc = ep_row["ep_defcon"] / defcon_pts if defcon_pts else 0.0
    out["defcon"] = _bernoulli_skew_kurt(p_dc)

    saves_per_pt = ep._sm(con, "saves_per_point", scoring_params_version)
    lam_saves = ep_row["ep_saves"] * saves_per_pt if position == "Goalkeeper" else 0.0
    out["saves"] = (1 / math.sqrt(lam_saves), 1 / lam_saves) if lam_saves > 1e-9 else (0.0, 0.0)

    p_r1, p_r2, p_r3 = ep_row["p_rank1"], ep_row["p_rank2"], ep_row["p_rank3"]
    p_r0 = max(0.0, 1 - p_r1 - p_r2 - p_r3)
    _, m2b, m3b, m4b = _discrete_moments([0, 1, 2, 3], [p_r0, p_r3, p_r2, p_r1])
    var_b = max(m2b, 1e-12)
    out["bonus"] = (m3b / var_b**1.5, m4b / var_b**2 - 3)

    out["goals_conceded"] = (0.0, 0.0)  # not modeled independently -- folded into "clean_sheet"-adjacent uncertainty at this scope
    return out


def _bernoulli_skew_kurt(p: float) -> tuple[float, float]:
    if p <= 1e-9 or p >= 1 - 1e-9:
        return 0.0, 0.0
    var = p * (1 - p)
    skew = (1 - 2 * p) / math.sqrt(var)
    excess_kurtosis = (1 - 6 * var) / var
    return skew, excess_kurtosis


def combined_skew_excess_kurtosis(variances: dict, per_category_sk: dict, var_total: float) -> tuple[float, float]:
    """Additive-third/fourth-moment approximation weighted by each category's own variance
    share -- ignores cross-category moment terms, a reasonable simplification given the
    covariance structure itself is already only approximately known (rho_residual)."""
    if var_total <= 1e-12:
        return 0.0, 0.0
    skew_total = 0.0
    kurt_total = 0.0
    for c in CATEGORIES:
        var_c = variances.get(f"var_{c}", 0.0)
        if var_c <= 1e-12 or c not in per_category_sk:
            continue
        skew_c, kurt_c = per_category_sk[c]
        skew_total += (var_c ** 1.5) * skew_c
        kurt_total += (var_c ** 2) * kurt_c
    return skew_total / var_total**1.5, kurt_total / var_total**2


def cornish_fisher_quantile(mean: float, var: float, skew: float, excess_kurtosis: float, q: float) -> float:
    z = norm.ppf(q)
    z_cf = (
        z + (z**2 - 1) / 6 * skew
        + (z**3 - 3 * z) / 24 * excess_kurtosis
        - (2 * z**3 - 5 * z) / 36 * skew**2
    )
    return mean + z_cf * math.sqrt(max(var, 0.0))


# ============================================================
# cross-player covariance, block-wise by fixture
# ============================================================

def cross_player_covariance_for_fixture(
    con, fixture_rows: list[dict], home_uid: str, away_uid: str, corr_params_version: int,
) -> list[tuple[str, str, str, float]]:
    """Returns (player_uid_a, player_uid_b, relationship, covariance) for every nonzero
    pair within one fixture. Teammates: positive via shared attacking (lambda_for) and
    defensive (clean-sheet) structure. Opponents: negative on the clean-sheet/goals-
    conceded axis. Different fixtures within a gameweek: zero, confirmed -- never called
    for those pairs at all."""
    def _corr(key):
        v, _ = params_mod.resolve_param(con, "cross_player_correlation_params", key, corr_params_version)
        return v

    teammate_attacking = _corr("teammate_attacking")
    opponent_attacking = _corr("opponent_attacking")
    teammate_defensive = _corr("teammate_defensive")
    opponent_defensive = _corr("opponent_defensive")

    out = []
    n = len(fixture_rows)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = fixture_rows[i], fixture_rows[j]
            is_teammate = a["team_uid"] == b["team_uid"]
            cov = 0.0

            # attacking co-movement: both teams' expected attacking returns share the same
            # match-tempo/goal-environment realization -- teammates share it directly
            # (same lambda_for), opponents share it more weakly and on the same sign
            # (a high-scoring match lifts both sides' attacking categories together) EXCEPT
            # on the clean-sheet/goals-conceded axis where one side's goal is the other's
            # concession -- handled as its own explicit term below.
            attack_a = a["var_goals"] + a["var_assists"] + a["var_bonus"]
            attack_b = b["var_goals"] + b["var_assists"] + b["var_bonus"]
            if attack_a > 0 and attack_b > 0:
                shared_tempo_corr = teammate_attacking if is_teammate else opponent_attacking
                cov += shared_tempo_corr * math.sqrt(attack_a * attack_b)

            # defensive: teammates' clean-sheet outcomes are the *same* Bernoulli draw
            # (same match, same team) -- strongly positively correlated.
            if is_teammate and a["var_clean_sheet"] > 0 and b["var_clean_sheet"] > 0:
                cov += teammate_defensive * math.sqrt(a["var_clean_sheet"] * b["var_clean_sheet"])
            # goals-conceded, same reasoning (same team, same match, shared concession count)
            if is_teammate and a["var_goals_conceded"] > 0 and b["var_goals_conceded"] > 0:
                cov += teammate_defensive * math.sqrt(a["var_goals_conceded"] * b["var_goals_conceded"])

            # opponents: team A's goals directly reduce team B's clean-sheet probability --
            # negative correlation on exactly this axis.
            if not is_teammate:
                cs_a, gc_b = a["var_clean_sheet"], b["var_goals_conceded"]
                cs_b, gc_a = b["var_clean_sheet"], a["var_goals_conceded"]
                if cs_a > 0 and gc_b > 0:
                    cov -= opponent_defensive * math.sqrt(cs_a * gc_b)
                if cs_b > 0 and gc_a > 0:
                    cov -= opponent_defensive * math.sqrt(cs_b * gc_a)

            if abs(cov) > 1e-9:
                out.append((a["player_uid"], b["player_uid"], "teammate" if is_teammate else "opponent", cov))
    return out


# ============================================================
# orchestrator
# ============================================================

def run(
    con: duckdb.DuckDBPyConnection,
    calibration_asof_date: date,
    ep_model_version: int,
    mm_model_version: int,
    ts_model_version: int,
    scoring_params_version: int,
    bps_params_version: int,
    tau_params_version: int,
    rho_residual_params_version: int,
    corr_params_version: int,
    season_priority: tuple[str, ...] = ("2026-2027", "2025-2026", "2024-2025"),
) -> int:
    rho_residual = _rho_residual(con, rho_residual_params_version)
    tau, _ = params_mod.resolve_param(con, "bps_dispersion_params", "tau", tau_params_version)
    mean_minutes = ep._mean_minutes_by_bucket(con)

    model_version = con.execute(
        """
        INSERT INTO uncertainty_model_versions
            (calibration_asof_date, ep_model_version, minutes_model_version,
             team_strength_model_version, rho_residual_params_version)
        VALUES (?, ?, ?, ?, ?) RETURNING model_version
        """,
        [calibration_asof_date, ep_model_version, mm_model_version, ts_model_version, rho_residual_params_version],
    ).fetchone()[0]

    fixtures = con.execute(
        "SELECT DISTINCT fixture_match_id FROM ep_outputs WHERE model_version = ?", [ep_model_version]
    ).fetchall()

    for (match_id,) in fixtures:
        home_uid, away_uid = con.execute(
            "SELECT home_team_uid, away_team_uid FROM fact_match WHERE match_id = ?", [match_id]
        ).fetchone()

        rows = con.execute(
            """
            SELECT o.player_uid, dp.position, o.ep_goals, o.ep_assists, o.ep_clean_sheet,
                   o.ep_goals_conceded, o.ep_defcon, o.ep_bonus, o.ep_saves, o.expected_bps,
                   m.p_0min, m.p_1_59min, m.p_60plus_min
            FROM ep_outputs o
            JOIN dim_player dp ON dp.player_uid = o.player_uid
            JOIN minutes_model_outputs m ON m.player_uid = o.player_uid AND m.model_version = ?
            WHERE o.model_version = ? AND o.fixture_match_id = ?
            """,
            [mm_model_version, ep_model_version, match_id],
        ).fetchall()
        if not rows:
            continue

        # team_uid per player, for the fixture-block covariance structure
        team_of = {}
        for team_uid in (home_uid, away_uid):
            found = ep.reconcile_mod._season_root_table(con, season_priority[0], "teams.csv")
            roster = con.execute(
                """
                SELECT DISTINCT dp.player_uid
                FROM player_alias pa JOIN dim_player dp ON dp.player_uid = pa.player_uid
                JOIN "{}" t ON t.code = pa.team_code
                JOIN team_alias ta ON ta.alias_name = t.name AND ta.season = pa.season
                WHERE pa.season = ? AND ta.team_uid = ?
                """.format(found[1]),
                [season_priority[0], team_uid],
            ).fetchall()
            for (pid,) in roster:
                team_of[pid] = team_uid

        strengths = {r[0]: math.exp(r[9] / tau) * (r[11] + r[12]) for r in rows}
        rank_dist = ep.plackett_luce_rank_distribution(strengths)

        fixture_rows = []
        for (player_uid, position, ep_goals, ep_assists, ep_clean_sheet, ep_goals_conceded,
             ep_defcon, ep_bonus, ep_saves, expected_bps, p0, p1, p2) in rows:
            team_uid = team_of.get(player_uid)
            if team_uid is None:
                continue
            # needed for every position, not just GK/DEF: clean_sheet's conditional state
            # mean uses P(opponent scores 0) regardless of the scoring player's own position.
            _lf, lambda_against, _ih = ep._fixture_lambdas(con, team_uid, match_id, ts_model_version)

            p_r1, p_r2, p_r3 = rank_dist.get(player_uid, (0.0, 0.0, 0.0))
            ep_row = {
                "ep_goals": ep_goals, "ep_assists": ep_assists, "ep_clean_sheet": ep_clean_sheet,
                "ep_goals_conceded": ep_goals_conceded, "ep_defcon": ep_defcon, "ep_bonus": ep_bonus,
                "ep_saves": ep_saves, "p_0": p0, "p_1_59": p1, "p_60plus": p2,
                "lambda_against": lambda_against, "p_rank1": p_r1, "p_rank2": p_r2, "p_rank3": p_r3,
            }
            rates = ep.player_rates_shrunk(con, player_uid, position, list(season_priority))
            def_rates = ep._defensive_action_rates_per_90(con, player_uid, position, list(season_priority))

            variances, var_total = total_variance(
                con, ep_row, position, rates, def_rates, mean_minutes, scoring_params_version, rho_residual
            )
            per_cat_sk = category_skew_excess_kurtosis(ep_row, position, con, scoring_params_version)
            skew, excess_kurt = combined_skew_excess_kurtosis(variances, per_cat_sk, var_total)
            mean_total = ep_goals + ep_assists + ep_clean_sheet + ep_goals_conceded + ep_defcon + ep_bonus + ep_saves + (1 * p1 + 2 * p2)

            quantiles = {
                key: cornish_fisher_quantile(mean_total, var_total, skew, excess_kurt, q)
                for key, q in {"quantile_05": 0.05, "quantile_25": 0.25, "quantile_75": 0.75, "quantile_95": 0.95}.items()
            }

            con.execute(
                """
                INSERT INTO uncertainty_outputs
                    (model_version, player_uid, fixture_match_id, var_appearance, var_goals, var_assists,
                     var_clean_sheet, var_goals_conceded, var_defcon, var_bonus, var_saves, var_total,
                     skew, excess_kurtosis, quantile_05, quantile_25, quantile_75, quantile_95)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (model_version, player_uid, fixture_match_id) DO NOTHING
                """,
                [model_version, player_uid, match_id, variances["var_appearance"], variances["var_goals"],
                 variances["var_assists"], variances["var_clean_sheet"], variances["var_goals_conceded"],
                 variances["var_defcon"], variances["var_bonus"], variances["var_saves"], var_total,
                 skew, excess_kurt, quantiles["quantile_05"], quantiles["quantile_25"],
                 quantiles["quantile_75"], quantiles["quantile_95"]],
            )
            fixture_rows.append({"player_uid": player_uid, "team_uid": team_uid, **variances})

        pairs = cross_player_covariance_for_fixture(con, fixture_rows, home_uid, away_uid, corr_params_version)
        for player_a, player_b, relationship, cov in pairs:
            lo, hi = sorted([player_a, player_b])
            con.execute(
                "INSERT INTO cross_player_covariance (model_version, player_uid_a, player_uid_b, "
                "fixture_match_id, relationship, covariance) VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (model_version, player_uid_a, player_uid_b) DO NOTHING",
                [model_version, lo, hi, match_id, relationship, cov],
            )

    return model_version


# ============================================================
# M9 adapter -- Cornish-Fisher risk display
# ============================================================

# The M9 spec's own words for the caveat this section must carry to the display layer,
# verbatim -- not this module's own claim (uncertainty.py's own docstring only says the
# Cornish-Fisher quantiles are "reporting/explainability output only... confirmed NOT wired
# into M5's optimization objective"; the "unvalidated" framing is M9's, quoted at its source
# rather than paraphrased into a second, possibly-drifting version of the same warning).
CORNISH_FISHER_DISPLAY_CAVEAT = (
    "Cornish-Fisher floor/ceiling quantiles, carried through with their 'unvalidated pending "
    "per-gameweek panel reconciliation' caveat attached at the display layer, not buried only "
    "in internal docs."
)


def explain_player_risk(con: duckdb.DuckDBPyConnection, uncertainty_model_version: int, player_uid: str) -> dict | None:
    """M9's risk-display section (analytic leg): M4's Cornish-Fisher quantiles, per-fixture,
    not aggregated across a horizon -- one row per (model_version, player_uid) is expected
    since M3/M4 are both single-gameweek per call. Returns None for a legitimate blank
    gameweek (no fixture_match_id row for this player at this model_version)."""
    row = con.execute(
        "SELECT fixture_match_id, var_total, skew, excess_kurtosis, quantile_05, quantile_25, "
        "quantile_75, quantile_95 FROM uncertainty_outputs WHERE model_version = ? AND player_uid = ?",
        [uncertainty_model_version, player_uid],
    ).fetchone()
    if row is None:
        return None
    fixture_match_id, var_total, skew, excess_kurtosis, q05, q25, q75, q95 = row
    return {
        "player_uid": player_uid, "fixture_match_id": fixture_match_id,
        "floor": q05, "q25": q25, "q75": q75, "ceiling": q95,
        "var_total": var_total, "skew": skew, "excess_kurtosis": excess_kurtosis,
        "caveat": CORNISH_FISHER_DISPLAY_CAVEAT,
    }
