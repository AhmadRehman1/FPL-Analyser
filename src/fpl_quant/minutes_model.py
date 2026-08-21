"""M2: Minutes Model.

Three-state distribution per player per gameweek: P(0 min), P(1-59 min), P(60+ min),
required to sum to 1 for every player-gameweek (M2's own dedicated invariant).

Historical base rate (position-shrunk, recency-weighted) -> logit-scale evidence
adjustment (shift-type + pull-type, M1b's effective_weight) -> conditional
minutes-given-appearance -> three-state output. GW0 friendly minutes are excluded from
the quantitative fit entirely by construction (the competition='Premier League' filter
already excludes 'Friendlies'/'Community Shield'/etc) and instead logged as a low-weight
preseason_involvement evidence claim for qualitative visibility only.
"""

import json
import math
import uuid
from datetime import date, datetime, timezone

import duckdb
import numpy as np
import pandas as pd

from . import evidence_blend as eb
from . import params as params_mod
from . import reconcile as reconcile_mod
from . import snapshot as snapshot_mod

PL = "Premier League"
EPS = 1e-6


def logit(p: float) -> float:
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------- historical fit ----

def _build_player_season_team_map(con: duckdb.DuckDBPyConnection, seasons: tuple[str, ...]) -> None:
    con.execute(
        "CREATE OR REPLACE TEMP TABLE _player_season_team (player_uid VARCHAR, season VARCHAR, team_uid VARCHAR)"
    )
    for season in seasons:
        found = reconcile_mod._season_root_table(con, season, "teams.csv")
        if not found:
            continue
        _relpath, table = found
        season_sql = season.replace("'", "''")
        con.execute(
            f"""
            INSERT INTO _player_season_team
            SELECT DISTINCT pa.player_uid, pa.season, ta.team_uid
            FROM player_alias pa
            JOIN "{table}" t ON t.code = pa.team_code
            JOIN team_alias ta ON ta.alias_name = t.name AND ta.season = pa.season
            WHERE pa.season = '{season_sql}'
            """
        )


def _team_match_weights(con: duckdb.DuckDBPyConnection, seasons: tuple[str, ...], asof_date: date, xi: float) -> pd.DataFrame:
    placeholders = ",".join(["?"] * len(seasons))
    df = con.execute(
        f"""
        SELECT home_team_uid AS team_uid, season, match_id, kickoff_time FROM fact_match
        WHERE competition = ? AND finished = TRUE AND season IN ({placeholders})
        UNION ALL
        SELECT away_team_uid AS team_uid, season, match_id, kickoff_time FROM fact_match
        WHERE competition = ? AND finished = TRUE AND season IN ({placeholders})
        """,
        [PL, *seasons, PL, *seasons],
    ).fetchdf()
    days_ago = (pd.Timestamp(asof_date) - pd.to_datetime(df["kickoff_time"])).dt.days.clip(lower=0)
    df["weight"] = np.exp(-xi * days_ago)
    return df


def compute_player_historical_components(
    con: duckdb.DuckDBPyConnection, seasons: tuple[str, ...], asof_date: date, xi: float
) -> pd.DataFrame:
    """Per player: weighted_total/weighted_starts (recency-weighted, for P_start_historical_own
    and the position-pooled average), plus raw (unweighted) counts for the
    conditional-on-not-starting sub-usage rate, which the spec doesn't ask to be time-decayed."""
    _build_player_season_team_map(con, seasons)
    weights_df = _team_match_weights(con, seasons, asof_date, xi)
    con.register("_team_match_weights_df", weights_df)
    try:
        df = con.execute(
            """
            SELECT
                pst.player_uid,
                sum(tmw.weight) AS weighted_total,
                sum(CASE WHEN pmst.start_min = 0 THEN tmw.weight ELSE 0 END) AS weighted_starts,
                sum(CASE WHEN pmst.minutes_played > 0 THEN 1 ELSE 0 END) AS competitive_matches,
                sum(CASE WHEN pmst.start_min = 0 THEN 1 ELSE 0 END) AS raw_starts,
                sum(CASE WHEN pmst.start_min > 0 AND pmst.minutes_played > 0 THEN 1 ELSE 0 END) AS raw_sub_appearances,
                count(*) AS raw_team_matches
            FROM _player_season_team pst
            JOIN _team_match_weights_df tmw ON tmw.team_uid = pst.team_uid AND tmw.season = pst.season
            LEFT JOIN fact_player_match_stats pmst
                ON pmst.player_uid = pst.player_uid AND pmst.match_id = tmw.match_id
            GROUP BY pst.player_uid
            """
        ).fetchdf()
    finally:
        con.unregister("_team_match_weights_df")
    return df


def compute_position_rates(con: duckdb.DuckDBPyConnection, per_player: pd.DataFrame) -> pd.DataFrame:
    positions = con.execute("SELECT player_uid, position FROM dim_player").fetchdf()
    df = per_player.merge(positions, on="player_uid", how="left")
    pos = df.groupby("position").agg(
        pos_weighted_starts=("weighted_starts", "sum"),
        pos_weighted_total=("weighted_total", "sum"),
        pos_raw_starts=("raw_starts", "sum"),
        pos_raw_sub_appearances=("raw_sub_appearances", "sum"),
        pos_raw_team_matches=("raw_team_matches", "sum"),
    )
    pos["p_start_historical_position_avg"] = pos.pos_weighted_starts / pos.pos_weighted_total
    not_started = pos.pos_raw_team_matches - pos.pos_raw_starts
    pos["p_used_as_sub_given_not_started"] = (pos.pos_raw_sub_appearances / not_started).where(not_started > 0, 0.0)
    return pos


def compute_conditional_minutes_rates(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Empirical P(60+ | started) and P(60+ | subbed on), per position. P(1-59 | .) is the
    complement in both cases -- a featuring player (started or subbed on) always has >0
    minutes by construction, so those two conditionals only ever split {1-59, 60+}."""
    return con.execute(
        """
        SELECT dp.position,
            avg(CASE WHEN pmst.start_min = 0 AND pmst.minutes_played >= 60 THEN 1.0
                     WHEN pmst.start_min = 0 THEN 0.0 END) AS p_60plus_given_started,
            avg(CASE WHEN pmst.start_min > 0 AND pmst.minutes_played > 0 AND pmst.minutes_played >= 60 THEN 1.0
                     WHEN pmst.start_min > 0 AND pmst.minutes_played > 0 THEN 0.0 END) AS p_60plus_given_subbed_on
        FROM fact_player_match_stats pmst
        JOIN dim_player dp ON dp.player_uid = pmst.player_uid
        GROUP BY dp.position
        """
    ).fetchdf().set_index("position")


# --------------------------------------------------------- evidence adjustment ----

_SHIFT_CLAIM_TYPES = ("injury_status", "manager_tendency", "transfer_likelihood")


def compute_logit_adjustment(
    con: duckdb.DuckDBPyConnection, player_uid: str, p_start_historical_final: float, asof: datetime,
    adjustment_params_version: int, decay_params_version: int, fact_multiplier_params_version: int,
) -> float:
    total = 0.0

    for claim_type in _SHIFT_CLAIM_TYPES:
        claims = snapshot_mod.get_claims_asof(
            con, asof, subject_entity_type="player", subject_entity_id=player_uid, claim_type=claim_type
        ).to_dict("records")
        for c in claims:
            payload = json.loads(c["claim_value"]) if c["claim_value"] else {}
            if claim_type == "transfer_likelihood" and payload.get("status") == "Complete":
                # "out-bound claims only" -- a completed move isn't an ongoing risk of
                # reduced minutes at the club they've already left.
                continue
            dims = {"claim_type": claim_type}
            category = payload.get("category")
            if category is not None:
                dims["category"] = category
            try:
                magnitude, _ = params_mod.resolve_param(
                    con, "minutes_adjustment_params", "magnitude", adjustment_params_version, dimensions=dims
                )
            except params_mod.ParamNotFoundError:
                continue
            if claim_type == "manager_tendency":
                sign = {"positive": 1, "negative": -1}.get(payload.get("valence"), 0)
                magnitude = magnitude * sign
            w = eb.effective_weight(con, c, asof, decay_params_version, fact_multiplier_params_version)
            total += magnitude * w

    claims = snapshot_mod.get_claims_asof(
        con, asof, subject_entity_type="player", subject_entity_id=player_uid, claim_type="predicted_xi"
    ).to_dict("records")
    if claims:
        try:
            pull_strength, _ = params_mod.resolve_param(
                con, "minutes_adjustment_params", "magnitude", adjustment_params_version,
                dimensions={"claim_type": "predicted_xi"},
            )
        except params_mod.ParamNotFoundError:
            pull_strength = None
        if pull_strength is not None:
            base_logit = logit(p_start_historical_final)
            for c in claims:
                if c["claim_value_numeric"] is None or pd.isna(c["claim_value_numeric"]):
                    continue
                w = eb.effective_weight(con, c, asof, decay_params_version, fact_multiplier_params_version)
                total += pull_strength * w * (logit(c["claim_value_numeric"]) - base_logit)

    try:
        cap, _ = params_mod.resolve_param(
            con, "minutes_adjustment_params", "cap", adjustment_params_version, dimensions={"scope": "global"}
        )
    except params_mod.ParamNotFoundError:
        cap = 6.0
    return max(-cap, min(cap, total))


# ============================================================
# M9 adapter -- evidence provenance
# ============================================================

def explain_player_adjustment(con: duckdb.DuckDBPyConnection, model_version: int, player_uid: str) -> list[dict]:
    """M9's evidence-provenance section: "for any player, the underlying evidence_claims
    behind their minutes/injury adjustment -- source, confidence, reliability tier,
    information_type, ... clearly labeled as not affecting the model's numbers" (for
    community/analyst evidence, which M1b never auto-consumes).

    compute_logit_adjustment() only ever returns the summed, capped float -- no per-claim
    contribution log exists anywhere else in this project. Rather than change that function's
    return type (would break every existing caller and test), this is a new, separate function
    with the *exact same* claim-filtering shape (same skip-completed-transfer check, same
    missing-param skip, same manager_tendency sign flip) -- kept side by side with the
    original specifically so the two are easy to keep in lockstep, not duplicated somewhere
    distant. Joins `sources` itself (get_claims_asof() doesn't) for source_name/source_type.
    Returns one row per claim *considered*, `included` distinguishing whether it actually
    contributed to the stored adjustment from whether it was merely logged and evaluated.
    """
    run_row = con.execute(
        "SELECT player_uid, p_start_historical_final, calibration_asof_date, adjustment_params_version, "
        "decay_params_version, fact_multiplier_params_version FROM minutes_model_outputs mo "
        "JOIN minutes_model_versions mv ON mv.model_version = mo.model_version "
        "WHERE mo.model_version = ? AND mo.player_uid = ?",
        [model_version, player_uid],
    ).fetchone()
    if not run_row:
        return []
    (_uid, p_start_historical_final, calibration_asof_date, adjustment_params_version,
     decay_params_version, fact_multiplier_params_version) = run_row
    asof = datetime.combine(calibration_asof_date, datetime.max.time(), tzinfo=timezone.utc)

    sources_by_id = {r[0]: (r[1], r[2]) for r in con.execute("SELECT source_id, source_name, source_type FROM sources").fetchall()}

    def _source_info(source_id):
        row = sources_by_id.get(source_id)
        return {"source_name": row[0], "source_type": row[1]} if row else {"source_name": None, "source_type": None}

    rows: list[dict] = []

    for claim_type in _SHIFT_CLAIM_TYPES:
        claims = snapshot_mod.get_claims_asof(
            con, asof, subject_entity_type="player", subject_entity_id=player_uid, claim_type=claim_type
        ).to_dict("records")
        for c in claims:
            payload = json.loads(c["claim_value"]) if c["claim_value"] else {}
            base = {
                "claim_id": c["claim_id"], "claim_type": claim_type, **_source_info(c["source_id"]),
                "information_type": c["information_type"], "confidence": c["confidence"],
                "reliability_score": c["source_reliability_score"], "observed_date": c["observed_date"],
                "raw_text": c["raw_text"], "reasoning": payload.get("reasoning"),
            }
            if claim_type == "transfer_likelihood" and payload.get("status") == "Complete":
                rows.append({**base, "included": False, "exclusion_reason": "transfer already completed"})
                continue
            dims = {"claim_type": claim_type}
            category = payload.get("category")
            if category is not None:
                dims["category"] = category
            try:
                magnitude, _ = params_mod.resolve_param(
                    con, "minutes_adjustment_params", "magnitude", adjustment_params_version, dimensions=dims
                )
            except params_mod.ParamNotFoundError:
                rows.append({**base, "included": False, "exclusion_reason": "no resolvable magnitude param"})
                continue
            if claim_type == "manager_tendency":
                sign = {"positive": 1, "negative": -1}.get(payload.get("valence"), 0)
                magnitude = magnitude * sign
            w = eb.effective_weight(con, c, asof, decay_params_version, fact_multiplier_params_version)
            rows.append({
                **base, "included": True, "exclusion_reason": None,
                "magnitude": magnitude, "weight": w, "contribution": magnitude * w,
            })

    claims = snapshot_mod.get_claims_asof(
        con, asof, subject_entity_type="player", subject_entity_id=player_uid, claim_type="predicted_xi"
    ).to_dict("records")
    if claims:
        try:
            pull_strength, _ = params_mod.resolve_param(
                con, "minutes_adjustment_params", "magnitude", adjustment_params_version,
                dimensions={"claim_type": "predicted_xi"},
            )
        except params_mod.ParamNotFoundError:
            pull_strength = None
        base_logit = logit(p_start_historical_final) if pull_strength is not None else None
        for c in claims:
            payload = json.loads(c["claim_value"]) if c["claim_value"] else {}
            base = {
                "claim_id": c["claim_id"], "claim_type": "predicted_xi", **_source_info(c["source_id"]),
                "information_type": c["information_type"], "confidence": c["confidence"],
                "reliability_score": c["source_reliability_score"], "observed_date": c["observed_date"],
                "raw_text": c["raw_text"], "reasoning": payload.get("reasoning"),
            }
            if pull_strength is None:
                rows.append({**base, "included": False, "exclusion_reason": "no resolvable magnitude param"})
                continue
            if c["claim_value_numeric"] is None or pd.isna(c["claim_value_numeric"]):
                rows.append({**base, "included": False, "exclusion_reason": "no numeric claim_value"})
                continue
            w = eb.effective_weight(con, c, asof, decay_params_version, fact_multiplier_params_version)
            contribution = pull_strength * w * (logit(c["claim_value_numeric"]) - base_logit)
            rows.append({**base, "included": True, "exclusion_reason": None, "weight": w, "contribution": contribution})

    return rows


# ------------------------------------------------------------- GW0 visibility ----

def log_preseason_involvement_claims(con: duckdb.DuckDBPyConnection, target_season: str) -> int:
    """GW0 friendly minutes never enter the quantitative fit (see module docstring) --
    logged here as low-weight, qualitative-only evidence claims instead (M2 spec).

    src_minutes_model_system_derived is this function's own source_id for these claims (they're
    derived from fact_player_match_stats, not pulled from any external outlet) -- the sources
    table's source_type CHECK constraint already anticipates 'system-derived' as a category, but
    nothing ever actually registered this row (a real, pre-existing gap: every other ingest
    module registers its own sources on the fly the same way, this one just never had one).
    Registered here, idempotently, the first time this function actually runs against real
    GW0 data rather than assuming it already exists.

    Deliberately NOT called 'src_system-derived' (as first written) -- ingest_workbook.py
    independently reserves that exact same source_id for its own, differently-scoped generic
    'system-derived' bucket (unrelated claims lacking a specific named source), a real
    same-name-different-owner collision only surfaced once a real run actually ingested both
    the main workbook and GW0 preseason data in the same pass. ON CONFLICT (source_name) DO
    NOTHING doesn't catch it either, since the two rows have different source_name values but
    the same derived source_id -- a source_id collision, not a source_name one."""
    con.execute(
        "INSERT INTO sources (source_id, source_name, source_type, base_reliability_score, "
        "citation_count, source_notes, last_reviewed_date) VALUES "
        "('src_minutes_model_system_derived', 'System-derived (fact_player_match_stats)', 'system-derived', 0.0, NULL, NULL, NULL) "
        "ON CONFLICT (source_name) DO NOTHING"
    )
    rows = con.execute(
        """
        SELECT pmst.player_uid, sum(pmst.minutes_played) AS total_minutes, count(*) AS appearances
        FROM fact_player_match_stats pmst
        JOIN fact_match m ON m.match_id = pmst.match_id
        WHERE m.season = ? AND m.gameweek = 0
        GROUP BY pmst.player_uid
        HAVING sum(pmst.minutes_played) IS NOT NULL
        """,
        [target_season],
    ).fetchall()
    ingested_date = datetime.now(timezone.utc)
    n = 0
    for player_uid, total_minutes, appearances in rows:
        con.execute(
            "INSERT INTO evidence_claims (claim_id, subject_entity_type, subject_entity_id, claim_type, "
            "claim_value, claim_value_numeric, information_type, source_id, source_reliability_score, "
            "confidence, observed_date, ingested_date, tab_origin, row_origin) "
            "VALUES (?, 'player', ?, 'preseason_involvement', ?, ?, 'FACT', 'src_minutes_model_system_derived', 0.0, "
            "0.3, ?, ?, 'fact_player_match_stats:GW0', NULL)",
            [
                str(uuid.uuid4()), player_uid,
                json.dumps({"total_preseason_minutes": int(total_minutes), "appearances": int(appearances)}),
                float(total_minutes), ingested_date.date(), ingested_date,
            ],
        )
        n += 1
    return n


# ------------------------------------------------------------------ orchestrator ----

def run(
    con: duckdb.DuckDBPyConnection,
    calibration_asof_date: date,
    target_season: str,
    decay_params_version: int,
    adjustment_params_version: int,
    shrinkage_params_version: int,
    fact_multiplier_params_version: int,
    lookback_seasons: tuple[str, ...] = ("2024-2025", "2025-2026"),
) -> int:
    xi, _ = params_mod.resolve_param(con, "minutes_model_decay_params", "xi", decay_params_version)
    threshold, _ = params_mod.resolve_param(
        con, "minutes_model_shrinkage_params", "competitive_matches_threshold", shrinkage_params_version
    )
    # end-of-day, not start-of-day: "as of this date" should include everything ingested
    # during that date, not just what existed at midnight before it (a claim ingested at
    # 09:34 on the asof date itself is legitimately knowable "as of" that date).
    asof = datetime.combine(calibration_asof_date, datetime.max.time(), tzinfo=timezone.utc)

    per_player = compute_player_historical_components(con, lookback_seasons, calibration_asof_date, xi)
    position_rates = compute_position_rates(con, per_player)  # merges position internally
    conditional_rates = compute_conditional_minutes_rates(con)

    target_players = con.execute(
        """
        SELECT DISTINCT dp.player_uid, dp.position
        FROM player_alias pa JOIN dim_player dp ON dp.player_uid = pa.player_uid
        WHERE pa.season = ?
        """,
        [target_season],
    ).fetchdf()

    per_player_idx = per_player.set_index("player_uid")

    model_version = con.execute(
        """
        INSERT INTO minutes_model_versions
            (calibration_asof_date, target_season, decay_params_version, adjustment_params_version,
             shrinkage_params_version, fact_multiplier_params_version, lookback_seasons)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        RETURNING model_version
        """,
        [calibration_asof_date, target_season, decay_params_version, adjustment_params_version,
         shrinkage_params_version, fact_multiplier_params_version, json.dumps(list(lookback_seasons))],
    ).fetchone()[0]

    for _, row in target_players.iterrows():
        player_uid, position = row["player_uid"], row["position"]
        pos_row = position_rates.loc[position] if position in position_rates.index else None
        p_start_pos_avg = float(pos_row["p_start_historical_position_avg"]) if pos_row is not None else 0.5
        p_sub_used = float(pos_row["p_used_as_sub_given_not_started"]) if pos_row is not None else 0.0

        if player_uid in per_player_idx.index:
            prow = per_player_idx.loc[player_uid]
            weighted_total = prow["weighted_total"] or 0.0
            p_start_own = float(prow["weighted_starts"] / weighted_total) if weighted_total > 0 else None
            competitive_matches = int(prow["competitive_matches"] or 0)
        else:
            p_start_own, competitive_matches = None, 0

        weight_own = min(1.0, competitive_matches / threshold) if p_start_own is not None else 0.0
        p_start_hist_final = weight_own * p_start_own + (1 - weight_own) * p_start_pos_avg if p_start_own is not None else p_start_pos_avg

        adjustment = compute_logit_adjustment(
            con, player_uid, p_start_hist_final, asof,
            adjustment_params_version, decay_params_version, fact_multiplier_params_version,
        )
        p_start_final = sigmoid(logit(p_start_hist_final) + adjustment)

        cond = conditional_rates.loc[position] if position in conditional_rates.index else None
        p_60_started = float(cond["p_60plus_given_started"]) if cond is not None and not pd.isna(cond["p_60plus_given_started"]) else 0.7
        p_60_subbed = float(cond["p_60plus_given_subbed_on"]) if cond is not None and not pd.isna(cond["p_60plus_given_subbed_on"]) else 0.1

        p_0 = (1 - p_start_final) * (1 - p_sub_used)
        p_60plus = p_start_final * p_60_started + (1 - p_start_final) * p_sub_used * p_60_subbed
        p_1_59 = 1.0 - p_0 - p_60plus  # by construction, not an independent third empirical estimate

        con.execute(
            """
            INSERT INTO minutes_model_outputs
                (model_version, player_uid, position, p_start_historical_own, p_start_historical_position_avg,
                 weight_own, p_start_historical_final, logit_adjustment_total, p_start_final,
                 p_used_as_sub_given_not_started, p_0min, p_1_59min, p_60plus_min, competitive_matches_last_2_seasons)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [model_version, player_uid, position, p_start_own, p_start_pos_avg, weight_own,
             p_start_hist_final, adjustment, p_start_final, p_sub_used, p_0, p_1_59, p_60plus, competitive_matches],
        )

    return model_version
