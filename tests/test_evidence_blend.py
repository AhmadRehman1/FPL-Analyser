import json
from datetime import datetime

from fpl_quant import evidence_blend as eb
from fpl_quant import params
from fpl_quant import snapshot


def _seed_player_and_sources(con):
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p1', 'Test Player', 'MID')")
    con.execute(
        "INSERT INTO sources (source_id, source_name, source_type, base_reliability_score) "
        "VALUES ('s_official', 'Official', 'official', 1.0)"
    )
    con.execute(
        "INSERT INTO sources (source_id, source_name, source_type, base_reliability_score) "
        "VALUES ('s_community', 'Community', 'community', 0.4)"
    )


def _insert_claim(con, claim_id, source_id, reliability, confidence, numeric=None, category=None,
                   information_type="OPINION", observed_date="2026-08-01"):
    value = json.dumps({"category": category}) if category else None
    con.execute(
        "INSERT INTO evidence_claims (claim_id, subject_entity_type, subject_entity_id, claim_type, "
        "claim_value, claim_value_numeric, information_type, source_id, source_reliability_score, "
        "confidence, observed_date, ingested_date) "
        "VALUES (?, 'player', 'p1', 'predicted_xi', ?, ?, ?, ?, ?, ?, ?, ?)",
        [claim_id, value, numeric, information_type, source_id, reliability, confidence,
         observed_date, datetime(2026, 8, 1)],
    )


def test_blend_numeric_weighted_average(con):
    _seed_player_and_sources(con)
    _insert_claim(con, "c1", "s_official", reliability=1.0, confidence=1.0, numeric=0.9)
    _insert_claim(con, "c2", "s_community", reliability=0.4, confidence=1.0, numeric=0.5)
    result = eb.blend_numeric(
        con, "player", "p1", "predicted_xi", datetime(2026, 8, 10),
        decay_params_version=1, fact_multiplier_params_version=1,
    )
    # weighted toward the higher-reliability official claim, strictly between the two inputs
    assert 0.5 < result < 0.9
    expected = (1.0 * 0.9 + 0.4 * 0.5) / (1.0 + 0.4)
    assert abs(result - expected) < 1e-9


def test_blend_numeric_no_evidence_returns_none(con):
    _seed_player_and_sources(con)
    result = eb.blend_numeric(
        con, "player", "p1", "predicted_xi", datetime(2026, 8, 10),
        decay_params_version=1, fact_multiplier_params_version=1,
    )
    assert result is None


def test_blend_categorical_distribution_sums_to_one(con):
    _seed_player_and_sources(con)
    _insert_claim(con, "c1", "s_official", reliability=1.0, confidence=1.0, category="Out")
    _insert_claim(con, "c2", "s_community", reliability=0.4, confidence=1.0, category="Doubtful")
    dist = eb.blend_categorical(
        con, "player", "p1", "predicted_xi", "category", datetime(2026, 8, 10),
        decay_params_version=1, fact_multiplier_params_version=1,
    )
    assert abs(sum(dist.values()) - 1.0) < 1e-9
    assert dist["Out"] > dist["Doubtful"]  # higher-reliability source should dominate


def test_fact_from_high_tier_source_gets_multiplier_boost(con):
    _seed_player_and_sources(con)
    params.write_param(con, "fact_type_multiplier_params", 1, "2026-08-10", "multiplier", value_numeric=1.5)
    _insert_claim(con, "c1", "s_official", reliability=1.0, confidence=1.0, numeric=0.8, information_type="FACT")
    boosted = eb.effective_weight(
        con,
        {"source_reliability_score": 1.0, "confidence": 1.0, "observed_date": None,
         "claim_type": "predicted_xi", "information_type": "FACT", "source_id": "s_official"},
        datetime(2026, 8, 10), decay_params_version=1, fact_multiplier_params_version=1,
    )
    assert boosted == 1.5  # 1.0 reliability * 1.0 confidence * 1.0 decay * 1.5 multiplier


def test_opinion_never_gets_fact_multiplier(con):
    _seed_player_and_sources(con)
    params.write_param(con, "fact_type_multiplier_params", 1, "2026-08-10", "multiplier", value_numeric=1.5)
    w = eb.effective_weight(
        con,
        {"source_reliability_score": 1.0, "confidence": 1.0, "observed_date": None,
         "claim_type": "predicted_xi", "information_type": "OPINION", "source_id": "s_official"},
        datetime(2026, 8, 10), decay_params_version=1, fact_multiplier_params_version=1,
    )
    assert w == 1.0


def test_null_confidence_via_dataframe_round_trip_defaults_to_1_not_nan(con):
    """Regression test for a real bug: a SQL NULL confidence comes back from
    snapshot.get_claims_asof (a pandas DataFrame) as float NaN, not Python None. Checking
    `is not None` doesn't catch it, and an unnoticed NaN silently poisons the whole
    weighted sum -- effective_weight must actually default it to 1.0, not stay NaN."""
    _seed_player_and_sources(con)
    _insert_claim(con, "c1", "s_official", reliability=1.0, confidence=None, numeric=0.9)
    claims = snapshot.get_claims_asof(con, datetime(2026, 8, 10), subject_entity_type="player", subject_entity_id="p1")
    assert len(claims) == 1
    claim_row = claims.to_dict("records")[0]
    w = eb.effective_weight(con, claim_row, datetime(2026, 8, 10), decay_params_version=1, fact_multiplier_params_version=1)
    assert w == 1.0  # reliability(1.0) * confidence(defaulted to 1.0) * decay(1.0)
    assert w == w  # not NaN (NaN != NaN)


# ============================================================
# Part 3 (squad-quality guardrails work): explain_analyst_debate_divergence /
# explain_community_evidence_roundup
# ============================================================

def _insert_debate_claim(con, claim_id, subject_uid, source_id, reliability, confidence, opinion, reason,
                          strength, row_origin, tab_origin="15_Analyst Debate Database", observed_date="2026-08-01"):
    value = json.dumps({"opinion": opinion, "reason": reason, "strength": strength})
    con.execute(
        "INSERT INTO evidence_claims (claim_id, subject_entity_type, subject_entity_id, claim_type, "
        "claim_value, claim_value_numeric, information_type, source_id, source_reliability_score, "
        "confidence, observed_date, ingested_date, tab_origin, row_origin) "
        "VALUES (?, 'player', ?, 'analyst_debate', ?, NULL, 'OPINION', ?, ?, ?, ?, ?, ?, ?)",
        [claim_id, subject_uid, value, source_id, reliability, confidence, observed_date,
         datetime(2026, 8, 1), tab_origin, row_origin],
    )


def _insert_roundup_claim(con, claim_id, subject_uid, claim_type, source_id, reliability, confidence, payload,
                           observed_date="2026-08-01"):
    con.execute(
        "INSERT INTO evidence_claims (claim_id, subject_entity_type, subject_entity_id, claim_type, "
        "claim_value, claim_value_numeric, information_type, source_id, source_reliability_score, "
        "confidence, observed_date, ingested_date) "
        "VALUES (?, 'player', ?, ?, ?, NULL, 'OPINION', ?, ?, ?, ?, ?)",
        [claim_id, subject_uid, claim_type, json.dumps(payload), source_id, reliability, confidence,
         observed_date, datetime(2026, 8, 1)],
    )


def test_explain_analyst_debate_divergence_pairs_both_sides_of_the_same_row(con):
    """The real structural link is (tab_origin, row_origin) shared by two claims tagged to two
    different players -- confirms both sides come back with their raw opinion text and a real
    weight difference driven by source reliability, nothing parsed from the text itself."""
    _seed_player_and_sources(con)
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p2', 'Other Player', 'FWD')")
    _insert_debate_claim(con, "d1a", "p1", "s_official", 1.0, 1.0, "Favours p1", "reason A", "strong", row_origin=5)
    _insert_debate_claim(con, "d1b", "p2", "s_community", 0.4, 1.0, "Favours p2", "reason B", "weak", row_origin=5)

    result = eb.explain_analyst_debate_divergence(
        con, "p1", datetime(2026, 8, 10), decay_params_version=1, fact_multiplier_params_version=1,
    )
    assert len(result) == 1
    sides_by_uid = {s["player_uid"]: s for s in result[0]["sides"]}
    assert set(sides_by_uid) == {"p1", "p2"}
    assert sides_by_uid["p1"]["opinion"] == "Favours p1"
    assert sides_by_uid["p2"]["opinion"] == "Favours p2"
    assert sides_by_uid["p1"]["weight"] > sides_by_uid["p2"]["weight"]  # higher-reliability source


def test_explain_analyst_debate_divergence_empty_when_player_has_no_debates(con):
    _seed_player_and_sources(con)
    result = eb.explain_analyst_debate_divergence(
        con, "p1", datetime(2026, 8, 10), decay_params_version=1, fact_multiplier_params_version=1,
    )
    assert result == []


def test_explain_analyst_debate_divergence_skips_a_single_sided_row(con):
    """A debate row with only one surviving claim (e.g. the other side got superseded/never
    resolved to a real player) isn't a real two-sided comparison -- must not be surfaced as one."""
    _seed_player_and_sources(con)
    _insert_debate_claim(con, "d2a", "p1", "s_official", 1.0, 1.0, "solo", "reason", "strength", row_origin=12)
    result = eb.explain_analyst_debate_divergence(
        con, "p1", datetime(2026, 8, 10), decay_params_version=1, fact_multiplier_params_version=1,
    )
    assert result == []


def test_explain_community_evidence_roundup_returns_verbatim_payloads_across_claim_types(con):
    """community_sentiment and youtube_evidence have different payload shapes -- confirms
    nothing here picks out specific keys and silently drops the rest."""
    _seed_player_and_sources(con)
    _insert_roundup_claim(
        con, "c1", "p1", "community_sentiment", "s_community", 0.4, 1.0,
        {"claim": "budget pick", "category": "Player Pick"},
    )
    _insert_roundup_claim(
        con, "c2", "p1", "youtube_evidence", "s_official", 1.0, 1.0,
        {"claim": "explosive underlying numbers", "supporting_data": "22 big chances"},
    )
    result = eb.explain_community_evidence_roundup(con, "p1", datetime(2026, 8, 10))
    assert len(result) == 2
    by_type = {r["claim_type"]: r for r in result}
    assert by_type["community_sentiment"]["payload"]["category"] == "Player Pick"
    assert by_type["youtube_evidence"]["payload"]["supporting_data"] == "22 big chances"


def test_explain_community_evidence_roundup_empty_when_no_claims(con):
    _seed_player_and_sources(con)
    result = eb.explain_community_evidence_roundup(con, "p1", datetime(2026, 8, 10))
    assert result == []


def test_community_tier_fact_gets_no_boost_even_if_flagged_fact(con):
    _seed_player_and_sources(con)
    params.write_param(con, "fact_type_multiplier_params", 1, "2026-08-10", "multiplier", value_numeric=1.5)
    w = eb.effective_weight(
        con,
        {"source_reliability_score": 0.4, "confidence": 1.0, "observed_date": None,
         "claim_type": "predicted_xi", "information_type": "FACT", "source_id": "s_community"},
        datetime(2026, 8, 10), decay_params_version=1, fact_multiplier_params_version=1,
    )
    assert w == 0.4  # community isn't high-tier -- no multiplier applied
