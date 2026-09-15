from datetime import datetime

from fpl_quant import field_rank

SEASON = "2026-2027"


def _seed_player(con, uid, name, position="Midfielder"):
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, ?)", [uid, name, position])


def _seed_stat(con, uid, gw, event_points, minutes=90):
    con.execute(
        "INSERT INTO fact_player_season_stats (player_uid, season, gw, minutes, event_points, _ingested_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [uid, SEASON, gw, minutes, event_points, datetime(2026, 9, 1)],
    )


def _seed_rival(con, entry_id, rank, picks, gw=1):
    """picks: [(player_uid, multiplier, is_captain), ...]."""
    for uid, mult, is_cap in picks:
        con.execute(
            "INSERT INTO fact_rival_squad_sample (entry_id, season, event, player_uid, is_captain, "
            "multiplier, league_rank, _ingested_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [entry_id, SEASON, gw, uid, is_cap, mult, rank, datetime(2026, 9, 1)],
        )


def _seed_scenario(con):
    for uid, name in [("p1", "One"), ("p2", "Two"), ("p3", "Three"), ("p4", "Four"),
                      ("p5", "Five"), ("p6", "Six"), ("p7", "Seven")]:
        _seed_player(con, uid, name)
    realized = {"p1": 2, "p2": 10, "p3": 6, "p4": 1, "p5": 8, "p6": 0, "p7": 15}
    for uid, pts in realized.items():
        _seed_stat(con, uid, 1, pts)
    # three rival squads: p1 is the field's captain (2 of 3), p1 owned by all three
    _seed_rival(con, 10, 5_000, [("p1", 2, True), ("p2", 1, False), ("p3", 1, False)])
    _seed_rival(con, 11, 90_000, [("p1", 1, False), ("p2", 2, True), ("p4", 1, False)])
    _seed_rival(con, 12, 500_000, [("p1", 2, True), ("p5", 1, False), ("p6", 1, False)])
    return realized


# ============================================================
# settled_rival_totals
# ============================================================

def test_settled_rival_totals_applies_the_multiplier(con):
    _seed_scenario(con)
    totals = {r["entry_id"]: r["points"] for r in field_rank.settled_rival_totals(con, SEASON, 1)}
    assert totals == {
        10: 2 * 2 + 10 + 6,   # p1 captained
        11: 2 + 10 * 2 + 1,   # p2 captained
        12: 2 * 2 + 8 + 0,    # p1 captained
    }


def test_settled_rival_totals_empty_when_no_sample(con):
    assert field_rank.settled_rival_totals(con, SEASON, 1) == []


# ============================================================
# field_ownership
# ============================================================

def test_field_ownership_effective_and_captaincy_rates(con):
    _seed_scenario(con)
    own = field_rank.field_ownership(con, SEASON, 1)
    assert own["p1"]["eo"] == 100.0                       # in all three XIs
    assert own["p1"]["captain_pct"] == round(200 / 3, 1)  # captained by 2 of 3
    assert own["p2"]["eo"] == round(200 / 3, 1)
    assert own["p3"]["owned_pct"] == round(100 / 3, 1)
    assert "p7" not in own                                 # nobody in the sample owns p7


# ============================================================
# realized_xi_points
# ============================================================

def test_realized_xi_points_doubles_the_captain(con):
    _seed_scenario(con)
    assert field_rank.realized_xi_points(con, SEASON, 1, ["p1", "p2"], "p1") == 2 + 10 + 2
    assert field_rank.realized_xi_points(con, SEASON, 1, ["p1", "p2"], "p2", captain_multiplier=3) == 2 + 10 + 10 * 2


# ============================================================
# score_squad_rank
# ============================================================

def test_score_squad_rank_places_the_squad_in_the_field(con):
    _seed_scenario(con)
    # XI p2..p6 + captain p2 -> 10*2 + 6 + 1 + 8 + 0 = 35; rivals scored 20 / 23 / 12
    score = field_rank.score_squad_rank(con, SEASON, 1, ["p2", "p3", "p4", "p5", "p6"], "p2", 1_000_000)
    assert score["realized_points"] == 35
    assert (score["n_rivals"], score["n_beaten"], score["n_tied"]) == (3, 3, 0)
    assert score["percentile"] == 100.0
    assert score["estimated_rank"] == 1


def test_score_squad_rank_no_sample_returns_zero_rivals(con):
    _seed_scenario(con)
    score = field_rank.score_squad_rank(con, SEASON, 7, ["p2"], "p2", 1_000_000)
    assert score["n_rivals"] == 0 and score["percentile"] is None


# ============================================================
# estimate_population_rank -- band-weighted, corrects for the stratified sample
# ============================================================

def _rivals(pairs):
    """pairs: [(league_rank, points), ...] -> settled_rival_totals()-shaped dicts."""
    return [{"entry_id": i, "league_rank": r, "points": p} for i, (r, p) in enumerate(pairs)]


def test_estimate_population_rank_beats_a_top_heavy_sample_by_more_than_the_flat_count():
    # a below-average score against a sample that is 4/6 elite: the flat count says it beat
    # ~1/3 of "the field" (really: 1/3 of a room of strong managers). Band-weighting knows the
    # two weak rivals stand in for millions of real managers, so the population percentile is
    # far higher and the projected rank far better.
    total = 10_000_000
    rivals = _rivals([(4_000, 82), (7_000, 78), (110_000, 61), (130_000, 59),
                      (1_400_000, 33), (1_900_000, 29)])
    est = field_rank.estimate_population_rank(48, rivals, total)
    assert est["method"] == "band_weighted"
    assert est["n_beaten"] == 2  # raw count unchanged: 48 only outscores the two tail rivals
    naive_pct = 2 / 6 * 100
    assert est["percentile"] > naive_pct + 20  # materially higher than the naive count
    assert est["estimated_rank"] < 0.5 * total  # nowhere near "last place"


def test_estimate_population_rank_is_monotonic_in_score():
    total = 5_000_000
    rivals = _rivals([(5_000, 70), (80_000, 55), (500_000, 44), (1_800_000, 30)])
    ranks = [field_rank.estimate_population_rank(s, rivals, total)["estimated_rank"] for s in (20, 40, 60, 90)]
    assert ranks == sorted(ranks, reverse=True)  # more points -> numerically smaller (better) rank


def test_estimate_population_rank_places_an_average_score_mid_field():
    # a realistic stratified sample (like DEFAULT_RANK_BANDS): ~64% of it in the top 140k, and
    # each band's scores straddle the subject's 50. An average score must land mid-field, not
    # near-last the way a flat count over this top-heavy sample would put it.
    total = 10_633_096
    rivals = _rivals(
        [(1_000 + 400 * i, 70 + 10 * (i % 2)) for i in range(40)]        # elite: 70/80
        + [(60_000 + 650 * i, 52 + 12 * (i % 2)) for i in range(120)]    # target: 52/64
        + [(400_000 + 5_000 * i, 44 + 12 * (i % 2)) for i in range(40)]  # mid-field: 44/56
        + [(600_000 + 28_000 * i, 40 + 20 * (i % 2)) for i in range(50)]  # tail: 40/60
    )
    est = field_rank.estimate_population_rank(50, rivals, total)
    assert est["method"] == "band_weighted"
    assert 30.0 < est["percentile"] < 70.0
    assert 3_000_000 < est["estimated_rank"] < 8_000_000
    # the naive flat count would be far more pessimistic (most of this sample outscores 50)
    flat_pct = est["n_beaten"] / est["sample_size"] * 100
    assert est["percentile"] > flat_pct


def test_estimate_population_rank_falls_back_to_flat_count_without_a_rank_axis():
    rivals = [{"entry_id": i, "league_rank": None, "points": p} for i, p in enumerate([30, 50, 70])]
    est = field_rank.estimate_population_rank(60, rivals, 1_000_000)
    assert est["method"] == "flat_count"
    assert est["percentile"] == round(2 / 3 * 100, 1)  # beat 2 of 3, no weighting possible


def test_estimate_population_rank_empty_sample():
    est = field_rank.estimate_population_rank(60, [], 1_000_000)
    assert est["percentile"] is None and est["estimated_rank"] is None and est["method"] is None


# ============================================================
# attribute_rank_gap
# ============================================================

def test_attribute_rank_gap_buckets(con):
    _seed_scenario(con)
    attr = field_rank.attribute_rank_gap(
        con, SEASON, 1, xi_uids=["p2", "p3", "p4", "p5", "p7"], captain_uid="p3", bench_uids=["p6"],
    )

    cap = attr["captaincy"]
    assert cap["captain_points"] == 6                       # p3 scored 6
    assert cap["field_captain"] == "One" and cap["field_captain_points"] == 2
    assert cap["vs_field_captain"] == 6 - 2
    assert cap["best_in_xi"] == "Seven" and cap["vs_best_in_xi"] == 6 - 15  # p7 hauled 15 in the XI

    tmpl = attr["template_coverage"]
    assert tmpl["template_size"] == 6                        # only six distinct sampled players
    assert tmpl["template_owned"] == 5                       # missed only p1
    assert [m["name"] for m in tmpl["missed"]] == ["One"]
    assert tmpl["eo_weighted_points_bled"] == -2.0           # 100% EO * 2 realized pts, signed negative

    diffs = attr["differentials"]
    assert diffs["n"] == 1 and diffs["players"][0]["name"] == "Seven"   # p7: 0% sampled ownership
    assert diffs["net_surprise"] == 0.0                      # no ep_model_version -> no surprise term

    assert attr["bench"]["points_left"] == 0                 # p6 scored 0


def test_attribute_rank_gap_bench_cost_is_signed_negative(con):
    _seed_scenario(con)
    attr = field_rank.attribute_rank_gap(
        con, SEASON, 1, xi_uids=["p2", "p3", "p4"], captain_uid="p2", bench_uids=["p5", "p6"],
    )
    assert attr["bench"]["points_left"] == -(8 + 0)


# ============================================================
# sample_shape
# ============================================================

def test_sample_shape_buckets_by_rank_band(con):
    _seed_scenario(con)
    shape = field_rank.sample_shape(con, SEASON, 1)
    assert shape["n_entries"] == 3
    assert shape["rank_min"] == 5_000 and shape["rank_max"] == 500_000
    assert shape["buckets"] == {"top_10k": 1, "60k_140k": 1, "400k_600k": 1}


def test_sample_shape_empty_without_a_sample(con):
    assert field_rank.sample_shape(con, SEASON, 1) == {}


# ============================================================
# gameweek_is_settled
# ============================================================

def test_gameweek_is_settled(con):
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('t1', 'Team One'), ('t2', 'Team Two')")
    con.execute(
        "INSERT INTO fact_match (match_id, season, gameweek, home_team_uid, away_team_uid, finished, _ingested_at) "
        "VALUES ('m1', ?, 1, 't1', 't2', TRUE, ?), ('m2', ?, 1, 't2', 't1', TRUE, ?), "
        "('m3', ?, 2, 't1', 't2', FALSE, ?)",
        [SEASON, datetime(2026, 9, 1)] * 3,
    )
    assert field_rank.gameweek_is_settled(con, SEASON, 1) is True
    assert field_rank.gameweek_is_settled(con, SEASON, 2) is False
    assert field_rank.gameweek_is_settled(con, SEASON, 3) is False   # no fixtures at all
