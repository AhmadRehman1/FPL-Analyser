import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import rank_autopsy as ra  # noqa: E402

SEASON = "2026-2027"


# ============================================================
# pure: finding() -- picks the biggest per-gameweek leak
# ============================================================

def _row(**over):
    base = {
        "gameweek": 3, "realized_points": 40, "percentile": 55.0, "estimated_rank": 4_400_000,
        "n_rivals": 200, "n_beaten": 110,
        "attribution": {
            "captaincy": {"captain": "X", "captain_points": 2, "best_in_xi": "Y", "vs_best_in_xi": 0.0,
                          "vs_field_captain": 0.0},
            "template_coverage": {"template_owned": 15, "template_size": 15, "missed": [],
                                  "eo_weighted_points_bled": 0.0},
            "differentials": {"n": 0, "net_surprise": 0.0, "players": []},
            "bench": {"points_left": 0.0},
        },
    }
    base["attribution"].update(over.pop("attribution", {}))
    base.update(over)
    return base


def test_finding_flags_captaincy_as_the_biggest_leak():
    row = _row(attribution={"captaincy": {
        "captain": "Haaland", "captain_points": 2, "best_in_xi": "Salah", "vs_best_in_xi": -9.0,
        "vs_field_captain": -4.0,
    }})
    line = ra.finding("model optimal", 3, row)
    assert "captaincy" in line and "Haaland" in line and "Salah" in line
    assert "projected rank ~4,400,000" in line


def test_finding_flags_template_when_that_is_worst():
    row = _row(attribution={"template_coverage": {
        "template_owned": 11, "template_size": 15,
        "missed": [{"name": "Palmer", "eo": 67.0, "points": 13, "eo_weighted_points": 8.71}],
        "eo_weighted_points_bled": -8.71,
    }})
    line = ra.finding("model optimal", 3, row)
    assert "template" in line and "Palmer" in line and "11/15" in line


def test_finding_no_dominant_leak():
    assert "No single dominant leak" in ra.finding("model optimal", 3, _row())


# ============================================================
# pure: roll_up() -- season-to-date sum of the additive buckets
# ============================================================

def test_roll_up_sums_signed_buckets_across_gameweeks():
    by_gw = [
        _row(gameweek=1, attribution={
            "captaincy": {"vs_best_in_xi": -9.0, "vs_field_captain": -4.0},
            "bench": {"points_left": -2.0},
        }),
        _row(gameweek=2, attribution={
            "captaincy": {"vs_best_in_xi": 3.0, "vs_field_captain": 1.0},
            "bench": {"points_left": -5.0},
        }),
    ]
    cumulative = ra.roll_up(by_gw)
    assert cumulative["captaincy_vs_best_in_xi"] == -6.0
    assert cumulative["bench_points_left"] == -7.0
    assert cumulative["template_eo_weighted_points_bled"] == 0.0


# ============================================================
# DB: score_subject + run_autopsy against a seeded field
# ============================================================

def _seed_player(con, uid, name):
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, 'Midfielder')", [uid, name])


def _seed_settled_gameweek(con, gw):
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES (?, ?) ON CONFLICT DO NOTHING",
                [f"t{gw}a", f"Team {gw}A"])
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES (?, ?) ON CONFLICT DO NOTHING",
                [f"t{gw}b", f"Team {gw}B"])
    con.execute(
        "INSERT INTO fact_match (match_id, season, gameweek, home_team_uid, away_team_uid, finished, _ingested_at) "
        "VALUES (?, ?, ?, ?, ?, TRUE, ?)",
        [f"m{gw}", SEASON, gw, f"t{gw}a", f"t{gw}b", datetime(2026, 9, 1)],
    )


def _seed_field(con, gw=1):
    for uid, name, pts in [("a", "Ay", 2), ("b", "Bee", 12), ("c", "Cee", 6), ("d", "Dee", 1), ("e", "Ee", 9)]:
        _seed_player(con, uid, name)
        con.execute(
            "INSERT INTO fact_player_season_stats (player_uid, season, gw, minutes, event_points, _ingested_at) "
            "VALUES (?, ?, ?, 90, ?, ?)", [uid, SEASON, gw, pts, datetime(2026, 9, 1)],
        )
    _seed_settled_gameweek(con, gw)
    for entry_id, rank, picks in [
        (1, 4_000, [("a", 2, True), ("b", 1, False), ("c", 1, False)]),
        (2, 95_000, [("a", 1, False), ("b", 2, True), ("d", 1, False)]),
        (3, 480_000, [("a", 2, True), ("e", 1, False), ("c", 1, False)]),
    ]:
        for uid, mult, is_cap in picks:
            con.execute(
                "INSERT INTO fact_rival_squad_sample (entry_id, season, event, player_uid, is_captain, "
                "multiplier, league_rank, _ingested_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [entry_id, SEASON, gw, uid, is_cap, mult, rank, datetime(2026, 9, 1)],
            )


def test_score_subject_persists_and_returns_a_row(con):
    _seed_field(con)
    squad = {"subject_id": "999", "xi": ["b", "c", "d", "e"], "bench": ["a"], "captain": "b", "ep_model_version": None}
    row = ra.score_subject(con, 1, "model_optimal", squad, None, 1_000_000)

    assert row["realized_points"] == 12 * 2 + 6 + 1 + 9   # b captained
    assert row["n_rivals"] == 3
    stored = con.execute(
        "SELECT percentile, realized_points, n_beaten FROM fact_squad_rank_score "
        "WHERE season = ? AND event = 1 AND subject = 'model_optimal' AND subject_id = '999'", [SEASON],
    ).fetchone()
    assert stored == (row["percentile"], row["realized_points"], row["n_beaten"])


def test_score_subject_is_idempotent(con):
    _seed_field(con)
    squad = {"subject_id": "999", "xi": ["b", "c"], "bench": [], "captain": "b", "ep_model_version": None}
    ra.score_subject(con, 1, "model_optimal", squad, None, 1_000_000)
    ra.score_subject(con, 1, "model_optimal", squad, None, 1_000_000)
    assert con.execute("SELECT count(*) FROM fact_squad_rank_score").fetchone()[0] == 1


def test_score_subject_none_without_a_settled_sample(con):
    _seed_field(con, gw=1)
    squad = {"subject_id": "999", "xi": ["b"], "bench": [], "captain": "b", "ep_model_version": None}
    assert ra.score_subject(con, 7, "model_optimal", squad, None, 1_000_000) is None


def test_run_autopsy_assembles_accounts_and_rolls_up(con):
    _seed_field(con)
    squads = {
        7139944: {"subject_id": "7139944", "xi": ["a", "c", "d", "e"], "bench": ["b"], "captain": "a", "ep_model_version": None},
        1305242: {"subject_id": "1305242", "xi": ["b", "c", "d", "e"], "bench": ["a"], "captain": "b", "ep_model_version": None},
    }
    payload = ra.run_autopsy(
        con, total_players=1_000_000, element_names={},
        account_squad_fn=lambda eid, ev: squads.get(eid),
        account_rank_fn=lambda eid, ev: {7139944: 250_000, 1305242: 120_000}[eid],
    )

    assert payload["gameweeks_scored"] == [1]
    assert set(payload["subjects"]) == {"account:7139944", "account:1305242"}
    acct = payload["subjects"]["account:7139944"]
    assert acct["by_gameweek"][0]["real_overall_rank"] == 250_000
    assert "cumulative_attribution" in acct and acct["latest_percentile"] is not None
    assert payload["sample_shape_by_gameweek"]["1"]["n_entries"] == 3
    assert len(payload["findings"]) == 2
