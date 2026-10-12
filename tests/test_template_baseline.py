"""The availability-gated ownership template, the walk-forward's model-free yardstick
(docs/plans/2026-10_beat_the_template.md, Req 1 and Req 2)."""

from datetime import datetime

import pytest

from fpl_quant import backtest as bt
from tests.test_backtest import _WALK_KWARGS, _seed_season_simulation_league

SEASON = "2025-2026"

# 18 players, one per club so the club cap never binds, all at 5.0m so the budget never binds:
# the template is then simply the most-owned players under the position quotas.
OWNERSHIP = {
    "g1": ("Goalkeeper", 30.0), "g2": ("Goalkeeper", 10.0),
    "d1": ("Defender", 40.0), "d2": ("Defender", 35.0), "d3": ("Defender", 20.0),
    "d4": ("Defender", 15.0), "d5": ("Defender", 12.0), "d6": ("Defender", 5.0),
    "m1": ("Midfielder", 60.0), "m2": ("Midfielder", 50.0), "m3": ("Midfielder", 25.0),
    "m4": ("Midfielder", 18.0), "m5": ("Midfielder", 14.0), "m6": ("Midfielder", 3.0),
    "f1": ("Forward", 55.0), "f2": ("Forward", 45.0), "f3": ("Forward", 8.0), "f4": ("Forward", 2.0),
}


def _seed_template_league(con, status_rows=None):
    """GW3 of 2025-26 scheduled (asof_scope needs its deadline); every player has a row at gw 1,
    2 and 3 with the same ownership. status_rows: {(uid, gw): status} overrides (default 'a')."""
    status_rows = status_rows or {}
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('team_a', 'A'), ('team_b', 'B')")
    for gw in (1, 2, 3):
        con.execute(
            "INSERT INTO fact_match (match_id, season, gameweek, home_team_uid, away_team_uid, finished, competition, "
            "kickoff_time, _ingested_at) VALUES (?, ?, ?, 'team_a', 'team_b', TRUE, 'Premier League', ?, ?)",
            [f"m{gw}", SEASON, gw, datetime(2025, 8, 7 * gw, 15), datetime.now()],
        )
    for i, (uid, (position, sbp)) in enumerate(sorted(OWNERSHIP.items()), start=1):
        con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, ?)", [uid, uid, position])
        con.execute(
            "INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) "
            "VALUES (?, ?, ?, ?, ?)", [uid, uid, str(i), SEASON, uid],
        )
        for gw in (1, 2, 3):
            con.execute(
                "INSERT INTO fact_player_season_stats (player_uid, season, gw, now_cost, selected_by_percent, status, "
                "event_points, _ingested_at) VALUES (?, ?, ?, 5.0, ?, ?, 2, current_timestamp)",
                [uid, SEASON, gw, sbp, status_rows.get((uid, gw), "a")],
            )


def _templates(con, gameweek=3):
    with bt.asof_scope(con, SEASON, gameweek):
        return bt.template_squads(con, SEASON, guardrail_cap=3.0)


def test_template_is_the_most_owned_legal_squad(con):
    _seed_template_league(con)
    t = _templates(con)
    gated = t["gated"]
    assert gated["squad"] == {"g1", "g2", "d1", "d2", "d3", "d4", "d5", "m1", "m2", "m3", "m4", "m5", "f1", "f2", "f3"}
    # the ten most-owned outfield squad players form a legal 4-4-2
    assert gated["xi"] == {"g1", "d1", "d2", "d3", "d4", "m1", "m2", "m3", "m4", "f1", "f2"}
    assert t["ungated"]["squad"] == gated["squad"]
    assert t["n_unavailable"] == 0


def test_template_captain_is_the_most_owned_outfield_xi_player(con):
    # A5: mu is ownership and the captain counts double, so the solve captains the top-owned
    # outfield player -- never the goalkeeper, even one owned more than anyone
    _seed_template_league(con)
    con.execute("UPDATE fact_player_season_stats SET selected_by_percent = 90.0 WHERE player_uid = 'g1'")
    gated = _templates(con)["gated"]
    assert gated["captain"] == "m1"
    assert gated["vice"] in gated["xi"] and gated["vice"] != "m1"


def test_a_player_suspended_before_the_deadline_is_gated_out(con):
    _seed_template_league(con, {("m1", 2): "s"})
    t = _templates(con)
    assert "m1" not in t["gated"]["squad"]
    assert "m6" in t["gated"]["squad"]
    assert t["gated"]["captain"] == "f1"
    assert t["n_unavailable"] == 1
    # the ungated template still starts him
    assert "m1" in t["ungated"]["xi"] and t["ungated"]["captain"] == "m1"


@pytest.mark.parametrize("status", ["i", "s", "u", "n", "I"])
def test_every_unavailable_code_is_gated_out(con, status):
    _seed_template_league(con, {("f1", 2): status})
    assert "f1" not in _templates(con)["gated"]["squad"]


def test_the_gate_reads_only_rows_before_the_deadline(con):
    # a suspension first recorded in row G is not knowable at G's deadline
    _seed_template_league(con, {("m1", 3): "s"})
    assert "m1" in _templates(con, gameweek=3)["gated"]["xi"]


def test_the_gate_uses_the_latest_row_and_a_missing_status_never_excludes(con):
    # injured at gw 1 but available again at gw 2; d1's latest status is missing
    _seed_template_league(con, {("m1", 1): "i", ("d1", 1): "i", ("d1", 2): None})
    xi = _templates(con)["gated"]["xi"]
    assert {"m1", "d1"} <= xi


def test_doubtful_players_stay_in(con):
    _seed_template_league(con, {("m1", 2): "d"})
    assert "m1" in _templates(con)["gated"]["xi"]


def test_template_is_none_when_the_pool_cannot_fill_a_squad(con):
    # GW1: no row before the deadline, so no ownership to pick from
    _seed_template_league(con)
    t = _templates(con, gameweek=1)
    assert t["gated"] is None and t["ungated"] is None


# ============================================================
# the walk-forward records the template every step it is asked to
# ============================================================

def _seed_owned_league(con):
    _seed_season_simulation_league(con)
    # distinct ownership so the template has no ties
    con.execute(
        "UPDATE fact_player_season_stats SET selected_by_percent = "
        "10.0 + now_cost * 3 + CAST(substr(player_uid, 2) AS INTEGER) * 0.1"
    )


def _metrics(con, run_id):
    rows = con.execute(
        "SELECT season, gameweek, metric_name, metric_value FROM backtest_metrics WHERE backtest_run_id = ?", [run_id],
    ).fetchall()
    out: dict = {}
    for season, gw, name, value in rows:
        out.setdefault((season, gw), {})[name] = value
    return out


TEMPLATE_METRICS = {
    "template_squad_realized_points", "template_squad_realized_points_ungated", "beats_template_points_delta",
    "template_xi_overlap", "non_template_points", "model_captain_points", "template_captain_points",
    "captain_ep_plus_eo_points", "model_xi_resid", "template_xi_resid", "template_solve_seconds",
}


def test_walk_forward_records_the_template_every_scored_step(tmp_path):
    from fpl_quant import db

    con = db.connect(tmp_path / "t.duckdb")
    _seed_owned_league(con)
    run_id = bt.run(con, n_antithetic_pairs=200, run_monte_carlo=False, seasons=(SEASON,), ownership_params_version=1,
                    template_baseline=True, **_WALK_KWARGS)
    by_step = _metrics(con, run_id)
    scored = [step for step, m in by_step.items() if "model_squad_realized_points" in m]
    assert len(scored) > 2
    for step in scored:
        m = by_step[step]
        assert TEMPLATE_METRICS <= set(m), step
        assert m["beats_template_points_delta"] == pytest.approx(
            m["model_squad_realized_points"] - m["template_squad_realized_points"]
        )
        assert 0 <= m["template_xi_overlap"] <= 11
        assert m["non_template_points"] <= m["model_squad_realized_points"] + 1e-9
        bands = [m[f"model_xi_spend:price_band={b}"] for b in ("<5.0", "5.0-7.0", "7.0-9.0", "9.0+")]
        assert 0 < sum(bands) <= 100.0
    con.close()


def test_the_template_leaves_every_model_metric_unchanged(tmp_path):
    from fpl_quant import db

    runs = {}
    for template_baseline in (False, True):
        con = db.connect(tmp_path / f"{template_baseline}.duckdb")
        _seed_owned_league(con)
        run_id = bt.run(con, n_antithetic_pairs=200, run_monte_carlo=False, seasons=(SEASON,),
                        ownership_params_version=1, template_baseline=template_baseline, **_WALK_KWARGS)
        runs[template_baseline] = _metrics(con, run_id)
        con.close()
    without, with_template = runs[False], runs[True]
    assert without.keys() == with_template.keys()
    for step, m in without.items():
        assert not (set(m) & TEMPLATE_METRICS)
        assert {k: v for k, v in with_template[step].items() if k in m} == m


def test_the_scorer_counts_points_the_gap_overlap_and_non_template_points(con):
    """Unit check of the scorer on a hand-built model squad and template: points, the gap, the
    overlap and the points from outside the template XI."""
    _seed_template_league(con)
    model = {"xi": frozenset({"g1", "d1", "d2", "d3", "m1", "m2", "m3", "m6", "f1", "f2", "f4"}),
             "captain": "f4", "contributions": {}}
    template = _templates(con)
    # every player scores 2; the model captains f4 who is not in the template
    model["contributions"] = {uid: (4.0 if uid == "f4" else 2.0) for uid in model["xi"]}
    run_id = con.execute("INSERT INTO backtest_runs (warm_up_gameweeks) VALUES (0) RETURNING backtest_run_id").fetchone()[0]
    bt._score_template_baseline(con, run_id, SEASON, 3, "warm", template, 999, None, model)  # no EP rows
    m = _metrics(con, run_id)[(SEASON, 3)]
    # template: 11 starters x 2, captain doubled; nobody blanked (minutes unknown), so no subs
    assert m["template_squad_realized_points"] == pytest.approx(24.0)
    assert m["beats_template_points_delta"] == pytest.approx(24.0 - 24.0)
    assert m["template_xi_overlap"] == 9  # all but the model's m6 and f4 (the template has d4 and m4)
    assert m["non_template_points"] == pytest.approx(2.0 + 4.0)  # m6 and the captained f4
    assert m["model_captain_points"] == 2 and m["template_captain_points"] == 2
    assert m["template_xi_spend:price_band=5.0-7.0"] == pytest.approx(55.0)
    assert "captain_ep_plus_eo_points" not in m  # needs ownership params
