"""The season simulation must give the same answer twice (docs/reports/2026-10_open_issues.md,
issue 2). Two local runs of the same code on the same DB produced different squads, chips and
transfers. Causes found and pinned here:

- DuckDB with more than one thread returns DISTINCT / GROUP BY / join rows in a different order
  each run, and sums floats over more than one row group in a different order; the pipeline
  feeds both into order-sensitive arithmetic and a MILP with many tied optima. db.connect() now
  pins one thread.
- monte_carlo.run() seeded its draws on its own model_version and the squad run id, DB sequence
  numbers that differ in a DB with a different history.
- rankings broke ties by set iteration order, which changes with PYTHONHASHSEED.
"""

import itertools
import random

from fpl_quant import backtest as bt
from fpl_quant import db
from fpl_quant import expected_points as ep
from fpl_quant import transfer_planner as tp
from tests.test_backtest import _SEASON_SIM_VERSIONS, _seed_season_simulation_league


def test_connect_pins_one_duckdb_thread_by_default(tmp_path):
    con = db.connect(tmp_path / "a.duckdb")
    assert con.execute("SELECT current_setting('threads')").fetchone()[0] == 1
    con.close()
    con = db.connect(tmp_path / "b.duckdb", threads=None)
    assert con.execute("SELECT current_setting('threads')").fetchone()[0] >= 1
    con.close()


def test_plackett_luce_does_not_depend_on_dict_order():
    rng = random.Random(3)
    strengths = {f"p{i}": rng.uniform(0.01, 5.0) for i in range(22)}
    reference = ep.plackett_luce_rank_distribution(strengths)
    keys = list(strengths)
    for _ in range(5):
        rng.shuffle(keys)
        shuffled = {k: strengths[k] for k in keys}
        assert ep.plackett_luce_rank_distribution(shuffled) == reference  # bit for bit


def _fingerprint(con, result):
    mc = con.execute(
        "SELECT r.query_id, s.player_uid, s.mean_total, s.var_total FROM monte_carlo_player_summary s "
        "JOIN monte_carlo_run_versions r ON r.model_version = s.model_version ORDER BY r.query_id, s.player_uid"
    ).fetchall()
    return {
        "weekly_points": result["weekly_points"],
        "actions": [(a["gameweek"], a["accepted_transfer_rank"], a["accepted_chip"]) for a in result["actions"]],
        "final_squad": sorted(h["player_uid"] for h in tp._read_holdings(con, result["final_state_version"])),
        "mc": mc,
    }


def test_season_simulation_repeats_exactly_in_a_db_with_a_different_history(tmp_path):
    """The second DB has already used up some model-version and squad-run ids, as a DB that ran
    an earlier simulation has. The draws and decisions must not change with them."""
    fingerprints = []
    for name, burn_ids in (("fresh", 0), ("used", 7)):
        con = db.connect(tmp_path / f"{name}.duckdb")
        _seed_season_simulation_league(con)
        for _ in range(burn_ids):
            con.execute("SELECT nextval('seq_monte_carlo_model_version')").fetchone()
            con.execute("SELECT nextval('seq_squad_optimizer_run')").fetchone()
        result = bt.run_season_simulation(
            con, "2025-2026", start_gameweek=2, end_gameweek=4, n_antithetic_pairs=200, **_SEASON_SIM_VERSIONS,
        )
        fingerprints.append(_fingerprint(con, result))
        con.close()
    assert fingerprints[0] == fingerprints[1]
    assert fingerprints[0]["mc"], "the walk ran Monte Carlo (Triple Captain) at least once"


def test_evaluate_transfers_breaks_ties_by_player_uid_not_holding_order(con, monkeypatch):
    """Two players worth exactly the same (both injured, 0 EP): which one the #1 transfer sells
    must not depend on the order the holdings came in."""
    horizon = {
        "out_b": {"total_ep": 0.0, "position": "Midfielder", "club": "A", "price": 6.0, "per_gw": {3: 0.0}},
        "out_a": {"total_ep": 0.0, "position": "Midfielder", "club": "B", "price": 6.0, "per_gw": {3: 0.0}},
        "keeper": {"total_ep": 5.0, "position": "Goalkeeper", "club": "C", "price": 4.5, "per_gw": {3: 5.0}},
        "in_x": {"total_ep": 9.0, "position": "Midfielder", "club": "D", "price": 6.0, "per_gw": {3: 9.0}},
    }
    monkeypatch.setattr(tp, "_horizon_ep_by_player", lambda *a, **k: horizon)
    holdings = [{"player_uid": uid, "in_xi": True, "is_captain": False, "is_vice": False} for uid in ("out_b", "out_a", "keeper")]
    tops = set()
    for order in itertools.permutations(holdings):
        results = tp.evaluate_transfers(con, list(order), "2025-2026", {3: (1, 1)}, free_transfers_available=1, points_per_hit=4.0)
        tops.add((results[0]["player_out"], results[0]["player_in"]))
    assert tops == {("out_a", "in_x")}
