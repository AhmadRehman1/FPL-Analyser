"""scripts/compute_ml_shadow.py's target selection: the shadow reads the ep_outputs built for the
live next gameweek (the first one whose deadline is after the DB's data cutoff), never just the
newest ep_model_version -- the nightly's own walk-forward and the planner/forward runs write newer
versions for other gameweeks (the 2026-10-04..07 shadows all reported GW38 that way)."""

import sys
from datetime import datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import compute_ml_shadow as cms  # noqa: E402

LIVE = "2026-2027"
# The cached 18:00 pipeline ingest the 02:00 nightly restores: GW5 played, GW6 deadline next.
INGESTED_AT = datetime(2026, 10, 6, 18, 5)
# (season, gameweek) -> its fixtures' kickoffs; GW6 has two.
KICKOFFS = {
    (LIVE, 1): [datetime(2026, 8, 22, 11, 30)],
    (LIVE, 5): [datetime(2026, 9, 19, 14, 0)],
    (LIVE, 6): [datetime(2026, 10, 10, 11, 30), datetime(2026, 10, 10, 14, 0)],
    (LIVE, 7): [datetime(2026, 10, 17, 11, 30)],
    (LIVE, 18): [datetime(2026, 12, 26, 15, 0)],
    (LIVE, 38): [datetime(2027, 5, 23, 15, 0)],
    ("2025-2026", 38): [datetime(2026, 5, 24, 15, 0)],
}


def _match_ids(season, gw):
    return [f"m_{season}_{gw}_{i}" for i in range(len(KICKOFFS[(season, gw)]))]


def _seed(con, ep_gameweeks, *, partial=()):
    """KICKOFFS' fixtures, then one ep_outputs version per (season, gameweek) in `ep_gameweeks`,
    in that order, each built from its own minutes model -- covering only its gameweek's first
    fixture for an index in `partial` (a half-written version). Returns
    {index: (ep_model_version, minutes_model_version)}."""
    for team in ("team_a", "team_b"):
        con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES (?, ?)", [team, team])
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p1', 'p1', 'Midfielder')")
    for (season, gw), kickoffs in KICKOFFS.items():
        for match_id, kickoff in zip(_match_ids(season, gw), kickoffs):
            con.execute(
                "INSERT INTO fact_match (match_id, season, gameweek, kickoff_time, home_team_uid, away_team_uid, "
                "finished, competition, _ingested_at) VALUES (?, ?, ?, ?, 'team_a', 'team_b', ?, 'Premier League', ?)",
                [match_id, season, gw, kickoff, kickoff < INGESTED_AT, INGESTED_AT],
            )
    ts_mv = con.execute(
        "INSERT INTO team_strength_model_versions (calibration_asof_date, home_advantage, xi_params_version, "
        "rho_params_version, reference_team_uid) VALUES ('2026-10-06', 0.2, 1, 1, 'team_a') RETURNING model_version"
    ).fetchone()[0]
    versions = {}
    for i, (season, gw) in enumerate(ep_gameweeks):
        mm_mv = con.execute(
            "INSERT INTO minutes_model_versions (calibration_asof_date, target_season, decay_params_version, "
            "adjustment_params_version, shrinkage_params_version, fact_multiplier_params_version, lookback_seasons) "
            "VALUES ('2026-10-06', ?, 1, 1, 1, 1, '[]') RETURNING model_version",
            [season],
        ).fetchone()[0]
        ep_mv = con.execute(
            "INSERT INTO ep_model_versions (calibration_asof_date, target_season, team_strength_model_version, "
            "minutes_model_version, scoring_matrix_params_version, bps_params_version, bps_tau_params_version) "
            "VALUES ('2026-10-06', ?, ?, ?, 1, 1, 1) RETURNING model_version",
            [season, ts_mv, mm_mv],
        ).fetchone()[0]
        match_ids = _match_ids(season, gw)
        for match_id in match_ids[:1] if i in partial else match_ids:
            con.execute(
                "INSERT INTO ep_outputs (model_version, player_uid, fixture_match_id, ep_appearance, ep_goals, "
                "ep_assists, ep_clean_sheet, ep_goals_conceded, ep_defcon, ep_bonus, ep_saves, ep_penalty_save, "
                "ep_cards, ep_own_goal, ep_total, expected_bps) "
                "VALUES (?, 'p1', ?, 1.0, 0.3, 0.1, 0, 0, 0, 0.2, 0, 0, 0, 0, 1.6, 20.0)",
                [ep_mv, match_id],
            )
        versions[i] = (ep_mv, mm_mv)
    return versions


def test_reads_the_next_gameweeks_version_when_later_gameweek_versions_are_newer(con, monkeypatch):
    # Production order: the live pipeline's GW6, then later 2026-27 gameweeks (a forward plan's
    # GW18, a season sim's GW38), then the nightly walk-forward's last step, 2025-26 GW38, newest.
    versions = _seed(con, [(LIVE, 6), (LIVE, 18), (LIVE, 38), ("2025-2026", 38)])
    calls = []
    monkeypatch.setattr(
        "research.ml.forward.predict_forward",
        lambda _con, season, gw, ep_mv, mm_mv: calls.append((season, gw, ep_mv, mm_mv)),
    )

    payload = cms.build_ml_shadow_payload(con, element_names={})

    assert payload["target_gameweek"] == 6
    assert payload["data_cutoff"] == INGESTED_AT.isoformat()
    assert payload["ep_model_version"] == versions[0][0]
    # GW6's own ep_outputs AND the minutes model they were built from, not the walk-forward's
    assert calls == [(LIVE, 6, *versions[0])]


def test_skips_a_half_written_version_of_the_next_gameweek(con):
    # a newer GW6 version covering one of its two fixtures (a horizon step that died mid-write)
    versions = _seed(con, [(LIVE, 6), (LIVE, 6)], partial={1})
    assert cms._ep_versions_for_gameweek(con, 6) == versions[0]


def test_placeholder_when_no_version_was_built_for_the_next_gameweek(con, monkeypatch):
    # GW1 (ingestion), later gameweeks and the walk-forward exist -- nothing for GW6
    _seed(con, [(LIVE, 1), (LIVE, 18), (LIVE, 38), ("2025-2026", 38)])
    monkeypatch.setattr("research.ml.forward.predict_forward", lambda *a, **k: pytest.fail("predicted a wrong week"))

    payload = cms.build_ml_shadow_payload(con, element_names={})

    assert payload["status"] == "no_ep_outputs_for_target_gameweek"
    assert payload["target_gameweek"] == 6
    assert payload["players"] == []


@pytest.mark.parametrize("cutoff, expected", [
    (INGESTED_AT, 6),                      # international break after GW5
    (datetime(2026, 10, 10, 18, 0), 7),    # GW6 under way
    (datetime(2027, 5, 24, 0, 0), None),   # season over
    (None, None),                          # nothing ingested
])
def test_target_gameweek_is_the_first_deadline_after_the_cutoff(con, cutoff, expected):
    _seed(con, [])
    assert cms._target_gameweek_from_db(con, cutoff) == expected
