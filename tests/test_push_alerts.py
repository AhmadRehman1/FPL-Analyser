"""App gap 1: which diffs are alert-worthy, the push-payload construction, and that an absent
subscription store / empty alert list never crashes a scheduled run."""

from datetime import datetime, timedelta, timezone

from fpl_quant import push_alerts as pa

NOW = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)


def _team(*squad_overrides):
    base = [
        {"player_id": 1, "web_name": "Raya", "is_captain": False, "is_vice_captain": False},
        {"player_id": 2, "web_name": "Haaland", "is_captain": True, "is_vice_captain": False},
        {"player_id": 3, "web_name": "Salah", "is_captain": False, "is_vice_captain": True},
        {"player_id": 4, "web_name": "Saka", "is_captain": False, "is_vice_captain": False},
    ]
    by_id = {p["player_id"]: p for p in base}
    for ov in squad_overrides:
        by_id[ov["player_id"]].update(ov)
    return {"entry_id": 7139944, "squad": list(by_id.values())}


def _players(*overrides):
    base = {i: {"id": i, "web_name": n, "status": "a", "chance_of_playing_next_round": None, "news": None}
            for i, n in [(1, "Raya"), (2, "Haaland"), (3, "Salah"), (4, "Saka")]}
    for ov in overrides:
        base[ov["id"]].update(ov)
    return base


# ---- captain / vice doubtful -------------------------------------------------

def test_no_alerts_when_everyone_is_fit():
    alerts = pa.compute_alerts(
        real_squad={}, team=_team(), players_by_id=_players(),
        price_watch={}, next_deadline_utc=None, now_utc=NOW,
    )
    assert alerts == []


def test_captain_flagged_injured_is_a_high_priority_alert():
    alerts = pa.compute_alerts(
        real_squad={}, team=_team(),
        players_by_id=_players({"id": 2, "status": "i", "news": "Hamstring - out"}),
        price_watch={}, next_deadline_utc=None, now_utc=NOW,
    )
    assert len(alerts) == 1
    a = alerts[0]
    assert a["kind"] == "captain_doubtful" and a["priority"] == "high"
    assert "Captain doubt: Haaland" in a["title"]
    assert a["body"] == "Hamstring - out"


def test_vice_captain_at_75pct_chance_is_flagged_even_without_a_status_code():
    alerts = pa.compute_alerts(
        real_squad={}, team=_team(),
        players_by_id=_players({"id": 3, "status": "a", "chance_of_playing_next_round": 75}),
        price_watch={}, next_deadline_utc=None, now_utc=NOW,
    )
    assert [a["kind"] for a in alerts] == ["captain_doubtful"]
    assert "Vice-captain doubt: Salah" in alerts[0]["title"]


def test_a_doubtful_non_captain_is_NOT_alerted():
    alerts = pa.compute_alerts(
        real_squad={}, team=_team(), players_by_id=_players({"id": 4, "status": "d", "chance_of_playing_next_round": 50}),
        price_watch={}, next_deadline_utc=None, now_utc=NOW,
    )
    assert alerts == []


# ---- price change on a held player -----------------------------------------

def test_price_change_only_fires_for_a_player_actually_in_the_squad():
    pw = {"risers": [{"id": 2, "web_name": "Haaland", "team": "MCI"}, {"id": 999, "web_name": "Someone", "team": "XYZ"}],
          "fallers": [{"id": 4, "web_name": "Saka", "team": "ARS"}]}
    alerts = pa.compute_alerts(
        real_squad={}, team=_team(), players_by_id=_players(),
        price_watch=pw, next_deadline_utc=None, now_utc=NOW,
    )
    kinds = [(a["kind"], a["title"]) for a in alerts]
    assert ("price_change", "Price rise tonight: Haaland") in kinds
    assert ("price_change", "Price fall tonight: Saka") in kinds
    assert not any("Someone" in t for _, t in kinds)
    assert all(a["priority"] == "medium" for a in alerts)


# ---- pending recommendation near the deadline -----------------------------

def test_pending_transfer_only_alerts_inside_the_lead_window():
    # 'transfer_now' is the real hold_recommendations.recommended_action value for "make a
    # transfer" (schema CHECK: hold | transfer_now | no_action_available).
    rs = {"hold_vs_transfer_now": {"recommended_action": "transfer_now"},
          "transfer_recommendations": [{"player_out": "Kinsky", "player_in": "Leno", "net": 8.9}]}
    far = pa.compute_alerts(real_squad=rs, team=_team(), players_by_id=_players(), price_watch={},
                            next_deadline_utc=NOW + timedelta(hours=10), now_utc=NOW)
    assert far == []
    near = pa.compute_alerts(real_squad=rs, team=_team(), players_by_id=_players(), price_watch={},
                             next_deadline_utc=NOW + timedelta(hours=2), now_utc=NOW, lead_hours=3)
    assert [a["kind"] for a in near] == ["pending_transfer"]
    assert "Kinsky → Leno" in near[0]["body"]


def test_pending_transfer_ignores_no_action_available():
    rs = {"hold_vs_transfer_now": {"recommended_action": "no_action_available"}, "transfer_recommendations": []}
    alerts = pa.compute_alerts(real_squad=rs, team=_team(), players_by_id=_players(), price_watch={},
                               next_deadline_utc=NOW + timedelta(hours=1), now_utc=NOW)
    assert alerts == []


def test_pending_transfer_not_alerted_when_the_model_says_hold():
    rs = {"hold_vs_transfer_now": {"recommended_action": "hold"}, "transfer_recommendations": []}
    alerts = pa.compute_alerts(real_squad=rs, team=_team(), players_by_id=_players(), price_watch={},
                               next_deadline_utc=NOW + timedelta(hours=1), now_utc=NOW)
    assert alerts == []


def test_recommended_chip_near_deadline_is_alerted():
    rs = {"chip_evaluations": [{"chip_type": "wildcard", "recommended": True},
                               {"chip_type": "bench_boost", "recommended": False}]}
    alerts = pa.compute_alerts(real_squad=rs, team=_team(), players_by_id=_players(), price_watch={},
                               next_deadline_utc=NOW + timedelta(hours=1), now_utc=NOW)
    assert [a["kind"] for a in alerts] == ["pending_chip"]
    assert "Wildcard recommended" in alerts[0]["title"]


def test_deadline_already_passed_is_not_within_lead():
    rs = {"hold_vs_transfer_now": {"recommended_action": "transfer_now"}, "transfer_recommendations": []}
    alerts = pa.compute_alerts(real_squad=rs, team=_team(), players_by_id=_players(), price_watch={},
                               next_deadline_utc=NOW - timedelta(hours=1), now_utc=NOW)
    assert alerts == []


# ---- payload construction --------------------------------------------------

def test_build_push_payload_none_when_no_alerts():
    assert pa.build_push_payload([]) is None


def test_build_push_payload_leads_with_the_highest_priority_and_counts_the_rest():
    alerts = [
        {"kind": "price_change", "priority": "medium", "title": "Price fall: Saka", "body": "..."},
        {"kind": "captain_doubtful", "priority": "high", "title": "Captain doubt: Haaland", "body": "Knock"},
    ]
    payload = pa.build_push_payload(alerts)
    assert payload["title"] == "Captain doubt: Haaland"
    assert payload["body"] == "Knock  (+1 more alert)"
    assert payload["tag"] == "fpl-quant-deadline"
    assert payload["url"].startswith("https://")
    assert payload["alert_kinds"] == ["captain_doubtful", "price_change"]


def test_missing_or_empty_inputs_never_raise():
    # the "no subscription / nothing ingested yet" shape a first scheduled run can hit
    assert pa.compute_alerts(real_squad=None, team=None, players_by_id={}, price_watch=None,
                             next_deadline_utc=None, now_utc=NOW) == []
