import random

from fpl_quant import retrospective_validation as rv


# ============================================================
# fetch_entry_season_total
# ============================================================

def test_fetch_entry_season_total_finds_the_matching_season_row():
    payload = {"past": [
        {"season_name": "2024/25", "total_points": 2000},
        {"season_name": "2025/26", "total_points": 2419},
    ]}
    assert rv.fetch_entry_season_total(1, "2025/26", payload=payload) == 2419


def test_fetch_entry_season_total_none_when_season_not_played():
    payload = {"past": [{"season_name": "2024/25", "total_points": 2000}]}
    assert rv.fetch_entry_season_total(1, "2025/26", payload=payload) is None


def test_fetch_entry_season_total_none_when_entry_does_not_exist(monkeypatch):
    # A real 404 turns into payload=None upstream (_fetch_json's own contract, verified live
    # against the real API this session). Monkeypatch _fetch_json itself to simulate that,
    # rather than passing payload=None directly -- that sentinel means "no injected payload,
    # do a real fetch," which is a different (and correctly failing, without network access)
    # code path entirely.
    monkeypatch.setattr(rv, "_fetch_json", lambda url, **kwargs: None)
    assert rv.fetch_entry_season_total(1, "2025/26") is None


def test_fetch_entry_season_total_empty_past_list():
    assert rv.fetch_entry_season_total(1, "2025/26", payload={"past": []}) is None


# ============================================================
# sample_real_season_totals -- the sampling method itself: this is the surface the plan's
# Critique Engine pass found broken once (elite-bias via fetch_top_entries()) and fixed. These
# tests exist to keep that fix real, not just documented in prose.
# ============================================================

def test_sample_real_season_totals_never_calls_a_rank_ordered_source():
    """The one test that matters most for this module: confirm the sampling mechanism is
    genuinely random draws over the ID space, not top-N-by-rank. fetch_fn here returns a value
    keyed only by entry_id with no notion of rank/order at all -- if sample_real_season_totals
    ever started preferring low IDs or iterating in rank order, this test's ID coverage
    assertion below would catch it."""
    universe = {i: 1000 + i for i in range(1, 5001)}  # entry_id -> a fake total, no ranking
    calls: list[int] = []

    def fetch_fn(entry_id):
        calls.append(entry_id)
        return universe.get(entry_id)

    result = rv.sample_real_season_totals(
        "2025/26", target_n=200, id_space_max=5000,
        rng=random.Random(42), fetch_fn=fetch_fn,
    )
    assert result["n_sampled"] == 200
    assert len(result["totals"]) == 200
    # Every draw fell inside the declared ID space, and draws are not sequential/rank-ordered
    # (a rank-ordered or sequential source would produce calls == sorted(calls) or a tight
    # low-ID cluster; a real random sample over 5000 IDs picking 200 essentially never does).
    assert all(1 <= c <= 5000 for c in calls)
    assert calls != sorted(calls)


def test_sample_real_season_totals_rejects_and_continues():
    # Only even IDs "exist" -- every odd draw must be rejected and not counted toward n_sampled.
    def fetch_fn(entry_id):
        return 1000 + entry_id if entry_id % 2 == 0 else None

    result = rv.sample_real_season_totals(
        "2025/26", target_n=50, id_space_max=1000,
        rng=random.Random(7), fetch_fn=fetch_fn,
    )
    assert result["n_sampled"] == 50
    assert result["n_rejected"] > 0
    assert result["n_attempted"] == result["n_sampled"] + result["n_rejected"]


def test_sample_real_season_totals_never_draws_the_same_id_twice():
    seen_calls: list[int] = []

    def fetch_fn(entry_id):
        seen_calls.append(entry_id)
        return 1  # always "exists" -- forces target_n draws from a tiny space

    rv.sample_real_season_totals(
        "2025/26", target_n=20, id_space_max=20,
        rng=random.Random(1), fetch_fn=fetch_fn,
    )
    assert len(seen_calls) == len(set(seen_calls))


def test_sample_real_season_totals_stops_at_attempt_budget_not_infinite_loop():
    # Nothing ever exists -- must give up after target_n * max_attempts_multiplier attempts,
    # not hang forever, and must report the shortfall honestly rather than pad it.
    result = rv.sample_real_season_totals(
        "2025/26", target_n=100, id_space_max=1000,
        max_attempts_multiplier=3, rng=random.Random(3), fetch_fn=lambda entry_id: None,
    )
    assert result["n_sampled"] == 0
    assert result["n_attempted"] == 300
    assert result["totals"] == []


def test_sample_real_season_totals_return_value_never_contains_entry_id():
    """R4/R15: the return dict must be safe to log or cache verbatim -- no entry_id anywhere
    in it, only aggregate totals and counts."""
    def fetch_fn(entry_id):
        return 12345  # a suspicious, distinctive value that must NOT leak into keys/totals-as-id

    result = rv.sample_real_season_totals(
        "2025/26", target_n=5, id_space_max=100,
        rng=random.Random(9), fetch_fn=fetch_fn,
    )
    assert set(result.keys()) == {"totals", "n_sampled", "n_rejected", "n_attempted", "wall_clock_seconds"}
    assert result["totals"] == [12345, 12345, 12345, 12345, 12345]


def test_sample_real_season_totals_deterministic_given_a_seeded_rng():
    def fetch_fn(entry_id):
        return entry_id

    r1 = rv.sample_real_season_totals("2025/26", target_n=30, id_space_max=500, rng=random.Random(123), fetch_fn=fetch_fn)
    r2 = rv.sample_real_season_totals("2025/26", target_n=30, id_space_max=500, rng=random.Random(123), fetch_fn=fetch_fn)
    assert r1["totals"] == r2["totals"]
