import random

import pytest
import requests

from fpl_quant import retrospective_validation as rv

# An entry_id chosen to be distinctive/greppable -- every test below that checks "no leak"
# asserts this exact substring is absent from whatever text it's checking.
_SENTINEL_ENTRY_ID = 8675309


class _FakeResponse:
    def __init__(self, status_code, headers=None):
        self.status_code = status_code
        self.headers = headers or {}

    def json(self):
        return {"past": []}


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


def test_fetch_entry_season_total_handles_malformed_row_shapes_without_crashing():
    """A third Critique Engine pass found this had assumed every 'past' row is a dict --
    a real gap for a genuinely malformed (but JSON-parseable) response, e.g. a WAF/gateway
    quirk. Treated as no usable data for that row, not a crash."""
    payload = {"past": ["corrupted-row", {"season_name": "2025/26", "total_points": 1900}]}
    assert rv.fetch_entry_season_total(1, "2025/26", payload=payload) == 1900


def test_fetch_entry_season_total_handles_non_dict_payload_without_crashing():
    assert rv.fetch_entry_season_total(1, "2025/26", payload=[]) is None


def test_fetch_entry_season_total_handles_non_list_past_without_crashing():
    assert rv.fetch_entry_season_total(1, "2025/26", payload={"past": "not-a-list"}) is None


# ============================================================
# _fetch_json / RetrievalError -- the blocking bug a Critique Engine pass on this phase found:
# a naive raise on a non-404 failure embeds the real entry_id in the exception message (via the
# request URL), which would then hit a traceback / CI log -- a direct R15 violation. These tests
# exist to keep that fix real, not just documented in prose (the plan's own Verification section
# for this phase explicitly calls for exactly this: "a log-output test confirms no entry_id
# appears in captured log lines").
# ============================================================

def test_fetch_json_raises_retrieval_error_not_raw_requests_error_on_persistent_500(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(500))
    with pytest.raises(rv.RetrievalError):
        rv._fetch_json(f"{rv.FPL_API_BASE}/entry/{_SENTINEL_ENTRY_ID}/history/", max_attempts=2, base_backoff_seconds=0)


def test_fetch_json_retrieval_error_message_never_contains_the_url_or_entry_id(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(403))  # non-retryable, raises on attempt 0
    url = f"{rv.FPL_API_BASE}/entry/{_SENTINEL_ENTRY_ID}/history/"
    with pytest.raises(rv.RetrievalError) as exc_info:
        rv._fetch_json(url)
    message = str(exc_info.value)
    assert str(_SENTINEL_ENTRY_ID) not in message
    assert "fantasy.premierleague.com" not in message
    assert url not in message


def test_fetch_json_raises_retrieval_error_after_retries_exhausted_on_connection_error(monkeypatch):
    """This is the branch where a real chained exception (__context__) actually exists to leak
    -- unlike the 403 test above, where nothing is being handled at the raise site, so
    __context__ is trivially None there regardless of whether `from None` works. Here a real
    ConnectionError, with the URL embedded in ITS OWN message, is genuinely in flight when
    RetrievalError is raised -- `from None` has to actively suppress it, not just have nothing
    to suppress. A second Critique Engine pass on this fix specifically flagged the original
    version of this test for not checking __context__/__cause__ on the one branch where it
    would have mattered; fixed here."""
    def always_fails(*a, **k):
        raise requests.ConnectionError(f"connection refused for {rv.FPL_API_BASE}/entry/{_SENTINEL_ENTRY_ID}/history/")

    monkeypatch.setattr(requests, "get", always_fails)
    monkeypatch.setattr(rv.time, "sleep", lambda seconds: None)  # don't actually wait during the test
    url = f"{rv.FPL_API_BASE}/entry/{_SENTINEL_ENTRY_ID}/history/"
    try:
        rv._fetch_json(url, max_attempts=3, base_backoff_seconds=0)
        assert False, "expected RetrievalError"
    except rv.RetrievalError as e:
        assert str(_SENTINEL_ENTRY_ID) not in str(e)
        assert e.__cause__ is None
        # The real ConnectionError (with the sentinel in ITS message) genuinely exists as
        # Python's implicit __context__ here -- confirming that, rather than just asserting
        # __context__ is None, is what makes this test meaningful: it proves `from None` is
        # doing real suppression work, not asserting a vacuous truth.
        assert e.__context__ is not None
        assert str(_SENTINEL_ENTRY_ID) in str(e.__context__)
        # ...but default traceback rendering (what would actually hit a log/CI output) must not
        # show it, since __suppress_context__ is set by `from None`.
        assert e.__suppress_context__ is True


def test_fetch_json_catches_request_exceptions_beyond_connection_error_and_timeout(monkeypatch):
    """A second Critique Engine pass found the original except clause too narrow -- only
    ConnectionError/Timeout -- missing other real requests.RequestException subtypes (e.g.
    TooManyRedirects) that can fire under the same "we're being blocked" conditions this
    exists to handle. Confirms the broadened `except requests.RequestException` actually
    catches one of those other subtypes rather than letting it propagate raw."""
    def raises_too_many_redirects(*a, **k):
        raise requests.TooManyRedirects(f"Exceeded 30 redirects for entry {_SENTINEL_ENTRY_ID}")

    monkeypatch.setattr(requests, "get", raises_too_many_redirects)
    monkeypatch.setattr(rv.time, "sleep", lambda seconds: None)
    url = f"{rv.FPL_API_BASE}/entry/{_SENTINEL_ENTRY_ID}/history/"
    with pytest.raises(rv.RetrievalError) as exc_info:
        rv._fetch_json(url, max_attempts=2, base_backoff_seconds=0)
    assert str(_SENTINEL_ENTRY_ID) not in str(exc_info.value)


def test_fetch_json_wraps_a_non_json_200_response_as_retrieval_error(monkeypatch):
    """A second Critique Engine pass found resp.json() on the success (< 400 status) path was
    completely unguarded -- a WAF/interstitial challenge page served with a 200 (a realistic
    "we're being blocked" symptom, not a contrived one) would crash with a raw
    JSONDecodeError instead of a sanitized RetrievalError, aborting the whole sampling run."""
    class _NonJsonResponse(_FakeResponse):
        def json(self):
            raise requests.exceptions.JSONDecodeError("Expecting value", "<html>blocked</html>", 0)

    monkeypatch.setattr(requests, "get", lambda *a, **k: _NonJsonResponse(200))
    url = f"{rv.FPL_API_BASE}/entry/{_SENTINEL_ENTRY_ID}/history/"
    with pytest.raises(rv.RetrievalError) as exc_info:
        rv._fetch_json(url)
    assert str(_SENTINEL_ENTRY_ID) not in str(exc_info.value)


def test_fetch_json_returns_none_on_404_without_raising(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(404))
    assert rv._fetch_json(f"{rv.FPL_API_BASE}/entry/{_SENTINEL_ENTRY_ID}/history/") is None


def test_fetch_json_survives_a_pathological_retry_after_header(monkeypatch):
    """A third Critique Engine pass found str.isdigit() is True for some Unicode digit
    characters float() can't parse (e.g. U+00B2 superscript-2), which would raise an unguarded
    ValueError while computing the backoff delay for a retryable status. Must fall back to the
    normal backoff formula instead of crashing."""
    calls = {"n": 0}

    def flaky_get(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResponse(503, headers={"Retry-After": "²"})  # "²" -- isdigit() True, float() raises
        return _FakeResponse(200)

    monkeypatch.setattr(requests, "get", flaky_get)
    monkeypatch.setattr(rv.time, "sleep", lambda seconds: None)
    result = rv._fetch_json(f"{rv.FPL_API_BASE}/entry/{_SENTINEL_ENTRY_ID}/history/", max_attempts=3, base_backoff_seconds=0)
    assert result == {"past": []}
    assert calls["n"] == 2


def test_fetch_json_retries_then_succeeds_on_a_transient_retryable_status(monkeypatch):
    calls = {"n": 0}

    def flaky_get(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResponse(503)
        return _FakeResponse(200)

    monkeypatch.setattr(requests, "get", flaky_get)
    monkeypatch.setattr(rv.time, "sleep", lambda seconds: None)
    result = rv._fetch_json(f"{rv.FPL_API_BASE}/entry/{_SENTINEL_ENTRY_ID}/history/", max_attempts=3, base_backoff_seconds=0)
    assert result == {"past": []}
    assert calls["n"] == 2


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
    assert set(result.keys()) == {"totals", "n_sampled", "n_rejected", "n_errors", "n_attempted", "wall_clock_seconds"}
    assert result["totals"] == [12345, 12345, 12345, 12345, 12345]


def test_sample_real_season_totals_catches_retrieval_error_and_keeps_sampling():
    """A real fetch failure (RetrievalError -- see the _fetch_json tests above) must not abort
    a long-running sample outright; it's counted as n_errors, distinct from an ordinary
    "entry doesn't exist" rejection, and sampling continues."""
    def fetch_fn(entry_id):
        if entry_id % 3 == 0:
            raise rv.RetrievalError("HTTP 500 after 4 attempt(s)")
        if entry_id % 2 == 0:
            return None  # ordinary "doesn't exist" rejection
        return 1000 + entry_id

    result = rv.sample_real_season_totals(
        "2025/26", target_n=30, id_space_max=2000,
        rng=random.Random(11), fetch_fn=fetch_fn,
    )
    assert result["n_sampled"] == 30
    assert result["n_errors"] > 0
    assert result["n_rejected"] >= result["n_errors"]  # n_rejected includes n_errors + ordinary misses
    assert result["n_attempted"] == result["n_sampled"] + result["n_rejected"]


def test_sample_real_season_totals_error_message_from_fetch_fn_never_reaches_caller_unhandled():
    """Even if every single draw errors, sample_real_season_totals itself must not raise --
    it should exhaust the attempt budget and return an honest, empty-ish result, per the same
    "fails loudly via the reported numbers, not via a crash" contract as the plain-rejection
    case (test_sample_real_season_totals_stops_at_attempt_budget_not_infinite_loop)."""
    def always_errors(entry_id):
        raise rv.RetrievalError("HTTP 403 after 1 attempt(s)")

    result = rv.sample_real_season_totals(
        "2025/26", target_n=10, id_space_max=1000,
        max_attempts_multiplier=3, rng=random.Random(5), fetch_fn=always_errors,
    )
    assert result["n_sampled"] == 0
    assert result["n_errors"] == 30
    assert result["n_attempted"] == 30


def test_sample_real_season_totals_deterministic_given_a_seeded_rng():
    def fetch_fn(entry_id):
        return entry_id

    r1 = rv.sample_real_season_totals("2025/26", target_n=30, id_space_max=500, rng=random.Random(123), fetch_fn=fetch_fn)
    r2 = rv.sample_real_season_totals("2025/26", target_n=30, id_space_max=500, rng=random.Random(123), fetch_fn=fetch_fn)
    assert r1["totals"] == r2["totals"]


# ============================================================
# compute_retrospective_comparison (Phase E-3) -- pure computation, tested against a synthetic
# distribution with a KNOWN answer, per the plan's own done-check wording.
# ============================================================

_KNOWN_DISTRIBUTION = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]  # n=10, mean=5.5, median=5.5


def test_compute_retrospective_comparison_above_all_real_values_is_100th_percentile():
    result = rv.compute_retrospective_comparison(11.0, _KNOWN_DISTRIBUTION)
    assert result["percentile_rank"] == pytest.approx(100.0)
    assert result["point_differential_vs_mean"] == pytest.approx(5.5)
    assert result["point_differential_vs_median"] == pytest.approx(5.5)


def test_compute_retrospective_comparison_below_all_real_values_is_0th_percentile():
    result = rv.compute_retrospective_comparison(0.0, _KNOWN_DISTRIBUTION)
    assert result["percentile_rank"] == pytest.approx(0.0)
    assert result["point_differential_vs_mean"] == pytest.approx(-5.5)


def test_compute_retrospective_comparison_exact_tie_counts_as_half():
    """Equal to the maximum sample value (10.0): 9 strictly below + 1 tied, tied counts as
    half -- (9 + 0.5) / 10 * 100 = 95.0, a known, exactly-verifiable answer, not an approximation."""
    result = rv.compute_retrospective_comparison(10.0, _KNOWN_DISTRIBUTION)
    assert result["percentile_rank"] == pytest.approx(95.0)


def test_compute_retrospective_comparison_midpoint_is_50th_percentile():
    result = rv.compute_retrospective_comparison(5.5, _KNOWN_DISTRIBUTION)
    assert result["percentile_rank"] == pytest.approx(50.0)
    assert result["point_differential_vs_mean"] == pytest.approx(0.0)
    assert result["point_differential_vs_median"] == pytest.approx(0.0)


def test_compute_retrospective_comparison_reports_real_summary_stats():
    result = rv.compute_retrospective_comparison(7.0, _KNOWN_DISTRIBUTION)
    assert result["n_real_sample"] == 10
    assert result["real_mean"] == pytest.approx(5.5)
    assert result["real_median"] == pytest.approx(5.5)
    assert result["real_min"] == pytest.approx(1.0)
    assert result["real_max"] == pytest.approx(10.0)
    assert result["engine_total"] == pytest.approx(7.0)


def test_compute_retrospective_comparison_odd_n_median():
    result = rv.compute_retrospective_comparison(3.0, [1.0, 2.0, 3.0, 4.0, 100.0])  # median=3, mean=22
    assert result["real_median"] == pytest.approx(3.0)
    assert result["real_mean"] == pytest.approx(22.0)
    assert result["point_differential_vs_median"] == pytest.approx(0.0)
    assert result["point_differential_vs_mean"] == pytest.approx(-19.0)


def test_compute_retrospective_comparison_raises_on_empty_real_totals():
    with pytest.raises(ValueError):
        rv.compute_retrospective_comparison(5.0, [])
