"""Track E: Retrospective Historical Validation -- Phase E-1 (sampling only). See
docs/plans/2026-08_retrospective_validation_and_ml_decision_plan.md for the full design; this
module implements R2/R4/R6/R15 from that plan.

Draws a genuinely random sample of real FPL entries' actual season-total points, via
**uniform-random entry_id sampling** over the real ID space -- deliberately NOT
ingest_fpl_entry_picks.fetch_top_entries(), which only ever returns the highest-ranked entries
from a league's standings (confirmed: that function's own docstring, "Top n_entries by rank").
Reusing it here would compare the engine's own simulated season against the world's elite
managers, not real managers generally -- a real design error this plan's own Critique Engine
pass caught and corrected (see the plan document's Current State section). This module must
never call fetch_top_entries() or any other rank-ordered endpoint, including as a fallback --
the plan's Open Items section is explicit that a "pool built from standings, then subsampled"
does not fix the bias either, since the pool itself is still elite-only.

R4/R15 (this plan's own privacy requirements): entry_id is used only transiently, inside this
module, to make the fetch and de-duplicate draws -- it is never included in any returned value,
cache file, or log line, INCLUDING exception messages. A blind Critique Engine pass on this
phase's first implementation found a real, reproduced hole here: requests' own HTTPError /
urllib3's own ConnectionError messages embed the full request URL (which contains entry_id), so
a naive `raise`/`resp.raise_for_status()` on a non-404 failure (403, exhausted retries, a
connection blip) would leak entry_id straight into a traceback -- exactly the "persisted or CI
log output" R15 names. Fixed here: _fetch_json raises the module's own RetrievalError, whose
message never includes the URL, `from None` (suppressing Python's implicit exception chaining,
which would otherwise still attach the original URL-bearing exception as __context__ and print
it anyway). sample_real_season_totals then treats a RetrievalError the same as "no data for
this entry" -- reject and continue -- rather than aborting a ~40-minute run on one transient
failure, while still reporting it separately (n_errors) so a caller can tell "many entries just
don't play FPL" (normal) apart from "we're getting real API errors" (worth investigating). This
preserves the original "loud failure, not silently swallowed" intent: a caller inspecting
n_errors/rejection_rate at the end gets an honest, aggregate signal, just not a hard crash mid-run.

Disclosed, not fully solved (see the plan's Assumptions Ledger [A12]): uniform-random sampling
over the ID space skews toward accounts by signup recency, not by engagement, so it may still
over-represent dormant "set once, never touched" accounts relative to actively-managed ones.
The only current backstop is a mean/median sanity check against known general-population FPL
score figures, applied by the caller (scripts/run_retrospective_sample.py), not this module.
"""

from __future__ import annotations

import random
import time

import requests

FPL_API_BASE = "https://fantasy.premierleague.com/api"

# Verified live this session (see the plan's Current State): bootstrap-static's total_players
# was 9,824,056. A fixed 500,000-signup margin (not re-fetched live each run, deliberately --
# this number only needs to be a generous upper bound on the real ID space, not exact) gives
# [1, 10_324_056] as the real draw range.
DEFAULT_ID_SPACE_MAX = 10_324_056

# Same retry rationale as ingest_fpl_entry_picks._fetch_json -- duplicated rather than shared,
# per this project's established one-fetch-function-per-module convention (see that module's
# own docstring: "_fetch_json() is isolated specifically so verification is a one-line swap").
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class RetrievalError(RuntimeError):
    """A real fetch failure (non-404, or a connection/timeout that outlasted all retries).
    Message is deliberately sanitized -- status code / error type only, never the URL, which
    would embed entry_id (R15)."""


# ============================================================
# fetch -- isolated network I/O, the one surface real verification and test mocking touch
# ============================================================

def _fetch_json(url: str, *, max_attempts: int = 4, base_backoff_seconds: float = 1.0) -> dict | None:
    """None on a real 404 (entry doesn't exist). Any other real failure raises RetrievalError
    after exhausting retries -- loud, never silently swallowed, but with a message that never
    contains `url` (see module docstring for why this matters here specifically).

    A second Critique Engine pass on this fix found the first version still had two gaps: the
    except clause only caught ConnectionError/Timeout, missing other real requests.RequestException
    subtypes (TooManyRedirects, ChunkedEncodingError, ...) that can fire under exactly the
    "we're being blocked" conditions this exists to handle; and a 200-status response with a
    non-JSON body (e.g. a WAF/interstitial challenge page, which some anti-bot layers serve
    with a 200 rather than an error status) would crash `resp.json()` completely unguarded.
    Both are fixed here: the except clause now catches requests.RequestException (the base
    class for everything requests can raise), and resp.json() is wrapped too."""
    for attempt in range(max_attempts):
        try:
            resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        except requests.RequestException as e:
            if attempt == max_attempts - 1:
                raise RetrievalError(f"request error after {max_attempts} attempts: {type(e).__name__}") from None
            time.sleep(base_backoff_seconds * (2 ** attempt))
            continue

        if resp.status_code == 404:
            return None
        if resp.status_code in _RETRYABLE_STATUS_CODES and attempt < max_attempts - 1:
            # str.isdigit() is True for some Unicode digit characters float() can't parse
            # (e.g. U+00B2 "²") -- a third Critique Engine pass on this phase found a
            # pathological Retry-After header could raise an unguarded ValueError here.
            retry_after = resp.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after and retry_after.isdigit() else base_backoff_seconds * (2 ** attempt)
            except ValueError:
                delay = base_backoff_seconds * (2 ** attempt)
            time.sleep(delay)
            continue

        if resp.status_code >= 400:
            raise RetrievalError(f"HTTP {resp.status_code} after {attempt + 1} attempt(s)") from None
        try:
            return resp.json()
        except ValueError as e:
            # requests.exceptions.JSONDecodeError is a ValueError subclass; catching the
            # broader type is deliberate in case a future requests version's json() raises
            # something else JSON-parsing-related. The response body (which could echo the
            # requested path on some WAF pages) never enters the sanitized message below.
            raise RetrievalError(f"invalid JSON response (HTTP {resp.status_code}): {type(e).__name__}") from None

    # Unreachable: every loop iteration above returns or raises before falling through, on
    # every branch including the final attempt -- kept only as a defensive backstop against a
    # future edit accidentally adding a fall-through path.
    raise RetrievalError(f"exhausted {max_attempts} attempts") from None


# ============================================================
# one entry's real season total -- entry_id touches this function and nothing downstream
# ============================================================

def fetch_entry_season_total(entry_id: int, season_name: str, *, payload: dict | None = None) -> int | None:
    """season_name is the real API's own slash format (e.g. "2025/26"), not this project's
    hyphenated "2025-2026" DB convention -- callers convert. Returns the real, final
    total_points for that season from the entry's own history() `past` array (verified live
    this session: entry/{id}/history/ still serves season-level totals for past seasons even
    though per-gameweek picks don't survive), or None if the entry doesn't exist or never
    played that season. Raises RetrievalError (sanitized, see module docstring) on a real fetch
    failure. payload can be injected directly for tests instead of a live fetch."""
    if payload is None:
        payload = _fetch_json(f"{FPL_API_BASE}/entry/{entry_id}/history/")
    if payload is None:
        return None
    # A third Critique Engine pass on this phase found this loop assumed payload is always a
    # dict and every element of "past" is always a dict -- neither is guaranteed just because
    # the response parsed as JSON (e.g. a gateway/WAF could return a bare JSON array, or one
    # malformed row, with a 200 status). Treated as "no usable season row," same as any other
    # unexpected shape, rather than crashing the whole sample on one anomalous response.
    if not isinstance(payload, dict):
        return None
    past = payload.get("past", [])
    if not isinstance(past, list):
        return None
    for row in past:
        if isinstance(row, dict) and row.get("season_name") == season_name:
            return row.get("total_points")
    return None


# ============================================================
# the sample -- uniform-random draws, reject-and-continue, entry_id never leaves this function
# ============================================================

def sample_real_season_totals(
    season_name: str, target_n: int, *,
    id_space_max: int = DEFAULT_ID_SPACE_MAX,
    max_attempts_multiplier: int = 20,
    request_interval_seconds: float = 0.1,
    rng: random.Random | None = None,
    fetch_fn=None,
) -> dict:
    """Uniform-random entry_id sampling over [1, id_space_max] -- see module docstring for why
    this, not fetch_top_entries(). Draws random IDs (never repeating one already tried this
    run), keeps only those with a real season_name row in their history, until target_n are
    collected or a generous attempt budget (target_n * max_attempts_multiplier) is exhausted --
    so a pathologically high real rejection rate fails loudly (via a short sample, with n_errors
    surfaced separately from ordinary "doesn't exist" rejections) rather than looping forever;
    the caller decides whether that's acceptable per the plan's Open Items fallback (draw more,
    or accept a smaller sample -- never switch to a rank-ordered source).

    A RetrievalError from fetch_fn (a real fetch failure, not "entry doesn't exist") is caught
    here and counted separately (n_errors) rather than aborting the whole run -- see module
    docstring for why this is the right resilience/honesty trade-off for a ~40-minute live run.

    Returns {"totals": [int, ...], "n_sampled": int, "n_rejected": int, "n_errors": int,
    "n_attempted": int, "wall_clock_seconds": float} -- deliberately no entry_id anywhere in the
    return value (R4), so a caller that logs or caches this dict verbatim automatically
    satisfies R15 too.

    fetch_fn (entry_id -> int | None, may raise RetrievalError) can be injected for tests
    instead of a live fetch; when absent, defaults to a real fetch_entry_season_total() call per
    draw, with a small delay between real requests (request_interval_seconds) to avoid hammering
    FPL's own API.
    """
    rng = rng or random.Random()
    live_fetch = fetch_fn is None
    fetch_fn = fetch_fn or (lambda entry_id: fetch_entry_season_total(entry_id, season_name))
    max_attempts = target_n * max_attempts_multiplier

    totals: list[int] = []
    n_attempted = 0
    n_errors = 0
    seen: set[int] = set()
    t0 = time.monotonic()
    while len(totals) < target_n and n_attempted < max_attempts:
        entry_id = rng.randint(1, id_space_max)
        if entry_id in seen:
            continue
        seen.add(entry_id)
        n_attempted += 1
        try:
            total = fetch_fn(entry_id)
        except RetrievalError:
            n_errors += 1
            total = None
        if total is not None:
            totals.append(total)
        if live_fetch and request_interval_seconds > 0:
            time.sleep(request_interval_seconds)

    return {
        "totals": totals,
        "n_sampled": len(totals),
        "n_rejected": n_attempted - len(totals),
        "n_errors": n_errors,
        "n_attempted": n_attempted,
        "wall_clock_seconds": time.monotonic() - t0,
    }
