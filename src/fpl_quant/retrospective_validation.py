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
cache file, or log line. Only aggregate point totals and counts leave this module.

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


# ============================================================
# fetch -- isolated network I/O, the one surface real verification and test mocking touch
# ============================================================

def _fetch_json(url: str, *, max_attempts: int = 4, base_backoff_seconds: float = 1.0) -> dict | None:
    """None on a real 404 (entry doesn't exist) -- any other real failure still raises after
    exhausting retries, rather than being silently swallowed."""
    for attempt in range(max_attempts):
        try:
            resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        except (requests.ConnectionError, requests.Timeout):
            if attempt == max_attempts - 1:
                raise
            time.sleep(base_backoff_seconds * (2 ** attempt))
            continue

        if resp.status_code == 404:
            return None
        if resp.status_code in _RETRYABLE_STATUS_CODES and attempt < max_attempts - 1:
            retry_after = resp.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else base_backoff_seconds * (2 ** attempt)
            time.sleep(delay)
            continue

        resp.raise_for_status()
        return resp.json()
    return None


# ============================================================
# one entry's real season total -- entry_id touches this function and nothing downstream
# ============================================================

def fetch_entry_season_total(entry_id: int, season_name: str, *, payload: dict | None = None) -> int | None:
    """season_name is the real API's own slash format (e.g. "2025/26"), not this project's
    hyphenated "2025-2026" DB convention -- callers convert. Returns the real, final
    total_points for that season from the entry's own history() `past` array (verified live
    this session: entry/{id}/history/ still serves season-level totals for past seasons even
    though per-gameweek picks don't survive), or None if the entry doesn't exist or never
    played that season. payload can be injected directly for tests instead of a live fetch."""
    if payload is None:
        payload = _fetch_json(f"{FPL_API_BASE}/entry/{entry_id}/history/")
    if payload is None:
        return None
    for row in payload.get("past", []):
        if row.get("season_name") == season_name:
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
    so a pathologically high real rejection rate fails loudly (via a short sample) rather than
    looping forever; the caller decides whether that's acceptable per the plan's Open Items
    fallback (draw more, or accept a smaller sample -- never switch to a rank-ordered source).

    Returns {"totals": [int, ...], "n_sampled": int, "n_rejected": int, "n_attempted": int,
    "wall_clock_seconds": float} -- deliberately no entry_id anywhere in the return value (R4),
    so a caller that logs or caches this dict verbatim automatically satisfies R15 too.

    fetch_fn (entry_id -> int | None) can be injected for tests instead of a live fetch; when
    absent, defaults to a real fetch_entry_season_total() call per draw, with a small delay
    between real requests (request_interval_seconds) to avoid hammering FPL's own API.
    """
    rng = rng or random.Random()
    live_fetch = fetch_fn is None
    fetch_fn = fetch_fn or (lambda entry_id: fetch_entry_season_total(entry_id, season_name))
    max_attempts = target_n * max_attempts_multiplier

    totals: list[int] = []
    n_attempted = 0
    seen: set[int] = set()
    t0 = time.monotonic()
    while len(totals) < target_n and n_attempted < max_attempts:
        entry_id = rng.randint(1, id_space_max)
        if entry_id in seen:
            continue
        seen.add(entry_id)
        n_attempted += 1
        total = fetch_fn(entry_id)
        if total is not None:
            totals.append(total)
        if live_fetch and request_interval_seconds > 0:
            time.sleep(request_interval_seconds)

    return {
        "totals": totals,
        "n_sampled": len(totals),
        "n_rejected": n_attempted - len(totals),
        "n_attempted": n_attempted,
        "wall_clock_seconds": time.monotonic() - t0,
    }
