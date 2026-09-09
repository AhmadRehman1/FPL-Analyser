# Priority 10 — Full Rival-Squad-Distribution Field Simulator: Design Doc

Status: **Phase A implemented** (`ingest_fpl_entry_picks.py`, `fact_rival_squad_sample`,
`scripts/run_rival_sample_ingestion.py`) with conservative defaults (FPL's Overall league,
n_entries=200, minimal fields — no manager/team names stored).

**Phase B started (rank instrumentation, 2026-09).** Not the full joint simulator described
in §3–4 — that is still §4's Phase B/C. This first slice is the *measurement* half, built
because the model-managed team's real goal was pinned down as **overall rank ≈ top 100k** and
nothing in the stack measured rank (`nightly_backtest.yml` scores a synthetic points delta).
What shipped:

- **Stratified sampling.** `ingest_fpl_entry_picks.fetch_entries_in_rank_bands()` +
  `DEFAULT_RANK_BANDS` — ~120 entries around the target band (ranks 60k–140k) plus a ~40
  elite slice and a ~40 mid-field slice, replacing the top-200-by-rank scaffold (which
  measured the model against the 200 best managers alive — the wrong bar for a top-100k
  goal). Answers §5's "sampling source" open question: **stratified around the user's rank
  target**, not the leaderboard top. Degrades gracefully at the standings API's
  deep-pagination frontier and reports the bands actually reached.
- **`field_rank.py`** — places any squad in that settled field (percentile + projected rank,
  reusing `live_tracking.estimate_live_rank`) and decomposes the rank gap into captaincy /
  template-coverage / differential / bench.
- **`fact_squad_rank_score`** (`schema/0018`), **`scripts/rank_autopsy.py`**,
  **`.github/workflows/rank_tracking.yml`** (weekly) — scores the model's optimal squad and
  the two tracked accounts each completed gameweek → `data/dashboard/rank_autopsy.json`.
- Forward-only, as §4's Phase D anticipated: no historical rival data exists, so scoring
  begins at 2026-27 GW1.

It does **not** touch the optimiser — that is Phase C (joint simulation as a solve input),
still design-only below.

Phases C/D below are **design only, not implemented**. Per the roadmap's own explicit
instruction: "this is a serious standalone project — do not start it until Priorities 0-6 are
solid, and scope it as its own multi-PR effort with its own design doc reviewed... before
implementation, not a single PR." This doc is that review artifact.

## 1. What exists today (Priority 1) and why it's a real limitation

`field_covariance.py`'s `compute_field_covariance()` builds **one synthetic EO-weighted
portfolio** — effectively a blended pseudo-squad representing "the average exposure of the
field" — and reuses M6's `Z_fixture`/Gamma-Poisson machinery to estimate how correlated a
candidate squad is with that single blended portfolio. This is a genuine, honestly-disclosed
**mean-field approximation**: it answers "how correlated am I with the field's *average*
exposure," not "what does my rank look like against the actual *distribution* of real rival
squads."

The gap this leaves: real FPL rank movement doesn't come from beating an average — it comes
from beating (or losing to) *specific* rival squads, whose ownership isn't independent
per-player. A manager who owns the template premium forward is also disproportionately
likely to own the popular budget enablers and to have captained the same premium — real
correlation structure a mean-field blend cannot represent by construction. This is exactly
what Priority 1's own field_covariance.py docstring already names as a scoped-down
simplification, and what Priority 10 is meant to replace.

## 2. The key finding that changes this project's feasibility calculus

Every other "field/rival" signal built so far in this project (EO, captaincy-concentration,
field-covariance) has had to work around **zero real rival-squad data** — confirmed
repeatedly: no FPL leaderboard or individual-entry data is ingested anywhere. That constraint
does not have to hold for Priority 10.

**The public FPL API exposes real individual managers' picks** via unauthenticated endpoints
(`entry/{id}/event/{gw}/picks/`, `leagues-classic/{id}/standings/`, `entry/{id}/history/`).
This is genuinely different from Understat's situation (Priority 7a) — this is FPL's own
official API, not a third-party site being scraped, and per-entry picks for a given completed
gameweek are public information the game itself displays. This means Priority 10 does **not**
need to fabricate a rival-squad model or an archetype clustering from imagined data — it can
sample real squads and build a genuine empirical distribution from them.

This is the single most important design decision this doc makes, and it should be confirmed
with the user before Phase A starts (see §6).

## 3. Target architecture

Four capabilities, replacing the mean-field EO portfolio with a genuine empirical
rival-squad distribution and a joint (not independent) simulation against it:

1. **A real sample of rival squads** for a target gameweek — e.g. the Overall top-N
   (some hundreds, not millions — see §5 on rate limits) or a specific large mini-league the
   user is in, each with real per-player ownership *and* real captaincy choice (unlike EO's
   own captaincy-concentration *estimate*, a real sample gets real captaincy directly).
2. **Joint simulation**, not N independent Monte Carlo runs: the manager's candidate squad
   and every sampled rival squad need to be simulated against the *same* underlying
   correlated match outcomes per fixture (shared `Z_fixture` draws), because that's where the
   real correlation comes from — two squads sharing a premium forward move together when that
   forward blanks or hauls, and that shared movement is exactly what a rank-delta signal has
   to capture. Independently simulating each squad separately (M6's current per-squad scope)
   would throw this away.
3. **A real rank-delta distribution** for the candidate squad against the sampled field —
   not a scalar proxy, an actual empirical distribution: "in what fraction of simulated
   gameweeks did this squad move up in rank against the sampled field, and by how much."
4. **A validation path** — compare the simulator's rank-delta predictions against realized
   outcomes where real historical rival-squad snapshots exist to check them against (a real,
   separately-scoped data gap — see §5).

## 4. Phased breakdown (multi-PR, per the roadmap's own instruction)

Each phase is independently reviewable and independently valuable — none blocks shipping the
others' groundwork, though C depends on A and B.

- **Phase A — `ingest_fpl_entry_picks.py`** (data) — **implemented**. Fetches a real,
  bounded sample of rival squads for a target gameweek from the public FPL API (Overall
  league by default, `n_entries=200`). New table `fact_rival_squad_sample` (player_uid ×
  entry × gw, captain flag, multiplier, public league rank — deliberately no manager/team
  name). Idempotent per (season, event). Same network-blocked-in-this-sandbox caveat as
  Understat (Priority 7a): the real API response shapes are taken from FPL's own extensive
  public documentation, not independently verified against a live fetch here —
  `_fetch_json()` is isolated for a one-line swap once verified. Deliberately a **separate**
  script (`scripts/run_rival_sample_ingestion.py`), not wired into `run_ingestion.py`'s
  default flow, so its own request volume doesn't ride along on every routine ingestion run.
- **Phase B — joint simulation engine.** Extends `monte_carlo.py` to simulate the manager's
  candidate squad *and* every sampled rival squad against shared per-fixture `Z_fixture`
  draws in one pass, not independent per-squad runs. This is the real technical core of "true"
  field simulation and the most implementation-risky phase — likely needs its own internal
  design pass once Phase A's real data shape is known (sample sizes, squad-legality edge
  cases in real historical picks, e.g. a rival on a since-injured captain).
- **Phase C — rank-distribution output + squad_optimizer integration.** A new
  `field_simulator.py` producing the real empirical rank-delta distribution, surfaced as a
  diagnostic (matching M6's own existing pattern: `monte_carlo.run()` only simulates the
  *already-solved* squad, never influences the MIQP search itself, because a full joint
  simulation across many candidate squads during solver search would be far too expensive).
  This likely **augments** field_covariance.py's cheap default rather than replacing it —
  the mean-field proxy stays the always-on default; the full simulator becomes an expensive,
  explicitly opt-in "deep field analysis" pass on the final squad, the same
  cost-tiering convention Priority 3's multi-transfer search and Priority 5's Free-Hit/TC
  combo already use.
- **Phase D — backtest validation.** Extends `backtest.py` to check the simulator's
  rank-delta predictions against realized outcomes. Real historical *per-gameweek* rival-squad
  snapshots are unlikely to be recoverable for old, already-completed gameweeks (the API
  reflects an entry's *current* picks more reliably than a full season history of past
  picks per gameweek) — this phase may end up validation-scoped to the current live season
  going forward, not the full 2024-25/2025-26 backtest window Priority 9 already covers. A
  real, disclosed limitation, not solved by this doc.

## 5. Open risks and questions — need your input before Phase A starts

- **Sampling source and size.** ~~"Overall top-N" (some hundreds)...~~ **Resolved (2026-09,
  Phase B rank instrumentation).** The top-N-by-rank scaffold measured the model against the
  200 best managers alive — the wrong bar for a top-100k goal. `DEFAULT_RANK_BANDS` now takes
  a **stratified** sample: ~120 around the target band (ranks 60k–140k), ~40 elite (top 10k,
  a ceiling reference), ~40 mid-field (400k–600k, a margin-over-median reference). ~200
  entries total, well below the old aspirational `n_entries=2000` — deliberately a volume
  whose deep-pagination reach is observable (`fetch_entries_in_rank_bands` reports the bands
  it actually reached) rather than assumed. Mini-league sampling is still available via
  `fetch_top_entries`/`ingest_rival_squad_sample(n_entries=)` but is not the default.
- **Rate limiting and respectful use.** Sampling even a few hundred real entries means a few
  hundred real HTTP requests per gameweek to FPL's own API — needs real backoff/caching
  discipline (fetch once per gameweek, cache the sample, never re-fetch on every report run),
  and should stay well inside whatever the FPL API's own informal rate tolerance is.
- **Data-handling scope.** Individual managers' picks are public within the game, but
  *aggregating and storing* many real people's team data at scale is a step up from a single
  glance at a leaderboard page — worth being deliberate about before building it, not an
  automatic yes just because the data is technically public.
- **Performance.** Joint simulation across a real rival sample (Phase B) at the cost of a
  single already-expensive M6 Monte Carlo run, multiplied by however many rivals are sampled,
  needs a real performance budget decided before implementation — this could be materially
  more expensive than every other opt-in feature built so far in this project.

## 6. Recommendation

Don't start Phase A until the sampling-source and data-handling questions above are answered
explicitly — everything else in this doc can be refined once real sample data exists to
design Phase B/C against, but those two questions gate whether Phase A should be built at
all, and how.
