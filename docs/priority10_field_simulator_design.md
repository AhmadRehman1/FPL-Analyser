# Priority 10 — Full Rival-Squad-Distribution Field Simulator: Design Doc

Status: **design only, no implementation**. Per the roadmap's own explicit instruction:
"this is a serious standalone project — do not start it until Priorities 0-6 are solid, and
scope it as its own multi-PR effort with its own design doc reviewed... before
implementation, not a single PR." Priorities 0-9 (excluding 7d, deliberately skipped — see
the session notes) are now merged or in review; this doc is that required review artifact,
written before any Priority 10 code exists.

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

- **Phase A — `ingest_fpl_entry_picks.py`** (data). Fetches a real, bounded sample of rival
  squads for a target gameweek from the public FPL API. New table
  `fact_rival_squad_sample` (player_uid × entry × gw, captain flag). Rate-limited and capped
  to a modest sample size (see §5) — this is the one phase genuinely blocked on a decision
  only the user can make (whose squads to sample, how many, how the sampling source is
  chosen), so it should not start without that answered.
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

- **Sampling source and size.** "Overall top-N" (some hundreds) is the most representative of
  genuine competitive rivals but requires paginating the real leaderboard API; a specific
  mini-league you're in is smaller, more directly relevant to your own rank, but less
  representative of the whole field. Which one Priority 10 samples from is a real product
  decision, not something to default silently.
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
