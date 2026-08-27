# Plan: Retrospective Historical Validation (Track E) and ML Residual Challenger Definitive Decision (Track F)

One-line goal: two independently shippable pieces of evidence exist that didn't before — a real,
honestly-labeled, genuinely blind comparison of the engine's own 2025-26 season strategy against
a representative sample of real managers' actual outcomes, and a real, statistically definitive
ship/no-ship call on whether a LightGBM residual correction beats the existing Quant model —
both produced from data and code that were verified to exist, not assumed to.

**Revision note:** this plan was run through two rounds of blind Critique Engine review after its
first draft (see the closing section). Round 1 found two design-level errors: the original
sampling approach would have compared against the global top-ranked elite, not real managers
generally; and using currently-active (post-recalibration) parameter versions would have
simulated 2025-26 with hindsight the engine never actually had. Round 2 verified both fixes hold
up at the source-code level, but caught two things the fixes themselves introduced: an Open
Items fallback that silently reintroduced the elite-bias bug it was supposed to guard against,
and an incomplete parameter list (`run_season_simulation()` needs 18 version kwargs; the first
fix only addressed 8). All four are corrected below.

## Classification

Two independent tracks, continuing this repo's existing `docs/plans/2026-08_roadmap_plan.md`
lettering (Tracks A-D already used there):

- **Track E — Retrospective Historical Validation**: Feature (new capability; reuses
  `run_season_simulation()`, adds new genuinely-random sampling and a new comparison/report layer
  on top — the earlier idea of reusing `fetch_top_entries()` wholesale was corrected, see below).
- **Track F — ML Residual Challenger, Definitive Decision**: Feature, with the first phase being
  dependency installation — the experiment scaffold (`research/ml/`) already exists from an
  earlier session; this track's job is to run it for real, for the first time, with LightGBM
  wired in as the primary nonlinear challenger.

Explicitly out of scope (see Out of Scope below): Track A (Priority 10 field simulator), Track C
expansion to more managers, a Stripe/paid-API layer, XGBoost as an unconditional fourth arm, and
any recovery/investigation of the `fpl-quant-v2-REAL-LOCAL-SAFE` directory's deleted working tree.

## Interview Ledger

1. Q: Plan a from-scratch rebuild per the attached generic FPL-system prompt, or extend the
   existing `fpl-quant-v2` roadmap? → **Extend the existing roadmap** (user), in this order:
   (1) generalization backtest across many real historical squads, (2) resolve the ML-challenger
   experiment to a ship/no-ship call using already-logged results, (3) only then revisit
   Stripe/paid-API on top of (not instead of) the existing free-tier architecture.
2. Q: "Generalization backtest" — build the retrospective engine-vs-real-season-totals
   comparison (usable immediately on already-ingested data), the forward expansion of Track C to
   more real managers (usable only as future gameweeks accumulate), or both? → **Both,
   retrospective first** (user) — with an explicit requirement that the retrospective number's
   methodology must state plainly it is the engine's own from-GW1 strategy vs. real managers'
   actual outcomes, never framed as replaying real managers' decisions.
3. Q: What population scale/privacy posture for the real-manager data both pieces touch — large
   anonymized sample, or a small named set? → **Large anonymized sample** (user), extending
   Priority 10's existing no-names precedent.
4. Q: For Track C's expansion specifically — keep its public git-committed mechanism at scale
   (real strangers' identities exposed, verified live), move it private, or don't expand it at
   all? → **Don't expand Track C** (user) — retrospective sample alone carries the claim; Track C
   stays exactly as it is today (2 known/consenting accounts).
5. Q: `fpl-quant-v2-REAL-LOCAL-SAFE`'s entire working tree, including its own database copy, has
   been deleted since last night's run; the main directory's own database is separately intact.
   Investigate first, or proceed on the main directory? → **Proceed** (user).
6. Q: Wire the retrospective number straight into the public `track-record.html` page, or stop at
   a written report for human review first? → **Report only** (user).
7. Q: Is scikit-learn's `GradientBoostingRegressor` an acceptable stand-in for "gradient
   boosting," or install LightGBM/XGBoost specifically? → **Install scikit-learn AND LightGBM
   now**, wire LightGBM in as the primary nonlinear challenger with ridge kept as a control arm;
   XGBoost added only later, conditionally. The decision is not "definitive" unless it compares
   Quant-alone / Quant+ridge / Quant+LightGBM on identical leakage-safe folds, reporting per-fold
   results, aggregate metrics, bootstrap confidence intervals, compute/runtime, and season-points
   decision impact (user, fully specified).

Seven questions spent, plus a blind Critique Engine pass after drafting (see Current State and
Key Decisions) that corrected two design choices the interview itself didn't surface: the
sampling method's elite-bias, and the hindsight-parameter risk in the engine's own simulated run.
No further forks remain that would change the shape of the plan.

## Current State (verified this session — live checks and file citations, reproducible)

- `fpl-quant-v2` (main directory) is clean on `master`, matches `origin/master`, and has its own
  intact, populated 2.2GB `db/fpl_quant_v2.duckdb` (verified: `ls -la db/`, unchanged since
  Aug 13). This is the sole working directory for this plan (per Q5).
- M0-M9 (verified: `README.md` Status section) and Tracks B/C/D of the existing roadmap
  (verified: `docs/plans/2026-08_roadmap_plan.md`, all phases `[x]`) are done. Track A (Priority
  10 field simulator) is designed but not built (all phases `[ ]`) — untouched by this plan.
- `research/ml/` (Phase-0 ML-vs-Quant challenger) is fully scaffolded, leakage-checked, and
  tested against synthetic fixtures, but has **never been run against real data** — verified:
  `research/ml/results/` does not exist as a directory yet, `REPORT.md` is still the unfilled
  template. `ep_outputs` — the table the real experiment would read from — **is** genuinely
  populated: `SELECT count(*) FROM ep_outputs` against the real `db/fpl_quant_v2.duckdb` returns
  **66,974 rows** (verified live this session; resolves what was originally an open assumption).
- `research/ml/residual_model.py` **already contains a working sklearn-based GBM arm**
  (`GradientBoostingResidualModel`, using `HistGradientBoostingRegressor`), wired into
  `experiment.py` as `"quant_gbm"`, gated only on sklearn being installed. Its own docstring
  states LightGBM/XGBoost were deliberately *not* added, citing the original research spec's
  §10: "do not add a large dependency unnecessarily." Installing scikit-learn (R7 below) will
  silently activate this existing arm — this plan states explicitly how it's treated (see Key
  Decisions and R16) rather than letting it appear unannounced in Phase F-4's output.
- The "overnight walk-forward results" referenced in the original ask (`data/outputs/
  walk_forward_best.json`, `walk_forward_trials.csv`) are from a **different, disconnected**
  script (`scripts/run_walk_forward.py`) — a standalone ridge window/alpha search with no
  Quant-baseline comparison and no leakage-protocol integration. Not reused by this plan.
- None of scikit-learn, LightGBM, or XGBoost are installed or listed in any requirements file
  (verified: direct import check in the project's own `.venv`).
- Live-verified against the real FPL API this session, with reproducible detail:
  - `GET https://fantasy.premierleague.com/api/entry/1/event/1/picks/` → HTTP 200, real picks —
    this is entry 1's picks for the *current* season's GW1 (confirmed by cross-checking against
    `entry/1/history/`'s `"current"` array, which shows exactly one completed event).
  - `GET https://fantasy.premierleague.com/api/entry/1/event/38/picks/` → HTTP 404 `{"detail":
    "Not found."}` — GW38 is a gameweek number that only ever belonged to a *past, completed*
    season under the current season's active numbering; the endpoint will not serve it.
  - `GET https://fantasy.premierleague.com/api/entry/1/history/` → HTTP 200, includes a `"past"`
    array with real season-total points/rank for every prior season back to 2014/15 — confirming
    season-level totals *do* survive for past seasons even though per-gameweek picks don't.
  - `GET https://fantasy.premierleague.com/api/entry/1/` → HTTP 200, includes
    `"player_first_name":"Chris","player_last_name":"Musson"` — confirming any `entry_id`
    trivially resolves to a real, unauthenticated first/last name.
  - `GET https://fantasy.premierleague.com/api/bootstrap-static/` → `"total_players": 9824056` —
    the real, current count of active FPL entries this season, used below as the basis for
    genuinely random (not top-ranked) sampling.
- `fit_seasons_for()` (`src/fpl_quant/backtest.py:68-82`) hardcodes 2024-2025 to fit only against
  itself and 2025-2026 to fit against `(2024-2025, 2025-2026)`. Combined with
  `has_fittable_history()`'s cold-start floor, this is why the M7 backtest skipped 2024-25 GW1
  specifically (confirmed via direct query: `fact_match` has all 38 gameweeks for both
  2024-2025 and 2025-2026). 2025-26 GW1 is the only season start point where "the engine played
  this from a true GW1" is honestly makeable via `run_season_simulation()`. The 2023-24 parquet
  data added today is not wired into this fitting path at all.
- **Critique finding, corrected below:** Track B's automated recalibration (already `[x]` done
  in the existing roadmap) has promoted `xi_params_version`/`rho_residual_params_version` etc. to
  newer versions, fit in part against 2025-26's own real outcomes. Simulating 2025-26 from GW1
  using *currently-active* parameter versions (as the plan originally specified) would give the
  engine hindsight it never had at the time — the same class of leakage `asof_scope()` exists
  elsewhere in this codebase specifically to prevent. Fixed in Approach/Phase E-2 below: the
  simulation uses the **default (v1) parameter versions**, not `active_recalibratable_versions()`.
  `run_season_simulation()` requires exactly 18 version kwargs (verified: its own signature,
  `backtest.py:920-945`); the real, existing production script `scripts/run_season_simulation.py`
  shows the authoritative split (verified: its own `_param_versions()` function) — **8 families
  are ever recalibratable** (`xi`, `rho`, `rho_residual`, `adjustment`, `shrinkage`,
  `fact_multiplier`, `lambda`, `kappa_tc` — exactly `RECALIBRATABLE_VERSION_ARGS`'s 8 keys,
  `backtest.py:1510-1524`, normally resolved via `active_recalibratable_versions()`) and **10
  families are never recalibratable at all** (`decay`, `scoring`, `bps`, `tau`, `corr`,
  `guardrail`, `horizon`, `transfer_cost`, `wildcard_threshold`, `free_hit_threshold` — always
  hardcoded to `1` everywhere in this codebase, including in this same real script). For a
  genuinely blind Track E run, **all 18** are hardcoded to `1` — including the 8 that are
  normally resolved via `active[...]` elsewhere, since using their currently-active values is
  exactly the hindsight risk being avoided here.
- **Critique finding, corrected below:** the original design reused `fetch_top_entries()`
  (`src/fpl_quant/ingest_fpl_entry_picks.py:97-121`) for Track E's comparison sample. That
  function's own docstring is "top n_entries by rank from a classic league's standings" — against
  the Overall league, that returns the highest-ranked ~2,000 managers *on Earth*, not a
  representative sample. Comparing the engine's simulated total to that elite subset and calling
  the result a "percentile" or "beats real managers" claim would be statistically misleading in
  either direction. Fixed in Approach/Phase E-1 below: genuine random `entry_id` sampling.
- `fpl-quant-v2-REAL-LOCAL-SAFE` (a sibling directory, one commit behind `origin/master`) has had
  its entire working tree — including its own local copy of the real database — deleted from
  disk since last night's walk-forward run, leaving only `.git`. Flagged to the user (Q5);
  explicitly not investigated or restored by this plan.

## Scope (v1)

1. **Track E**: draw a genuinely random sample of ~2,000 real FPL entries (uniform over the valid
   ID space, not top-ranked), pull their actual 2025-26 season totals via `history()`; run the
   engine's own strategy for 2025-26 from a true GW1 via `run_season_simulation()` using the
   **default, pre-recalibration parameter versions** (not currently-active ones, to avoid
   hindsight bias); compute percentile rank and point differential; write a report (not
   published) with a mandatory, explicit methodology disclaimer covering both the "not a replay
   of real decisions" and "blind simulation, not hindsight-tuned" points.
2. **Track F**: install and pin scikit-learn + LightGBM; wire LightGBM into
   `research/ml/residual_model.py` as the primary nonlinear challenger, ridge retained as control,
   the existing sklearn `quant_gbm` arm reported as a bonus informational fourth arm (not part of
   the ship/no-ship criterion, per R16); add bootstrap confidence intervals and compute/runtime
   instrumentation; run the real experiment for the first time against the production DB; record
   an explicit ship/no-ship decision in `REPORT.md` against a defined statistical-credibility bar
   that also checks calibration and per-slice performance.

## Out of Scope & Parked Items

- Track A (Priority 10 field simulator) — never part of this ask; stays exactly as documented.
- Track C's expansion to more real managers — explicitly declined (Q4).
- Stripe/paid-API layer — explicitly deferred (Q1); captured as an Open Item.
- XGBoost as an unconditional fourth arm — only added later if LightGBM's result is positive or
  borderline (Q7).
- Publishing Track E's number to `track-record.html` or any public surface — deferred to a
  separate follow-up after human review (Q6).
- Investigating or restoring `fpl-quant-v2-REAL-LOCAL-SAFE`'s deleted working tree — explicitly
  declined (Q5).
- Extending `fit_seasons_for()` to unlock 2024-25 GW1 using the new 2023-24 data — a real,
  non-trivial change to a calibrated model input; named in Open Items, not attempted here.

## Approach

**Track E** reuses `run_season_simulation()` (bootstraps the engine's own M5 squad at a real GW1
and walks it forward on the engine's own M8-informed decisions) but calls it with **all 18
required parameter-version kwargs hardcoded to `1`** — adapting `scripts/run_season_simulation.py`'s
own `_param_versions()` function verbatim, but replacing every `active[...]` lookup (for the 8
families that are normally recalibratable) with a literal `1`, since using their currently-active
values is exactly the hindsight risk being avoided (see Current State for the full 18-name
breakdown). This is a deliberate, corrected choice so the simulation reflects what the engine
would genuinely have done at the time, not what current hindsight-tuned parameters make it look
like it would have done. Sampling real comparison data does **not** reuse `fetch_top_entries()`
wholesale (that function is right for Track A's field-diversity use case, wrong for a population
claim here, and must never be used as a fallback either — see Open Items); instead it draws
uniform-random integers over `[1, 10,324,056]` (`total_players` verified live at 9,824,056 this
session, plus a fixed 500,000 margin for continued signups before build time), calling
`history()` for each, discarding IDs that don't exist or have no `"2025/26"` row in `past` (never
played that season), until ~2,000 valid real season totals are collected. This is slower than
`fetch_top_entries()`'s paginated approach but is the only way to get a population-representative
sample rather than an elite one — with one disclosed, not fully solved, residual limitation (see
Risks & Landmines): uniform-random over the ID space skews toward accounts by signup recency, not
toward engagement, so it may still over-represent dormant "set once, never touched" accounts
relative to active managers. The comparison/report layer is new code.

**Track F** runs the existing `research/ml/` Phase-0 design for the first time against real data,
extended with LightGBM as specified in Q7. The experiment's dataset is built from `ep_outputs`
rows the M7 backtest already produced and persisted — verified live this session at 66,974 real
rows, so Phase F-4 does not need an unplanned backtest re-run. The already-existing sklearn
`quant_gbm` arm will activate once scikit-learn is installed (R7); rather than letting it appear
unannounced, this plan reports it as a bonus fourth, informational arm — the actual ship/no-ship
decision is governed by R11 applied specifically to the LightGBM arm, per the user's explicit Q7
instruction, not by whichever arm happens to look best.

## Requirements

- **R1**: WHEN Track E's engine-run phase executes THE SYSTEM SHALL bootstrap a real M5 squad at
  2025-26 GW1 via `run_season_simulation()`, passing **all 18 required version kwargs hardcoded
  to `1`** (both the 8 normally-recalibratable families and the 10 that are never recalibrated
  anywhere in this codebase) rather than resolving any of them via
  `active_recalibratable_versions()`, and walk forward to GW38 using the engine's own M8-informed
  decisions.
- **R2**: WHEN the sampling phase executes THE SYSTEM SHALL draw a **uniform-random sample of
  entry IDs** (not top-ranked) over the real, currently valid ID space and fetch real 2025-26
  season-total points via `history()` for ~2,000 entries that actually played that season.
- **R3**: THE SYSTEM SHALL compute and report the engine's simulated total's percentile rank and
  point differential against the real sampled distribution.
- **R4**: THE SYSTEM SHALL NOT store or surface any real manager's name, team name, or entry_id
  in any output artifact of Track E, **including log output**; retained/logged data is aggregate
  point totals and counts only.
- **R5**: THE SYSTEM SHALL produce a written report whose methodology section explicitly states
  (a) the comparison is the engine's own from-GW1 strategy versus real managers' actual outcomes,
  never a replay of real managers' decisions, and (b) the simulation used default, pre-
  recalibration parameters rather than current hindsight-tuned ones, and SHALL NOT wire this
  report into any public page.
- **R6**: Sampling SHALL reuse the existing `_fetch_json` backoff/retry pattern, SHALL cache the
  pulled sample rather than re-fetching on every run, and SHALL log a real measured wall-clock
  time and rejection rate (IDs that don't exist or didn't play 2025-26).
- **R7**: THE SYSTEM SHALL install scikit-learn and LightGBM as pinned dependencies.
- **R8**: `research/ml/residual_model.py` SHALL expose a LightGBM-backed residual model as the
  primary nonlinear challenger, conforming to the same interface `experiment.py` already expects,
  alongside the existing ridge (control) and sklearn `quant_gbm` (bonus/informational) arms.
- **R9**: THE SYSTEM SHALL run `research/ml/experiment.py` for real, for the first time, against
  the production database, computing Quant-alone, Quant+ridge, Quant+LightGBM, and (now that
  scikit-learn is installed) Quant+`quant_gbm` on identical leakage-safe walk-forward folds.
- **R10**: Reporting SHALL include per-fold results, aggregate metrics, bootstrap confidence
  intervals, compute/runtime per model, and season-points (decision-impact) results, for all
  arms produced.
- **R11**: The ship/no-ship decision SHALL be governed by the **LightGBM** arm's result: "ship"
  only if it shows a statistically credible improvement (confidence-interval-based, not a point
  estimate) in decision-relevant metrics without regressing calibration or per-slice performance
  (position/season/price/minutes/ownership/fixture-difficulty).
- **R12**: `research/ml/REPORT.md` SHALL be filled with real results for all arms and an explicit
  decision recorded in its existing §9 checklist, referencing R11's criterion by name.
- **R13**: XGBoost SHALL NOT be added in this pass; a follow-up decision SHALL be flagged if
  LightGBM's result is positive or borderline.
- **R14**: IF LightGBM fails to install or run in this environment THEN THE SYSTEM SHALL still
  report Quant-vs-ridge (and Quant+`quant_gbm` if available) and document LightGBM as
  unavailable, rather than blocking the whole workstream.
- **R15**: THE SYSTEM SHALL scrub `entry_id` from any persisted or CI log output during Track E's
  sampling, not only from the final cached/reported artifact.
- **R16**: THE SYSTEM SHALL clearly label the sklearn `quant_gbm` arm in `REPORT.md` as
  informational/bonus, explicitly stating it does not factor into the R11 ship/no-ship decision.

## Key Decisions

- Track E scoped to 2025-26 only, from a true GW1 — (verified: `fit_seasons_for()` +
  `has_fittable_history()` + real `fact_match` query) [A5].
- Track E's engine run passes **all 18 required parameter-version kwargs hardcoded to `1`**, not
  currently-active recalibrated ones — (Critique Engine finding, corrected across two rounds) —
  avoids simulating with hindsight the engine never had, since 2025-26's own outcomes fed the
  current recalibration [A10].
- Track E's comparison sample is **genuinely random over the real ID space (~9.8M entries,
  verified live), not `fetch_top_entries()`'s top-ranked mechanism** — (Critique Engine finding,
  corrected) — the latter would have compared against the global elite, not real managers
  generally [A1].
- Track E's real-manager comparison data stores aggregate point totals only, scrubbed from logs
  too — (user Q3/Q4, extended per critique finding #5) [A1].
- Track C is not touched by this plan — (user, Q4).
- LightGBM is the primary nonlinear residual challenger and the sole arm governing the ship/no-
  ship call; the pre-existing sklearn `quant_gbm` arm is reported as bonus/informational only,
  not silently left unaddressed — (user Q7, extended per critique finding #3) [A11].
- The ship/no-ship bar requires confidence intervals and per-slice checks, not a point-estimate
  MAE comparison — (user, Q7) [A7].

## Data & State Changes

- **Track E**: a new, uncommitted-to-git cache file for the sampled real season totals (no
  entry_id retained past the fetch step, and not present in log output either); a new report
  file, e.g. `docs/reports/2025-26_retrospective_validation.md`. No schema migration — read-only
  against existing `run_season_simulation()` machinery plus a new external fetch loop.
- **Track F**: `requirements.txt`/`requirements.lock`/`pyproject.toml` gain pinned
  `scikit-learn`/`lightgbm` entries. `research/ml/residual_model.py` gains a LightGBM code path.
  `research/ml/evaluate.py` gains bootstrap-CI and timing helpers. `research/ml/results/` and
  `REPORT.md` go from nonexistent/template to populated with real output for all four arms
  (Quant, ridge, LightGBM, `quant_gbm`). No production schema or live-recommendation-path
  changes — `research/ml/` remains explicitly non-production.

## Edge Cases & Failure Handling

- A randomly sampled entry_id doesn't exist, is private, or never played 2025-26 (no `"2025/26"`
  row in `past`) → discard and draw another; log only the aggregate rejection rate (R6), never
  the individual rejected IDs.
- Rate limiting across the (larger, due to rejection) number of real `history()` calls needed to
  reach ~2,000 valid entries → reuse existing backoff; fetch once, cache, never re-fetch on every
  report run.
- LightGBM fails to install or import on this environment → report Quant/ridge/`quant_gbm` only
  (whichever are available), document LightGBM as unavailable (R14) rather than blocking Track F.
- `ep_outputs` turns out not to be populated → already ruled out this session (66,974 real rows
  verified); Phase F-4 still re-checks as a cheap first-step guard in case the DB changes before
  build time.
- ML result is a near-tie or fails the per-slice check → recorded as an explicit "no-ship" in
  `REPORT.md`, consistent with this project's own "a negative result is a successful research
  result" framing.
- The engine's simulated 2025-26 total lands implausibly outside the real sample's range →
  treated as a signal to sanity-check the simulation (including confirming the default-parameter
  choice was applied correctly) before reporting, not reported as-is.

## Risks, Landmines & Adaptations

- **Real third-party data at new scale** (≈2,000+ entries touched, including rejected draws) →
  adaptation: aggregate-only retention, no names/entry_ids in any output artifact *or log* (R4,
  R15), extending Priority 10's precedent and closing the log-leakage gap a blind critique pass
  found in the first draft.
- **The retrospective number could be misread as "we replayed real managers' decisions and beat
  them"** → adaptation: the methodology disclaimer (R5) is a hard requirement, and the report is
  not published anywhere until human review (Q6).
- **The retrospective number could silently reflect hindsight-tuned parameters rather than a
  genuinely blind simulation** → adaptation (critique finding): R1/R5 mandate default (v1)
  parameter versions and require the report to say so explicitly.
- **The sampling method could silently compare against the global elite rather than real managers
  generally** → adaptation (critique finding, round 1): R2 mandates genuine random sampling over
  the real ID space, not `fetch_top_entries()`'s top-ranked mechanism — and (critique finding,
  round 2) this prohibition is now stated as unconditional, including in the Open Items fallback
  for a high rejection rate, which an earlier revision had accidentally reintroduced via a
  "subsample from a fetch_top_entries()-built pool" idea that doesn't actually avoid the bias.
- **Uniform-random ID sampling has its own, different skew** (toward accounts by signup
  recency/dormancy, not engagement) → **disclosed, not fully solved** (critique finding, round
  2): [A12] names this explicitly; the mean/median sanity check in Phase E-1 is the only current
  backstop, and this is flagged in Open Items as worth a closer look if the sanity check ever
  looks off, rather than assumed away.
- **`run_season_simulation()`'s 18-parameter call could be built incorrectly** (only 8 of 18
  addressed) → adaptation (critique finding, round 2): the full 18-name split, verified against
  `scripts/run_season_simulation.py`'s own real code, is spelled out in Current State, R1, and
  Phase E-2's Steps — a cold executor no longer has to rediscover this by reading source code.
- **`fpl-quant-v2-REAL-LOCAL-SAFE`'s working tree, including its own DB copy, was deleted since
  last night's run** → flagged to the user; the main directory's own DB is separately verified
  intact and is the sole directory this plan operates on (Q5).
- **The ML ship/no-ship call could look like it dodged the original "LightGBM/XGBoost" ask** →
  adaptation: LightGBM specifically installed and wired as the primary, ship-governing challenger
  (R11), with the pre-existing sklearn arm explicitly demoted to informational-only (R16) rather
  than left to silently compete for the headline result.
- **A single point-estimate MAE improvement could look like "ML wins" while hiding a regression
  in some slice or season** → adaptation: R11's ship bar requires confidence intervals and the
  existing per-slice check (`sliced_metrics`) to hold.

## Assumptions Ledger

| ID | Assumption | Basis | Blast radius if wrong | Check |
|----|-----------|-------|------------------------|-------|
| A1 | Retrospective sample: ~2,000 real entries via **uniform-random ID sampling** over `[1, 9,824,056+margin]` (verified live `total_players`), not top-ranked | corrected via blind Critique Engine pass — the original `fetch_top_entries()`-based design would have sampled the global elite | a materially misleading "percentile"/"beats real managers" claim if not fixed | Phase E-1's rejection-rate/timing log, plus a sanity check that the sampled distribution's mean/median look like plausible general-population FPL scores, not elite-only scores |
| A5 | Track E is scoped to 2025-26 only, from a true GW1 | `fit_seasons_for()` + `has_fittable_history()` + verified real `fact_match` coverage | a stakeholder expecting a 2-season number gets one season instead | stated in Current State/Key Decisions |
| A6 | Bootstrap CI uses 1,000 resamples at a 95% interval | standard default, no project-specific precedent to match instead | trivial — a config number | Phase F-3 |
| A7 | "Statistically credible improvement" reuses this project's own existing pattern from Track B's `evaluate_and_promote_proposal()` | consistency with an already-built, already-tested mechanism in this repo | none — deliberately conservative | Phase F-4 |
| A8 | scikit-learn/LightGBM versions: latest stable compatible with `requirements.lock`'s existing constraints at install time | normal dependency-management default | trivial — a version bump | Phase F-1 |
| A10 | Track E's engine run passes **all 18 required version kwargs hardcoded to `1`** (8 normally-recalibratable + 10 never-recalibratable), not `active_recalibratable_versions()`'s currently-active ones | corrected across two blind Critique Engine rounds — round 1 found the hindsight risk, round 2 found the fix only covered 8 of the 18 required kwargs; the full 18-name split is verified against `scripts/run_season_simulation.py`'s own real `_param_versions()` function | a headline number quietly inflated/deflated by hindsight, undermining the "if this engine had played 2025-26 blind" claim; or, if the 10 non-recalibratable kwargs were left unsupplied, the call would simply fail to execute | Phase E-2's done-check explicitly confirms all 18 versions used, logged in the run's audit trail |
| A12 | Uniform-random ID sampling skews toward accounts by signup recency, not by engagement — dormant "set once, never touched" accounts may be over-represented relative to active managers | disclosed, not fully solved: round 2 of the Critique Engine named this as a real residual risk the fix doesn't fully close | the sample could undercount actively-managed rivals even though it's no longer elite-only | Phase E-1's mean/median sanity check against known general-population FPL score figures is the only current backstop; named explicitly here rather than silently assumed away |
| A11 | The pre-existing sklearn `quant_gbm` arm is reported as bonus/informational, not part of the R11 ship criterion | corrected via blind Critique Engine pass — it would otherwise silently activate once sklearn is installed with no stated treatment | a confusing REPORT.md where it's unclear which arm actually decided ship/no-ship | R16, Phase F-4's done-check |

`ep_outputs` populated with 66,974 real rows was an open assumption in the first draft; verified
live this session (see Current State) and is no longer tracked as a ledger risk.

## Open Items (none blocking)

- Stripe/paid-API layer — deferred until Track E and F produce real numbers to decide from (Q1).
- XGBoost as a fourth arm — add only if LightGBM's Track F result is positive or borderline (Q7).
- `fpl-quant-v2-REAL-LOCAL-SAFE`'s deleted working tree — flagged, not investigated (Q5); worth a
  human glance to confirm the deletion was intentional.
- Whether to extend `fit_seasons_for()` using the new 2023-24 data to unlock 2024-25 GW1 — a
  real, non-trivial change to a calibrated model input; not attempted here.
- Once Track E's report is human-reviewed, a separate follow-up decides whether/how to publish it
  (Q6).
- Whether random-ID sampling's rejection rate (many IDs will be invalid/never-played) makes
  ~2,000 valid entries impractically slow to collect — Phase E-1's own first real run is the
  measurement. **The fallback, if rejection rate is impractically high, is to draw more random
  IDs (widen the draw count, not the method) or reduce the target sample size** — never to
  substitute any rank-ordered source such as `fetch_top_entries()`, including as a "pool" to
  subsample from, since a pool built entirely from standings is elite-only regardless of how it's
  subsampled afterward (this was a real error caught in Critique Engine round 2 in an earlier
  version of this Open Item; corrected here, and reiterated because Verification above forbids
  calling any rank-ordered endpoint unconditionally, with no exception carved out for this case).

## Verification

- Track E: `pytest tests/ -q` stays green after the new sampling/report code is added; the
  percentile/differential computation gets its own unit test against a synthetic distribution
  with a known answer; the sampling method's own test confirms it does not call
  `fetch_top_entries()` or any rank-ordered endpoint; a log-output test confirms no `entry_id`
  appears in captured log lines; the real `run_season_simulation()` invocation is sanity-checked
  against the real sampled distribution's range, and its done-check explicitly records which
  parameter versions were used.
- Track F: `pytest research/ml/tests/ -q` stays green after LightGBM is wired in; the real
  `python -m research.ml.experiment` run produces non-empty, non-template output in
  `research/ml/results/` for all four arms; `REPORT.md`'s §9 decision checkbox is explicitly
  checked based on the LightGBM arm alone, with `quant_gbm` clearly labeled informational.
- Cross-cutting: full existing suite (`pytest`, plus Node planner tests) stays green throughout —
  neither track touches any live production recommendation path.

## Build Phases

- [ ] **Phase E-1: Draw a genuine random sample of ~2,000 real 2025-26 season totals** *(risky —
  real third-party data at scale, and the sampling method itself was already found broken once by
  the Critique Engine; run it again at build time)*
  Done when: a cached dataset of ~2,000 real season-total point values exists, drawn via
  uniform-random `entry_id` sampling (not `fetch_top_entries()` or any rank-ordered source), with
  no entry_id or name retained past the fetch step **and none appearing in log output**, a logged
  rejection rate, and a logged wall-clock time; a sanity check confirms the sampled distribution's
  shape looks like a general population, not an elite-only one (e.g., compare its mean/median
  against publicly known "average FPL score" figures rather than assuming).
  Steps: draw random integers over `[1, 9,824,056 + margin]`; call `history()` per draw via the
  existing `_fetch_json` backoff pattern; keep only entries with a `"2025/26"` row in `past`;
  strip identifying fields before caching and before logging; log aggregate counts only.
  Covers: R2, R4, R6, R15; checks: [A1]
- [ ] **Phase E-2: Run the engine's own 2025-26 season from a true GW1, blind** *(risky — the
  parameter-version choice was already found to risk hindsight bias once by the Critique Engine;
  run it again at build time)*
  Done when: `run_season_simulation(con, "2025-2026", 1, 38, ...)` completes with a real final
  total and a full per-gameweek action log, **explicitly invoked with all 18 required version
  kwargs hardcoded to `1`** (not `active_recalibratable_versions()` for any of them); the
  done-check record lists all 18 kwarg names and their values used; the total is sanity-checked
  against Phase E-1's real distribution range before proceeding.
  Steps: write a `_blind_param_versions()` function adapting `scripts/run_season_simulation.py`'s
  own `_param_versions()` verbatim, replacing every `active[...]` lookup with a literal `1`
  (covering both the 8 normally-recalibratable families and the 10 that are hardcoded there
  already); invoke `run_season_simulation()` with it; persist the run's `run_id`, action log, and
  the full 18-kwarg dict used for audit.
  Covers: R1; checks: [A5], [A10]
- [ ] **Phase E-3: Compute the comparison and write the report** *(risky — this report's wording
  is the mitigation for two separate misreading risks; run the Critique Engine at build time)*
  Done when: `docs/reports/2025-26_retrospective_validation.md` exists, stating the engine's
  simulated total, its percentile rank and point differential against Phase E-1's real sample,
  and an explicit methodology section covering both (a) this is the engine's own from-GW1
  strategy versus real managers' actual outcomes, not a replay of any real manager's decisions,
  and (b) the simulation used default, pre-recalibration parameters, not current hindsight-tuned
  ones; the report is not linked from or wired into any public page.
  Steps: compute percentile/differential (unit-tested against a synthetic distribution with a
  known answer); write the report with both disclaimers; do not touch `track-record.html` or
  `index.html`.
  Covers: R3, R5; checks: [A1], [A10]
- [ ] **Phase F-1: Install and pin scikit-learn + LightGBM**
  Done when: both import successfully in the project's `.venv`; `requirements.txt`/
  `requirements.lock`/`pyproject.toml` list pinned versions; the full existing test suite stays
  green after the addition.
  Steps: `pip install scikit-learn lightgbm`; pin versions in the dependency files; run the full
  suite to confirm no regression.
  Covers: R7; checks: [A8]
- [ ] **Phase F-2: Wire LightGBM into `residual_model.py` as the primary challenger, and label
  the existing `quant_gbm` arm** *(risky — must preserve the existing leakage protocol exactly;
  run the Critique Engine at build time)*
  Done when: a LightGBM-backed residual model exists conforming to the same fit/predict interface
  `experiment.py` already expects; the existing ridge and `quant_gbm` models are unchanged;
  `leakage_checks.assert_split_invariants` gates all arms identically; unit tests (extending
  `test_residual_model.py`) cover LightGBM, including the R14 fallback when it's unavailable;
  `quant_gbm`'s informational/non-ship-governing status (R16) is documented in the module.
  Steps: add the LightGBM wrapper; wire the R14 fallback path; extend tests; document `quant_gbm`
  treatment; confirm no leakage check is bypassed for any arm.
  Covers: R8, R14, R16
- [ ] **Phase F-3: Add bootstrap confidence intervals and compute/runtime instrumentation**
  Done when: `evaluate.py`'s comparison output includes a bootstrap CI per metric per model
  (all four arms), and each model's real fit/predict wall-clock time is captured and reported;
  unit tests cover the CI helper against a synthetic case with a known interval.
  Steps: add `bootstrap_ci()`; wire timing capture around each model's fit/predict calls; tests.
  Covers: R10; checks: [A6]
- [ ] **Phase F-4: Run the real experiment and record the definitive decision** *(risky — this is
  the actual ship/no-ship call the whole workstream exists to produce; run the Critique Engine at
  build time)*
  Done when: `research/ml/results/` holds real, non-template output for all four arms
  (Quant-alone, Quant+ridge, Quant+LightGBM, Quant+`quant_gbm`) on identical leakage-safe folds;
  `REPORT.md` is filled with real numbers including per-fold results, aggregate metrics,
  bootstrap CIs, compute/runtime, and season-points impact; §9's decision checkbox is explicitly
  marked based on the **LightGBM arm alone** (R11), with `quant_gbm` clearly labeled
  informational (R16) and the per-slice check confirmed before any "ship" verdict; if LightGBM's
  result is positive or borderline, a flagged follow-up recommends whether to add XGBoost (R13);
  a measured total runtime is logged, with an explicit note if any fold had to be abandoned for
  time and why.
  Steps: confirm `ep_outputs` still has real rows (cheap re-check of the 66,974-row fact already
  verified this session); run `python -m research.ml.experiment` for real; fill `REPORT.md`;
  apply R11's ship bar to the LightGBM arm specifically; record the decision.
  Covers: R9, R11, R12, R13; checks: [A7]

## The Critique Engine (for Build Phases marked *risky* above)

This plan is important enough to warrant it, and a first blind pass on the plan document itself
already found two design-level errors before any code was written (see Current State/Key
Decisions) — direct evidence this step catches real problems here, not just process theater. For
every phase marked *(risky)* above — E-1, E-2, E-3, F-2, F-4 — the executor must, before marking
that phase done:

1. Build the phase.
2. Hand it to a blind critic (a fresh sub-agent if the executor's environment supports spawning
   one; otherwise a deliberate fresh-eyes reread as a hostile stranger who did not write it),
   armed with only: the phase's stated Done-when criteria, the relevant Key-Decision/Risk text
   from this document, and the diff — never the builder's own rationale.
3. The critic returns PASS/FAIL plus concrete required fixes, defaulting to "not good enough."
4. Apply every blocker, loop, stop at consensus or after three rounds; any residual non-blocker
   weakness gets added to this document's Open Items, not hidden.

---

*This plan was produced via a structured "Dry-Run Interview" process: recon of the existing
codebase, roadmap docs, and live external state (the FPL API, this session's `.venv`, and all
three sibling working directories) — cited throughout via file:line references and reproducible
live-check results — seven targeted questions to the operator (Interview Ledger above), explicit
tagging of every assumption made on the operator's behalf (Assumptions Ledger above), and two
rounds of blind Critique Engine review on the finished draft (fresh sub-agents with no access to
this conversation, armed only with the plan, the repo, and a values rubric). Round 1 found and
drove the correction of two design errors the interview itself had not surfaced: the
sampling-population bias (comparing against the global elite, not real managers) and the
hindsight-parameter risk (simulating a season with parameters partly tuned on that season's own
outcomes). Round 2 verified both fixes against the actual source code — not just re-reading the
prose — and caught two things the fixes themselves had introduced: an Open Items fallback that
silently reintroduced the elite-bias bug, and an incomplete parameter list (8 of the 18 kwargs
`run_season_simulation()` actually requires). All four are corrected in this version, verified
against real code (`scripts/run_season_simulation.py`'s own `_param_versions()` function) rather
than inferred. Corrections to any assumption, decision, or scope call above are welcome — flag
them and this document will be revised.*
