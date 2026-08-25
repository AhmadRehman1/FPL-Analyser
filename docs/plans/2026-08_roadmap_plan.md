# Plan: Field Simulator Phase B+, Automated Recalibration, Planner Decision Backtest, and Platform Polish

One-line goal: four independent workstreams from the existing roadmap (`docs/BUSINESS_PLAN.md`, `docs/priority10_field_simulator_design.md`) move from "designed but not built" or "not yet cadenced" to shipped, each with its own build phases an executor can run without further questions.

## Classification

This is a roadmap-level plan spanning four tracks, not a single track — the user explicitly asked for "all of the above [three named roadmap items] please and more" after being shown the roadmap. Each track is classified independently:

- **Track A — Priority 10 Field Simulator (Phases B/C/D + a scale-up)**: Feature (extends an existing, partially-shipped design).
- **Track B — Recalibration cadence + automated promotion**: Feature with a Refactor component (the active-parameter-version mechanism must move from hardcoded literals to a single source of truth before automation is safe).
- **Track C — Planner decision backtest (hold vs. use)**: Feature, built as new logging rather than reusing an existing unwired mechanism (see Track C's Key Decisions — this was a deliberate user override of my recommendation).
- **Track D — Platform polish (4 P2 items)**: bundle of small Feature/Quick-task items (origin-aware fetch, wiring the last unwired export, analytics, custom domain).

Parked secondary asks: none — full coverage was requested and is provided below. Two Track D items (analytics, custom domain) are planned and code-scaffolded but cannot be made *live* without the operator provisioning external infrastructure (an analytics host, a registered domain) — this is stated explicitly wherever it applies, not hidden.

## Interview Ledger

1. Q: What's the actual software task to plan? → "Bigger change" (user).
2. Q: Which bigger change (offered: Priority 10 Phase B+, recalibration cadence, planner backtest, or something else)? → "All of the above please and more" (user) — expanded scope to include the full remaining roadmap (P1 items + P2 items), not just the three named.
3. Q: Keep Priority 10's current sampling scale (n=200, Overall league) or scale up first? → **Scale up sampling first** (user, overriding my recommendation to keep current scale).
4. Q: Automate just the recalibration *run*, or the confirm/promotion step too? → **Automate confirm too** (user, overriding my recommendation to keep confirm manual) — flagged as a landmine, see Track B.
5. Q: Reuse the existing unwired `run_season_simulation()` walk-forward mechanism for the planner backtest, or build new decision logging? → **Build new decision logging from scratch** (user, overriding my recommendation to reuse the existing mechanism).
6. Q: Include analytics + custom domain (need external resources I can't provision) in this plan, or leave as open items? → **Plan all four P2 items now** (user).

Six questions spent, all answered with explicit directional choices (three of them overriding my recommendation — each is followed through as its own re-derived sub-plan below, not patched in one line, per the four "Key Decisions" sections). No further forks remain that would change the shape of the plan; remaining unknowns are default-and-tagged in the Assumptions Ledger.

## Current State (verified via 4 parallel recon passes)

- Repo is clean on `master`; no in-flight PR or branch work exists for any of these four items (verified: `git status`, `git log`).
- **Priority 10**: Phase A (real rival-squad ingestion) is implemented and tested but not scheduled — it's a manual-only script, not referenced anywhere under `.github/` (verified: `scripts/run_rival_sample_ingestion.py`, confirmed via grep; `README.md:402-406` states this explicitly). Phases B/C/D are design-only prose in `docs/priority10_field_simulator_design.md:87-109`, with no schema, function signature, or code existing yet (verified: no `field_simulator.py`, no joint-simulation test file found).
- **Recalibration**: the full mechanism already exists — `backtest.recalibrate()` (`src/fpl_quant/backtest.py:1955-2014`), `propose_recalibration`, a seed-file promotion path, and a manual human-review CLI (`scripts/review_recalibration.py`) — but it only *runs* weekly via `.github/workflows/weekly_backtest.yml:41-43` and promotion is 100% manual today; active parameter versions are hardcoded integer literals duplicated across `scripts/run_ingestion.py:158-248` and `scripts/export_track_record.py:38-45` (both verified).
- **Planner backtest**: the track-record page backtests per-player probabilistic calibration only (verified: `src/fpl_quant/reporting.py:520-570`, metric names at `src/fpl_quant/backtest.py:643-644,699-700`) — no transfer/hold decision is ever compared to its outcome anywhere. A parallel, unwired mechanism (`run_season_simulation()`, `src/fpl_quant/backtest.py:920-1095`) does simulate forward-walk planner decisions but is used only for an unrelated, also-unwired leaderboard feature (`scripts/export_leaderboard.py`) — this track deliberately does not reuse it (see Track C).
- **P2 items**: origin-aware fetch is fully absent from `index.html` (hardcoded raw URL at `index.html:662,1142-1158,1165`) but a working, production-tested equivalent already exists in `track-record.html:133-158`. Of the "3 unwired `_latest` exports," 2 are now actually wired (`export_projections.py` into `scheduled_pipeline.yml:238-241`; `export_leaderboard.py` into `weekly_backtest.yml:138-139`) — `docs/BUSINESS_PLAN.md` is stale on this point. Only `track_elite.py` remains unwired, gated on `data/elite_managers.json` currently having an empty `managers` list (verified: `data/elite_managers.json:2-3`). Analytics and a CNAME/custom-domain step are both fully absent (verified via repo-wide grep, zero matches).

## Scope (v1)

All four tracks, each shipping independently:
1. Priority 10: scale-up validation → joint simulation (Phase B) → rank-distribution diagnostic (Phase C) → current-season-only backtest validation (Phase D).
2. Recalibration: single source of truth for active parameter versions → regression-gated automated promotion → wired into the existing weekly cadence, with full audit trail and a reversible pointer.
3. Planner backtest: log the planner's recommendation at report-generation time → realize the outcome once results land → surface an aggregate once enough gameweeks exist.
4. Platform polish: origin-aware fetch in `index.html`; wire `track_elite.py`; scaffold self-hosted analytics; scaffold a custom domain.

## Out of Scope & Parked Items

- Adding specific mini-leagues to Priority 10's sampling (would need a mini-league ID from the operator) — parked as Open Item, not blocking the scale-up as scoped.
- Changing the underlying recalibration math (xi/rho/minutes/lambda refits) — untouched; only the *promotion decision* is automated.
- Feeding Priority 10's field-simulator diagnostic into `squad_optimizer`'s actual search — stays diagnostic-only, matching the design doc's own stated intent (`docs/priority10_field_simulator_design.md:93-102`).
- Reconstructing planner decisions from before this plan ships — `report_history/`'s existing snapshots have no labeled recommendation field (verified: `data/report_history/2026-2027_gw14.json` keys), so the decision log starts counting from Phase C-1's ship date, not retroactively.
- Actually registering a domain or standing up analytics hosting — I can write code and config, not provision external infrastructure or spend the operator's money.

---

## Track A — Priority 10 Field Simulator (Phases B/C/D)

### Approach

**A0 (new prerequisite phase, added because of Q3's answer):** scale up Phase A's sampling before building on top of it. The design doc's own unresolved questions (`docs/priority10_field_simulator_design.md:111-129` — sampling size, rate-limit/caching discipline, data-handling scope, performance budget) are about scaling *beyond* the shipped n=200 baseline, so scaling up must re-validate them, not skip them.

Then Phase B (joint simulation core), Phase C (rank-distribution diagnostic, opt-in/diagnostic-only), Phase D (backtest validation, current-season-only — real historical rival snapshots predating Phase A's ingestion start are unrecoverable, already disclosed in the design doc).

### Key Decisions

- **[A1] Sampling scale: 2,000 entries from the Overall league** (10x the current 200), Overall-league-only (no additional mini-leagues — none was named). *Basis:* [assumed: 2,000 as a round order-of-magnitude increase — if wrong: the A-1 build phase measures real timing/error-rate at this scale first, so the number is easy to revise before Phase B depends on it]. Blast radius if wrong: re-run A-1 at a different N; nothing downstream breaks because Phase B reads whatever sample size exists.
- **[A9] Retention policy: `fact_rival_squad_sample` retains only the current season's rows.** A scheduled purge removes rows from any prior season. *Basis:* Phase D (backtest validation) is already scoped to the current live season only (verified: `docs/priority10_field_simulator_design.md:103-109`) — no downstream use case in this plan ever reads a prior season's rival-squad sample, so nothing is lost by not keeping it. This is the actual engineering control for the landmine named in Risks & Landmines below ("increased real personal data retention at scale") — rate-limiting (A-1) and excluding mini-leagues bound *load* and *breadth*, but only this purge bounds the *volume of real people's picks retained over time* as sampling scales 10x. Without it, "scale up sampling" plus "keep running this indefinitely" means unbounded accumulation of real managers' squad choices, which the no-names design in `fact_rival_squad_sample` (schema `0017_priority10_phaseA_rival_squad_sample.sql`) reduces the sensitivity of but does not itself bound the volume of.
- Phase C remains diagnostic-only, never wired into `squad_optimizer`'s search (verified: `docs/priority10_field_simulator_design.md:93-102` — matches the existing, already-shipped pattern of `monte_carlo.run()` never influencing the MIQP search).
- Phase D is scoped to the current live season only (verified: `docs/priority10_field_simulator_design.md:103-109`).

### Data Changes

New additive migration `schema/00NN_priority10_phaseB_joint_simulation.sql` (exact number assigned at execution time, after whichever of Track A/B/C is built first) adding a table for per-run diagnostic output (candidate identifier, season, gameweek, rank-distribution summary). No existing table is altered. A season-scoped purge (per [A9]) deletes rows rather than adding a table.

### Edge Cases

- A gameweek with too few sampled rival squads (e.g., early in a season before Phase A had accumulated data) → Phase C reports "insufficient field sample," not a fabricated distribution.
- A sampled entry that goes private or is deleted between ingestion and simulation → already handled by Phase A (404→`None`, verified: `ingest_fpl_entry_picks.py`).
- A purge (per [A9]) runs while a backtest validation (Phase D) job for the current season is mid-run → purge only ever targets rows tagged with a season strictly earlier than the current one, so it cannot race with or delete data a same-season job is using.

### Requirements

- **R1**: WHEN the operator runs the scaled-up ingestion THE SYSTEM SHALL complete within a measured, logged time/error-rate budget and record that measurement in this doc.
- **R1b**: WHEN a season rolls over THE SYSTEM SHALL purge `fact_rival_squad_sample` rows belonging to any prior season, retaining only the current season's data.
- **R2**: WHEN Phase B is invoked with a fixed random seed THE SYSTEM SHALL produce identical simulated totals across repeated runs (determinism, matching `field_covariance.py`'s existing test pattern).
- **R3**: WHEN Phase C's diagnostic is not explicitly opted into THE SYSTEM SHALL NOT invoke it from `squad_optimizer`'s default path.
- **R4**: WHEN backtest validation runs on a gameweek with no rival-squad samples THE SYSTEM SHALL report "insufficient sample," not a distribution.

### Build Phases

- [ ] **Phase A-1: Scale up ingestion at n=2,000, with a season-scoped retention policy** *(risky — run the Critique Engine at build time, see below)*
  Done when: a manual run of `run_rival_sample_ingestion.py 2000` completes with a logged error rate under 5% and a recorded wall-clock time; `tests/test_ingest_fpl_entry_picks.py` gains a parametrized larger-n case (mocked); a purge routine exists and is tested to confirm it deletes only prior-season rows and leaves the current season's rows untouched.
  Steps: bump the default/CLI arg; add timing + error-rate logging; run once against a real live gameweek; add the test case; record the actual measured numbers back into [A1] above; add the season-scoped purge (per [A9]) and a test covering both the delete and the preserve path.
  Covers: R1, R1b; checks: [A1], [A9]
- [ ] **Phase A-2: Joint simulation core (Phase B)** *(risky)*
  Done when: a new function produces per-draw simulated totals for the candidate squad plus every sampled rival squad, unit-tested for determinism and against a hand-computed 2-rival-squad case.
  Steps: add `joint_field_simulation.py` (new module, to avoid overloading `monte_carlo.py`'s existing scope) reusing `deterministic_seed`/`sample_z_fixture`; write unit tests; document the I/O contract in the module's header, mirroring `field_covariance.py`'s docstring style.
  Covers: R2
- [ ] **Phase A-3: Rank-distribution diagnostic (Phase C)** *(risky)*
  Done when: `field_simulator.py` exposes a function returning rank-delta distribution stats, callable only as an explicit opt-in (matching Priority 3/5's cost-tiering), with a test confirming it is never invoked from the default optimizer path.
  Steps: build `field_simulator.py`; wire the opt-in flag; persist output to the new schema table; tests.
  Covers: R3
- [ ] **Phase A-4: Current-season backtest validation (Phase D)**
  Done when: `backtest.py` can replay Phase C's diagnostic for any gameweek with rival-squad samples and compare it to real final rank movement, with an explicit insufficient-sample guard for earlier gameweeks.
  Steps: extend `backtest.py`; add the guard; tests; document the current-season-only scope directly in the function's docs.
  Covers: R4

---

## Track B — Recalibration Cadence + Automated Promotion

### Approach

Move from "recalibration runs weekly, a human manually confirms via `review_recalibration.py`" to "recalibration runs weekly and promotes itself when safe." This requires two things the roadmap item's one-line description doesn't mention but which recon surfaced as necessary: (1) a single source of truth for "the active parameter version," replacing the hardcoded literals in `run_ingestion.py` and `export_track_record.py`, since automated promotion has nothing to update otherwise; (2) a compensating safety control in place of the human reviewer, since versions the auto-promoted parameters are immutable but *live* — a bad promotion would silently degrade real user-facing recommendations with nobody watching.

### Key Decisions

- **Automate confirm too, not just the run** (user, overriding my recommendation). This is the plan's single highest-blast-radius decision: it removes a deliberate, already-shipped human review gate (`review_recalibration.py`, only `status=='confirmed'` rows are ever auto-loaded, verified: `backtest.py:1405-1467`) from a path that drives what real users see. It is followed through here, not patched with one line — see [A2]/[A3] and the compensating controls below, and Risks & Landmines.
- **[A2] Regression threshold: no backtest metric (`log_score_*`, `brier_*`) may regress by more than 2% relative to the currently active version, or the proposal stays `pending` for manual fallback review.** *Basis:* [assumed — this specific number is a placeholder; the operator should tune it once real proposal history exists to show what "normal" week-to-week metric noise looks like. If wrong: too loose lets a bad recalibration through, too tight blocks every proposal — Phase B-2's done-check requires testing both a promote and a block case, which will surface if 2% is unworkably tight or loose on real data]. This is the minimum safe reading of "automate confirm" — automating promotion with *zero* guard was not, in my judgment, actually what "automate confirm too" was asking for as opposed to "remove all safety," so a guard is included as a compensating control rather than asked about again.
- **[A3] Active-version source of truth: new table `active_param_versions(family, key, version, promoted_at, promoted_by)`.** Executor's choice of exact column shape; the requirement is a single place every script reads from.
- Reversibility: because `params.py`'s versions are immutable and append-only (verified: `params.py:28-119`), "rollback" is just re-pointing `active_param_versions` at a prior version — no data is ever deleted, so this is cheap and safe to implement.
- `review_recalibration.py`'s manual `--confirm`/`--reject` stays usable as an override/audit tool even though it's no longer the only path to promotion.

### Data Changes

New additive migration adding `active_param_versions` (per [A3]) and a `status='auto_confirmed'` value alongside the existing `recalibration_proposals.status` values (verified table: `schema/0008_m7_backtest.sql:61-77`). No destructive change.

### Edge Cases

- A proposal that improves some metrics and regresses others → any single metric crossing the threshold blocks promotion (conservative default).
- Two candidate proposals for the same `(family, key)` in one run → promote at most one, latest wins (executor's choice).

### Requirements

- **R5**: WHEN any script needs the active version for a parameter family/key THE SYSTEM SHALL read it from `active_param_versions`, not a hardcoded literal.
- **R6**: WHEN a recalibration proposal's backtest metrics are within the regression threshold of the active version's THE SYSTEM SHALL auto-promote it and log the decision; WHEN not, THE SYSTEM SHALL leave it `pending` with a logged reason.
- **R7**: WHEN an operator needs to undo an auto-promotion THE SYSTEM SHALL support re-pointing the active version to any prior (immutable, still-intact) version via a documented command.

### Build Phases

- [ ] **Phase B-1: Single source of truth for active parameter versions**
  Done when: `run_ingestion.py` and `export_track_record.py` read from the new table instead of hardcoded literals; behavior is unchanged because the table is seeded with today's literal values; full existing test suite still passes.
  Steps: schema migration; `resolve_active_version()` helper in `params.py`; replace call sites; seed the table; run the full suite.
  Covers: R5
- [ ] **Phase B-2: Regression-gated automated promotion** *(risky — run the Critique Engine at build time)*
  Done when: a new function evaluates each pending `recalibration_proposals` row against the active version's own backtest metrics and either promotes (updates `active_param_versions`, marks `auto_confirmed`) or leaves it `pending` with a logged, machine-readable reason; unit tests cover both the promote and the block path using [A2]'s threshold.
  Steps: implement the comparison; extend status handling; log every decision; tests for both paths.
  Covers: R6; checks: [A2]
- [ ] **Phase B-3: Wire into the weekly cadence + reversibility** *(risky)*
  Done when: `weekly_backtest.yml`'s existing Sunday run calls Phase B-2's function right after `recalibrate()`; a documented rollback command exists and is tested end-to-end on a scratch DB, confirming it restores prior behavior exactly.
  Steps: workflow edit; write/document the rollback command (extends `review_recalibration.py`'s CLI shape); dry-run by invoking `scripts/run_backtest.py` followed by Phase B-2's promotion function directly via CLI against a scratch copy of the DuckDB file — the same way `tests/test_backtest.py` already exercises `recalibrate()`, not by running the YAML workflow through a GitHub Actions emulator; verify rollback.
  Covers: R7; checks: [A3]

---

## Track C — Planner Decision Backtest (Hold vs. Use)

### Approach

Log what the planner recommends for the operator's tracked manager(s) — the same scope already covered by the existing scheduled report-generation path (`data/report_history/`) — then, once real results for that gameweek are known, record whether following the recommendation would have scored better than holding. Surface an aggregate once enough gameweeks have accumulated.

### Non-Goal (explicit, load-bearing — this is the guardrail for the riskiest part of this track)

This track never touches `planner/storage.js`'s locally-drafted plans (verified: `planner/storage.js:1-11` — user drafts are `localStorage`-only, never transmitted, and the app's own privacy statement says it "sends nothing about the user anywhere," `docs/BUSINESS_PLAN.md:100-105`). It only extends the *server-side* report-generation path that already runs for the operator's own tracked team via the scheduled pipeline. If this track were ever extended to other users' planner sessions, that would contradict an already-shipped privacy commitment and needs its own explicit decision — flagged in Open Items, not silently done here.

### Key Decisions

- **Build new decision logging, not the existing `run_season_simulation()` mechanism** (user, overriding my recommendation). *Trade-off made explicit:* `run_season_simulation()` already works today and would show results immediately; new logging is slower (needs real gameweeks to pass) but reflects what actually happened to the real tracked manager rather than a simulated walk-forward. `run_season_simulation()` is left untouched, still serving its existing (also unwired) leaderboard use.
- **[A4] Minimum sample size before surfacing publicly: 8 completed gameweeks.** *Basis:* [assumed — no variance data exists yet to justify a specific number; if wrong, adjust once real week-to-week variance in the logged outcomes is visible]. Below this, the page shows an honest "not enough data yet" state, mirroring the site's existing "un-recalibrated" honesty pattern (`docs/BUSINESS_PLAN.md:52-53`).

### Data Changes

New additive table `planner_decision_log(season, gameweek, recommended_action, actual_action_taken, realized_points_actual, realized_points_if_recommendation_followed, logged_at)`.

### Edge Cases

- A gameweek where the recommendation was explicitly "hold" → still logged as a record (not an absence of a record), so "hold was correct" is measurable too.
- A recommendation that depended on a doubtful-flag player's late team news → logged as given at generation time, never retroactively adjusted (matches how `report_history/` already snapshots at generation time).

### Requirements

- **R8**: WHEN a scheduled report is generated for the tracked manager THE SYSTEM SHALL log the planner's recommended action (transfer, captain, chip, or explicit hold) to `planner_decision_log`.
- **R9**: WHEN the next gameweek's real results are ingested THE SYSTEM SHALL back-fill that gameweek's logged row with realized actual points and realized if-followed points.
- **R10**: WHEN fewer than [A4]'s minimum gameweeks have been realized THE SYSTEM SHALL show an explicit "not enough data yet" state instead of a computed aggregate.

### Build Phases

- [ ] **Phase C-1: Log the recommendation at generation time** *(risky — touches what data is captured about a real report; run the Critique Engine at build time to confirm the non-goal above is actually honored in the diff)*
  Done when: every scheduled report-generation run appends a record to `planner_decision_log` for the tracked manager(s), verified by inspecting the table after one real scheduled run; the "hold" case is explicitly unit-tested (must log a record, not skip).
  Steps: schema migration; extend the existing report-generation script; unit test the hold case.
  Covers: R8
- [ ] **Phase C-2: Realize the outcome once results are known**
  Done when: a step added immediately after the "Run ingestion (M0-M6)" step in `scheduled_pipeline.yml` (verified: `.github/workflows/scheduled_pipeline.yml:133-134` — this is where each run's real gameweek results become available in the DB) fills in the two realized-points fields for the prior gameweek's row, tested against at least one real completed gameweek's fixture data.
  Steps: add the new step right after `scheduled_pipeline.yml:134`; test with a known-result fixture.
  Covers: R9
- [ ] **Phase C-3: Surface the aggregate once enough data exists**
  Done when: `app_track_record.json`/`track-record.html` show a new "planner decision accuracy" section once [A4]'s threshold is met, and an honest below-threshold state before that; both states are tested.
  Steps: extend `export_track_record.py`/`reporting.py`; extend `track-record.html`; tests for both states.
  Covers: R10; checks: [A4]

---

## Track D — Platform Polish (P2 batch, all 4 items)

### Key Decisions

- **D1** ports `track-record.html:133-158`'s already-working `isLocalDev`/`DATA_BASE`/`RAW_FALLBACK` pattern into `index.html` verbatim, rather than inventing a new mechanism — it's already production-tested code.
- **D2** wires `track_elite.py` into `scheduled_pipeline.yml` (matching `export_projections.py`'s existing pattern, `scheduled_pipeline.yml:238-241`); it will safely no-op until `data/elite_managers.json`'s empty `managers` list is populated (matches commit `c381c36`'s own documented intent — `load_elite_managers()` is "empty by default, no entry_ids hardcoded anywhere") — populating it with real entry IDs is a content decision for the operator, listed in Open Items, not a code gap.
- **D3**: **[A5] analytics vendor: Plausible** [assumed — if wrong, swap the one script tag's `src`/`data-domain` for Umami's equivalent; the integration shape is identical either way]. Self-hosting the analytics instance is external infrastructure I cannot provision — Open Item.
- **D4**: I can add the `CNAME` file and the `deploy_pages.yml` staging step; registering a domain and configuring DNS/Cloudflare are the operator's own external actions — Open Item with exact steps listed.

### Requirements

- **R11**: WHEN `index.html` is served from the same origin as its data THE SYSTEM SHALL fetch relatively; WHEN not, THE SYSTEM SHALL fall back to the existing raw-URL path — and the existing add-team SLA (`docs/BUSINESS_PLAN.md:91-98`) SHALL be unaffected.
- **R12**: WHEN the scheduled pipeline runs THE SYSTEM SHALL also invoke `track_elite.py`, producing `elite_divergence_latest.json` (empty while `data/elite_managers.json` stays unpopulated).
- **R13**: THE SYSTEM SHALL include a single, easily-removable analytics script tag plus an in-app disclosure line consistent with the app's stated privacy posture.
- **R14**: THE SYSTEM SHALL include a `CNAME` file (placeholder domain) staged by `deploy_pages.yml`, inert until the operator supplies a real domain and DNS record.

### Build Phases

- [ ] **Phase D-1: Origin-aware fetch in index.html** *(risky — must not break the add-team SLA; run the Critique Engine at build time)*
  Done when: `index.html` loads data relatively when same-origin and falls back to the existing raw-GitHub URL otherwise; manually verified on both localhost and the deployed GitHub Pages URL that data still loads.
  Steps: port `track-record.html:133-158`'s logic into `index.html`'s `fetchJSON` (`index.html:1142-1158`) and the live-match fetch (`index.html:1165`); manual test both environments.
  Covers: R11
- [ ] **Phase D-2: Wire track_elite.py**
  Done when: a workflow step invokes `track_elite.py` on the existing pipeline cadence, confirmed to no-op safely on the current empty managers list.
  Steps: add the workflow step; confirm the no-op; note the entry-ID population decision in Open Items.
  Covers: R12
- [ ] **Phase D-3: Self-hosted analytics integration**
  Done when: the script tag and disclosure line are present on `index.html`/`landing.html`/`track-record.html`, configured from one constant.
  Steps: add the tag; add disclosure copy; note the external hosting requirement in Open Items.
  Covers: R13; checks: [A5]
- [ ] **Phase D-4: Custom domain scaffolding**
  Done when: a placeholder `CNAME` exists at repo root and is staged by `deploy_pages.yml`; Open Items lists the exact external DNS/registration steps.
  Steps: add `CNAME`; update `deploy_pages.yml`'s staged-file list; document DNS steps.
  Covers: R14

---

## Assumptions Ledger

| ID | Assumption | Basis | Blast radius if wrong | Check |
|----|-----------|-------|------------------------|-------|
| A1 | Scale up Priority 10 sampling to n=2,000, Overall league only | round order-of-magnitude increase, no basis for a more precise number yet | wasted API load / too little field diversity | Phase A-1 measures real timing/error-rate and can be re-tuned before Phase B depends on it |
| A2 | Auto-promotion regression threshold: 2% on any backtest metric | placeholder, no real proposal history exists yet to calibrate | too loose lets a bad recalibration through; too tight blocks everything | Phase B-2's done-check requires testing both a promote and a block case |
| A3 | Active-version table shape: `active_param_versions(family, key, version, promoted_at, promoted_by)` | executor's latitude — any shape serves as long as it's a single source of truth | none (internal implementation detail) | Phase B-1 |
| A4 | Minimum 8 realized gameweeks before surfacing planner-decision accuracy publicly | no variance data exists yet | premature aggregate could mislead with too little data | Phase C-3, revisit once real variance is visible |
| A5 | Analytics vendor: Plausible over Umami | named first in the roadmap doc's own "Plausible/Umami" phrasing; integration shape is near-identical either way | trivial — one script tag's attributes change | Phase D-3 |
| A6 | New schema migration numbers assigned at execution time (whichever track is built first claims the next number) | tracks may be built in any order | none — purely a filename/ordering detail | at build time |
| A7 | D1's origin-aware change must not alter the add-team SLA behavior the raw-URL design was chosen for | explicit statement in `docs/BUSINESS_PLAN.md:91-98` | could silently regress the one thing the raw-URL design was protecting | Phase D-1 done-check includes manual verification on the deployed Pages URL |
| A8 | Recommended execution order across tracks: D1/D2 (quick, low-risk) → C-1 (start the data clock early since Track C needs months to mature) → B (highest standalone value, "core moat") → A (largest, most externally-sensitive) | effort/risk sequencing, not a hard dependency | none — tracks are independently executable; reordering costs nothing | executor's choice |
| A9 | `fact_rival_squad_sample` retains only the current season's rows; prior-season rows are purged | Phase D never reads a prior season's rival-squad sample (it's already scoped current-season-only), so nothing downstream needs the older rows kept | without this, scaling sampling 10x means real people's squad picks accumulate indefinitely with no bound | Phase A-1 tests both the delete and preserve path of the purge |

## Risks, Landmines & Adaptations

- **Automated parameter promotion removes a deliberate, already-shipped human safety gate** (`review_recalibration.py`) from a path that drives live user-facing recommendations → adaptation: a regression-gated auto-promotion (threshold [A2]) plus a full audit trail plus a cheap, tested rollback path (versions are immutable, so reverting is just re-pointing a pointer) — see Track B throughout, flagged again here because it is the single highest-stakes decision in this plan and was a deliberate user override of my recommendation.
- **Priority 10 at increased scale retains more real individual FPL managers' picks** (still no names, per Phase A's existing design) → adaptation: a season-scoped retention purge ([A9]) bounds how long that data is kept — rows from any season but the current one are deleted, since Phase D never uses them — rather than letting 10x sampling accumulate indefinitely; Phase A-1 also re-validates rate-limiting/timing at the new scale, and no mini-leagues are added without an explicit future decision (parked in Open Items). Rate-limiting and scope-narrowing alone would only have bounded *load* and *breadth*, not the growing *volume* of retained real personal data — the purge is what actually closes that gap.
- **Building new planner-decision logging could, if scoped wrong, contradict the app's existing "sends nothing about the user anywhere" privacy commitment** → adaptation: the explicit Non-Goal in Track C restricts this to the already-server-side, already-scheduled report path for the operator's own tracked manager(s); Phase C-1 is marked risky specifically so its done-check double-checks this boundary was honored in the actual diff, not just the plan.
- **D1 touches `index.html`'s data-loading path, which the roadmap doc calls out as an intentional design choice** (`docs/BUSINESS_PLAN.md:91-98`) → adaptation: reuse the exact pattern already proven in `track-record.html` rather than a new design, and require manual verification on the deployed Pages URL as part of the done-check.
- **Residual risk, not further mitigated here**: analytics and custom domain (Track D3/D4) cannot be verified end-to-end without external infrastructure the operator must provision — named explicitly in Open Items rather than glossed over.

## Open Items (none blocking)

- Which specific FPL entry IDs to add to `data/elite_managers.json` so Phase D-2's export produces non-empty output — proceed with the workflow wired and no-op until the operator supplies IDs.
- Whether to later add specific mini-leagues to Priority 10's sampling (beyond the Overall league) — proceed with Overall-league-only for now.
- Standing up Plausible (or Umami) hosting for Track D3 — proceed with the code scaffolded and inert until hosted.
- Registering a domain and configuring its DNS record (plus optional Cloudflare front-ending) for Track D4 — proceed with a placeholder `CNAME` until the operator supplies a real domain.
- Tuning [A2]'s 2% regression threshold and [A4]'s 8-gameweek minimum once each track has produced enough real data to show what "normal" variance looks like.

## Verification

- Each track's Build Phases carry their own done-checks (exact commands/tests/observable behavior) — see above; every phase is independently runnable and testable.
- Cross-cutting: run the full existing test suite (`pytest`, plus the Node planner tests) after every phase in every track to confirm no regression to already-shipped behavior (Priority 1's `field_covariance.py`, the existing recalibration mechanism, the existing per-player track record, and `track-record.html`'s working origin-aware fetch must all keep passing their current tests throughout).
- Track B and Track C both require manual inspection of a real database/table after their first live run (not just unit tests) before being considered done, because both introduce genuinely new automated behavior affecting production data.

## The Critique Engine (for Build Phases marked *risky* above)

This plan is important enough to warrant it: it removes a human safety gate from a path affecting real users (Track B), retains more real third-party personal data at scale (Track A), and touches an explicit, already-documented privacy commitment (Track C) and an explicit, already-documented SLA-protecting design choice (Track D1). For every phase marked *(risky)* above — A-1, A-2, A-3, B-2, B-3, C-1, D-1 — the executor must, before marking that phase done:

1. Build the phase.
2. Hand it to a blind critic (a fresh sub-agent if the executor's environment supports spawning one; otherwise a deliberate fresh-eyes reread as a hostile stranger who did not write it) armed with only: the phase's stated Done-when criteria, the relevant Non-Goal/Key-Decision text from this document, and the diff — never the builder's own rationale.
3. The critic returns PASS/FAIL plus concrete required fixes, defaulting to "not good enough."
4. Apply every blocker, loop, stop at consensus or after three rounds; any residual non-blocker weakness gets added to this document's Open Items, not hidden.

---

*This plan was produced via a structured "Dry-Run Interview" process: recon of the existing codebase and roadmap docs (cited throughout via file:line references), six targeted questions to the operator (Interview Ledger above), and explicit tagging of every assumption made on the operator's behalf (Assumptions Ledger above). Corrections to any assumption, decision, or scope call above are welcome — flag them and this document will be revised.*
