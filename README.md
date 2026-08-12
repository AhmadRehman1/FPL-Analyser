# FPL Quant v2

Implementation of the frozen M0-M9 module specs (`FPL Quant v2`). This is a from-scratch
rebuild, separate from the prior attempt (`../fpl-quant-engine`), following the kickoff
notes' implementation order and the architectural corrections the spec-freezing process
converged on (versioned parameters, a real `evidence_claims` layer, MIQP not MILP, etc).

## Status

- **M0 (Data Schema & Ingestion Architecture): done.** Three-layer model
  (`fact_raw` -> `fact_reconciled` -> `evidence_claims`), generic versioned-parameter
  mechanism, `data_asof` snapshot discipline, deprecation allowlist (including M8's
  tab 31-36 addition), all built and tested against the real source data.
- **M1b (Evidence Integration & Reliability Weighting): done.** `base_reliability_score`
  construction, `evidence_blend.py`'s weighted-average/categorical-distribution conflict
  resolution with the FACT/high-tier multiplier, cross-check and community source-notes
  audit fields.
- **M1 (Team Strength Model): done.** Full Dixon-Coles MLE fit (`tau(x,y;rho)`, `xi`
  time-decay) over real 2024-25+2025-26 Premier League results, Elo-regression prior,
  continuous shrinkage, versioned snapshots. Calibration sanity-checked against real
  football knowledge (Man City/Arsenal/Liverpool top attack, Spurs' well-known leaky
  defence despite decent attack, promoted clubs falling back to a pure Elo prior).
- **M2 (Minutes Model): done.** Position-pooled, recency-weighted historical start-rate
  fit with continuous shrinkage, logit-scale evidence adjustment (shift-type + pull-type,
  reusing M1b's `effective_weight`) clipped to +/-6.0, empirical conditional
  minutes-given-appearance rates, GW0 friendly minutes logged as low-weight
  `preseason_involvement` claims rather than entering the quantitative fit. Required
  invariant (`P(0)+P(1-59)+P(60+)=1`) holds exactly across all 577 real 2026-27 players;
  spot-checked against real evidence (Saliba's "Out" injury claim cuts his start
  probability from 83.5% to 43.8%).
- **M3 (Expected Points Engine): done.** Implementation-time verification gate cleared
  first (base scoring matrix + BPS formula confirmed against current 2026/27 sources, not
  assumed -- see below). Every category its own sub-model conditioned on M2's minutes
  distribution: appearance, goals/assists (per-90 rates shrunk toward the position average
  by sample size), clean sheet (exact binary 60+ gate), goals conceded (exact
  `E[floor(X/2)]` under Poisson, not a linear approximation), DefCon (thresholded count
  distribution), and bonus via genuine sequential Plackett-Luce marginalization. Verified:
  Plackett-Luce `E[bonus]` sums to exactly 6.0 per fixture (3+2+1) across every real GW1
  2026-27 match; only Haaland exceeds 1.5 expected goals league-wide. Required
  non-double-counting audit is a structured, tested artifact (`non_double_counting_audit()`),
  not an unwritten claim. Saves/penalty-saves/cards/own-goals are explicitly left at 0
  rather than guessed -- no penalty-taker or cards/OG per-90 rate was ever reconciled into
  `fact_reconciled`, and BPS's passing/crossing/key-pass components are omitted for the
  same reason (see `expected_points.py`'s module docstring for the full scope statement).
- **M4 (Uncertainty & Correlation Layer): done.** Per-category variance from each
  category's own M3 distributional form (Poisson, Bernoulli/threshold, full categorical
  for bonus). Within-player covariance via the law of total covariance on M2's three-state
  minutes gate, plus a residual term on the pinned `rho_residual=0.15` placeholder.
  Cross-player Sigma built block-wise by fixture. Cornish-Fisher quantiles for reporting
  only (confirmed not wired into M5's objective). Verified against real GW1 2026-27 data:
  zero negative/NaN variances across 577 players, all quantiles monotonic, Arsenal's
  back-line shows the strongest positive teammate covariance in the league (~3.5-4.0) --
  exactly the concentration signal M5's guardrails exist to see -- and opposing
  goalkeepers show the most negative cross-fixture covariance, both as expected.
- **M5 (Squad Optimizer): done.** The module the original v1 rebuild exists because of --
  the documented lambda=0 back-five failure happened here. Real MIQP via SCIP (not a
  MIP-only solver that would silently drop the quadratic term): the risk term is moved into
  a constraint via the standard epigraph reformulation (`t >= w'Sigma*w`, convex since
  Sigma is PSD; objective becomes `linear_EP - lambda*t`) since SCIP requires a linear
  objective. The lambda=0-vs-lambda=0.15 divergence check runs first, before anything else
  from this module is trusted -- `squad_optimizer.run()` raises `DivergenceCheckFailedError`
  and refuses to store a squad selection if the two solves land on the same squad (the run
  itself is still logged, so the failure is auditable). Solves the real 577-player, 20-club
  candidate pool in ~19 seconds. The real GW1 2026-27 optimal XI is maximally
  club-diversified (11 different clubs, one player each, well under the guardrail's cap of
  3) -- direct evidence the covariance signal from M4 is doing real work, not just the hard
  cap. Notably favors several high-DefCon defenders (including the captain) over some
  premium attackers under the frozen budget/risk trade-off -- a real, striking model output
  worth a human sanity check (exactly what M9's later "could a human beat this by eye?"
  prompt exists for), not silently smoothed over.
- **M6 (Monte Carlo Simulation Engine): done.** Supersedes M4's `rho_residual=0.15`
  placeholder with a real generative mechanism -- a shared match-intensity latent factor
  `Z_fixture` (Gamma-Poisson mixture, mean 1) drawn once per simulated fixture and scaling
  every involved player's goal/assist/bonus-strength rate together, closed-form calibrated
  (`sigma_z^2 = rho/(lambda*(1-rho))`) so the underlying goals+assists Poisson-count
  correlation hits exactly 0.15 (verified both analytically and empirically at 200k draws).
  Scope, a genuine judgment call: simulates M5's *chosen 15-player squad* for one
  `squad_optimizer_runs.run_id` (not M5's ~577-player input pool -- see module docstring for
  why that's the correct reading of "candidate pool ... not the full 577-player league").
  Clean sheet and goals-conceded are read directly off the same joint (home_goals,
  away_goals) draw already sampled from the Dixon-Coles bivariate Poisson grid, not
  independently redrawn -- a strictly more correct generative link than a second Poisson
  draw would give. 5,000 antithetic pairs (10,000 realizations) per squad player, seeded
  deterministically via `sha256(model_version|calibration_asof_date|query_id)`. Verified
  against the real GW1 2026-27 squad from M5's run: all 15 players' simulated mean tracks
  M3's analytical `ep_total` closely (e.g. Bruno Fernandes 4.40 simulated vs 4.18 analytical,
  Jason Steele 0.96 vs 0.97) -- strong evidence the whole draw pipeline is unbiased, not just
  internally consistent. Zero negative variances across all 15 players, all quantiles
  monotonic, and empirical minutes-state frequencies match M2's own probabilities almost
  exactly (e.g. Bruno: 68.8% simulated 60+ vs 69.4% modeled). A genuinely notable finding,
  not silently smoothed over: teammate/opponent *total-points* correlation in the real run
  comes out far below 0.15 (0.02-0.08, not ~0.15) even though the underlying goals+assists
  mechanism is calibrated exactly to 0.15 -- see Design notes for why this is real dilution,
  not a bug, and what it implies for M4's blanket `rho_residual` application.
- **M7 (Walk-Forward Backtesting Framework): done.** Full M1-M6 pipeline re-run against every
  historical 2024-25/2025-26 gameweek under a real `data_asof` cutoff -- the first genuine
  exercise of the snapshot-discipline guarantee every module has carried since M0 (confirmed
  by code review before building this: every module except `minutes_model.py`'s evidence-claim
  path only *stored* `calibration_asof_date` for audit, never filtered a fact-table query with
  it -- invisible until now because a live run's historical facts trivially predate
  `data_asof`). Enforcement is connection-scoped `TEMP TABLE` shadowing (`backtest.asof_scope()`)
  so M1-M5's existing, already-tested SQL runs completely unmodified against an asof-truncated
  view -- verified both by a dedicated leak-prevention regression test and, for real, by the
  actual run. 76 gameweeks total (both seasons confirmed fully complete, per this spec's own
  research correction); 71 actually walked -- 1 skipped for the cold-start guard (2024-25 GW1
  has zero prior matches to MLE-fit against), 4 for real historical Double Gameweeks the
  pipeline can't yet handle (confirmed in the real data at 2024-25 GW25 and 2025-26
  GW26/33/36 -- a genuine gap in M3's existing DGW-out-of-scope boundary, extended here rather
  than patched with new aggregation logic mid-M7). Tiered cold/warm/mature scoring (log score +
  Brier for M2/M3, Poisson calibration for M1) on the real 71-gameweek run: mature tier's
  Poisson-calibration residual is essentially unbiased (-0.030), every other metric improves
  cold->warm->mature exactly as this spec's own known-limitation note predicted, and the
  cold-tier MLE-divergence guard fired exactly once across the whole run (3 degenerate
  fixture-sides, isolated entirely to cold tier) -- real evidence the instability really is a
  cold-start-specific phenomenon, not a broader fragility. `recalibrate()` produced 5 real,
  human-reviewable proposals (all `pending`, none auto-activated -- see Design notes): `xi`
  0.0018->0.005 (fit NLL nearly halves, 1302->677), `rho_residual` 0.15->0.0 (independent
  confirmation of M6's dilution finding above), `fact_type_multiplier` 1.2->1.0,
  `competitive_matches_threshold` 10->20, and `lambda_value` 0.15->0.0 -- the last one
  specifically *not* a recommendation to remove M5's risk mechanism (see Design notes: a
  dedicated sensitivity report empirically vindicates keeping `xi_club_concentration_cap` out
  of this recalibration, exactly the redundancy the guardrail was designed for).
- **M8 (Transfer & Chip Strategy Planner): done.** Implementation-time verification gate
  cleared first (same discipline as M3's scoring matrix): the -4-points-per-extra-transfer
  hit, 5-transfer banking cap, and 2-sets-of-4-chips/GW19-no-carryover structure all confirmed
  unchanged for 2026-27 via live web search against the Premier League's own site and Fantasy
  Football Scout, not assumed from convention. Operates on an *existing* squad -- a genuinely
  new concept nothing in M0-M7 tracked (`manager_state_versions`/`manager_squad_holdings`,
  bootstrapped once from a real `squad_optimizer_runs` selection and evolved forward only by
  M8's own accepted recommendations, never silently re-derived from a fresh M5 call). Two
  real findings that turned out to make the hardest-looking parts of this module cheap, not
  expensive: no multi-gameweek EP horizon existed anywhere (`ep.run()` takes one gameweek per
  call), but `team_strength`/`minutes_model` are gameweek-agnostic snapshots, so a 5-gameweek
  horizon costs only 5 extra `ep.run()`/`uncertainty.run()` calls, not 5x the full M1-M4
  chain; and Triple Captain needs *zero* new simulation, since captaincy has no effect on
  `monte_carlo.py`'s mechanics (grepped, confirmed) -- `monte_carlo_player_summary.mean_total`/
  `sqrt(var_total)` already *are* `E[marginal_value]`/`StdDev[marginal_value]` for every
  candidate. Verified against the real GW1 2026-27 squad, planning for GW2: top transfer
  recommendation is Jason Steele (backup GK) -> Martin Dúbravka, +8.97 points over the
  5-gameweek horizon at zero cost (within the free allocation); Wildcard and Free Hit both
  correctly *not* recommended this early (a fresh full rebuild actually scores *worse* than
  the current squad plus one good transfer -- gains of -4.98 and +1.59 respectively, neither
  clearing the versioned threshold); Bench Boost recommends GW4. Triple Captain is a genuine
  near-tie, not a confident pick, and says something real about the model, not a bug -- see
  Design notes.
- **M9 (Reporting/Explainability Layer): done.** The last module in the frozen sequence.
  Grepped the whole `src/fpl_quant/` tree first for any existing `explain()`-style adapter:
  zero matches -- this isn't one new module, it's one small, additive adapter function in each
  of six existing modules (`expected_points.explain_player_ep`, `uncertainty.
  explain_player_risk`, `monte_carlo.explain_player_risk_empirical`, `squad_optimizer.
  explain_run`, `minutes_model.explain_player_adjustment`, `backtest.
  explain_backtest_summary`, `transfer_planner.explain_plan`, `params.transparency_panel`),
  plus one new assembling module (`reporting.py`) -- exactly the loose-coupling integration
  pattern the spec itself locks in, not a design choice made here. Minimal headline by default,
  every other section its own dict key; automated pattern-detection flags kept genuinely
  separate from a fixed human prompt ("does this squad look defensible to you?"), never a
  self-certification, per the spec's own stated reason (the original `lambda=0` bug passed
  whatever automated checks existed at the time). Verified against the real GW1 2026-27 squad:
  15 players, captain James Tarkowski correctly *not* flagged as a goalkeeper, zero clubs at
  either concentration cap, real M7 backtest metrics and real M8 transfer/chip rationale both
  render correctly in the same report, and the parameter-transparency panel counts **65 of 71**
  active parameters as still purely invented, not yet touched by M7's recalibration -- an
  honest, real number, not an estimate. Two real bugs the test suite caught before the real
  run could: an exact repeat of M6's own documented `dict()`-on-3-tuples mistake (see below),
  and a captain-position check that silently no-opped whenever club-resolution data was
  missing because it was wrongly coupled to that unrelated join -- see Design notes for both.
- **M7 extension: `kappa_tc` recalibration (follow-up, not a new milestone).** M8's own spec
  flags `kappa_tc` for M7 recalibration explicitly, but M7 predates M8 so had no technique for
  it. Added `backtest.refit_kappa_tc()`, wired into `recalibrate()` as a new opt-in
  `refit_kappa_tc_flag`/`current_kappa_tc_version` pair (default off, so existing M7-only
  callers are unaffected). Unlike `refit_lambda`, this needed zero re-solving: `kappa_tc` never
  changes which XI gets picked, only which XI player would be captained, and every real
  backtest gameweek step already has a real Monte Carlo simulation of its own XI stored from
  the original 71-gameweek run -- captain choice under a candidate `kappa_tc` is a pure argmax
  read against `monte_carlo_player_summary`, the same TC-score formula `transfer_planner.
  evaluate_triple_captain()` already uses live. Run for real against the actual
  `backtest_run_id=1` data (62 of the 70 Monte-Carlo-bearing steps had usable warm/mature-tier
  data): `kappa_tc` 0.15->0.5, `realized_sharpe` 1.092->1.196 -- proposal #6, `pending`, human
  review required same as the other five. `wildcard_gain_threshold_params` is deliberately
  *not* covered by this extension -- see Design notes for why. Re-running the M9 report after
  writing this proposal now shows **64 of 71** (`tc_risk_aversion_params` moved out of the
  never-backtested set) -- see Design notes for the important caveat before treating proposal
  #6 as a clean recommendation.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# Extract FPL-Core-Insights and place the evidence workbook under data/external/ first
# (see data/external/ layout below), then:
.venv\Scripts\python scripts\run_ingestion.py

.venv\Scripts\python -m pytest tests/ -v
```

## Layout

```
schema/0001_core_schema.sql   -- DDL: fact_raw log, fact_reconciled tables, evidence_claims,
                                  sources, generic param_versions mechanism, model_runs
schema/0002_m1_team_strength.sql -- DDL: team_strength_model_versions, team_strength_snapshots
schema/0003_m2_minutes_model.sql -- DDL: minutes_model_versions, minutes_model_outputs
schema/0004_m3_expected_points.sql -- DDL: ep_model_versions, ep_outputs
schema/0005_m4_uncertainty.sql -- DDL: uncertainty_model_versions, uncertainty_outputs, cross_player_covariance
schema/0006_m5_squad_optimizer.sql -- DDL: squad_optimizer_runs, squad_optimizer_selections
schema/0007_m6_monte_carlo.sql -- DDL: monte_carlo_run_versions/player_totals/player_summary/empirical_covariance
schema/0008_m7_backtest.sql -- DDL: backtest_runs/gameweek_steps/metrics, recalibration_proposals
schema/0009_m8_transfer_planner.sql -- DDL: manager_state_versions/squad_holdings, transfer_plan_runs/recommendations, chip_evaluations
schema/0010_m8_manager_snapshot_flag.sql -- squad_optimizer_runs.is_manager_snapshot (additive; see Design notes)
src/fpl_quant/
    db.py                     -- DuckDB connection + schema application
    ingest_csv.py             -- generic fact_raw ingestion: one table per (season, relpath),
                                  all-VARCHAR, append-only across ingestion runs
    entity_resolution.py      -- name normalization, deterministic team/player UIDs
    reconcile.py              -- fact_raw -> fact_reconciled: entity resolution, match_id
                                  dedup, column-semantics tagging
    params.py                 -- generic versioned-parameter read/write (immutable versions)
    decay.py                  -- pinned exponential evidence-decay formula
    snapshot.py                -- data_asof query helpers (look-ahead prevention)
    ingest_workbook.py         -- evidence workbook -> evidence_claims, deprecation allowlist
    ingest_research_pull.py    -- second (flat-columns) evidence source -> evidence_claims
    evidence_blend.py          -- M1b: effective-weight formula + conflict-resolution blending
    team_strength.py           -- M1: Dixon-Coles MLE fit, Elo-regression prior, shrinkage
    minutes_model.py           -- M2: historical fit, evidence adjustment, three-state output
    expected_points.py         -- M3: per-category sub-models, Plackett-Luce bonus, EP total
    uncertainty.py             -- M4: variance, within/cross-player covariance, Cornish-Fisher
    squad_optimizer.py         -- M5: MIQP via SCIP, lambda=0-vs-real divergence check
    monte_carlo.py              -- M6: Z_fixture Gamma-Poisson mixture, antithetic-variate
                                    gameweek simulation, empirical Sigma vs M4
    backtest.py                  -- M7: asof_scope() TEMP TABLE shadowing, walk-forward loop +
                                     tiered scoring, per-family recalibration + proposal gate
    transfer_planner.py           -- M8: horizon EP, transfer search, Wildcard/Free Hit/Triple
                                     Captain/Bench Boost evaluation, manager-state evolution
    reporting.py                   -- M9: automated sanity-check flags + report assembly/
                                     rendering, calling every other module's explain() adapter
scripts/run_ingestion.py       -- end-to-end pipeline runner (M0-M6, one live gameweek)
scripts/run_backtest.py         -- M7: full walk-forward backtest + recalibration, all 76 gameweeks
scripts/review_recalibration.py  -- M7: human review/confirm/reject gate for recalibration_proposals
scripts/run_transfer_planner.py   -- M8: bootstrap manager state + plan transfers/chips for one gameweek
scripts/run_report.py              -- M9: build + print a real squad report from the project database
tests/                         -- pytest, one file per module concern
data/external/                 -- gitignored; extracted FPL-Core-Insights repo,
                                   FPL_202627_Master_Evidence_Database.xlsx, and
                                   FPL_Evidence_Claims_Research_Pull.xlsx go here
db/fpl_quant_v2.duckdb         -- gitignored; rebuild via scripts/run_ingestion.py
```

## Design notes worth knowing before touching this

- **`evidence_claims` matching is normalized, not literal-string.** The workbook spells
  player names differently than FPL-Core-Insights does (accents, dropped middle names).
  `player_alias.normalized_alias_name` is the join key; a word-subset fallback handles
  compound-surname mismatches (e.g. workbook's "Manuel Ugarte" vs the registered "Manuel
  Ugarte Ribeiro"), but only when it resolves to exactly one distinct player.
- **`matches.csv`'s `home_team`/`away_team` reference team `code`, not `id`**, despite the
  source README stating `id` -- verified against real data (Arsenal code=3/id=1 appears as
  `home_team='3.0'` in its own home fixture). `reconcile.py` joins on `code`.
- **Compound free-text evidence (6_Manager Database, 17_Pre-season Match Reports) is staged,
  not auto-decomposed.** Per M1b's frozen spec, decomposing a cell like "Alonso trialled
  4-2-3-1... gave minutes to academy players..." into atomic claims is a permanent
  human-curation step, not an NLP problem. Both tabs land in
  `claims_pending_manual_decomposition` with `raw_text` preserved for a human to work from.
- **Every module's versioned parameters share one physical table** (`param_versions`),
  distinguished by `param_family`. `claim_type_decay_params` and `source_tier_weights` are
  read-only views over it, named to match the spec docs exactly. `params.resolve_param()`
  hard-errors on a missing lookup -- it never silently falls back to a default (this is
  M5's explicit, non-negotiable requirement, generalized to the whole mechanism).
- **Superseded claims are excluded "outright" per M0's wording, but `snapshot.py` treats
  that as asof-relative**, not an unconditional filter -- otherwise a backtest run before a
  correction landed would incorrectly see no evidence at all, which is a real look-ahead
  bug, not a faithful reading of the spec's intent (M7 depends on this being right).
- **2024-2025's season-root files live one directory level down** (`teams/teams.csv`,
  `players/players.csv`, `playerstats/playerstats.csv`) instead of at the season root like
  2025-2026/2026-2027. `reconcile._season_root_table()` tries both layouts -- an exact-match
  lookup silently drops all of 2024-2025 otherwise (a real bug this project hit and fixed,
  not a hypothetical).
- **M1's `seasons_of_topflight_data >= 3` Elo-regression population is capped at what the
  loaded data can ever produce.** FPL-Core-Insights ships exactly 3 seasons; since 2026-27
  is the target season, only 2 prior seasons are ever available, so no team can literally
  reach 3. `weight_own_data = min(1, seasons/3)` stays on the frozen literal divisor (so it
  honestly tops out at 2/3 rather than being silently rebased to look fully confident); only
  the regression's *eligible-team filter* uses a capped effective threshold, so that
  population is never empty by construction.
- **Dixon-Coles zero-centering is one degree of freedom, not two.** `attack_i - defence_j`
  is invariant under a shared additive shift applied to every attack *and* defence value
  together, not under independent shifts of each. `team_strength.py` fits with one reference
  team pinned at (0,0) for numerical stability, then re-centers so the mean attack is
  exactly 0 -- mean defence generally won't also be exactly 0, which is a necessary
  consequence of there being only one true degree of freedom, not a bug.
- **The real injury-status vocabulary in this workbook is `Out`/`Doubt`/`Doubt (improving)`/
  `Doubt (minutes)`, not the spec's illustrative `Out`/`Doubtful`/`Minor-knock`/`Fit`.**
  `minutes_adjustment_params` is seeded against what the data actually contains, with the
  spec's original strings kept too (for whenever a future data pull uses them), and the
  sub-variants ("improving"/"minutes") given milder shift magnitudes than a plain "Doubt".
- **A `snapshot.py` NULL becomes NaN, not None, once it round-trips through a pandas
  DataFrame.** `evidence_blend.effective_weight()` checked `is not None` for
  `confidence`/`source_reliability_score` and missed every NaN, silently poisoning whole
  weighted sums (`nan * anything = nan`, and `max(-cap, min(cap, nan))` resolves to the cap
  through Python's NaN-comparison semantics -- this is exactly what happened: two real
  players' evidence adjustments both silently pinned at +6.0 before the fix). Fixed with
  `pd.isna()`; regression-tested in `test_evidence_blend.py`.
- **A model run's `data_asof` needs to mean end-of-that-day, not start-of-it.** Evidence
  ingested earlier the same calendar day as a run's `calibration_asof_date` is legitimately
  knowable "as of" that date; converting the date to midnight excluded it. `minutes_model.run()`
  combines with `datetime.max.time()`, not `datetime.min.time()`.
- **M3's implementation-time verification gate (kickoff notes' hard precondition) was
  cleared via live web search**, not memory: base scoring matrix and BPS formula confirmed
  against the Premier League's own site, Fantasy Football Scout, and Draft Fantasy, cross-
  checked against the workbook's own `13_Rule Changes Database`. One genuine ambiguity
  survived the cross-check and is documented (not silently resolved) in
  `expected_points.seed_v1_params()`: whether outside-box GK saves are removed from BPS
  entirely (the workbook's claim) or retain a base +2 with only the box bonus removed
  (what current web sources suggest) -- went with the workbook's more specific claim.
- **A per-90 rate from a handful of minutes is noise, not signal, and must be shrunk.**
  Real bug this project hit: a player with a single 2-minute cameo and one small xG
  contribution extrapolated to `expected_goals_per_90 = 3.6`, which briefly outranked
  Haaland's expected goals for a gameweek before `expected_points.player_rates_shrunk()`
  added minutes-sample-size shrinkage toward the position average (same discipline M1/M2
  already apply to their own historical rates, just missing here on the first pass).
- **`ep_outputs`' saves/penalty-saves/cards/own-goals sit at 0, not a guessed rate.** No
  penalty-taker identity or cards/OG per-90 rate was ever reconciled into
  `fact_reconciled`, and BPS's passing/crossing/key-pass components are omitted for the
  same reason -- explicit absence of a signal, not an invented one.
- **M4's residual within-category covariance term needs Var(category | playing)**, which
  would require each category's within-state second moment, not just its conditional mean.
  `uncertainty.py` proxies it with the category's unconditional variance instead --
  reasonable given `rho_residual` is itself already an invented placeholder explicitly
  flagged for full *replacement* (not mere recalibration) once M6's Monte Carlo engine
  exists to capture this structure directly. Named in code and here, not a hidden shortcut.
- **The cross-player block-structure correlation coefficients (teammate/opponent x
  attacking/defensive) are invented v1 defaults**, same status as `rho_residual` -- no
  literature, ordinal reasoning only (a shared clean-sheet Bernoulli draw within one match
  is near-deterministic for two teammates, so pinned high at 0.9; shared attacking tempo is
  real but far looser, so pinned low at 0.25). Versioned through the same generic
  `param_versions` mechanism as everything else in this project, not hardcoded literals --
  an earlier draft had them hardcoded directly in `cross_player_covariance_for_fixture()`,
  caught and fixed before the first real run.
- **M6's real generative mechanism shows M4's flat correlation pins overstate total-points
  correlation by diluting into the wrong denominator, not by picking the wrong number.**
  `rho_residual=0.15` was calibrated as a goals+assists *count* correlation (the thing
  `Z_fixture` actually links across fixture participants), but `uncertainty.total_variance()`
  applies it as a flat pairwise correlation between *every* active category pair for a given
  player -- including pairs like defcon-bonus or appearance-goals that share no generative
  channel at all in M6's mechanism. Appearance, DefCon, saves, and bonus-rank are independent
  per-player draws in M6 (no shared `Z_fixture` term), and those categories make up most of a
  typical player's points variance, so whatever correlation the goals+assists piece
  contributes gets diluted once summed into `total_points`. Verified on the real GW1 2026-27
  squad run: empirical total-points correlation lands at 0.02-0.08 for every teammate/
  opponent pair, not ~0.15. The same pattern shows up cross-player: `teammate_defensive=0.9`
  is a reasonable *directional* call (clean sheet genuinely is the same Bernoulli draw for
  two teammates) but overstates it at the total-points level once pinned as a flat
  coefficient -- e.g. Ethan Ampadu/Joel Piroe's M4 covariance (0.67) is ~2.3x M6's empirical
  value (0.29); Dara O'Shea/Emerson's M4 covariance (0.51) is ~4.7x M6's empirical value
  (0.11). The fix M6 makes possible isn't a smaller flat rho -- it's category-specific
  correlation (goals/assists through the real `Z_fixture` channel, clean-sheet/goals-conceded
  exact per M6's own joint score draw, everything else left near zero), which is exactly the
  full *replacement* the M4 docstring and this README both already flagged `rho_residual` for.
- **A second, differently-shaped evidence source** (`FPL_Evidence_Claims_Research_Pull.xlsx`
  -- flat columns per tab: Transfers/Injuries/SetPieceTakers/PriceNotes) is ingested by its
  own small module, `ingest_research_pull.py`, sharing M1b's source-tier classification and
  M0's `evidence_claims` schema rather than a parallel implementation. `PriceNotes` is
  ingested for audit/provenance visibility only (`claim_type='fpl_price_note'`) -- never
  promoted into `fact_reconciled`'s authoritative `now_cost`, per M0's own architectural
  boundary between a stat and an opinion about a stat.
- **`player_alias` has multiple rows per (player, season)`** (full name and web name each
  get their own row -- see `reconcile.build_dim_player`). Joining straight from
  `ep_outputs`/`squad_optimizer` candidates against `player_alias` without deduping first
  fans out to ~2x the true row count -- a real bug this project hit building M5's candidate
  pool (1151 "players" instead of 577). Fixed by deduping to one `(player_uid, team_code)`
  pair per season before joining; verified no player actually has two *different* team_codes
  in the same season before trusting that dedupe.
- **`str.lower()` doesn't fold the German sharp-s to "ss"; `str.casefold()` does.** Real bug:
  a research-pull source's "Pascal Gross" (plain double-s) failed to resolve against the
  registered "Pascal Groß" (sharp-s) until `entity_resolution.normalize_name()` switched
  from `.lower()` to `.casefold()`. NFKD normalization doesn't catch this either -- sharp-s
  isn't a diacritic, so it has no combining-character decomposition to strip.
- **M6's "candidate pool relevant to the specific query ... not the full 577-player league"
  is read as M5's *chosen 15-player squad*, not M5's ~577-player input pool.** The input pool
  IS the "full 577-player league" M6's own Research section just finished naming as the
  computational-scope concern to avoid, so reading "candidate pool" as that same 577 would
  contradict the sentence's own contrast; the chosen squad is also the only pool M6's own
  Outputs bullets actually need (M5's chosen squad's distribution, M8's chip-value estimation
  over that squad+bench). A "query" is therefore one `squad_optimizer_runs.run_id`. Stated
  plainly in `monte_carlo.py`'s module docstring, not silently picked.
- **Non-squad fixture participants (needed for a realistic Plackett-Luce bonus-ranking pool,
  since a real match has ~22 first-team-relevant players, not just the squad's own) are NOT
  independently re-simulated each realization.** Full fidelity would mean re-drawing every
  non-squad player's own minutes state 10,000 times per fixture for a class of players whose
  own distributional output nothing downstream ever consumes -- exactly the computational-
  scope blowup the query-level scope restriction above exists to avoid. They instead use
  their mean-based M3 strength (`exp(expected_bps/tau)*p_played`, identical to what M3/M4
  already compute) scaled by that fixture's own per-realization `Z_fixture` -- still reactive
  to the shared tempo factor, just not independently re-drawn.
- **`Z_fixture`'s variance is solved in closed form, not fit numerically.** Modeling
  `X_i | Z ~ Poisson(Z*lambda_i)` with `Z ~ Gamma(mean=1, var=sigma_z^2)` (the standard
  Gamma-Poisson mixture -- keeps every downstream rate nonnegative by construction, unlike a
  Normal or log-Normal multiplier) gives `Cov(X_i,X_j) = sigma_z^2*lambda_i*lambda_j` and
  `Var(X_i) = lambda_i + sigma_z^2*lambda_i^2` exactly, so at `lambda_i=lambda_j=lambda`,
  solving `rho_residual = sigma_z^2*lambda^2/(lambda+sigma_z^2*lambda^2)` for `sigma_z^2`
  gives `sigma_z^2 = rho_residual/(lambda*(1-rho_residual))` directly -- no root-finding, no
  numerical calibration loop. `lambda` itself (`lambda_representative`) is computed from real
  data (mean expected goals+assists COUNT, not points, across the actual squad's real
  GW1 2026-27 fixtures: 0.141 in the real run) rather than invented, so the only literal
  constant feeding this whole mechanism is `rho_residual` itself -- already versioned by M4.
  Verified both analytically (`test_z_fixture_variance_matches_closed_form`) and empirically
  via a 200k-draw Monte Carlo of the actual mixture
  (`test_z_fixture_variance_reproduces_target_correlation_empirically`, within 0.02 of target).
- **DefCon and saves are intentionally NOT scaled by `Z_fixture`.** M6's own spec names
  goals, assists, and "bonus-relevant BPS components" explicitly as what the shared tempo
  factor scales; defensive actions and goalkeeper saves are a different mechanism (a busier,
  higher-tempo match doesn't obviously inflate one defender's own CBI count the way it
  inflates goal-scoring across a match), and inventing a second, untested scaling channel
  beyond what the spec actually states would be a bigger unstated simplification than leaving
  them at their M3 rate, not a smaller one.
- **Clean sheet and goals-conceded are read directly off the same joint `(home_goals,
  away_goals)` draw already sampled from the Dixon-Coles bivariate Poisson grid for that
  fixture, not independently redrawn from `lambda_against`.** This is a strictly more correct
  generative link than a second Poisson draw would give: it makes two teammates' clean-sheet
  outcomes literally the *same* underlying event (not just parametrically correlated) and two
  opponents' clean-sheet/goals-conceded outcomes exact complements of the same scoreline, with
  no extra covariance machinery required to produce that structure.
- **A real bug the first end-to-end run against real data caught**: `dict(rows)` on
  `(player_uid_a, player_uid_b, covariance)` 3-tuples raised `ValueError: dictionary update
  sequence element #0 has length 3; 2 is required` when building the M4-covariance lookup for
  the validation table -- `dict()` needs 2-tuples. The unit test suite (which mocks nothing
  and never exercises `run()`'s SQL against a populated DB, per the same
  unit-the-math/integrate-against-real-data split `test_squad_optimizer.py` already
  established) couldn't have caught this; only the real pipeline run could, and did. Fixed
  with an explicit dict comprehension (`{(a,b): cov for a,b,cov in rows}`). Consistent with
  every other real bug logged in this file: caught by running against real data, not by
  memory or assumption.
- **A genuine, notable finding from the first real run, not silently smoothed over: teammate/
  opponent* total-points* correlation comes out far below `rho_residual=0.15`** (0.017-0.078
  observed across the real GW1 2026-27 squad's within-fixture pairs) even though the
  underlying goals+assists Poisson-count mechanism is calibrated to hit exactly 0.15 (and
  does, verified in isolation). The reason is real, not a bug: `Z_fixture` only injects
  correlation into the goals/assists/bonus-strength slice of a player's variance; appearance,
  DefCon, saves, and (for the specific real pairs observed) one side's zero-weight clean-sheet
  category (Forwards score 0 clean-sheet points under the base scoring matrix) are either
  independently drawn per player or simply don't carry a clean-sheet term for that position,
  which dilutes any TOTAL-points correlation well below the category-level 0.15 by ordinary
  variance-mixing arithmetic. M4's `rho_residual`, by contrast, was applied as a flat residual
  term across *every* category pair (appearance-vs-appearance included) for every teammate/
  opponent, which this real comparison suggests overstates the true total-points correlation
  relative to a mechanistically-honest model. Flagged here for M7's calibration pass, exactly
  the kind of disagreement `monte_carlo_empirical_covariance.m4_covariance` exists to surface,
  not a discrepancy to paper over.
- **Full pipeline integration is verified by running the real `scripts/run_ingestion.py`
  end-to-end against the real 2024-27 data, not by a large synthetic-DB pytest fixture.**
  `tests/test_monte_carlo.py` thoroughly unit-tests every standalone function (seeding,
  Z_fixture calibration, the bivariate-Poisson grid sampler, vectorized Poisson/categorical/
  Plackett-Luce samplers) the same way `test_squad_optimizer.py` unit-tests `solve()` against
  a synthetic pool rather than building out the full `fact_match`/`ep_outputs`/
  `minutes_model_outputs`/`player_alias`/`team_alias` join chain `run()` actually depends on.
  That real run is what caught the `dict()` bug above -- direct evidence the split is doing
  its job, not a gap being rationalized away.
- **A real bug the M7 dry run caught: a strict `kickoff_time < deadline` asof cutoff hides the
  very fixtures being predicted, not just future ones.** The gameweek being backtested has its
  own `kickoff_time` exactly at (not before) the pinned deadline, so a naive `asof_scope()`
  shadow made `expected_points.run()`/`monte_carlo.run()` unable to see which players faced
  which fixture that week -- the whole point of the prediction, not a leak. Fixed with one
  deliberate exception in the `fact_match` shadow: the target gameweek's own fixture *schedule*
  (match_id/teams/kickoff_time) stays visible with `home_score`/`away_score`/`finished` nulled
  out, so the result stays genuinely unknowable while the schedule -- announced well before any
  real deadline -- doesn't. Every other future gameweek stays fully invisible, schedule
  included.
- **A second real bug the dry run caught: real historical Double Gameweeks crash
  `squad_optimizer_selections`' primary key.** 2024-25 GW25 and 2025-26 GW26/33/36 each have a
  team playing twice under the same gameweek label (rearranged fixtures) -- confirmed in the
  actual ingested data, not a hypothetical. `expected_points.run()` emits one `ep_outputs` row
  per player per fixture by its own explicit v1 design ("DGW/multi-fixture handling is out of
  scope for v1, per M8's own research finding that 2026-27 currently has no scheduled
  doubles/blanks" -- `expected_points.py`'s own module docstring), so a DGW player's duplicate
  rows crash the squad-selection insert with no aggregation semantics defined for what a DGW
  player's combined squad value should even mean. That finding was scoped to the live target
  gameweek specifically, not a guarantee about the historical seasons M7 walks through --
  `backtest.has_double_gameweek()` extends the same existing v1 scope boundary into the
  walk-forward loop (skip those 4 gameweeks, recorded plainly in `warm_up_gameweeks`) rather
  than inventing new DGW-aggregation modeling mid-M7.
- **`source_tier_weights` turned out not to be recalibratable the way this module's own design
  first assumed.** They are not resolved live per model-run at all: `ingest_workbook.
  build_sources()` bakes `tier_weight * log-scaled(citation_count)` into
  `evidence_claims.source_reliability_score` once, at ingestion time, and
  `evidence_blend.effective_weight()` never re-joins it (deliberately -- "snapshotted at
  ingestion, so later re-scoring of a source never silently reweights old claims" per M0's own
  architecture). Re-testing a candidate tier weight would mean re-running the whole evidence
  workbook ingestion per candidate, not re-running `minutes_model.run()`. Caught before it
  shipped with the wrong wiring, not after -- `refit_minutes_and_evidence_params()`'s
  coordinate descent covers `fact_type_multiplier_params` (which *is* live-resolved every run)
  in tier weights' place.
- **`evidence_claims.ingested_date <= asof` is a wall-clock-per-ingestion-batch stamp, not a
  per-claim-knowable-date one** (`ingest_workbook.py`: one `datetime.now()` shared by every
  claim in a batch) -- meaning every historical claim in this DB carries an ~Aug-2026
  timestamp regardless of which season it describes. Live runs never notice
  (`CALIBRATION_ASOF_DATE = date.today()` in `run_ingestion.py`, so the check is a harmless
  no-op), but a backtest `asof` pinned into 2024-25/2025-26 is always *before* that timestamp,
  so the existing check would have silently excluded all evidence for every single backtest
  gameweek -- confirmed the bug is backtest-only before trusting anything already built on
  M0-M6's live output. Fixed with `snapshot.get_claims_asof(..., enforce_ingested_date=False)`,
  a new opt-in parameter (default `True` leaves the live path untouched) that drops the
  `ingested_date` condition and relies on `observed_date` + the existing asof-relative
  `superseded_by` logic alone -- there is no real per-claim ingestion-timing data to fall back
  on historically, so `observed_date` is the honest ceiling on precision, not a workaround.
- **`xi_club_concentration_cap` is deliberately excluded from `recalibrate()`, and the real
  backtest data now empirically justifies that exclusion, not just the original argument for
  it.** M5's own spec frames the cap as a redundant backstop against `lambda`'s mean-variance
  mechanism silently failing (stub Sigma, solver falling back to linear-only -- the project's
  own documented history), "written as its own independent constraint so the protection holds
  even if the squad-level cap were ever loosened." Grid-searching it against the same
  `realized_sharpe` signal `lambda` is tuned against would erode exactly the redundancy it
  exists for. `backtest.report_concentration_sensitivity()` (read-only, writes nothing) tested
  this directly on the real 63-gameweek warm+mature backtest: at the current `lambda=0.15`,
  `realized_sharpe` is *identical* across cap in {2,3,4,5} -- the risk-aversion term alone
  already keeps concentration below even the tightest cap tried, so the guardrail is fully
  inert there. At the recalibration's proposed `lambda=0.0`, the cap suddenly matters: cap=2
  scores measurably worse (3.764) than cap in {3,4,5} (3.809, identical to each other) -- the
  guardrail becomes genuinely load-bearing exactly when the primary mechanism is switched off,
  and the *current* cap value (3) already captures the full benefit a looser one would give.
  Real, data-backed confirmation of the redundancy relationship the design intended, not just
  an argument for it.
- **The `lambda_value` recalibration proposal (0.15 -> 0.0) needs careful framing, not literal
  application.** `refit_lambda()`'s out-of-sample grid search found `lambda=0.0` scored a
  higher `realized_sharpe` (3.809 vs 3.709) than the current pin across the real 63-gameweek
  warm+mature window -- but this is not evidence the risk mechanism should be removed. The
  concentration-sensitivity finding directly above shows *why*: `xi_club_concentration_cap`
  was held fixed at 3 throughout this search (deliberately, per its own exclusion above) and
  was doing real, independent diversification work at `lambda=0` that a bare Sharpe ratio on
  realized points doesn't itself capture -- the original `lambda=0` back-five failure this
  whole module exists because of was a *concentration* failure, and the guardrail that would
  have caught it was never disabled during this test. Recorded as a `pending`
  `recalibration_proposals` row, exactly the kind of output the spec's own qualitative "could a
  human beat this by eye" review step exists to weigh, not to auto-apply.
- **Everything M7 touches was verified against the real project database, not a scratch
  fixture, following the same discipline as every prior milestone** -- but only after two
  scratch dry runs against a full copy of the real DB first caught the schedule-visibility and
  Double-Gameweek bugs above, cheaply and safely, before either could corrupt the real run.
  The real production run (`scripts/run_backtest.py`) completed all 71 gameweeks and 2 of 5
  recalibration techniques before an external process interruption (not a code bug -- confirmed
  by the partial results already being fully valid, immutable `param_versions`/
  `recalibration_proposals` rows); resumed cleanly by calling `backtest.recalibrate()` again
  against the same `backtest_run_id` with the completed techniques' flags turned off, without
  needing to redo the expensive `run()` pass -- direct evidence the "each proposal is its own
  immutable row, nothing is activated automatically" design (see `propose_recalibration()`) is
  also what makes this kind of interruption cheap to recover from, not just auditable.
- **DuckDB refuses to `ALTER` a table that other tables have foreign keys into, and validates
  FK constraints against the real catalog table even when a query would otherwise resolve a
  bare table name through a `TEMP TABLE` shadow -- both confirmed the hard way building M8,
  not assumed from documentation.** Triple Captain needs a real `monte_carlo.run()` simulation
  of the manager's *actual* current holdings, but `monte_carlo_run_versions.
  squad_optimizer_run_id` is a `NOT NULL` FK into `squad_optimizer_runs`, and the manager's
  holdings were never a real M5 solve. Two approaches were tried and failed before the one
  that works: (1) making the FK column nullable -- blocked, DuckDB won't `ALTER` a table
  (`monte_carlo_run_versions`) that other tables (`monte_carlo_player_totals`/`summary`/
  `empirical_covariance`) FK into, confirmed by direct testing against the real schema, not a
  design choice; (2) reusing M7's connection-scoped `TEMP TABLE` shadow of
  `squad_optimizer_runs` (the same trick `backtest.asof_scope()` uses for `fact_match`) --
  satisfies `monte_carlo.run()`'s own `SELECT` queries fine (confirmed it got that far in a
  real run before crashing), but the subsequent `INSERT INTO monte_carlo_run_versions` still
  fails FK validation, because DuckDB checks the constraint against `main.squad_optimizer_runs`
  regardless of what a `TEMP TABLE` of the same name would resolve a plain query through. Also
  confirmed no escape hatch exists: no `SET foreign_keys=...` pragma, and `ALTER TABLE ... DROP
  CONSTRAINT` raises `NotImplementedException` outright in this DuckDB version. The fix that
  actually works: `squad_optimizer_runs.is_manager_snapshot` (schema/0010, a plain `ADD COLUMN
  ... DEFAULT FALSE` with no inline `NOT NULL` -- the one `ALTER` variant DuckDB *does* permit
  on a table with dependents), and `transfer_planner._write_manager_snapshot_as_optimizer_run()`
  inserts a real, permanent, clearly-flagged row there instead of a temporary shadow.
  `monte_carlo.run()` itself needed zero changes. The row can never be deleted afterward either
  (the same "can't modify a table with FK dependents" limitation, once
  `monte_carlo_run_versions` references it) -- a disclosed, one-way trade-off, not a hidden one.
- **The same "can't modify a row with FK-referencing children" limitation showed up a second
  time**, independent of the above: `apply_recommendation()` originally tried to `UPDATE
  transfer_plan_runs SET output_state_version = ...` after accepting a recommendation, and
  DuckDB rejected it because `transfer_recommendations`/`chip_evaluations` both FK into
  `transfer_plan_runs` -- even though `output_state_version` isn't the referenced key column at
  all. Fixed by flipping the link's direction: `manager_state_versions.produced_by_run_id` is
  set once, at `INSERT` time, rather than back-filled onto the parent row afterward -- avoids
  ever needing to `UPDATE` a row with dependents, not just for this case but as the general
  pattern worth knowing before adding the next module here.
- **`evaluate_transfers()`'s exhaustive single-transfer search enforces three constraints a
  transfer must satisfy to be real, not optional embellishments**: same position, incoming
  price no greater than outgoing (no banked-budget tracking exists yet -- out of the locked
  spec's stated scope, conservatively assumes zero bank rather than silently ignoring budget
  entirely), and the post-swap per-club count staying within the same `<=3` guardrail M5
  itself enforces. All three are cheap, pure in-memory checks once the horizon EP is
  pre-fetched once per candidate pool (not per candidate pair) -- roughly 8,000 evaluations
  per planning call, negligible next to the real cost centers (the SCIP solves and Monte
  Carlo simulation).
- **The spec's "M4's variance naturally widening for further-out gameweeks" is not actually
  true of the current implementation, confirmed by reading the code before building anything
  to compensate for it.** `team_strength.calibrate()` produces one horizon-agnostic snapshot
  reused unchanged for every target gameweek; `uncertainty.run()` has no calendar-distance-to-
  target term at all. No new uncertainty-inflation formula is invented here -- the locked spec
  presents this as a description of an assumed mechanism, not a numbered requirement, so
  `compute_horizon_ep()` inherits whatever variance M4 actually produces per horizon gameweek,
  undecorated. Named here so a future module doesn't assume it's already handled.
- **A genuine, real finding from the verified GW1->GW2 2026-27 run, not silently smoothed
  over: Triple Captain comes out a near-exact tie between a defender and a premium
  attacker, not a confident pick.** Bruno Fernandes has the highest raw simulated mean
  (4.34 points) of any XI candidate, but also the highest variance (`sqrt(var)` ~4.29);
  Marcos Senesi Barón's mean is lower (4.23) but his variance is under half of Bruno's
  (`sqrt(var)` ~2.93). At the invented `kappa_tc=0.15` (pinned to match `lambda_value` for
  lack of any other anchor, same status as every other risk-preference default in this
  project), the risk-adjusted scores land at 3.79 vs 3.71 -- Senesi barely ahead, well within
  what 5,000-antithetic-pair simulation noise could plausibly move. Directly continues the
  pattern M5's own README entry already documented for its real GW1 captain pick ("notably
  favors several high-DefCon defenders... over some premium attackers") -- independent
  evidence of the same real signal from a completely different evaluation path, not a
  coincidence and not a bug.
- **Verified against the real project database following the same two-stage discipline as
  M7**: a scratch dry run against a full copy of the real DB first (confirmed the whole
  pipeline runs clean end to end, ~22 minutes), then the real bootstrap-and-plan run against
  `db/fpl_quant_v2.duckdb` itself, producing results consistent with the dry run's (same top
  transfer recommendation, same near-tie Triple Captain pattern) -- reproducibility across two
  independent runs against the same real GW1 squad, not a one-off.
- **M9's evidence-provenance adapter needed "which claims actually moved a given minutes
  adjustment," but `minutes_model.compute_logit_adjustment()` only ever returned the summed,
  capped float -- no per-claim log existed anywhere.** Changing that function's return type
  would break every existing caller and test (a real, tested, frozen function, not a free
  rewrite). Fixed with a new, separate function, `explain_player_adjustment()`, kept
  deliberately side by side with the original so the two claim-filtering code paths (same
  skip-completed-transfer check, same missing-param skip, same manager_tendency sign flip)
  stay easy to keep in lockstep rather than duplicated somewhere distant. Regression-tested
  directly against the property that actually matters: summing `explain_player_adjustment()`'s
  per-claim `contribution` values and applying the same cap reproduces
  `compute_logit_adjustment()`'s real output exactly, for the same synthetic claims -- the two
  are required to agree, not just both "look reasonable" independently.
- **The exact same `dict()`-on-3-tuples mistake this README already documents from M6's first
  real run happened again, verbatim, building M9's evidence-provenance adapter** --
  `dict(con.execute("SELECT source_id, source_name, source_type FROM sources").fetchall())` on
  3-column rows, same `ValueError: dictionary update sequence element #0 has length 3; 2 is
  required`. Caught this time by the unit test suite before any real run, not by a live crash
  -- direct evidence that regression-testing an adapter against a real synthetic fixture (not
  a mock) catches the same class of bug a live run would have, not a weaker substitute for one.
  Fixed the same way as M6's: an explicit dict comprehension, not the `dict()` constructor, on
  anything wider than 2 columns.
- **A second real bug the test suite caught: `squad_optimizer.explain_run()`'s captained-
  goalkeeper check silently no-opped whenever club-resolution data wasn't found, because it
  was wrongly folded into the same query as the club-count audit.** Both the captain check and
  the club audit originally lived inside one query gated on `reconcile._season_root_table()`
  finding a raw teams table for that season -- reasonable for the club audit (which genuinely
  needs that join), but the captained-GK check only ever needs `squad_optimizer_selections.
  is_captain` joined straight to `dim_player.position`, no club data involved at all. Coupling
  them meant a missing/delayed season-root table would silently disable the *cheaper and
  arguably more important* of the two checks along with the one that actually depended on it.
  Split into two independent lookups: captain-position detection is now unconditional; the
  club audit remains best-effort and degrades gracefully on its own. A test seeding a
  goalkeeper captain without season-root data caught this before it shipped.
- **The parameter-transparency panel's real count from the live GW1 2026-27 report: 65 of 71
  active parameters are still purely invented, never touched by M7's recalibration** (`SELECT
  DISTINCT param_family FROM recalibration_proposals`) -- a real, load-bearing number now that
  it's computed rather than estimated, and a concrete measure of how much of this project's
  own stated goal (replacing invented defaults with backtested values) remains. All four of
  M8's own families (`planning_horizon_params`, `transfer_cost_params`,
  `tc_risk_aversion_params`, `wildcard_gain_threshold_params`) were in that 65 at M9's own
  completion -- M7 predates M8, so `kappa_tc`'s own spec-stated flag ("flagged for M7
  recalibration") had not yet been acted on. Named here, not left implicit, as the natural next
  piece of follow-up work -- see the two notes below for what was actually done about it.
- **`refit_kappa_tc()`'s realized_sharpe objective does not have a clean interior optimum --
  disclosed, not hidden behind a conveniently narrow default grid.** Diagnostic grid search
  beyond the default candidates (0.75, 1.0, 1.5, 2.0) against the real `backtest_run_id=1` data
  shows `realized_sharpe` keeps climbing as `kappa_tc` grows (1.196 at 0.5, 1.500 at 2.0, no
  turnaround found) -- extreme risk aversion in captaincy trivially minimizes the *realized
  variance* term in the denominator by always captaining the flattest, lowest-ceiling player in
  the XI, and Sharpe rewards that even as mean captained points keep falling (9.16 at the
  current 0.15 down to 7.45 at 2.0). This is a real weakness of using the same realized_sharpe
  metric `refit_lambda()` uses for squad-level risk aversion, applied here to a single-player
  argmax choice with no portfolio effect to offset it -- worth a human's explicit skepticism
  before accepting proposal #6 (0.15->0.5), not a reason to silently widen the grid until it
  produces whatever answer looks most defensible. The default `kappa_tc_grid` in `recalibrate()`
  deliberately stays bounded at 0.50, matching `lambda_grid`, rather than being hand-tuned
  around this finding.
- **`wildcard_gain_threshold_params.min_horizon_gain` is deliberately not covered by the
  `kappa_tc` extension, or by any M7 technique.** Backtesting it would mean re-running M8's own
  manager-state bootstrap and evolution across the full 71-gameweek backtest history -- M7's
  walk-forward squad is M5's from-scratch pick every single step, not an evolving manager
  holding, so no equivalent "what would the manager have owned at gameweek N" state exists in
  M7's infrastructure to compare a wildcard's gain against. A materially larger, separately-
  scoped piece of work, not a small addition to `recalibrate()` -- named as a real, open gap.
