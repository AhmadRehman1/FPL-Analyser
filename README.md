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
scripts/run_ingestion.py       -- end-to-end pipeline runner
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
