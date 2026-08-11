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
    evidence_blend.py          -- M1b: effective-weight formula + conflict-resolution blending
    team_strength.py           -- M1: Dixon-Coles MLE fit, Elo-regression prior, shrinkage
    minutes_model.py           -- M2: historical fit, evidence adjustment, three-state output
scripts/run_ingestion.py       -- end-to-end pipeline runner
tests/                         -- pytest, one file per module concern
data/external/                 -- gitignored; extracted FPL-Core-Insights repo +
                                   FPL_202627_Master_Evidence_Database.xlsx go here
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
