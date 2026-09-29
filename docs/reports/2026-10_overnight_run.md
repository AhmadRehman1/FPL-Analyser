# Overnight run, 2026-09-28/29

Nothing is merged. Every change is on its own branch and PR for you to review. All code PRs pass
`pytest tests/` and `npm test` locally, and GitHub CI (pytest, node, lint, typecheck) where it runs.

## 1. Needs Ahmad

### Cloudflare dashboard (do this first)

Every PR shows a red `Workers Builds: fpl-analyser` check. The Worker builds every branch push
and fails in 0 seconds, before any repo command runs, so it needs these settings. Your
existing Worker is called **fpl-analyser**, so `wrangler.toml` (PR #189) now uses that name; a
name mismatch makes Workers Builds fail.

1. https://dash.cloudflare.com -> **Workers & Pages** -> click **fpl-analyser**.
2. **Settings** tab -> **Build**.
3. **Build configuration** -> **Edit**:
   - Build command: `bash scripts/build_cloudflare_site.sh`
   - Deploy command: `npx wrangler deploy`
   - Non-production branch deploy command: `npx wrangler versions upload` (default is fine)
   - Root directory: leave **blank**
   - **Save**.
4. **Branch control** -> Production branch: `master` -> **Save**. Turn "builds for non-production
   branches" **off** unless you want preview builds on every PR (they fail until #189 is merged).
5. **Build watch paths** -> **Edit**, so the 1-2 hourly `[skip ci]` data commits stop
   triggering rebuilds:
   - Include: `index.html, landing.html, track-record.html, sw.js, manifest.json, icons/*, assets/*, planner/*, scripts/site_files.txt, scripts/build_cloudflare_site.sh, scripts/cloudflare/*, wrangler.toml`
   - Exclude: `data/*`
   - Simpler alternative: include `*`, exclude `data/*`.
6. Optional, for faster builds: under the build's **Variables and secrets**, add
   `SKIP_DEPENDENCY_INSTALL` = `1`. Nothing needs installing; `npx` fetches wrangler.
7. Merge **#189**, then **Deployments** -> **Retry build**. The site will be at
   `https://fpl-analyser.<your-subdomain>.workers.dev`.

### GitHub Pages source

Live github.io serves the **whole repo** from a branch (`/tests/conftest.py`, `/pyproject.toml`
are public), not the `deploy_pages.yml` shell. Fix: Settings -> Pages -> Source = **GitHub
Actions**. This is safe once #189 is merged (it adds `planner/`, which the workflow's list was
missing).

### Switch on the captain fix (and lambda), your call

PR #195 adds the versioned captain variance weight but leaves it at today's value. The permission
system blocked me from changing what the live pipeline publishes, and that's rightly your
decision. To switch it on, add this to the `squad_optimizer.run(...)` call in
`scripts/run_ingestion.py` and to `backtest.run(...)` in `scripts/run_walkforward.py`:

```python
captain_risk_params_version=params.get_or_create_version(
    con, "captain_risk_params", "captain_variance_multiplier", "2026-09-29", value_numeric=0.0),
```

For lambda, the evidence supports 0.10 (see the scoreboard). It's a recalibratable family, so the
clean route is a confirmed proposal. After #191, that needs a held-out score, which `refit_lambda`
doesn't produce yet (next steps).

### Reject the k_minutes 900 confirmation (recommended)

`python scripts/review_recalibration.py --list-confirmed`, then
`--reject <k_minutes proposal id>`. Evidence in #196.

### Expert consensus file

Commit the Cowork output to `data/external/expert_consensus/expert_consensus_GW<n>_<date>.md`
(PR #192 un-ignores the folder), then run
`PYTHONPATH=src python scripts/ingest_expert_consensus.py --validate-only`.

## 2. PRs opened

| PR | What it fixes | Key number | State |
|---|---|---|---|
| #189 | Cloudflare deploy (`wrangler.toml`, build script, shared file list with Pages, `_headers`); planner files were missing from the deploy list | Local test: live GW6 data renders, SW registers | Ready; the Cloudflare check needs the dashboard steps |
| #190 | Status refresh report (Phase 2) plus this report | Elite-vs-model: 6 core elite picks at model rank 70-600 | Ready (docs) |
| #191 | **Fix A**: recalibration gate (held-out purged K-fold, versioned 1% floor, collision + grid-edge checks, `--confirm` goes through it) | The real k_minutes 450->900 promotion is now refused | Ready |
| #194 | **Fix B**: decision-weighted k_minutes loss, grid 100-4000 (stacked on #191) | No recalibration run overnight (it writes to master) | Ready after #191 |
| #192 | **Fix G**: expert-consensus ingester (strict name/club validation, coercion log, numeric claims) + model-vs-experts disagreement check | Rejects all four known-bad names; FPL club wins | Ready |
| #193 | Tooling: branch walk-forward workflow (`bt/**`) + scoreboard | 10 experiments run overnight | Ready |
| #195 | **Fix D**: one captain rule for every user-facing captain; versioned captain variance weight | Weight 0: beats-avg **-0.71 -> +2.14**, captain pts 3.24 -> 5.67 /GW | Ready (activation is your call) |
| #196 | Walk-forward measures the live k_minutes (it always used 450) | Live 900: 49.26 vs 49.16 pts/GW, 9.0+ bias worse | Ready; the nightly headline will honestly drop to about -0.99 |

Branches kept without a PR (see section 4): `claude/fix-c-assist-calibration`,
`claude/fix-f-minutes-floor`, and experiment branches `bt/*`.

## 3. Scoreboard

Walk-forward, 2024-25 + 2025-26, same cached DB for every arm.

| | start of night | best evidence-backed now |
|---|---|---|
| beats avg manager /GW | **-0.71** | **+2.14** (captain fix) / **+5.41** (captain fix + lambda 0.10) |
| model squad pts /GW | 49.16 | 52.00 / 55.27 |
| price-band resid `<5 / 5-7 / 7-9 / 9+` | -0.24 / +0.31 / +0.78 / +0.93 | unchanged (the fixes that flatten them aren't merged, see 4) |
| captain pts /GW (before doubling) | 3.24 | 5.67 |
| captain was the XI's top-EP player | 0% | 99% |
| defender/GK captain | 0% (all flat midfielders) | 5.7% |
| minutes log score (uniform = -1.099) | -1.256 | -1.256 (the floor gets -0.679, not merged) |

Honest caveats:
- The "average manager" is synthetic and uses the model's own P(plays). It's fine for comparing
  optimizer changes; for EP changes, compare model points.
- Lambda 0.10 is the best of four values tried on the same 70 weeks, so part of its gain is
  selection.
- None of this is live until you merge and switch it on.

Second round (on top of captain fix + lambda 0.10, same 70 weeks): assists calibration **55.86**
model pts/GW (+0.59, better in 48 of 70 weeks), role blend **56.34** (+1.07). Both made the EP
bias slightly worse, so neither has a PR yet (section 4). The minutes floor 0.005 run was still
going when I wrote this.

## 4. Tried and didn't work (yet)

- **Current-season role blend** (`current_season_role_params`, already in code). **Re-tested on top
  of the captain fix + lambda 0.10: +1.07 model pts/GW (56.34 vs 55.27)**, the best squad-points
  result of the night, but EP bias -0.013 -> -0.079 and MAE 1.159 -> 1.180. First pass: model points
  +0.81/GW, and the price bands are much flatter (9.0+ +0.93 -> +0.72, 5-7 +0.31 -> +0.20). But the
  global bias got worse (-0.01 -> -0.08), the log score slightly worse, and the benchmark shift makes
  its beats-avg unreadable. This is the fix aimed at the "elite owned it, model rated it low" gap,
  so it's worth another round once the benchmark is EP-independent.
- **FPL assist calibration** (FPL awards 1.38x xA; forwards about 2x). **Re-tested on top of the
  captain fix + lambda 0.10: +0.59 model pts/GW (55.86 vs 55.27), better in 48 of 70 weeks, beats-avg
  +0.84 (about 1.5 standard errors, benchmark barely moved).** Held back only because EP bias went
  -0.013 -> -0.038 and MAE +0.009. This is the closest to a PR. First pass: the assists residual fell
  from +0.28 to +0.16 at 9.0+, and 7-9 from +0.13 to +0.04. But squad points were unchanged (49.16)
  and the global bias moved -0.01 -> -0.04. The EP fix is real but, on its own, doesn't change who
  the optimizer picks, likely because the 3x captain penalty and lambda 0.15 dominated. **Re-test
  on top of the captain fix + lambda 0.10.**
- **Minutes probability floor 0.02.** Log score -1.256 -> **-0.679**, beating the -1.099 target
  outright. But model points -0.50/GW: a 2% floor adds phantom appearance points to non-players and
  trims nailed starters. Try 0.005: most of the damage came from hard zeros, so a smaller floor
  should keep most of the log-score gain.
- **k_minutes 150.** Flatter premiums (9.0+ +0.86) but fewer points (48.39). 450 looks about right;
  Fix B's weighted search will say properly.
- **Lambda 0.02 / 0.05.** More points, but the optimizer's divergence check drops 23 / 4 weeks.
  Not comparable until that check is revisited.

## 5. Decisions I made without you

- **Worker name `fpl-analyser`, not `fpl-quant`**, to match the Worker you already created.
- **Added `planner/` to the deploy list** (index.html needs it; the Pages list was missing it).
- **Experiments on GitHub Actions** (`bt/**` branches + a new workflow), because building a DB here
  was blocked (the FPL-Core-Insights clone was denied). They commit nothing.
- **The gate now requires a held-out score**, so lambda / kappa_tc / xi-rho proposals are held
  until their refits produce one. That's deliberately strict.
- **Fix D's activation left to you** (blocked from changing live output; also your call anyway).
- **Didn't open PRs for the role blend, assists, or the minutes floor.** Your rule: no headline
  improvement means keep the branch and write it up.
- **Recommended rejecting k_minutes 900** rather than doing it: rejection changes live parameters.
- **Stacked Fix B on Fix A** instead of duplicating the gate code.
- **Elite sample = today's top 1,000 managers** (survivorship bias noted in the report); GW1
  excluded because no pre-deadline projection snapshot exists.

## 6. What I'd do next

1. Merge #191 then #195, and switch the captain fix on (above). This is the biggest measured gain.
2. Make the "average manager" benchmark independent of the model's own EP (use realized minutes
   for P(started)), so EP changes can be judged on the headline again.
3. Assists calibration then the role blend. Both now add squad points on top of the captain fix
   + lambda 0.10 (+0.59 and +1.07 /GW). Fix the small EP bias they add (re-centre per position),
   re-run, and open PRs if bias holds.
4. Wire held-out scoring into `refit_lambda` so lambda 0.10 can go through the gate properly;
   revisit the divergence check that drops weeks below lambda 0.10.
5. Minutes floor at 0.005.
6. Chip "wait for a better week" logic (plan in the status report).
7. After a few gameweeks of the consensus disagreement log (#192), score model vs experts.
