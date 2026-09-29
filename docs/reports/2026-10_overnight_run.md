# Overnight run, 2026-09-28/29

_Draft - scoreboard filled in when the walk-forwards finish._

## 1. Needs Ahmad

### Cloudflare dashboard (do this first)

Your existing Worker is called **fpl-analyser**, so `wrangler.toml` now uses that name (a mismatch
makes Workers Builds fail). Open https://dash.cloudflare.com and then:

1. **Workers & Pages** -> click **fpl-analyser**.
2. **Settings** tab -> scroll to **Build**.
3. **Build configuration** -> **Edit**:
   - Build command: `bash scripts/build_cloudflare_site.sh`
   - Deploy command: `npx wrangler deploy`
   - Non-production branch deploy command: `npx wrangler versions upload` (the default is fine)
   - Root directory: leave **blank** (`/`)
   - **Save**.
4. **Branch control** -> **Edit** -> Production branch: `master` -> **Save**. Leave "builds for
   non-production branches" on if you want preview URLs on PRs, off if you don't.
5. **Build watch paths** -> **Edit** (this stops the 1-2 hourly `[skip ci]` data commits
   rebuilding the site):
   - Include paths: `index.html, landing.html, track-record.html, sw.js, manifest.json, icons/*, assets/*, planner/*, scripts/site_files.txt, scripts/build_cloudflare_site.sh, scripts/cloudflare/*, wrangler.toml`
   - Exclude paths: `data/*`
   - (Simpler alternative if the include list is fiddly: include `*`, exclude `data/*`.)
   - **Save**.
6. Optional, faster builds: **Variables and secrets** (build section) -> add `SKIP_DEPENDENCY_INSTALL`
   = `1`. The build needs no npm or pip packages (`npx` fetches wrangler itself); without it the
   build image may spend time installing `requirements.txt`.
7. **Deployments** tab -> **Retry build** on the latest one (or merge PR #189, which triggers it).
   The site is then at `https://fpl-analyser.<your-subdomain>.workers.dev`.

Check it worked: the page loads, the header says "Live data - updated ...", and the Plan tab
(the multi-GW planner) opens.

### GitHub Pages source (one setting)

Live github.io is serving the whole repo from a branch (e.g. `/tests/conftest.py` and
`/pyproject.toml` are public), not the `deploy_pages.yml` shell. Settings -> Pages -> Build and
deployment -> Source = **GitHub Actions**. Safe once PR #189 is merged (it adds the planner, which
the workflow's old file list was missing).

### Expert consensus file

Commit your Cowork file to `data/external/expert_consensus/expert_consensus_GW<n>_<YYYY-MM-DD>.md`
(PR #192 un-ignores that folder), then run
`PYTHONPATH=src python scripts/ingest_expert_consensus.py --validate-only` to see what it rejects.

## 2. PRs opened

_(filled in below)_

## 3. Scoreboard

_(filled in below)_

## 4. Tried and didn't work

_(filled in below)_

## 5. Decisions I made without you

_(filled in below)_

## 6. What I'd do next

_(filled in below)_
