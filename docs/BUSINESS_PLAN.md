# FPL Quant — Business Readiness & Free-Tier Roadmap

This document is an honest, outside-in read on what it would take to turn FPL Quant from a
working personal project into a sustainable, revenue-capable product — and how far that can go
without spending anything on infrastructure. It is written to be read by a maintainer deciding
what to build next, not as marketing copy.

It assumes the reader already knows the project's architecture (see `README.md`). It does not
re-explain the M0–M9 quant pipeline.

---

## 1. Where the project already is (strengths to preserve)

- **A genuinely working product, not a mockup.** The PWA loads real FPL data for two real
  accounts, renders six tabs (Home / Team / Transfers / Fixtures / Leagues / Plan), and gives
  model-computed transfer, captain, and chip-timing directives. Onboarding modal, dev/mock mode,
  pull-to-refresh, a local-only planning sandbox, and live-match polling are all already there.
- **An honest free-tier backend that scales to "any user, no server bill."** The whole multi-user
  path costs $0 in hosting:
  - **GitHub Issues** as the request queue (a user opens `[add-manager] <id>`).
  - **GitHub Actions** as the compute backend (`on_demand_report.yml` restores a cached DuckDB
    snapshot and runs the M8/M9 planner for that entry — ~1–2 min).
  - **raw.githubusercontent.com** as the data host (reflects a freshly committed JSON within
    ~1 min, which the add-team SLA depends on).
  - **A per-author rate limit (3/day)** and a serialized `concurrency` group as the anti-abuse
    control, instead of a manual collaborator gate.
- **A real, defensible quant core.** Dixon-Coles team strength, a genuine minutes model, a
  Plackett-Luce bonus marginalization, MIQP squad optimization via SCIP, Monte-Carlo simulation,
  and an explainability layer ("Explain my move", "what would change my mind"). That depth is the
  thing a competitor cannot trivially copy in a weekend.
- **Tested.** ~480 Python tests + 76 Node planner tests gate `master`. This is rare for a side
  project and is the single biggest reason it can be trusted as a product.

The conclusion from this section: **do not throw away the free-tier architecture.** It is the
project's structural advantage. The roadmap below builds on it rather than replacing it.

---

## 2. Gaps blocking "real business"

### 2.1 Public hosting & first impression (P0 — cheap, high leverage)
The PWA itself has no reliable public URL today. A new user has no link to click. This is the
single cheapest, highest-leverage gap to close, and it is addressed in this PR:

- **GitHub Pages** hosts the static shell for free at `https://<owner>.github.io/FPL-Analyser/`
  (see `.github/workflows/deploy_pages.yml`). The shell is intentionally the only thing deployed;
  data stays on `raw.githubusercontent.com` for fast per-request refresh (the add-team SLA depends
  on raw, and Pages' CDN cadence is too coarse for that path).
- One-time setup: repo **Settings → Pages → Source = "GitHub Actions"**. No secrets needed.

### 2.2 Trust & conversion (P0)
The Home screen already carries an honest disclaimer ("v1 quant model, un-recalibrated, not yet
backtested"). That honesty is correct and should stay — but it currently undercuts conversion
because there is no proof the model works. To turn visitors into users:

- **Backtest + calibration proof on a public page.** The pipeline already has a backtest module
  (`backtest.py`, `run_backtest.py`, `schema/0008_m7_backtest.sql`) and a recalibration flow
  (`review_recalibration.py`, `data/recalibration/`). Publish a simple "model track record" page
  from `data/report_history/` showing realized accuracy vs. the model's projections over the
  season. `app_track_record.json` already exists; surface it as a real page, not a hidden file.
- **Transparent data freshness.** The "Last run" timestamp is already shown. Add a visible
  "data sources" note (official FPL API, Understat xG) so users know this is real data, not guesses.

### 2.3 Model quality (P1 — the core moat, larger effort)
The disclaimer is honest because the model is genuinely v1. For a paid product this must improve,
but it is the project's *moat*, not a quick fix. Prioritized sub-items:

- Run the existing recalibration flow on a cadence and feed `review_recalibration.py` output back
  into versioned parameters (the `params.py` mechanism already supports immutable versions).
- Close the known modeling gaps documented in `expected_points.py`'s scope statement (penalty
  takers, cards/own-goals, BPS passing components) — these are intentionally 0 today, which is
  honest but costs accuracy.
- Backtest the *transfer planner's* realized decisions, not just per-player EP — a manager cares
  whether "hold vs. use" calls were right, not just whether Haaland's xP was calibrated.

### 2.4 Optional feature exports currently 404 (P2 — wire once inputs are stable)
Three optional `_latest` files are fetched with `.catch(() => null)` and currently 404 because
their generator scripts are not wired into any workflow:

- `data/dashboard/projections_latest.json` (`scripts/export_projections.py`)
- `data/dashboard/leaderboard_latest.json` (`scripts/export_leaderboard.py`)
- `data/dashboard/elite_divergence_latest.json` (`scripts/track_elite.py`)

The app degrades gracefully (these are non-blocking), but wiring them into
`scheduled_pipeline.yml` would complete the Home screen's "model track record / elite
divergence" sections. Do this only after confirming each script's inputs are stable and the run
is cheap/safe — wiring a script that can fail into scheduled CI expands the failure surface for
every scheduled run. (Left as a follow-up, intentionally not in this PR.)

### 2.5 Fork-friendliness / decoupling the data URL (P2 — done, app gap 7)
`index.html` used to unconditionally use `RAW_FALLBACK` on every deployed host. Now
`resolveDataBase()` is origin-aware **without changing the canonical deployment**:

- **Canonical site** (`ahmadrehman1.github.io/FPL-Analyser/`) — still `RAW_FALLBACK`. The
  add-team SLA (raw reflects a commit in seconds; Pages' CDN cadence is too coarse for the
  per-request data path) is untouched, and `deploy_pages.yml` still stages the shell only.
- **localhost / 127.0.0.1** — same-origin `data/` on disk, as before.
- **A staging fork** that wants to experiment against its *own* data without touching the raw
  feed the real live-followed teams read: set the repo variable `FQ_STAGING=1` (so
  `deploy_pages.yml` also bundles `data/dashboard/` into the Pages artifact) and add
  `<meta name="fq-data-base" content="">` to `index.html` (or set `window.FQ_DATA_BASE`).
  `resolveDataBase()` then serves data same-origin from the fork's own Pages site.

### 2.6 Analytics & engagement (P2 — privacy decision required)
There is currently no measurement of which recommendations users act on, retention, or funnel
drop-off. For a business this is a real gap, but **do not add tracking code without an explicit
privacy decision** — the app currently sends nothing about the user anywhere (local-only state is
in `localStorage` and never transmitted). If measurement is wanted, prefer a privacy-first,
self-hostable option (e.g. Plausible/Umami) over third-party trackers, and disclose it in-app.

### 2.7 Custom domain & branding (P2)
For a real brand, point a custom domain at the GitHub Pages site (free via a CNAME; Cloudflare's
free tier can sit in front as CDN + DNS). The repo already has its own icons/manifest. A custom
domain is a config change, not code.

---

## 3. Monetization paths (all free-tier-compatible)

The architecture supports growth without a server bill, so monetization can start before any
infra spend:

1. **Freemium, gated by the existing rate limit.** The add-team flow already caps 3 requests per
   GitHub author per day. Keep that as the free tier; raise it / remove it for supporters. No new
   infra — just a role/label check in `on_demand_report.yml`.
2. **Supporter / "buy me a coffee" tier.** Lowest-friction first revenue: a link in the Profile
   sheet. Zero infra, zero payment integration in the app itself.
3. **Premium model tier.** The moat is the quant depth. A paid tier could expose the multi-week
   transfer planner (`planner/`), chip-combo evaluation, and Monte-Carlo scenarios — features
   already built — as advanced views. Monetization is an access gate, not new model code.
4. **B2B / league & content-creator tools.** Private mini-league tools (already partially present
   via `app_leagues_<id>.json` ownership/captaincy breakdowns) could be sold to league admins or
   FPL content creators who want a branded standings + differential view.

The key point: **none of these require leaving the free tier to start.** Paid infra only becomes
worth it when GitHub's free Actions allowance for public repos (or the raw/Pages rate limits) is
actually hit — and by then revenue exists to fund it.

---

## 4. Prioritized roadmap

| Priority | Item | Effort | Why |
|----------|------|--------|-----|
| P0 | Public GitHub Pages hosting of the shell (this PR) | S | No public URL = no users. Free. |
| P0 | One-time Pages setup (Settings → Pages → GitHub Actions) | S | Required for the workflow above to deploy. |
| P0 | Publish a real "model track record" page from existing `data/report_history/` + `app_track_record.json` | M | Converts visitors by proving the model works. |
| P1 | Cadence the recalibration flow; feed it back into versioned params | L | Core moat. Turns "honest v1" into "calibrated". |
| P1 | Backtest the *planner's* realized decisions (hold vs. use), not just per-player xP | L | A manager's real question. |
| P2 | Wire the 3 optional `_latest` exports into `scheduled_pipeline.yml` (after input stability) | S–M | Completes Home screen sections; removes silent 404s. |
| P2 | Privacy-first analytics (Plausible/Umami, self-hosted) behind an explicit privacy decision | M | Enables retention/funnel measurement. |
| P2 | Custom domain via CNAME + Cloudflare free tier | S | Real brand. |
| P2 | Origin-aware data base (relative when same-origin, raw fallback) | S | Fork/rename-friendliness. Deliberately deferred. |

---

## 5. What this PR does and does not do

**Does:** adds free GitHub Pages hosting for the static shell (`.nojekyll` +
`deploy_pages.yml`), and this business-readiness roadmap.

**Does not (deliberately):** touch the core data-loading path in `index.html` (the raw-URL design
is intentional and better for the add-team SLA); wire the optional `_latest` exports into CI
(pending input stability); add analytics tracking (privacy decision pending); change the quant
model. Each of these is a documented follow-up above rather than an oversight.
