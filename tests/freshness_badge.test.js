// PR #137 Tier-A / Tier-B freshness split follow-up: the DOM-free freshnessBadge renderer and
// its age/state helpers, extracted from index.html's main <script>. Regression guard on the
// tier thresholds (Tier A "live": amber 90m / red 4h; Tier B "model": amber 14h / red 30h),
// the fresh/stale/failed/unknown state machine, and the "updated Nm ago" text.

const { test } = require("node:test");
const assert = require("node:assert");
const { extractHtmlFn, extractHtmlConst } = require("./_extract_html_fn");

const MIN = 60 * 1000;
const HOUR = 60 * MIN;
const DAY = 24 * HOUR;

const FN_NAMES = ["escapeHtml", "fmtAge", "_freshnessText", "freshnessState", "_freshnessAge", "freshnessTsOf", "freshnessBadge", "tickFreshnessBadges"];

function load() {
  const src =
    `const fmtTime = (iso) => "at:" + iso;\n` +
    extractHtmlConst("FRESHNESS_TIERS") + "\n" +
    FN_NAMES.map(extractHtmlFn).join("\n\n");
  // tickFreshnessBadges touches `document`; give it a minimal stub so the module loads even
  // though the tests below exercise the pure functions.
  const sandbox = { document: { querySelectorAll: () => [] } };
  const keys = Object.keys(sandbox);
  const factory = new Function(...keys, src + "\nreturn { " + FN_NAMES.join(", ") + ", FRESHNESS_TIERS };");
  return factory(...keys.map((k) => sandbox[k]));
}

const api = load();
const NOW = Date.parse("2026-09-04T12:00:00Z");
const iso = (msAgo) => new Date(NOW - msAgo).toISOString();

// ---- thresholds are what the follow-up spec pinned -------------------------------------------

test("FRESHNESS_TIERS matches the spec (Tier A 90m/4h, Tier B 14h/30h)", () => {
  assert.deepStrictEqual(api.FRESHNESS_TIERS.live, { amberMs: 90 * MIN, redMs: 4 * HOUR });
  assert.deepStrictEqual(api.FRESHNESS_TIERS.model, { amberMs: 14 * HOUR, redMs: 30 * HOUR });
});

// ---- freshnessState ------------------------------------------------------------------------

test("freshnessState: live tier crosses fresh -> stale -> failed at 90m / 4h", () => {
  assert.strictEqual(api.freshnessState(10 * MIN, "live"), "fresh");
  assert.strictEqual(api.freshnessState(89 * MIN, "live"), "fresh");
  assert.strictEqual(api.freshnessState(90 * MIN, "live"), "stale");
  assert.strictEqual(api.freshnessState(3 * HOUR, "live"), "stale");
  assert.strictEqual(api.freshnessState(4 * HOUR, "live"), "failed");
  assert.strictEqual(api.freshnessState(9 * HOUR, "live"), "failed");
});

test("freshnessState: model tier crosses at 14h / 30h", () => {
  assert.strictEqual(api.freshnessState(6 * HOUR, "model"), "fresh");
  assert.strictEqual(api.freshnessState(13 * HOUR + 59 * MIN, "model"), "fresh");
  assert.strictEqual(api.freshnessState(14 * HOUR, "model"), "stale");
  assert.strictEqual(api.freshnessState(29 * HOUR, "model"), "stale");
  assert.strictEqual(api.freshnessState(30 * HOUR, "model"), "failed");
});

test("freshnessState: null/NaN age is 'unknown'; unknown tier name falls back to model", () => {
  assert.strictEqual(api.freshnessState(null, "live"), "unknown");
  assert.strictEqual(api.freshnessState(NaN, "model"), "unknown");
  assert.strictEqual(api.freshnessState(13 * HOUR, "not-a-tier"), "fresh"); // model fallback: <14h
  assert.strictEqual(api.freshnessState(20 * HOUR, "not-a-tier"), "stale");
});

test("freshnessState accepts an explicit {amberMs, redMs}", () => {
  const thr = { amberMs: 1000, redMs: 2000 };
  assert.strictEqual(api.freshnessState(500, thr), "fresh");
  assert.strictEqual(api.freshnessState(1500, thr), "stale");
  assert.strictEqual(api.freshnessState(2500, thr), "failed");
});

// ---- _freshnessAge / fmtAge / _freshnessText ---------------------------------------------------

test("_freshnessAge parses ISO datetimes and clamps clock-skew futures to 0", () => {
  assert.strictEqual(api._freshnessAge(iso(3 * HOUR), NOW), 3 * HOUR);
  assert.strictEqual(api._freshnessAge(new Date(NOW + 5 * MIN).toISOString(), NOW), 0);
  assert.strictEqual(api._freshnessAge("", NOW), null);
  assert.strictEqual(api._freshnessAge("not a date", NOW), null);
  assert.strictEqual(api._freshnessAge(null, NOW), null);
});

test("_freshnessAge accepts a YYYY-MM-DD data_asof (parsed as UTC midnight)", () => {
  const age = api._freshnessAge("2026-09-04", Date.parse("2026-09-04T09:30:00Z"));
  assert.strictEqual(age, 9 * HOUR + 30 * MIN);
});

test("fmtAge is coarse and human", () => {
  assert.strictEqual(api.fmtAge(0), "0m");
  assert.strictEqual(api.fmtAge(43 * MIN), "43m");
  assert.strictEqual(api.fmtAge(90 * MIN), "1h 30m");
  assert.strictEqual(api.fmtAge(3 * HOUR), "3h");
  assert.strictEqual(api.fmtAge(50 * HOUR), "2d 2h");
});

test("_freshnessText: 'age unknown' / 'updated just now' / 'updated Nm ago'", () => {
  assert.strictEqual(api._freshnessText(null), "age unknown");
  assert.strictEqual(api._freshnessText(30 * 1000), "updated just now");
  assert.strictEqual(api._freshnessText(12 * MIN), "updated 12m ago");
  assert.strictEqual(api._freshnessText(26 * HOUR), "updated 1d 2h ago");
});

// ---- freshnessTsOf -----------------------------------------------------------------------------

test("freshnessTsOf prefers generated_at, falls back to data_asof, else null", () => {
  assert.strictEqual(api.freshnessTsOf({ generated_at: "2026-09-04T10:00:00Z", data_asof: "2026-09-04" }), "2026-09-04T10:00:00Z");
  assert.strictEqual(api.freshnessTsOf({ data_asof: "2026-09-04" }), "2026-09-04");
  assert.strictEqual(api.freshnessTsOf({}), null);
  assert.strictEqual(api.freshnessTsOf(null), null);
});

// ---- freshnessBadge (HTML string) ------------------------------------------------------------

test("freshnessBadge renders the state class, tier, label and re-age data-* attributes", () => {
  const html = api.freshnessBadge(iso(2 * HOUR), "model", "Model", NOW);
  assert.match(html, /class="fresh-badge is-fresh"/);
  assert.match(html, /data-fresh-tier="model"/);
  assert.match(html, /data-fresh-label="Model"/);
  assert.match(html, /data-fresh-ts="[^"]+"/);
  assert.match(html, /Model · updated 2h ago<\/span>/);
});

test("freshnessBadge: live tier past 4h is failed", () => {
  const html = api.freshnessBadge(iso(5 * HOUR), "live", "Live data", NOW);
  assert.match(html, /class="fresh-badge is-failed"/);
  assert.match(html, /Live data · updated 5h ago/);
});

test("freshnessBadge: unparseable/missing timestamp -> unknown, no title attr, still carries tier", () => {
  const html = api.freshnessBadge(null, "live", "Live data", NOW);
  assert.match(html, /class="fresh-badge is-unknown"/);
  assert.match(html, /Live data · age unknown/);
  assert.doesNotMatch(html, /title=/);
});

test("freshnessBadge escapes a hostile label", () => {
  const html = api.freshnessBadge(iso(MIN), "model", '<img src=x onerror=1>', NOW);
  assert.doesNotMatch(html, /<img/);
  assert.match(html, /&lt;img/);
});

test("freshnessBadge omits the label separator when no label is given", () => {
  const html = api.freshnessBadge(iso(30 * 1000), "live", "", NOW);
  assert.match(html, />updated just now<\/span>/);
  assert.doesNotMatch(html, /·/);
});
