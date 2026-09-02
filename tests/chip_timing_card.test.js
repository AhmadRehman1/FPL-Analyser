// The "Chip timing" card + its detail sheet are DOM-free renderers inside index.html's main
// <script>. They read data/chip_timing/chip_timing_latest.json (state.chipTiming) -- the full
// forced-Wildcard MIQP sweep -- and turn it into a "hold your Wildcard, best week is GWn" card,
// unlike movesThisWeekCard()'s directive which always says "play it this week". Tested against
// the real committed sweep file so a shape drift is caught.

const { test } = require("node:test");
const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");
const { extractHtmlFn } = require("./_extract_html_fn");

const REAL = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "data", "chip_timing", "chip_timing_latest.json"), "utf8"));

function harness() {
  const src = ["chipTimingReportForAccount", "chipTimingCard", "openChipTimingSheet"].map(extractHtmlFn).join("\n\n");
  const state = {};
  let sheet = null;
  const sandbox = {
    state,
    escapeHtml: (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])),
    provenanceLine: () => "<!--prov-->",
    openSheet: () => {},
    document: { getElementById: () => ({ set innerHTML(v) { sheet = v; }, get innerHTML() { return sheet; } }) },
  };
  const keys = Object.keys(sandbox);
  const fn = new Function(...keys, src + "\nreturn { chipTimingCard, openChipTimingSheet, chipTimingReportForAccount };");
  return { api: fn(...keys.map((k) => sandbox[k])), state, getSheet: () => sheet };
}

test("no card when the sweep feed is absent", () => {
  const h = harness();
  h.state.chipTiming = null;
  assert.strictEqual(h.api.chipTimingCard(), "");
});

test("real feed: card leads with the swept best Wildcard week and a hold recommendation", () => {
  const h = harness();
  h.state.chipTiming = REAL;
  h.state.accountId = REAL.teams[0].entry_id;
  h.state.realSquad = { plan_for_gameweek: 4 };
  h.state.team = { gameweek: 3 };
  const card = h.api.chipTimingCard();
  const best = REAL.teams[0].report.comparison.swept_best_gameweek;
  assert.ok(card.includes(`Wildcard &middot; GW${best}`), "headline shows the swept best GW");
  assert.ok(card.includes("Hold your Wildcard"), "best week is in the future -> hold");
  assert.ok(card.includes("pts vs holding"));
  assert.ok(card.includes("See the full sweep"));
});

test("real feed: 'play it now' framing when the best week is the upcoming one", () => {
  const h = harness();
  const doctored = JSON.parse(JSON.stringify(REAL));
  const best = doctored.teams[0].report.comparison.swept_best_gameweek;
  h.state.chipTiming = doctored;
  h.state.accountId = doctored.teams[0].entry_id;
  h.state.realSquad = { plan_for_gameweek: best }; // upcoming GW == the best week
  h.state.team = { gameweek: best - 1 };
  const card = h.api.chipTimingCard();
  assert.ok(!card.includes("Hold your Wildcard"));
  assert.ok(card.includes("playing it is live"));
});

test("real feed: partial-window caveat shows when the sweep didn't cover GW4-19", () => {
  const h = harness();
  h.state.chipTiming = REAL;
  h.state.accountId = REAL.teams[0].entry_id;
  h.state.realSquad = { plan_for_gameweek: 4 };
  h.state.team = { gameweek: 3 };
  const sg = REAL.teams[0].report.comparison.sweep_gameweeks || [];
  const partial = Math.min(...sg) > 4 || Math.max(...sg) < 19;
  assert.strictEqual(h.api.chipTimingCard().includes("Sweep so far covers"), partial);
});

test("real feed: the detail sheet renders the swept table, trajectory and free-hit scan", () => {
  const h = harness();
  h.state.chipTiming = REAL;
  h.state.accountId = REAL.teams[1].entry_id;
  h.state.realSquad = { plan_for_gameweek: 4 };
  h.state.team = { gameweek: 3 };
  h.api.openChipTimingSheet();
  const s = h.getSheet();
  const r = REAL.teams[1].report;
  assert.ok(s.includes("Forced-Wildcard sweep"));
  assert.ok(s.includes(`GW${r.comparison.swept_table[0].gameweek}`));
  assert.ok(s.includes("Free Hit scan"));
  assert.ok(s.includes(escapeSafe(r.entry_label)));
});

function escapeSafe(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
