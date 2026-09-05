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

// chipTimingCard() demotes its verdict once the GW4-19 (16-week) sweep is still "early" (under
// half swept) -- see index.html's own sweepEarly comment -- so a test exercising the confident
// "Hold your Wildcard"/"play it now" copy needs a team whose committed sweep is past that
// threshold. Picked dynamically (not a hardcoded team index) for the same reason the swept-best-
// gameweek comment above already gives: which team is further along drifts with every real
// chip_timing_latest.json refresh.
const FULL_SWEEP_WEEKS = 16;
function nonEarlyTeam() {
  const ranked = REAL.teams
    .map((t) => ({ t, n: (t.report.comparison.sweep_gameweeks || []).length }))
    .sort((a, b) => b.n - a.n);
  assert.ok(
    ranked[0].n >= FULL_SWEEP_WEEKS / 2,
    "expected at least one tracked team's committed sweep to be past the early-sweep threshold -- " +
      "if this fails, chip_timing_latest.json refreshed with both teams still early; adapt these " +
      "tests to doctor sweep_gameweeks synthetically instead of relying on real data",
  );
  return ranked[0].t;
}

test("real feed: card leads with the swept best Wildcard week and a hold recommendation", () => {
  const h = harness();
  h.state.chipTiming = REAL;
  const team = nonEarlyTeam();
  h.state.accountId = team.entry_id;
  const best = team.report.comparison.swept_best_gameweek;
  // Upcoming gameweek strictly BEFORE the swept best week, so the "hold" branch fires whatever
  // value the committed sweep currently carries -- swept_best_gameweek moves every re-run
  // (hardcoding "4" here broke this test on an unrelated chip_timing_latest.json refresh where
  // a partial sweep landed with best == 4).
  h.state.realSquad = { plan_for_gameweek: best - 1 };
  h.state.team = { gameweek: best - 2 };
  const card = h.api.chipTimingCard();
  assert.ok(card.includes(`Wildcard &middot; GW${best}`), "headline shows the swept best GW");
  assert.ok(card.includes("Hold your Wildcard"), "best week is in the future -> hold");
  assert.ok(card.includes("pts vs holding"));
  assert.ok(card.includes("See the full sweep"));
});

test("real feed: 'play it now' framing when the best week is the upcoming one", () => {
  const h = harness();
  const doctored = JSON.parse(JSON.stringify(REAL));
  const team = doctored.teams.find((t) => t.entry_id === nonEarlyTeam().entry_id);
  const best = team.report.comparison.swept_best_gameweek;
  h.state.chipTiming = doctored;
  h.state.accountId = team.entry_id;
  h.state.realSquad = { plan_for_gameweek: best }; // upcoming GW == the best week
  h.state.team = { gameweek: best - 1 };
  const card = h.api.chipTimingCard();
  assert.ok(!card.includes("Hold your Wildcard"));
  assert.ok(card.includes("playing it is live"));
});

test("real feed: partial-window caveat shows when a non-early sweep didn't cover GW4-19", () => {
  const h = harness();
  h.state.chipTiming = REAL;
  const team = nonEarlyTeam();
  h.state.accountId = team.entry_id;
  h.state.realSquad = { plan_for_gameweek: 4 };
  h.state.team = { gameweek: 3 };
  const sg = team.report.comparison.sweep_gameweeks || [];
  const partial = Math.min(...sg) > 4 || Math.max(...sg) < 19;
  assert.strictEqual(h.api.chipTimingCard().includes("Sweep so far covers"), partial);
});

test("real feed: an early sweep (<8 of 16 weeks) gets the honest 'so far' framing, not a confident verdict", () => {
  const h = harness();
  const early = REAL.teams
    .map((t) => ({ t, n: (t.report.comparison.sweep_gameweeks || []).length }))
    .sort((a, b) => a.n - b.n)[0];
  assert.ok(early.n > 0 && early.n < FULL_SWEEP_WEEKS / 2, "expected a real team with an early (<8/16) committed sweep -- adapt with doctored data if this now fails");
  h.state.chipTiming = REAL;
  h.state.accountId = early.t.entry_id;
  h.state.realSquad = { plan_for_gameweek: 4 };
  h.state.team = { gameweek: 3 };
  const card = h.api.chipTimingCard();
  assert.ok(card.includes("(so far)"), "verdict pill is qualified, not stated as confident fact");
  assert.ok(card.includes(`Only <b style="color:var(--ink);">${early.n} of ${FULL_SWEEP_WEEKS}</b> candidate weeks`));
  assert.ok(card.includes("too early to call this the best week yet"));
  assert.ok(!card.includes("Hold your Wildcard"), "early sweep must not assert a confident recommendation");
  assert.ok(!card.includes("Sweep so far covers"), "the sweepEarly paragraph replaces, not duplicates, the smaller partial-window caveat");
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
