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

// 2026-09-14 fix: chipTimingCard()'s own bestGw is `swept_best_gameweek ?? (the swept_table row
// with the highest delta_vs_hold)` -- swept_best_gameweek is null whenever the sweep concluded
// NO forced week actually beats holding (compare_wildcard_timing()'s own documented behavior,
// not a bug), which is exactly the real state both committed teams' sweeps reached once their
// window matured -- a state these tests never exercised before. Mirrors the card's own fallback
// exactly, rather than reading swept_best_gameweek directly, so these tests track whatever the
// card actually renders instead of breaking every time a real sweep concludes "hold is best."
function cardBestGameweek(comparison) {
  if (comparison.swept_best_gameweek != null) return comparison.swept_best_gameweek;
  const swept = (comparison.swept_table || []).slice().sort((a, b) => b.delta_vs_hold - a.delta_vs_hold)[0];
  return swept ? swept.gameweek : null;
}

test("real feed: card leads with the swept best Wildcard week and a hold recommendation", () => {
  const h = harness();
  h.state.chipTiming = REAL;
  const team = nonEarlyTeam();
  h.state.accountId = team.entry_id;
  const best = cardBestGameweek(team.report.comparison);
  // Upcoming gameweek strictly BEFORE the swept best week, so the "hold" branch fires whatever
  // value the committed sweep currently carries -- the best GW moves every re-run (hardcoding
  // "4" here broke this test once already on an unrelated chip_timing_latest.json refresh).
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
  const best = cardBestGameweek(team.report.comparison);
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

// 2026-09-14 fix: both real committed teams' sweeps have now matured past FULL_SWEEP_WEEKS/2
// (15/16 each, as of this fix), so no real team is left in an "early" state to exercise this
// copy against -- exactly the scenario this test's own prior comment anticipated ("adapt with
// doctored data if this now fails"). Doctors a real team's own comparison down to a genuinely
// early sweep (5 of the same real swept_table rows, sweep_gameweeks truncated to match) rather
// than inventing shapes from scratch, so everything the card reads besides sweep size/coverage
// stays real.
function doctorToEarlySweep(team, nSwept) {
  const doctored = JSON.parse(JSON.stringify(team));
  const c = doctored.report.comparison;
  c.sweep_gameweeks = (c.sweep_gameweeks || []).slice(0, nSwept);
  c.swept_table = (c.swept_table || []).filter((row) => c.sweep_gameweeks.includes(row.gameweek));
  // An early, thin sweep's own best-so-far is real signal, not a global conclusion -- clearing
  // swept_best_gameweek here matches compare_wildcard_timing()'s real behavior when the swept
  // arms haven't settled the question yet (it's None until enough of the window is covered).
  c.swept_best_gameweek = null;
  return doctored;
}

test("real feed: an early sweep (<8 of 16 weeks) gets the honest 'so far' framing, not a confident verdict", () => {
  const h = harness();
  const nSwept = 5;
  const early = doctorToEarlySweep(nonEarlyTeam(), nSwept);
  assert.ok(early.report.comparison.swept_table.length > 0, "the real team's swept_table must have rows within the first 5 sweep_gameweeks to doctor from");
  h.state.chipTiming = { teams: [early] };
  h.state.accountId = early.entry_id;
  h.state.realSquad = { plan_for_gameweek: 4 };
  h.state.team = { gameweek: 3 };
  const card = h.api.chipTimingCard();
  assert.ok(card.includes("(so far)"), "verdict pill is qualified, not stated as confident fact");
  assert.ok(card.includes(`Only <b style="color:var(--ink);">${nSwept} of ${FULL_SWEEP_WEEKS}</b> candidate weeks`));
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
