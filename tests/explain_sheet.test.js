// Gap 5: the "Explain this" sheet renderers live inside index.html's main <script> (they build
// HTML strings from feeds already in `state`, no DOM/network). This test extracts them and
// checks the rendering contract for the real feed shapes -- expected_points.explain_player_ep()
// / uncertainty.explain_player_risk() output attached to real_squad_<id>.json's `explain` block
// by run_transfer_planner_for_real_squad.py, plus decision_<id>_latest.json's sensitivity list.
//
// Same "DOM-free logic, tested separately from the UI" split as tests/planner/*.js.

const { test } = require("node:test");
const assert = require("node:assert");
const { extractHtmlFn: extractFn } = require("./_extract_html_fn");

const FN_NAMES = [
  "realSquadExplain", "epBreakdownBlock", "riskRangeBlock", "playerBreakdownSection",
  "whatWouldChangeBlock", "openExplainSheet", "explainTransfer", "explainCaptain",
];

function loadHarness() {
  const src = FN_NAMES.map(extractFn).join("\n\n");
  const state = {};
  let lastSheet = null;
  const sandbox = {
    state,
    escapeHtml: (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])),
    humanizeKey: (k) => k.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()),
    EP_CATEGORY_LABELS: {
      appearance: "Appearance (60+ mins)", goals: "Goals", assists: "Assists", clean_sheet: "Clean sheet",
      goals_conceded: "Goals conceded", defcon: "Defensive contribution", bonus: "Bonus", saves: "Saves",
      penalty_save: "Penalty saves", cards: "Cards", own_goal: "Own goals",
    },
    document: { getElementById: () => ({ set innerHTML(v) { lastSheet = v; }, get innerHTML() { return lastSheet; } }) },
    openSheet: () => {},
  };
  const args = Object.keys(sandbox);
  const fn = new Function(...args, src + "\nreturn { " + FN_NAMES.join(", ") + " };");
  const api = fn(...args.map((k) => sandbox[k]));
  return { api, state, getSheet: () => lastSheet };
}

const EP = (cats, total) => ({ player_uid: "p", fixture_match_id: "m", categories: cats, total, expected_bps: 20 });
const RISK = () => ({ player_uid: "p", fixture_match_id: "m", floor: 1.2, q25: 3.1, q75: 7.4, ceiling: 13.6, var_total: 12, skew: 0.5, excess_kurtosis: 0.3, caveat: "Cornish-Fisher approximation" });

test("epBreakdownBlock: biggest contributor first, near-zero categories dropped, total shown", () => {
  const { api } = loadHarness();
  const out = api.epBreakdownBlock(EP({ goals: 3.05, appearance: 0.92, clean_sheet: 0.01, bonus: 0.61, defcon: 0.0 }, 5.27));
  assert.ok(out.includes("Goals"));
  assert.ok(!out.includes("Clean sheet"), "0.01 clean sheet is below the 0.05 floor");
  assert.ok(!out.includes("Defensive contribution"), "exactly 0 dropped");
  assert.ok(out.indexOf("Goals") < out.indexOf("Bonus"), "sorted by magnitude");
  assert.ok(out.includes("5.27"), "expected total rendered");
});

test("epBreakdownBlock: empty/absent breakdown renders nothing", () => {
  const { api } = loadHarness();
  assert.strictEqual(api.epBreakdownBlock(null), "");
  assert.strictEqual(api.epBreakdownBlock({ categories: {} }), "");
});

test("riskRangeBlock: floor/ceiling + middle-half from explain_player_risk output", () => {
  const { api } = loadHarness();
  const out = api.riskRangeBlock(RISK());
  assert.ok(out.includes("1.2") && out.includes("13.6"));
  assert.strictEqual(api.riskRangeBlock(null), "");
  assert.strictEqual(api.riskRangeBlock({ floor: null }), "");
});

test("explainCaptain: renders the recommended captain's EP split + risk range from real_squad.explain", () => {
  const h = loadHarness();
  h.state.realSquad = {
    explain: {
      captain_breakdown: {
        gameweek: 3,
        recommended: { player_uid: "player_erling_haaland", name: "Erling Haaland", ep: EP({ goals: 3.0, appearance: 0.9, bonus: 0.6 }, 5.1), risk: RISK() },
        current: null,
      },
    },
  };
  h.state.decision = null;
  h.api.explainCaptain({ recommended_name: "Erling Haaland", recommended_uid: "player_erling_haaland", recommended_expected_points: 5.1, current_name: "Erling Haaland", matches_current: true, potential_gain: 0 });
  const sheet = h.getSheet();
  assert.ok(sheet.includes("Keep Erling Haaland"), "title");
  assert.ok(sheet.includes("Goals"), "EP category rendered");
  assert.ok(sheet.includes("13.6"), "risk ceiling rendered");
  assert.ok(sheet.includes("armband already matches"));
});

test("explainCaptain: a switch shows both the recommended and current captain breakdowns", () => {
  const h = loadHarness();
  h.state.realSquad = {
    explain: {
      captain_breakdown: {
        recommended: { name: "Mohamed Salah", ep: EP({ goals: 2.2, assists: 1.1 }, 4.4), risk: RISK() },
        current: { name: "Erling Haaland", ep: EP({ goals: 1.9, appearance: 0.9 }, 3.6), risk: RISK() },
      },
    },
  };
  h.api.explainCaptain({ recommended_name: "Mohamed Salah", current_name: "Erling Haaland", recommended_expected_points: 4.4, current_expected_points: 3.6, matches_current: false, potential_gain: 0.8 });
  const sheet = h.getSheet();
  assert.ok(sheet.includes("Captain Mohamed Salah"));
  assert.ok(sheet.includes("Erling Haaland") && sheet.includes("current"));
  assert.ok(sheet.includes("Assists"));
});

test("explainTransfer: matches the breakdown by canonical name and shows in + out + the swap reason", () => {
  const h = loadHarness();
  h.state.realSquad = {
    explain: {
      transfer_breakdowns: [{
        rank: 1, gameweek: 3,
        player_in: { name: "Bernd Leno", ep: EP({ clean_sheet: 1.3, saves: 1.1, appearance: 1.0 }, 3.5), risk: RISK() },
        player_out: { name: "Antonín Kinsky", ep: EP({ appearance: 0.2, clean_sheet: 0.2 }, 0.8), risk: RISK() },
      }],
    },
  };
  h.state.decision = {
    swaps: [{ in_name: "Bernd Leno", out_name: "Antonín Kinsky", reason: "Kinsky has started 2 of a possible 3 and lost the gloves to Vicario" }],
    downside_ci: [42.1, 88.3],
    sensitivity: [{ if_condition_display: "Kinsky starts GW3", then_action_display: "hold", delta_ep: -1.2 }],
  };
  h.api.explainTransfer("Bernd Leno", "Antonín Kinsky");
  const sheet = h.getSheet();
  assert.ok(sheet.includes("lost the gloves"), "swap reason from decision.swaps[0]");
  assert.ok(sheet.includes("Clean sheet"), "incoming player's EP split");
  assert.ok(sheet.includes("42.1") && sheet.includes("88.3"), "squad downside from decision.downside_ci");
  assert.ok(sheet.includes("Kinsky starts GW3"), "what-would-change-my-mind from decision.sensitivity");
});

test("explainTransfer: no matching breakdown still renders the reason + downside, never an empty sheet", () => {
  const h = loadHarness();
  h.state.realSquad = { explain: { transfer_breakdowns: [] } };
  h.state.decision = { swaps: [{ in_name: "X", out_name: "Y", reason: "form" }], downside_ci: [1, 2], sensitivity: [] };
  h.api.explainTransfer("X", "Y");
  const sheet = h.getSheet();
  assert.ok(sheet.includes("form"));
  assert.ok(sheet.includes("Y") && sheet.includes("X"));
  assert.ok(!sheet.includes("isn't in the latest run yet"), "reason present, so not the empty state");
});

test("whatWouldChangeBlock: empty without a decision sensitivity list", () => {
  const h = loadHarness();
  h.state.decision = null;
  assert.strictEqual(h.api.whatWouldChangeBlock(), "");
  h.state.decision = { sensitivity: [] };
  assert.strictEqual(h.api.whatWouldChangeBlock(), "");
});
