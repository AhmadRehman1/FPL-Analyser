// The forward-plan card + sheet are DOM-free renderers inside index.html's main <script>.
// They read data/forward_plan/forward_plan_latest.json (state.forwardPlan) -- one model_choice
// planner walk per tracked squad -- and show the model's whole plan to GW18: per-week action,
// captain, projected points, and the Wildcard squad. chipPreviewSquadBlock (its own tested
// renderer) is stubbed here so this focuses on the plan-specific reshaping.

const { test } = require("node:test");
const assert = require("node:assert");
const { extractHtmlFn, extractHtmlConst } = require("./_extract_html_fn");

function harness() {
  const src = [
    extractHtmlConst("FP_CHIP_LABEL"),
    extractHtmlFn("forwardPlanForEntity"),
    extractHtmlFn("forwardPlanChipLine"),
    extractHtmlFn("forwardPlanCard"),
    extractHtmlFn("forwardPlanWeekRow"),
    extractHtmlFn("openForwardPlanSheet"),
  ].join("\n\n");
  const state = {};
  let sheet = null;
  const sandbox = {
    state,
    escapeHtml: (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])),
    chipPreviewSquadBlock: (type, squad) => `<!--squad:${type}:${(squad || []).length}-->`,
    openSheet: () => {},
    document: { getElementById: () => ({ set innerHTML(v) { sheet = v; }, get innerHTML() { return sheet; } }) },
  };
  const keys = Object.keys(sandbox);
  const fn = new Function(...keys, src + "\nreturn { forwardPlanForEntity, forwardPlanChipLine, forwardPlanCard, forwardPlanWeekRow, openForwardPlanSheet };");
  return { api: fn(...keys.map((k) => sandbox[k])), state, getSheet: () => sheet };
}

function squad15() {
  return Array.from({ length: 15 }, (_, i) => ({
    player_name: `P${i}`, club: "Arsenal", in_xi: i < 11, is_captain: i === 0, is_vice: i === 1,
  }));
}

function planFixture(overrides = {}) {
  return {
    entities: {
      model_team: {
        entity_key: "model_team", label: "FPL Quant Model Team", entry_id: null,
        base_gameweek: 2, start_gameweek: 3, end_gameweek: 18,
        total_projected_points: 812.4, band: [540, 1084],
        chips_planned: [{ chip: "wildcard", gameweek: 10 }],
        wildcard: { gameweek: 10, projected_gain: 24.1, projected_points: 61.2, captain: "P0", squad: squad15() },
        wildcard_held_until: null, free_hit: null,
        weeks: [
          { gameweek: 3, action: "hold", summary: "Hold — no transfer", transfers: [], chip: null, captain: "P0", projected_points: 51.1, band: [37, 66], wildcard_gain: -5.1, wildcard_recommended: false, squad: squad15() },
          { gameweek: 4, action: "transfer", summary: "P3 → P9", transfers: [{ out: "P3", in: "P9", net: 3.2 }], chip: null, captain: "P0", projected_points: 53.4, band: [39, 68], wildcard_gain: -2.0, wildcard_recommended: false, squad: squad15() },
          { gameweek: 10, action: "wildcard", summary: "Wildcard — full rebuild", transfers: [], chip: "wildcard", captain: "P0", projected_points: 61.2, band: [44, 79], wildcard_gain: 24.1, wildcard_recommended: true, squad: squad15() },
        ],
        ...overrides,
      },
    },
    ...(overrides.__root || {}),
  };
}

test("forwardPlanForEntity returns null when the feed is absent or the key is unknown", () => {
  const h = harness();
  h.state.forwardPlan = null;
  assert.strictEqual(h.api.forwardPlanForEntity("model_team"), null);
  h.state.forwardPlan = planFixture();
  assert.strictEqual(h.api.forwardPlanForEntity("nope"), null);
  assert.ok(h.api.forwardPlanForEntity("model_team"));
});

test("forwardPlanChipLine lists planned chips, and a held Wildcard when it's never played", () => {
  const h = harness();
  const plan = planFixture().entities.model_team;
  assert.strictEqual(h.api.forwardPlanChipLine(plan), "Wildcard GW10");
  const held = { chips_planned: [], wildcard: null, wildcard_held_until: { gameweek: 12, projected_gain: 8.0 } };
  assert.strictEqual(h.api.forwardPlanChipLine(held), "holding Wildcard for GW12");
});

test("forwardPlanCard summarises move count + projected points and links to the sheet", () => {
  const h = harness();
  h.state.forwardPlan = planFixture();
  const card = h.api.forwardPlanCard("model_team");
  assert.ok(card.includes("The plan to GW18"));
  assert.ok(card.includes("2 moves"), "3 weeks, 1 hold + 1 transfer + 1 wildcard = 2 non-hold");
  assert.ok(card.includes("812 projected points") || card.includes("~812"));
  assert.ok(card.includes("location.hash='forward-plan-model_team'"));
});

test("forwardPlanCard is empty when there is no plan for that entity", () => {
  const h = harness();
  h.state.forwardPlan = planFixture();
  assert.strictEqual(h.api.forwardPlanCard("1305242"), "");
});

test("openForwardPlanSheet renders the wildcard squad, the chip week, and every gameweek row", () => {
  const h = harness();
  h.state.forwardPlan = planFixture();
  h.api.openForwardPlanSheet("model_team");
  const s = h.getSheet();
  assert.ok(s.includes("Wildcard squad &middot; GW10"));
  assert.ok(s.includes("<!--squad:wildcard:15-->"), "wildcard pitch rendered via chipPreviewSquadBlock");
  assert.ok(s.includes("GW3") && s.includes("GW4") && s.includes("GW10"));
  assert.ok(s.includes("P3 → P9"), "transfer summary shown");
  assert.ok(s.includes("Early-season caveat"));
});

test("openForwardPlanSheet shows a held-Wildcard note when the model never plays it in-window", () => {
  const h = harness();
  const plan = planFixture();
  plan.entities.model_team.wildcard = null;
  plan.entities.model_team.chips_planned = [];
  plan.entities.model_team.wildcard_held_until = { gameweek: 20, projected_gain: 12.5 };
  plan.entities.model_team.weeks = plan.entities.model_team.weeks.filter((w) => w.action !== "wildcard");
  h.state.forwardPlan = plan;
  h.api.openForwardPlanSheet("model_team");
  const s = h.getSheet();
  assert.ok(s.includes("holds its Wildcard through this window"));
  assert.ok(s.includes("GW20"));
});

test("openForwardPlanSheet is graceful when the plan is missing", () => {
  const h = harness();
  h.state.forwardPlan = null;
  h.api.openForwardPlanSheet("model_team");
  assert.ok(h.getSheet().includes("hasn't produced a forward plan"));
});
