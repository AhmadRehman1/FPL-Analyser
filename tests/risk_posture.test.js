// app gap 6: the Plan-tab risk-posture toggle helpers live in index.html's main <script>
// (localStorage + which plan the rec surfaces read). Extract and test: per-account
// persistence, the fallback for a stale value, attackPostureActive() gating on the variant
// plan actually being loaded, activeRealSquad() selection, and riskPostureCard() rendering.

const { test, beforeEach } = require("node:test");
const assert = require("node:assert");
const { extractHtmlFn: extractFn, extractHtmlConst: extractConst } = require("./_extract_html_fn");

const FN_NAMES = ["riskPosture", "setRiskPosture", "activePosture", "attackPostureActive", "activeRealSquad", "riskPostureCard"];

function loadHarness() {
  const store = new Map();
  const localStorage = {
    getItem: (k) => (store.has(k) ? store.get(k) : null),
    setItem: (k, v) => store.set(k, String(v)),
    removeItem: (k) => store.delete(k),
  };
  const consts = [extractConst("RISK_POSTURES"), extractConst("POSTURE_VARIANT")].join("\n") + '\nconst RISK_POSTURE_KEY = "fq_risk_posture";\nconst CHIP_LABELS = { wildcard: "Wildcard", free_hit: "Free Hit", triple_captain: "Triple Captain", bench_boost: "Bench Boost" };\n';
  const src = consts + FN_NAMES.map(extractFn).join("\n\n");
  const state = { accountId: "7139944" };
  const sandbox = {
    state, localStorage,
    hapticTick: () => {}, renderCurrentTab: () => {},
    escapeHtml: (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])),
  };
  const args = Object.keys(sandbox);
  const fn = new Function(...args, src + "\nreturn { " + FN_NAMES.join(", ") + " };");
  return { api: fn(...args.map((k) => sandbox[k])), state };
}

let H;
beforeEach(() => { H = loadHarness(); });

test("default posture is balanced; set + read round-trips per account", () => {
  assert.strictEqual(H.api.riskPosture(), "balanced");
  H.api.setRiskPosture("attack");
  assert.strictEqual(H.api.riskPosture(), "attack");
  H.state.accountId = "1305242";
  assert.strictEqual(H.api.riskPosture(), "balanced", "other account unaffected");
  H.api.setRiskPosture("attack");
  H.state.accountId = "7139944";
  assert.strictEqual(H.api.riskPosture(), "attack");
});

test("a stale / unknown stored posture falls back to balanced", () => {
  H.state.__ls || 0;
  H.api.setRiskPosture("attack");
  // simulate a future value this build doesn't know
  H.api.setRiskPosture("protect"); // rejected by setRiskPosture (unknown)
  assert.strictEqual(H.api.riskPosture(), "attack");
});

test("attackPostureActive() is true only when attack is picked AND the variant plan loaded", () => {
  H.api.setRiskPosture("attack");
  assert.strictEqual(H.api.attackPostureActive(), false, "no variant loaded yet");
  H.state.realSquadAttack = { captain_recommendation: { recommended_name: "Salah" }, chip_evaluations: [], transfer_recommendations: [] };
  assert.strictEqual(H.api.attackPostureActive(), true);
  H.api.setRiskPosture("balanced");
  assert.strictEqual(H.api.attackPostureActive(), false);
});

test("activeRealSquad() returns the attack plan when active, else the balanced plan", () => {
  H.state.realSquad = { tag: "balanced" };
  H.state.realSquadAttack = { tag: "attack" };
  assert.strictEqual(H.api.activeRealSquad().tag, "balanced");
  H.api.setRiskPosture("attack");
  assert.strictEqual(H.api.activeRealSquad().tag, "attack");
});

test("ML shadow lane: activePosture / activeRealSquad select real_squad_<id>_ml when published", () => {
  H.state.realSquad = { tag: "balanced" };
  H.state.realSquadMl = { tag: "ml" };
  H.api.setRiskPosture("ml");
  assert.strictEqual(H.api.activePosture(), "ml");
  assert.strictEqual(H.api.activeRealSquad().tag, "ml");
  assert.strictEqual(H.api.attackPostureActive(), false, "ml is not attack");
  delete H.state.realSquadMl;
  assert.strictEqual(H.api.activePosture(), "balanced", "falls back when the ml plan isn't loaded");
});

test("riskPostureCard renders all three segments; a variant is disabled until published", () => {
  const card = H.api.riskPostureCard();
  assert.ok(card.includes(">Balanced<") && card.includes(">Attack rank<") && card.includes(">ML shadow<"));
  assert.ok(/Attack rank<\/button>/.test(card));
  assert.ok(card.includes("disabled"), "Attack disabled with no variant loaded");

  H.state.realSquad = { captain_recommendation: { recommended_name: "Haaland" } };
  H.state.realSquadAttack = {
    captain_recommendation: { recommended_name: "Salah" },
    chip_evaluations: [{ chip_type: "wildcard", recommended: true }],
    transfer_recommendations: [{ player_out: "A", player_in: "B" }],
  };
  H.api.setRiskPosture("attack");
  const card2 = H.api.riskPostureCard();
  // the Attack button is now enabled (ML shadow, still unpublished, stays disabled)
  assert.ok(!/setRiskPosture\('attack'\)" disabled/.test(card2), "Attack enabled once the variant is loaded");
  assert.ok(/setRiskPosture\('ml'\)" disabled/.test(card2), "ML shadow still disabled");
  assert.ok(card2.includes("Salah") && card2.includes("Haaland"), "shows attack captain vs balanced captain");
  assert.ok(card2.includes("Wildcard"));
});
