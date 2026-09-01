// app gap 4: the local decision log + its in-app self-audit view live inside index.html's main
// <script> (localStorage only, no DOM/network). Extract and test: storage round-trips per
// account+gameweek+kind, explicit vs soft (auto-fill) writes, the three-way control's selected
// state, and the Profile-sheet summary rendering. Same DOM-free split as tests/planner/*.js.

const { test, beforeEach } = require("node:test");
const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");

const html = fs.readFileSync(path.join(__dirname, "..", "index.html"), "utf8");

function extractFn(name) {
  const start = html.search(new RegExp("function\\s+" + name + "\\s*\\("));
  if (start < 0) throw new Error("function not found: " + name);
  let i = html.indexOf("{", start);
  const stack = ["code"];
  let depth = 0;
  for (; i < html.length; i++) {
    const c = html[i], p = html[i - 1], mode = stack[stack.length - 1];
    if (mode === "'" || mode === '"') { if (c === mode && p !== "\\") stack.pop(); continue; }
    if (mode === "`") {
      if (c === "`" && p !== "\\") stack.pop();
      else if (c === "$" && html[i + 1] === "{" && p !== "\\") { stack.push("code"); i++; }
      continue;
    }
    if (c === "/" && html[i + 1] === "/") { const nl = html.indexOf("\n", i); i = nl < 0 ? html.length : nl; continue; }
    if (c === "'" || c === '"' || c === "`") { stack.push(c); continue; }
    if (c === "{") depth++;
    else if (c === "}") { if (stack.length > 1) { stack.pop(); continue; } depth--; if (depth === 0) return html.slice(start, i + 1); }
  }
  throw new Error("unbalanced braces: " + name);
}

const FN_NAMES = [
  "loadDecisionLogAll", "loadDecisionLog", "decisionChoiceFor", "logDecision",
  "decisionLogControl", "decisionLogBlock",
];

// Pull the two module-level consts the functions close over out of the file verbatim.
function extractConst(name) {
  const m = html.match(new RegExp("const " + name + "\\s*=\\s*[^;]+;"));
  if (!m) throw new Error("const not found: " + name);
  return m[0];
}

function loadHarness() {
  const store = new Map();
  const localStorage = {
    getItem: (k) => (store.has(k) ? store.get(k) : null),
    setItem: (k, v) => store.set(k, String(v)),
    removeItem: (k) => store.delete(k),
  };
  const consts = ["DECISION_LOG_KEY", "DECISION_CHOICES", "DECISION_CHOICE_LABELS"].map(extractConst).join("\n");
  const src = consts + "\n" + FN_NAMES.map(extractFn).join("\n\n");
  const state = { accountId: "7139944" };
  const sandbox = {
    state, localStorage,
    hapticTick: () => {},
    renderCurrentTab: () => {},
    escapeHtml: (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])),
  };
  const args = Object.keys(sandbox);
  const fn = new Function(...args, src + "\nreturn { " + FN_NAMES.join(", ") + " };");
  return { api: fn(...args.map((k) => sandbox[k])), state, store };
}

let H;
beforeEach(() => { H = loadHarness(); });

test("logDecision round-trips per account + gameweek + kind", () => {
  H.api.logDecision("captain", "followed", 3, "Keep Haaland (C)");
  H.api.logDecision("transfer", "modified", 3, "my own pick");
  H.api.logDecision("chip", "skipped", 4, "Play Wildcard");
  const log = H.api.loadDecisionLog("7139944");
  assert.strictEqual(log[3].captain.choice, "followed");
  assert.strictEqual(log[3].captain.rec, "Keep Haaland (C)");
  assert.strictEqual(log[3].transfer.choice, "modified");
  assert.strictEqual(log[4].chip.choice, "skipped");
  assert.ok(typeof log[3].captain.ts === "number");
});

test("logs are namespaced by account", () => {
  H.api.logDecision("captain", "followed", 3, "x");
  H.state.accountId = "1305242";
  H.api.logDecision("captain", "skipped", 3, "y");
  assert.strictEqual(H.api.loadDecisionLog("7139944")[3].captain.choice, "followed");
  assert.strictEqual(H.api.loadDecisionLog("1305242")[3].captain.choice, "skipped");
});

test("an explicit choice always wins; a soft (explicit=false) write never overwrites it", () => {
  H.api.logDecision("transfer", "skipped", 3, "explicit", true);
  H.api.logDecision("transfer", "followed", 3, "soft auto-fill", false);
  assert.strictEqual(H.api.decisionChoiceFor("transfer", 3), "skipped");
});

test("a soft write DOES land when nothing is logged for that slot yet", () => {
  H.api.logDecision("transfer", "followed", 3, "matched the rec", false);
  assert.strictEqual(H.api.decisionChoiceFor("transfer", 3), "followed");
});

test("logDecision rejects an unknown choice and a null gameweek", () => {
  H.api.logDecision("captain", "banana", 3, "x");
  H.api.logDecision("captain", "followed", null, "x");
  assert.deepStrictEqual(H.api.loadDecisionLogAll(), {});
});

test("decisionLogControl marks the stored choice active and carries the gameweek + kind into the onclick", () => {
  H.api.logDecision("captain", "modified", 5, "Switch to Salah");
  const c = H.api.decisionLogControl("captain", 5, "Switch to Salah");
  assert.ok(/dlog-btn active[^>]*>Did my own thing/.test(c) || c.includes('class="dlog-btn active"'));
  assert.ok(c.includes("logDecision('captain','modified',5"));
  assert.ok(c.includes("logDecision('captain','skipped',5"));
  assert.strictEqual(H.api.decisionLogControl("captain", null, "x"), "", "no control without a gameweek");
});

test("decisionLogBlock: empty state prompts the user to start logging", () => {
  const out = H.api.decisionLogBlock();
  assert.ok(out.includes("Nothing logged yet"));
  assert.ok(out.includes("Your decisions"));
});

test("decisionLogBlock: summarises follow rate, own-call count, captain overrides and a per-GW list", () => {
  H.api.logDecision("captain", "followed", 3, "keep");
  H.api.logDecision("transfer", "followed", 3, "a->b");
  H.api.logDecision("captain", "modified", 4, "switch");
  H.api.logDecision("chip", "skipped", 4, "wc");
  const out = H.api.decisionLogBlock();
  assert.ok(out.includes(">2<"), "2 followed tiles"); // followed count
  assert.ok(out.includes("50%"), "2 of 4 followed");
  assert.ok(/own captain 1 time/.test(out), "captain override surfaced");
  assert.ok(out.includes("GW3") && out.includes("GW4"), "per-gameweek rows");
});
