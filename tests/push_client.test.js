// app gap 1: the client-side Web Push helpers in index.html's main <script> -- the base64->
// Uint8Array VAPID-key decode, whether push is considered configured, the [push-subscribe]
// issue URL, and the 3-state bell (off / on / push-linked).

const { test } = require("node:test");
const assert = require("node:assert");
const { extractHtmlFn, extractHtmlConst } = require("./_extract_html_fn");

const FN_NAMES = ["urlBase64ToUint8Array", "pushConfigured", "loadPushSub", "pushSubscribeIssueUrl", "renderNotifBell", "notifPrefs", "setNotifPrefs"];

function loadHarness({ vapidKey = "" } = {}) {
  const store = new Map();
  const localStorage = {
    getItem: (k) => (store.has(k) ? store.get(k) : null),
    setItem: (k, v) => store.set(k, String(v)),
    removeItem: (k) => store.delete(k),
  };
  let bell = { innerHTML: "", classes: new Set(), attrs: {}, title: "" };
  const document = {
    getElementById: (id) => (id === "notif-bell" ? {
      set innerHTML(v) { bell.innerHTML = v; }, get innerHTML() { return bell.innerHTML; },
      setAttribute: (k, v) => { bell.attrs[k] = v; },
      classList: { toggle: (c, on) => { on ? bell.classes.add(c) : bell.classes.delete(c); } },
      set title(v) { bell.title = v; }, get title() { return bell.title; },
    } : null),
  };
  const src =
    `const VAPID_PUBLIC_KEY = ${JSON.stringify(vapidKey)};\n` +
    `const PUSH_SUB_KEY = "fq_push_sub";\nconst NOTIF_KEY = "fq_notifications_v1";\n` +
    `const REPO_SLUG = "AhmadRehman1/FPL-Analyser";\n` +
    `const BELL_ON = "on"; const BELL_OFF = "off";\n` +
    FN_NAMES.map(extractHtmlFn).join("\n\n");
  const sandbox = {
    localStorage, document, atob: (s) => Buffer.from(s, "base64").toString("binary"),
    URLSearchParams,
  };
  const args = Object.keys(sandbox);
  const api = new Function(...args, src + "\nreturn { " + FN_NAMES.join(", ") + ", _bell: () => (" + "0,arguments" + ") };")(...args.map((k) => sandbox[k]));
  return { api, bell: () => bell };
}

test("urlBase64ToUint8Array decodes a URL-safe base64 VAPID key", () => {
  const { api } = loadHarness();
  const out = api.urlBase64ToUint8Array("aGVsbG8"); // "hello", unpadded, no url-safe chars
  assert.ok(out instanceof Uint8Array);
  assert.deepStrictEqual([...out], [...Buffer.from("hello")]);
  // url-safe chars round-trip
  const urlsafe = Buffer.from([0xfb, 0xff, 0xbf]).toString("base64").replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
  assert.deepStrictEqual([...api.urlBase64ToUint8Array(urlsafe)], [0xfb, 0xff, 0xbf]);
});

test("pushConfigured is false without a real key, true with one", () => {
  assert.strictEqual(loadHarness({ vapidKey: "" }).api.pushConfigured(), false);
  assert.strictEqual(loadHarness({ vapidKey: "short" }).api.pushConfigured(), false);
  assert.strictEqual(loadHarness({ vapidKey: "B" + "x".repeat(86) }).api.pushConfigured(), true);
});

test("pushSubscribeIssueUrl embeds the subscription as a json block for the workflow to parse", () => {
  const { api } = loadHarness();
  const sub = { endpoint: "https://fcm.example/abc", keys: { p256dh: "p", auth: "a" } };
  const url = api.pushSubscribeIssueUrl(sub);
  assert.ok(url.startsWith("https://github.com/AhmadRehman1/FPL-Analyser/issues/new?"));
  const body = new URL(url).searchParams.get("body");
  const m = body.match(/```json\s*([\s\S]*?)```/);
  assert.deepStrictEqual(JSON.parse(m[1].trim()), sub);
  assert.strictEqual(new URL(url).searchParams.get("title"), "[push-subscribe]");
});

test("renderNotifBell has three states: off, on (local), push-linked", () => {
  const h = loadHarness();
  h.api.setNotifPrefs({ enabled: false });
  h.api.renderNotifBell();
  assert.strictEqual(h.bell().innerHTML, "off");
  assert.strictEqual(h.bell().classes.has("is-push"), false);

  h.api.setNotifPrefs({ enabled: true });
  h.api.renderNotifBell();
  assert.strictEqual(h.bell().innerHTML, "on");
  assert.strictEqual(h.bell().classes.has("is-push"), false);
  assert.match(h.bell().title, /app open only/);

  h.api.setNotifPrefs({ enabled: true, pushRegistered: true });
  h.api.renderNotifBell();
  assert.strictEqual(h.bell().classes.has("is-push"), true);
  assert.match(h.bell().title, /Closed-app push/);
});
