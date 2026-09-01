// app gap 7: resolveDataBase() must keep the CANONICAL deployment on the raw URL (the add-team
// SLA depends on it) while letting a staging fork opt into same-origin via a meta tag or a
// window.FQ_DATA_BASE override. Regression guard for the core data-loading path.

const { test } = require("node:test");
const assert = require("node:assert");
const { extractHtmlFn } = require("./_extract_html_fn");

const RAW = "https://raw.githubusercontent.com/AhmadRehman1/FPL-Analyser/master";

function run({ hostname = "example.github.io", protocol = "https:", metaContent = null, winOverride }) {
  const src =
    `const RAW_FALLBACK = ${JSON.stringify(RAW)};\n` +
    `const isLocalDev = location.hostname === "localhost" || location.hostname === "127.0.0.1";\n` +
    extractHtmlFn("resolveDataBase");
  const sandbox = {
    location: { hostname, protocol },
    document: { querySelector: () => (metaContent === null ? null : { getAttribute: () => metaContent }) },
    window: {},
  };
  if (winOverride !== undefined) sandbox.window.FQ_DATA_BASE = winOverride;
  const fn = new Function(...Object.keys(sandbox), src + "\nreturn resolveDataBase();");
  return fn(...Object.values(sandbox));
}

test("canonical deployed host -> raw URL (unchanged production behaviour)", () => {
  assert.strictEqual(run({ hostname: "ahmadrehman1.github.io" }), RAW);
});

test("localhost -> same-origin (unchanged local-dev behaviour)", () => {
  assert.strictEqual(run({ hostname: "localhost" }), "");
  assert.strictEqual(run({ hostname: "127.0.0.1" }), "");
});

test("file:// -> raw URL", () => {
  assert.strictEqual(run({ hostname: "", protocol: "file:" }), RAW);
});

test("staging fork: <meta name=fq-data-base content=''> -> same-origin", () => {
  assert.strictEqual(run({ hostname: "my-fork.github.io", metaContent: "" }), "");
});

test("staging fork: meta with a sub-path -> that path, trailing slash trimmed", () => {
  assert.strictEqual(run({ hostname: "my-fork.github.io", metaContent: "/staging/" }), "/staging");
});

test("window.FQ_DATA_BASE overrides everything, incl. an empty string on a non-canonical host", () => {
  assert.strictEqual(run({ hostname: "somewhere.example", winOverride: "" }), "");
  assert.strictEqual(run({ hostname: "localhost", winOverride: "https://cdn.example/d" }), "https://cdn.example/d");
});
