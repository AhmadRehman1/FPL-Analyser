// chipTimingLine() — the sweep-verdict line inside the chip detail sheet. A DOM-free renderer
// that turns a chip_evaluations[].timing block (written by
// run_transfer_planner_for_real_squad.reconcile_chips_with_timing_sweep) into one sentence.

const { test } = require("node:test");
const assert = require("node:assert");
const { extractHtmlFn } = require("./_extract_html_fn");

function load() {
  const src = extractHtmlFn("chipTimingLine");
  const sandbox = {
    escapeHtml: (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])),
  };
  const keys = Object.keys(sandbox);
  return new Function(...keys, src + "\nreturn chipTimingLine;")(...keys.map((k) => sandbox[k]));
}

const chipTimingLine = load();

test("no timing block -> empty string", () => {
  assert.strictEqual(chipTimingLine({ chip_type: "wildcard" }), "");
  assert.strictEqual(chipTimingLine({ chip_type: "wildcard", timing: { available: false } }), "");
});

test("best week is now -> a clear go signal", () => {
  const out = chipTimingLine({ chip_type: "wildcard", timing: { available: true, is_best_week_now: true } });
  assert.match(out, /best week to play it/);
});

test("not the best week -> surfaces the hold detail", () => {
  const out = chipTimingLine({
    chip_type: "wildcard",
    detail_timing: "Hold -- the chip-timing sweep's best wildcard week is GW12 (+18 projected pts vs playing now).",
    timing: { available: true, is_best_week_now: false, best_gameweek: 12 },
  });
  assert.match(out, /GW12/);
  assert.match(out, /Hold/);
});

test("falls back to best_gameweek when there is no detail string", () => {
  const out = chipTimingLine({ chip_type: "bench_boost", timing: { available: true, is_best_week_now: false, best_gameweek: 30 } });
  assert.match(out, /GW30/);
});

test("empty-window note is shown when there is no best week at all", () => {
  const out = chipTimingLine({
    chip_type: "bench_boost",
    timing: { available: true, is_best_week_now: false, note: "no positive bench-boost week in the sweep horizon" },
  });
  assert.match(out, /no positive bench-boost week/);
});
