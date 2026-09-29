// Pure logic behind the players table, player sheet and fixture ticker, extracted from
// index.html: typo-tolerant search, multi-column sort (missing values last), same-position
// percentiles, the 1-5 FDR scale, and the ticker's run scoring.

const { test } = require("node:test");
const assert = require("node:assert");
const { extractHtmlFn, extractHtmlConst } = require("./_extract_html_fn");

const FN_NAMES = ["foldText", "withinOneEdit", "fuzzyMatch", "compareBySortKeys", "percentileRank", "fdrLevel", "fdrTickerRows"];

function load() {
  const src =
    extractHtmlConst("TICKER_BLANK_SCORE") + "\n" +
    FN_NAMES.map(extractHtmlFn).join("\n\n");
  const factory = new Function(src + "\nreturn { " + FN_NAMES.join(", ") + " };");
  return factory();
}
const api = load();

test("foldText strips accents and special letters", () => {
  assert.strictEqual(api.foldText("Ødegaard"), "odegaard");
  assert.strictEqual(api.foldText("Mbappé  MÜLLER"), "mbappe  muller");
  assert.strictEqual(api.foldText(null), "");
});

test("withinOneEdit allows one insert, delete or substitution", () => {
  assert.ok(api.withinOneEdit("haland", "haaland"));
  assert.ok(api.withinOneEdit("salha", "salhb"));
  assert.ok(api.withinOneEdit("saka", "saka"));
  assert.ok(!api.withinOneEdit("haland", "haalands"));
  assert.ok(!api.withinOneEdit("palmer", "plamre"));
});

test("fuzzyMatch: every token must hit, typos allowed on 4+ letters", () => {
  const hay = "B.Fernandes Bruno Borges Fernandes MUN Man Utd MID";
  assert.ok(api.fuzzyMatch("", hay));
  assert.ok(api.fuzzyMatch("fernandes", hay));
  assert.ok(api.fuzzyMatch("bruno mun", hay));
  assert.ok(api.fuzzyMatch("fernandez", hay), "one typo");
  assert.ok(api.fuzzyMatch("man utd mid", hay));
  assert.ok(!api.fuzzyMatch("bruno liv", hay), "every token must match");
  assert.ok(!api.fuzzyMatch("mun", "Haaland MCI"), "short tokens need a real substring");
  assert.ok(api.fuzzyMatch("odegard", "Ødegaard Martin Ødegaard ARS"));
});

test("compareBySortKeys: direction, tie-breaks and missing values last", () => {
  const cols = {
    price: { get: (p) => p.price },
    x5: { get: (p, m) => m.x5 },
    name: { get: (p) => p.name },
  };
  const metrics = { a: { x5: 10 }, b: { x5: 20 }, c: { x5: null }, d: { x5: 20 } };
  const rows = [
    { id: "a", price: 5, name: "Zed" },
    { id: "b", price: 5, name: "Amy" },
    { id: "c", price: 9, name: "Bob" },
    { id: "d", price: 4, name: "Cat" },
  ];
  const sortBy = (keys) => rows.slice().sort((x, y) => api.compareBySortKeys(x, y, keys, cols, (p) => metrics[p.id])).map((p) => p.id);
  assert.deepStrictEqual(sortBy([{ key: "price", dir: "desc" }]), ["c", "a", "b", "d"]);
  assert.deepStrictEqual(sortBy([{ key: "price", dir: "asc" }, { key: "x5", dir: "desc" }]), ["d", "b", "a", "c"]);
  assert.deepStrictEqual(sortBy([{ key: "x5", dir: "desc" }, { key: "name", dir: "asc" }]), ["b", "d", "a", "c"]);
  // null x5 stays last whichever way the column is sorted
  assert.strictEqual(sortBy([{ key: "x5", dir: "asc" }]).at(-1), "c");
  assert.strictEqual(sortBy([{ key: "x5", dir: "desc" }]).at(-1), "c");
});

test("percentileRank counts ties as half", () => {
  assert.strictEqual(api.percentileRank(5, [1, 2, 3, 4, 5]), 90);
  assert.strictEqual(api.percentileRank(1, [1, 2, 3, 4]), 13);
  assert.strictEqual(api.percentileRank(3, [3, 3, 3, 3]), 50);
  assert.strictEqual(api.percentileRank(null, [1, 2]), null);
  assert.strictEqual(api.percentileRank(2, []), null);
});

test("fdrLevel clamps to the 1-5 scale", () => {
  assert.strictEqual(api.fdrLevel(1), 1);
  assert.strictEqual(api.fdrLevel(5), 5);
  assert.strictEqual(api.fdrLevel(7), 5);
  assert.strictEqual(api.fdrLevel(0), 1);
  assert.strictEqual(api.fdrLevel(null), null);
  assert.strictEqual(api.fdrLevel(undefined), null);
});

test("fdrTickerRows: doubles, blanks and run score", () => {
  const side = (short, difficulty) => ({ short_name: short, name: short, difficulty });
  const gws = [
    { gameweek: 5, fixtures: [{ home: side("AAA", 2), away: side("BBB", 4) }] },
    { gameweek: 6, fixtures: [{ home: side("AAA", 3), away: side("CCC", 3) }, { home: side("BBB", 2), away: side("AAA", 2) }] },
    { gameweek: 7, fixtures: [{ home: side("CCC", 2), away: side("BBB", 5) }] },
  ];
  const rows = Object.fromEntries(api.fdrTickerRows(gws, 6, 2).map((r) => [r.short, r]));
  assert.deepStrictEqual(rows.AAA.cells.map((c) => c.gw), [6, 7]);
  // GW6 double (3 and 2 -> mean 2.5, minus one for the extra game = 1.5); GW7 blank = 5
  assert.strictEqual(rows.AAA.cells[0].games.length, 2);
  assert.deepStrictEqual(rows.AAA.cells[1].games, []);
  assert.strictEqual(rows.AAA.score, (1.5 + 5) / 2);
  assert.deepStrictEqual(rows.BBB.cells[0].games[0], { opp: "AAA", home: true, difficulty: 2 });
  assert.strictEqual(rows.BBB.cells[1].games[0].home, false);
  assert.strictEqual(rows.CCC.score, (3 + 2) / 2);
});
