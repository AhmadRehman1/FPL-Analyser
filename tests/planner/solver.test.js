"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const Solver = require("../../planner/solver.js");

function player(id, position, team, price, webName) {
  return { id, position, team, price, web_name: webName || ("P" + id) };
}

function projectionsFixture() {
  // element ids 100..104, EP for GW2 and GW3.
  return {
    players: [
      { fpl_element_id: 100, ep_per_gw: [{ gw: 2, ep: 3.0 }, { gw: 3, ep: 3.5 }] }, // owned, weak
      { fpl_element_id: 101, ep_per_gw: [{ gw: 2, ep: 7.0 }, { gw: 3, ep: 6.5 }] }, // strong replacement
      { fpl_element_id: 102, ep_per_gw: [{ gw: 2, ep: 4.0 }, { gw: 3, ep: 4.0 }] }, // mild upgrade
      { fpl_element_id: 103, ep_per_gw: [{ gw: 2, ep: 1.0 }, { gw: 3, ep: 1.0 }] }, // downgrade
      { fpl_element_id: 104, ep_per_gw: [{ gw: 2, ep: 3.0 }, { gw: 3, ep: 3.0 }] }, // same as owned
    ],
  };
}

test("projectionsByElementId keys by fpl_element_id and sums per-gw bands", () => {
  const byElement = Solver.projectionsByElementId(projectionsFixture());
  assert.equal(byElement[101][2], 7.0);
  assert.equal(byElement[101][3], 6.5);
});

test("projectionsByElementId skips rows with no resolved element id", () => {
  const payload = { players: [{ fpl_element_id: null, ep_per_gw: [{ gw: 2, ep: 9 }] }] };
  const byElement = Solver.projectionsByElementId(payload);
  assert.deepEqual(byElement, {});
});

test("horizonEp sums across the requested gameweeks, 0 for missing ones", () => {
  const byElement = Solver.projectionsByElementId(projectionsFixture());
  assert.equal(Solver.horizonEp(byElement, 101, [2, 3]), 13.5);
  assert.equal(Solver.horizonEp(byElement, 999, [2, 3]), 0); // unknown player
  assert.equal(Solver.horizonEp(byElement, 101, [2, 3, 4]), 13.5); // GW4 has no band
});

test("hasProjections distinguishes an empty payload from a real one", () => {
  assert.equal(Solver.hasProjections({}), false);
  assert.equal(Solver.hasProjections(Solver.projectionsByElementId(projectionsFixture())), true);
});

test("candidatesForSlot only offers same-position, unowned, affordable players", () => {
  const outPlayer = player(100, "MID", "ARS", 6.0);
  const candidates = [
    player(101, "MID", "LIV", 8.0),  // affordable if bank covers the 2.0 gap
    player(102, "FWD", "MUN", 5.0),  // wrong position
    player(100, "MID", "ARS", 6.0),  // the outgoing player itself
    player(103, "MID", "CHE", 20.0), // unaffordable
  ];
  const pool = Solver.candidatesForSlot(outPlayer, candidates, { bank: 2.0, ownedIds: new Set([100]) });
  assert.deepEqual(pool.map((p) => p.id), [101]);
});

test("suggestTransfers picks the best net-positive replacement, ranked best-first", () => {
  const playersById = {
    100: player(100, "MID", "ARS", 6.0, "Weak"),
    101: player(101, "MID", "LIV", 6.0, "Strong"),
    102: player(102, "MID", "MUN", 6.0, "Mild"),
    103: player(103, "MID", "CHE", 6.0, "Worse"),
    104: player(104, "MID", "TOT", 6.0, "Same"),
  };
  const candidates = Object.values(playersById);
  const state = { squad: [{ playerId: 100, inXI: true, isCaptain: false, isVice: false }], bank: 0 };
  const projByElement = Solver.projectionsByElementId(projectionsFixture());

  const suggestions = Solver.suggestTransfers(state, playersById, candidates, projByElement, {
    horizonGameweeks: [2, 3], freeTransfersAvailable: 1,
  });
  assert.equal(suggestions.length, 1); // one owned slot -> at most one suggestion
  assert.equal(suggestions[0].inId, 101); // the strongest replacement wins, not just any upgrade
  assert.equal(suggestions[0].cost, 0); // a free transfer was available
  assert.ok(suggestions[0].net > 0);
});

test("suggestTransfers respects locked players (never suggested out)", () => {
  const playersById = {
    100: player(100, "MID", "ARS", 6.0),
    101: player(101, "MID", "LIV", 6.0),
  };
  const candidates = Object.values(playersById);
  const state = { squad: [{ playerId: 100, inXI: true, isCaptain: false, isVice: false }], bank: 0 };
  const projByElement = Solver.projectionsByElementId(projectionsFixture());
  const suggestions = Solver.suggestTransfers(state, playersById, candidates, projByElement, {
    horizonGameweeks: [2, 3], lockedIds: new Set([100]),
  });
  assert.deepEqual(suggestions, []);
});

test("suggestTransfers respects ignored replacement targets", () => {
  const playersById = {
    100: player(100, "MID", "ARS", 6.0),
    101: player(101, "MID", "LIV", 6.0), // the best replacement, but ignored
    102: player(102, "MID", "MUN", 6.0), // the next-best
  };
  const candidates = Object.values(playersById);
  const state = { squad: [{ playerId: 100, inXI: true, isCaptain: false, isVice: false }], bank: 0 };
  const projByElement = Solver.projectionsByElementId(projectionsFixture());
  const suggestions = Solver.suggestTransfers(state, playersById, candidates, projByElement, {
    horizonGameweeks: [2, 3], ignoredIds: new Set([101]), freeTransfersAvailable: 1,
  });
  assert.equal(suggestions.length, 1);
  assert.equal(suggestions[0].inId, 102);
});

test("suggestTransfers applies the -4 hit cost when no free transfer is available, and drops net-negative moves", () => {
  const playersById = {
    100: player(100, "MID", "ARS", 6.0),
    102: player(102, "MID", "MUN", 6.0), // EP gain is only +1.0 over the horizon (4.0+4.0 vs 3.0+3.5=6.5 -> gain 1.5)
  };
  const candidates = Object.values(playersById);
  const state = { squad: [{ playerId: 100, inXI: true, isCaptain: false, isVice: false }], bank: 0 };
  const projByElement = Solver.projectionsByElementId(projectionsFixture());
  const suggestions = Solver.suggestTransfers(state, playersById, candidates, projByElement, {
    horizonGameweeks: [2, 3], freeTransfersAvailable: 0,
  });
  // gain = 8.0 - 6.5 = 1.5; net = 1.5 - 4 = -2.5 -> should be dropped entirely (net must be > 0)
  assert.deepEqual(suggestions, []);
});

// ---- suggestMultiTransfers (bounded 2-for-2 combinatorial search) ---------------------------

function multiFixtureSamePosition(extra) {
  extra = extra || {};
  const playersById = {
    100: player(100, "MID", "ARS", 5.0, "Mid1"), 101: player(101, "MID", "LIV", 5.0, "Mid2"),
    200: player(200, "MID", "MUN", 7.0, "Strong1"), 201: player(201, "MID", "CHE", 7.0, "Strong2"),
  };
  Object.assign(playersById, extra.morePlayers || {});
  const state = {
    squad: [
      { playerId: 100, inXI: true, isCaptain: false, isVice: false },
      { playerId: 101, inXI: true, isCaptain: false, isVice: false },
    ],
    bank: extra.bank != null ? extra.bank : 4,
  };
  const projByElement = Object.assign({ 100: { 2: 2 }, 101: { 2: 2 }, 200: { 2: 6 }, 201: { 2: 6 } }, extra.moreProjections || {});
  return { playersById, state, projByElement, candidates: Object.values(playersById) };
}

test("suggestMultiTransfers finds a same-position 2-for-2 combo, net of price and EP", () => {
  const { playersById, state, projByElement, candidates } = multiFixtureSamePosition();
  const results = Solver.suggestMultiTransfers(state, playersById, candidates, projByElement, {
    horizonGameweeks: [2], freeTransfersAvailable: 2,
  });
  assert.equal(results.length, 1);
  assert.deepEqual(results[0].outIds.slice().sort((a, b) => a - b), [100, 101]);
  assert.deepEqual(results[0].inIds.slice().sort((a, b) => a - b), [200, 201]);
  assert.equal(results[0].gain, 8); // (6+6) - (2+2)
  assert.equal(results[0].cost, 0);
  assert.equal(results[0].net, 8);
});

test("suggestMultiTransfers finds a cross-position (DEF+FWD) combo matching the outgoing position multiset", () => {
  const playersById = {
    100: player(100, "DEF", "ARS", 5.0, "D1"), 101: player(101, "FWD", "LIV", 7.0, "F1"),
    200: player(200, "DEF", "MUN", 6.0, "SD"), 201: player(201, "FWD", "CHE", 8.0, "SF"),
    // A same-position decoy that must never appear -- nothing of this position is being sold.
    300: player(300, "MID", "TOT", 9.0, "MidDecoy"),
  };
  const state = {
    squad: [
      { playerId: 100, inXI: true, isCaptain: false, isVice: false },
      { playerId: 101, inXI: true, isCaptain: false, isVice: false },
    ],
    bank: 2,
  };
  const projByElement = { 100: { 2: 2 }, 101: { 2: 3 }, 200: { 2: 7 }, 201: { 2: 9 }, 300: { 2: 20 } };
  const results = Solver.suggestMultiTransfers(state, playersById, Object.values(playersById), projByElement, {
    horizonGameweeks: [2], freeTransfersAvailable: 2,
  });
  assert.equal(results.length, 1); // the MID decoy must never surface -- no MID was sold
  assert.deepEqual(results[0].outIds.slice().sort((a, b) => a - b), [100, 101]);
  assert.deepEqual(results[0].inIds.slice().sort((a, b) => a - b), [200, 201]);
});

test("suggestMultiTransfers respects the combined budget constraint", () => {
  const { playersById, state, projByElement, candidates } = multiFixtureSamePosition({ bank: 0 }); // needs bank >= 4
  const results = Solver.suggestMultiTransfers(state, playersById, candidates, projByElement, {
    horizonGameweeks: [2], freeTransfersAvailable: 2,
  });
  assert.deepEqual(results, []);
});

test("suggestMultiTransfers rejects a combo that would push a club over the 3-player cap", () => {
  const playersById = {
    100: player(100, "MID", "LIV", 5.0, "M1"), 101: player(101, "MID", "MUN", 5.0, "M2"),
    102: player(102, "GKP", "ARS", 5.0, "GK"), 103: player(103, "DEF", "ARS", 4.0, "D1"), 104: player(104, "DEF", "ARS", 4.0, "D2"),
    200: player(200, "MID", "ARS", 7.0, "BreachesCap"), // a 4th ARS player if bought alongside a non-ARS partner
    201: player(201, "MID", "TOT", 7.0, "Partner"),
    202: player(202, "MID", "TOT", 6.5, "SafePartner"),
  };
  const state = {
    squad: [100, 101, 102, 103, 104].map((id) => ({ playerId: id, inXI: true, isCaptain: false, isVice: false })),
    bank: 10,
  };
  const projByElement = {
    100: { 2: 2 }, 101: { 2: 2 }, 200: { 2: 9 }, 201: { 2: 8 }, 202: { 2: 7 },
  };
  const results = Solver.suggestMultiTransfers(state, playersById, Object.values(playersById), projByElement, {
    horizonGameweeks: [2], freeTransfersAvailable: 2,
  });
  assert.ok(!results.some((r) => r.inIds.includes(200)), "a combo bringing ARS to 4 players must never appear");
  assert.ok(results.some((r) => r.inIds.slice().sort((a, b) => a - b).join(",") === "201,202"), "the club-safe combo should still surface");
});

test("suggestMultiTransfers charges a -4 hit when fewer than 2 free transfers are available, and drops non-positive-net combos", () => {
  const { playersById, state, projByElement, candidates } = multiFixtureSamePosition();
  const oneFree = Solver.suggestMultiTransfers(state, playersById, candidates, projByElement, {
    horizonGameweeks: [2], freeTransfersAvailable: 1,
  });
  assert.equal(oneFree.length, 1);
  assert.equal(oneFree[0].cost, 4);
  assert.equal(oneFree[0].net, 4); // gain 8 - hit 4

  const zeroFree = Solver.suggestMultiTransfers(state, playersById, candidates, projByElement, {
    horizonGameweeks: [2], freeTransfersAvailable: 0,
  });
  assert.deepEqual(zeroFree, []); // gain 8 - hit 8 (2 hits) = net 0, filtered out as non-positive
});

test("suggestMultiTransfers respects locked players -- too few sellable players yields no combos", () => {
  const { playersById, state, projByElement, candidates } = multiFixtureSamePosition();
  const results = Solver.suggestMultiTransfers(state, playersById, candidates, projByElement, {
    horizonGameweeks: [2], freeTransfersAvailable: 2, lockedIds: new Set([100]),
  });
  assert.deepEqual(results, []);
});

test("suggestMultiTransfers respects ignored replacement targets", () => {
  const { playersById, state, projByElement, candidates } = multiFixtureSamePosition({
    bank: 4,
    morePlayers: { 202: player(202, "MID", "BOU", 6.0, "Strong3") },
    moreProjections: { 202: { 2: 5 } },
  });
  const results = Solver.suggestMultiTransfers(state, playersById, candidates, projByElement, {
    horizonGameweeks: [2], freeTransfersAvailable: 2, ignoredIds: new Set([200]),
  });
  assert.ok(results.every((r) => !r.inIds.includes(200)));
  assert.ok(results.some((r) => r.inIds.slice().sort((a, b) => a - b).join(",") === "201,202"));
});

test("suggestMultiTransfers ranks best-net-first and respects maxSuggestions", () => {
  const playersById = {
    100: player(100, "MID", "LIV", 5.0, "M1"), 101: player(101, "MID", "MUN", 5.0, "M2"),
    200: player(200, "MID", "ARS", 7.0, "Best"), 201: player(201, "MID", "TOT", 7.0, "Mid"),
    202: player(202, "MID", "BOU", 7.0, "Worst"), 203: player(203, "MID", "EVE", 7.0, "Extra"),
  };
  const state = {
    squad: [
      { playerId: 100, inXI: true, isCaptain: false, isVice: false },
      { playerId: 101, inXI: true, isCaptain: false, isVice: false },
    ],
    bank: 10,
  };
  const projByElement = { 100: { 2: 1 }, 101: { 2: 1 }, 200: { 2: 9 }, 201: { 2: 7 }, 202: { 2: 5 }, 203: { 2: 3 } };
  const results = Solver.suggestMultiTransfers(state, playersById, Object.values(playersById), projByElement, {
    horizonGameweeks: [2], freeTransfersAvailable: 2, maxSuggestions: 2,
  });
  assert.equal(results.length, 2);
  for (let i = 1; i < results.length; i++) assert.ok(results[i - 1].net >= results[i].net);
  assert.deepEqual(results[0].inIds.slice().sort((a, b) => a - b), [200, 201]); // the two highest-EP incoming players
});

// ---- boundedIncomingPool ---------------------------------------------------------------------

test("boundedIncomingPool keeps only the top-K candidates per position by horizon EP", () => {
  const playersById = {};
  for (let i = 1; i <= 5; i++) playersById[i] = player(i, "MID", "ARS", 5.0, "M" + i);
  const projByElement = { 1: { 2: 1 }, 2: { 2: 2 }, 3: { 2: 3 }, 4: { 2: 4 }, 5: { 2: 5 } };
  const pool = Solver.boundedIncomingPool(Object.values(playersById), projByElement, [2], { candidatePoolLimitPerPosition: 2 });
  assert.equal(pool.length, 2);
  assert.deepEqual(pool.map((r) => r.player.id).sort((a, b) => a - b), [4, 5]);
});

test("boundedIncomingPool excludes owned/ignored ids regardless of EP rank", () => {
  const playersById = { 1: player(1, "MID", "ARS", 5.0, "Best"), 2: player(2, "MID", "LIV", 5.0, "Worst") };
  const projByElement = { 1: { 2: 10 }, 2: { 2: 1 } };
  const pool = Solver.boundedIncomingPool(Object.values(playersById), projByElement, [2], {
    excludeIds: new Set([1]), candidatePoolLimitPerPosition: 5,
  });
  assert.deepEqual(pool.map((r) => r.player.id), [2]);
});

test("suggestTransfers caps results at maxSuggestions", () => {
  const playersById = { 100: player(100, "MID", "ARS", 6.0), 200: player(200, "MID", "LIV", 6.0) };
  const squad = [];
  for (let i = 0; i < 3; i++) {
    playersById[100 + i] = player(100 + i, "MID", "ARS", 6.0);
    squad.push({ playerId: 100 + i, inXI: true, isCaptain: false, isVice: false });
  }
  playersById[300] = player(300, "MID", "LIV", 6.0);
  const projByElement = {
    100: { 2: 1 }, 101: { 2: 1 }, 102: { 2: 1 }, 300: { 2: 9 },
  };
  const state = { squad, bank: 0 };
  const suggestions = Solver.suggestTransfers(state, playersById, Object.values(playersById), projByElement, {
    horizonGameweeks: [2], freeTransfersAvailable: 1, maxSuggestions: 2,
  });
  assert.equal(suggestions.length, 2);
});
