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
