"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const Model = require("../../planner/model.js");

function player(id, position, team, price) {
  return { id, position, team, price };
}

function holding(playerId, inXI) {
  return { playerId, inXI, isCaptain: false, isVice: false };
}

function twoPlayerFixture() {
  // Just enough for transfer/bank arithmetic tests -- these don't need a legal 15-man squad,
  // only two players and a couple of replacement candidates.
  const playersById = {
    100: player(100, "MID", "ARS", 8.0),
    101: player(101, "MID", "LIV", 9.0), // pricier replacement
    102: player(102, "MID", "MUN", 6.0), // cheaper replacement
  };
  const baseSquad = [holding(100, true)];
  return { playersById, baseSquad };
}

// ---- base state / no-op gameweeks ----------------------------------------------------------

test("computeStateAtGameweek at the base gameweek returns the base state untouched", () => {
  const { playersById, baseSquad } = twoPlayerFixture();
  const draft = Model.createDraft({ baseGameweek: 2, baseSquad, baseBank: 3.5, baseFreeTransfers: 1 });
  const state = Model.computeStateAtGameweek(draft, 2, playersById);
  assert.equal(state.bank, 3.5);
  assert.equal(state.freeTransfersAvailable, 1);
  assert.deepEqual(state.squad, baseSquad);
  assert.equal(state.transferCostThisGw, 0);
});

test("free transfers accrue by 1 each untouched gameweek, capped at 5", () => {
  const { playersById, baseSquad } = twoPlayerFixture();
  const draft = Model.createDraft({ baseGameweek: 1, baseSquad, baseBank: 0, baseFreeTransfers: 1 });
  // GW1 base=1 free transfer. Walk forward with nothing planned: GW2 should show 2 (accrued
  // from GW1's own untouched step), GW3 -> 3, etc, capping at 5 rather than growing forever.
  assert.equal(Model.computeStateAtGameweek(draft, 2, playersById).freeTransfersAvailable, 2);
  assert.equal(Model.computeStateAtGameweek(draft, 5, playersById).freeTransfersAvailable, 5);
  assert.equal(Model.computeStateAtGameweek(draft, 8, playersById).freeTransfersAvailable, 5);
});

// ---- transfers: bank, free-transfer consumption, hits --------------------------------------

test("a single free transfer updates bank by the price difference and consumes the free transfer", () => {
  const { playersById, baseSquad } = twoPlayerFixture();
  let draft = Model.createDraft({ baseGameweek: 2, baseSquad, baseBank: 2.0, baseFreeTransfers: 1 });
  draft = Model.applyTransfer(draft, 2, 100, 101); // sell 8.0m, buy 9.0m -> bank drops by 1.0
  const state = Model.computeStateAtGameweek(draft, 2, playersById);
  assert.equal(state.bank, 1.0);
  assert.equal(state.transferCostThisGw, 0);
  assert.equal(state.transfersMadeThisGw, 1);
  assert.ok(state.squad.some((h) => h.playerId === 101));
  assert.ok(!state.squad.some((h) => h.playerId === 100));
  // The next gameweek should show the free transfer used, then +1 accrual -> still 1 (not 2).
  const nextState = Model.computeStateAtGameweek(draft, 3, playersById);
  assert.equal(nextState.freeTransfersAvailable, 1);
});

test("selling for less than buying reduces bank correctly, never below what was validated", () => {
  const { playersById, baseSquad } = twoPlayerFixture();
  let draft = Model.createDraft({ baseGameweek: 2, baseSquad, baseBank: 5.0, baseFreeTransfers: 1 });
  draft = Model.applyTransfer(draft, 2, 100, 102); // sell 8.0m, buy 6.0m -> bank grows by 2.0
  const state = Model.computeStateAtGameweek(draft, 2, playersById);
  assert.equal(state.bank, 7.0);
});

test("a transfer beyond the free allocation costs 4 points per extra transfer", () => {
  const { playersById, baseSquad } = twoPlayerFixture();
  const squad = [holding(100, true), holding(102, true)];
  let draft = Model.createDraft({
    baseGameweek: 3, baseSquad: squad, baseBank: 10.0, baseFreeTransfers: 1,
  });
  // 2 transfers made, only 1 free -> 1 hit -> -4.
  draft = Model.applyTransfer(draft, 3, 100, 101);
  draft = Model.applyTransfer(draft, 3, 102, 101 /* arbitrary second swap target */);
  const state = Model.computeStateAtGameweek(draft, 3, playersById);
  assert.equal(state.transfersMadeThisGw, 2);
  assert.equal(state.transferCostThisGw, Model.POINTS_PER_HIT);
});

test("undoTransfer removes a recorded transfer and its cost", () => {
  const { playersById, baseSquad } = twoPlayerFixture();
  let draft = Model.createDraft({ baseGameweek: 2, baseSquad, baseBank: 5.0, baseFreeTransfers: 1 });
  draft = Model.applyTransfer(draft, 2, 100, 101);
  draft = Model.undoTransfer(draft, 2, 0);
  const state = Model.computeStateAtGameweek(draft, 2, playersById);
  assert.equal(state.transfersMadeThisGw, 0);
  assert.equal(state.bank, 5.0);
  assert.deepEqual(state.squad, baseSquad);
});

test("free-transfer accounting carries correctly across multiple planned gameweeks", () => {
  const { playersById, baseSquad } = twoPlayerFixture();
  let draft = Model.createDraft({ baseGameweek: 1, baseSquad, baseBank: 20.0, baseFreeTransfers: 1 });
  // GW1: no transfer -> GW2 has 2 free transfers banked.
  // GW2: use 1 transfer (free) -> GW3 should have 1+1=2... wait: at GW2, freeTransfersAvailable
  // going in is 2 (from GW1 accrual); using 1 of them leaves 1 consumed, then +1 accrual -> 2.
  draft = Model.applyTransfer(draft, 2, 100, 102);
  const gw2 = Model.computeStateAtGameweek(draft, 2, playersById);
  assert.equal(gw2.freeTransfersAvailable, 2); // reflects the +1 accrual baked into this GW's result
  const gw3 = Model.computeStateAtGameweek(draft, 3, playersById);
  assert.equal(gw3.freeTransfersAvailable, 2);
});

// ---- chips ----------------------------------------------------------------------------------

test("wildcard active in a gameweek makes transfers free and uncapped", () => {
  const { playersById, baseSquad } = twoPlayerFixture();
  const squad = [holding(100, true), holding(102, true)];
  let draft = Model.createDraft({ baseGameweek: 5, baseSquad: squad, baseBank: 0, baseFreeTransfers: 1 });
  draft = Model.setChip(draft, 5, "wildcard");
  draft = Model.applyTransfer(draft, 5, 100, 101);
  draft = Model.applyTransfer(draft, 5, 102, 101);
  const state = Model.computeStateAtGameweek(draft, 5, playersById);
  assert.equal(state.transferCostThisGw, 0);
  assert.equal(state.activeChip, "wildcard");
  assert.equal(state.unlimitedTransfersThisGw, true);
  // Wildcard doesn't drain the free-transfer bank -- next gameweek just accrues normally.
  const nextState = Model.computeStateAtGameweek(draft, 6, playersById);
  assert.equal(nextState.freeTransfersAvailable, 2);
});

test("canAssignChip blocks a chip already used in the same half-season", () => {
  const { baseSquad } = twoPlayerFixture();
  let draft = Model.createDraft({ baseGameweek: 3, baseSquad, baseBank: 0, baseFreeTransfers: 1 });
  draft = Model.setChip(draft, 3, "bench_boost");
  const check = Model.canAssignChip(draft, 10, "bench_boost");
  assert.equal(check.allowed, false);
  assert.match(check.reason, /already used/);
});

test("canAssignChip allows the same chip type once in each half-season", () => {
  const { baseSquad } = twoPlayerFixture();
  let draft = Model.createDraft({ baseGameweek: 3, baseSquad, baseBank: 0, baseFreeTransfers: 1 });
  draft = Model.setChip(draft, 3, "wildcard"); // set 1 (before GW19)
  const checkSet2 = Model.canAssignChip(draft, 25, "wildcard"); // set 2 (after GW19)
  assert.equal(checkSet2.allowed, true);
});

test("GW19 itself already belongs to chip set 2, independent of set-1 usage", () => {
  const { baseSquad } = twoPlayerFixture();
  const draft = Model.createDraft({ baseGameweek: 1, baseSquad, baseBank: 0, baseFreeTransfers: 1 });
  // triple_captain was never used in set 1 -- it's simply forfeited (see checkGw19Deadline),
  // not "still available but blocked." Assigning it at GW19 draws from the fresh set-2
  // allocation and must be allowed.
  const check = Model.canAssignChip(draft, 19, "triple_captain");
  assert.equal(check.allowed, true);
  assert.equal(Model.chipSetFor(19), "set2");
  assert.equal(Model.chipSetFor(18), "set1");
});

test("setChip throws when assigning an already-used chip (fail fast, don't silently no-op)", () => {
  const { baseSquad } = twoPlayerFixture();
  let draft = Model.createDraft({ baseGameweek: 3, baseSquad, baseBank: 0, baseFreeTransfers: 1 });
  draft = Model.setChip(draft, 3, "free_hit");
  assert.throws(() => Model.setChip(draft, 12, "free_hit"));
});

test("checkGw19Deadline flags urgency inside the warning window and forfeiture at GW19", () => {
  const notUrgentYet = Model.checkGw19Deadline(10, []);
  assert.equal(notUrgentYet.urgent, false);
  assert.equal(notUrgentYet.forfeitedNow, false);

  const urgent = Model.checkGw19Deadline(17, ["wildcard"]); // 2 gameweeks left, 3 chips unused
  assert.equal(urgent.urgent, true);
  assert.equal(urgent.unusedSet1Chips.length, 3);

  const forfeited = Model.checkGw19Deadline(19, []);
  assert.equal(forfeited.forfeitedNow, true);
  assert.equal(forfeited.urgent, false); // never both at once (real bug this mirrors the fix for)

  const allUsed = Model.checkGw19Deadline(19, Model.ALL_CHIP_TYPES);
  assert.equal(allUsed.forfeitedNow, false);
});

// ---- starting-XI/bench swaps (drag-and-drop lineup rearrangement) ---------------------------

test("swapXIStatus exchanges two squad members' starting-XI/bench status", () => {
  const squad = [holding(1, true), holding(2, false)]; // 1 starts, 2 is benched
  const playersById = { 1: player(1, "MID", "ARS", 5.0), 2: player(2, "MID", "LIV", 5.0) };
  let draft = Model.createDraft({ baseGameweek: 2, baseSquad: squad, baseBank: 0, baseFreeTransfers: 1 });
  draft = Model.swapXIStatus(draft, 2, playersById, 1, 2);
  const state = Model.computeStateAtGameweek(draft, 2, playersById);
  assert.equal(state.squad.find((h) => h.playerId === 1).inXI, false);
  assert.equal(state.squad.find((h) => h.playerId === 2).inXI, true);
});

test("swapXIStatus is a harmless no-op when both players already share the same XI status", () => {
  const squad = [holding(1, true), holding(2, true)];
  const playersById = { 1: player(1, "MID", "ARS", 5.0), 2: player(2, "MID", "LIV", 5.0) };
  let draft = Model.createDraft({ baseGameweek: 2, baseSquad: squad, baseBank: 0, baseFreeTransfers: 1 });
  draft = Model.swapXIStatus(draft, 2, playersById, 1, 2);
  const state = Model.computeStateAtGameweek(draft, 2, playersById);
  assert.equal(state.squad.find((h) => h.playerId === 1).inXI, true);
  assert.equal(state.squad.find((h) => h.playerId === 2).inXI, true);
});

test("swapXIStatus throws if either player isn't in the squad at that gameweek", () => {
  const squad = [holding(1, true)];
  const playersById = { 1: player(1, "MID", "ARS", 5.0), 99: player(99, "MID", "LIV", 5.0) };
  const draft = Model.createDraft({ baseGameweek: 2, baseSquad: squad, baseBank: 0, baseFreeTransfers: 1 });
  assert.throws(() => Model.swapXIStatus(draft, 2, playersById, 1, 99));
});

test("an XI override carries forward into future gameweeks until changed again, same as captain", () => {
  const squad = [holding(1, true), holding(2, false)];
  const playersById = { 1: player(1, "MID", "ARS", 5.0), 2: player(2, "MID", "LIV", 5.0) };
  let draft = Model.createDraft({ baseGameweek: 2, baseSquad: squad, baseBank: 0, baseFreeTransfers: 1 });
  draft = Model.swapXIStatus(draft, 2, playersById, 1, 2);
  const laterState = Model.computeStateAtGameweek(draft, 5, playersById); // nothing planned for GW3-5
  assert.equal(laterState.squad.find((h) => h.playerId === 1).inXI, false);
  assert.equal(laterState.squad.find((h) => h.playerId === 2).inXI, true);
});

test("multiple XI overrides in the same gameweek compose instead of overwriting each other", () => {
  const squad = [holding(1, true), holding(2, false), holding(3, true), holding(4, false)];
  const playersById = {
    1: player(1, "MID", "ARS", 5.0), 2: player(2, "MID", "LIV", 5.0),
    3: player(3, "DEF", "MUN", 4.0), 4: player(4, "DEF", "CHE", 4.0),
  };
  let draft = Model.createDraft({ baseGameweek: 2, baseSquad: squad, baseBank: 0, baseFreeTransfers: 1 });
  draft = Model.swapXIStatus(draft, 2, playersById, 1, 2); // MID swap
  draft = Model.swapXIStatus(draft, 2, playersById, 3, 4); // DEF swap, same gameweek
  const state = Model.computeStateAtGameweek(draft, 2, playersById);
  assert.equal(state.squad.find((h) => h.playerId === 1).inXI, false);
  assert.equal(state.squad.find((h) => h.playerId === 2).inXI, true);
  assert.equal(state.squad.find((h) => h.playerId === 3).inXI, false);
  assert.equal(state.squad.find((h) => h.playerId === 4).inXI, true);
});

// ---- squad value / spending power -----------------------------------------------------------

test("squadValue and spendingPower sum real player prices plus bank", () => {
  const { playersById, baseSquad } = twoPlayerFixture();
  const value = Model.squadValue(baseSquad, playersById);
  assert.equal(value, 8.0);
  assert.equal(Model.spendingPower(baseSquad, 2.5, playersById), 10.5);
});

// ---- integration with Rules.validateDraftSquad ----------------------------------------------

test("a transfer that breaks the club-limit rule is still applied by the reducer, but flagged by Rules", () => {
  const Rules = require("../../planner/rules.js");
  const playersById = {
    1: player(1, "GKP", "ARS", 5.0), 2: player(2, "GKP", "AVL", 4.0),
    3: player(3, "DEF", "ARS", 5.5), 4: player(4, "DEF", "LIV", 5.5), 5: player(5, "DEF", "MUN", 4.5),
    6: player(6, "DEF", "CHE", 4.5), 7: player(7, "DEF", "TOT", 4.0),
    8: player(8, "MID", "LIV", 8.0), 9: player(9, "MID", "MCI", 7.5), 10: player(10, "MID", "MUN", 6.5),
    11: player(11, "MID", "CHE", 5.5), 12: player(12, "MID", "TOT", 5.0),
    13: player(13, "FWD", "MCI", 11.0), 14: player(14, "FWD", "ARS", 8.0), 15: player(15, "FWD", "LIV", 6.0),
    16: player(16, "FWD", "ARS", 6.0), // a 3rd Arsenal forward, used to push the club over the cap
  };
  const benchIds = new Set([2, 7, 12, 15]);
  const baseSquad = Object.keys(playersById).filter((id) => Number(id) !== 16).map((id) => holding(Number(id), !benchIds.has(Number(id))));
  let draft = Model.createDraft({ baseGameweek: 2, baseSquad, baseBank: 0, baseFreeTransfers: 1 });
  draft = Model.applyTransfer(draft, 2, 15, 16); // sell the Liverpool FWD, buy a 3rd Arsenal FWD (already have 1, 3, 14)
  const state = Model.computeStateAtGameweek(draft, 2, playersById);
  const validation = Rules.validateDraftSquad(state.squad, playersById, state.bank);
  assert.equal(validation.valid, false);
  assert.ok(validation.warnings.some((w) => w.code === "club_limit" && w.club === "ARS"));
});
