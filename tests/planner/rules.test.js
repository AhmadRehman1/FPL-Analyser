"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const rules = require("../../planner/rules.js");

// ---- fixture builders -------------------------------------------------------------------

function player(id, position, team, price) {
  return { id: id, position: position, team: team, price: price };
}

function holding(playerId, inXI) {
  return { playerId: playerId, inXI: inXI, isCaptain: false, isVice: false };
}

/** A legal 15-man squad: 2 GKP, 5 DEF, 5 MID, 3 FWD, spread across enough clubs to respect the
 * 3-per-club cap, with a legal 1-4-4-2 starting XI (GK1 + 4 DEF + 4 MID + 2 FWD) and 4 on the bench
 * (GK2 + 1 DEF + 1 MID + 1 FWD). Total price = 100.0 (the standard starting budget). */
function legalSquadFixture() {
  const players = [
    player(1, "GKP", "ARS", 5.0), player(2, "GKP", "AVL", 4.0),
    player(3, "DEF", "ARS", 5.5), player(4, "DEF", "LIV", 5.5), player(5, "DEF", "MUN", 4.5),
    player(6, "DEF", "CHE", 4.5), player(7, "DEF", "TOT", 4.0),
    player(8, "MID", "LIV", 8.0), player(9, "MID", "MCI", 7.5), player(10, "MID", "MUN", 6.5),
    player(11, "MID", "CHE", 5.5), player(12, "MID", "TOT", 5.0),
    player(13, "FWD", "MCI", 11.0), player(14, "FWD", "ARS", 8.0), player(15, "FWD", "LIV", 6.0),
  ];
  const playersById = {};
  players.forEach((p) => { playersById[p.id] = p; });

  const benchIds = new Set([2, 7, 12, 15]);
  const squad = players.map((p) => holding(p.id, !benchIds.has(p.id)));
  return { squad, playersById };
}

// ---- club limits --------------------------------------------------------------------------

test("validateClubLimits passes a squad with at most 3 per club", () => {
  const { squad, playersById } = legalSquadFixture();
  const { resolved } = rules.resolvePlayers(squad, playersById);
  const result = rules.validateClubLimits(resolved);
  assert.equal(result.valid, true);
  assert.deepEqual(result.violations, []);
});

test("validateClubLimits flags a 4th player from the same club by name", () => {
  const { squad, playersById } = legalSquadFixture();
  // Arsenal already has players 1, 3, 14 (3 players) -- swap player 15 (Liverpool FWD) for a
  // 4th Arsenal player to push the club over the cap.
  playersById[15] = player(15, "FWD", "ARS", 6.0);
  const { resolved } = rules.resolvePlayers(squad, playersById);
  const result = rules.validateClubLimits(resolved);
  assert.equal(result.valid, false);
  assert.equal(result.violations.length, 1);
  assert.equal(result.violations[0].club, "ARS");
  assert.equal(result.violations[0].count, 4);
  assert.match(result.violations[0].message, /Too many selected from ARS \(4\/3\)/);
});

// ---- squad composition ---------------------------------------------------------------------

test("validateSquadComposition passes the standard 2-5-5-3 split", () => {
  const { squad, playersById } = legalSquadFixture();
  const { resolved } = rules.resolvePlayers(squad, playersById);
  const result = rules.validateSquadComposition(resolved);
  assert.equal(result.valid, true);
});

test("validateSquadComposition rejects 3 GKP / 4 DEF (wrong split, still 15 total)", () => {
  const { squad, playersById } = legalSquadFixture();
  playersById[7] = player(7, "GKP", "TOT", 4.0); // was DEF, now a 3rd GKP
  const { resolved } = rules.resolvePlayers(squad, playersById);
  const result = rules.validateSquadComposition(resolved);
  assert.equal(result.valid, false);
  const codes = result.violations.map((v) => v.position);
  assert.ok(codes.includes("GKP"));
  assert.ok(codes.includes("DEF"));
});

// ---- squad size / duplicates ----------------------------------------------------------------

test("validateSquadSize rejects a 14-player squad", () => {
  const { squad } = legalSquadFixture();
  squad.pop();
  const result = rules.validateSquadSize(squad);
  assert.equal(result.valid, false);
  assert.equal(result.violations[0].code, "squad_size");
});

test("validateSquadSize rejects a duplicate player id", () => {
  const { squad } = legalSquadFixture();
  squad.pop();
  squad.push(holding(1, false)); // player 1 already in the squad (as GK1, in XI)
  const result = rules.validateSquadSize(squad);
  assert.equal(result.valid, false);
  assert.ok(result.violations.some((v) => v.code === "duplicate_player"));
});

// ---- formation ----------------------------------------------------------------------------

test("validateFormation passes a legal 1-4-4-2", () => {
  const { squad, playersById } = legalSquadFixture();
  const { resolved } = rules.resolvePlayers(squad, playersById);
  const result = rules.validateFormation(resolved);
  assert.equal(result.valid, true);
  assert.equal(result.xiCount, 11);
  assert.equal(result.benchCount, 4);
});

test("validateFormation rejects 2 goalkeepers in the starting XI", () => {
  const { squad, playersById } = legalSquadFixture();
  // Bench player 7 (a DEF) instead of GK2 (player 2), so both GKs start.
  const s = squad.map((h) => {
    if (h.playerId === 2) return holding(2, true);
    if (h.playerId === 7) return holding(7, false);
    return h;
  });
  const { resolved } = rules.resolvePlayers(s, playersById);
  const result = rules.validateFormation(resolved);
  assert.equal(result.valid, false);
  assert.ok(result.violations.some((v) => v.code === "formation" && v.position === "GKP"));
});

test("validateFormation rejects fewer than 3 starting defenders", () => {
  const { squad, playersById } = legalSquadFixture();
  // Bench 2 more DEF (5, 6) and start 2 more MID (bench player 12 stays benched but pull in
  // no replacement -- simplest is to directly rebuild an XI with only 2 DEF via a fresh fixture).
  const s = squad.map((h) => {
    if (h.playerId === 5 || h.playerId === 6) return holding(h.playerId, false);
    if (h.playerId === 12) return holding(h.playerId, true); // free up an XI slot with an extra MID
    return h;
  });
  const { resolved } = rules.resolvePlayers(s, playersById);
  const result = rules.validateFormation(resolved);
  assert.equal(result.valid, false);
  const defViolation = result.violations.find((v) => v.code === "formation" && v.position === "DEF");
  assert.ok(defViolation, "expected a DEF formation violation");
  assert.equal(defViolation.count, 2);
});

test("validateFormation rejects more than 3 starting forwards", () => {
  // A real 15-man squad only ever holds 3 FWD total (validateSquadComposition's own job), so
  // exercising the ">3 forwards in the XI" branch needs a formation-only fixture rather than a
  // mutation of the composition-legal one above -- validateFormation is tested here in
  // isolation from validateSquadComposition, exactly as they run independently in practice.
  const players = [
    player(1, "GKP", "ARS", 5.0),
    player(2, "DEF", "LIV", 5.0), player(3, "DEF", "MUN", 5.0),
    player(4, "DEF", "CHE", 5.0), player(5, "DEF", "TOT", 5.0),
    player(6, "MID", "MCI", 6.0), player(7, "MID", "AVL", 6.0),
    player(8, "FWD", "ARS", 8.0), player(9, "FWD", "LIV", 8.0),
    player(10, "FWD", "MUN", 8.0), player(11, "FWD", "CHE", 8.0),
  ];
  const playersById = {};
  players.forEach((p) => { playersById[p.id] = p; });
  const squad = players.map((p) => holding(p.id, true)); // all 11 start
  const { resolved } = rules.resolvePlayers(squad, playersById);
  const result = rules.validateFormation(resolved);
  assert.equal(result.valid, false);
  const fwdViolation = result.violations.find((v) => v.code === "formation" && v.position === "FWD");
  assert.ok(fwdViolation);
  assert.equal(fwdViolation.count, 4);
});

// ---- budget ---------------------------------------------------------------------------------

test("validateBudget accepts zero and positive bank", () => {
  assert.equal(rules.validateBudget(0).valid, true);
  assert.equal(rules.validateBudget(4.5).valid, true);
});

test("validateBudget rejects negative bank", () => {
  const result = rules.validateBudget(-0.5);
  assert.equal(result.valid, false);
  assert.equal(result.violations[0].code, "negative_budget");
});

test("validateBudget tolerates floating-point dust around zero", () => {
  const dust = 0.3 - 0.2 - 0.1; // classic float artefact, slightly negative
  assert.equal(rules.validateBudget(dust).valid, true);
});

// ---- captaincy ------------------------------------------------------------------------------

test("validateCaptaincy passes a captain and vice both in the XI", () => {
  const { squad, playersById } = legalSquadFixture();
  const { resolved } = rules.resolvePlayers(squad, playersById);
  const result = rules.validateCaptaincy(resolved, 13, 8);
  assert.equal(result.valid, true);
});

test("validateCaptaincy rejects a captain on the bench", () => {
  const { squad, playersById } = legalSquadFixture();
  const { resolved } = rules.resolvePlayers(squad, playersById);
  const result = rules.validateCaptaincy(resolved, 15, 8); // player 15 is benched
  assert.equal(result.valid, false);
  assert.ok(result.violations.some((v) => v.code === "captain_not_in_xi"));
});

test("validateCaptaincy rejects vice-captain equal to captain", () => {
  const { squad, playersById } = legalSquadFixture();
  const { resolved } = rules.resolvePlayers(squad, playersById);
  const result = rules.validateCaptaincy(resolved, 13, 13);
  assert.equal(result.valid, false);
  assert.ok(result.violations.some((v) => v.code === "vice_equals_captain"));
});

// ---- validateDraftSquad (the one entry point the UI calls) -----------------------------------

test("validateDraftSquad is valid end-to-end for a legal squad within budget", () => {
  const { squad, playersById } = legalSquadFixture();
  const result = rules.validateDraftSquad(squad, playersById, 0.0);
  assert.equal(result.valid, true);
  assert.deepEqual(result.warnings, []);
});

test("validateDraftSquad collects violations from every rule at once", () => {
  const { squad, playersById } = legalSquadFixture();
  playersById[15] = player(15, "FWD", "ARS", 6.0); // 4th Arsenal player -> club-limit violation
  const result = rules.validateDraftSquad(squad, playersById, -1.0); // also over budget
  assert.equal(result.valid, false);
  const codes = result.warnings.map((w) => w.code);
  assert.ok(codes.includes("club_limit"));
  assert.ok(codes.includes("negative_budget"));
});

test("validateDraftSquad flags a missing/unknown player id instead of throwing", () => {
  const { squad, playersById } = legalSquadFixture();
  delete playersById[15];
  const result = rules.validateDraftSquad(squad, playersById, 0.0);
  assert.equal(result.valid, false);
  assert.ok(result.warnings.some((w) => w.code === "missing_player" && w.playerId === 15));
});
