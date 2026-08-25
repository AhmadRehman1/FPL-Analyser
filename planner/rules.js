/* planner/rules.js
 *
 * Pure squad-validity rules for the multi-gameweek transfer planner. No DOM, no fetch, no
 * localStorage -- everything here is a plain function of (squad, playersById) so it can be
 * unit tested directly (see tests/planner/rules.test.js) and reused unmodified from both the
 * browser (index.html's "Plan" tab) and Node's test runner.
 *
 * A "holding" is { playerId, inXI, isCaptain, isVice } -- the draft's own per-player record.
 * A "player" is the app_players.json shape merged in: { id, position, team, price, ... }.
 * Position strings match app_players.json's own convention: "GKP" | "DEF" | "MID" | "FWD".
 *
 * Rules mirrored here are real, standard FPL rules (not invented): a 15-man squad is a FIXED
 * 2 GKP / 5 DEF / 5 MID / 3 FWD composition (not just "15 players, any mix"), max 3 players
 * from any one real-life club, and a starting XI of exactly 11 with 1 GKP, 3-5 DEF, 2-5 MID,
 * 1-3 FWD (the rest of the 15 sit on a 4-man bench). Budget can never go negative.
 */
(function (root) {
  "use strict";

  var POSITIONS = ["GKP", "DEF", "MID", "FWD"];
  var SQUAD_COMPOSITION = { GKP: 2, DEF: 5, MID: 5, FWD: 3 };
  var SQUAD_SIZE = 15;
  var STARTING_XI_SIZE = 11;
  var BENCH_SIZE = SQUAD_SIZE - STARTING_XI_SIZE;
  var MAX_PER_CLUB = 3;
  var FORMATION_LIMITS = {
    GKP: { min: 1, max: 1 },
    DEF: { min: 3, max: 5 },
    MID: { min: 2, max: 5 },
    FWD: { min: 1, max: 3 },
  };
  var BUDGET_EPSILON = 1e-6;

  /** Resolves each holding to its player record. Returns { resolved, missingIds } rather than
   * throwing -- a stale/removed player id is a real state a draft can end up in (e.g. a player
   * transferred out of the league), and callers need to be able to surface that as a warning
   * instead of crashing the whole planner. */
  function resolvePlayers(squad, playersById) {
    var resolved = [];
    var missingIds = [];
    for (var i = 0; i < squad.length; i++) {
      var holding = squad[i];
      var player = playersById[holding.playerId];
      if (!player) {
        missingIds.push(holding.playerId);
        continue;
      }
      resolved.push({ holding: holding, player: player });
    }
    return { resolved: resolved, missingIds: missingIds };
  }

  function countByClub(resolved) {
    var counts = {};
    for (var i = 0; i < resolved.length; i++) {
      var club = resolved[i].player.team || "?";
      counts[club] = (counts[club] || 0) + 1;
    }
    return counts;
  }

  function countByPosition(resolved) {
    var counts = { GKP: 0, DEF: 0, MID: 0, FWD: 0 };
    for (var i = 0; i < resolved.length; i++) {
      var pos = resolved[i].player.position;
      if (counts[pos] === undefined) counts[pos] = 0;
      counts[pos]++;
    }
    return counts;
  }

  /** Max 3 players from any one real-life club -- across the whole 15, not just the XI. */
  function validateClubLimits(resolved) {
    var counts = countByClub(resolved);
    var violations = [];
    Object.keys(counts).forEach(function (club) {
      if (counts[club] > MAX_PER_CLUB) {
        violations.push({
          code: "club_limit",
          club: club,
          count: counts[club],
          max: MAX_PER_CLUB,
          message: "Too many selected from " + club + " (" + counts[club] + "/" + MAX_PER_CLUB + ")",
        });
      }
    });
    return { valid: violations.length === 0, violations: violations, counts: counts };
  }

  /** The 15-man squad's fixed composition: exactly 2 GKP, 5 DEF, 5 MID, 3 FWD. */
  function validateSquadComposition(resolved) {
    var counts = countByPosition(resolved);
    var violations = [];
    POSITIONS.forEach(function (pos) {
      var required = SQUAD_COMPOSITION[pos];
      if (counts[pos] !== required) {
        violations.push({
          code: "squad_composition",
          position: pos,
          count: counts[pos],
          required: required,
          message: pos + ": " + counts[pos] + " selected, squad needs exactly " + required,
        });
      }
    });
    return { valid: violations.length === 0, violations: violations, counts: counts };
  }

  function validateSquadSize(squad) {
    var ids = squad.map(function (h) { return h.playerId; });
    var uniqueCount = new Set(ids).size;
    var hasDuplicates = uniqueCount !== squad.length;
    var violations = [];
    if (squad.length !== SQUAD_SIZE) {
      violations.push({
        code: "squad_size",
        count: squad.length,
        required: SQUAD_SIZE,
        message: "Squad has " + squad.length + " players, needs exactly " + SQUAD_SIZE,
      });
    }
    if (hasDuplicates) {
      violations.push({
        code: "duplicate_player",
        message: "The same player appears more than once in the squad",
      });
    }
    return { valid: violations.length === 0, violations: violations };
  }

  /** Starting XI: exactly 11 players, 1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD; the remaining 4 form
   * the bench. Bench-size mismatches are reported here too since they're the direct consequence
   * of an XI that isn't exactly 11 out of a 15-man squad. */
  function validateFormation(resolved) {
    var xi = resolved.filter(function (r) { return r.holding.inXI; });
    var bench = resolved.filter(function (r) { return !r.holding.inXI; });
    var counts = countByPosition(xi);
    var violations = [];

    if (xi.length !== STARTING_XI_SIZE) {
      violations.push({
        code: "xi_size",
        count: xi.length,
        required: STARTING_XI_SIZE,
        message: "Starting XI has " + xi.length + " players, needs exactly " + STARTING_XI_SIZE,
      });
    }
    if (bench.length !== BENCH_SIZE && resolved.length === SQUAD_SIZE) {
      violations.push({
        code: "bench_size",
        count: bench.length,
        required: BENCH_SIZE,
        message: "Bench has " + bench.length + " players, needs exactly " + BENCH_SIZE,
      });
    }
    POSITIONS.forEach(function (pos) {
      var limit = FORMATION_LIMITS[pos];
      var count = counts[pos] || 0;
      if (count < limit.min || count > limit.max) {
        violations.push({
          code: "formation",
          position: pos,
          count: count,
          min: limit.min,
          max: limit.max,
          message: pos + ": " + count + " in starting XI (needs " + limit.min +
            (limit.max !== limit.min ? "-" + limit.max : "") + ")",
        });
      }
    });
    return { valid: violations.length === 0, violations: violations, counts: counts, xiCount: xi.length, benchCount: bench.length };
  }

  /** Bank (remaining budget) can never go negative -- float-safe with a small epsilon so a
   * price like 4.5 + 4.5 - 9.0 doesn't spuriously read as -0.0000000001. */
  function validateBudget(bank) {
    var valid = bank >= -BUDGET_EPSILON;
    return {
      valid: valid,
      bank: bank,
      violations: valid ? [] : [{
        code: "negative_budget",
        bank: bank,
        message: "Budget would go negative (bank " + bank.toFixed(1) + "m)",
      }],
    };
  }

  /** Captain must be a starting XI player; vice-captain must be a different starting XI player.
   * Kept separate from validateDraftSquad()'s blocking checks -- these matter for scoring, not
   * for whether the squad/transfer itself is legal, so a caller can surface them as a distinct,
   * non-blocking warning class if it wants to. */
  function validateCaptaincy(resolved, captainId, viceId) {
    var byId = {};
    resolved.forEach(function (r) { byId[r.holding.playerId] = r; });
    var violations = [];
    if (captainId == null) {
      violations.push({ code: "no_captain", message: "No captain selected" });
    } else if (!byId[captainId] || !byId[captainId].holding.inXI) {
      violations.push({ code: "captain_not_in_xi", message: "Captain must be in the starting XI" });
    }
    if (viceId != null) {
      if (viceId === captainId) {
        violations.push({ code: "vice_equals_captain", message: "Vice-captain must be different from the captain" });
      } else if (!byId[viceId] || !byId[viceId].holding.inXI) {
        violations.push({ code: "vice_not_in_xi", message: "Vice-captain must be in the starting XI" });
      }
    }
    return { valid: violations.length === 0, violations: violations };
  }

  /** The one entry point the UI should call before letting a transfer/lineup change be
   * confirmed. Runs every blocking check (squad size, composition, club limits, formation,
   * budget) and flattens their violations into one warnings array the UI can render directly
   * (each has a ready-made `.message`). `valid` is false if ANY blocking check fails. */
  function validateDraftSquad(squad, playersById, bank) {
    var resolution = resolvePlayers(squad, playersById);
    var warnings = resolution.missingIds.map(function (id) {
      return { code: "missing_player", playerId: id, message: "Player id " + id + " not found in the player pool" };
    });

    var sizeResult = validateSquadSize(squad);
    var compositionResult = validateSquadComposition(resolution.resolved);
    var clubResult = validateClubLimits(resolution.resolved);
    var formationResult = validateFormation(resolution.resolved);
    var budgetResult = validateBudget(bank);

    [sizeResult, compositionResult, clubResult, formationResult, budgetResult].forEach(function (r) {
      warnings = warnings.concat(r.violations);
    });

    return {
      valid: warnings.length === 0,
      warnings: warnings,
      size: sizeResult,
      composition: compositionResult,
      clubLimits: clubResult,
      formation: formationResult,
      budget: budgetResult,
    };
  }

  var api = {
    POSITIONS: POSITIONS,
    SQUAD_COMPOSITION: SQUAD_COMPOSITION,
    SQUAD_SIZE: SQUAD_SIZE,
    STARTING_XI_SIZE: STARTING_XI_SIZE,
    BENCH_SIZE: BENCH_SIZE,
    MAX_PER_CLUB: MAX_PER_CLUB,
    FORMATION_LIMITS: FORMATION_LIMITS,
    resolvePlayers: resolvePlayers,
    countByClub: countByClub,
    countByPosition: countByPosition,
    validateClubLimits: validateClubLimits,
    validateSquadComposition: validateSquadComposition,
    validateSquadSize: validateSquadSize,
    validateFormation: validateFormation,
    validateBudget: validateBudget,
    validateCaptaincy: validateCaptaincy,
    validateDraftSquad: validateDraftSquad,
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    root.PlannerRules = api;
  }
})(typeof self !== "undefined" ? self : this);
