/* planner/model.js
 *
 * The draft data model: an event-sourced reducer over a base squad snapshot. A draft is NOT a
 * stack of 38 independently-stored squads (that would let per-GW state drift out of sync with
 * the transfers that supposedly produced it) -- it's one base state (the real squad at the
 * gameweek the draft was created from) plus a sparse map of per-gameweek "plans" (transfers
 * made, chip played, captain/vice/bench choice). computeStateAtGameweek() replays those plans
 * forward from the base gameweek, so bank/free-transfers/squad at any viewed gameweek is always
 * *derived*, never stored redundantly -- carrying state forward automatically is just what
 * replaying the reducer does, by construction, matching how FPL's own free-transfer/bank
 * mechanics actually work (each gameweek's state is a pure function of the previous one).
 *
 * Real, standard FPL rules encoded here (matching src/fpl_quant/transfer_planner.py's own
 * confirmed-not-assumed constants): free transfers bank up to a cap of 5; each transfer beyond
 * the free allocation costs 4 points; chips come in two sets of 4 (wildcard, free_hit,
 * bench_boost, triple_captain) -- set 1 usable GW1-18, forfeited (not carried over) if unused by
 * the GW19 deadline, set 2 usable GW19-38, one of each type per set. Wildcard/Free Hit relax the
 * free-transfer limit for their own gameweek only (unlimited changes, zero hit cost) and don't
 * consume/grant free transfers beyond the normal weekly +1 accrual.
 *
 * Depends on planner/rules.js (loaded first) for validateDraftSquad -- kept as a peer dependency
 * rather than re-implemented here so squad legality is checked in exactly one place.
 */
(function (root, factory) {
  if (typeof module !== "undefined" && module.exports) {
    module.exports = factory(require("./rules.js"));
  } else {
    root.PlannerModel = factory(root.PlannerRules);
  }
})(typeof self !== "undefined" ? self : this, function (Rules) {
  "use strict";

  var FREE_TRANSFER_CAP = 5;
  var POINTS_PER_HIT = 4;
  var GW19_DEADLINE_GAMEWEEK = 19;
  var FIRST_GAMEWEEK = 1;
  var LAST_GAMEWEEK = 38;
  var ALL_CHIP_TYPES = ["wildcard", "free_hit", "bench_boost", "triple_captain"];
  var UNLIMITED_TRANSFER_CHIPS = { wildcard: true, free_hit: true };
  var MAX_SAVED_DRAFTS = 5;

  function clone(obj) { return JSON.parse(JSON.stringify(obj)); }

  /** A fresh draft, bootstrapped from a real (or another draft's) squad snapshot.
   * `baseSquad` is an array of { playerId, inXI, isCaptain, isVice }; `baseFreeTransfers`
   * defaults to 1 (a fresh account's real starting allocation, same assumption
   * transfer_planner.py's bootstrap makes, documented there rather than invented fresh here). */
  function createDraft(opts) {
    opts = opts || {};
    var now = new Date().toISOString();
    return {
      id: opts.id || ("draft_" + Date.now() + "_" + Math.random().toString(36).slice(2, 8)),
      name: opts.name || "Untitled draft",
      entryId: opts.entryId != null ? opts.entryId : null,
      baseGameweek: opts.baseGameweek || FIRST_GAMEWEEK,
      baseSquad: clone(opts.baseSquad || []),
      baseBank: opts.baseBank != null ? opts.baseBank : 0,
      baseFreeTransfers: opts.baseFreeTransfers != null ? opts.baseFreeTransfers : 1,
      chipsUsedSet1: clone(opts.chipsUsedSet1 || []),
      chipsUsedSet2: clone(opts.chipsUsedSet2 || []),
      gwPlans: clone(opts.gwPlans || {}),
      createdAt: opts.createdAt || now,
      updatedAt: opts.updatedAt || now,
    };
  }

  function chipSetFor(gameweek) {
    return gameweek < GW19_DEADLINE_GAMEWEEK ? "set1" : "set2";
  }

  function emptyPlan() {
    return { transfersOut: [], transfersIn: [], chip: null, captainId: null, viceId: null, benchOrder: null, xiOverrides: {} };
  }

  function planFor(draft, gameweek) {
    return draft.gwPlans[String(gameweek)] || emptyPlan();
  }

  /** Every gameweek from the draft's base up to (and including) `gameweek`, in order --
   * the exact replay sequence computeStateAtGameweek() walks. */
  function gameweekSequence(draft, gameweek) {
    var seq = [];
    for (var gw = draft.baseGameweek; gw <= gameweek; gw++) seq.push(gw);
    return seq;
  }

  /** Applies one gameweek's plan on top of a running { squad, bank } to produce the squad/bank
   * that carries into the NEXT gameweek, plus this gameweek's own transfer-cost/chip bookkeeping.
   * `squad` is the array of holdings surviving from the previous gameweek. */
  function applyPlanToState(squad, bank, freeTransfersAvailable, plan, playersById) {
    var squadById = {};
    squad.forEach(function (h) { squadById[h.playerId] = h; });

    var chipActive = plan.chip;
    var unlimitedTransfers = chipActive && UNLIMITED_TRANSFER_CHIPS[chipActive];
    var transfersMade = plan.transfersOut.length;

    // Apply the swaps in order: each transfersOut[i] leaves, transfersIn[i] arrives in the same
    // squad slot (inXI carried over from the departing player, so a starter is replaced by a
    // starter and a bench player by a bench player, matching how a real FPL transfer works).
    var nextSquad = squad.slice();
    var bankDelta = 0;
    for (var i = 0; i < transfersMade; i++) {
      var outId = plan.transfersOut[i];
      var inId = plan.transfersIn[i];
      var outIdx = -1;
      for (var j = 0; j < nextSquad.length; j++) {
        if (nextSquad[j].playerId === outId) { outIdx = j; break; }
      }
      var outHolding = outIdx >= 0 ? nextSquad[outIdx] : { playerId: outId, inXI: true, isCaptain: false, isVice: false };
      var outPlayer = playersById[outId];
      var inPlayer = playersById[inId];
      var priceOut = outPlayer ? outPlayer.price : 0;
      var priceIn = inPlayer ? inPlayer.price : 0;
      bankDelta += priceOut - priceIn;
      var newHolding = {
        playerId: inId,
        inXI: outHolding.inXI,
        isCaptain: outHolding.isCaptain,
        isVice: outHolding.isVice,
      };
      if (outIdx >= 0) nextSquad[outIdx] = newHolding;
      else nextSquad.push(newHolding);
    }

    var freeUsed = unlimitedTransfers ? 0 : Math.min(transfersMade, freeTransfersAvailable);
    var hits = unlimitedTransfers ? 0 : Math.max(0, transfersMade - freeTransfersAvailable);
    var transferCost = hits * POINTS_PER_HIT;

    // Real FPL rule (see transfer_planner.py's own documented fix for this exact bug): a new
    // free transfer accrues every gameweek regardless of whether one was used, capped at 5.
    // A hit transfer (paid because the bank was already empty) does NOT draw the bank down
    // further -- only genuinely-free transfers consume from it.
    var nextFreeTransfers = unlimitedTransfers
      ? Math.min(FREE_TRANSFER_CAP, freeTransfersAvailable + 1)
      : Math.min(FREE_TRANSFER_CAP, Math.max(0, freeTransfersAvailable - freeUsed) + 1);

    // Captain/vice overrides for this gameweek, applied on top of the post-transfer squad (a
    // transfer can hand a fresh player the armband the same week it arrives).
    if (plan.captainId != null || plan.viceId != null) {
      nextSquad = nextSquad.map(function (h) {
        return {
          playerId: h.playerId,
          inXI: h.inXI,
          isCaptain: plan.captainId != null ? h.playerId === plan.captainId : h.isCaptain,
          isVice: plan.viceId != null ? h.playerId === plan.viceId : h.isVice,
        };
      });
    }

    // Starting-XI/bench overrides -- e.g. a drag-and-drop swap of two squad members' XI
    // status. Explicit per-player, not a swap primitive at this layer: swapXIStatus() below
    // computes both sides' target inXI value before calling here, so this stays a simple,
    // order-independent "set inXI for these player ids" apply, the same shape captain/vice
    // overrides already use. Like captain/vice, once applied it's baked into nextSquad and so
    // carries forward into future gameweeks by default (a lineup choice persists until
    // explicitly changed again), matching real FPL behavior.
    if (plan.xiOverrides && Object.keys(plan.xiOverrides).length) {
      nextSquad = nextSquad.map(function (h) {
        var override = plan.xiOverrides[h.playerId];
        return override === undefined ? h : {
          playerId: h.playerId, inXI: override, isCaptain: h.isCaptain, isVice: h.isVice,
        };
      });
    }

    return {
      squad: nextSquad,
      bank: bank + bankDelta,
      freeTransfersAvailable: nextFreeTransfers,
      freeTransfersUsedThisGw: freeUsed,
      transfersMadeThisGw: transfersMade,
      transferCostThisGw: transferCost,
      chipActive: chipActive,
      unlimitedTransfers: !!unlimitedTransfers,
    };
  }

  /** The core query: what does this draft's squad/bank/free-transfers/chip state look like at
   * a given gameweek, given everything planned from the draft's base gameweek up to it? Pure
   * function of the draft object -- never mutates it. */
  function computeStateAtGameweek(draft, gameweek, playersById) {
    var squad = clone(draft.baseSquad);
    var bank = draft.baseBank;
    var freeTransfers = draft.baseFreeTransfers;
    var chipsUsedSet1 = draft.chipsUsedSet1.slice();
    var chipsUsedSet2 = draft.chipsUsedSet2.slice();
    var cumulativeTransferCost = 0;
    // "freeTransfersAvailable" must report the count GOING INTO the viewed gameweek (what a
    // manager would see in FPL's own UI before making that week's transfers), not the count
    // accrued for the week AFTER it -- captured each iteration before that iteration's own
    // consumption/accrual is folded into `freeTransfers` for the next step.
    var reportedFreeTransfers = freeTransfers;
    var transferCostThisGw = 0;
    var transfersMadeThisGw = 0;
    var activeChip = null;
    var unlimitedTransfersThisGw = false;

    var sequence = gameweekSequence(draft, gameweek);
    for (var i = 0; i < sequence.length; i++) {
      var gw = sequence[i];
      var plan = planFor(draft, gw);
      reportedFreeTransfers = freeTransfers;
      var stepResult = applyPlanToState(squad, bank, freeTransfers, plan, playersById);
      squad = stepResult.squad;
      bank = stepResult.bank;
      freeTransfers = stepResult.freeTransfersAvailable;
      cumulativeTransferCost += stepResult.transferCostThisGw;
      transferCostThisGw = stepResult.transferCostThisGw;
      transfersMadeThisGw = stepResult.transfersMadeThisGw;
      activeChip = stepResult.chipActive;
      unlimitedTransfersThisGw = stepResult.unlimitedTransfers;
      if (plan.chip) {
        if (chipSetFor(gw) === "set1") chipsUsedSet1.push(plan.chip);
        else chipsUsedSet2.push(plan.chip);
      }
    }

    return {
      gameweek: gameweek,
      squad: squad,
      bank: bank,
      freeTransfersAvailable: reportedFreeTransfers,
      transferCostThisGw: transferCostThisGw,
      transfersMadeThisGw: transfersMadeThisGw,
      cumulativeTransferCost: cumulativeTransferCost,
      activeChip: activeChip,
      unlimitedTransfersThisGw: unlimitedTransfersThisGw,
      chipsUsedSet1: chipsUsedSet1,
      chipsUsedSet2: chipsUsedSet2,
    };
  }

  /** Squad value = sum of each held player's current price (what selling them all would return
   * before accounting for bank). "Spending power" = bank + squad value, the number a manager
   * actually has to work with when shopping for a replacement. */
  function squadValue(squad, playersById) {
    return squad.reduce(function (sum, h) {
      var p = playersById[h.playerId];
      return sum + (p ? p.price : 0);
    }, 0);
  }
  function spendingPower(squad, bank, playersById) {
    return bank + squadValue(squad, playersById);
  }

  function mutatePlan(draft, gameweek, patch) {
    var key = String(gameweek);
    var next = clone(draft);
    var current = next.gwPlans[key] || emptyPlan();
    next.gwPlans[key] = Object.assign({}, current, patch);
    next.updatedAt = new Date().toISOString();
    return next;
  }

  /** Records a transfer (out -> in) for `gameweek`, appended after any transfers already
   * recorded there this session. Does NOT validate legality itself -- callers should run the
   * resulting computeStateAtGameweek() squad through Rules.validateDraftSquad() and surface
   * warnings before letting the change be "confirmed" (kept separate so the reducer stays a
   * pure state-transition function, and the UI decides what "confirm" means). */
  function applyTransfer(draft, gameweek, outId, inId) {
    var current = planFor(draft, gameweek);
    return mutatePlan(draft, gameweek, {
      transfersOut: current.transfersOut.concat([outId]),
      transfersIn: current.transfersIn.concat([inId]),
    });
  }

  /** Removes the transfer at `index` (0-based, in the order it was made) for `gameweek`. */
  function undoTransfer(draft, gameweek, index) {
    var current = planFor(draft, gameweek);
    var transfersOut = current.transfersOut.slice();
    var transfersIn = current.transfersIn.slice();
    transfersOut.splice(index, 1);
    transfersIn.splice(index, 1);
    return mutatePlan(draft, gameweek, { transfersOut: transfersOut, transfersIn: transfersIn });
  }

  function clearTransfers(draft, gameweek) {
    return mutatePlan(draft, gameweek, { transfersOut: [], transfersIn: [] });
  }

  /** Whether `chipType` can legally be assigned to `gameweek` given what's already used.
   * Returns { allowed, reason } rather than throwing -- the UI needs a reason string to show
   * ("Already used this season", "Wildcard set 1 is closed after GW18") more than it needs an
   * exception. */
  function canAssignChip(draft, gameweek, chipType) {
    if (ALL_CHIP_TYPES.indexOf(chipType) < 0) {
      return { allowed: false, reason: "Unknown chip type" };
    }
    var targetSet = chipSetFor(gameweek);
    var usedThisSet = targetSet === "set1" ? draft.chipsUsedSet1 : draft.chipsUsedSet2;

    // A chip already spent EARLIER in this draft's own timeline (in a prior planned gameweek,
    // not yet reflected in draft.chipsUsedSet*) also blocks re-assignment -- scan every plan in
    // the same chip-set window up to (but not including) `gameweek`.
    var sequence = gameweekSequence(draft, gameweek - 1);
    var usedInWindowSoFar = usedThisSet.slice();
    sequence.forEach(function (gw) {
      if (chipSetFor(gw) !== targetSet) return;
      var plan = planFor(draft, gw);
      if (plan.chip) usedInWindowSoFar.push(plan.chip);
    });

    if (usedInWindowSoFar.indexOf(chipType) >= 0) {
      return {
        allowed: false,
        reason: targetSet === "set1"
          ? (chipType + " already used in the first half of the season (before GW19)")
          : (chipType + " already used in the second half of the season"),
      };
    }
    // Note: there's no separate "set 1 is closed" guard here beyond chipSetFor()'s own
    // gameweek < GW19 split -- GW19 itself already resolves to "set2" above, so assigning a
    // chip type AT GW19 draws from the fresh second-half allocation, not a blocked first-half
    // one. An unused set-1 chip isn't blocked from being assigned here; it's simply gone (see
    // checkGw19Deadline()'s forfeitedNow flag) -- there's nothing left in set1 to assign once
    // gameweek >= GW19.
    return { allowed: true, reason: null };
  }

  function setChip(draft, gameweek, chipType) {
    if (chipType != null) {
      var check = canAssignChip(draft, gameweek, chipType);
      if (!check.allowed) {
        throw new Error("Cannot assign " + chipType + " to GW" + gameweek + ": " + check.reason);
      }
    }
    return mutatePlan(draft, gameweek, { chip: chipType });
  }

  /** use-it-or-lose-it flag for chip set 1, mirroring transfer_planner.py's own
   * check_gw19_deadline(): urgent inside the last `warningWindow` gameweeks before GW19 with
   * unused set-1 chips still on the table; forfeited_now once GW19 itself has arrived. */
  function checkGw19Deadline(currentGameweek, chipsUsedSet1, warningWindow) {
    warningWindow = warningWindow || 3;
    var unused = ALL_CHIP_TYPES.filter(function (c) { return chipsUsedSet1.indexOf(c) < 0; });
    var gameweeksRemaining = GW19_DEADLINE_GAMEWEEK - currentGameweek;
    return {
      unusedSet1Chips: unused,
      gameweeksUntilGw19: gameweeksRemaining,
      urgent: gameweeksRemaining >= 1 && gameweeksRemaining <= warningWindow && unused.length > 0,
      forfeitedNow: currentGameweek >= GW19_DEADLINE_GAMEWEEK && unused.length > 0,
    };
  }

  function setCaptain(draft, gameweek, playerId) {
    return mutatePlan(draft, gameweek, { captainId: playerId });
  }
  function setVice(draft, gameweek, playerId) {
    return mutatePlan(draft, gameweek, { viceId: playerId });
  }
  function setBenchOrder(draft, gameweek, orderedPlayerIds) {
    return mutatePlan(draft, gameweek, { benchOrder: orderedPlayerIds });
  }

  /** Sets explicit starting-XI/bench status for one or more players at `gameweek` -- merged
   * on top of whatever's already overridden that week, not replacing it wholesale, so several
   * separate drag-and-drop swaps in the same session compose correctly. `overrides` is a plain
   * { playerId: boolean } map (true = starting, false = benched). */
  function setXIStatus(draft, gameweek, overrides) {
    var current = planFor(draft, gameweek);
    var merged = Object.assign({}, current.xiOverrides, overrides);
    return mutatePlan(draft, gameweek, { xiOverrides: merged });
  }

  /** The drag-and-drop lineup swap: exchanges two squad members' starting-XI/bench status at
   * `gameweek` (e.g. dragging a bench player onto a starting slot). Reads each player's CURRENT
   * inXI value at that gameweek (via computeStateAtGameweek, so it's correct even after prior
   * transfers/overrides) rather than assuming the caller already knows it. Both ids must
   * already be in the squad at that gameweek -- throws rather than silently no-opping,
   * matching setChip()'s own fail-fast convention for an invalid request. */
  function swapXIStatus(draft, gameweek, playersById, playerIdA, playerIdB) {
    var state = computeStateAtGameweek(draft, gameweek, playersById);
    var holdingA = state.squad.find(function (h) { return h.playerId === playerIdA; });
    var holdingB = state.squad.find(function (h) { return h.playerId === playerIdB; });
    if (!holdingA || !holdingB) {
      throw new Error("swapXIStatus: both players must already be in the squad at GW" + gameweek);
    }
    var overrides = {};
    overrides[playerIdA] = holdingB.inXI;
    overrides[playerIdB] = holdingA.inXI;
    return setXIStatus(draft, gameweek, overrides);
  }
  function renameDraft(draft, name) {
    var next = clone(draft);
    next.name = name;
    next.updatedAt = new Date().toISOString();
    return next;
  }

  var api = {
    FREE_TRANSFER_CAP: FREE_TRANSFER_CAP,
    POINTS_PER_HIT: POINTS_PER_HIT,
    GW19_DEADLINE_GAMEWEEK: GW19_DEADLINE_GAMEWEEK,
    FIRST_GAMEWEEK: FIRST_GAMEWEEK,
    LAST_GAMEWEEK: LAST_GAMEWEEK,
    ALL_CHIP_TYPES: ALL_CHIP_TYPES,
    MAX_SAVED_DRAFTS: MAX_SAVED_DRAFTS,
    createDraft: createDraft,
    chipSetFor: chipSetFor,
    computeStateAtGameweek: computeStateAtGameweek,
    squadValue: squadValue,
    spendingPower: spendingPower,
    applyTransfer: applyTransfer,
    undoTransfer: undoTransfer,
    clearTransfers: clearTransfers,
    canAssignChip: canAssignChip,
    setChip: setChip,
    checkGw19Deadline: checkGw19Deadline,
    setCaptain: setCaptain,
    setVice: setVice,
    setBenchOrder: setBenchOrder,
    setXIStatus: setXIStatus,
    swapXIStatus: swapXIStatus,
    renameDraft: renameDraft,
    planFor: planFor,
  };

  return api;
});
