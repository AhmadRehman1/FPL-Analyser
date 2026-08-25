/* planner/solver.js
 *
 * Transfer suggestion engine -- v1 is deliberately a greedy single-transfer search (per the
 * task's own staged scope: "start simple... later become a more exhaustive combinatorial
 * search"), not a full multi-transfer optimizer. Mirrors the shape of the existing backend's
 * search_single_transfer() (src/fpl_quant/transfer_planner.py): for each owned, unlocked
 * player, find the best same-position replacement the manager can actually afford, ranked by
 * (summed EP gain over the horizon) minus (transfer-hit cost), using real per-player xPts from
 * data/dashboard/projections_latest.json when it's available.
 *
 * Pure and DOM-free: takes plain data in, returns plain data out, so it's testable without a
 * browser and swappable later for a wider search without touching any UI code.
 */
(function (root, factory) {
  if (typeof module !== "undefined" && module.exports) {
    module.exports = factory();
  } else {
    root.PlannerSolver = factory();
  }
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  var POINTS_PER_HIT = 4;

  /** Builds a lookup of playerUid/elementId -> { gw: ep } from projections_latest.json's own
   * `players` array (each row already carries per-gameweek EP bands -- see
   * scripts/export_projections.py's ProjectionRow.ep_per_gw). Keyed by `fpl_element_id` since
   * that's the id space app_players.json (and therefore this planner's squads) uses; a row
   * with no resolved element id is skipped (can't be joined to a squad player either way, same
   * "disclosed, not guessed" convention export_projections.py itself documents). */
  function projectionsByElementId(projectionsPayload) {
    var byElement = {};
    if (!projectionsPayload || !projectionsPayload.players) return byElement;
    projectionsPayload.players.forEach(function (row) {
      if (row.fpl_element_id == null) return;
      var byGw = {};
      (row.ep_per_gw || []).forEach(function (band) { byGw[band.gw] = band.ep; });
      byElement[row.fpl_element_id] = byGw;
    });
    return byElement;
  }

  /** Summed projected points for one player across `gameweeks` -- 0 for any gameweek with no
   * projection (e.g. a blank gameweek), never guessed. */
  function horizonEp(projByElement, elementId, gameweeks) {
    var byGw = projByElement[elementId];
    if (!byGw) return 0;
    return gameweeks.reduce(function (sum, gw) { return sum + (byGw[gw] || 0); }, 0);
  }

  /** Whether real projection data actually covers this player pool at all -- lets the UI show
   * "no projections published yet" instead of a silently-zero suggestion list. */
  function hasProjections(projByElement) {
    return Object.keys(projByElement).length > 0;
  }

  /** Ranks every legal same-position replacement for one outgoing player by net horizon value
   * (EP gain minus this single transfer's own hit cost, i.e. treating it as the one extra
   * transfer beyond whatever's already free). `candidates` is the full player pool
   * (app_players.json rows); `ownedIds`/`ignoredIds` are Sets of player ids. */
  function candidatesForSlot(outPlayer, candidates, opts) {
    var ownedIds = opts.ownedIds || new Set();
    var ignoredIds = opts.ignoredIds || new Set();
    var bank = opts.bank || 0;
    var affordableUpTo = bank + outPlayer.price;
    return candidates.filter(function (p) {
      return p.position === outPlayer.position
        && p.id !== outPlayer.id
        && !ownedIds.has(p.id)
        && !ignoredIds.has(p.id)
        && p.price <= affordableUpTo + 1e-9;
    });
  }

  /** The greedy single-transfer suggester: for each owned, non-locked player, find their single
   * best replacement, then rank those suggestions across the whole squad. `state` is a
   * computeStateAtGameweek()-shaped object ({ squad, bank }); `opts.horizonGameweeks` is the
   * list of future gameweeks to sum EP over (e.g. the next 5); `opts.lockedIds`/`ignoredIds`
   * are Sets; `opts.freeTransfersAvailable` decides whether the top suggestion is actually free
   * or a -4 hit. Returns suggestions sorted best-first, capped at `opts.maxSuggestions` (default
   * 10). Each suggestion never includes a hit unless the net EP gain (before the hit) still
   * beats zero, matching the same "isn't the hit itself the point" discipline the backend's own
   * search applies. */
  function suggestTransfers(state, playersById, candidates, projByElement, opts) {
    opts = opts || {};
    var horizonGameweeks = opts.horizonGameweeks || [];
    var lockedIds = opts.lockedIds || new Set();
    var ignoredIds = opts.ignoredIds || new Set();
    var freeTransfersAvailable = opts.freeTransfersAvailable != null ? opts.freeTransfersAvailable : 0;
    var maxSuggestions = opts.maxSuggestions || 10;
    var hitCost = freeTransfersAvailable >= 1 ? 0 : POINTS_PER_HIT;

    var ownedIds = new Set(state.squad.map(function (h) { return h.playerId; }));
    var suggestions = [];

    state.squad.forEach(function (holding) {
      if (lockedIds.has(holding.playerId)) return;
      var outPlayer = playersById[holding.playerId];
      if (!outPlayer) return;
      var outEp = horizonEp(projByElement, outPlayer.id, horizonGameweeks);
      var pool = candidatesForSlot(outPlayer, candidates, { ownedIds: ownedIds, ignoredIds: ignoredIds, bank: state.bank });

      var best = null;
      pool.forEach(function (inPlayer) {
        var inEp = horizonEp(projByElement, inPlayer.id, horizonGameweeks);
        var gain = inEp - outEp;
        var net = gain - hitCost;
        if (!best || net > best.net) {
          best = {
            outId: outPlayer.id, outName: outPlayer.web_name,
            inId: inPlayer.id, inName: inPlayer.web_name,
            gain: gain, cost: hitCost, net: net,
            priceDelta: inPlayer.price - outPlayer.price,
          };
        }
      });
      if (best && best.net > 0) suggestions.push(best);
    });

    suggestions.sort(function (a, b) { return b.net - a.net; });
    return suggestions.slice(0, maxSuggestions);
  }

  // ==========================================================================================
  // Bounded 2-for-2 combinatorial multi-transfer search -- mirrors
  // src/fpl_quant/transfer_planner.py's own evaluate_multi_transfers() design (same bound,
  // same reasoning for it) rather than inventing a separate approach for the frontend. Full
  // combinatorial explosion at 15 held players x a ~600-player pool is C(15,2)*C(600,2) ~=
  // 18.9 million outgoing/incoming pairings before even checking position/budget/club-cap
  // feasibility -- real, but far too slow to run synchronously on a click. Bounding the
  // INCOMING side to the top candidatePoolLimitPerPosition players per position by horizon EP
  // (a genuinely low-EP candidate essentially never appears in an EP-maximizing combo) reduces
  // this to a tractable few hundred thousand combinations. The one real gap this bound
  // accepts, same as the Python version: a specifically cheap, low-EP player that's the only
  // way to afford another leg of the same combo within budget can be pruned away by a pure
  // EP-ranked top-K. Stopped at 2-for-2 for the same reason the backend stops there: C(15,3)*
  // C(4K,3) runs into the billions even after pruning, and real FPL play essentially never
  // makes 3+ simultaneous incremental transfers outside a Wildcard/Free Hit (a full-squad
  // rebuild, not an incremental combo -- out of this module's scope).
  // ==========================================================================================

  var DEFAULT_CANDIDATE_POOL_LIMIT_PER_POSITION = 20;
  var DEFAULT_MAX_CLUB_COUNT = 3;

  /** Top-K candidates PER POSITION by horizon EP, among players not owned/locked/ignored --
   * the incoming side of a multi-transfer combo search. Exposed on its own (not just inlined
   * into suggestMultiTransfers) so its bounding behavior is directly testable. */
  function boundedIncomingPool(candidates, projByElement, horizonGameweeks, opts) {
    opts = opts || {};
    var excludeIds = opts.excludeIds || new Set();
    var limit = opts.candidatePoolLimitPerPosition || DEFAULT_CANDIDATE_POOL_LIMIT_PER_POSITION;
    var byPosition = {};
    candidates.forEach(function (p) {
      if (excludeIds.has(p.id)) return;
      var withEp = { player: p, ep: horizonEp(projByElement, p.id, horizonGameweeks) };
      (byPosition[p.position] = byPosition[p.position] || []).push(withEp);
    });
    var bounded = [];
    Object.keys(byPosition).forEach(function (pos) {
      byPosition[pos].sort(function (a, b) { return b.ep - a.ep; });
      bounded = bounded.concat(byPosition[pos].slice(0, limit));
    });
    return bounded;
  }

  /** All (i, j) pairs of a list with i < j -- the JS equivalent of Python's
   * itertools.combinations(list, 2), used both for the outgoing (owned) side and, when both
   * legs share a position, the incoming side too. */
  function pairs(list) {
    var out = [];
    for (var i = 0; i < list.length; i++) {
      for (var j = i + 1; j < list.length; j++) out.push([list[i], list[j]]);
    }
    return out;
  }

  /** The 2-for-2 search itself: sell any 2 currently-held, unlocked players; buy any 2 real,
   * unignored candidates whose combined POSITION MULTISET exactly matches the two sold (squad
   * position quotas are fixed, so a legal swap's incoming positions must balance the outgoing
   * ones as a set, not player-for-player by a specific pairing) -- ranked by combined net value
   * over the horizon, exactly like suggestTransfers() but for the combination as a whole, not
   * two independent single transfers. `state`/`playersById`/`candidates`/`projByElement` and
   * `opts.horizonGameweeks`/`lockedIds`/`ignoredIds`/`freeTransfersAvailable` match
   * suggestTransfers()'s own signature; `opts.maxClubCount` defaults to the real FPL cap (3),
   * `opts.candidatePoolLimitPerPosition` to 20, `opts.maxSuggestions` to 10. */
  function suggestMultiTransfers(state, playersById, candidates, projByElement, opts) {
    opts = opts || {};
    var horizonGameweeks = opts.horizonGameweeks || [];
    var lockedIds = opts.lockedIds || new Set();
    var ignoredIds = opts.ignoredIds || new Set();
    var freeTransfersAvailable = opts.freeTransfersAvailable != null ? opts.freeTransfersAvailable : 0;
    var maxClubCount = opts.maxClubCount || DEFAULT_MAX_CLUB_COUNT;
    var maxSuggestions = opts.maxSuggestions || 10;
    var bank = state.bank || 0;

    // Exactly 2 transfers are made in any combo this function considers; a hit is only
    // charged for transfers beyond whatever's currently free, same max(0, n - free) rule
    // suggestTransfers() applies at n=1, generalized here at n=2.
    var hits = Math.max(0, 2 - freeTransfersAvailable);
    var transferCost = hits * POINTS_PER_HIT;

    var ownedIds = new Set(state.squad.map(function (h) { return h.playerId; }));
    var heldClubCounts = {};
    state.squad.forEach(function (h) {
      var p = playersById[h.playerId];
      if (p) heldClubCounts[p.team] = (heldClubCounts[p.team] || 0) + 1;
    });

    var sellable = state.squad
      .filter(function (h) { return !lockedIds.has(h.playerId); })
      .map(function (h) { return playersById[h.playerId]; })
      .filter(Boolean)
      .map(function (p) { return { player: p, ep: horizonEp(projByElement, p.id, horizonGameweeks) }; });

    var incoming = boundedIncomingPool(candidates, projByElement, horizonGameweeks, {
      excludeIds: new Set([...ownedIds, ...ignoredIds]),
      candidatePoolLimitPerPosition: opts.candidatePoolLimitPerPosition,
    });
    var incomingByPosition = {};
    incoming.forEach(function (c) { (incomingByPosition[c.player.position] = incomingByPosition[c.player.position] || []).push(c); });

    var results = [];
    pairs(sellable).forEach(function (outPair) {
      var outA = outPair[0], outB = outPair[1];
      var combinedPriceOut = outA.player.price + outB.player.price;
      var combinedEpOut = outA.ep + outB.ep;
      var posSorted = [outA.player.position, outB.player.position].sort();

      var inPairs;
      if (posSorted[0] === posSorted[1]) {
        inPairs = pairs(incomingByPosition[posSorted[0]] || []);
      } else {
        inPairs = [];
        (incomingByPosition[posSorted[0]] || []).forEach(function (x) {
          (incomingByPosition[posSorted[1]] || []).forEach(function (y) { inPairs.push([x, y]); });
        });
      }

      inPairs.forEach(function (inPair) {
        var inX = inPair[0], inY = inPair[1];
        var combinedPriceIn = inX.player.price + inY.player.price;
        if (combinedPriceIn > combinedPriceOut + bank + 1e-9) return;

        var deltas = {};
        [outA.player, outB.player].forEach(function (p) { deltas[p.team] = (deltas[p.team] || 0) - 1; });
        [inX.player, inY.player].forEach(function (p) { deltas[p.team] = (deltas[p.team] || 0) + 1; });
        var breaksClubCap = Object.keys(deltas).some(function (club) {
          return (heldClubCounts[club] || 0) + deltas[club] > maxClubCount;
        });
        if (breaksClubCap) return;

        var combinedEpIn = inX.ep + inY.ep;
        var gain = combinedEpIn - combinedEpOut;
        var net = gain - transferCost;
        if (net <= 0) return;
        results.push({
          outIds: [outA.player.id, outB.player.id], outNames: [outA.player.web_name, outB.player.web_name],
          inIds: [inX.player.id, inY.player.id], inNames: [inX.player.web_name, inY.player.web_name],
          combinedPriceOut: combinedPriceOut, combinedPriceIn: combinedPriceIn,
          gain: gain, cost: transferCost, net: net,
        });
      });
    });

    results.sort(function (a, b) { return b.net - a.net; });
    return results.slice(0, maxSuggestions);
  }

  return {
    POINTS_PER_HIT: POINTS_PER_HIT,
    projectionsByElementId: projectionsByElementId,
    horizonEp: horizonEp,
    hasProjections: hasProjections,
    candidatesForSlot: candidatesForSlot,
    suggestTransfers: suggestTransfers,
    boundedIncomingPool: boundedIncomingPool,
    suggestMultiTransfers: suggestMultiTransfers,
  };
});
