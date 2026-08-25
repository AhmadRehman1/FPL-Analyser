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

  return {
    POINTS_PER_HIT: POINTS_PER_HIT,
    projectionsByElementId: projectionsByElementId,
    horizonEp: horizonEp,
    hasProjections: hasProjections,
    candidatesForSlot: candidatesForSlot,
    suggestTransfers: suggestTransfers,
  };
});
