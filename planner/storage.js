/* planner/storage.js
 *
 * localStorage persistence for saved drafts, capped at 5 (MAX_SAVED_DRAFTS). Per the project's
 * own design note (index.html's "local-only team state" section): this app has no accounts
 * system, so drafts live entirely client-side, same convention as the existing
 * fq_local_team_<accountId> / fq_custom_accounts keys -- never submitted anywhere.
 *
 * The storage backend is injected (not read from a global `localStorage`) so this module is
 * testable in Node without a DOM: pass `window.localStorage` from the browser, or any object
 * implementing getItem/setItem/removeItem in tests.
 */
(function (root, factory) {
  if (typeof module !== "undefined" && module.exports) {
    module.exports = factory();
  } else {
    root.PlannerStorage = factory();
  }
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  var STORAGE_KEY = "fq_planner_drafts_v1";
  var MAX_SAVED_DRAFTS = 5;

  function loadAll(storage) {
    try {
      var raw = storage.getItem(STORAGE_KEY);
      var parsed = raw ? JSON.parse(raw) : {};
      return parsed && typeof parsed === "object" ? parsed : {};
    } catch (e) {
      return {};
    }
  }

  function saveAll(storage, draftsById) {
    storage.setItem(STORAGE_KEY, JSON.stringify(draftsById));
  }

  /** All saved drafts, most-recently-updated first. */
  function listDrafts(storage) {
    var all = loadAll(storage);
    return Object.keys(all)
      .map(function (id) { return all[id]; })
      .sort(function (a, b) { return new Date(b.updatedAt) - new Date(a.updatedAt); });
  }

  function getDraft(storage, id) {
    var all = loadAll(storage);
    return all[id] || null;
  }

  /** Saves (creates or updates) a draft. A brand-new draft is rejected once 5 are already
   * saved -- returns { ok: false, reason } instead of silently evicting the oldest one, since
   * silently losing a saved plan is exactly the kind of surprise this module exists to avoid.
   * Updating an EXISTING draft (its id already present) is always allowed, even at the cap. */
  function saveDraft(storage, draft) {
    var all = loadAll(storage);
    var isNew = !all[draft.id];
    if (isNew && Object.keys(all).length >= MAX_SAVED_DRAFTS) {
      return {
        ok: false,
        reason: "You already have " + MAX_SAVED_DRAFTS + " saved drafts (the maximum) -- delete one before saving a new one.",
      };
    }
    var next = Object.assign({}, draft, { updatedAt: new Date().toISOString() });
    all[draft.id] = next;
    saveAll(storage, all);
    return { ok: true, draft: next };
  }

  function deleteDraft(storage, id) {
    var all = loadAll(storage);
    if (!all[id]) return { ok: false, reason: "Draft not found" };
    delete all[id];
    saveAll(storage, all);
    return { ok: true };
  }

  /** Duplicates a saved draft under a new id/name -- subject to the same 5-draft cap as any
   * other new draft (it's a new entry, not an update). */
  function duplicateDraft(storage, id, newName) {
    var source = getDraft(storage, id);
    if (!source) return { ok: false, reason: "Draft not found" };
    var copy = Object.assign({}, source, {
      id: "draft_" + Date.now() + "_" + Math.random().toString(36).slice(2, 8),
      name: newName || (source.name + " (copy)"),
      createdAt: new Date().toISOString(),
    });
    return saveDraft(storage, copy);
  }

  function draftCount(storage) {
    return Object.keys(loadAll(storage)).length;
  }

  return {
    STORAGE_KEY: STORAGE_KEY,
    MAX_SAVED_DRAFTS: MAX_SAVED_DRAFTS,
    listDrafts: listDrafts,
    getDraft: getDraft,
    saveDraft: saveDraft,
    deleteDraft: deleteDraft,
    duplicateDraft: duplicateDraft,
    draftCount: draftCount,
  };
});
