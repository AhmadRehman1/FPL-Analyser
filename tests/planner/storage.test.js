"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const Storage = require("../../planner/storage.js");
const Model = require("../../planner/model.js");

/** Minimal in-memory stand-in for the browser's localStorage. */
function makeMemoryStorage() {
  const map = new Map();
  return {
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => map.set(k, String(v)),
    removeItem: (k) => map.delete(k),
  };
}

function draft(name) {
  return Model.createDraft({ name, baseGameweek: 2, baseSquad: [], baseBank: 0, baseFreeTransfers: 1 });
}

test("saveDraft then getDraft round-trips", () => {
  const storage = makeMemoryStorage();
  const d = draft("My plan");
  const result = Storage.saveDraft(storage, d);
  assert.equal(result.ok, true);
  const loaded = Storage.getDraft(storage, d.id);
  assert.equal(loaded.name, "My plan");
});

test("listDrafts returns most-recently-updated first", async () => {
  const storage = makeMemoryStorage();
  const d1 = draft("First");
  Storage.saveDraft(storage, d1);
  await new Promise((r) => setTimeout(r, 2));
  const d2 = draft("Second");
  Storage.saveDraft(storage, d2);
  const list = Storage.listDrafts(storage);
  assert.equal(list[0].name, "Second");
  assert.equal(list[1].name, "First");
});

test("a 6th new draft is rejected once 5 are saved", () => {
  const storage = makeMemoryStorage();
  for (let i = 0; i < 5; i++) {
    const result = Storage.saveDraft(storage, draft("Draft " + i));
    assert.equal(result.ok, true);
  }
  assert.equal(Storage.draftCount(storage), 5);
  const sixth = Storage.saveDraft(storage, draft("Draft 5"));
  assert.equal(sixth.ok, false);
  assert.match(sixth.reason, /5 saved drafts/);
  assert.equal(Storage.draftCount(storage), 5);
});

test("updating an existing draft is allowed even at the 5-draft cap", () => {
  const storage = makeMemoryStorage();
  const drafts = [];
  for (let i = 0; i < 5; i++) {
    const d = draft("Draft " + i);
    Storage.saveDraft(storage, d);
    drafts.push(d);
  }
  const updated = Object.assign({}, drafts[0], { name: "Renamed" });
  const result = Storage.saveDraft(storage, updated);
  assert.equal(result.ok, true);
  assert.equal(Storage.getDraft(storage, drafts[0].id).name, "Renamed");
  assert.equal(Storage.draftCount(storage), 5);
});

test("deleteDraft removes a draft and frees a cap slot", () => {
  const storage = makeMemoryStorage();
  const drafts = [];
  for (let i = 0; i < 5; i++) {
    const d = draft("Draft " + i);
    Storage.saveDraft(storage, d);
    drafts.push(d);
  }
  const del = Storage.deleteDraft(storage, drafts[0].id);
  assert.equal(del.ok, true);
  assert.equal(Storage.draftCount(storage), 4);
  const sixth = Storage.saveDraft(storage, draft("New one"));
  assert.equal(sixth.ok, true);
});

test("duplicateDraft copies under a new id and name", () => {
  const storage = makeMemoryStorage();
  const d = draft("Original");
  Storage.saveDraft(storage, d);
  const result = Storage.duplicateDraft(storage, d.id);
  assert.equal(result.ok, true);
  assert.notEqual(result.draft.id, d.id);
  assert.match(result.draft.name, /copy/);
  assert.equal(Storage.draftCount(storage), 2);
});

test("loadAll tolerates corrupted JSON without throwing", () => {
  const storage = makeMemoryStorage();
  storage.setItem(Storage.STORAGE_KEY, "{not valid json");
  assert.deepEqual(Storage.listDrafts(storage), []);
});
