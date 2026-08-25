"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const Fixtures = require("../../planner/fixtures.js");

function fixturesFixture() {
  return {
    gameweeks: [
      {
        gameweek: 1,
        deadline_time: "2026-08-21T17:30:00Z",
        fixtures: [
          { home: { short_name: "ARS", difficulty: 2 }, away: { short_name: "COV", difficulty: 5 } },
          { home: { short_name: "LIV", difficulty: 3 }, away: { short_name: "MUN", difficulty: 3 } },
        ],
      },
      {
        gameweek: 2,
        deadline_time: "2026-08-28T17:30:00Z",
        fixtures: [
          { home: { short_name: "COV", difficulty: 2 }, away: { short_name: "ARS", difficulty: 4 } },
          { home: { short_name: "MUN", difficulty: 3 }, away: { short_name: "LIV", difficulty: 3 } },
        ],
      },
      {
        gameweek: 3,
        deadline_time: "2026-09-04T17:30:00Z",
        fixtures: [
          { home: { short_name: "ARS", difficulty: 1 }, away: { short_name: "MUN", difficulty: 4 } },
        ],
      },
    ],
  };
}

test("upcomingFixturesForClub returns fixtures in gameweek order with home/away resolved", () => {
  const rows = Fixtures.upcomingFixturesForClub(fixturesFixture(), "ARS", 1, 3);
  assert.equal(rows.length, 3);
  assert.deepEqual(rows.map((r) => r.gameweek), [1, 2, 3]);
  assert.equal(rows[0].home, true);
  assert.equal(rows[0].opponent, "COV");
  assert.equal(rows[1].home, false); // ARS is away in GW2
  assert.equal(rows[1].opponent, "COV");
});

test("upcomingFixturesForClub respects the fromGw/n window", () => {
  const rows = Fixtures.upcomingFixturesForClub(fixturesFixture(), "ARS", 2, 1);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].gameweek, 2);
});

test("upcomingFixturesForClub returns empty for a club with no fixtures in range", () => {
  const rows = Fixtures.upcomingFixturesForClub(fixturesFixture(), "EVE", 1, 3);
  assert.deepEqual(rows, []);
});

test("averageDifficulty is null for no fixtures, otherwise the mean", () => {
  assert.equal(Fixtures.averageDifficulty([]), null);
  const rows = Fixtures.upcomingFixturesForClub(fixturesFixture(), "ARS", 1, 3);
  assert.equal(Fixtures.averageDifficulty(rows), (2 + 4 + 1) / 3);
});

test("fixtureRunByClub sorts easiest average-difficulty first", () => {
  const rows = Fixtures.fixtureRunByClub(fixturesFixture(), 1, 3);
  const arsIdx = rows.findIndex((r) => r.club === "ARS"); // avg (2+4+1)/3 = 2.33
  const covIdx = rows.findIndex((r) => r.club === "COV"); // avg (5+2)/2 = 3.5
  assert.ok(arsIdx < covIdx, "ARS's easier run should sort before COV's harder run");
});

test("playersByFixtureEase filters to a max average difficulty when given", () => {
  const players = [
    { id: 1, team: "ARS" }, // avg 2.33
    { id: 2, team: "COV" }, // avg 3.5
  ];
  const easy = Fixtures.playersByFixtureEase(players, fixturesFixture(), 1, 3, 2.5);
  assert.equal(easy.length, 1);
  assert.equal(easy[0].player.id, 1);
});

test("playersByFixtureEase with no cutoff just sorts, keeping every player", () => {
  const players = [{ id: 1, team: "COV" }, { id: 2, team: "ARS" }];
  const sorted = Fixtures.playersByFixtureEase(players, fixturesFixture(), 1, 3, null);
  assert.equal(sorted.length, 2);
  assert.equal(sorted[0].player.id, 2); // ARS (easier) first
});

test("deadlineCountdown computes days/hours/minutes remaining", () => {
  const now = new Date("2026-08-27T17:30:00Z"); // exactly 1 day before GW2's deadline
  const result = Fixtures.deadlineCountdown("2026-08-28T17:30:00Z", now);
  assert.equal(result.isPast, false);
  assert.equal(result.days, 1);
  assert.equal(result.hours, 0);
  assert.equal(result.minutes, 0);
});

test("deadlineCountdown flags a past deadline", () => {
  const now = new Date("2026-08-22T00:00:00Z");
  const result = Fixtures.deadlineCountdown("2026-08-21T17:30:00Z", now);
  assert.equal(result.isPast, true);
});

test("deadlineCountdown returns null for a missing deadline", () => {
  assert.equal(Fixtures.deadlineCountdown(null, new Date()), null);
});
