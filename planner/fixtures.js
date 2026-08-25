/* planner/fixtures.js
 *
 * Fixture-difficulty helpers built directly on data/dashboard/app_fixtures.json's own shape
 * (gameweeks[].fixtures[].{home,away}.{short_name,difficulty}) -- the same FDR 1-5 scale and
 * fixture list index.html's own Fixtures tab already renders (see FDR_COLOR there). No new
 * data source: this module only reshapes what's already fetched, per club instead of per match.
 */
(function (root, factory) {
  if (typeof module !== "undefined" && module.exports) {
    module.exports = factory();
  } else {
    root.PlannerFixtures = factory();
  }
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  /** Every fixture for `clubShortName` (e.g. "ARS") from `fromGw` for the next `n` gameweeks,
   * in gameweek order, as { gameweek, opponent, home, difficulty }. A blank gameweek for that
   * club (no fixture scheduled) is simply absent from the result -- not padded with a fake
   * entry, so callers can tell "no fixture" apart from "fixture with unknown difficulty". */
  function upcomingFixturesForClub(fixturesData, clubShortName, fromGw, n) {
    var gameweeks = (fixturesData && fixturesData.gameweeks) || [];
    var out = [];
    for (var i = 0; i < gameweeks.length; i++) {
      var gw = gameweeks[i];
      if (gw.gameweek < fromGw || gw.gameweek >= fromGw + n) continue;
      (gw.fixtures || []).forEach(function (f) {
        if (f.home && f.home.short_name === clubShortName) {
          out.push({ gameweek: gw.gameweek, opponent: f.away.short_name, home: true, difficulty: f.home.difficulty });
        } else if (f.away && f.away.short_name === clubShortName) {
          out.push({ gameweek: gw.gameweek, opponent: f.home.short_name, home: false, difficulty: f.away.difficulty });
        }
      });
    }
    out.sort(function (a, b) { return a.gameweek - b.gameweek; });
    return out;
  }

  /** Mean FDR across a club's fixtures in the window -- null (not 0, which would misleadingly
   * read as "very easy") when the club has no scheduled fixtures in that window at all (a real
   * blank-gameweek possibility, even though M8's own research found none scheduled this
   * season -- see docs/specs/M8_Transfer_Chip_Strategy_Planner.md's research findings). */
  function averageDifficulty(fixturesForClub) {
    var withDifficulty = fixturesForClub.filter(function (f) { return f.difficulty != null; });
    if (!withDifficulty.length) return null;
    var sum = withDifficulty.reduce(function (s, f) { return s + f.difficulty; }, 0);
    return sum / withDifficulty.length;
  }

  /** Every club's upcoming-N-gameweek run in one pass, sorted easiest-first -- the "fixture
   * swing" table the planner's fixture view and player-search sort both read from. */
  function fixtureRunByClub(fixturesData, fromGw, n) {
    var gameweeks = (fixturesData && fixturesData.gameweeks) || [];
    var clubs = new Set();
    gameweeks.forEach(function (gw) {
      (gw.fixtures || []).forEach(function (f) {
        if (f.home) clubs.add(f.home.short_name);
        if (f.away) clubs.add(f.away.short_name);
      });
    });
    var rows = [];
    clubs.forEach(function (club) {
      var fixtures = upcomingFixturesForClub(fixturesData, club, fromGw, n);
      rows.push({ club: club, fixtures: fixtures, averageDifficulty: averageDifficulty(fixtures) });
    });
    rows.sort(function (a, b) {
      var aAvg = a.averageDifficulty == null ? 99 : a.averageDifficulty;
      var bAvg = b.averageDifficulty == null ? 99 : b.averageDifficulty;
      return aAvg - bAvg;
    });
    return rows;
  }

  /** Filters+sorts a player list by how easy their club's upcoming fixtures are -- the "show me
   * players with easy fixtures over the next 3 GWs" search/sort mode. `maxAverageDifficulty`
   * defaults to no filtering (just a fixture-ease sort); pass e.g. 2.5 to filter down to clubs
   * with a genuinely easy run. */
  function playersByFixtureEase(players, fixturesData, fromGw, n, maxAverageDifficulty) {
    var runByClub = {};
    fixtureRunByClub(fixturesData, fromGw, n).forEach(function (row) { runByClub[row.club] = row; });
    var withRun = players.map(function (p) {
      var run = runByClub[p.team];
      return { player: p, averageDifficulty: run ? run.averageDifficulty : null, fixtures: run ? run.fixtures : [] };
    });
    var filtered = maxAverageDifficulty == null
      ? withRun
      : withRun.filter(function (r) { return r.averageDifficulty != null && r.averageDifficulty <= maxAverageDifficulty; });
    filtered.sort(function (a, b) {
      var aAvg = a.averageDifficulty == null ? 99 : a.averageDifficulty;
      var bAvg = b.averageDifficulty == null ? 99 : b.averageDifficulty;
      return aAvg - bAvg;
    });
    return filtered;
  }

  /** Deadline countdown for a given gameweek's deadline_time (ISO 8601) relative to `now`
   * (defaults to the current time) -- { totalMs, days, hours, minutes, isPast }. Pure/testable:
   * `now` is injectable so a test doesn't depend on wall-clock time. */
  function deadlineCountdown(deadlineIso, now) {
    if (!deadlineIso) return null;
    var nowMs = (now instanceof Date ? now : new Date(now || Date.now())).getTime();
    var deadlineMs = new Date(deadlineIso).getTime();
    var totalMs = deadlineMs - nowMs;
    var isPast = totalMs <= 0;
    var abs = Math.abs(totalMs);
    var days = Math.floor(abs / 86400000);
    var hours = Math.floor((abs % 86400000) / 3600000);
    var minutes = Math.floor((abs % 3600000) / 60000);
    return { totalMs: totalMs, days: days, hours: hours, minutes: minutes, isPast: isPast };
  }

  return {
    upcomingFixturesForClub: upcomingFixturesForClub,
    averageDifficulty: averageDifficulty,
    fixtureRunByClub: fixtureRunByClub,
    playersByFixtureEase: playersByFixtureEase,
    deadlineCountdown: deadlineCountdown,
  };
});
