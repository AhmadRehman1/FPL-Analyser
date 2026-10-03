"""FPL's rules that change from one season to the next, in one table every scorer and planner
reads (docs/reports/2026-10_open_issues.md, issue 1).

Before this, 2024-25 was scored and planned under 2025-26 rules except where a special case said
otherwise: DefCon had its own helper (expected_points.defcon_in_force) and everything else --
the chips, the bonus-points weights -- had no season at all. A rule that differs by season goes
here, and code asks this module rather than comparing season strings itself.

What changed, for the parts this project models (sources: the Premier League's own "what's new"
articles for 2025/26 and 2026/27, Fantasy Football Scout's BPS explainers for both seasons,
FootballFanCast's 2024/25 chip guide):

- Chips. 2024-25: two Wildcards, one per half; one Free Hit, Bench Boost and Triple Captain for
  the whole season; and an Assistant Manager chip from January for three gameweeks, which this
  project does not model. From 2025-26: two of every chip, one set per half. The first set
  expires at the Gameweek 19 deadline -- a chip played *in* GW19 is activated before that
  deadline, so GW19 is the first half's last week -- and the second set is available from GW20.
  (This code used to treat GW19 as a second-half week.)
- Defensive contributions: from 2025-26 only.
- Bonus points, in the terms expected_points' BPS estimate uses: +1 per 2 clearances, blocks
  and interceptions until 2025-26 and per 3 from 2026-27; a keeper save was a flat 2 in 2024-25
  and 3 inside the box (2 outside) from 2025-26 -- the estimate applies the inside-box value to
  every save. bps_formula_params v1 holds the 2026-27 weights; a season's overrides replace
  those keys for that season.
- Unchanged across 2024-25 to 2026-27 for what is modelled: goal, assist, clean sheet and
  appearance points (a goalkeeper's goal has been worth 10 since 2024-25).

Known differences not modelled: 2025-26 relaxed the assist definition (41 more assists had it
applied in 2024-25) -- the FPL/xA assist ratio is fitted from the seasons visible at the cutoff,
so it mixes the two definitions; and the BPS terms the estimate omits (tackles, goalline
clearances, penalty saves and goals, passing).
"""

from __future__ import annotations

from dataclasses import dataclass, field

FIRST_HALF_LAST_GAMEWEEK = 19
LAST_GAMEWEEK = 38
CHIP_TYPES = ("wildcard", "free_hit", "bench_boost", "triple_captain")

HALVES = ((1, FIRST_HALF_LAST_GAMEWEEK), (FIRST_HALF_LAST_GAMEWEEK + 1, LAST_GAMEWEEK))
WHOLE_SEASON = ((1, LAST_GAMEWEEK),)


@dataclass(frozen=True)
class SeasonRules:
    season: str
    defcon: bool
    # chip -> the gameweek windows it can be played in, once per window
    chip_windows: dict[str, tuple[tuple[int, int], ...]]
    # bps_formula_params keys whose value that season differs from the configured version
    bps_overrides: dict[str, float] = field(default_factory=dict)
    not_modelled: tuple[str, ...] = ()


RULES: dict[str, SeasonRules] = {
    "2024-2025": SeasonRules(
        season="2024-2025",
        defcon=False,
        chip_windows={"wildcard": HALVES, "free_hit": WHOLE_SEASON, "bench_boost": WHOLE_SEASON, "triple_captain": WHOLE_SEASON},
        bps_overrides={"cbi_per_point": 2.0, "save_inside_box": 2.0},
        not_modelled=("assistant_manager chip (GW24 on, three gameweeks)", "stricter assist definition"),
    ),
    "2025-2026": SeasonRules(
        season="2025-2026",
        defcon=True,
        chip_windows={chip: HALVES for chip in CHIP_TYPES},
        bps_overrides={"cbi_per_point": 2.0},
    ),
    "2026-2027": SeasonRules(
        season="2026-2027",
        defcon=True,
        chip_windows={chip: HALVES for chip in CHIP_TYPES},
    ),
}


def rules_for(season: str | None) -> SeasonRules:
    """The rules of `season`. None (no season known) and a season after the table get the
    latest rules; a season before it the earliest. "YYYY-YYYY" labels sort chronologically."""
    if season in RULES:
        return RULES[season]
    if season is None or season > max(RULES):
        return RULES[max(RULES)]
    return RULES[min(RULES)]


def bps_override(season: str | None, key: str) -> float | None:
    """The season's own value for a bps_formula_params key, or None to use the configured one."""
    return rules_for(season).bps_overrides.get(key)


def half_of(gameweek: int) -> int:
    """1 for GW1-19, 2 for GW20-38: which of manager_state_versions' chip lists records a chip
    played that week."""
    return 1 if gameweek <= FIRST_HALF_LAST_GAMEWEEK else 2


def chip_window(season: str | None, chip: str, gameweek: int) -> tuple[int, int] | None:
    """The (first, last) gameweek window of `chip` that `gameweek` falls in, or None."""
    return next((w for w in rules_for(season).chip_windows.get(chip, ()) if w[0] <= gameweek <= w[1]), None)


def chip_available(
    season: str | None, chip: str, gameweek: int, chips_used_set1, chips_used_set2,
) -> bool:
    """Whether `chip` can still be played in `gameweek`: its window covering that week has not
    been used. Uses are recorded by half (half_of()), so a whole-season window is spent by a use
    in either half, a half window only by a use in its own half."""
    window = chip_window(season, chip, gameweek)
    if window is None:
        return False
    first_half, second_half = HALVES
    used_first = chip in set(chips_used_set1 or ()) and window[0] <= first_half[1]
    used_second = chip in set(chips_used_set2 or ()) and window[1] >= second_half[0]
    return not (used_first or used_second)
