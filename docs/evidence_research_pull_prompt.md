# Perplexity prompt — FPL 2026/27 in-season evidence database

You are compiling a structured evidence database for a quantitative Fantasy Premier League
model. The model already derives team strength, fixture difficulty, expected goals, minutes
history, ownership and prices from raw data — **do not research any of those.** It needs
exactly one class of input from you: **current, individually-sourced, human-observable facts
about player availability and on-pitch roles** that raw stats cannot yet reflect — a knock
reported this week, a change of penalty taker, a manager who rests one specific player for
European away games.

**Season:** Premier League 2026/27. Gameweeks 1 and 2 have been played; Gameweek 3 is next.
**Today:** {{FILL IN TODAY'S DATE}}. Every row must be current as of today. Preseason or
last-season information is only acceptable if nothing has changed **and you say so explicitly
in the notes** with the reason.

---

## Output: six tables. One fact per row. A fact appears in exactly one table.

The tables are deliberately non-overlapping. Before writing a row, confirm it belongs in this
table and not another:

| Situation | Goes in | NOT in |
|---|---|---|
| Doubtful with a hamstring strain | Injuries | PredictedXI |
| Fully fit but benched because a new signing took his spot | RoleChange | Injuries, PredictedXI |
| Fit, nailed, but the manager rests him for every Thursday Europa game | Rotation | PredictedXI |
| Fit, no competition problem, but expected to be a ~30-min sub | PredictedXI | Rotation |
| Takes his club's penalties / free kicks / corners | SetPieces | every other table |
| Close to a price rise/fall | PriceWatch | every other table |

---

### Table 1 — `Injuries` — every player carrying a fitness issue right now

`player` | `club` | `status` | `issue` | `date_reported` | `expected_return` | `source_name` | `source_type` | `confidence_1_10` | `information_type` | `observed_date` | `notes`

- `status` — exactly one of **Out** / **Doubt** / **Fit**. Use **Fit** only to record that a
  player has just returned and the injury risk has cleared (useful signal in itself).
- `issue` — short: "left hamstring strain", "in concussion protocol", "knee — surgery".
- `date_reported`, `expected_return` — `YYYY-MM-DD`, or `unknown`, or `YYYY-MM-xx` if only the
  month is known.
- Coverage: **every** player with a fitness flag on the official FPL site or in a credible
  press report, across all 20 clubs. Completeness beats a shortlist.

### Table 2 — `PredictedXI` — how nailed each fit player is

`player` | `club` | `predicted_starter` | `start_confidence_pct` | `expected_minutes` | `position` | `source_name` | `source_type` | `confidence_1_10` | `information_type` | `observed_date` | `notes`

- Fit players only. An injured player belongs in `Injuries`, not here.
- `predicted_starter` — `Yes` / `No`.
- `start_confidence_pct` — 0–100: probability he is in the starting XI for the next match,
  **given he is available**.
- `expected_minutes` — a band: `75-90`, `60-75`, `20-45 (sub)`.
- `position` — the role he actually plays now (e.g. `RW`, `false 9`, `LWB`), not his FPL
  position.
- Coverage: every player owned by >2% of FPL managers, **plus** every player priced ≥ £6.0m,
  **plus** any sub-£5.0m player who is a genuine nailed starter. Roughly 180–220 players.

### Table 3 — `Rotation` — managers who systematically rest a specific player

`player` | `club` | `manager` | `valence` | `pattern` | `trigger` | `source_name` | `source_type` | `confidence_1_10` | `information_type` | `observed_date` | `notes`

- Only where there is a **specific, repeated** pattern — not a generic "could be rotated".
- `valence` — **negative** (rested / rotated out) or **positive** (a fringe player who
  *reliably* starts in a defined situation, e.g. "always starts the domestic cups").
- `pattern` — "rested for every European away game", "subbed at 60–65 when the team leads",
  "alternates weekly with {player}".
- `trigger` — the condition: "Thursday Europa fixture", "3 games in 7 days", "vs a deep block".
- Clubs in European competition this season are the priority. A club with a settled XI and no
  midweek football may have zero rows — that is a valid answer.

### Table 4 — `RoleChange` — a fit player whose starting outlook changed since GW1

`player` | `club` | `change` | `cause` | `effective_from` | `source_name` | `source_type` | `confidence_1_10` | `information_type` | `observed_date` | `notes`

- `change` — one of **lost_starting_spot** / **won_starting_spot** / **new_position** /
  **frozen_out** / **likely_january_exit**.
- `cause` — "new signing {name} took the role", "switch to a back three", "transfer request",
  "manager fallout".
- A *durable* change to a fit player's outlook that two gameweeks of minutes don't yet fully
  price in. Nothing here may duplicate `Injuries` or `Rotation`.

### Table 5 — `SetPieces` — dead-ball duties as they stand now, all 20 clubs

`club` | `duty` | `primary_taker` | `secondary_taker` | `deputy_if_primary_absent` | `source_name` | `source_type` | `confidence_1_10` | `information_type` | `observed_date` | `notes`

- One row per **(club, duty)**. `duty` — exactly one of **Penalties** / **Direct Free Kicks** /
  **Corners**. 60 rows total.
- If a duty is genuinely unclear or shared, still write the row, name the likeliest taker, and
  set a low `confidence_1_10` — do not omit it.
- Base this on what has actually happened in GW1–2 plus credible reporting. In `notes`, flag
  any change from last season.

### Table 6 — `PriceWatch` — near-term price movements (low priority, provenance only)

`player` | `club` | `direction` | `note` | `source_name` | `source_type` | `confidence_1_10` | `information_type` | `observed_date`

- `direction` — `rising` / `falling` / `imminent_rise` / `imminent_fall`.
- Only players within ~0.1m of a change per the mainstream price-prediction trackers. Keep it
  short. Always `information_type = OPINION`.

---

## Rules for every row

- **`source_name`** — the specific outlet or person: "David Ornstein / The Athletic",
  "Official FPL site", "Fabrizio Romano", "BBC Sport match report", "@PhysioRoom". Never
  "various", "reports", "sources".
- **`source_type`** — one of **official** (club / league / FPL official channel),
  **journalist** (established newspaper or broadcaster reporter), **specialist** (an
  FPL-focused analyst or community expert with a track record), **community** (forums, general
  social media).
- **`confidence_1_10`** — calibrated: 9–10 official confirmation · 6–8 one credible journalist
  · 3–5 a single specialist's read or conflicting reports · 1–2 rumour.
- **`information_type`** — **FACT** (a verifiable event: a lineup that was actually named, an
  official injury listing, a penalty that was actually taken) or **OPINION** (a prediction or
  judgement: "expected to start", "likely rotated").
- **`observed_date`** — `YYYY-MM-DD`, the date the source published. For anything
  time-sensitive it should be within the last ~3 weeks; if the best source is older, say so in
  `notes` and lower the confidence.
- **`notes`** — one line maximum. Source conflicts, caveats, or "unchanged from last season,
  still current — {reason}".
- Where sources disagree, write **one row per source** with its own confidence. Do not average
  or merge them.

## Do NOT include

- Fixture difficulty, opponent strength, schedule congestion **as a rating** — the model
  computes fixtures itself. (Congestion is allowed only as a `trigger` value in `Rotation`.)
- Team attack / defence quality, xG, form tables, league position.
- Price-change **predictions presented as fact** — `PriceWatch` only, always marked OPINION.
- Captaincy picks, differentials, "template" advice, transfer or chip suggestions.
- Player-ability opinions ("world class", "underrated", "on form").
- Anything predating the 2026/27 season unless explicitly marked "still current, unchanged"
  with a reason.

## Format

Return each table as a clean markdown table using the exact column headers above, ready to
paste into a spreadsheet. Under each table put a one-line count ("Injuries — 34 rows, 18
clubs"). If you have no credible information for a table, say so rather than filling it with
guesses.
