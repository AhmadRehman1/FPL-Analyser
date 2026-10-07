Data current as of 2026-10-07. Season: Premier League 2026/27, GW1-5 played, GW6 upcoming (deadline Sat 10 Oct 10:00 UTC). Source: Fantasy Football Fix, "Best FPL Forwards for Gameweek 6: Stats & Wildcard Picks" (GW6 positional review: forwards), published 7 Oct 2026[^1]. Every row was cross-checked against FPL's bootstrap-static and element-summary on 2026-10-07[^2].

Scope note: only the article's role and set-piece facts are tabled. Its stats (xG, shots, big chances), fixture ratings, points projections and Wildcard picks are left out, because docs/evidence_research_pull_prompt.md excludes them and the model computes its own. Its four fitness doubts are also left out on purpose. FPL already flags each one at 75% (Isak thigh, Havertz hamstring, Brobbey hamstring, João Pedro knee), and the minutes model applies that live flag directly (minutes_model.live_availability_by_player), so an Injuries row here would count the same knock twice.

## Table 4 — RoleChange

| player | club | change | cause | effective_from | source_name | source_type | confidence_1_10 | information_type | observed_date | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Charalampos Kostoulas | Brighton | won_starting_spot | Brighton's centre-forward since Danny Welbeck's move to Chelsea | 2026-08-28 | Fantasy Football Fix | specialist | 6 | FACT | 2026-10-07 | FPL data: started GW2-5 (79/90/83/75 min) after a 26-min GW1 sub cameo; 2 starts in all of 2025-26[^1][^2]. |

RoleChange — 1 row, 1 club.

## Table 5 — SetPieces

| club | duty | primary_taker | secondary_taker | deputy_if_primary_absent | source_name | source_type | confidence_1_10 | information_type | observed_date | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Leeds United | Penalties | Dominic Calvert-Lewin | | | Fantasy Football Fix | specialist | 4 | OPINION | 2026-10-07 | Article: "potential penalty duties". Agrees with the 2026-09-01 pull[^1]. |
| Leeds United | Penalties | Dominic Calvert-Lewin | Lukas Nmecha | | Premier League official site | official | 8 | FACT | 2026-10-07 | FPL's set-piece order: Calvert-Lewin 1, Nmecha 2, Piroe 3[^2]. |
| Hull City | Penalties | Oli McBurnie | | | Fantasy Football Fix | specialist | 4 | OPINION | 2026-10-07 | Article: "potential penalty duties"[^1]. |
| Hull City | Penalties | Oli McBurnie | | | Premier League official site | official | 8 | FACT | 2026-10-07 | FPL's set-piece order lists McBurnie first and no second taker. The 2026-09-01 pull's Crooks-first row (fplcopilot) disagrees[^2]. |

SetPieces — 4 rows, 2 clubs.

## References

[^1]: Fantasy Football Fix, "Best FPL Forwards for Gameweek 6: Stats & Wildcard Picks", 7 Oct 2026. https://www.fantasyfootballfix.com/blog-index/fpl-gw6-forwards-positional-review-2026/
[^2]: Official FPL API, read 2026-10-07: https://fantasy.premierleague.com/api/bootstrap-static/ (set-piece orders, injury flags) and https://fantasy.premierleague.com/api/element-summary/138/ (Kostoulas's minutes and starts).
