# Expert consensus - GW6 (fixture)

Small test fixture in the same table formats as the Cowork research output, including the
known-bad rows from the real GW6 file.

Caveat on manager attribution: the tool believes some managers were fetch errors. (Ignored.)

## CaptainPicks

| player | club | n_sources_captaining | n_sources_total | share | notes |
|---|---|---|---|---|---|
| Erling Haaland | Man City | 7 | 10 | 0.70 | home fixture |
| Bruno Fernandes | Man Utd | 2 | 10 | 0.20 | |
| Antoine Semenyo | Bournemouth | 1 | 10 | n/a | club wrong in source |
| Alessandro Donnarumma | Man City | 1 | 10 | 0.1 | wrong first name |

## TransferTargets

| player | club | position | price | n_sources_buy | n_sources_sell | net_score | reason_tags | notes |
|---|---|---|---|---|---|---|---|---|
| Pascal Groß | Brighton | MID | 5.8 | 6 | 0 | 6 | form;fixtures | |
| Cole Palmer | Chelsea | MID | 9.7 | 1 | 4 | -3 | minutes | |
| Wouter Lacroix | Crystal Palace | DEF | 5.0 | 3 | 0 | 3 | cheap | no such player |
| Robert Sanchez / Martinez | Chelsea | GK | 5.0 | 2 | 0 | 2 | | two players |

## Differentials

| player | club | ownership_pct | n_sources | reason_tags | notes |
|---|---|---|---|---|---|
| Ismaila Minteh | Brighton | 3.1 | 2 | pace | wrong first name |
| Pascal Groß | Brighton | 95+ | 3 | set pieces | |

## Avoid

| player | club | n_sources | reason | notes |
|---|---|---|---|---|
| Cole Palmer | Chelsea | 3 | rotation | |

## Sources

| source_name | source_type | url | published_date | track_record_note | weight_1_10 |
|---|---|---|---|---|---|
| Fantasy Football Scout | specialist | https://example.com/a | 2026-10-01 | long record | 8 |
| FPL Harry | specialist | https://example.com/b | 2026-10-01 | | 6 |
| Random forum | community | https://example.com/c | 2026-09-30 | | 2 |

## Table 1 - Injuries

| player | club | status | issue | date_reported | expected_return | source_name | source_type | confidence_1_10 | information_type | observed_date | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Cole Palmer | Chelsea | Doubt | groin | 2026-09-29 | after the international break | Official FPL site | official | 9 | FACT | 2026-09-29 | |

## Table 2 - PredictedXI

| player | club | predicted_starter | start_confidence_pct | expected_minutes | position | source_name | source_type | confidence_1_10 | information_type | observed_date | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Pascal Groß | Brighton | Yes | 72-80 | 75-90 | CM | Fantasy Football Scout | specialist | 7 | OPINION | 2026-10-01 | |
| Erling Haaland | Man City | Yes | 95+ | 75-90 | ST | Fantasy Football Scout | specialist | 8 | OPINION | 2026-10-01 | |
| Cole Palmer | Chelsea | Yes, if fit | 50 | 60-75 | AM | Fantasy Football Scout | specialist | 5 | OPINION | 2026-10-01 | |
| Bruno Fernandes | Man Utd | Doubt | n/a | 60-75 | AM | FPL Harry | specialist | 4 | OPINION | 2026-10-01 | |
