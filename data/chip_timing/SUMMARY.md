# Two-team chip timing -- Wildcard / Bench Boost / Free Hit

_Generated 2026-09-05T01:47:53.800932+00:00. Projected expected points, not realised. Param bundles compared: active._

The full-horizon `force_wildcard_at` sweep is the primary Wildcard signal; the greedy `model_choice` walk only sees ~5 gameweeks ahead at each step and is shown for contrast. See `src/fpl_quant/chip_timing_analysis.py` for the method.

## Matippy toes (entry 1305242)

_Param bundle: active (xi v2, rho_residual=0.0, lambda=0.15, kappa_tc=0.15). Projected EP over GW4-22. Not realised points._

**Data flags:**
- [!] stale injury/rotation evidence (>14d) for held players: Riccardo Calafiori, Rayan Cherki, Pascal Groß
- [!] no availability evidence at all for held players: player_regan_slater, player_bobby_thomas
- [!] entry has already used chips: set1=['triple_captain'] set2=[]

**Wildcard:** the sweep finds no forced week that beats holding the chip over the evaluation window -- hold it and re-run as the season fills in.
- Greedy model_choice walk: plays GW4
- Hold arm's own threshold read-off: GW3 (+57.4)
- No forced-Wildcard gameweek in the sweep window beats the hold-Wildcard baseline on total projected points -- the sweep says hold the chip, not that any particular week is best.

| forced WC GW | total proj pts | 80% band | vs hold |
|----|----|----|----|
| 4 | 1400.3 | 1061-1740 | -91.5 |
| 5 | 1396.5 | 1060-1733 | -95.3 |
| 6 | 1408.8 | 1066-1752 | -83.0 |
| 7 | 1333.5 | 1017-1650 | -158.3 |
| 8 | 1327.6 | 1019-1637 | -164.2 |
| 9 | 1322.5 | 1012-1633 | -169.3 |
| 10 | 1327.6 | 1019-1637 | -164.2 |
| 11 | 1321.2 | 1011-1631 | -170.6 |
| 13 | 1335.4 | 1020-1651 | -156.4 |
| 14 | 1333.5 | 1017-1650 | -158.3 |
| 15 | 1335.4 | 1020-1651 | -156.4 |
| 16 | 1333.5 | 1017-1650 | -158.3 |
| 17 | 1351.1 | 1029-1673 | -140.7 |
| 18 | 1351.1 | 1029-1673 | -140.7 |
| 19 | 1333.5 | 1017-1650 | -158.3 |

**Bench Boost combo** (fresh post-WC squad's bench, WC week + next 3):

**Free Hit:** GW3 (+21.3), GW4 (+19.5), GW5 (+15.8), GW6 (+12.3), GW7 (+15.5), GW8 (+6.9), GW9 (+10.2), GW10 (+5.2), GW11 (+4.7), GW12 (+2.0), GW13 (+8.8), GW15 (+1.8), GW16 (+10.4), GW17 (+4.5), GW18 (+3.7), GW19 (+2.9), GW20 (+3.9), GW21 (+5.8)

**Recalibration sensitivity:** Wildcard week by bundle: {'active': None}. Stable across bundles.


## Ahmad sucks (entry 7139944)

_Param bundle: active (xi v2, rho_residual=0.0, lambda=0.15, kappa_tc=0.15). Projected EP over GW3-22. Not realised points._

**Data flags:**
- [!] stale injury/rotation evidence (>14d) for held players: Pascal Groß
- [!] no availability evidence at all for held players: player_bobby_thomas, player_regan_slater

**Wildcard:** the sweep finds no forced week that beats holding the chip over the evaluation window -- hold it and re-run as the season fills in.
- Greedy model_choice walk: plays GW4
- Hold arm's own threshold read-off: GW3 (+82.1)
- No forced-Wildcard gameweek in the sweep window beats the hold-Wildcard baseline on total projected points -- the sweep says hold the chip, not that any particular week is best.

| forced WC GW | total proj pts | 80% band | vs hold |
|----|----|----|----|
| 4 | 1421.8 | 1074-1770 | -62.3 |
| 5 | 1409.1 | 1067-1751 | -75.0 |
| 6 | 1395.7 | 1057-1735 | -88.4 |
| 7 | 1421.8 | 1074-1770 | -62.3 |
| 8 | 1367.3 | 1039-1695 | -116.8 |
| 9 | 1404.2 | 1062-1746 | -79.9 |
| 10 | 1404.2 | 1062-1746 | -79.9 |
| 12 | 1391.9 | 1056-1728 | -92.2 |
| 13 | 1391.9 | 1056-1728 | -92.2 |
| 14 | 1391.9 | 1056-1728 | -92.2 |
| 15 | 1407.3 | 1066-1749 | -76.8 |
| 16 | 1421.8 | 1074-1770 | -62.3 |
| 17 | 1398.3 | 1064-1733 | -85.8 |
| 18 | 1391.9 | 1056-1728 | -92.2 |
| 19 | 1407.9 | 1069-1747 | -76.2 |

**Bench Boost combo** (fresh post-WC squad's bench, WC week + next 3):

**Free Hit:** GW3 (+25.9), GW4 (+23.5), GW5 (+19.0), GW6 (+9.9), GW7 (+17.8), GW8 (+9.0), GW9 (+4.0), GW10 (+4.8), GW11 (+3.6), GW13 (+8.0), GW16 (+10.2), GW17 (+2.2), GW18 (+2.9), GW19 (+4.4), GW20 (+2.9), GW21 (+2.8)

**Recalibration sensitivity:** Wildcard week by bundle: {'active': None}. Stable across bundles.


