# Two-team chip timing -- Wildcard / Bench Boost / Free Hit

_Generated 2026-09-01T03:43:16.886114+00:00. Projected expected points, not realised. Param bundles compared: active._

The full-horizon `force_wildcard_at` sweep is the primary Wildcard signal; the greedy `model_choice` walk only sees ~5 gameweeks ahead at each step and is shown for contrast. See `src/fpl_quant/chip_timing_analysis.py` for the method.

## Matippy toes (entry 1305242)

_Param bundle: active (xi v2, rho_residual=0.0, lambda=0.15, kappa_tc=0.15). Projected EP over GW3-16. Not realised points._

**Data flags:**
- [!] stale injury/rotation evidence (>14d) for held players: Antonín Kinsky, Virgil van Dijk, Riccardo Calafiori, Luke Shaw, Bruno Borges Fernandes, Dominik Szoboszlai, Bryan Mbeumo, Pascal Groß, Erling Haaland, Dominic Calvert-Lewin, João Pedro Junqueira de Jesus, Joe Rodon, Bobby Thomas
- [!] no availability evidence at all for held players: player_bart_verbruggen, player_regan_slater

**Wildcard -- full-horizon sweep says GW12** (+65.4 projected pts vs holding, 660.0 total).
- Greedy model_choice walk: plays GW3
- Hold arm's own threshold read-off: GW3 (+30.6)
- Greedy plays GW3; full-horizon sweep prefers GW12 (+65.4 projected pts vs holding). At GW3 the model only sees GW3..GW7, so GW12 is outside its visible horizon and cannot be compared against GW3 at decision time. Per-GW evaluate_wildcard gain: GW3~30.6, GW12~-7.3.

| forced WC GW | total proj pts | 80% band | vs hold |
|----|----|----|----|
| 9 | 656.9 | 470-844 | +62.3 |
| 10 | 659.5 | 472-847 | +64.9 |
| 11 | 658.0 | 471-845 | +63.4 |
| 12 | 660.0 | 472-848 | +65.4 |

**Bench Boost combo** (fresh post-WC squad's bench, WC week + next 3):

**Free Hit:** GW3 (+17.0), GW4 (+12.9), GW5 (+10.8), GW6 (+8.1), GW7 (+7.4), GW8 (+10.7), GW9 (+7.3), GW10 (+11.2), GW11 (+11.6), GW12 (+5.8), GW13 (+8.5), GW14 (+4.2), GW15 (+4.6)

**Recalibration sensitivity:** Wildcard week by bundle: {'active': 12}. Stable across bundles.


## Ahmad sucks (entry 7139944)

_Param bundle: active (xi v2, rho_residual=0.0, lambda=0.15, kappa_tc=0.15). Projected EP over GW3-16. Not realised points._

**Data flags:**
- [!] stale injury/rotation evidence (>14d) for held players: Antonín Kinsky, Gabriel dos Santos Magalhães, Harry Maguire, Bruno Borges Fernandes, Bryan Mbeumo, Pascal Groß, Dominic Calvert-Lewin, Erling Haaland, João Pedro Junqueira de Jesus, Martin Dúbravka, Bobby Thomas
- [!] no availability evidence at all for held players: player_trai_hume, Christos Tzolis, player_regan_slater, player_leif_davis

**Wildcard -- full-horizon sweep says GW12** (+57.0 projected pts vs holding, 663.9 total).
- Greedy model_choice walk: plays GW3
- Hold arm's own threshold read-off: GW3 (+55.0)
- Greedy plays GW3; full-horizon sweep prefers GW12 (+57.0 projected pts vs holding). At GW3 the model only sees GW3..GW7, so GW12 is outside its visible horizon and cannot be compared against GW3 at decision time. Per-GW evaluate_wildcard gain: GW3~55.0, GW12~-19.8.

| forced WC GW | total proj pts | 80% band | vs hold |
|----|----|----|----|
| 9 | 658.0 | 471-845 | +51.1 |
| 10 | 658.0 | 471-845 | +51.1 |
| 11 | 658.0 | 471-845 | +51.1 |
| 12 | 663.9 | 475-853 | +57.0 |

**Bench Boost combo** (fresh post-WC squad's bench, WC week + next 3):

**Free Hit:** GW3 (+17.9), GW4 (+11.9), GW5 (+12.8), GW6 (+7.8), GW7 (+8.2), GW8 (+9.1), GW9 (+4.3), GW10 (+10.1), GW11 (+8.9), GW12 (+4.4), GW13 (+8.9), GW14 (+5.5), GW15 (+3.7), GW16 (+1.5)

**Recalibration sensitivity:** Wildcard week by bundle: {'active': 12}. Stable across bundles.


