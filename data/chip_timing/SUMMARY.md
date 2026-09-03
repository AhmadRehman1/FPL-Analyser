# Two-team chip timing -- Wildcard / Bench Boost / Free Hit

_Generated 2026-09-03T02:19:24.371632+00:00. Projected expected points, not realised. Param bundles compared: active._

The full-horizon `force_wildcard_at` sweep is the primary Wildcard signal; the greedy `model_choice` walk only sees ~5 gameweeks ahead at each step and is shown for contrast. See `src/fpl_quant/chip_timing_analysis.py` for the method.

## Matippy toes (entry 1305242)

_Param bundle: active (xi v2, rho_residual=0.0, lambda=0.15, kappa_tc=0.15). Projected EP over GW3-22. Not realised points._

**Data flags:**
- [!] stale injury/rotation evidence (>14d) for held players: Riccardo Calafiori, Pascal Groß
- [!] no availability evidence at all for held players: player_bobby_thomas, player_regan_slater

**Wildcard -- full-horizon sweep says GW4** (+25.1 projected pts vs holding, 1028.8 total).
- Greedy model_choice walk: plays GW3
- Hold arm's own threshold read-off: GW5 (+41.5)
- Greedy plays GW3; full-horizon sweep prefers GW4 (+25.1 projected pts vs holding). At GW3 the model only sees GW3..GW7, so GW4 is outside its visible horizon and cannot be compared against GW3 at decision time. Per-GW evaluate_wildcard gain: GW3~11.9, GW4~0.7.

| forced WC GW | total proj pts | 80% band | vs hold |
|----|----|----|----|
| 4 | 1028.8 | 761-1296 | +25.1 |
| 5 | 1028.8 | 761-1296 | +25.1 |

**Bench Boost combo** (fresh post-WC squad's bench, WC week + next 3):
- GW3: synergy -4.88 (combo 52.9 vs naive 57.8) 
- GW4: synergy -8.81 (combo 44.8 vs naive 53.6) 
- GW5: synergy -4.89 (combo 51.5 vs naive 56.4) 
- GW6: synergy -5.00 (combo 44.4 vs naive 49.4) 

**Free Hit:** GW3 (+15.9), GW4 (+17.4), GW5 (+17.1), GW6 (+8.2), GW7 (+12.5), GW8 (+8.3), GW9 (+11.8), GW10 (+12.8), GW11 (+10.9), GW12 (+4.1), GW14 (+4.8), GW15 (+2.9), GW16 (+7.0), GW17 (+8.2), GW18 (+3.7), GW19 (+6.1), GW21 (+10.1), GW22 (+5.8)

**Wildcard squad robustness (GW3, 8 perturbed solves):** fragile -- 0/15 core, 47 fragile.
- fragile: player_adrien_truffert, player_bart_verbruggen, player_bruno_borges_fernandes, player_bryan_mbeumo, player_dan_bentley, player_declan_rice, player_dillon_phillips, player_eberechi_eze, player_elliot_anderson, player_emiliano_martinez_romero, player_enzo_fernandez, player_enzo_le_fee, player_erling_haaland, player_ethan_ampadu, player_ezri_konsa_ngoyo, player_francisco_evanilson_de_lima_barbosa, player_gianluigi_donnarumma, player_iliman_ndiaye, player_james_garner, player_joachim_andersen, player_joelinton_cassio_apolinario_de_lira, player_liam_kitching, player_marcos_senesi_baron, player_matheus_nunes, player_matheus_santos_carneiro_da_cunha, player_mikkel_damsgaard, player_moises_caicedo_corozo, player_morgan_rogers, player_nathan_collins, player_nikola_milenkovic, player_ollie_watkins, player_ruben_dos_santos_gato_alves_dias, player_ryan_gravenberch, player_sepp_van_den_berg, player_shumaira_mheuka, player_sindre_walle_egeli, player_tyler_fredricson, player_vitezslav_jaros, player_wellity_lucky, player_wesley_fofana, player_will_dennis, player_will_hughes, player_william_osula, player_william_saliba, player_wilson_isidor, player_yoane_wissa, player_zach_abbott

**Recalibration sensitivity:** Wildcard week by bundle: {'active': 4}. Stable across bundles.


## Ahmad sucks (entry 7139944)

_Param bundle: active (xi v2, rho_residual=0.0, lambda=0.15, kappa_tc=0.15). Projected EP over GW3-22. Not realised points._

**Data flags:**
- [!] stale injury/rotation evidence (>14d) for held players: Pascal Groß
- [!] no availability evidence at all for held players: player_bobby_thomas, player_regan_slater

**Wildcard -- full-horizon sweep says GW10** (+23.7 projected pts vs holding, 1028.8 total).
- Greedy model_choice walk: plays GW3
- Hold arm's own threshold read-off: GW5 (+41.0)
- Greedy plays GW3; full-horizon sweep prefers GW10 (+23.7 projected pts vs holding). At GW3 the model only sees GW3..GW7, so GW10 is outside its visible horizon and cannot be compared against GW3 at decision time. Per-GW evaluate_wildcard gain: GW3~31.3, GW10~-17.4.

| forced WC GW | total proj pts | 80% band | vs hold |
|----|----|----|----|
| 5 | 1028.8 | 761-1296 | +23.7 |
| 6 | 1028.8 | 761-1296 | +23.7 |
| 7 | 1028.8 | 761-1296 | +23.7 |
| 8 | 1028.8 | 761-1296 | +23.7 |
| 9 | 1028.8 | 761-1296 | +23.7 |
| 10 | 1028.8 | 761-1296 | +23.7 |
| 11 | 1028.8 | 761-1296 | +23.7 |
| 12 | 1028.8 | 761-1296 | +23.7 |
| 14 | 1028.8 | 761-1296 | +23.7 |
| 15 | 1028.8 | 761-1296 | +23.7 |
| 16 | 1028.8 | 761-1296 | +23.7 |
| 17 | 1028.8 | 761-1296 | +23.7 |
| 18 | 1028.8 | 761-1296 | +23.7 |
| 19 | 1028.8 | 761-1296 | +23.7 |

**Bench Boost combo** (fresh post-WC squad's bench, WC week + next 3):
- GW3: synergy -2.48 (combo 52.9 vs naive 55.4) 
- GW4: synergy -3.69 (combo 44.8 vs naive 48.5) 
- GW5: synergy -2.90 (combo 51.5 vs naive 54.4) 
- GW6: synergy -3.15 (combo 44.4 vs naive 47.6) 

**Free Hit:** GW3 (+18.4), GW4 (+16.3), GW5 (+20.1), GW6 (+9.1), GW7 (+11.1), GW8 (+11.7), GW9 (+5.9), GW10 (+12.4), GW11 (+8.9), GW12 (+2.3), GW13 (+7.4), GW14 (+2.4), GW15 (+2.0), GW16 (+5.9), GW17 (+7.5), GW18 (+2.8), GW19 (+4.7), GW20 (+4.0), GW21 (+6.3), GW22 (+5.5)

**Wildcard squad robustness (GW3, 8 perturbed solves):** fragile -- 0/15 core, 47 fragile.
- fragile: player_adrien_truffert, player_bart_verbruggen, player_bruno_borges_fernandes, player_bryan_mbeumo, player_declan_rice, player_eberechi_eze, player_elliot_anderson, player_emiliano_martinez_romero, player_enzo_fernandez, player_enzo_le_fee, player_erling_haaland, player_ethan_ampadu, player_ezri_konsa_ngoyo, player_francisco_evanilson_de_lima_barbosa, player_gianluigi_donnarumma, player_iliman_ndiaye, player_james_garner, player_joachim_andersen, player_joelinton_cassio_apolinario_de_lira, player_jorrel_hato, player_liam_kitching, player_marcos_senesi_baron, player_martin_dubravka, player_matheus_nunes, player_matheus_santos_carneiro_da_cunha, player_maxence_lacroix, player_mikkel_damsgaard, player_moises_caicedo_corozo, player_morgan_rogers, player_nathan_collins, player_nikola_milenkovic, player_ollie_watkins, player_robin_roefs, player_ruben_dos_santos_gato_alves_dias, player_ryan_gravenberch, player_sepp_van_den_berg, player_shumaira_mheuka, player_sindre_walle_egeli, player_tyler_fredricson, player_vitezslav_jaros, player_wellity_lucky, player_will_dennis, player_will_hughes, player_william_osula, player_wilson_isidor, player_yoane_wissa, player_zach_abbott

**Recalibration sensitivity:** Wildcard week by bundle: {'active': 10}. Stable across bundles.


