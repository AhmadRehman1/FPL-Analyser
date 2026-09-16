# Season-horizon chip-timing sensitivity study (Workstream B's "fuller ask")

2 arms landed. Realized-points, evolving-manager season simulation (`backtest.run_season_simulation`). "off" = current greedy approach (PR #173's magnitude floor only); "on" = adds the wider-window season-horizon timing gate (triple_captain_timing_params_version=1, bench_boost_timing_params_version=1).

## Arms

| arm | season | total | mean/GW | sharpe | max drawdown | transfers | chips played | GWs |
|---|---|---|---|---|---|---|---|---|
| off | 2025-2026 | 822 | 43.26 | 3.563 | 45.11 | 12 | 6 | 19 |
| on | 2025-2026 | 818 | 43.05 | 3.401 | 50.42 | 12 | 6 | 19 |

## Reading the evidence

- **2025-2026**: timing-on regresses total realized points by -4.0 (822 -> 818), Sharpe -0.162. the timing gate HELD 3 chip(s) the greedy arm played: [(4, 'triple_captain'), (6, 'bench_boost'), (16, 'free_hit')]; the timing gate additionally played 3 chip(s) the greedy arm didn't: [(4, 'free_hit'), (7, 'triple_captain'), (8, 'bench_boost')]; 3 chip decision(s) unchanged: [(9, 'wildcard'), (19, 'free_hit'), (20, 'bench_boost')]

_This is evidence, not a decision. triple_captain_timing_params_version/bench_boost_timing_params_version stay None (off) in forward_season_sim._resolve_versions() and backtest.run_season_simulation()'s own defaults until the project owner reviews this and explicitly turns them on._
