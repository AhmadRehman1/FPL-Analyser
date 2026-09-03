# Risk-aversion (lambda) / concentration-cap sensitivity study

5 arms landed. Realized-points, evolving-manager season simulation (`backtest.run_season_simulation`). Live pins: lambda=0.15, cap=3.

## lambda_value sweep

| lambda | season | total | mean/GW | sharpe | max drawdown | transfers | chips | GWs |
|---|---|---|---|---|---|---|---|---|
| 0.15 (live pin) | 2025-2026 | 223 | 44.60 | 9.544 | 8.20 | 2 | 2 | 5 |

## xi_club_concentration_cap sweep (lambda held at the live pin)

| cap | season | total | mean/GW | sharpe | max drawdown | transfers | chips | GWs |
|---|---|---|---|---|---|---|---|---|
| 2.0 | 2025-2026 | 213 | 42.60 | 6.434 | 13.20 | 2 | 2 | 5 |
| 3.0 (live pin) | 2025-2026 | 222 | 44.40 | 16.734 | 4.40 | 1 | 3 | 5 |
| 4.0 | 2025-2026 | 187 | 37.40 | 3.334 | 24.80 | 4 | 0 | 5 |
| 5.0 | 2025-2026 | 223 | 44.60 | 9.544 | 8.20 | 2 | 2 | 5 |

## Reading the evidence

- **2025-2026**: best realized Sharpe at lambda=0.15 (sharpe 9.544, total 223); best total at lambda=0.15 (223 pts). Live pin 0.15 scored sharpe 9.544, total 223, drawdown 8.20.

_This is evidence, not a decision. seeds_1.json stays parked; promotion is scripts/review_recalibration.py's human gate (README: the lambda study gates it, the 'attack' posture default, and the deferred 'protect rank' toggle)._
