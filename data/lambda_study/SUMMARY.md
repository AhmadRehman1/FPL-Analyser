# Risk-aversion (lambda) / concentration-cap sensitivity study

2 arms landed. Realized-points, evolving-manager season simulation (`backtest.run_season_simulation`). Live pins: lambda=0.15, cap=3.

## lambda_value sweep

| lambda | season | total | mean/GW | sharpe | max drawdown | transfers | chips | GWs |
|---|---|---|---|---|---|---|---|---|
| 0.1 | 2024-2025 | 2069 | 55.92 | 3.297 | 79.78 | 29 | 6 | 37 |
| 0.15 (live pin) | 2025-2026 | 2123 | 57.38 | 3.687 | 83.95 | 26 | 7 | 37 |

## xi_club_concentration_cap sweep (lambda held at the live pin)

_No cap arms landed._

## Reading the evidence

- **2024-2025**: best realized Sharpe at lambda=0.1 (sharpe 3.297, total 2069); best total at lambda=0.1 (2069 pts).
- **2025-2026**: best realized Sharpe at lambda=0.15 (sharpe 3.687, total 2123); best total at lambda=0.15 (2123 pts). Live pin 0.15 scored sharpe 3.687, total 2123, drawdown 83.95.

_This is evidence, not a decision. seeds_1.json stays parked; promotion is scripts/review_recalibration.py's human gate (README: the lambda study gates it, the 'attack' posture default, and the deferred 'protect rank' toggle)._
