# squad_optimizer solve-time wiring sensitivity study

Full M7 walk-forward (`backtest.run()`), both historical seasons. "off" = current production behavior (the four families never activated at either squad_optimizer.run() call site in backtest.py); "on" = adds ownership/risk_posture/field_covariance/bench_quality/concentration_risk at their un-recalibrated v1 defaults.

## beats_crowd_points_delta by tier

| tier | off | on | delta | verdict |
|---|---|---|---|---|
| cold | +5.388 | +15.763 | +10.375 | improvement |
| mature | -1.681 | +1.969 | +3.650 | improvement |
| warm | -1.807 | +0.907 | +2.714 | improvement |

## segment_calibration (mature tier, signed resid = realized - predicted)

| segment | off resid | on resid | delta (closer to 0 is better) |
|---|---|---|---|
| position=Defender | -0.156 | -0.156 | -0.000 |
| position=Forward | +0.009 | +0.009 | -0.000 |
| position=Goalkeeper | -0.425 | -0.425 | +0.000 |
| position=Midfielder | -0.099 | -0.099 | +0.000 |
| price_band=5.0-7.0 | +0.063 | +0.063 | +0.000 |
| price_band=7.0-9.0 | +0.507 | +0.507 | -0.000 |
| price_band=9.0+ | +0.314 | +0.314 | -0.000 |
| price_band=<5.0 | -0.282 | -0.282 | +0.000 |

## Reading the evidence

- **"on" does not regress beats_crowd_points_delta on any tier** -- the specific bar the deferred task set. That clears the way to consider defaulting this wiring on, but is not by itself a recommendation to do so (segment_calibration and divergence-check pass rates above are also worth a human look before deciding).

_This is evidence, not a decision. squad_optimizer.run()'s five Priority 1/2 terms stay unwired-by-default at both backtest.py call sites until the project owner reviews this and explicitly opts a caller in._