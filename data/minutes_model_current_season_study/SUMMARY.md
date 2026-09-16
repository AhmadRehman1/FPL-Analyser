# Minutes-model current-season-role sensitivity study

Full M7 walk-forward (`backtest.run()`), both historical seasons. "off" = current production behavior (current_season_role_params_version never activated); "on" = adds the fast-reacting current-season-own-rate blend at its v1 default (current_season_matches_threshold=4). Both arms already include minutes_model.run()'s own unconditional lookback_seasons fix (provably backtest-neutral, not what this study is testing).

## beats_crowd_points_delta by tier

| tier | off | on | delta | verdict |
|---|---|---|---|---|
| - | - | - | - | one or both arms missing |

## Minutes-model calibration (direct read, not just downstream points)

| metric | off | on | delta |
|---|---|---|---|
| - | - | - | - |

## Reading the evidence

_One or both arms missing -- no comparison possible._

_This is evidence, not a decision. current_season_role_params_version stays unwired-by-default at both backtest.py call sites until the project owner reviews this and explicitly opts a caller in._