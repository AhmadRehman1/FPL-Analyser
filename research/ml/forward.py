"""Forward (live) residual prediction -- the shadow ML view.

`research/ml/` is otherwise a pure walk-forward *evaluator*: it fits per-fold and scores
against realised outcomes. This module is the one forward path -- fit the shipped Huber δ=4
residual model on ALL walk-forward history at once, then predict the residual for the
UPCOMING gameweek's players, giving `ep_total_ml = ep_total + predicted_residual`.

This never replaces the Quant EP and feeds no recommendation on its own. `scripts/
compute_ml_shadow.py` writes it to `data/dashboard/ml_shadow.json` for a side-by-side
"the ML view agrees / disagrees" panel; promoting it to an actual decision input stays a
separate, human-gated step (REPORT.md §10b's frozen-forward-test process).

Leakage: training uses `dataset_builder.build_minimal_dataset` (asof-scoped per step) and the
forward inference frame is built inside `backtest.asof_scope(con, target_season,
target_gameweek)` -- so the rolling features for the upcoming gameweek see only gameweeks
already played, exactly as a real pre-deadline run would.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from fpl_quant import backtest as bt

from . import contract as C
from . import dataset_builder, feature_engineering
from .residual_model import (
    LightGBMResidualModel,
    Preprocessor,
    ResidualModelUnavailableError,
)

MIN_TRAIN_ROWS = 200
HUBER_ALPHA = 4.0


def _forward_inference_frame(
    con: duckdb.DuckDBPyConnection, target_season: str, target_gameweek: int,
    ep_model_version: int, mm_model_version: int | None,
) -> pd.DataFrame:
    """The upcoming gameweek's players as a dataset row shape (identifiers + Q(x) + provenance,
    no label/residual), built inside the same asof shadow a real pre-deadline run would see."""
    deadline = bt.gameweek_deadline(con, target_season, target_gameweek)
    if deadline is None:
        return pd.DataFrame()
    with bt.asof_scope(con, target_season, target_gameweek):
        quant = dataset_builder._quant_predictions_for_step(con, ep_model_version)
        if quant.empty:
            return quant
        quant = dataset_builder._team_and_fixture_context(con, quant, target_season)
    quant[C.COL_SEASON] = target_season
    quant[C.COL_GAMEWEEK] = int(target_gameweek)
    quant[C.COL_EP_MODEL_VERSION] = int(ep_model_version)
    quant[C.COL_PRED_TIMESTAMP] = pd.Timestamp(deadline)
    quant["_mm_model_version"] = int(mm_model_version) if mm_model_version is not None else pd.NA
    return quant


def predict_forward(
    con: duckdb.DuckDBPyConnection, target_season: str, target_gameweek: int,
    ep_model_version: int, mm_model_version: int | None,
    *, min_train_rows: int | None = None, random_state: int = 42,
) -> pd.DataFrame | None:
    """Fit Huber δ=4 on the whole walk-forward history and predict `ep_total_ml` for
    `target_gameweek`. Returns a frame [player_uid, ep_quant, predicted_residual, ep_total_ml],
    or None when it cannot run: no `backtest_gameweek_steps` yet, too few training rows, no
    live `ep_outputs` for the target gameweek, or lightgbm not installed."""
    try:
        train = dataset_builder.build_minimal_dataset(con)
    except RuntimeError:
        return None  # no walk-forward history ingested yet
    train = feature_engineering.add_features(con, train)
    train = train.dropna(subset=[C.COL_RESIDUAL])
    if len(train) < (MIN_TRAIN_ROWS if min_train_rows is None else min_train_rows):
        return None

    feat_cols = feature_engineering.feature_columns()
    pp = Preprocessor().fit(train, feat_cols)
    x_tr, _ = pp.transform(train)
    y_tr = train[C.COL_RESIDUAL].to_numpy(dtype=float)
    try:
        model = LightGBMResidualModel(
            random_state=random_state, objective="huber", alpha=HUBER_ALPHA
        ).fit(x_tr, y_tr)
    except ResidualModelUnavailableError:
        return None

    fwd = _forward_inference_frame(con, target_season, target_gameweek, ep_model_version, mm_model_version)
    if fwd is None or fwd.empty:
        return None
    fwd = feature_engineering.add_features(con, fwd)
    x_fwd, _ = pp.transform(fwd)
    resid = model.predict(x_fwd)

    out = pd.DataFrame({
        "player_uid": fwd["player_uid"].to_numpy(),
        "ep_quant": fwd[C.COL_QUANT_PRED].to_numpy(dtype=float),
        "predicted_residual": resid,
    })
    out["ep_total_ml"] = out["ep_quant"] + out["predicted_residual"]
    out.attrs["n_train_rows"] = len(train)
    out.attrs["train_seasons"] = train[C.COL_SEASON].drop_duplicates().tolist()
    return out
