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


def fit_residual_model(
    con: duckdb.DuckDBPyConnection, *, min_train_rows: int | None = None, random_state: int = 42,
):
    """Fit the shipped Huber δ=4 residual model on the whole walk-forward history. Returns
    (model, preprocessor, meta) or None when it can't fit: no `backtest_gameweek_steps` yet,
    too few training rows, or lightgbm not installed. `meta` = {n_train_rows, train_seasons}."""
    try:
        train = dataset_builder.build_minimal_dataset(con)
    except RuntimeError:
        return None  # no walk-forward history ingested yet
    train = feature_engineering.add_features(con, train)
    train = train.dropna(subset=[C.COL_RESIDUAL])
    if len(train) < (MIN_TRAIN_ROWS if min_train_rows is None else min_train_rows):
        return None

    pp = Preprocessor().fit(train, feature_engineering.feature_columns())
    x_tr, _ = pp.transform(train)
    y_tr = train[C.COL_RESIDUAL].to_numpy(dtype=float)
    try:
        model = LightGBMResidualModel(
            random_state=random_state, objective="huber", alpha=HUBER_ALPHA
        ).fit(x_tr, y_tr)
    except ResidualModelUnavailableError:
        return None
    meta = {
        "n_train_rows": len(train),
        "train_seasons": train[C.COL_SEASON].drop_duplicates().tolist(),
    }
    return model, pp, meta


def _predict_one(con, model, pp, target_season, gameweek, ep_model_version, mm_model_version):
    """ep_total_ml for one gameweek's players. None if there is no live ep_outputs for it."""
    fwd = _forward_inference_frame(con, target_season, gameweek, ep_model_version, mm_model_version)
    if fwd is None or fwd.empty:
        return None
    fwd = feature_engineering.add_features(con, fwd)
    x_fwd, _ = pp.transform(fwd)
    resid = model.predict(x_fwd)
    out = pd.DataFrame({
        "player_uid": fwd["player_uid"].to_numpy(),
        "gameweek": int(gameweek),
        "ep_quant": fwd[C.COL_QUANT_PRED].to_numpy(dtype=float),
        "predicted_residual": resid,
    })
    out["ep_total_ml"] = out["ep_quant"] + out["predicted_residual"]
    return out


def predict_forward(
    con: duckdb.DuckDBPyConnection, target_season: str, target_gameweek: int,
    ep_model_version: int, mm_model_version: int | None,
    *, min_train_rows: int | None = None, random_state: int = 42,
) -> pd.DataFrame | None:
    """Fit and predict `ep_total_ml` for a single upcoming gameweek. Returns a frame
    [player_uid, ep_quant, predicted_residual, ep_total_ml] (+ .attrs), or None -- see
    fit_residual_model / _predict_one for the reasons."""
    fit = fit_residual_model(con, min_train_rows=min_train_rows, random_state=random_state)
    if fit is None:
        return None
    model, pp, meta = fit
    out = _predict_one(con, model, pp, target_season, target_gameweek, ep_model_version, mm_model_version)
    if out is None:
        return None
    out = out.drop(columns=["gameweek"])
    out.attrs.update(meta)
    return out


def write_ml_horizon_ep_versions(
    con: duckdb.DuckDBPyConnection, target_season: str,
    horizon_ep_versions: dict[int, tuple[int, int]], mm_model_version: int | None,
    *, min_train_rows: int | None = None,
) -> dict[int, tuple[int, int]] | None:
    """For each horizon gameweek, write a shadow copy of that gameweek's `ep_outputs` with
    every point-contributing column scaled by `ep_total_ml / ep_total` per player, under a new
    `ep_model_version`. Returns `{gw: (ml_ep_model_version, un_model_version)}` -- the
    uncertainty version is REUSED from the quant horizon (the residual model corrects the mean,
    not the variance) -- ready to hand to `transfer_planner.run(horizon_ep_versions=...)`.
    None if the model can't be fit.

    This is the only place `research/ml/` writes to the production `ep_*` tables, and it writes
    NEW versions only -- it never touches the quant rows. `scripts/run_ml_shadow_planner.py`
    is the sole caller; the shipped balanced recommendation is unaffected.
    """
    preds = predict_forward_horizon(
        con, target_season, horizon_ep_versions, mm_model_version, min_train_rows=min_train_rows
    )
    if preds is None:
        return None

    ml_versions: dict[int, tuple[int, int]] = {}
    for gw, (quant_ep_mv, un_mv) in horizon_ep_versions.items():
        gw_pred = preds[preds["gameweek"] == gw]
        if gw_pred.empty:
            continue
        ml_ep_mv = con.execute(
            """
            INSERT INTO ep_model_versions
                (calibration_asof_date, target_season, team_strength_model_version,
                 minutes_model_version, scoring_matrix_params_version, bps_params_version,
                 bps_tau_params_version)
            SELECT calibration_asof_date, target_season, team_strength_model_version,
                   minutes_model_version, scoring_matrix_params_version, bps_params_version,
                   bps_tau_params_version
            FROM ep_model_versions WHERE model_version = ?
            RETURNING model_version
            """,
            [quant_ep_mv],
        ).fetchone()[0]

        scale = gw_pred[["player_uid", "ep_quant", "ep_total_ml"]].copy()
        scale["s"] = scale["ep_total_ml"] / scale["ep_quant"].where(scale["ep_quant"].abs() > 1e-9, other=pd.NA)
        con.register("_ml_scale_df", scale[["player_uid", "s", "ep_total_ml"]])
        try:
            # scale the point-contributing components so explain_player_ep stays internally
            # consistent; ep_total is set to the ML value directly (the planner reads only this).
            # A player the ML model didn't cover (no scale row) keeps the quant row unchanged.
            con.execute(
                """
                INSERT INTO ep_outputs
                SELECT ?, o.player_uid, o.fixture_match_id,
                       o.ep_appearance * coalesce(sc.s, 1.0),
                       o.ep_goals * coalesce(sc.s, 1.0),
                       o.ep_assists * coalesce(sc.s, 1.0),
                       o.ep_clean_sheet * coalesce(sc.s, 1.0),
                       o.ep_goals_conceded,
                       o.ep_defcon * coalesce(sc.s, 1.0),
                       o.ep_bonus * coalesce(sc.s, 1.0),
                       o.ep_saves * coalesce(sc.s, 1.0),
                       o.ep_penalty_save,
                       o.ep_cards,
                       o.ep_own_goal,
                       coalesce(sc.ep_total_ml, o.ep_total),
                       o.expected_bps
                FROM ep_outputs o
                LEFT JOIN _ml_scale_df sc ON sc.player_uid = o.player_uid
                WHERE o.model_version = ?
                """,
                [ml_ep_mv, quant_ep_mv],
            )
        finally:
            con.unregister("_ml_scale_df")
        ml_versions[gw] = (ml_ep_mv, un_mv)

    return ml_versions or None


def predict_forward_horizon(
    con: duckdb.DuckDBPyConnection, target_season: str,
    horizon_ep_versions: dict[int, tuple[int, int]], mm_model_version: int | None,
    *, min_train_rows: int | None = None, random_state: int = 42,
) -> pd.DataFrame | None:
    """One model fit, then predict `ep_total_ml` for every gameweek in `horizon_ep_versions`
    (the {gw: (ep_model_version, un_model_version)} map transfer_planner.compute_horizon_ep
    returns). Returns a long frame [player_uid, gameweek, ep_quant, predicted_residual,
    ep_total_ml] (+ .attrs), or None if the model can't be fit. A horizon gameweek with no
    live ep_outputs is simply absent from the result, not fatal."""
    fit = fit_residual_model(con, min_train_rows=min_train_rows, random_state=random_state)
    if fit is None:
        return None
    model, pp, meta = fit
    parts = []
    for gw, (ep_mv, _un_mv) in sorted(horizon_ep_versions.items()):
        one = _predict_one(con, model, pp, target_season, gw, ep_mv, mm_model_version)
        if one is not None and not one.empty:
            parts.append(one)
    if not parts:
        return None
    out = pd.concat(parts, ignore_index=True)
    out.attrs.update(meta)
    return out
