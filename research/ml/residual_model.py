"""Residual models for the Phase-0 ML experiment.

All models predict the RESIDUAL r = y - Q(x), never the raw points -- per spec §9/§10, ML
never replaces the Quant model, it corrects it:

    ML_corrected_prediction = QuantPrediction + predicted_residual

Model 1 (spec §9): linear / ridge regression. Simple, interpretable, first out of the gate.
Model 2 (spec §10): gradient boosting via sklearn's HistGradientBoostingRegressor. Originally
the only nonlinear challenger, chosen specifically because it ships inside scikit-learn with no
extra native-dependency footprint, citing spec §10's "do not add a large dependency
unnecessarily" as the reason LightGBM/XGBoost/CatBoost were deliberately left out.
Model 3 (Track F, docs/plans/2026-08_retrospective_validation_and_ml_decision_plan.md, R8):
LightGBM, added later as the PRIMARY nonlinear challenger and the sole arm that governs the
ship/no-ship decision (R11) -- overriding spec §10's original dependency-minimalism stance was
an explicit operator decision (that plan's Q7), not a default. Model 2 (`quant_gbm`) is kept
exactly as it was and still runs, but is now reported as bonus/informational only (R16) -- it
was already a real, working arm before LightGBM existed, and removing it would throw away a
real (if secondary) result for no reason; it just no longer decides anything on its own.

scikit-learn is NOT a hard dependency of this repo's main requirements.txt (it lives in
requirements-research.txt instead, installed only for research/ml/ work) -- all three ML models
therefore lazy-import their backend and fail gracefully with a clear, actionable message if it's
absent, so the rest of the pipeline (Quant baseline, historical baseline, dataset build, leakage
checks) still runs and produces results without any of them (R14).

Preprocessing (spec §9: "handle missing values explicitly, do not silently drop large
portions of the dataset"):
- Numeric features: missing values imputed with the TRAINING set's median (fit on train,
  applied to test -- never computed from test data, which would leak test-set statistics into
  the model).
- Categorical features (position, status): one-hot encoded, categories fixed from the training
  set; an unseen test-time category maps to the all-zero row rather than crashing. `position` is
  an approved static-identity feature source (EXISTING_MODEL_AUDIT.md §9, LEAKAGE_PROTOCOL.md
  §4) that dataset_builder already attaches to every row, but it was not actually listed here
  until this feature-audit pass found the gap -- no model (linear, quant_gbm, or quant_lightgbm)
  was ever conditioning on it, despite a goalkeeper's and a forward's point distributions
  differing enormously.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CATEGORICAL_FEATURES = ("status", "position")


class ResidualModelUnavailableError(RuntimeError):
    """Raised when a model's required dependency (e.g. scikit-learn) is not installed."""


@dataclass
class Preprocessor:
    """Fit on training data only; transform train and test identically. Prevents leakage of
    test-set statistics (medians, category sets) into the feature matrix."""

    numeric_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    medians_: dict[str, float] = field(default_factory=dict)
    categories_: dict[str, list[str]] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, feature_cols: list[str]) -> "Preprocessor":
        self.categorical_cols = [c for c in feature_cols if c in CATEGORICAL_FEATURES and c in df.columns]
        self.numeric_cols = [c for c in feature_cols if c not in self.categorical_cols and c in df.columns]
        for c in self.numeric_cols:
            med = pd.to_numeric(df[c], errors="coerce").median()
            self.medians_[c] = 0.0 if pd.isna(med) else float(med)
        for c in self.categorical_cols:
            self.categories_[c] = sorted(df[c].dropna().astype(str).unique().tolist())
        return self

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        cols: list[np.ndarray] = []
        names: list[str] = []
        for c in self.numeric_cols:
            series = pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series(np.nan, index=df.index)
            filled = series.fillna(self.medians_.get(c, 0.0))
            cols.append(filled.to_numpy(dtype=float))
            names.append(c)
        for c in self.categorical_cols:
            series = df[c].astype(str) if c in df.columns else pd.Series("unknown", index=df.index)
            for cat in self.categories_.get(c, []):
                cols.append((series == cat).to_numpy(dtype=float))
                names.append(f"{c}={cat}")
        if not cols:
            return np.zeros((len(df), 0)), names
        return np.column_stack(cols), names


# ============================================================
# Model 1: Linear / Ridge residual model
# ============================================================

class LinearResidualModel:
    """Ridge regression predicting the residual. sklearn preferred; a closed-form numpy ridge
    (no external ML dependency) is used automatically if sklearn is unavailable, so this model
    always works."""

    name = "quant_linear"

    def __init__(self, alpha: float = 1.0, random_state: int = 42):
        self.alpha = alpha
        self.random_state = random_state
        self._backend = None
        self._coef: np.ndarray | None = None
        self._intercept: float = 0.0
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearResidualModel":
        try:
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=self.alpha, random_state=self.random_state)
            model.fit(X, y)
            self._backend = model
            return self
        except ImportError:
            pass
        # numpy closed-form ridge fallback: standardize, solve (X'X + alpha*I)^-1 X'y
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0)
        self._std[self._std < 1e-12] = 1.0
        Xs = (X - self._mean) / self._std
        Xb = np.column_stack([np.ones(len(Xs)), Xs])
        n_features = Xb.shape[1]
        reg = self.alpha * np.eye(n_features)
        reg[0, 0] = 0.0  # do not regularise the intercept
        coef = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y)
        self._intercept = float(coef[0])
        self._coef = coef[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._backend is not None:
            return self._backend.predict(X)
        Xs = (X - self._mean) / self._std
        return self._intercept + Xs @ self._coef

    def feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
        """Absolute standardized coefficient magnitude, signed direction."""
        if self._backend is not None:
            coef = np.asarray(self._backend.coef_, dtype=float)
        else:
            coef = self._coef if self._coef is not None else np.zeros(len(feature_names))
        return pd.DataFrame({
            "feature": feature_names,
            "importance": np.abs(coef),
            "direction": np.sign(coef),
        }).sort_values("importance", ascending=False).reset_index(drop=True)


# ============================================================
# Model 2: Gradient boosting residual model
# ============================================================

class GradientBoostingResidualModel:
    """HistGradientBoostingRegressor predicting the residual. Requires scikit-learn; raises
    ResidualModelUnavailableError (caught by the experiment orchestrator, not fatal to the
    rest of the pipeline) if it is not installed OR if it fails to run -- same fix, and same
    reasoning, as LightGBMResidualModel's own fit() (see its docstring); applied here too for
    consistency even though R14's text names LightGBM specifically."""

    name = "quant_gbm"

    def __init__(self, max_depth: int = 4, max_iter: int = 100, learning_rate: float = 0.05,
                 random_state: int = 42, loss: str = "squared_error"):
        self.max_depth = max_depth
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.random_state = random_state
        # "squared_error" = L2. "absolute_error" aligns the training loss with the experiment's
        # MAE metric and is robust to the heavy right tail of FPL points -- see
        # LightGBMResidualModel.__init__ for the full reasoning; kept a constructor default so
        # nothing that builds this class unexpectedly changes behaviour.
        self.loss = loss
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingResidualModel":
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
        except ImportError as exc:
            raise ResidualModelUnavailableError(
                "scikit-learn is not installed -- GradientBoostingResidualModel requires it. "
                "Install with `pip install scikit-learn` to enable this model; the Quant "
                "baseline, historical baseline, and linear residual model all still run without it."
            ) from exc
        try:
            model = HistGradientBoostingRegressor(
                loss=self.loss,
                max_depth=self.max_depth, max_iter=self.max_iter,
                learning_rate=self.learning_rate, random_state=self.random_state,
            )
            model.fit(X, y)
        except Exception as exc:
            raise ResidualModelUnavailableError(
                f"scikit-learn HistGradientBoostingRegressor failed to fit ({type(exc).__name__}: {exc}) "
                "-- treating this arm as unavailable for this fold rather than aborting the rest of the experiment."
            ) from exc
        self._model = model
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ResidualModelUnavailableError("model was never fit (sklearn unavailable)")
        return self._model.predict(X)

    def feature_importance(self, X: np.ndarray, y: np.ndarray, feature_names: list[str], n_repeats: int = 5) -> pd.DataFrame:
        """Permutation importance -- model-agnostic, appropriate for a tree ensemble (spec §19)."""
        try:
            from sklearn.inspection import permutation_importance
        except ImportError as exc:
            raise ResidualModelUnavailableError("scikit-learn is not installed") from exc
        result = permutation_importance(
            self._model, X, y, n_repeats=n_repeats, random_state=self.random_state, scoring="neg_mean_absolute_error",
        )
        return pd.DataFrame({
            "feature": feature_names,
            "importance": result.importances_mean,
            "direction": np.sign(result.importances_mean),
        }).sort_values("importance", ascending=False).reset_index(drop=True)


# ============================================================
# Model 3: LightGBM residual model -- the primary nonlinear challenger (Track F, R8/R11)
# ============================================================

class LightGBMResidualModel:
    """LGBMRegressor predicting the residual. Requires lightgbm; raises
    ResidualModelUnavailableError (caught by the experiment orchestrator, same as
    GradientBoostingResidualModel -- not fatal to the rest of the pipeline, R14) if it is not
    installed OR if it fails to run. This is the arm R11's ship/no-ship decision is governed
    by -- see module docstring for why it exists alongside, not instead of, the original
    quant_gbm arm.

    R14's own text is "fails to install OR RUN" -- a Critique Engine pass on this phase found
    the first version only caught ImportError, so a real runtime failure (LightGBM's native
    library has a known history of environment-specific issues, e.g. OpenMP/threading problems
    on Windows) would propagate raw and abort the entire multi-fold experiment run, not just
    this one arm. Fixed: the actual fit() call is wrapped too, not just the import."""

    name = "quant_lightgbm"

    def __init__(
        self,
        max_depth: int = 4,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        # No L1/L2 or row/column subsampling by default is fine for large datasets, but
        # exhaustive gameweek walk-forward means the earliest folds train on a few hundred
        # rows, where that risks memorising noise rather than finding real signal -- a
        # one-time, principled choice of more conservative defaults (not a hyperparameter
        # search, which spec §3 forbids; this is picking better defaults once).
        reg_alpha: float = 0.1,
        reg_lambda: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        # "regression" = L2. The experiment's primary metric is MAE (evaluate.py), so training
        # the objective LightGBM optimises to L1 ("regression_l1") aligns the loss with the
        # metric and is more robust to the heavy right tail of FPL points (a rare double-digit
        # haul). Same category as the reg_alpha/subsample defaults above: a one-time principled
        # choice, not a hyperparameter search.
        objective: str = "regression",
    ):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.objective = objective
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LightGBMResidualModel":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ResidualModelUnavailableError(
                "lightgbm is not installed -- LightGBMResidualModel requires it. Install with "
                "`pip install lightgbm` (see requirements-research.txt) to enable this model; "
                "the Quant baseline, historical baseline, linear residual model, and quant_gbm "
                "(if scikit-learn is available) all still run without it."
            ) from exc
        try:
            # verbose=-1 silences LightGBM's own stdout logging (e.g. "[LightGBM] [Info] ..."),
            # which would otherwise interleave with this project's own per-fold progress output
            # across potentially dozens of walk-forward folds.
            model = LGBMRegressor(
                objective=self.objective,
                max_depth=self.max_depth, n_estimators=self.n_estimators,
                learning_rate=self.learning_rate, reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda, subsample=self.subsample, subsample_freq=1,
                colsample_bytree=self.colsample_bytree,
                random_state=self.random_state, verbose=-1,
            )
            model.fit(X, y)
        except Exception as exc:
            # R14: a genuine runtime failure (not an import problem) must degrade the same way
            # an unavailable install does -- one exception type for the experiment orchestrator
            # to catch, regardless of which reason this arm couldn't produce a result.
            raise ResidualModelUnavailableError(
                f"lightgbm failed to fit ({type(exc).__name__}: {exc}) -- treating this arm as "
                "unavailable for this fold rather than aborting the rest of the experiment."
            ) from exc
        self._model = model
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ResidualModelUnavailableError("model was never fit (lightgbm unavailable)")
        return self._model.predict(X)

    def feature_importance(self, X: np.ndarray, y: np.ndarray, feature_names: list[str], n_repeats: int = 5) -> pd.DataFrame:
        """Permutation importance -- model-agnostic (same technique as
        GradientBoostingResidualModel's own feature_importance, deliberately reused rather than
        LightGBM's built-in split/gain importances, so all nonlinear arms report importance on
        the same footing)."""
        try:
            from sklearn.inspection import permutation_importance
        except ImportError as exc:
            raise ResidualModelUnavailableError(
                "scikit-learn is not installed -- LightGBMResidualModel's feature_importance "
                "reuses sklearn.inspection.permutation_importance (it works on any estimator "
                "with predict(), not just sklearn's own models)."
            ) from exc
        result = permutation_importance(
            self._model, X, y, n_repeats=n_repeats, random_state=self.random_state, scoring="neg_mean_absolute_error",
        )
        return pd.DataFrame({
            "feature": feature_names,
            "importance": result.importances_mean,
            "direction": np.sign(result.importances_mean),
        }).sort_values("importance", ascending=False).reset_index(drop=True)


# ============================================================
# Model 4: XGBoost residual model -- independent-implementation confirmation of the LightGBM
# result (Track F R13: "Flag whether XGBoost should be added as a further, independent-
# implementation confirmation"). It does NOT govern the R11 ship/no-ship decision (only
# quant_lightgbm does) -- it exists to answer a different question: if two independently
# implemented gradient-boosting libraries agree that there is (or isn't) out-of-sample signal
# in the residual, that agreement is itself evidence the result isn't an artifact of one
# library's specific defaults/splitting heuristic. Reported alongside quant_gbm as
# informational (R16-style), never substituting for quant_lightgbm's own numbers.
# ============================================================

class XGBoostResidualModel:
    """XGBRegressor predicting the residual. Requires xgboost; raises
    ResidualModelUnavailableError (caught by the experiment orchestrator, same treatment as
    GradientBoostingResidualModel/LightGBMResidualModel, R14) if it is not installed OR if it
    fails to run, so a real runtime failure degrades the same way an absent install does and
    never aborts the rest of the multi-fold experiment run."""

    name = "quant_xgboost"

    def __init__(
        self,
        max_depth: int = 4,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        # Same reasoning as LightGBMResidualModel's own defaults: exhaustive gameweek
        # walk-forward means the earliest folds train on a few hundred rows, where
        # unregularised boosting risks memorising noise rather than finding real signal.
        reg_alpha: float = 0.1,
        reg_lambda: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        # "reg:squarederror" = L2. "reg:absoluteerror" aligns the training loss with the
        # experiment's MAE metric -- see LightGBMResidualModel.__init__ for the reasoning.
        # Constructor default preserved so nothing that builds this class changes behaviour.
        objective: str = "reg:squarederror",
    ):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.objective = objective
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostResidualModel":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ResidualModelUnavailableError(
                "xgboost is not installed -- XGBoostResidualModel requires it. Install with "
                "`pip install xgboost` (see requirements-research.txt) to enable this model; "
                "the Quant baseline, historical baseline, linear residual model, quant_gbm, and "
                "quant_lightgbm (whichever dependencies are available) all still run without it."
            ) from exc
        try:
            model = XGBRegressor(
                objective=self.objective,
                max_depth=self.max_depth, n_estimators=self.n_estimators,
                learning_rate=self.learning_rate, reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda, subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=self.random_state, verbosity=0,
            )
            model.fit(X, y)
        except Exception as exc:
            raise ResidualModelUnavailableError(
                f"xgboost failed to fit ({type(exc).__name__}: {exc}) -- treating this arm as "
                "unavailable for this fold rather than aborting the rest of the experiment."
            ) from exc
        self._model = model
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ResidualModelUnavailableError("model was never fit (xgboost unavailable)")
        return self._model.predict(X)

    def feature_importance(self, X: np.ndarray, y: np.ndarray, feature_names: list[str], n_repeats: int = 5) -> pd.DataFrame:
        """Permutation importance -- same model-agnostic technique GradientBoostingResidualModel
        and LightGBMResidualModel already use, so all nonlinear arms report importance on the
        same footing rather than each using its own library's built-in (differently-scaled)
        importance measure."""
        try:
            from sklearn.inspection import permutation_importance
        except ImportError as exc:
            raise ResidualModelUnavailableError(
                "scikit-learn is not installed -- XGBoostResidualModel's feature_importance "
                "reuses sklearn.inspection.permutation_importance (it works on any estimator "
                "with predict(), not just sklearn's own models)."
            ) from exc
        result = permutation_importance(
            self._model, X, y, n_repeats=n_repeats, random_state=self.random_state, scoring="neg_mean_absolute_error",
        )
        return pd.DataFrame({
            "feature": feature_names,
            "importance": result.importances_mean,
            "direction": np.sign(result.importances_mean),
        }).sort_values("importance", ascending=False).reset_index(drop=True)


def sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401
        return True
    except ImportError:
        return False


def lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401
        return True
    except ImportError:
        return False


def xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401
        return True
    except ImportError:
        return False
