"""Residual models for the Phase-0 ML experiment.

Both models predict the RESIDUAL r = y - Q(x), never the raw points -- per spec §9/§10, ML
never replaces the Quant model, it corrects it:

    ML_corrected_prediction = QuantPrediction + predicted_residual

Model 1 (spec §9): linear / ridge regression. Simple, interpretable, first out of the gate.
Model 2 (spec §10): gradient boosting. sklearn's HistGradientBoostingRegressor is used because
it ships inside scikit-learn with no extra native-dependency footprint (unlike XGBoost/LightGBM/
CatBoost, which are NOT in this repo's requirements.txt/requirements.lock and are not added
here per spec §10's "do not add a large dependency unnecessarily"). scikit-learn itself is also
NOT currently a dependency of this repo -- both models therefore lazy-import sklearn and fail
gracefully with a clear, actionable message if it is absent, so the rest of the pipeline
(Quant baseline, historical baseline, dataset build, leakage checks) still runs and produces
results without it.

Preprocessing (spec §9: "handle missing values explicitly, do not silently drop large
portions of the dataset"):
- Numeric features: missing values imputed with the TRAINING set's median (fit on train,
  applied to test -- never computed from test data, which would leak test-set statistics into
  the model).
- Categorical features (position, status): one-hot encoded, categories fixed from the training
  set; an unseen test-time category maps to the all-zero row rather than crashing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CATEGORICAL_FEATURES = ("status",)


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
    rest of the pipeline) if it is not installed."""

    name = "quant_gbm"

    def __init__(self, max_depth: int = 4, max_iter: int = 100, learning_rate: float = 0.05, random_state: int = 42):
        self.max_depth = max_depth
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.random_state = random_state
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
        self._model = HistGradientBoostingRegressor(
            max_depth=self.max_depth, max_iter=self.max_iter,
            learning_rate=self.learning_rate, random_state=self.random_state,
        )
        self._model.fit(X, y)
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


def sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401
        return True
    except ImportError:
        return False
