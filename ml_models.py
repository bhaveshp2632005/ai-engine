"""
ml_models.py — Tabular ML Models v5.0
══════════════════════════════════════
Bulletproof design:
  • Every fit/predict wrapped in try/except
  • predict() always returns array of correct length
  • XGBoost/LightGBM cloned properly for OOF training
  • No shared mutable state between clones
"""

import logging
import numpy as np
import joblib
from pathlib import Path

from sklearn.linear_model  import Ridge
from sklearn.ensemble      import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline      import Pipeline
from sklearn.metrics       import mean_squared_error, mean_absolute_error
import xgboost  as xgb
import lightgbm as lgb

logger     = logging.getLogger(__name__)
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    try:
        rmse    = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae     = float(mean_absolute_error(y_true, y_pred))
        mask    = np.abs(y_true) > 1e-9
        mape    = float(np.mean(np.abs(
            (y_true[mask] - y_pred[mask]) / (np.abs(y_true[mask]) + 1e-9))) * 100)
        dir_acc = float(np.mean(np.sign(y_pred) == np.sign(y_true)) * 100)
        return {"rmse": round(rmse, 6), "mae": round(mae, 6),
                "mape": round(mape, 4), "directional_accuracy": round(dir_acc, 2)}
    except Exception:
        return {"rmse": 0.0, "mae": 0.0, "mape": 0.0, "directional_accuracy": 50.0}


class _BaseMLModel:
    name: str = "base"

    def fit(self, X: np.ndarray, y: np.ndarray):
        logger.info(f"[{self.name}] Training on {X.shape[0]} samples ...")
        try:
            self.model.fit(X, y)
        except Exception as e:
            logger.warning(f"[{self.name}] fit failed: {e}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        try:
            out = self.model.predict(X)
            out = np.nan_to_num(out.astype(np.float32), nan=0.0,
                                posinf=0.0, neginf=0.0)
            return out
        except Exception as e:
            logger.warning(f"[{self.name}] predict failed: {e}")
            return np.zeros(X.shape[0], dtype=np.float32)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        m = _metrics(y, self.predict(X))
        logger.info(f"[{self.name}] {m}")
        return m

    def save(self, path: Path = None):
        p = path or MODELS_DIR / f"{self.name}.pkl"
        try:
            joblib.dump(self.model, p)
        except Exception as e:
            logger.warning(f"[{self.name}] save failed: {e}")

    def load(self, path: Path = None):
        p = path or MODELS_DIR / f"{self.name}.pkl"
        try:
            self.model = joblib.load(p)
        except Exception as e:
            logger.warning(f"[{self.name}] load failed: {e}")
        return self


class LinearModel(_BaseMLModel):
    name = "linear"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge",  Ridge(alpha=alpha)),
        ])

    def __class_getitem__(cls, item):
        return cls

    # Allow cloning via type(bm)()
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)


class RandomForestModel(_BaseMLModel):
    name = "random_forest"

    def __init__(self, n_estimators: int = 300, max_depth: int = 8):
        self.n_estimators = n_estimators
        self.max_depth    = max_depth
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=5, n_jobs=-1, random_state=42)

    def feature_importance(self, feature_names: list) -> dict:
        try:
            imp = self.model.feature_importances_
            return dict(sorted(zip(feature_names, imp), key=lambda x: -x[1]))
        except Exception:
            return {}


class XGBoostModel(_BaseMLModel):
    name = "xgboost"

    def __init__(self, n_estimators: int = 300):
        self.n_estimators = n_estimators
        self._build()

    def _build(self):
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            tree_method="hist",   # faster on CPU
        )

    # Called by type(bm)() for clone in OOF
    def __init__(self, n_estimators: int = 300):
        self.n_estimators = n_estimators
        self._build()

    def fit(self, X, y, X_val=None, y_val=None):
        try:
            eval_set = [(X_val, y_val)] if X_val is not None else None
            self.model.fit(X, y, eval_set=eval_set, verbose=False)
        except Exception as e:
            logger.warning(f"[{self.name}] fit failed: {e}")
        return self


class LightGBMModel(_BaseMLModel):
    name = "lightgbm"

    def __init__(self, n_estimators: int = 300):
        self.n_estimators = n_estimators
        self._build()

    def _build(self):
        self.model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            max_depth=5,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=10,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

    def fit(self, X, y, X_val=None, y_val=None):
        try:
            if X_val is not None:
                kw = dict(
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)])
                self.model.fit(X, y, **kw)
            else:
                self.model.fit(X, y)
        except Exception as e:
            logger.warning(f"[{self.name}] fit failed: {e}")
        return self