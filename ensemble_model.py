"""
ensemble_model.py — PRODUCTION Lightweight Ensemble
═════════════════════════════════════════════════════
CHANGES vs dev version:
  ✗ REMOVED: LSTMModel, GRUModel, TemporalFusionModel (torch removed)
  ✗ REMOVED: HybridEnsemble (uses seq models)
  ✗ REMOVED: MC Dropout uncertainty (torch-only)
  ✓ KEPT:    LinearModel, XGBoostModel, LightGBMModel
  ✓ KEPT:    GradientBoosting meta-learner
  ✓ KEPT:    Sentiment blending
  ✓ KEPT:    Confidence scoring
  ✓ ADDED:   LightEnsemble class (tabular-only, no seq models)
  ✓ NOTE:    HybridEnsemble alias kept for backward compat
"""

import logging
import pickle
from dataclasses import dataclass
from pathlib     import Path
from typing      import Optional, Tuple

import numpy as np
from sklearn.ensemble        import GradientBoostingRegressor
from sklearn.linear_model    import Ridge
from sklearn.metrics         import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing   import StandardScaler

from feature_engineering import FeatureEngineer
from ml_models           import LightGBMModel, LinearModel, XGBoostModel
from sentiment_analysis  import SentimentAnalyzer

logger     = logging.getLogger(__name__)
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

_MAX_RETURN_PCT  = 25.0
_MIN_CI_HALF_PCT = 1.5
_MAX_CI_HALF_PCT = 12.0


@dataclass
class PredictionResult:
    symbol:           str
    current_price:    float
    predicted_price:  float
    price_low:        float
    price_high:       float
    predicted_return: float
    trend:            str
    confidence:       float
    horizon_days:     int
    sentiment:        dict
    model_metrics:    dict
    feature_count:    int
    currency:         str = "USD"

    def to_dict(self) -> dict:
        return {
            "symbol":          self.symbol,
            "currentPrice":    round(self.current_price,   2),
            "predictedPrice":  round(self.predicted_price, 2),
            "priceRange": {
                "low":  round(self.price_low,  2),
                "high": round(self.price_high, 2),
            },
            "predictedReturn": round(self.predicted_return, 2),
            "trend":           self.trend,
            "confidence":      round(self.confidence, 1),
            "horizonDays":     self.horizon_days,
            "sentiment":       self.sentiment,
            "modelMetrics":    self.model_metrics,
            "featureCount":    self.feature_count,
            "currency":        self.currency,
        }


class LightEnsemble:
    """
    Tabular-only stacking ensemble (no LSTM/GRU/torch).
    L0: LinearModel + XGBoostModel + LightGBMModel
    L1: GradientBoosting meta-learner
    ~10x faster than HybridEnsemble on CPU.
    """

    def __init__(
        self,
        horizon:          int   = 5,
        sentiment_weight: float = 0.12,
    ):
        self.horizon          = horizon
        self.sentiment_weight = sentiment_weight
        self.fe               = FeatureEngineer()

        self.tabular_models = [
            LinearModel(),
            XGBoostModel(n_estimators=150),
            LightGBMModel(n_estimators=150),
        ]
        self.meta_model  = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=5, random_state=42,
        )
        self.meta_scaler        = StandardScaler()
        self.sentiment_analyzer = SentimentAnalyzer()
        self._is_fitted         = False

    # ── Public ────────────────────────────────────────────────────────────────

    def predict(
        self,
        symbol:         str,
        df_raw,
        currency:       str  = "USD",
        skip_sentiment: bool = False,
    ) -> PredictionResult:
        logger.info("[LightEnsemble] Predicting %s …", symbol)

        df_feat = self.fe.transform(df_raw)
        X_tab, y_tab, feat_names = self.fe.build_supervised(df_feat, self.horizon)

        # Sentiment
        ss, sr = self._get_sentiment(symbol, skip_sentiment)

        # Train L0 tabular models (OOF)
        tab_oof = self._train_tabular(X_tab, y_tab)

        # Train L1 meta-learner
        # Features: [tab_pred, sentiment_score] — 2 features
        ml = len(tab_oof)
        meta_X  = np.column_stack([tab_oof, np.full(ml, ss, dtype=np.float32)])
        meta_Xs = self.meta_scaler.fit_transform(meta_X)
        if len(meta_Xs) >= 20:
            self.meta_model.fit(meta_Xs, y_tab[-ml:])
        else:
            fb = Ridge(alpha=1.0)
            fb.fit(meta_Xs, y_tab[-ml:])
            self.meta_model = fb
        self._is_fitted = True

        # Final prediction
        pred_log, ci_lo_log, ci_hi_log, conf = self._final_predict(X_tab, ss)

        # Convert log-return → %
        pred_pct  = float(pred_log  * 100)
        ci_lo_pct = float(ci_lo_log * 100)
        ci_hi_pct = float(ci_hi_log * 100)

        # Sentiment nudge
        sentiment_nudge = float(np.clip(ss * 1.5, -1.5, 1.5))
        pred_pct = (pred_pct * (1 - self.sentiment_weight)
                    + sentiment_nudge * self.sentiment_weight)

        # Direction agreement boost
        preds_dir = [np.sign(float(m.predict(X_tab[[-1]])[0])) for m in self.tabular_models]
        if len(set(preds_dir)) == 1 and preds_dir[0] != 0:
            conf = min(conf + 5.0, 85.0)
        if ss != 0 and np.sign(ss) == np.sign(pred_pct):
            conf = min(conf + 3.0, 85.0)

        # Price targets
        cp = float(df_raw["close"].iloc[-1])
        assert cp > 0, f"Current price must be positive, got {cp}"

        pred_pct  = float(np.clip(pred_pct,  -_MAX_RETURN_PCT, _MAX_RETURN_PCT))
        ci_lo_pct = float(np.clip(ci_lo_pct, -_MAX_RETURN_PCT, _MAX_RETURN_PCT))
        ci_hi_pct = float(np.clip(ci_hi_pct, -_MAX_RETURN_PCT, _MAX_RETURN_PCT))

        pp = cp * (1 + pred_pct  / 100)
        pl = cp * (1 + ci_lo_pct / 100)
        ph = cp * (1 + ci_hi_pct / 100)

        pl = max(pl, cp * 0.50)
        ph = max(ph, cp * 1.005)
        pl = min(pl, pp * 0.99)
        ph = max(ph, pp * 1.01)

        # Trend
        trend = "Bullish" if pred_pct > 1.0 else ("Bearish" if pred_pct < -1.0 else "Neutral")
        if trend == "Bullish" and ss < -0.35: trend = "Neutral"
        if trend == "Bearish" and ss >  0.35: trend = "Neutral"

        # Metrics
        y_pred_meta = self.meta_model.predict(meta_Xs)
        y_true_meta = y_tab[-ml:]
        dir_acc = float(np.mean(np.sign(y_true_meta) == np.sign(y_pred_meta)) * 100)
        metrics = {
            "rmse": round(float(np.sqrt(mean_squared_error(y_true_meta, y_pred_meta))), 6),
            "mae":  round(float(mean_absolute_error(y_true_meta, y_pred_meta)), 6),
            "directional_accuracy": round(dir_acc, 2),
        }
        dir_boost = float(np.clip((dir_acc - 50.0) * 0.6, 0.0, 15.0))
        conf = min(conf + dir_boost, 85.0)

        logger.info("[LightEnsemble] %s → %s %+.2f%% conf=%.0f%%",
                    symbol, trend, pred_pct, conf)

        return PredictionResult(
            symbol=symbol, current_price=cp,
            predicted_price=round(pp, 2),
            price_low=round(pl, 2), price_high=round(ph, 2),
            predicted_return=round(pred_pct, 2),
            trend=trend, confidence=round(conf, 1),
            horizon_days=self.horizon,
            sentiment=sr, model_metrics=metrics,
            feature_count=len(feat_names), currency=currency,
        )

    # ── Sentiment ─────────────────────────────────────────────────────────────

    def _get_sentiment(self, symbol: str, skip: bool) -> Tuple[float, dict]:
        _neutral = {"score": 0.0, "label": "Neutral", "confidence": 0.0,
                    "article_count": 0, "model_used": "skipped"}
        if skip:
            return 0.0, _neutral
        try:
            s = self.sentiment_analyzer.analyze(symbol)
            return float(s.score), s.to_dict()
        except Exception as e:
            logger.warning("Sentiment failed: %s", e)
            return 0.0, {**_neutral, "model_used": "error", "error": str(e)}

    # ── Training ──────────────────────────────────────────────────────────────

    def _train_tabular(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        tscv = TimeSeriesSplit(n_splits=5)
        oof  = np.zeros((len(y), len(self.tabular_models)), dtype=np.float32)
        for _, (tr, va) in enumerate(tscv.split(X)):
            for j, bm in enumerate(self.tabular_models):
                try:
                    clone = type(bm)()
                    clone.fit(X[tr], y[tr])
                    oof[va, j] = clone.predict(X[va])
                except Exception as e:
                    logger.warning("Tab OOF fold failed [%s]: %s", bm.name, e)
        for bm in self.tabular_models:
            try:
                bm.fit(X, y)
            except Exception as e:
                logger.warning("Tab refit failed [%s]: %s", bm.name, e)
        return oof.mean(axis=1)

    # ── Final Prediction ──────────────────────────────────────────────────────

    def _final_predict(
        self,
        X_tab: np.ndarray,
        ss: float,
    ) -> Tuple[float, float, float, float]:
        tab_p = []
        for bm in self.tabular_models:
            try:
                v = float(bm.predict(X_tab[[-1]])[0])
                if np.isfinite(v):
                    tab_p.append(v)
            except Exception:
                pass
        if not tab_p:
            tab_p = [0.0]

        tab_mean  = float(np.mean(tab_p))
        meta_feat = np.array([tab_mean, ss], dtype=np.float32).reshape(1, -1)

        try:
            pred_log = float(
                self.meta_model.predict(
                    self.meta_scaler.transform(meta_feat))[0])
            if not np.isfinite(pred_log) or abs(pred_log) > 1.0:
                pred_log = tab_mean
        except Exception:
            pred_log = tab_mean

        pred_log = float(np.clip(pred_log, -0.25, 0.25))

        all_preds = np.array(tab_p, dtype=np.float32)
        all_preds = all_preds[np.isfinite(all_preds)]
        std_log   = float(np.std(all_preds)) if len(all_preds) > 1 else 0.01

        conf = float(52.0 + 33.0 * np.exp(-std_log / 0.012))
        conf = float(np.clip(conf, 52.0, 85.0))

        ci_half_min = _MIN_CI_HALF_PCT / 100
        ci_half_max = _MAX_CI_HALF_PCT / 100
        ci_lo_log   = pred_log - max(std_log * 1.96, ci_half_min)
        ci_hi_log   = pred_log + max(std_log * 1.96, ci_half_min)

        if ci_lo_log >= pred_log:
            ci_lo_log = pred_log - ci_half_min
        if ci_hi_log <= pred_log:
            ci_hi_log = pred_log + ci_half_min

        ci_lo_log = max(ci_lo_log, pred_log - ci_half_max)
        ci_hi_log = min(ci_hi_log, pred_log + ci_half_max)

        return pred_log, ci_lo_log, ci_hi_log, conf

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, symbol: str):
        path = MODELS_DIR / f"{symbol}_light_ensemble.pkl"
        try:
            with open(path, "wb") as f:
                pickle.dump({
                    "meta_model":  self.meta_model,
                    "meta_scaler": self.meta_scaler,
                }, f)
            logger.info("[LightEnsemble] Saved → %s", path)
        except Exception as e:
            logger.warning("[LightEnsemble] Save failed: %s", e)

    def load(self, symbol: str) -> bool:
        path = MODELS_DIR / f"{symbol}_light_ensemble.pkl"
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                s = pickle.load(f)
            self.meta_model   = s["meta_model"]
            self.meta_scaler  = s["meta_scaler"]
            self._is_fitted   = True
            logger.info("[LightEnsemble] Loaded ← %s", path)
            return True
        except Exception as e:
            logger.warning("[LightEnsemble] Load failed: %s", e)
            return False


# ── Backward compatibility alias ──────────────────────────────────────────────
# Old code that imports HybridEnsemble will get LightEnsemble instead.
HybridEnsemble = LightEnsemble
