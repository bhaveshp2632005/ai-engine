

import logging
import warnings
from dataclasses import dataclass
from typing      import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster      import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


@dataclass
class RegimeResult:
    regime:         str    # "Bull" | "Bear" | "Sideways"
    probability:    float  # 0.0–1.0
    regime_id:      int    # 0=Bear, 1=Sideways, 2=Bull
    volatility:     str    # "Low" | "Medium" | "High"
    trend_strength: float  # -1.0 to +1.0
    bull_weight:    float = 1.0
    bear_weight:    float = 1.0

    def to_dict(self) -> dict:
        return {
            "regime":        self.regime,
            "currentRegime": self.regime,
            "probability":   round(self.probability, 3),
            "volatility":    self.volatility,
            "trendStrength": round(self.trend_strength, 3),
            "modelWeights": {
                "bullWeight": round(self.bull_weight, 2),
                "bearWeight": round(self.bear_weight, 2),
            },
        }


_FALLBACK = RegimeResult(
    regime="Sideways", probability=0.55, regime_id=1,
    volatility="Medium", trend_strength=0.0)


class RegimeDetector:
    """
    KMeans-only regime detector (hmmlearn removed for free-tier compatibility).
    Guaranteed to never crash.
    """

    N_REGIMES = 3

    def __init__(self):
        self.scaler = StandardScaler()

    def detect(self, df: pd.DataFrame) -> RegimeResult:
        try:
            return self._detect_inner(df)
        except Exception as e:
            logger.warning("[Regime] detect crashed, using fallback: %s", e)
            return _FALLBACK

    def _detect_inner(self, df: pd.DataFrame) -> RegimeResult:
        if len(df) < 60:
            return self._rule_based(df)

        feats = self._build_features(df)
        if feats is None or len(feats) < 20:
            return self._rule_based(df)

        regime_id, prob = self._kmeans_regime(feats)

        vol   = self._volatility_label(df)
        ts    = self._trend_strength(df)
        label = {0: "Bear", 1: "Sideways", 2: "Bull"}[regime_id]

        if vol == "High":
            prob = min(prob, 0.68)

        prob = float(np.clip(prob, 0.40, 0.94))
        bull_w, bear_w = self._model_weights(label, ts)

        return RegimeResult(
            regime=label, probability=prob, regime_id=regime_id,
            volatility=vol, trend_strength=ts,
            bull_weight=bull_w, bear_weight=bear_w,
        )

    def _build_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        try:
            c       = df["close"].values.astype(float)
            if len(c) < 40:
                return None
            log_ret = np.log(np.maximum(c[1:], 1e-9) / np.maximum(c[:-1], 1e-9))
            log_ret = np.clip(log_ret, -0.5, 0.5)
            s       = pd.Series(log_ret)
            ret_5   = s.rolling(5,  min_periods=1).mean().values
            ret_20  = s.rolling(20, min_periods=5).mean().values
            vol_21  = s.rolling(21, min_periods=5).std().fillna(0).values * np.sqrt(252)
            vol_63  = s.rolling(63, min_periods=10).std().fillna(0).values * np.sqrt(252)
            cs      = pd.Series(c)
            sma50   = cs.rolling(50, min_periods=10).mean().values[1:]
            sma50   = np.where(sma50 > 0, sma50, 1e-9)
            trend   = (c[1:] - sma50) / sma50
            feat    = np.column_stack([ret_5, ret_20, vol_21, vol_63, trend])
            feat    = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
            return feat[-200:]
        except Exception as e:
            logger.warning("[Regime] feature build failed: %s", e)
            return None

    def _kmeans_regime(self, features: np.ndarray) -> Tuple[int, float]:
        X      = self.scaler.fit_transform(features)
        km     = KMeans(n_clusters=self.N_REGIMES, random_state=42,
                        n_init=15, max_iter=300)
        km.fit(X)
        labels = km.labels_

        mean_ret = {}
        for cid in range(self.N_REGIMES):
            mask = labels == cid
            mean_ret[cid] = float(features[mask, 1].mean()) if mask.any() else 0.0

        sorted_by_ret = sorted(mean_ret, key=mean_ret.get)
        semantic      = {sorted_by_ret[i]: i for i in range(self.N_REGIMES)}

        cur_cluster = int(km.predict(X[-1:].reshape(1, -1))[0])
        regime_id   = semantic[cur_cluster]
        recent      = labels[-30:]
        frac        = float(np.sum(recent == cur_cluster) / len(recent))
        prob        = float(np.clip(0.40 + frac * 0.55, 0.42, 0.90))
        return regime_id, prob

    def _volatility_label(self, df: pd.DataFrame) -> str:
        try:
            c       = df["close"].values.astype(float)
            log_ret = np.log(np.maximum(c[1:], 1e-9) / np.maximum(c[:-1], 1e-9))
            s       = pd.Series(log_ret)
            vol_21  = float(s.rolling(21, min_periods=5).std().iloc[-1]) * np.sqrt(252)
            vol_63  = float(s.rolling(63, min_periods=10).std().mean())  * np.sqrt(252)
            ratio   = vol_21 / max(vol_63, 1e-9)
            if ratio > 1.35: return "High"
            if ratio < 0.70: return "Low"
            return "Medium"
        except Exception:
            return "Medium"

    def _trend_strength(self, df: pd.DataFrame) -> float:
        try:
            c    = df["close"].values.astype(float)
            if len(c) < 50:
                return 0.0
            ma50 = float(np.mean(c[-50:]))
            cur  = float(c[-1])
            diff = (cur - ma50) / max(ma50, 1e-9)
            return float(np.clip(diff * 10, -1.0, 1.0))
        except Exception:
            return 0.0

    def _model_weights(self, regime: str, ts: float) -> Tuple[float, float]:
        if regime == "Bull":
            return (1.3 + ts * 0.2, 0.7)
        elif regime == "Bear":
            return (0.7, 1.3 + abs(ts) * 0.2)
        return (1.0, 1.0)

    def _rule_based(self, df: pd.DataFrame) -> RegimeResult:
        try:
            c   = df["close"].values.astype(float)
            n   = min(len(c), 5)
            if n < 2:
                return _FALLBACK
            ret = float(np.log(c[-1] / max(c[-n], 1e-9)))
            if ret > 0.02:
                return RegimeResult("Bull", 0.62, 2, "Medium", min(ret * 5, 1.0))
            if ret < -0.02:
                return RegimeResult("Bear", 0.62, 0, "Medium", max(ret * 5, -1.0))
        except Exception:
            pass
        return _FALLBACK
