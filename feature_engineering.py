
import logging
import numpy  as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Adds 70+ features to OHLCV data and builds ML matrices.

    Usage
    -----
    fe      = FeatureEngineer()
    df_feat = fe.transform(df_1d, macro_df=macro)
    X, y, names = fe.build_supervised(df_feat)
    X_seq,  y   = fe.build_sequences(df_feat)
    """

    def transform(self, df: pd.DataFrame,
                  df_htf: pd.DataFrame = None,
                  macro_df: pd.DataFrame = None) -> pd.DataFrame:
        d = df.copy()
        d.columns = [c.lower() for c in d.columns]
        d = self._trend(d)
        d = self._momentum(d)
        d = self._volatility(d)
        d = self._volume(d)
        d = self._price_features(d)
        d = self._statistical(d)
        if df_htf is not None:
            d = self._multi_timeframe(d, df_htf)
        if macro_df is not None:
            d = self._macro_overlay(d, macro_df)
        d = self._target(d)
        before = len(d)
        d.dropna(inplace=True)
        logger.info("Features: %d → %d rows, %d cols", before, len(d), len(d.columns))
        return d

    def build_supervised(self, df_feat: pd.DataFrame, horizon: int = 5):
        exclude   = {"target","open","high","low","close","volume"}
        feat_cols = [c for c in df_feat.columns if c not in exclude]
        X         = df_feat[feat_cols].values.astype(np.float32)
        log_ret   = df_feat["log_return"].values
        y = np.array([log_ret[i+1:i+1+horizon].sum()
                      for i in range(len(log_ret)-horizon)], dtype=np.float32)
        return X[:len(y)], y, feat_cols

    def build_sequences(self, df_feat: pd.DataFrame,
                        seq_len: int = 60, horizon: int = 5):
        exclude   = {"target","open","high","low","close","volume"}
        feat_cols = [c for c in df_feat.columns if c not in exclude]
        data      = df_feat[feat_cols].values.astype(np.float32)
        log_ret   = df_feat["log_return"].values
        X_list, y_list = [], []
        for i in range(seq_len, len(data) - horizon):
            X_list.append(data[i-seq_len:i])
            y_list.append(log_ret[i+1:i+1+horizon].sum())
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

    # ── Trend ──────────────────────────────────────────────────────────────

    def _trend(self, d: pd.DataFrame) -> pd.DataFrame:
        for w in [20, 50, 200]:
            d[f"ma{w}"]         = d["close"].rolling(w).mean()
            d[f"close_vs_ma{w}"] = d["close"] / (d[f"ma{w}"] + 1e-9) - 1
        d["ema12"]       = d["close"].ewm(span=12, adjust=False).mean()
        d["ema26"]       = d["close"].ewm(span=26, adjust=False).mean()
        d["macd"]        = d["ema12"] - d["ema26"]
        d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
        d["macd_hist"]   = d["macd"] - d["macd_signal"]
        d["golden_cross"] = (d["ma20"] > d["ma50"]).astype(int)
        d["above_ma200"]  = (d["close"] > d["ma200"]).astype(int)
        # Trend strength (ADX-like)
        dm_up   = d["high"].diff().clip(lower=0)
        dm_down = (-d["low"].diff()).clip(lower=0)
        d["adx_proxy"] = (dm_up - dm_down).abs().rolling(14).mean()
        return d

    # ── Momentum ───────────────────────────────────────────────────────────

    def _momentum(self, d: pd.DataFrame) -> pd.DataFrame:
        # RSI
        delta = d["close"].diff()
        gain  = delta.clip(lower=0); loss = (-delta).clip(lower=0)
        d["rsi"] = 100 - 100 / (1 + gain.rolling(14).mean() /
                                     (loss.rolling(14).mean().replace(0, np.nan)))
        # Stochastic
        low14 = d["low"].rolling(14).min(); high14 = d["high"].rolling(14).max()
        d["stoch_k"] = 100 * (d["close"] - low14) / (high14 - low14 + 1e-9)
        d["stoch_d"] = d["stoch_k"].rolling(3).mean()
        # Williams %R
        d["williams_r"] = -100 * (high14 - d["close"]) / (high14 - low14 + 1e-9)
        # Returns
        for p in [1, 5, 10, 20]:
            d[f"returns_{p}d"] = d["close"].pct_change(p) * 100
        d["roc10"]      = d["close"].pct_change(10) * 100
        d["momentum10"] = d["close"] - d["close"].shift(10)
        return d

    # ── Volatility ─────────────────────────────────────────────────────────

    def _volatility(self, d: pd.DataFrame) -> pd.DataFrame:
        sma20 = d["close"].rolling(20).mean()
        std20 = d["close"].rolling(20).std()
        d["bb_upper"] = sma20 + 2 * std20; d["bb_lower"] = sma20 - 2 * std20
        d["bb_mid"]   = sma20
        d["bb_width"] = (d["bb_upper"] - d["bb_lower"]) / (sma20 + 1e-9)
        d["bb_pct"]   = (d["close"] - d["bb_lower"]) / (d["bb_upper"] - d["bb_lower"] + 1e-9)
        # ATR
        hl = d["high"] - d["low"]
        hc = (d["high"] - d["close"].shift()).abs()
        lc = (d["low"]  - d["close"].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        d["atr14"]   = tr.rolling(14).mean()
        d["atr_pct"] = d["atr14"] / (d["close"] + 1e-9)
        # Historical volatility (21d, annualised)
        log_ret       = np.log(d["close"] / d["close"].shift())
        d["hist_vol21"] = log_ret.rolling(21).std() * np.sqrt(252)
        # Volatility regime (1=low, 2=normal, 3=high)
        vol_pct = d["hist_vol21"].rolling(252).rank(pct=True)
        d["vol_regime"] = pd.cut(vol_pct, bins=[-np.inf, 0.33, 0.67, np.inf],
                                  labels=[1, 2, 3]).astype(float)
        return d

    # ── Volume ─────────────────────────────────────────────────────────────

    def _volume(self, d: pd.DataFrame) -> pd.DataFrame:
        direction    = np.sign(d["close"].diff()).fillna(0)
        d["obv"]     = (direction * d["volume"]).cumsum()
        d["vol_ma20"] = d["volume"].rolling(20).mean()
        d["vol_ratio"] = d["volume"] / (d["vol_ma20"] + 1e-9)
        d["vol_surge"] = (d["vol_ratio"] > 2.0).astype(int)
        # VWAP (20d rolling)
        typical      = (d["high"] + d["low"] + d["close"]) / 3
        d["vwap20"]  = ((typical * d["volume"]).rolling(20).sum() /
                        (d["volume"].rolling(20).sum() + 1e-9))
        d["close_vs_vwap"] = d["close"] / (d["vwap20"] + 1e-9) - 1
        # Liquidity score (inverse of spread proxy)
        d["liquidity_score"] = d["volume"] / (d["atr14"].fillna(1) + 1e-9)
        # Volume imbalance (buy pressure proxy)
        d["vol_imbalance"] = (d["close"] - d["open"]) / (d["atr14"].fillna(1) + 1e-9) * d["vol_ratio"]
        return d

    # ── Price-derived ──────────────────────────────────────────────────────

    def _price_features(self, d: pd.DataFrame) -> pd.DataFrame:
        d["log_return"]   = np.log(d["close"] / d["close"].shift())
        d["high_low_pct"] = (d["high"] - d["low"])         / (d["close"] + 1e-9)
        d["body_size"]    = (d["close"] - d["open"]).abs() / (d["close"] + 1e-9)
        d["upper_shadow"] = (d["high"] - d[["open","close"]].max(axis=1)) / (d["close"] + 1e-9)
        d["lower_shadow"] = (d[["open","close"]].min(axis=1) - d["low"])  / (d["close"] + 1e-9)
        d["is_bullish"]   = (d["close"] > d["open"]).astype(int)
        d["gap_pct"]      = (d["open"] - d["close"].shift()) / (d["close"].shift() + 1e-9)
        for lag in range(1, 6):
            d[f"return_lag{lag}"] = d["log_return"].shift(lag)
        d["return_mean5"]  = d["log_return"].rolling(5).mean()
        d["return_std5"]   = d["log_return"].rolling(5).std()
        d["return_mean20"] = d["log_return"].rolling(20).mean()
        d["return_std20"]  = d["log_return"].rolling(20).std()
        # 52-week position
        d["52w_high"]  = d["close"].rolling(252).max()
        d["52w_low"]   = d["close"].rolling(252).min()
        d["52w_pos"]   = ((d["close"] - d["52w_low"]) /
                          (d["52w_high"] - d["52w_low"] + 1e-9))
        return d

    # ── Statistical ────────────────────────────────────────────────────────

    def _statistical(self, d: pd.DataFrame) -> pd.DataFrame:
        lr = d["log_return"]
        d["skew20"]   = lr.rolling(20).skew()
        d["kurt20"]   = lr.rolling(20).kurt()
        d["zscore20"] = ((d["close"] - d["close"].rolling(20).mean()) /
                         (d["close"].rolling(20).std() + 1e-9))
        d["zscore50"] = ((d["close"] - d["close"].rolling(50).mean()) /
                         (d["close"].rolling(50).std() + 1e-9))
        d["autocorr5"] = lr.rolling(20).apply(
            lambda x: pd.Series(x).autocorr(lag=5) if len(x) >= 6 else np.nan, raw=False)
        return d

    # ── Multi-timeframe ────────────────────────────────────────────────────

    def _multi_timeframe(self, d: pd.DataFrame,
                         df_htf: pd.DataFrame) -> pd.DataFrame:
        """Merge higher-timeframe close/RSI as extra features."""
        htf = df_htf.copy()
        htf.columns = [c.lower() for c in htf.columns]
        if "close" in htf.columns:
            # Weekly RSI
            delta   = htf["close"].diff()
            gain    = delta.clip(lower=0); loss = (-delta).clip(lower=0)
            htf_rsi = 100 - 100 / (1 + gain.rolling(14).mean() /
                                       (loss.rolling(14).mean().replace(0, np.nan)))
            htf_rsi.name = "htf_rsi"
            htf_ma50 = htf["close"].rolling(50).mean()
            htf_ma50.name = "htf_ma50"
            htf_trend = (htf["close"] > htf_ma50).astype(int); htf_trend.name = "htf_trend"
            for s in [htf_rsi, htf_ma50, htf_trend]:
                s_reindexed = s.reindex(d.index, method="ffill")
                d[s.name]   = s_reindexed.values
        return d

    # ── Macro overlay ──────────────────────────────────────────────────────

    def _macro_overlay(self, d: pd.DataFrame,
                       macro_df: pd.DataFrame) -> pd.DataFrame:
        """Add normalised macro features aligned to OHLCV index."""
        macro_aligned = macro_df.reindex(d.index, method="ffill")
        for col in macro_aligned.columns:
            # Normalise macro to pct-change + rolling z-score
            pct = macro_aligned[col].pct_change()
            mu  = pct.rolling(60).mean(); sd = pct.rolling(60).std()
            d[f"macro_{col}_z"] = (pct - mu) / (sd + 1e-9)
        return d

    # ── Target ─────────────────────────────────────────────────────────────

    def _target(self, d: pd.DataFrame) -> pd.DataFrame:
        d["target"] = np.log(d["close"] / d["close"].shift()).shift(-1)
        return d