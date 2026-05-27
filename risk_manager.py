
import logging
from dataclasses import dataclass
from typing      import Dict, Optional

import numpy  as np
import pandas as pd

logger = logging.getLogger(__name__)

RISK_LEVELS = [(0.25, "Low"), (0.50, "Medium"), (0.75, "High"), (1.0, "Critical")]


@dataclass
class RiskAssessment:
    symbol:             str
    risk_level:         str
    risk_score:         float
    suggested_position: float
    stop_loss_price:    float
    stop_loss_pct:      float
    take_profit_price:  float
    take_profit_pct:    float
    var_95:             float
    cvar_95:            float
    max_drawdown:       float
    volatility:         float
    beta:               Optional[float] = None
    notes:              str = ""

    def to_dict(self) -> dict:
        return {
            "symbol":            self.symbol,
            "riskLevel":         self.risk_level,
            "riskScore":         round(self.risk_score        * 100, 1),
            "suggestedPosition": round(self.suggested_position * 100, 2),
            "stopLossPrice":     round(self.stop_loss_price,    2),
            "stopLossPct":       round(self.stop_loss_pct      * 100, 2),
            "takeProfitPrice":   round(self.take_profit_price,  2),
            "takeProfitPct":     round(self.take_profit_pct    * 100, 2),
            "var95":             round(self.var_95              * 100, 2),
            "cvar95":            round(self.cvar_95             * 100, 2),
            "maxDrawdown":       round(self.max_drawdown        * 100, 2),
            "volatility":        round(self.volatility          * 100, 2),
            "notes":             self.notes,
        }


class RiskManager:
    """
    Kelly Criterion + ATR stop-loss + historical VaR/CVaR.

    Usage
    -----
    rm     = RiskManager()
    result = rm.assess(symbol="TSLA", df=df, predicted_return=0.03)
    """

    def __init__(
        self,
        portfolio_value:     float = 100_000.0,
        max_risk_per_trade:  float = 0.02,
        max_position_size:   float = 0.15,
        kelly_fraction:      float = 0.25,
        atr_stop_multiplier: float = 2.5,
        reward_risk_ratio:   float = 2.0,
    ):
        self.portfolio_value      = portfolio_value
        self.max_risk_per_trade   = max_risk_per_trade
        self.max_position_size    = max_position_size
        self.kelly_fraction       = kelly_fraction
        self.atr_stop_multiplier  = atr_stop_multiplier
        self.reward_risk_ratio    = reward_risk_ratio

    # ── Public ──────────────────────────────────────────────────────────────

    def assess(
        self,
        df:               pd.DataFrame,
        symbol:           str   = "UNKNOWN",
        predicted_return: float = 0.0,
        win_rate:         float = 0.52,
        confidence:       float = 0.7,    # accepts 0-1 OR 0-100
        # Accept both naming conventions:
        market_regime:    str   = "",
        regime:           str   = "",
    ) -> RiskAssessment:

        # Normalise confidence to 0-1
        conf = confidence / 100.0 if confidence > 1 else confidence
        # Merge regime kwargs
        reg = (market_regime or regime or "Neutral").strip()

        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        cp = float(df["close"].iloc[-1])

        # Volatility + VaR
        log_ret   = np.log(df["close"] / df["close"].shift()).dropna()
        vol_daily = float(log_ret.std())
        vol_ann   = vol_daily * np.sqrt(252)

        try:
            var_95  = float(np.percentile(log_ret, 5))
            mask    = log_ret <= var_95
            cvar_95 = float(log_ret[mask].mean()) if mask.any() else var_95
        except Exception:
            var_95 = cvar_95 = -vol_daily * 1.65

        # ATR-based stop-loss
        atr         = max(self._calc_atr(df, 14), cp * 0.005)   # floor at 0.5%
        sl_distance = self.atr_stop_multiplier * atr
        sl_price    = cp - sl_distance
        sl_pct      = sl_distance / (cp + 1e-9)
        tp_distance = sl_distance * self.reward_risk_ratio
        tp_price    = cp + tp_distance
        tp_pct      = tp_distance / (cp + 1e-9)

        # Max Drawdown
        equity = (1 + log_ret).cumprod()
        peak   = equity.cummax()
        mdd    = float(((equity - peak) / (peak + 1e-9)).min())

        risk_score = self._compute_risk_score(
            vol_ann, mdd, abs(var_95), sl_pct, conf, reg)

        position = self._size_position(
            vol_ann, risk_score, win_rate, predicted_return, sl_pct)

        risk_level = next(
            (label for thresh, label in RISK_LEVELS if risk_score <= thresh),
            "Critical")

        notes = []
        if vol_ann > 0.5: notes.append("High volatility — reduce position size")
        if mdd < -0.4:    notes.append("Deep historical drawdown detected")
        if reg == "Bear": notes.append("Bear market — defensive sizing applied")

        return RiskAssessment(
            symbol=symbol, risk_level=risk_level, risk_score=risk_score,
            suggested_position=position,
            stop_loss_price=sl_price, stop_loss_pct=sl_pct,
            take_profit_price=tp_price, take_profit_pct=tp_pct,
            var_95=abs(var_95), cvar_95=abs(cvar_95),
            max_drawdown=mdd, volatility=vol_ann,
            notes=" | ".join(notes) if notes else "Normal risk conditions",
        )

    # ── Position sizing ──────────────────────────────────────────────────────

    def _size_position(self, vol, risk_score, win_rate, pred_ret, sl_pct) -> float:
        base  = self.max_risk_per_trade / (vol + 1e-9) * (1 - risk_score * 0.5)
        avg_win  = max(pred_ret, 0.01)
        avg_loss = max(sl_pct, 0.005)
        kelly = ((win_rate * avg_win - (1 - win_rate) * avg_loss) /
                 (avg_win + 1e-9)) * self.kelly_fraction
        size  = min(base, kelly)
        return float(np.clip(size, 0.005, self.max_position_size))

    # ── Risk score ───────────────────────────────────────────────────────────

    def _compute_risk_score(self, vol, mdd, var, sl_pct, confidence, regime) -> float:
        vol_score    = min(vol / 0.8, 1.0)
        mdd_score    = min(abs(mdd) / 0.6, 1.0)
        var_score    = min(var / 0.05, 1.0)
        conf_score   = 1.0 - confidence
        regime_bonus = 0.2 if regime == "Bear" else (0.1 if regime == "Sideways" else 0.0)
        raw = (0.35 * vol_score + 0.25 * mdd_score +
               0.20 * var_score + 0.20 * conf_score + regime_bonus)
        return float(np.clip(raw, 0.0, 1.0))

    # ── ATR ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _calc_atr(df: pd.DataFrame, period: int = 14) -> float:
        try:
            hl = df["high"] - df["low"]
            hc = (df["high"] - df["close"].shift()).abs()
            lc = (df["low"]  - df["close"].shift()).abs()
            tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
            atr = float(tr.rolling(period).mean().iloc[-1])
            return atr if np.isfinite(atr) and atr > 0 else float(df["close"].iloc[-1]) * 0.02
        except Exception:
            return float(df["close"].iloc[-1]) * 0.02

    # ── Batch assessment ─────────────────────────────────────────────────────

    def assess_portfolio(
        self,
        symbols_data:      Dict[str, pd.DataFrame],
        predicted_returns: Optional[Dict[str, float]] = None,
    ) -> Dict[str, RiskAssessment]:
        result = {}
        for sym, df in symbols_data.items():
            pr = (predicted_returns or {}).get(sym, 0.0) / 100
            try:
                result[sym] = self.assess(df=df, symbol=sym, predicted_return=pr)
            except Exception as exc:
                logger.warning("Risk assess failed for %s: %s", sym, exc)
        return result