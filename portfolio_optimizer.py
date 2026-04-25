"""
portfolio_optimizer.py — Modern Portfolio Theory + ML-enhanced optimization.

FIXES vs original:
  • Constructor accepts symbols OR infers from price_data keys
  • optimize() signature unified: optimize(price_data, method, ml_returns)
  • efficient_frontier() robust (returns [] on failure, never throws)
  • All scipy minimize calls wrapped in try/except → equal-weight fallback
"""
import logging
from dataclasses import dataclass, field
from typing      import Dict, List, Optional

import numpy  as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass
class PortfolioResult:
    weights:         Dict[str, float]
    expected_return: float
    expected_risk:   float
    sharpe_ratio:    float
    method:          str

    def to_dict(self) -> dict:
        return {
            "weights":        {k: round(v * 100, 2) for k, v in self.weights.items()},
            "expectedReturn": round(self.expected_return * 100, 2),
            "expectedRisk":   round(self.expected_risk   * 100, 2),
            "sharpeRatio":    round(self.sharpe_ratio, 3),
            "method":         self.method,
        }


class PortfolioOptimizer:
    """
    Institutional portfolio optimiser.

    Supports
    --------
    • max_sharpe    — maximise Sharpe ratio (default)
    • min_variance  — minimum variance frontier
    • equal_weight  — 1/N baseline
    • risk_parity   — equal risk contribution

    Usage
    -----
    # Option A: pass symbols at construction
    opt = PortfolioOptimizer(symbols=["AAPL","TSLA","NVDA","MSFT"])
    result = opt.optimize(price_data_dict, method="max_sharpe")

    # Option B: infer symbols from price_data at optimize() time
    opt = PortfolioOptimizer()
    result = opt.optimize(price_data_dict, method="max_sharpe")
    """

    def __init__(
        self,
        symbols:     Optional[List[str]] = None,
        risk_free:   float = 0.05,
        min_weight:  float = 0.02,
        max_weight:  float = 0.40,
        cash_buffer: float = 0.05,
    ):
        self.symbols     = [s.upper() for s in symbols] if symbols else None
        self.risk_free   = risk_free
        self.min_weight  = min_weight
        self.max_weight  = max_weight
        self.cash_buffer = cash_buffer

    # ── Public ──────────────────────────────────────────────────────────────

    def optimize(
        self,
        price_data:  Dict[str, pd.DataFrame],
        method:      str = "max_sharpe",
        ml_returns:  Optional[Dict[str, float]] = None,
        # legacy compat — api_server.py passes these as keyword args
        prices_dict: Optional[Dict[str, pd.DataFrame]] = None,
        regime:      str = "Sideways",
    ) -> PortfolioResult:
        # Accept either price_data or prices_dict (legacy compat)
        data = price_data if price_data else (prices_dict or {})

        # Infer symbols from data keys if not set at construction
        syms = self.symbols if self.symbols else [s.upper() for s in data.keys()]
        if not syms:
            return self._equal_weight_result([], method)

        returns_df = self._build_returns(data, syms)
        if returns_df.empty or len(returns_df) < 60:
            logger.warning("Insufficient return data — using equal weights")
            return self._equal_weight_result(syms, method)

        mu_hist = returns_df.mean().values * 252
        cov     = returns_df.cov().values  * 252

        # Blend ML predicted returns when available
        if ml_returns:
            mu_ml = np.array([ml_returns.get(s, 0.0) / 100 * 252 for s in syms])
            mu    = 0.5 * mu_hist + 0.5 * mu_ml
        else:
            mu = mu_hist

        n = len(syms)
        if method == "equal_weight":
            w = self._equal_weight(n)
        elif method == "min_variance":
            w = self._min_variance(mu, cov, n)
        elif method == "risk_parity":
            w = self._risk_parity(cov, n)
        else:  # max_sharpe (default)
            w = self._max_sharpe(mu, cov, n)

        return self._build_result(w, mu, cov, syms, method)

    # ── Optimisation routines ────────────────────────────────────────────────

    def _max_sharpe(self, mu, cov, n) -> np.ndarray:
        def neg_sharpe(w):
            ret  = w @ mu
            risk = np.sqrt(w @ cov @ w + 1e-9)
            return -(ret - self.risk_free) / risk
        return self._run_opt(neg_sharpe, n)

    def _min_variance(self, mu, cov, n) -> np.ndarray:
        def portfolio_var(w):
            return w @ cov @ w
        return self._run_opt(portfolio_var, n)

    def _risk_parity(self, cov, n) -> np.ndarray:
        def risk_parity_obj(w):
            sigma = np.sqrt(w @ cov @ w + 1e-9)
            mrc   = cov @ w / sigma
            rc    = w * mrc
            target = sigma / n
            return float(np.sum((rc - target) ** 2))
        return self._run_opt(risk_parity_obj, n)

    @staticmethod
    def _equal_weight(n: int) -> np.ndarray:
        return np.ones(n) / n

    def _run_opt(self, objective, n: int) -> np.ndarray:
        bounds      = [(self.min_weight, self.max_weight)] * n
        constraints = [{"type": "eq",
                        "fun": lambda w: w.sum() - (1 - self.cash_buffer)}]
        x0 = np.ones(n) / n
        try:
            result = minimize(objective, x0, method="SLSQP",
                              bounds=bounds, constraints=constraints,
                              options={"maxiter": 500, "ftol": 1e-9})
            if result.success:
                w = np.clip(result.x, self.min_weight, self.max_weight)
                return w / w.sum() * (1 - self.cash_buffer)
        except Exception as e:
            logger.warning("Optimisation failed: %s — using equal weights", e)
        return self._equal_weight(n) * (1 - self.cash_buffer)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _build_returns(self, price_data: Dict[str, pd.DataFrame],
                       syms: List[str]) -> pd.DataFrame:
        series = {}
        for sym in syms:
            if sym not in price_data:
                continue
            df = price_data[sym].copy()
            df.columns = [c.lower() for c in df.columns]
            if "close" in df.columns:
                series[sym] = np.log(df["close"] / df["close"].shift()).dropna()
        if not series:
            return pd.DataFrame()
        return pd.DataFrame(series).dropna()

    def _build_result(self, w, mu, cov, syms, method) -> PortfolioResult:
        weights_dict = {s: float(wi) for s, wi in zip(syms, w)}
        # Add cash allocation
        allocated    = sum(weights_dict.values())
        weights_dict["CASH"] = round(max(0.0, 1.0 - allocated), 4)
        exp_ret  = float(w @ mu)
        exp_risk = float(np.sqrt(w @ cov @ w + 1e-9))
        sharpe   = (exp_ret - self.risk_free) / (exp_risk + 1e-9)
        return PortfolioResult(
            weights=weights_dict,
            expected_return=exp_ret,
            expected_risk=exp_risk,
            sharpe_ratio=sharpe,
            method=method,
        )

    def _equal_weight_result(self, syms: List[str], method: str) -> PortfolioResult:
        n = max(len(syms), 1)
        w = {s: round((1 - self.cash_buffer) / n, 4) for s in syms}
        w["CASH"] = self.cash_buffer
        return PortfolioResult(weights=w, expected_return=0.0,
                               expected_risk=0.0, sharpe_ratio=0.0, method=method)

    # ── Efficient frontier ───────────────────────────────────────────────────

    def efficient_frontier(self, price_data: Dict[str, pd.DataFrame],
                           n_points: int = 50) -> list:
        """Return list of (risk, return, sharpe) dicts for plotting."""
        try:
            syms       = self.symbols if self.symbols else list(price_data.keys())
            returns_df = self._build_returns(price_data, syms)
            if returns_df.empty or len(returns_df) < 60:
                return []
            mu  = returns_df.mean().values * 252
            cov = returns_df.cov().values  * 252
            n   = len(syms)
            frontier = []
            for target_ret in np.linspace(mu.min(), mu.max(), n_points):
                try:
                    constraints = [
                        {"type": "eq", "fun": lambda w: w.sum() - 1},
                        {"type": "eq", "fun": lambda w, t=target_ret: w @ mu - t},
                    ]
                    res = minimize(
                        lambda w: np.sqrt(w @ cov @ w + 1e-9),
                        np.ones(n) / n, method="SLSQP",
                        bounds=[(0, 1)] * n, constraints=constraints,
                    )
                    if res.success:
                        w      = res.x
                        risk   = float(np.sqrt(w @ cov @ w + 1e-9))
                        ret    = float(w @ mu)
                        sharpe = (ret - self.risk_free) / (risk + 1e-9)
                        frontier.append({
                            "risk":   round(risk * 100, 2),
                            "return": round(ret  * 100, 2),
                            "sharpe": round(sharpe,     3),
                        })
                except Exception:
                    continue
            return frontier
        except Exception as e:
            logger.warning("efficient_frontier failed: %s", e)
            return []