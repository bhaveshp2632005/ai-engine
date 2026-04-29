"""
main.py — AI Engine v5.0 PRODUCTION
═════════════════════════════════════
ARCHITECTURE CHANGE v5.0:
  OLD: Python fetches data itself (Stooq/Yahoo) → double fetch
  NEW: Node.js sends pre-fetched OHLCV + indicators → Python only predicts

  /predict now accepts:
    {
      "symbol":       "TSLA",
      "horizon":      5,
      "currentPrice": 248.50,
      "prevClose":    245.20,
      "indicators":   { "rsi": 58, "macd": 1.2, "ema20": 246, ... },
      "chart":        [ {date, open, high, low, close, volume}, ... ]
    }

  data_loader.py is NOT called in the predict flow.
  All other endpoints (/regime, /sentiment, /risk, etc.) unchanged.

BUGS ALREADY FIXED (from v4.x):
  ✓ run_in_executor removed (sklearn not thread-safe)
  ✓ /portfolio/optimize alias added
  ✓ Pydantic camelCase alias_generator
  ✓ LRU cache with max 50 entries
  ✓ matplotlib plt.close("all") after charts
  ✓ gc.collect() after predictions
"""

import dataclasses
import gc
import logging
import numpy as np
import os
import sys
import time
from collections import OrderedDict
from contextlib  import asynccontextmanager
from pathlib     import Path
from typing      import Any, Dict, List, Optional

_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from dotenv import load_dotenv
load_dotenv()

from utils.logger import setup_logging
setup_logging(os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("ai-engine")

from fastapi                   import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors   import CORSMiddleware
from fastapi.responses         import JSONResponse
from pydantic                  import BaseModel, Field, ConfigDict
from pydantic.alias_generators import to_camel

# data_loader still imported for non-predict endpoints (regime, risk, etc.)
from data_loader         import DataLoader, MacroLoader, StockDataLoader
from ensemble_model      import LightEnsemble
from feature_engineering import FeatureEngineer
from sentiment_analysis  import SentimentAnalyzer
from backtesting_engine  import BacktestEngine
from visualization       import Visualizer
from regime_detection    import RegimeDetector
from portfolio_optimizer import PortfolioOptimizer
from risk_manager        import RiskManager
from websocket_server    import ws_endpoint


# ══════════════════════════════════════════════════════════════════════════════
# REQUEST MODELS
# ══════════════════════════════════════════════════════════════════════════════

class _CamelModel(BaseModel):
    """Base: accepts camelCase from frontend AND snake_case directly."""
    model_config = ConfigDict(
        alias_generator  = to_camel,
        populate_by_name = True,
        extra            = "ignore",
    )


class PredictRequest(_CamelModel):
    """
    NEW v5.0: Node.js sends pre-fetched data.
    chart[] must have at least 60 rows (recommend 300+).
    indicators{} are optional — used for quick signal enrichment only.
    """
    symbol:        str            = Field(...,  example="TSLA")
    horizon:       int            = Field(5,    ge=1, le=30)
    current_price: Optional[float] = None      # from Node.js live price
    prev_close:    Optional[float] = None
    indicators:    Optional[Dict[str, Any]] = None   # rsi, macd, ema, etc.
    chart:         Optional[List[Dict[str, Any]]] = None  # OHLCV rows
    currency:      str            = "USD"
    skip_sentiment: bool          = False
    include_risk:   bool          = True
    include_chart:  bool          = False
    # kept for API compat — not used (no LSTM in prod)
    lstm_epochs:   int            = Field(60, ge=1, le=200)


class PortfolioRequest(_CamelModel):
    model_config = ConfigDict(extra="ignore")
    symbols: List[str] = Field(..., min_length=2)
    method:  str       = "max_sharpe"
    regime:  str       = "Sideways"


class BacktestRequest(_CamelModel):
    symbol:           str   = Field(..., example="AAPL")
    initial_cash:     float = 100_000.0
    signal_threshold: float = 0.8
    stop_loss_pct:    float = 0.06
    take_profit_pct:  float = 0.12


# ══════════════════════════════════════════════════════════════════════════════
# SERVICE SINGLETONS
# ══════════════════════════════════════════════════════════════════════════════

class _Svc:
    loader    = DataLoader()       # used by non-predict endpoints only
    fe        = FeatureEngineer()
    sentiment = SentimentAnalyzer()
    bt        = BacktestEngine()
    regime    = RegimeDetector()
    risk      = RiskManager()
    viz       = Visualizer(output_dir=str(_HERE / "charts"))

svc = _Svc()


# ══════════════════════════════════════════════════════════════════════════════
# LRU CACHE (max 50 entries, TTL from env)
# ══════════════════════════════════════════════════════════════════════════════

class _LRUCache:
    def __init__(self, max_size: int = 50, ttl: int = 900):
        self._store:    OrderedDict = OrderedDict()
        self._max_size: int         = max_size
        self._ttl:      int         = ttl

    def get(self, key: str):
        if key not in self._store:
            return None
        entry = self._store[key]
        if (time.time() - entry["t"]) > self._ttl:
            del self._store[key]
            return None
        self._store.move_to_end(key)
        return entry["d"]

    def set(self, key: str, data):
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = {"d": data, "t": time.time()}
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)

    def __len__(self):
        return len(self._store)


_cache     = _LRUCache(max_size=50, ttl=int(os.getenv("AI_CACHE_TTL", "900")))


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _regime_dict(r) -> dict:
    d = r.to_dict()
    if "currentRegime" not in d:
        d["currentRegime"] = d.get("regime", "Sideways")
    return d

def _articles_json(articles: list) -> list:
    out = []
    for a in articles:
        if dataclasses.is_dataclass(a):
            out.append(dataclasses.asdict(a))
        elif isinstance(a, dict):
            out.append(a)
    return out


def _chart_to_df(chart: List[Dict[str, Any]]):
    """
    Convert Node.js chart[] array → clean pandas DataFrame.
    chart rows: { date, open, high, low, close, volume }
    Returns DataFrame with DatetimeIndex, columns = open/high/low/close/volume.
    """
    import pandas as pd

    df = pd.DataFrame(chart)
    df.columns = [c.lower() for c in df.columns]

    # Flexible date column names
    date_col = next((c for c in df.columns if c in ("date", "timestamp", "time", "datetime")), None)
    if date_col is None:
        raise ValueError("chart[] rows must have a 'date' field")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.set_index(date_col).sort_index()
    df.index.name = "date"

    # Keep only OHLCV
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            if col == "volume":
                df["volume"] = 0   # volume is optional — fill with 0
            else:
                raise ValueError(f"chart[] rows missing required column: '{col}'")

    df = df[["open", "high", "low", "close", "volume"]].copy()

    # Coerce to float, drop bad rows
    df = df.apply(pd.to_numeric, errors="coerce")
    df.dropna(how="all", inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df = df[df["close"] > 0]
    df = df[~df.index.duplicated(keep="last")]

    if len(df) < 60:
        raise ValueError(
            f"chart[] has only {len(df)} valid rows after cleaning. "
            "Send at least 60 rows (recommend 300+ for best accuracy)."
        )

    return df


# ══════════════════════════════════════════════════════════════════════════════
# APP LIFECYCLE
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AI Engine v5.0-prod starting …")
    try:
        import pandas as pd
        dummy = pd.DataFrame({
            "open": [100.0]*300, "high": [101.0]*300,
            "low":  [99.0]*300,  "close": [100.5]*300,
            "volume": [1_000_000]*300,
        })
        svc.fe.transform(dummy)
        gc.collect()
        logger.info("FeatureEngineer warm-up OK")
    except Exception as e:
        logger.warning("Warm-up failed (non-fatal): %s", e)
    yield
    logger.info("AI Engine shutting down")


# ══════════════════════════════════════════════════════════════════════════════
# CORS
# ══════════════════════════════════════════════════════════════════════════════

_cors    = os.getenv("CORS_ORIGINS", "*")
_origins = [o.strip() for o in _cors.split(",") if o.strip()] or ["*"]

app = FastAPI(
    title     = "StockAnalyzer AI Engine",
    version   = "5.0.0-prod",
    lifespan  = lifespan,
    docs_url  = "/docs",
    redoc_url = None,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins     = _origins,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
@app.get("/ping")
def health():
    return {
        "status":        "ok",
        "version":       "5.0.0-prod",
        "mode":          "data-bridge",   # Node.js sends data, Python predicts
        "cache_entries": len(_cache),
    }


# ── Quick Analyze (rule-based, no ML, < 1s) ───────────────────────────────────
# Node.js sends indicators it already calculated — Python uses them directly.

@app.post("/analyze")
async def analyze_quick(body: dict):
    try:
        sym        = (body.get("symbol") or "UNKNOWN").upper()
        ind        = body.get("indicators") or {}
        candles    = body.get("chart") or []
        import pandas as pd

        # If indicators provided by Node.js, use them directly (fastest path)
        if ind and ind.get("rsi") is not None:
            rsi      = float(ind.get("rsi",      50))
            macd     = float(ind.get("macd",      0))
            macd_sig = float(ind.get("macdSignal") or ind.get("macd_signal") or 0)
            bb_pct   = float(ind.get("bbPct")     or ind.get("bb_pct")      or 0.5)
            trend    = float(ind.get("trend")     or ind.get("close_vs_ma50") or 0)
        elif candles and len(candles) >= 20:
            # Fallback: compute from candles
            df       = _chart_to_df(candles)
            df_f     = svc.fe.transform(df)
            last     = df_f.iloc[-1]
            rsi      = float(last.get("rsi",          50))
            macd     = float(last.get("macd",          0))
            macd_sig = float(last.get("macd_signal",   0))
            bb_pct   = float(last.get("bb_pct",        0.5))
            trend    = float(last.get("close_vs_ma50", 0))
        else:
            return JSONResponse({"action": "HOLD", "confidence": 30,
                                 "summary": "Insufficient data", "score": 0})

        score, reasons = 0.0, []
        if rsi < 35:        score += 2; reasons.append(f"RSI oversold ({rsi:.0f})")
        elif rsi > 65:      score -= 2; reasons.append(f"RSI overbought ({rsi:.0f})")
        if macd > macd_sig: score += 1; reasons.append("MACD bullish")
        else:               score -= 1; reasons.append("MACD bearish")
        if bb_pct < 0.2:    score += 1; reasons.append("Near lower BB")
        elif bb_pct > 0.8:  score -= 1; reasons.append("Near upper BB")
        if trend > 0.02:    score += 1; reasons.append("Above MA50")
        elif trend < -0.02: score -= 1; reasons.append("Below MA50")

        action     = "BUY" if score >= 2 else ("SELL" if score <= -2 else "HOLD")
        confidence = min(95, max(30, 50 + abs(score) * 10))
        return JSONResponse({"action": action, "confidence": int(confidence),
                             "summary": " | ".join(reasons) or "Mixed signals",
                             "score": round(score, 2)})
    except Exception as ex:
        logger.warning("[analyze] %s", ex)
        return JSONResponse({"action": "HOLD", "confidence": 30,
                             "summary": "Analysis unavailable", "score": 0})


# ── Full Prediction ───────────────────────────────────────────────────────────
# v5.0: Uses chart[] from Node.js — NO data_loader.py call.

@app.post("/predict")
async def predict(req: PredictRequest):
    sym = req.symbol.upper().strip()
    ck  = f"predict:{sym}:{req.horizon}"

    cached = _cache.get(ck)
    if cached:
        return JSONResponse({**cached, "fromCache": True})

    try:
        out = _run_predict(req, sym)
        _cache.set(ck, out)
        return JSONResponse({**out, "fromCache": False})

    except ValueError as ex:
        raise HTTPException(422, str(ex))
    except MemoryError:
        gc.collect()
        raise HTTPException(503, "Server low on memory — retry in 30s")
    except Exception as ex:
        logger.exception("[predict] %s", sym)
        raise HTTPException(500, f"Prediction failed: {ex}")


def _run_predict(req: PredictRequest, sym: str) -> dict:
    """
    v5.0 core: builds DataFrame from req.chart[], then predicts.
    NO external API calls. All data comes from Node.js.
    """
    t0 = time.time()

    # ── 1. Build DataFrame from Node.js chart data ────────────────────────────
    if not req.chart:
        raise ValueError(
            "chart[] is required. Node.js must send OHLCV candle data. "
            "Send at least 300 candles for best accuracy."
        )

    df  = _chart_to_df(req.chart)
    cur = req.currency or "USD"

    # Override current price if Node.js provides live price
    # (chart[-1].close may be slightly delayed)
    current_price_override = req.current_price

    # ── 2. Regime detection (uses provided DataFrame) ─────────────────────────
    regime_res  = svc.regime.detect(df)
    regime_dict = _regime_dict(regime_res)

    # ── 3. ML Prediction ──────────────────────────────────────────────────────
    ensemble = LightEnsemble(horizon=req.horizon)
    result   = ensemble.predict(
        symbol         = sym,
        df_raw         = df,
        currency       = cur,
        skip_sentiment = req.skip_sentiment,
    )
    out = result.to_dict()

    # Use Node.js live price if provided (more accurate than chart[-1])
    if current_price_override and current_price_override > 0:
        cp = float(current_price_override)
        # Recalculate price targets with accurate current price
        pred_pct = result.predicted_return / 100
        out["currentPrice"]    = round(cp, 2)
        out["predictedPrice"]  = round(cp * (1 + pred_pct), 2)
        out["priceRange"]["low"]  = round(cp * (1 + (result.price_low  / result.current_price - 1)), 2)
        out["priceRange"]["high"] = round(cp * (1 + (result.price_high / result.current_price - 1)), 2)

    out["confidence"]   = round(max(float(out.get("confidence", 50.0)), 52.0), 1)
    out["marketRegime"] = regime_dict

    # ── 4. Enrich with Node.js indicators (if provided) ───────────────────────
    # These don't change the ML prediction but are included in response
    # so frontend has everything in one place.
    if req.indicators:
        out["nodeIndicators"] = req.indicators

    out["meta"] = {
        "symbol":        sym,
        "currency":      cur,
        "dataPoints":    len(df),
        "latestDate":    str(df.index[-1].date()),
        "computeTime":   round(time.time() - t0, 2),
        "engineVersion": "5.0.0-prod",
        "dataSource":    "node-bridge",   # confirms no double-fetch
    }

    # ── 5. Risk assessment (uses provided DataFrame) ──────────────────────────
    if req.include_risk:
        try:
            risk_res = svc.risk.assess(
                symbol           = sym,
                df               = df,
                predicted_return = result.predicted_return / 100,
                market_regime    = regime_res.regime,
            )
            out["risk"] = risk_res.to_dict()
        except Exception as e:
            logger.warning("[predict] risk failed: %s", e)

    # ── 6. Chart (optional, uses provided DataFrame) ──────────────────────────
    if req.include_chart:
        try:
            import matplotlib.pyplot as plt
            df_f         = svc.fe.transform(df)
            fig          = svc.viz.prediction_chart(df_f, result)
            out["chart"] = svc.viz.fig_to_base64(fig)
            plt.close("all")
        except Exception as e:
            logger.warning("[predict] chart failed: %s", e)

    gc.collect()
    return out


# ── Regime (still uses data_loader for standalone calls) ──────────────────────

@app.get("/regime/{symbol}")
async def regime(symbol: str):
    sym = symbol.upper().strip()
    ck  = f"regime:{sym}"
    cached = _cache.get(ck)
    if cached:
        return JSONResponse({**cached, "fromCache": True})
    try:
        df  = svc.loader.get(sym)
        res = svc.regime.detect(df)
        out = {**_regime_dict(res), "symbol": sym}
        _cache.set(ck, out)
        return JSONResponse({**out, "fromCache": False})
    except Exception as ex:
        raise HTTPException(500, str(ex))


# ── Sentiment ─────────────────────────────────────────────────────────────────

@app.get("/sentiment/{symbol}")
async def sentiment(symbol: str, max_articles: int = 20):
    sym = symbol.upper().strip()
    ck  = f"sent:{sym}"
    cached = _cache.get(ck)
    if cached:
        return JSONResponse({**cached, "fromCache": True})
    try:
        res = svc.sentiment.analyze(sym, max_articles)
        out = {**res.to_dict(), "articles": _articles_json(res.articles[:10])}
        _cache.set(ck, out)
        return JSONResponse({**out, "fromCache": False})
    except Exception as ex:
        raise HTTPException(500, str(ex))


# ── Portfolio ─────────────────────────────────────────────────────────────────

@app.post("/portfolio")
@app.post("/portfolio/optimize")
async def portfolio(req: PortfolioRequest):
    try:
        syms   = [s.upper() for s in req.symbols]
        prices = {s: svc.loader.get(s) for s in syms}
        opt    = PortfolioOptimizer(symbols=syms)
        result = opt.optimize(prices, method=req.method)
        frontier = []
        try:
            frontier = opt.efficient_frontier(prices, n_points=30)
        except Exception:
            pass
        return JSONResponse({**result.to_dict(), "frontier": frontier})
    except ValueError as ex:
        raise HTTPException(422, str(ex))
    except Exception as ex:
        raise HTTPException(500, str(ex))


# ── Risk ──────────────────────────────────────────────────────────────────────

@app.get("/risk/{symbol}")
async def risk(symbol: str):
    sym = symbol.upper().strip()
    try:
        df  = svc.loader.get(sym)
        res = svc.risk.assess(symbol=sym, df=df)
        return JSONResponse({**res.to_dict(), "symbol": sym})
    except Exception as ex:
        raise HTTPException(500, str(ex))


# ── Backtest ──────────────────────────────────────────────────────────────────

@app.post("/backtest")
async def backtest(req: BacktestRequest):
    sym = req.symbol.upper().strip()
    ck  = f"bt:{sym}:{req.signal_threshold}"
    cached = _cache.get(ck)
    if cached:
        return JSONResponse({**cached, "fromCache": True})
    try:
        t0   = time.time()
        df   = svc.loader.get(sym)
        df_f = svc.fe.transform(df)
        from ml_models import XGBoostModel, LightGBMModel
        X_tab, y_tab, _ = svc.fe.build_supervised(df_f, 5)
        xgb_m = XGBoostModel(n_estimators=50); xgb_m.fit(X_tab, y_tab)
        lgb_m = LightGBMModel(n_estimators=50); lgb_m.fit(X_tab, y_tab)
        sigs  = svc.bt.generate_signals_from_model(df_f, [xgb_m, lgb_m], [], svc.fe)
        bt_r  = svc.bt.run(df, sigs,
            signal_threshold = req.signal_threshold,
            stop_loss_pct    = req.stop_loss_pct,
            take_profit_pct  = req.take_profit_pct,
        )
        del xgb_m, lgb_m
        gc.collect()
        out = {**bt_r.to_dict(), "symbol": sym, "computeTime": round(time.time()-t0, 2)}
        _cache.set(ck, out)
        return JSONResponse({**out, "fromCache": False})
    except Exception as ex:
        raise HTTPException(500, str(ex))


# ── Indicators ────────────────────────────────────────────────────────────────

@app.get("/indicators/{symbol}")
async def indicators(symbol: str, n_days: int = 30):
    sym = symbol.upper().strip()
    ck  = f"ind:{sym}:{n_days}"
    cached = _cache.get(ck)
    if cached:
        return JSONResponse({**cached, "fromCache": True})
    try:
        import matplotlib.pyplot as plt
        df   = svc.loader.get(sym)
        df_f = svc.fe.transform(df)
        chart = None
        try:
            chart = svc.viz.fig_to_base64(svc.viz.indicator_chart(df_f, sym))
            plt.close("all")
        except Exception:
            pass
        out = {
            "symbol": sym, "n_days": n_days,
            "latest": {k: round(float(v), 4) for k, v in df_f.iloc[-1].items()
                      if not hasattr(v, "__len__")},
            "chart": chart,
        }
        _cache.set(ck, out)
        return JSONResponse({**out, "fromCache": False})
    except Exception as ex:
        raise HTTPException(500, str(ex))


# ── Chart ─────────────────────────────────────────────────────────────────────

@app.get("/chart/{symbol}")
async def chart(symbol: str, chart_type: str = "price"):
    sym = symbol.upper().strip()
    try:
        import matplotlib.pyplot as plt
        df   = svc.loader.get(sym)
        df_f = svc.fe.transform(df)
        fig  = svc.viz.indicator_chart(df_f, sym) if chart_type == "indicator" \
               else svc.viz.price_chart(df_f, sym)
        result = JSONResponse({"symbol": sym, "chart": svc.viz.fig_to_base64(fig)})
        plt.close("all")
        return result
    except Exception as ex:
        raise HTTPException(500, str(ex))


# ── Signal ────────────────────────────────────────────────────────────────────

@app.get("/signal/{symbol}")
async def signal_engine(symbol: str):
    sym = symbol.upper().strip()
    ck  = f"signal:{sym}"
    cached = _cache.get(ck)
    if cached:
        return JSONResponse({**cached, "fromCache": True})
    try:
        from backtesting_engine import SignalGenerator
        df  = svc.loader.get(sym)
        sig = SignalGenerator.latest(df)
        try:
            s  = svc.sentiment.analyze(sym, max_articles=10)
            sd = s.to_dict()
            if sd.get("score", 0) > 0.15 and sig["score"] > 0:
                sig["confidence"] = min(sig["confidence"] + 5, 95)
                sig["reasons"].append(f"Positive news ({sd['label']})")
            elif sd.get("score", 0) < -0.15 and sig["score"] < 0:
                sig["confidence"] = min(sig["confidence"] + 5, 95)
                sig["reasons"].append(f"Negative news ({sd['label']})")
            sig["sentiment"] = sd
        except Exception:
            sig["sentiment"] = {"label": "Neutral", "score": 0, "confidence": 0}
        out = {**sig, "symbol": sym,
               "currentPrice": round(float(df["close"].iloc[-1]), 2)}
        _cache.set(ck, out)
        return JSONResponse({**out, "fromCache": False})
    except Exception as ex:
        raise HTTPException(500, str(ex))


# ── Macro ─────────────────────────────────────────────────────────────────────

@app.get("/macro")
async def macro_snapshot():
    ck = "macro:latest"
    cached = _cache.get(ck)
    if cached:
        return JSONResponse({**cached, "fromCache": True})
    try:
        snap = MacroLoader().snapshot()
        vix  = snap.get("vix") or 20
        snap["vix_regime"] = "Low" if vix < 15 else ("High" if vix > 30 else "Normal")
        snap["risk_on"]    = snap.get("risk_on_score", 0.5)
        _cache.set(ck, snap)
        return JSONResponse({**snap, "fromCache": False})
    except Exception as ex:
        raise HTTPException(500, str(ex))


# ── Timeframes ────────────────────────────────────────────────────────────────

@app.get("/timeframes/{symbol}")
async def timeframes(symbol: str):
    sym    = symbol.upper().strip()
    result = {}
    for tf, (period, interval) in {"1d": ("2y","1d"), "1wk": ("5y","1wk")}.items():
        try:
            df = StockDataLoader(sym, period=period, interval=interval).load()
            result[tf] = {
                "rows":         len(df),
                "latest_close": round(float(df["close"].iloc[-1]), 2),
                "latest_date":  str(df.index[-1].date()),
            }
        except Exception as e:
            result[tf] = {"error": str(e)}
    return JSONResponse({"symbol": sym, "timeframes": result})


# ── Portfolio Eval ────────────────────────────────────────────────────────────

@app.post("/portfolio/eval")
async def portfolio_eval(body: dict):
    import pandas as pd
    holdings   = body.get("holdings", [])
    capital    = float(body.get("capital", 100000))
    results, prices_map = [], {}

    for h in holdings:
        sym    = str(h.get("symbol", "")).upper().strip()
        shares = float(h.get("shares", 0))
        avg_c  = float(h.get("avgCost", 0))
        if not sym or shares <= 0: continue
        try:
            df  = svc.loader.get(sym)
            cur = float(df["close"].iloc[-1])
            prices_map[sym] = df["close"]
            mv  = shares * cur
            cv  = shares * avg_c if avg_c > 0 else mv
            pnl = mv - cv
            log_r = df["close"].pct_change().dropna()
            results.append({
                "symbol":       sym,
                "shares":       shares,
                "avgCost":      round(avg_c, 2),
                "currentPrice": round(cur,   2),
                "marketValue":  round(mv,    2),
                "costBasis":    round(cv,    2),
                "pnl":          round(pnl,   2),
                "pnlPct":       round((pnl/cv*100 if cv > 0 else 0), 2),
                "volatility":   round(float(log_r.std())*252**0.5*100, 2),
            })
        except Exception as e:
            results.append({"symbol": sym, "error": str(e)})

    total_val  = sum(r.get("marketValue", 0) for r in results)
    total_cost = sum(r.get("costBasis",   0) for r in results)
    total_pnl  = sum(r.get("pnl",         0) for r in results)
    for r in results:
        r["allocation"] = round(r.get("marketValue", 0)/total_val*100, 2) if total_val > 0 else 0

    port_vol = None
    try:
        if len(prices_map) >= 2:
            ret_df  = pd.DataFrame({s: p.pct_change() for s, p in prices_map.items()}).dropna()
            syms_in = [r["symbol"] for r in results if r.get("symbol") in ret_df.columns]
            weights = [r["allocation"]/100 for r in results if r.get("symbol") in ret_df.columns]
            if syms_in:
                w        = np.array(weights)
                cov      = ret_df[syms_in].cov().values * 252
                port_vol = round(float((w @ cov @ w)**0.5 * 100), 2)
    except Exception:
        pass

    return JSONResponse({
        "holdings":            results,
        "totalValue":          round(total_val,  2),
        "totalCost":           round(total_cost, 2),
        "totalPnl":            round(total_pnl,  2),
        "totalPnlPct":         round(total_pnl/total_cost*100 if total_cost > 0 else 0, 2),
        "portfolioVolatility": port_vol,
        "capital":             capital,
    })


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws_endpoint(ws)


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host    = "0.0.0.0",
        port    = int(os.getenv("PORT", os.getenv("AI_PORT", "10000"))),
        reload  = False,
        workers = 1,
    )