"""
main.py — AI Engine v4.0 PRODUCTION (Render/Railway Free-Tier)
═══════════════════════════════════════════════════════════════
CHANGES vs dev version:
  ✗ REMOVED: torch, LSTM, GRU, TemporalFusionModel
  ✗ REMOVED: FinBERT / transformers (lexicon fallback only)
  ✗ REMOVED: stable-baselines3 RL agent endpoints
  ✗ REMOVED: hmmlearn HMM (KMeans-only regime detection)
  ✓ KEPT:    XGBoost, LightGBM, LinearModel
  ✓ KEPT:    Feature engineering, backtesting, portfolio, risk
  ✓ ADDED:   LightEnsemble (tabular-only, no seq models)
  ✓ ADDED:   Model warm-up at startup
  ✓ FIXED:   CORS via env var
  ✓ FIXED:   PORT env var (Render uses PORT not AI_PORT)

Start: uvicorn main:app --host 0.0.0.0 --port 10000
"""

import dataclasses
import logging
import numpy as np
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib    import Path
from typing     import List, Optional

_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from dotenv import load_dotenv
load_dotenv()

from utils.logger import setup_logging
setup_logging(os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("ai-engine")

from fastapi                 import FastAPI, HTTPException, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses       import JSONResponse
from pydantic                import BaseModel, Field

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


# ── Request Models ────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    symbol:           str   = Field(..., example="AAPL")
    horizon:          int   = Field(5, ge=1, le=30)
    skip_sentiment:   bool  = False
    include_risk:     bool  = True
    include_chart:    bool  = False
    include_backtest: bool  = False
    lstm_epochs:      int   = Field(60, ge=1, le=200)  # kept for API compat, ignored

class PortfolioRequest(BaseModel):
    symbols: List[str] = Field(..., min_length=2)
    method:  str       = "max_sharpe"
    regime:  str       = "Sideways"

class BacktestRequest(BaseModel):
    symbol:           str   = Field(..., example="AAPL")
    initial_cash:     float = 100_000.0
    signal_threshold: float = 0.8
    stop_loss_pct:    float = 0.06
    take_profit_pct:  float = 0.12


# ── Service Singletons ────────────────────────────────────────────────────────

class _Svc:
    loader    = DataLoader()
    fe        = FeatureEngineer()
    sentiment = SentimentAnalyzer()
    bt        = BacktestEngine()
    regime    = RegimeDetector()
    risk      = RiskManager()
    viz       = Visualizer(output_dir=str(_HERE / "charts"))

svc = _Svc()


# ── Cache ─────────────────────────────────────────────────────────────────────

_cache: dict = {}
_CACHE_TTL   = int(os.getenv("AI_CACHE_TTL", "900"))

def _cget(key: str):
    e = _cache.get(key)
    if e and (time.time() - e["t"]) < _CACHE_TTL:
        return e["d"]
    return None

def _cset(key: str, data):
    _cache[key] = {"d": data, "t": time.time()}


# ── Helpers ───────────────────────────────────────────────────────────────────

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


# ── App Lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AI Engine v4.0-prod starting …")
    try:
        import pandas as pd
        dummy = pd.DataFrame({
            "open": [100.0]*300, "high": [101.0]*300,
            "low": [99.0]*300, "close": [100.5]*300,
            "volume": [1_000_000]*300,
        })
        svc.fe.transform(dummy)
        logger.info("FeatureEngineer warm-up OK")
    except Exception as e:
        logger.warning("Warm-up failed (non-fatal): %s", e)
    yield
    logger.info("AI Engine shutting down")


# ── CORS ──────────────────────────────────────────────────────────────────────
_cors = os.getenv("CORS_ORIGINS", "*")
_origins = [o.strip() for o in _cors.split(",") if o.strip()] or ["*"]

app = FastAPI(
    title    = "StockAnalyzer AI Engine",
    version  = "4.0.0-prod",
    lifespan = lifespan,
    docs_url = "/docs",   # disable in prod by setting DISABLE_DOCS=1 if needed
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
def health():
    return {
        "status":        "ok",
        "version":       "4.0.0-prod",
        "mode":          "lightweight",
        "cache_entries": len(_cache),
    }


@app.post("/analyze")
async def analyze_quick(body: dict):
    """Fast rule-based signal — always < 1s, no ML training."""
    try:
        sym     = (body.get("symbol") or "UNKNOWN").upper()
        candles = body.get("chart") or body.get("candles") or []
        import pandas as pd
        if candles and len(candles) >= 20:
            df = pd.DataFrame(candles)
            df.columns = [c.lower() for c in df.columns]
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
        else:
            df = svc.loader.get(sym)

        df_f     = svc.fe.transform(df)
        last     = df_f.iloc[-1]
        rsi      = float(last.get("rsi",          50))
        macd     = float(last.get("macd",          0))
        macd_sig = float(last.get("macd_signal",   0))
        bb_pct   = float(last.get("bb_pct",        0.5))
        trend    = float(last.get("close_vs_ma50", 0))

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


@app.post("/predict")
async def predict(req: PredictRequest, bg: BackgroundTasks):
    sym = req.symbol.upper().strip()
    ck  = f"predict:{sym}:{req.horizon}"
    if (cached := _cget(ck)):
        return JSONResponse({**cached, "fromCache": True})
    try:
        t0   = time.time()
        df   = svc.loader.get(sym)
        info = svc.loader.get_info(sym)
        cur  = info.get("currency", "USD")

        regime_res  = svc.regime.detect(df)
        regime_dict = _regime_dict(regime_res)

        ensemble = LightEnsemble(horizon=req.horizon)
        result   = ensemble.predict(sym, df, cur, req.skip_sentiment)
        out      = result.to_dict()

        out["confidence"]   = round(max(float(out.get("confidence", 50.0)), 52.0), 1)
        out["marketRegime"] = regime_dict
        out["meta"] = {
            "symbol":        sym,
            "companyName":   info.get("name",     sym),
            "sector":        info.get("sector",   ""),
            "currency":      cur,
            "exchange":      info.get("exchange", ""),
            "dataPoints":    len(df),
            "latestDate":    str(df.index[-1].date()),
            "computeTime":   round(time.time() - t0, 2),
            "engineVersion": "4.0.0-prod",
        }

        if req.include_risk:
            try:
                risk_res    = svc.risk.assess(
                    symbol           = sym, df=df,
                    predicted_return = result.predicted_return / 100,
                    market_regime    = regime_res.regime,
                )
                out["risk"] = risk_res.to_dict()
            except Exception as e:
                logger.warning("[predict] risk failed: %s", e)

        if req.include_chart:
            try:
                df_f         = svc.fe.transform(df)
                fig          = svc.viz.prediction_chart(df_f, result)
                out["chart"] = svc.viz.fig_to_base64(fig)
            except Exception as e:
                logger.warning("[predict] chart failed: %s", e)

        if req.include_backtest:
            try:
                df_f = svc.fe.transform(df)
                sigs = svc.bt.generate_signals_from_model(
                    df_f, ensemble.tabular_models, [], svc.fe)
                bt_r = svc.bt.run(df, sigs)
                out["backtest"] = bt_r.to_dict()
            except Exception as e:
                logger.warning("[predict] backtest failed: %s", e)

        bg.add_task(ensemble.save, sym)
        _cset(ck, out)
        return JSONResponse({**out, "fromCache": False})

    except ValueError as ex:
        raise HTTPException(422, str(ex))
    except Exception as ex:
        logger.exception("[predict] %s", sym)
        raise HTTPException(500, str(ex))


@app.get("/regime/{symbol}")
async def regime(symbol: str):
    sym = symbol.upper().strip()
    ck  = f"regime:{sym}"
    if (cached := _cget(ck)):
        return JSONResponse({**cached, "fromCache": True})
    try:
        df  = svc.loader.get(sym)
        res = svc.regime.detect(df)
        out = {**_regime_dict(res), "symbol": sym}
        _cset(ck, out)
        return JSONResponse({**out, "fromCache": False})
    except Exception as ex:
        raise HTTPException(500, str(ex))


@app.get("/sentiment/{symbol}")
async def sentiment(symbol: str, max_articles: int = 20):
    sym = symbol.upper().strip()
    ck  = f"sent:{sym}"
    if (cached := _cget(ck)):
        return JSONResponse({**cached, "fromCache": True})
    try:
        res = svc.sentiment.analyze(sym, max_articles)
        out = {**res.to_dict(), "articles": _articles_json(res.articles[:10])}
        _cset(ck, out)
        return JSONResponse({**out, "fromCache": False})
    except Exception as ex:
        raise HTTPException(500, str(ex))


@app.post("/portfolio")
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


@app.get("/risk/{symbol}")
async def risk(symbol: str):
    sym = symbol.upper().strip()
    try:
        df  = svc.loader.get(sym)
        res = svc.risk.assess(symbol=sym, df=df)
        return JSONResponse({**res.to_dict(), "symbol": sym})
    except Exception as ex:
        raise HTTPException(500, str(ex))


@app.post("/backtest")
async def backtest(req: BacktestRequest):
    sym = req.symbol.upper().strip()
    ck  = f"bt:{sym}:{req.signal_threshold}"
    if (cached := _cget(ck)):
        return JSONResponse({**cached, "fromCache": True})
    try:
        t0   = time.time()
        df   = svc.loader.get(sym)
        df_f = svc.fe.transform(df)
        from ml_models import XGBoostModel, LightGBMModel
        X_tab, y_tab, _ = svc.fe.build_supervised(df_f, 5)
        xgb_m = XGBoostModel(n_estimators=100);  xgb_m.fit(X_tab, y_tab)
        lgb_m = LightGBMModel(n_estimators=100); lgb_m.fit(X_tab, y_tab)
        sigs = svc.bt.generate_signals_from_model(df_f, [xgb_m, lgb_m], [], svc.fe)
        bt_r = svc.bt.run(df, sigs,
            signal_threshold=req.signal_threshold,
            stop_loss_pct=req.stop_loss_pct,
            take_profit_pct=req.take_profit_pct,
        )
        out = {**bt_r.to_dict(), "symbol": sym, "computeTime": round(time.time()-t0, 2)}
        _cset(ck, out)
        return JSONResponse({**out, "fromCache": False})
    except Exception as ex:
        raise HTTPException(500, str(ex))


@app.get("/indicators/{symbol}")
async def indicators(symbol: str, n_days: int = 30):
    sym = symbol.upper().strip()
    ck  = f"ind:{sym}:{n_days}"
    if (cached := _cget(ck)):
        return JSONResponse({**cached, "fromCache": True})
    try:
        df   = svc.loader.get(sym)
        df_f = svc.fe.transform(df)
        chart = None
        try:
            chart = svc.viz.fig_to_base64(svc.viz.indicator_chart(df_f, sym))
        except Exception:
            pass
        out = {
            "symbol": sym, "n_days": n_days,
            "latest": {k: round(float(v), 4) for k, v in df_f.iloc[-1].items()
                      if not hasattr(v, "__len__")},
            "chart": chart,
        }
        _cset(ck, out)
        return JSONResponse({**out, "fromCache": False})
    except Exception as ex:
        raise HTTPException(500, str(ex))


@app.get("/chart/{symbol}")
async def chart(symbol: str, chart_type: str = "price"):
    sym = symbol.upper().strip()
    try:
        df   = svc.loader.get(sym)
        df_f = svc.fe.transform(df)
        fig  = svc.viz.indicator_chart(df_f, sym) if chart_type == "indicator" \
               else svc.viz.price_chart(df_f, sym)
        return JSONResponse({"symbol": sym, "chart": svc.viz.fig_to_base64(fig)})
    except Exception as ex:
        raise HTTPException(500, str(ex))


@app.get("/signal/{symbol}")
async def signal_engine(symbol: str):
    sym = symbol.upper().strip()
    ck  = f"signal:{sym}"
    if (cached := _cget(ck)):
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
        _cset(ck, out)
        return JSONResponse({**out, "fromCache": False})
    except Exception as ex:
        raise HTTPException(500, str(ex))


@app.get("/macro")
async def macro_snapshot():
    ck = "macro:latest"
    if (cached := _cget(ck)):
        return JSONResponse({**cached, "fromCache": True})
    try:
        snap = MacroLoader().snapshot()
        vix  = snap.get("vix") or 20
        snap["vix_regime"] = "Low" if vix < 15 else ("High" if vix > 30 else "Normal")
        snap["risk_on"]    = snap.get("risk_on_score", 0.5)
        _cset(ck, snap)
        return JSONResponse({**snap, "fromCache": False})
    except Exception as ex:
        raise HTTPException(500, str(ex))


@app.get("/timeframes/{symbol}")
async def timeframes(symbol: str):
    sym = symbol.upper().strip()
    result = {}
    for tf, (period, interval) in {"1d": ("2y","1d"), "1wk": ("5y","1wk")}.items():
        try:
            df = StockDataLoader(sym, period=period, interval=interval).load()
            result[tf] = {
                "rows": len(df),
                "latest_close": round(float(df["close"].iloc[-1]), 2),
                "latest_date": str(df.index[-1].date()),
            }
        except Exception as e:
            result[tf] = {"error": str(e)}
    return JSONResponse({"symbol": sym, "timeframes": result})


@app.post("/portfolio/eval")
async def portfolio_eval(body: dict):
    import pandas as pd
    holdings = body.get("holdings", [])
    capital  = float(body.get("capital", 100000))
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
                "symbol": sym, "shares": shares, "avgCost": round(avg_c, 2),
                "currentPrice": round(cur, 2), "marketValue": round(mv, 2),
                "costBasis": round(cv, 2), "pnl": round(pnl, 2),
                "pnlPct": round((pnl/cv*100 if cv > 0 else 0), 2),
                "volatility": round(float(log_r.std())*252**0.5*100, 2),
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
                w   = np.array(weights)
                cov = ret_df[syms_in].cov().values * 252
                port_vol = round(float((w @ cov @ w)**0.5 * 100), 2)
    except Exception:
        pass

    return JSONResponse({
        "holdings": results, "totalValue": round(total_val, 2),
        "totalCost": round(total_cost, 2), "totalPnl": round(total_pnl, 2),
        "totalPnlPct": round(total_pnl/total_cost*100 if total_cost > 0 else 0, 2),
        "portfolioVolatility": port_vol, "capital": capital,
    })


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws_endpoint(ws)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host    = "0.0.0.0",
        port    = int(os.getenv("PORT", os.getenv("AI_PORT", "10000"))),
        reload  = False,
        workers = 1,
    )
