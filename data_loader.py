"""
data_loader.py — Stock data pipeline PRODUCTION
════════════════════════════════════════════════
CHANGES vs dev version:
  ✓ FIXED:  DataLoader.get() now accepts interval= kwarg (was missing)
  ✓ FIXED:  All HTTP timeouts reduced to 10s (was 20s)
  ✓ FIXED:  MacroLoader snapshot() catches all errors gracefully
  ✓ KEPT:   Stooq primary → AlphaVantage fallback → yfinance fallback
"""

import io, logging, os, time
from datetime import datetime, timedelta
from pathlib  import Path

import pandas  as pd
import requests

logger = logging.getLogger(__name__)

DATA_DIR        = Path(os.getenv("DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL_HOURS = int(os.getenv("CACHE_TTL_HOURS", "4"))

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
})

_PERIOD_DAYS = {
    "5d": 5, "1mo": 30, "3mo": 90, "6mo": 180,
    "1y": 365, "2y": 730, "5y": 1825, "10y": 3650,
}

def _dates(period: str):
    days  = _PERIOD_DAYS.get(period, 730)
    end   = datetime.today()
    start = end - timedelta(days=days)
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")

def _stooq_sym(symbol: str) -> str:
    s = symbol.upper().strip()
    _map = {
        "^GSPC": "^SPX", "^DJI": "^DJI", "^IXIC": "^NDQ",
        "^VIX": "^VIX", "^TNX": "^TNX",
        "CL=F": "CL.F", "GC=F": "GC.F",
        "DX-Y.NYB": "DXY.F",
        "^NSEI": "^NII50", "^BSESN": "^BSE",
        "NIFTY50": "^NII50", "SENSEX": "^BSE",
    }
    if s in _map:      return _map[s]
    if s.startswith("^"): return s
    if "=F" in s:      return s.replace("=F", ".F")
    if "-USD" in s:    return s.lower().replace("-usd", "usd.cx")
    if s.endswith(".NS"): return s
    if s.endswith(".BO"): return s
    if s.endswith(".L"):  return s[:-2] + ".UK"
    if s.endswith(".TO"): return s[:-3] + ".CA"
    if s.endswith(".AX"): return s[:-3] + ".AU"
    return f"{s}.US"


class StockDataLoader:
    def __init__(self, symbol: str, period: str = "2y", interval: str = "1d"):
        self.symbol     = symbol.upper().strip()
        self.period     = period
        self.interval   = interval
        self.cache_path = DATA_DIR / f"{self.symbol}_{interval}.csv"

    def load(self, force_refresh: bool = False) -> pd.DataFrame:
        if not force_refresh and self._cache_fresh():
            try:
                df = self._read_cache()
                if not df.empty:
                    logger.info("[%s] cache hit (%d rows)", self.symbol, len(df))
                    return df
            except Exception:
                pass
        logger.info("[%s] downloading …", self.symbol)
        df = self._download()
        self._save(df)
        return df

    def get_info(self) -> dict:
        try:
            import yfinance as yf
            info = yf.Ticker(self.symbol, session=_SESSION).info or {}
            return {
                "symbol":   self.symbol,
                "name":     info.get("longName",    self.symbol),
                "sector":   info.get("sector",      "N/A"),
                "currency": info.get("currency",    "USD"),
                "exchange": info.get("exchange",    "N/A"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "52w_high": info.get("fiftyTwoWeekHigh"),
                "52w_low":  info.get("fiftyTwoWeekLow"),
                "beta":     info.get("beta"),
            }
        except Exception as e:
            logger.warning("[%s] get_info failed (ok): %s", self.symbol, e)
            return {"symbol": self.symbol, "name": self.symbol, "currency": "USD"}

    def get_headlines(self, max_items: int = 15) -> list:
        try:
            import yfinance as yf
            news = yf.Ticker(self.symbol, session=_SESSION).news or []
            return [n.get("title", "") for n in news[:max_items] if n.get("title")]
        except Exception:
            return []

    def _download(self) -> pd.DataFrame:
        errors = []
        try:
            df = self._from_stooq()
            if not df.empty:
                logger.info("[%s] Stooq OK", self.symbol)
                return df
        except Exception as e:
            errors.append(f"Stooq:{e}")

        key = os.getenv("ALPHA_VANTAGE_KEY", "").strip()
        if key:
            try:
                df = self._from_alphavantage(key)
                if not df.empty:
                    logger.info("[%s] AlphaVantage OK", self.symbol)
                    return df
            except Exception as e:
                errors.append(f"AV:{e}")

        try:
            df = self._from_yfinance()
            if not df.empty:
                logger.info("[%s] yfinance OK", self.symbol)
                return df
        except Exception as e:
            errors.append(f"yf:{e}")

        if self.cache_path.exists():
            try:
                df = self._read_cache()
                if not df.empty:
                    logger.warning("[%s] using stale cache", self.symbol)
                    return df
            except Exception:
                pass

        raise RuntimeError(
            f"[{self.symbol}] all sources failed: {'; '.join(errors)}")

    def _from_stooq(self) -> pd.DataFrame:
        sym    = _stooq_sym(self.symbol)
        d1, d2 = _dates(self.period)
        url    = f"https://stooq.com/q/d/l/?s={sym}&d1={d1}&d2={d2}&i=d"
        r      = _SESSION.get(url, timeout=10)
        r.raise_for_status()
        text = r.text.strip()
        if len(text) < 50 or "No data" in text:
            raise ValueError(f"empty response for {sym}")
        df = pd.read_csv(io.StringIO(text))
        df.columns = [c.strip().lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"])
        return self._clean(df.set_index("date").sort_index())

    @staticmethod
    def _av_sym(symbol: str) -> str:
        s = symbol.upper()
        if s.endswith(".NS"):  return s[:-3] + ".BSE"
        if s.endswith(".BO"):  return s[:-3] + ".BSE"
        if s.endswith(".L"):   return s[:-2] + ".LON"
        if s.endswith(".TO"):  return s[:-3] + ".TRT"
        if s.endswith(".AX"):  return s[:-3] + ".ASX"
        return s

    def _from_alphavantage(self, key: str) -> pd.DataFrame:
        av_sym = self._av_sym(self.symbol)
        url    = (f"https://www.alphavantage.co/query"
                  f"?function=TIME_SERIES_DAILY_ADJUSTED&symbol={av_sym}"
                  f"&outputsize=full&apikey={key}&datatype=csv")
        r    = _SESSION.get(url, timeout=10)
        r.raise_for_status()
        text = r.text.strip()
        if text.startswith("{") or "Invalid" in text or "Thank you" in text:
            import json
            try:
                msg  = json.loads(text)
                note = msg.get("Note") or msg.get("Information") or msg.get("Error Message") or text[:120]
            except Exception:
                note = text[:120]
            raise ValueError(f"AV error: {note}")
        df = pd.read_csv(io.StringIO(text))
        df.columns = [c.lower().strip() for c in df.columns]
        date_col = next((c for c in df.columns if "time" in c or "date" in c), None)
        if date_col is None:
            raise ValueError(f"No date column. Columns: {list(df.columns)}")
        df = df.rename(columns={date_col: "date"})
        if "adjusted_close" in df.columns:
            df = df.rename(columns={"adjusted_close": "close"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date").sort_index()
        d1_str = (datetime.today() - timedelta(days=_PERIOD_DAYS.get(self.period, 730))).strftime("%Y-%m-%d")
        df = df[df.index >= d1_str]
        cols = [c for c in ["open","high","low","close","volume"] if c in df.columns]
        return self._clean(df[cols])

    def _from_yfinance(self) -> pd.DataFrame:
        days  = _PERIOD_DAYS.get(self.period, 730)
        end   = int(datetime.today().timestamp())
        start = int((datetime.today() - timedelta(days=days)).timestamp())
        iv_map = {"1d": "1d", "1wk": "1wk", "1mo": "1mo"}
        iv     = iv_map.get(self.interval, "1d")
        url    = (f"https://query1.finance.yahoo.com/v8/finance/chart/{self.symbol}"
                  f"?period1={start}&period2={end}&interval={iv}&events=history")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://finance.yahoo.com/",
        }
        resp = _SESSION.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data   = resp.json()
        result = data.get("chart", {}).get("result")
        if not result:
            err = data.get("chart", {}).get("error", {})
            raise ValueError(f"Yahoo v8 error: {err}")
        r          = result[0]
        timestamps = r.get("timestamp", [])
        q          = r.get("indicators", {}).get("quote", [{}])[0]
        adj        = r.get("indicators", {}).get("adjclose", [{}])
        adjclose   = adj[0].get("adjclose", []) if adj else []
        if not timestamps:
            raise ValueError("empty response")
        close_col = adjclose if len(adjclose) == len(timestamps) else q.get("close", [])
        df = pd.DataFrame({
            "date":   pd.to_datetime(timestamps, unit="s"),
            "open":   q.get("open",   [None]*len(timestamps)),
            "high":   q.get("high",   [None]*len(timestamps)),
            "low":    q.get("low",    [None]*len(timestamps)),
            "close":  close_col,
            "volume": q.get("volume", [None]*len(timestamps)),
        })
        df["date"] = df["date"].dt.tz_localize(None)
        return self._clean(df.set_index("date").sort_index())

    def _read_cache(self) -> pd.DataFrame:
        df = pd.read_csv(self.cache_path, index_col=0, parse_dates=True)
        return self._clean(df)

    def _save(self, df: pd.DataFrame):
        try:
            df.to_csv(self.cache_path)
        except Exception as e:
            logger.warning("[%s] save failed: %s", self.symbol, e)

    def _cache_fresh(self) -> bool:
        if not self.cache_path.exists():
            return False
        age = (datetime.now().timestamp() - self.cache_path.stat().st_mtime) / 3600
        return age < CACHE_TTL_HOURS

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        cols = [c for c in ["open","high","low","close","volume"] if c in df.columns]
        if "close" not in cols:
            return pd.DataFrame()
        df = df[cols]
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index = pd.to_datetime(df.index)
        df.dropna(how="all", inplace=True)
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        df = df[df["close"] > 0]
        df = df[~df.index.duplicated(keep="last")]
        df.sort_index(inplace=True)
        return df


class DataLoader:
    """Service singleton — supports interval kwarg."""

    def __init__(self):
        self._cache: dict = {}

    def _loader(self, symbol: str, interval: str = "1d") -> StockDataLoader:
        key = f"{symbol}_{interval}"
        if key not in self._cache:
            self._cache[key] = StockDataLoader(symbol, interval=interval)
        return self._cache[key]

    def get(self, symbol: str, interval: str = "1d",
            force_refresh: bool = False) -> pd.DataFrame:
        return self._loader(symbol, interval).load(force_refresh)

    def get_info(self, symbol: str) -> dict:
        return self._loader(symbol).get_info()

    def get_headlines(self, symbol: str, n: int = 15) -> list:
        return self._loader(symbol).get_headlines(n)

    def get_macro(self):
        """Alias used by old code."""
        import pandas as pd
        try:
            return MacroLoader().get()
        except Exception:
            return pd.DataFrame()


def load_stock(symbol: str, period: str = "2y") -> pd.DataFrame:
    return StockDataLoader(symbol, period=period).load()


class MacroLoader:
    _SYMBOLS = {
        "vix": "^VIX", "sp500": "^GSPC", "dxy": "DX-Y.NYB",
        "oil": "CL=F", "gold": "GC=F", "tnx": "^TNX",
    }
    _CACHE = DATA_DIR / "_macro.csv"
    _TTL   = 6 * 3600

    def get(self, period: str = "1y", force_refresh: bool = False) -> "pd.DataFrame":
        if not force_refresh and self._fresh():
            try:
                import pandas as pd
                df = pd.read_csv(self._CACHE, index_col=0, parse_dates=True)
                if not df.empty:
                    return df
            except Exception:
                pass
        df = self._download(period)
        if not df.empty:
            try:
                self._CACHE.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(self._CACHE)
            except Exception:
                pass
        return df

    def snapshot(self) -> dict:
        snap: dict = {}
        try:
            df = self.get()
            for col in self._SYMBOLS:
                try:
                    snap[col] = round(float(df[col].dropna().iloc[-1]), 2)
                except Exception:
                    snap[col] = None
        except Exception:
            snap = {k: None for k in self._SYMBOLS}

        vix = snap.get("vix") or 20.0
        snap["risk_on_score"] = round(max(0.0, min(1.0, 1.0 - (vix - 15) / 25)), 2)
        try:
            import pandas as pd
            df   = self.get()
            sp   = df["sp500"].dropna()
            snap["sp500_trend"] = "up" if sp.iloc[-1] > sp.rolling(20).mean().iloc[-1] else "down"
        except Exception:
            snap["sp500_trend"] = "unknown"
        return snap

    def _download(self, period: str) -> "pd.DataFrame":
        import pandas as pd
        frames = {}
        for name, sym in self._SYMBOLS.items():
            try:
                df = StockDataLoader(sym, period=period)._from_stooq()
                if not df.empty:
                    frames[name] = df["close"].rename(name)
            except Exception as e:
                logger.warning("[MacroLoader] %s failed: %s", name, e)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames.values(), axis=1).sort_index()

    def _fresh(self) -> bool:
        return self._CACHE.exists() and (time.time() - self._CACHE.stat().st_mtime) < self._TTL
