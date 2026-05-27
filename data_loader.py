

import io, logging, os, time
from datetime  import datetime, timedelta
from pathlib   import Path
from urllib.parse import unquote

import pandas   as pd
import requests

logger = logging.getLogger(__name__)

DATA_DIR        = Path(os.getenv("DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL_HOURS = int(os.getenv("CACHE_TTL_HOURS", "4"))

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept":          "application/json, text/html, */*",
    "Referer":         "https://finance.yahoo.com/",
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


# ══════════════════════════════════════════════════════════════════════════════
# CANONICAL SYMBOL MAP  ← THE CORE FIX
# Every alias variant maps to the Yahoo Finance v8 canonical symbol.
# _canonicalise() is called at the top of StockDataLoader.__init__ so
# bad symbols never reach Stooq, AlphaVantage, or Yahoo with the wrong string.
# ══════════════════════════════════════════════════════════════════════════════

_CANONICAL: dict[str, str] = {
    # ── Indian indices ────────────────────────────────────────────────────────
    "^NSEI":         "^NSEI",
    "NIFTY50":       "^NSEI",
    "NIFTY50.NS":    "^NSEI",
    "NIFTY50.BO":    "^NSEI",
    "NIFTY.NS":      "^NSEI",
    "NIFTY":         "^NSEI",

    "^BSESN":        "^BSESN",
    "SENSEX":        "^BSESN",
    "SENSEX.NS":     "^BSESN",   # ← was crashing
    "SENSEX.BO":     "^BSESN",
    "BSE":           "^BSESN",
    "BSE.NS":        "^BSESN",

    "^NSEBANK":      "^NSEBANK",
    "BANKNIFTY":     "^NSEBANK",
    "BANKNIFTY.NS":  "^NSEBANK",
    "BANKNIFTY.BO":  "^NSEBANK",
    "NSEBANK":       "^NSEBANK",
    "NSEBANK.NS":    "^NSEBANK",

    # ── Global indices ────────────────────────────────────────────────────────
    "^GSPC":    "^GSPC",   "SP500":    "^GSPC",
    "^DJI":     "^DJI",    "DOW":      "^DJI",
    "^IXIC":    "^IXIC",   "NASDAQ":   "^IXIC",
    "^VIX":     "^VIX",
    "^TNX":     "^TNX",
    "^FTSE":    "^FTSE",
    "^GDAXI":   "^GDAXI",  "DAX":      "^GDAXI",
    "^FCHI":    "^FCHI",   "CAC40":    "^FCHI",
    "^N225":    "^N225",   "NIKKEI":   "^N225",
    "^HSI":     "^HSI",    "HANGSENG": "^HSI",
    "^AXJO":    "^AXJO",
    "^KS11":    "^KS11",
}


def _canonicalise(symbol: str) -> str:
    """
    Decode URL-encoding (%5ENSEI → ^NSEI), upper-case, then map any
    known alias to the Yahoo-canonical symbol. Returns unchanged if unknown.
    """
    s = unquote(symbol).upper().strip()
    return _CANONICAL.get(s, s)


def is_index(sym: str) -> bool:
    """True for any market index (^ prefix after canonicalisation)."""
    return _canonicalise(sym).startswith("^")


# ══════════════════════════════════════════════════════════════════════════════
# STOOQ SYMBOL TRANSLATION  (equities only — never called for indices)
# ══════════════════════════════════════════════════════════════════════════════

_YF_TO_STOOQ: list[tuple[str, str]] = sorted([
    (".NS", ".NS"), (".BO", ".BO"),
    (".L",  ".UK"), (".IL", ".UK"),
    (".TO", ".CA"), (".V",  ".CA"), (".CN", ".CA"),
    (".T",  ".JP"),
    (".HK", ".HK"),
    (".AX", ".AU"), (".NZ", ".NZ"), (".SG", ".SG"),
    (".TW", ".TW"), (".TWO", ".TW"),
    (".KS", ".KR"), (".KQ", ".KR"),
    (".SS", ".CN"), (".SZ", ".CN"),
    (".KL", ".MY"), (".BK", ".TH"), (".JK", ".JK"),
    (".DE", ".DE"), (".PA", ".FR"), (".SW", ".SW"),
    (".AS", ".NL"), (".MI", ".IT"), (".MC", ".ES"),
    (".OL", ".NO"), (".ST", ".SE"), (".CO", ".DK"),
    (".HE", ".FI"), (".LS", ".PT"), (".BR", ".BE"),
    (".VI", ".AT"), (".WA", ".PL"), (".IS", ".IS"),
    (".ME", ".RU"), (".SA", ".BR"), (".JO", ".SJ"),
], key=lambda x: -len(x[0]))


def _stooq_sym(symbol: str) -> str:
    s = symbol.upper().strip()
    _exact = {
        "^GSPC": "^SPX", "^DJI": "^DJI", "^IXIC": "^NDQ",
        "^VIX":  "^VIX", "^TNX": "^TNX",
        "^FTSE": "^FTM", "^GDAXI": "^DAX", "^FCHI": "^CAC",
        "^N225": "^NKX", "^HSI": "^HSI",   "^AXJO": "^AXJO",
        "CL=F":  "CL.F", "GC=F": "GC.F",   "SI=F": "SI.F", "NG=F": "NG.F",
    }
    if s in _exact:
        return _exact[s]
    if s.startswith("^"):
        return s
    if s.endswith("=X"):
        return s[:-2].lower() + ".fx"
    if s.endswith("=F") or "=F" in s:
        return s.replace("=F", ".F")
    if "-USD" in s:
        return s.lower().replace("-usd", "usd") + ".cx"
    for yf_sfx, stooq_sfx in _YF_TO_STOOQ:
        if s.endswith(yf_sfx.upper()):
            return s[: len(s) - len(yf_sfx)] + stooq_sfx
    if s.endswith(".F"):
        base = s[:-2]
        if len(base) >= 3:
            return base + ".DE"
        return s
    return s + ".US"


# ══════════════════════════════════════════════════════════════════════════════
# ALPHAVANTAGE SYMBOL TRANSLATION
# ══════════════════════════════════════════════════════════════════════════════

_YF_TO_AV: list[tuple[str, str]] = sorted([
    (".NS", ".BSE"), (".BO", ".BSE"), (".L", ".LON"),  (".TO", ".TRT"),
    (".T",  ".TSE"), (".HK", ".HKEX"),(".AX", ".ASX"), (".SG", ".SGX"),
    (".TW", ".TSEC"),(".TWO",".TSEC"),(".KS", ".KSC"), (".SS", ".SHH"),
    (".SZ", ".SHZ"), (".DE", ".DEX"), (".PA", ".PAR"), (".SW", ".SWX"),
    (".AS", ".AMS"), (".MI", ".MIL"), (".MC", ".BME"), (".OL", ".OSL"),
    (".ST", ".STO"), (".ME", ".MCX"), (".SA", ".SAO"), (".JO", ".JSE"),
], key=lambda x: -len(x[0]))


def _av_sym(symbol: str) -> str:
    s = symbol.upper().strip()
    # All index symbols are unsupported by AlphaVantage
    if is_index(s):
        return "__INDEX_NOT_SUPPORTED__"
    if s.endswith("=X"):
        pair = s[:-2]
        return (pair[:3] + "/" + pair[3:]) if len(pair) == 6 else pair
    if s.endswith("=F") or "=F" in s:
        return s.replace("=F", "").replace("=", "")
    if "-USD" in s or "-BTC" in s or "-ETH" in s:
        return s.split("-")[0]
    if s.endswith(".F"):
        base = s[:-2]
        return (base + ".DEX") if len(base) >= 3 else base
    for yf_sfx, av_sfx in _YF_TO_AV:
        if s.endswith(yf_sfx.upper()):
            return s[: len(s) - len(yf_sfx)] + av_sfx
    return s


# ══════════════════════════════════════════════════════════════════════════════
# STOCK DATA LOADER
# ══════════════════════════════════════════════════════════════════════════════

class StockDataLoader:
    def __init__(self, symbol: str, period: str = "2y", interval: str = "1d"):
        # _canonicalise MUST be first — converts SENSEX.NS → ^BSESN etc.
        self.symbol   = _canonicalise(symbol)
        self.period   = period
        self.interval = interval
        if self.symbol != symbol.upper().strip():
            logger.info("[DataLoader] '%s' → canonical '%s'", symbol, self.symbol)
        safe = (self.symbol
                .replace("/", "_").replace("^", "IDX_")
                .replace("=", "_").replace("-", "_"))
        self.cache_path = DATA_DIR / f"{safe}_{interval}.csv"

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self, force_refresh: bool = False) -> pd.DataFrame:
        if not force_refresh and self._cache_fresh():
            try:
                df = self._read_cache()
                if not df.empty:
                    logger.info("[%s] cache hit (%d rows)", self.symbol, len(df))
                    return df
            except Exception:
                pass
        logger.info("[%s] downloading…", self.symbol)
        df = self._download()
        self._save(df)
        return df

    def get_info(self) -> dict:
        base = {
            "symbol":   self.symbol,
            "name":     self.symbol,
            "currency": "INR" if self._is_indian() else "USD",
        }
        if is_index(self.symbol):
            try:
                url  = (f"https://query1.finance.yahoo.com/v8/finance/chart/"
                        f"{self.symbol}?range=5d&interval=1d")
                r    = _SESSION.get(url, timeout=5)
                meta = r.json()["chart"]["result"][0]["meta"]
                return {
                    **base,
                    "name":     meta.get("shortName",    self.symbol),
                    "currency": meta.get("currency",     "INR"),
                    "exchange": meta.get("exchangeName", "NSE"),
                }
            except Exception:
                return base
        try:
            import yfinance as yf
            info = yf.Ticker(self.symbol, session=_SESSION).info or {}
            return {
                "symbol":     self.symbol,
                "name":       info.get("longName",          self.symbol),
                "sector":     info.get("sector",            "N/A"),
                "currency":   info.get("currency",          "USD"),
                "exchange":   info.get("exchange",          "N/A"),
                "market_cap": info.get("marketCap"),
                "pe_ratio":   info.get("trailingPE"),
                "52w_high":   info.get("fiftyTwoWeekHigh"),
                "52w_low":    info.get("fiftyTwoWeekLow"),
                "beta":       info.get("beta"),
            }
        except Exception as e:
            logger.warning("[%s] get_info failed (ok): %s", self.symbol, e)
            return base

    def get_headlines(self, max_items: int = 15) -> list:
        try:
            import yfinance as yf
            news = yf.Ticker(self.symbol, session=_SESSION).news or []
            return [n.get("title", "") for n in news[:max_items] if n.get("title")]
        except Exception:
            return []

    def get_live_price(self) -> dict:
        """Fast single-price lookup used by WebSocket / indices API."""
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/"
               f"{self.symbol}?range=5d&interval=1d")
        r = _SESSION.get(url, timeout=6)
        r.raise_for_status()
        data   = r.json()
        result = data.get("chart", {}).get("result")
        if not result:
            raise RuntimeError(f"[{self.symbol}] live price: no result from Yahoo")
        meta      = result[0]["meta"]
        price     = meta["regularMarketPrice"]
        prevClose = (meta.get("previousClose")
                     or meta.get("chartPreviousClose")
                     or price)
        change    = round(price - prevClose, 2)
        changePct = round((change / prevClose) * 100, 2) if prevClose else 0
        return {
            "symbol":        self.symbol,
            "price":         round(price, 2),
            "change":        change,
            "changePercent": changePct,
            "open":          meta.get("regularMarketOpen"),
            "currency":      meta.get("currency",
                                      "INR" if self._is_indian() else "USD"),
        }

    # ── Download orchestration ────────────────────────────────────────────────

    def _download(self) -> pd.DataFrame:
        errors = []

        # ── INDEX FAST PATH ───────────────────────────────────────────────────
        # Stooq now requires a paid API key for ^BSE — skip it entirely.
        # Yahoo v8 handles all ^ symbols natively and reliably.
        if is_index(self.symbol):
            try:
                df = self._from_yfinance()
                if not df.empty:
                    logger.info("[%s] yfinance OK (index fast-path)", self.symbol)
                    return df
            except Exception as e:
                errors.append(f"yfinance: {e}")

            # NSE India as secondary fallback (NIFTY 50 + BANK NIFTY only)
            try:
                _NSE_MAP = {"^NSEI": "NIFTY 50", "^NSEBANK": "NIFTY BANK"}
                name = _NSE_MAP.get(self.symbol)
                if not name:
                    raise ValueError(f"NSE India: no mapping for {self.symbol}")
                from stock_nse_india import NseIndia  # type: ignore
                nse  = NseIndia()
                data = nse.getEquityIndices(name)
                if not data or not data.get("last"):
                    raise ValueError(f"NSE India: empty data for {name}")
                price = float(data["last"])
                df = pd.DataFrame([{
                    "open":   float(data.get("open",  price)),
                    "high":   float(data.get("high",  price)),
                    "low":    float(data.get("low",   price)),
                    "close":  price,
                    "volume": float(data.get("totalTradedVolume", 0) or 0),
                }], index=[pd.Timestamp.today().normalize()])
                df = self._clean(df)
                if not df.empty:
                    logger.info("[%s] NSE India OK", self.symbol)
                    return df
            except Exception as e:
                errors.append(f"NSE India: {e}")

            # Stale cache as last resort
            if self.cache_path.exists():
                try:
                    df = self._read_cache()
                    if not df.empty:
                        logger.warning("[%s] using stale cache (index)", self.symbol)
                        return df
                except Exception:
                    pass

            raise RuntimeError(
                f"[{self.symbol}] index data failed: {'; '.join(errors)}"
            )

        # ── EQUITY PATH: Stooq → AlphaVantage → Yahoo ─────────────────────────
        try:
            df = self._from_stooq()
            if not df.empty:
                logger.info("[%s] Stooq OK", self.symbol)
                return df
        except Exception as e:
            errors.append(f"Stooq: {e}")

        key = os.getenv("ALPHA_VANTAGE_KEY", "").strip()
        if key:
            try:
                df = self._from_alphavantage(key)
                if not df.empty:
                    logger.info("[%s] AlphaVantage OK", self.symbol)
                    return df
            except Exception as e:
                errors.append(f"AV: {e}")

        try:
            df = self._from_yfinance()
            if not df.empty:
                logger.info("[%s] yfinance OK", self.symbol)
                return df
        except Exception as e:
            errors.append(f"yfinance: {e}")

        if self.cache_path.exists():
            try:
                df = self._read_cache()
                if not df.empty:
                    logger.warning("[%s] using stale cache", self.symbol)
                    return df
            except Exception:
                pass

        raise RuntimeError(
            f"[{self.symbol}] all sources failed: {'; '.join(errors)}"
        )

    # ── Source: Yahoo Finance v8 direct ───────────────────────────────────────

    def _from_yfinance(self) -> pd.DataFrame:
        days  = _PERIOD_DAYS.get(self.period, 730)
        end   = int(datetime.today().timestamp())
        start = int((datetime.today() - timedelta(days=days)).timestamp())
        iv    = {"1d": "1d", "1wk": "1wk", "1mo": "1mo"}.get(self.interval, "1d")

        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{self.symbol}"
               f"?period1={start}&period2={end}&interval={iv}&events=history")
        resp = _SESSION.get(url, timeout=12)
        resp.raise_for_status()
        data   = resp.json()
        result = data.get("chart", {}).get("result")
        if not result:
            err = data.get("chart", {}).get("error", {})
            raise ValueError(f"Yahoo v8: {err}")

        r          = result[0]
        timestamps = r.get("timestamp", [])
        q          = r.get("indicators", {}).get("quote",    [{}])[0]
        adj        = r.get("indicators", {}).get("adjclose", [{}])
        adjclose   = adj[0].get("adjclose", []) if adj else []

        if not timestamps:
            raise ValueError("Yahoo v8: empty timestamp list")

        close_vals = (adjclose if len(adjclose) == len(timestamps)
                      else q.get("close", [None] * len(timestamps)))

        df = pd.DataFrame({
            "date":   pd.to_datetime(timestamps, unit="s"),
            "open":   q.get("open",   [None] * len(timestamps)),
            "high":   q.get("high",   [None] * len(timestamps)),
            "low":    q.get("low",    [None] * len(timestamps)),
            "close":  close_vals,
            "volume": q.get("volume", [None] * len(timestamps)),
        })
        df["date"] = df["date"].dt.tz_localize(None)
        return self._clean(df.set_index("date").sort_index())

    # ── Source: Stooq ─────────────────────────────────────────────────────────

    def _from_stooq(self) -> pd.DataFrame:
        sym    = _stooq_sym(self.symbol)
        d1, d2 = _dates(self.period)
        url    = f"https://stooq.com/q/d/l/?s={sym}&d1={d1}&d2={d2}&i=d"
        r      = _SESSION.get(url, timeout=10)
        r.raise_for_status()
        text = r.text.strip()
        if (len(text) < 50
                or "No data" in text
                or "Exceeded" in text
                or "apikey" in text.lower()
                or "captcha" in text.lower()):
            raise ValueError(f"Stooq: empty/blocked/rate-limited for '{sym}'")
        try:
            df = pd.read_csv(io.StringIO(text))
        except Exception as e:
            raise ValueError(
                f"Stooq: CSV parse error for '{sym}' "
                f"(raw={text[:120]!r}): {e}"
            ) from e
        df.columns = [c.strip().lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"])
        return self._clean(df.set_index("date").sort_index())

    # ── Source: AlphaVantage ──────────────────────────────────────────────────

    def _from_alphavantage(self, key: str) -> pd.DataFrame:
        s         = self.symbol
        av_symbol = _av_sym(s)
        if av_symbol == "__INDEX_NOT_SUPPORTED__":
            raise ValueError(f"AV: index symbol '{s}' not supported")

        if s.endswith("=X"):
            parts    = av_symbol.split("/")
            from_cur = parts[0] if len(parts) == 2 else av_symbol[:3]
            to_cur   = parts[1] if len(parts) == 2 else av_symbol[3:]
            url = (f"https://www.alphavantage.co/query"
                   f"?function=FX_DAILY&from_symbol={from_cur}&to_symbol={to_cur}"
                   f"&outputsize=full&apikey={key}&datatype=csv")
        elif "-USD" in s or "-BTC" in s or "-ETH" in s:
            base   = s.split("-")[0]
            market = s.split("-")[1] if "-" in s else "USD"
            url = (f"https://www.alphavantage.co/query"
                   f"?function=DIGITAL_CURRENCY_DAILY&symbol={base}&market={market}"
                   f"&apikey={key}&datatype=csv")
        else:
            url = (f"https://www.alphavantage.co/query"
                   f"?function=TIME_SERIES_DAILY_ADJUSTED&symbol={av_symbol}"
                   f"&outputsize=full&apikey={key}&datatype=csv")

        r    = _SESSION.get(url, timeout=10)
        r.raise_for_status()
        text = r.text.strip()
        if text.startswith("{") or "Invalid" in text or "Thank you" in text:
            import json
            try:
                msg  = json.loads(text)
                note = (msg.get("Note") or msg.get("Information")
                        or msg.get("Error Message") or text[:200])
            except Exception:
                note = text[:200]
            raise ValueError(f"AV: {note}")

        df = pd.read_csv(io.StringIO(text))
        df.columns = [c.lower().strip() for c in df.columns]
        date_col = next(
            (c for c in df.columns if "time" in c or "date" in c), None
        )
        if not date_col:
            raise ValueError(f"AV: no date column. Got: {list(df.columns)}")
        df = df.rename(columns={date_col: "date"})
        if "adjusted_close" in df.columns:
            df = df.rename(columns={"adjusted_close": "close"})
        close_col = next(
            (c for c in df.columns if c == "close" or c.startswith("close")), None
        )
        if close_col and close_col != "close":
            df = df.rename(columns={close_col: "close"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date").sort_index()
        cutoff = (datetime.today() - timedelta(
            days=_PERIOD_DAYS.get(self.period, 730)
        )).strftime("%Y-%m-%d")
        df = df[df.index >= cutoff]
        cols = [c for c in ["open", "high", "low", "close", "volume"]
                if c in df.columns]
        return self._clean(df[cols])

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _is_indian(self) -> bool:
        s = self.symbol
        return (any(s.endswith(x) for x in (".NS", ".BO", ".NSE", ".BSE"))
                or s in ("^NSEI", "^BSESN", "^NSEBANK"))

    def _read_cache(self) -> pd.DataFrame:
        df = pd.read_csv(self.cache_path, index_col=0, parse_dates=True)
        return self._clean(df)

    def _save(self, df: pd.DataFrame):
        try:
            df.to_csv(self.cache_path)
        except Exception as e:
            logger.warning("[%s] cache save failed: %s", self.symbol, e)

    def _cache_fresh(self) -> bool:
        if not self.cache_path.exists():
            return False
        age_hours = (
            (datetime.now().timestamp() - self.cache_path.stat().st_mtime) / 3600
        )
        return age_hours < CACHE_TTL_HOURS

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        cols = [c for c in ["open", "high", "low", "close", "volume"]
                if c in df.columns]
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


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADER SERVICE SINGLETON
# ══════════════════════════════════════════════════════════════════════════════

class DataLoader:
    def __init__(self):
        self._cache: dict = {}

    def _loader(self, symbol: str, interval: str = "1d") -> StockDataLoader:
        sym = _canonicalise(symbol)          # normalise before cache key
        key = f"{sym}_{interval}"
        if key not in self._cache:
            self._cache[key] = StockDataLoader(sym, interval=interval)
        return self._cache[key]

    def get(self, symbol: str, interval: str = "1d",
            force_refresh: bool = False) -> pd.DataFrame:
        return self._loader(symbol, interval).load(force_refresh)

    def get_info(self, symbol: str) -> dict:
        return self._loader(symbol).get_info()

    def get_headlines(self, symbol: str, n: int = 15) -> list:
        return self._loader(symbol).get_headlines(n)

    def get_live_price(self, symbol: str) -> dict:
        return self._loader(symbol).get_live_price()

    def get_macro(self) -> pd.DataFrame:
        try:
            return MacroLoader().get()
        except Exception:
            return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE
# ══════════════════════════════════════════════════════════════════════════════

def load_stock(symbol: str, period: str = "2y") -> pd.DataFrame:
    return StockDataLoader(symbol, period=period).load()


# ══════════════════════════════════════════════════════════════════════════════
# MACRO LOADER
# ══════════════════════════════════════════════════════════════════════════════

class MacroLoader:
    _SYMBOLS = {
        "vix":   "^VIX",
        "sp500": "^GSPC",
        "dxy":   "DX-Y.NYB",
        "oil":   "CL=F",
        "gold":  "GC=F",
        "tnx":   "^TNX",
    }
    _CACHE = DATA_DIR / "_macro.csv"
    _TTL   = 6 * 3600

    def get(self, period: str = "1y", force_refresh: bool = False) -> pd.DataFrame:
        if not force_refresh and self._fresh():
            try:
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
        snap["risk_on_score"] = round(
            max(0.0, min(1.0, 1.0 - (vix - 15) / 25)), 2
        )
        try:
            df  = self.get()
            sp  = df["sp500"].dropna()
            snap["sp500_trend"] = (
                "up" if sp.iloc[-1] > sp.rolling(20).mean().iloc[-1] else "down"
            )
        except Exception:
            snap["sp500_trend"] = "unknown"
        return snap

    def _download(self, period: str) -> pd.DataFrame:
        frames: dict = {}
        for name, sym in self._SYMBOLS.items():
            try:
                # Use yfinance directly for all macro symbols (indices + futures)
                df = StockDataLoader(sym, period=period)._from_yfinance()
                if not df.empty:
                    frames[name] = df["close"].rename(name)
            except Exception as e:
                logger.warning("[MacroLoader] %s (%s) failed: %s", name, sym, e)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames.values(), axis=1).sort_index()

    def _fresh(self) -> bool:
        return (self._CACHE.exists()
                and (time.time() - self._CACHE.stat().st_mtime) < self._TTL)