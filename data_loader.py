"""
data_loader.py — Stock data pipeline PRODUCTION v4.2
═════════════════════════════════════════════════════
SYMBOL SUPPORT — all asset classes:
  ✓ US Stocks/ETFs        AAPL, SPY, QQQ
  ✓ US Indices            ^GSPC, ^DJI, ^IXIC, ^VIX, ^TNX
  ✓ Crypto                BTC-USD, ETH-USD, SOL-USD
  ✓ Forex                 EURUSD=X, GBPUSD=X, USDJPY=X
  ✓ Commodities/Futures   GC=F, CL=F, SI=F, NG=F
  ✓ India NSE/BSE         RELIANCE.NS, TCS.NS, INFY.BO
  ✓ India Indices         ^NSEI, ^BSESN, NIFTY50, SENSEX
                          NIFTY50.NS, SENSEX.BO  ← v4.2 fix
  ✓ UK / Ireland          HSBA.L, BP.L
  ✓ Canada                SHOP.TO, BB.V
  ✓ Japan                 7203.T, 6758.T
  ✓ Hong Kong             0700.HK, 9988.HK
  ✓ Australia             CBA.AX, BHP.AX
  ✓ Germany XETRA         SAP.DE, BMW.DE
  ✓ Frankfurt stocks      SIE.F, BMW.F  → mapped to .DE
  ✓ France                MC.PA, AIR.PA
  ✓ Switzerland           NESN.SW, ROG.SW
  ✓ Netherlands           ASML.AS, RDSA.AS
  ✓ Italy                 ENI.MI, ISP.MI
  ✓ Spain                 ITX.MC, SAN.MC
  ✓ Norway                EQNR.OL, DNB.OL
  ✓ Sweden                VOLV-B.ST, ERIC-B.ST
  ✓ Denmark               NOVO-B.CO, MAERSK-B.CO
  ✓ Finland               NOKIA.HE, FORTUM.HE
  ✓ Portugal              EDP.LS, GALP.LS
  ✓ Belgium               AB.BR, UCB.BR
  ✓ Austria               OMV.VI, ERS.VI
  ✓ Poland                PKN.WA, PKO.WA
  ✓ Turkey                THYAO.IS, GARAN.IS
  ✓ South Korea           005930.KS, 000660.KS
  ✓ Taiwan                2330.TW, 2317.TWO
  ✓ China                 600519.SS, 000858.SZ
  ✓ Malaysia              1155.KL, 5168.KL
  ✓ Thailand              PTT.BK, ADVANC.BK
  ✓ Indonesia             BBCA.JK, TLKM.JK
  ✓ Singapore             D05.SG, U11.SG
  ✓ New Zealand           FPH.NZ, AIR.NZ
  ✓ Brazil                PETR4.SA, VALE3.SA
  ✓ Russia                SBER.ME, GAZP.ME
  ✓ South Africa          NPN.JO, AGL.JO

DATA SOURCES (priority order):
  1. Stooq     — primary, free, no API key
  2. AlphaVantage — fallback if ALPHA_VANTAGE_KEY env set
  3. yfinance  — final fallback (direct Yahoo Finance v8 API)
  4. Stale cache — last resort if all live sources fail

CHANGES vs v4.1:
  ✓ FIXED:  NIFTY50.NS / SENSEX.BO / NIFTY.NS alias normalisation in
            _stooq_sym() — callers that append .NS/.BO to index names no
            longer fall through to the suffix table and produce a bad Stooq
            symbol, causing "Expected 1 fields … saw 2" CSV parse errors.
  ✓ FIXED:  _from_stooq() now wraps pd.read_csv in try/except and raises a
            descriptive ValueError (including raw response snippet) instead
            of the opaque pandas C-engine tokenisation error.
  ✓ FIXED:  ^AXJO exact-map entry corrected — was wrongly mapped to ^ATX
            (Austrian index); now mapped to ^AXJO passthrough (Stooq
            supports it directly). Added note about ^ATX ambiguity.
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


# ══════════════════════════════════════════════════════════════════════════════
# STOOQ SYMBOL TRANSLATION
# ══════════════════════════════════════════════════════════════════════════════
#
# Priority order (first match wins):
#   0. Index/alias prefix normalisation  ← NEW in v4.2
#      Strips exchange suffixes from well-known index aliases so that
#      NIFTY50.NS, NIFTY50, and ^NSEI all resolve to the same Stooq symbol.
#   1. Exact hardcoded map  (special indices, well-known aliases)
#   2. Index passthrough    (^ prefix → as-is)
#   3. Forex                (=X suffix → .fx)
#   4. Futures              (=F suffix → .F)
#   5. Crypto               (-USD infix → usd.cx)
#   6. Exchange suffixes    (longest-suffix-first table — 40+ exchanges)
#   7. Frankfurt stocks     (.F suffix with ≥3-char base → .DE)
#   8. US default           (everything else → .US)
#
# Sorted longest-first so e.g. ".TWO" is matched before ".T" would be.

_YF_TO_STOOQ: list[tuple[str, str]] = sorted([
    # ── India ────────────────────────────────────────────────────────────────
    (".NS",  ".NS"),   # NSE (National Stock Exchange)
    (".BO",  ".BO"),   # BSE (Bombay Stock Exchange)

    # ── UK / Ireland ─────────────────────────────────────────────────────────
    (".L",   ".UK"),   # London Stock Exchange
    (".IL",  ".UK"),   # London (pence-quoted instruments)

    # ── North America ─────────────────────────────────────────────────────────
    (".TO",  ".CA"),   # Toronto Stock Exchange (TSX)
    (".V",   ".CA"),   # TSX Venture Exchange (Vancouver)
    (".CN",  ".CA"),   # Canadian Securities Exchange

    # ── Asia-Pacific ──────────────────────────────────────────────────────────
    (".T",   ".JP"),   # Tokyo Stock Exchange
    (".HK",  ".HK"),   # Hong Kong Stock Exchange (HKEX)
    (".AX",  ".AU"),   # Australian Securities Exchange (ASX)
    (".NZ",  ".NZ"),   # New Zealand Exchange (NZX)
    (".SG",  ".SG"),   # Singapore Exchange (SGX)
    (".TW",  ".TW"),   # Taiwan Stock Exchange (TWSE)
    (".TWO", ".TW"),   # Taiwan OTC (Taipei Exchange) — check before .TW
    (".KS",  ".KR"),   # Korea Stock Exchange (KOSPI)
    (".KQ",  ".KR"),   # KOSDAQ
    (".SS",  ".CN"),   # Shanghai Stock Exchange
    (".SZ",  ".CN"),   # Shenzhen Stock Exchange
    (".KL",  ".MY"),   # Bursa Malaysia
    (".BK",  ".TH"),   # Stock Exchange of Thailand (SET)
    (".JK",  ".JK"),   # Indonesia Stock Exchange (IDX)

    # ── Europe — major ────────────────────────────────────────────────────────
    (".DE",  ".DE"),   # Germany XETRA
    (".PA",  ".FR"),   # Euronext Paris
    (".SW",  ".SW"),   # SIX Swiss Exchange
    (".AS",  ".NL"),   # Euronext Amsterdam
    (".MI",  ".IT"),   # Borsa Italiana (Milan)
    (".MC",  ".ES"),   # Bolsa de Madrid
    (".OL",  ".NO"),   # Oslo Bors (Norway)
    (".ST",  ".SE"),   # Nasdaq Stockholm (Sweden)
    (".CO",  ".DK"),   # Nasdaq Copenhagen (Denmark)
    (".HE",  ".FI"),   # Nasdaq Helsinki (Finland)

    # ── Europe — smaller ──────────────────────────────────────────────────────
    (".LS",  ".PT"),   # Euronext Lisbon (Portugal)
    (".BR",  ".BE"),   # Euronext Brussels (Belgium)
    (".VI",  ".AT"),   # Wiener Börse (Austria)
    (".WA",  ".PL"),   # Warsaw Stock Exchange (Poland)
    (".IS",  ".IS"),   # Borsa Istanbul (Turkey)
    (".PR",  ".PR"),   # Prague Stock Exchange
    (".BD",  ".BD"),   # Budapest Stock Exchange
    (".IC",  ".IC"),   # Nasdaq Iceland

    # ── Other regions ─────────────────────────────────────────────────────────
    (".ME",  ".RU"),   # Moscow Exchange (Russia)
    (".SA",  ".BR"),   # B3 (Brazil) — Stooq uses .BR for Brazil
    (".JO",  ".SJ"),   # Johannesburg Stock Exchange (South Africa)
], key=lambda x: -len(x[0]))   # ← longest suffix first — critical for .TWO vs .T etc.


# ── v4.2: index/alias prefix normalisation ────────────────────────────────────
#
# Maps a bare alias prefix → its canonical Stooq symbol.
# Matched when the incoming symbol equals the alias exactly, OR starts with
# the alias followed by "." or "-" (e.g. NIFTY50.NS, NIFTY50-USD).
# The delimiter check prevents e.g. "NIFTY500.NS" from matching "NIFTY50".
#
# Sorted longest-first so more-specific prefixes (NIFTY50) are checked
# before shorter ones (NIFTY) and can't be shadowed.
_INDEX_ALIAS_PREFIX: list[tuple[str, str]] = sorted([
    ("NIFTY50",  "^NII50"),   # Nifty 50 — canonical
    ("NIFTY500", "^NII500"),  # Nifty 500 (if Stooq carries it)
    ("NIFTY",    "^NII50"),   # bare "NIFTY" alias
    ("SENSEX",   "^BSE"),     # BSE Sensex
    ("BANKNIFTY","^NSEBANK"), # Bank Nifty (Stooq symbol may vary)
], key=lambda x: -len(x[0]))  # longest first


def _stooq_sym(symbol: str) -> str:
    """
    Translate a yfinance-style ticker to its Stooq equivalent.
    Never raises — always returns a best-effort string.
    """
    s = symbol.upper().strip()

    # 0. Index/alias prefix normalisation (v4.2)
    #    Handles NIFTY50.NS, NIFTY50, SENSEX.BO, SENSEX, BANKNIFTY.NS, …
    for alias, stooq_canonical in _INDEX_ALIAS_PREFIX:
        if s == alias or s.startswith(alias + ".") or s.startswith(alias + "-"):
            return stooq_canonical

    # 1. Exact hardcoded map — special indices and well-known aliases
    _exact: dict[str, str] = {
        "^GSPC":    "^SPX",     # S&P 500
        "^DJI":     "^DJI",     # Dow Jones
        "^IXIC":    "^NDQ",     # Nasdaq Composite
        "^VIX":     "^VIX",     # CBOE Volatility Index
        "^TNX":     "^TNX",     # 10-Year Treasury Yield
        "CL=F":     "CL.F",     # Crude Oil WTI futures
        "GC=F":     "GC.F",     # Gold futures
        "SI=F":     "SI.F",     # Silver futures
        "NG=F":     "NG.F",     # Natural Gas futures
        "HG=F":     "HG.F",     # Copper futures
        "ZW=F":     "ZW.F",     # Wheat futures
        "ZC=F":     "ZC.F",     # Corn futures
        "DX-Y.NYB": "DXY.F",    # US Dollar Index
        "^NSEI":    "^NII50",   # Nifty 50
        "^BSESN":   "^BSE",     # BSE Sensex
        "^FTSE":    "^FTM",     # FTSE 100
        "^GDAXI":   "^DAX",     # DAX
        "^FCHI":    "^CAC",     # CAC 40
        "^N225":    "^NKX",     # Nikkei 225
        "^HSI":     "^HSI",     # Hang Seng
        # NOTE: ^AXJO (ASX 200) passes through to Stooq as-is.
        # ^ATX is the Austrian ATX index on Stooq — do NOT conflate with ASX 200.
        "^AXJO":    "^AXJO",    # ASX 200 — Stooq supports directly
        "^KS11":    "^KS11",    # KOSPI
        "^TWII":    "^TWII",    # Taiwan TAIEX
        "^STI":     "^STI",     # Straits Times Index
    }
    if s in _exact:
        return _exact[s]

    # 2. Index passthrough — all other ^ symbols go straight to Stooq as-is
    if s.startswith("^"):
        return s

    # 3. Forex — yfinance EURUSD=X → Stooq eurusd.fx
    if s.endswith("=X"):
        return s[:-2].lower() + ".fx"

    # 4. Futures — yfinance GC=F → Stooq GC.F
    #    Handle both trailing =F and embedded =F (e.g. CL=F already in exact map)
    if s.endswith("=F") or "=F" in s:
        return s.replace("=F", ".F")

    # 5. Crypto — yfinance BTC-USD → Stooq btcusd.cx
    if "-USD" in s:
        return s.lower().replace("-usd", "usd") + ".cx"

    # 6. International exchange suffixes (longest-first table)
    for yf_sfx, stooq_sfx in _YF_TO_STOOQ:
        if s.endswith(yf_sfx.upper()):
            base = s[: len(s) - len(yf_sfx)]
            return base + stooq_sfx

    # 7. Frankfurt .F stock tickers (e.g. SIE.F, BMW.F, VOW3.F)
    #    Futures commodity codes are always 2 chars (GC, CL, SI …)
    #    Frankfurt stock tickers are ≥ 3 chars → use .DE (XETRA)
    if s.endswith(".F"):
        base = s[:-2]
        if len(base) >= 3:
            return base + ".DE"
        # 2-char base → treat as futures passthrough
        return s

    # 8. Default: assume US-listed stock/ETF
    return s + ".US"


# ══════════════════════════════════════════════════════════════════════════════
# ALPHAVANTAGE SYMBOL TRANSLATION
# ══════════════════════════════════════════════════════════════════════════════
#
# AV has its own suffix conventions, different from both Stooq and yfinance.
# Also handles special API function routing (FX_DAILY, DIGITAL_CURRENCY_DAILY).

_YF_TO_AV: list[tuple[str, str]] = sorted([
    # India
    (".NS",  ".BSE"),
    (".BO",  ".BSE"),
    # UK
    (".L",   ".LON"),
    (".IL",  ".LON"),
    # Canada
    (".TO",  ".TRT"),
    (".V",   ".TRT"),
    (".CN",  ".TRT"),
    # Asia-Pacific
    (".T",   ".TSE"),
    (".HK",  ".HKEX"),
    (".AX",  ".ASX"),
    (".NZ",  ".NZX"),
    (".SG",  ".SGX"),
    (".TW",  ".TSEC"),
    (".TWO", ".TSEC"),
    (".KS",  ".KSC"),
    (".KQ",  ".KSC"),
    (".SS",  ".SHH"),
    (".SZ",  ".SHZ"),
    (".KL",  ".MYX"),
    (".BK",  ".SET"),
    (".JK",  ".JKT"),
    # Europe
    (".DE",  ".DEX"),
    (".PA",  ".PAR"),
    (".SW",  ".SWX"),
    (".AS",  ".AMS"),
    (".MI",  ".MIL"),
    (".MC",  ".BME"),
    (".OL",  ".OSL"),
    (".ST",  ".STO"),
    (".CO",  ".CPH"),
    (".HE",  ".HEL"),
    (".LS",  ".LIS"),
    (".BR",  ".BRU"),
    (".VI",  ".WBO"),
    (".WA",  ".WAR"),
    (".IS",  ".IST"),
    # Other
    (".ME",  ".MCX"),
    (".SA",  ".SAO"),
    (".JO",  ".JSE"),
], key=lambda x: -len(x[0]))


def _av_sym(symbol: str) -> str:
    """
    Translate a yfinance-style ticker to its AlphaVantage equivalent.
    Returns a (av_symbol, av_function) tuple is NOT needed here —
    the function selection is done in _from_alphavantage() based on the
    original symbol, which is cleaner. This just returns the symbol string.
    """
    s = symbol.upper().strip()

    # Normalise index aliases first (mirrors step 0 in _stooq_sym)
    # AV doesn't have these indices, but normalisation prevents AV from
    # being called with NIFTY50.NS and treating it as an equity.
    for alias, _ in _INDEX_ALIAS_PREFIX:
        if s == alias or s.startswith(alias + ".") or s.startswith(alias + "-"):
            # Return a clearly invalid AV symbol so _from_alphavantage() fails
            # fast and we fall through to yfinance rather than hitting AV quota.
            return "__INDEX_NOT_SUPPORTED__"

    # Forex: EURUSD=X → "EUR/USD"
    if s.endswith("=X"):
        pair = s[:-2]
        if len(pair) == 6:
            return pair[:3] + "/" + pair[3:]
        return pair

    # Futures: strip =F — AV uses the root symbol
    if s.endswith("=F") or "=F" in s:
        return s.replace("=F", "").replace("=", "")

    # Crypto: BTC-USD → "BTC"
    if "-USD" in s or "-BTC" in s or "-ETH" in s:
        return s.split("-")[0]

    # Frankfurt .F stock tickers → .DEX
    if s.endswith(".F"):
        base = s[:-2]
        if len(base) >= 3:
            return base + ".DEX"
        return base   # commodity root symbol

    # International exchange suffixes
    for yf_sfx, av_sfx in _YF_TO_AV:
        if s.endswith(yf_sfx.upper()):
            base = s[: len(s) - len(yf_sfx)]
            return base + av_sfx

    # US / default — return as-is
    return s


# ══════════════════════════════════════════════════════════════════════════════
# STOCK DATA LOADER
# ══════════════════════════════════════════════════════════════════════════════

class StockDataLoader:
    def __init__(self, symbol: str, period: str = "2y", interval: str = "1d"):
        self.symbol     = symbol.upper().strip()
        self.period     = period
        self.interval   = interval
        # Sanitise the symbol for use as a filename (remove /, ^, =, - etc.)
        safe = self.symbol.replace("/", "_").replace("^", "").replace("=", "_").replace("-", "_")
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
        logger.info("[%s] downloading …", self.symbol)
        df = self._download()
        self._save(df)
        return df

    def get_info(self) -> dict:
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
            return {"symbol": self.symbol, "name": self.symbol, "currency": "USD"}

    def get_headlines(self, max_items: int = 15) -> list:
        try:
            import yfinance as yf
            news = yf.Ticker(self.symbol, session=_SESSION).news or []
            return [n.get("title", "") for n in news[:max_items] if n.get("title")]
        except Exception:
            return []

    # ── Download orchestration ────────────────────────────────────────────────

    def _download(self) -> pd.DataFrame:
        errors = []

        # 1. Stooq (primary — free, no key required)
        try:
            df = self._from_stooq()
            if not df.empty:
                logger.info("[%s] Stooq OK", self.symbol)
                return df
        except Exception as e:
            errors.append(f"Stooq: {e}")

        # 2. AlphaVantage (fallback — requires ALPHA_VANTAGE_KEY env var)
        key = os.getenv("ALPHA_VANTAGE_KEY", "").strip()
        if key:
            try:
                df = self._from_alphavantage(key)
                if not df.empty:
                    logger.info("[%s] AlphaVantage OK", self.symbol)
                    return df
            except Exception as e:
                errors.append(f"AV: {e}")

        # 3. yfinance direct Yahoo Finance v8 API (final live fallback)
        try:
            df = self._from_yfinance()
            if not df.empty:
                logger.info("[%s] yfinance OK", self.symbol)
                return df
        except Exception as e:
            errors.append(f"yfinance: {e}")

        # 4. Stale cache (last resort — keeps app alive if all live sources fail)
        if self.cache_path.exists():
            try:
                df = self._read_cache()
                if not df.empty:
                    logger.warning("[%s] all live sources failed — using stale cache", self.symbol)
                    return df
            except Exception:
                pass

        raise RuntimeError(f"[{self.symbol}] all sources failed: {'; '.join(errors)}")

    # ── Source: Stooq ─────────────────────────────────────────────────────────

    def _from_stooq(self) -> pd.DataFrame:
        sym    = _stooq_sym(self.symbol)
        d1, d2 = _dates(self.period)
        url    = f"https://stooq.com/q/d/l/?s={sym}&d1={d1}&d2={d2}&i=d"
        r      = _SESSION.get(url, timeout=10)
        r.raise_for_status()
        text   = r.text.strip()
        if len(text) < 50 or "No data" in text or "Exceeded" in text:
            raise ValueError(f"Stooq: empty/blocked response for symbol '{sym}'")
        # v4.2: wrap CSV parsing so callers get a descriptive error instead of
        # an opaque pandas C-engine tokenisation traceback.
        try:
            df = pd.read_csv(io.StringIO(text))
        except Exception as e:
            raise ValueError(
                f"Stooq: CSV parse error for symbol '{sym}' "
                f"(raw={text[:120]!r}): {e}"
            ) from e
        df.columns = [c.strip().lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"])
        return self._clean(df.set_index("date").sort_index())

    # ── Source: AlphaVantage ──────────────────────────────────────────────────

    def _from_alphavantage(self, key: str) -> pd.DataFrame:
        s = self.symbol.upper()
        av_symbol = _av_sym(self.symbol)

        # Fast-fail for symbols that AV does not support (e.g. Indian indices)
        if av_symbol == "__INDEX_NOT_SUPPORTED__":
            raise ValueError(
                f"AV: symbol '{self.symbol}' is a known index not supported by AlphaVantage"
            )

        # Route to the correct AV endpoint based on asset class
        if s.endswith("=X"):
            # Forex pair
            parts = av_symbol.split("/")
            if len(parts) == 2:
                from_cur, to_cur = parts
            else:
                from_cur, to_cur = av_symbol[:3], av_symbol[3:]
            url = (f"https://www.alphavantage.co/query"
                   f"?function=FX_DAILY&from_symbol={from_cur}&to_symbol={to_cur}"
                   f"&outputsize=full&apikey={key}&datatype=csv")

        elif "-USD" in s or "-BTC" in s or "-ETH" in s:
            # Cryptocurrency
            base   = s.split("-")[0]
            market = s.split("-")[1] if "-" in s else "USD"
            url = (f"https://www.alphavantage.co/query"
                   f"?function=DIGITAL_CURRENCY_DAILY&symbol={base}&market={market}"
                   f"&apikey={key}&datatype=csv")

        else:
            # Equities, ETFs, indices, commodities, international stocks
            url = (f"https://www.alphavantage.co/query"
                   f"?function=TIME_SERIES_DAILY_ADJUSTED&symbol={av_symbol}"
                   f"&outputsize=full&apikey={key}&datatype=csv")

        r    = _SESSION.get(url, timeout=10)
        r.raise_for_status()
        text = r.text.strip()

        # AV returns JSON on error (rate limit, invalid symbol, etc.)
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

        # Normalise date column (AV uses "timestamp" or "date" depending on endpoint)
        date_col = next((c for c in df.columns if "time" in c or "date" in c), None)
        if date_col is None:
            raise ValueError(f"AV: no date column found. Got: {list(df.columns)}")
        df = df.rename(columns={date_col: "date"})

        # Normalise close column (adjusted_close preferred; crypto uses "close (usd)")
        if "adjusted_close" in df.columns:
            df = df.rename(columns={"adjusted_close": "close"})
        close_col = next(
            (c for c in df.columns if c == "close" or c.startswith("close")), None
        )
        if close_col and close_col != "close":
            df = df.rename(columns={close_col: "close"})

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date").sort_index()

        # Trim to requested period
        cutoff = (datetime.today() - timedelta(
            days=_PERIOD_DAYS.get(self.period, 730)
        )).strftime("%Y-%m-%d")
        df = df[df.index >= cutoff]

        cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        return self._clean(df[cols])

    # ── Source: yfinance (Yahoo Finance v8 direct) ────────────────────────────

    def _from_yfinance(self) -> pd.DataFrame:
        days  = _PERIOD_DAYS.get(self.period, 730)
        end   = int(datetime.today().timestamp())
        start = int((datetime.today() - timedelta(days=days)).timestamp())
        iv_map = {"1d": "1d", "1wk": "1wk", "1mo": "1mo"}
        iv     = iv_map.get(self.interval, "1d")

        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{self.symbol}"
               f"?period1={start}&period2={end}&interval={iv}&events=history")
        headers = {
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/122.0.0.0 Safari/537.36"),
            "Accept":          "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer":         "https://finance.yahoo.com/",
        }
        resp = _SESSION.get(url, headers=headers, timeout=10)
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

        # Prefer adjusted close when available and length matches
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

    # ── Cache helpers ─────────────────────────────────────────────────────────

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
        age_hours = (datetime.now().timestamp() - self.cache_path.stat().st_mtime) / 3600
        return age_hours < CACHE_TTL_HOURS

    # ── Data cleaning ─────────────────────────────────────────────────────────

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        """Normalise any raw OHLCV dataframe into a clean, consistent format."""
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        # Keep only standard OHLCV columns that exist
        cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        if "close" not in cols:
            return pd.DataFrame()
        df = df[cols]

        # Ensure tz-naive DatetimeIndex
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index = pd.to_datetime(df.index)

        # Drop all-NaN rows, then forward-fill and back-fill gaps
        df.dropna(how="all", inplace=True)
        df.ffill(inplace=True)
        df.bfill(inplace=True)

        # Remove nonsense rows
        df = df[df["close"] > 0]
        df = df[~df.index.duplicated(keep="last")]
        df.sort_index(inplace=True)
        return df


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADER — service singleton used by main.py
# ══════════════════════════════════════════════════════════════════════════════

class DataLoader:
    """
    Service-level singleton that wraps StockDataLoader instances.
    Caches loader objects in memory so repeated get() calls for the same
    symbol don't re-construct the loader or hit disk unnecessarily.
    """

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

    def get_macro(self) -> pd.DataFrame:
        """Alias used by legacy code."""
        try:
            return MacroLoader().get()
        except Exception:
            return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def load_stock(symbol: str, period: str = "2y") -> pd.DataFrame:
    """One-liner helper for scripts/notebooks."""
    return StockDataLoader(symbol, period=period).load()


# ══════════════════════════════════════════════════════════════════════════════
# MACRO LOADER
# ══════════════════════════════════════════════════════════════════════════════

class MacroLoader:
    """
    Downloads and caches a small set of macro indicators:
    VIX, S&P 500, DXY, WTI Oil, Gold, 10-Year Treasury.
    """
    _SYMBOLS = {
        "vix":   "^VIX",
        "sp500": "^GSPC",
        "dxy":   "DX-Y.NYB",
        "oil":   "CL=F",
        "gold":  "GC=F",
        "tnx":   "^TNX",
    }
    _CACHE = DATA_DIR / "_macro.csv"
    _TTL   = 6 * 3600   # seconds

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
        """Return latest values for all macro indicators plus derived signals."""
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

        # Risk-on score: 0 (risk-off) → 1 (risk-on), inverse of VIX normalised [15, 40]
        vix = snap.get("vix") or 20.0
        snap["risk_on_score"] = round(max(0.0, min(1.0, 1.0 - (vix - 15) / 25)), 2)

        # S&P 500 trend vs its 20-day moving average
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
                df = StockDataLoader(sym, period=period)._from_stooq()
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