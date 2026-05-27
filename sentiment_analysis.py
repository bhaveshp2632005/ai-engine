

import logging
import os
import re
from dataclasses import dataclass, field
from typing      import List, Optional

import httpx
import numpy as np

logger    = logging.getLogger(__name__)
LABEL_MAP = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}


def _load_finbert():
    """Stub — FinBERT disabled in production. Called in lifespan, safe to no-op."""
    logger.info("FinBERT disabled in production — using lexicon scorer")
    return None, None


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class NewsArticle:
    headline: str
    source:   str = "unknown"
    url:      str = ""


@dataclass
class SentimentResult:
    symbol:        str
    score:         float
    label:         str
    confidence:    float
    article_count: int
    articles:      list = field(default_factory=list)
    model_used:    str  = "lexicon"
    error:         Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "symbol":        self.symbol,
            "score":         round(self.score,      4),
            "label":         self.label,
            "confidence":    round(self.confidence * 100, 1),
            "article_count": self.article_count,
            "model_used":    self.model_used,
            "error":         self.error,
        }


# ── News Fetcher ──────────────────────────────────────────────────────────────

class NewsFetcher:
    HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; StockAnalyzer/3.0)"}

    def fetch(self, symbol: str, max_articles: int = 20) -> List[NewsArticle]:
        articles: List[NewsArticle] = []
        articles.extend(self._yahoo_rss(symbol, max_articles))

        gnews_key = os.getenv("GNEWS_API_KEY", "").strip()
        if gnews_key and len(articles) < max_articles:
            articles.extend(self._gnews(symbol, gnews_key, max_articles - len(articles)))

        fh_key = (os.getenv("FINNHUB_KEY", "") or os.getenv("FINNHUB_API_KEY", "")).strip()
        if fh_key and len(articles) < max_articles:
            articles.extend(self._finnhub(symbol, fh_key, max_articles - len(articles)))

        seen, unique = set(), []
        for a in articles:
            key = re.sub(r"\W+", "", a.headline.lower())[:60]
            if key and key not in seen:
                seen.add(key)
                unique.append(a)
        return unique[:max_articles]

    def _yahoo_rss(self, symbol: str, n: int) -> List[NewsArticle]:
        try:
            url  = (f"https://feeds.finance.yahoo.com/rss/2.0/headline"
                    f"?s={symbol}&region=US&lang=en-US")
            resp = httpx.get(url, headers=self.HEADERS, timeout=5, follow_redirects=True)
            items = re.findall(r"<title>(.*?)</title>", resp.text)[1:]
            return [
                NewsArticle(headline=h.strip(), source="Yahoo Finance")
                for h in items[:n] if len(h.strip()) > 10
            ]
        except Exception as e:
            logger.debug("Yahoo RSS failed for %s: %s", symbol, e)
            return []

    def _gnews(self, symbol: str, key: str, n: int) -> List[NewsArticle]:
        try:
            clean = re.sub(r"\.(NS|BO|L|TO)$", "", symbol.upper())
            data  = httpx.get(
                f"https://gnews.io/api/v4/search?q={clean}+stock&lang=en&max={n}&apikey={key}",
                timeout=5).json()
            return [
                NewsArticle(headline=a["title"], source="GNews", url=a.get("url", ""))
                for a in data.get("articles", []) if a.get("title")
            ]
        except Exception as e:
            logger.debug("GNews failed: %s", e)
            return []

    def _finnhub(self, symbol: str, key: str, n: int) -> List[NewsArticle]:
        try:
            from datetime import datetime, timedelta
            t = datetime.today()
            w = t - timedelta(days=7)
            data = httpx.get(
                f"https://finnhub.io/api/v1/company-news"
                f"?symbol={symbol}&from={w.strftime('%Y-%m-%d')}"
                f"&to={t.strftime('%Y-%m-%d')}&token={key}",
                timeout=5).json()
            if not isinstance(data, list):
                return []
            return [
                NewsArticle(headline=a["headline"], source=a.get("source","Finnhub"),
                            url=a.get("url",""))
                for a in data[:n] if a.get("headline")
            ]
        except Exception as e:
            logger.debug("Finnhub failed: %s", e)
            return []


# ── Sentiment Analyzer (lexicon-only) ────────────────────────────────────────

class SentimentAnalyzer:
    """
    Financial lexicon scorer — instant, zero RAM overhead.
    analyze() is guaranteed to never raise.
    """

    _POS = frozenset({
        "rise", "rises", "surges", "surge", "gain", "gains", "bull", "bullish",
        "record", "beat", "beats", "profit", "growth", "strong", "upgrade",
        "buy", "rally", "rallies", "positive", "exceeded", "upbeat", "outperform",
        "revenue", "robust", "momentum", "breakthrough", "expand",
    })
    _NEG = frozenset({
        "fall", "falls", "drop", "drops", "loss", "losses", "bear", "bearish",
        "miss", "misses", "decline", "weak", "downgrade", "sell", "crash",
        "negative", "concern", "concerns", "risk", "warning", "cut", "cuts",
        "layoff", "layoffs", "debt", "investigation", "fraud", "lawsuit",
        "downside", "underperform", "recession", "collapse",
    })

    def __init__(self):
        self.fetcher = NewsFetcher()

    def analyze(self, symbol: str, max_articles: int = 20) -> SentimentResult:
        try:
            return self._analyze_inner(symbol, max_articles)
        except Exception as e:
            logger.warning("SentimentAnalyzer.analyze failed for %s: %s", symbol, e)
            return SentimentResult(
                symbol=symbol, score=0.0, label="Neutral",
                confidence=0.0, article_count=0,
                error=str(e)[:100], model_used="error")

    def _analyze_inner(self, symbol: str, max_articles: int) -> SentimentResult:
        articles = self.fetcher.fetch(symbol, max_articles)
        if not articles:
            return SentimentResult(
                symbol=symbol, score=0.0, label="Neutral",
                confidence=0.3, article_count=0, error="No articles found")

        headlines     = [a.headline for a in articles]
        scores, confs = self._lexicon_score(headlines)
        scores        = np.array(scores, dtype=np.float32)
        confs         = np.array(confs,  dtype=np.float32)
        scores        = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        confs         = np.clip(np.nan_to_num(confs, nan=0.3), 0.05, 1.0)

        weights     = confs + 0.01
        final_score = float(np.clip(np.average(scores, weights=weights), -1.0, 1.0))
        final_conf  = float(np.mean(confs))

        return SentimentResult(
            symbol=symbol,
            score=round(final_score, 4),
            label=self._label(final_score),
            confidence=round(final_conf, 4),
            article_count=len(articles),
            articles=[{
                "headline": a.headline,
                "source":   a.source,
                "score":    round(float(s), 3),
                "label":    self._label(float(s)),
            } for a, s in zip(articles, scores)],
            model_used="lexicon",
        )

    def _lexicon_score(self, headlines):
        scores, confs = [], []
        for h in headlines:
            words = set(re.findall(r"\b\w+\b", h.lower()))
            p     = len(words & self._POS)
            n     = len(words & self._NEG)
            total = p + n
            if total == 0:
                scores.append(0.0); confs.append(0.2)
            else:
                scores.append(float((p - n) / total))
                confs.append(float(min(total / 4.0, 0.8)))
        return scores, confs

    @staticmethod
    def _label(score: float) -> str:
        if score > 0.15:  return "Positive"
        if score < -0.15: return "Negative"
        return "Neutral"
