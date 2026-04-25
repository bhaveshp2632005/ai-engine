"""
visualization.py — Professional dark-theme charts for AI prediction engine.
Charts: price, indicators, prediction, signals, portfolio, full_dashboard.
"""

import io, logging
from pathlib  import Path
from typing   import Optional

import matplotlib; matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot   as plt
import numpy  as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Dark-theme palette ──────────────────────────────────────────────────────
BG   = "#0D1117"; CARD = "#161B22"; TEXT = "#E6EDF3"
BLUE = "#58A6FF"; GREEN = "#3FB950"; RED  = "#F85149"
YEL  = "#E3B341"; GRID = "#21262D"

plt.rcParams.update({
    "figure.facecolor": BG,   "axes.facecolor":  CARD,
    "axes.edgecolor":   GRID, "axes.labelcolor": TEXT,
    "axes.titlecolor":  TEXT, "xtick.color":     TEXT,
    "ytick.color":      TEXT, "text.color":      TEXT,
    "grid.color":       GRID, "grid.linestyle":  "--",
    "grid.linewidth":   0.5,  "grid.alpha":      0.6,
    "legend.facecolor": CARD, "legend.edgecolor": GRID,
    "legend.labelcolor": TEXT, "font.size":        9,
})


def _style(*axes):
    for ax in axes:
        if ax is None: continue
        ax.set_facecolor(CARD)
        ax.tick_params(colors=TEXT, labelsize=7)
        ax.grid(True, color=GRID, linestyle="--", linewidth=0.5, alpha=0.6)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)


class Visualizer:
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _save(self, fig, filename: str):
        if self.output_dir:
            fig.savefig(self.output_dir / filename, dpi=120,
                        bbox_inches="tight", facecolor=BG)

    def fig_to_base64(self, fig) -> str:
        import base64
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=BG)
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.read()).decode()

    # ── Price Chart (Candlestick + Volume) ────────────────────────────────────

    def price_chart(self, df: pd.DataFrame, symbol: str = "", n_days: int = 180):
        df  = df.tail(n_days).copy()
        idx = np.arange(len(df))

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 8),
            gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
        fig.patch.set_facecolor(BG)
        _style(ax1, ax2)

        for i, (_, r) in enumerate(df.iterrows()):
            c = GREEN if r["close"] >= r["open"] else RED
            ax1.bar(i, r["close"] - r["open"], bottom=r["open"],
                    color=c, width=0.7, alpha=0.9)
            ax1.plot([i, i], [r["low"], r["high"]], color=c, linewidth=0.8)

        for col, col_c, lw in [("ma20", BLUE, 1.2), ("ma50", YEL, 1.5), ("ma200", "#FF7A7A", 1.8)]:
            if col in df.columns:
                ax1.plot(idx, df[col], color=col_c, linewidth=lw, label=col.upper())
        if "bb_upper" in df.columns:
            ax1.fill_between(idx, df["bb_lower"], df["bb_upper"], alpha=0.12, color=BLUE)

        vc = [GREEN if r["close"] >= r["open"] else RED for _, r in df.iterrows()]
        ax2.bar(idx, df["volume"], color=vc, alpha=0.6, width=0.7)

        ts = max(1, len(df) // 10)
        ax2.set_xticks(idx[::ts])
        ax2.set_xticklabels([d.strftime("%b %d") for d in df.index[::ts]],
                             rotation=30, ha="right", fontsize=8)
        ax1.set_title(f"{symbol} | Price Chart", fontsize=12, fontweight="bold")
        ax1.legend(fontsize=8)
        plt.tight_layout()
        self._save(fig, f"{symbol}_price.png")
        return fig

    # ── Indicator Chart (RSI / MACD / Stochastic) ────────────────────────────

    def indicator_chart(self, df: pd.DataFrame, symbol: str = "", n_days: int = 120):
        df  = df.tail(n_days).copy()
        idx = np.arange(len(df))

        fig = plt.figure(figsize=(14, 10))
        gs  = gridspec.GridSpec(3, 1, hspace=0.4)
        fig.patch.set_facecolor(BG)
        a1, a2, a3 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
        _style(a1, a2, a3)

        if "rsi" in df.columns:
            a1.plot(idx, df["rsi"], color=BLUE, linewidth=1.5, label="RSI")
            a1.axhline(70, color=RED,   linewidth=1, linestyle="--")
            a1.axhline(30, color=GREEN, linewidth=1, linestyle="--")
            a1.set_ylim(0, 100)
        a1.set_title(f"{symbol} — RSI(14)", fontweight="bold", fontsize=10)
        a1.legend(fontsize=8)

        if "macd" in df.columns:
            a2.plot(idx, df["macd"],        color=BLUE, linewidth=1.5, label="MACD")
            a2.plot(idx, df["macd_signal"], color=YEL,  linewidth=1.2, label="Signal")
            hist_colors = [GREEN if v >= 0 else RED for v in df["macd_hist"]]
            a2.bar(idx, df["macd_hist"], color=hist_colors, alpha=0.6, width=0.7)
            a2.axhline(0, color=GRID)
        a2.set_title("MACD", fontweight="bold", fontsize=10)
        a2.legend(fontsize=8)

        if "stoch_k" in df.columns:
            a3.plot(idx, df["stoch_k"], color=BLUE, linewidth=1.5, label="%K")
            a3.plot(idx, df["stoch_d"], color=YEL,  linewidth=1.2, label="%D")
            a3.axhline(80, color=RED,   linewidth=1, linestyle="--")
            a3.axhline(20, color=GREEN, linewidth=1, linestyle="--")
            a3.set_ylim(0, 100)
        a3.set_title("Stochastic", fontweight="bold", fontsize=10)
        a3.legend(fontsize=8)

        plt.tight_layout()
        self._save(fig, f"{symbol}_indicators.png")
        return fig

    # ── Prediction Chart ──────────────────────────────────────────────────────

    def prediction_chart(self, df: pd.DataFrame, pred_result, n_history: int = 60):
        hist   = df["close"].tail(n_history)
        hor    = pred_result.horizon_days
        future = pd.bdate_range(hist.index[-1], periods=hor + 1)[1:]

        fig, ax = plt.subplots(figsize=(14, 6))
        fig.patch.set_facecolor(BG)
        _style(ax)

        ax.plot(hist.index, hist.values, color=BLUE, linewidth=2, label="Historical")

        pv = np.linspace(float(hist.iloc[-1]), pred_result.predicted_price, hor + 1)[1:]
        lv = np.linspace(float(hist.iloc[-1]), pred_result.price_low,       hor + 1)[1:]
        hv = np.linspace(float(hist.iloc[-1]), pred_result.price_high,      hor + 1)[1:]

        ax.plot(future, pv, color=YEL, linewidth=2.5, linestyle="--",
                label=f"Predicted (+{hor}d)")
        ax.fill_between(future, lv, hv, alpha=0.25, color=YEL, label="95% CI")

        tc = GREEN if pred_result.trend == "Bullish" else (
             RED   if pred_result.trend == "Bearish" else YEL)
        ax.annotate(
            f"  {pred_result.trend}\n  {pred_result.predicted_price:.2f}",
            xy=(future[-1], pred_result.predicted_price),
            fontsize=11, fontweight="bold", color=tc)

        ax.set_title(
            f"{pred_result.symbol} | Prediction — {hor}d "
            f"| Confidence: {pred_result.confidence:.0f}%",
            fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        plt.tight_layout()
        self._save(fig, f"{pred_result.symbol}_prediction.png")
        return fig

    # ── Full Dashboard ────────────────────────────────────────────────────────

    def full_dashboard(self, df: pd.DataFrame, symbol: str,
                       pred_result, signals=None, backtest_result=None):
        n   = min(120, len(df))
        df  = df.tail(n).copy()
        idx = np.arange(len(df))

        fig = plt.figure(figsize=(20, 20))
        gs  = gridspec.GridSpec(5, 2, hspace=0.5, wspace=0.35)
        fig.patch.set_facecolor(BG)

        ax_p    = fig.add_subplot(gs[0, :])
        ax_rsi  = fig.add_subplot(gs[1, 0])
        ax_macd = fig.add_subplot(gs[1, 1])
        ax_st   = fig.add_subplot(gs[2, 0])
        ax_obv  = fig.add_subplot(gs[2, 1])
        ax_pr   = fig.add_subplot(gs[3, :])
        ax_info = fig.add_subplot(gs[4, :])
        _style(ax_p, ax_rsi, ax_macd, ax_st, ax_obv, ax_pr, ax_info)

        # Candlestick
        for i, (_, r) in enumerate(df.iterrows()):
            c = GREEN if r["close"] >= r["open"] else RED
            ax_p.bar(i, r["close"] - r["open"], bottom=r["open"], color=c, width=0.7, alpha=0.9)
            ax_p.plot([i, i], [r["low"], r["high"]], color=c, linewidth=0.7)
        for col, col_c, lw in [("ma20", BLUE, 1.2), ("ma50", YEL, 1.5)]:
            if col in df.columns:
                ax_p.plot(idx, df[col], color=col_c, linewidth=lw, label=col.upper())
        if "bb_upper" in df.columns:
            ax_p.fill_between(idx, df["bb_lower"], df["bb_upper"], alpha=0.1, color=BLUE)
        ax_p.set_title(f"{symbol} | AI Prediction Dashboard",
                        fontsize=14, fontweight="bold")
        ax_p.legend(fontsize=8, loc="upper left")
        ts = max(1, len(df) // 10)
        ax_p.set_xticks(idx[::ts])
        ax_p.set_xticklabels([d.strftime("%b %d") for d in df.index[::ts]],
                              rotation=30, ha="right", fontsize=7)

        # RSI
        if "rsi" in df.columns:
            ax_rsi.plot(idx, df["rsi"], color=BLUE, linewidth=1.5)
            ax_rsi.axhline(70, color=RED,   linewidth=0.8, linestyle="--")
            ax_rsi.axhline(30, color=GREEN, linewidth=0.8, linestyle="--")
            ax_rsi.set_ylim(0, 100)
        ax_rsi.set_title("RSI(14)", fontsize=9, fontweight="bold")

        # MACD
        if "macd" in df.columns:
            ax_macd.plot(idx, df["macd"],        color=BLUE, linewidth=1.5, label="MACD")
            ax_macd.plot(idx, df["macd_signal"], color=YEL,  linewidth=1.2, label="Signal")
            ax_macd.bar(idx, df["macd_hist"],
                        color=[GREEN if v >= 0 else RED for v in df["macd_hist"]],
                        alpha=0.6, width=0.7)
            ax_macd.axhline(0, color=GRID, linewidth=0.6)
        ax_macd.set_title("MACD", fontsize=9, fontweight="bold")
        ax_macd.legend(fontsize=7)

        # Stochastic
        if "stoch_k" in df.columns:
            ax_st.plot(idx, df["stoch_k"], color=BLUE, linewidth=1.5, label="%K")
            ax_st.plot(idx, df["stoch_d"], color=YEL,  linewidth=1.2, label="%D")
            ax_st.axhline(80, color=RED,   linewidth=0.8, linestyle="--")
            ax_st.axhline(20, color=GREEN, linewidth=0.8, linestyle="--")
            ax_st.set_ylim(0, 100)
        ax_st.set_title("Stochastic", fontsize=9, fontweight="bold")
        ax_st.legend(fontsize=7)

        # OBV
        if "obv" in df.columns:
            ax_obv.plot(idx, df["obv"], color=BLUE, linewidth=1.5, label="OBV")
        ax_obv.set_title("OBV", fontsize=9, fontweight="bold")
        ax_obv.legend(fontsize=7)

        # Prediction
        h      = df["close"]
        hor    = pred_result.horizon_days
        future = pd.bdate_range(h.index[-1], periods=hor + 1)[1:]
        ax_pr.plot(h.index, h.values, color=BLUE, linewidth=2, label="Historical")
        pv = np.linspace(float(h.iloc[-1]), pred_result.predicted_price, hor + 1)[1:]
        lv = np.linspace(float(h.iloc[-1]), pred_result.price_low,       hor + 1)[1:]
        hv = np.linspace(float(h.iloc[-1]), pred_result.price_high,      hor + 1)[1:]
        ax_pr.plot(future, pv, color=YEL, linewidth=2.5, linestyle="--", label="Predicted")
        ax_pr.fill_between(future, lv, hv, alpha=0.25, color=YEL, label="95% CI")
        tc = GREEN if pred_result.trend == "Bullish" else (
             RED   if pred_result.trend == "Bearish" else YEL)
        ax_pr.annotate(
            f"  {pred_result.trend} {pred_result.predicted_price:.2f}",
            xy=(future[-1], pred_result.predicted_price),
            fontsize=11, fontweight="bold", color=tc)
        ax_pr.set_title(
            f"Prediction +{hor}d | Confidence {pred_result.confidence:.0f}%"
            f" | {pred_result.predicted_return:+.2f}%",
            fontsize=10, fontweight="bold")
        ax_pr.legend(fontsize=9)

        # Info bar
        ax_info.axis("off")
        s    = pred_result.sentiment
        info = (f"Symbol: {symbol}  |  Current: {pred_result.current_price:.2f}  |  "
                f"Predicted: {pred_result.predicted_price:.2f}  |  "
                f"Range: {pred_result.price_low:.2f}–{pred_result.price_high:.2f}  |  "
                f"Trend: {pred_result.trend}  |  Confidence: {pred_result.confidence:.0f}%  |  "
                f"Sentiment: {s.get('label','N/A')} ({s.get('score',0):.3f})  |  "
                f"Articles: {s.get('article_count', 0)}")
        ax_info.text(0.5, 0.5, info, transform=ax_info.transAxes,
                     fontsize=10, ha="center", va="center", color=TEXT,
                     fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.5", facecolor=CARD,
                               edgecolor=GRID, alpha=0.8))

        self._save(fig, f"{symbol}_dashboard.png")
        return fig