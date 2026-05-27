
import asyncio, json, logging, time
from datetime import datetime, timezone
from typing   import Dict, Set

import httpx
from fastapi import WebSocket, WebSocketDisconnect

logger   = logging.getLogger(__name__)
_YAHOO_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
             "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36")


async def _fetch_live_quote(symbol: str) -> dict | None:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=1d"
    try:
        async with httpx.AsyncClient(timeout=8) as c:
            r    = await c.get(url, headers={"User-Agent": _YAHOO_UA, "Accept": "application/json"})
            data = r.json()
        meta     = data["chart"]["result"][0]["meta"]
        price    = float(meta["regularMarketPrice"])
        prev     = float(meta.get("chartPreviousClose") or meta.get("previousClose") or price)
        change   = round(price - prev, 2)
        change_p = round((change / prev) * 100, 2) if prev else 0.0
        return {
            "price":     round(price, 2),
            "change":    change,
            "changePct": change_p,
            "high":      round(float(meta.get("regularMarketDayHigh", price)), 2),
            "low":       round(float(meta.get("regularMarketDayLow",  price)), 2),
            "volume":    int(meta.get("regularMarketVolume", 0)),
            "currency":  meta.get("currency", "USD"),
        }
    except Exception as e:
        logger.debug("[WS] quote failed %s: %s", symbol, e)
        return None


class ConnectionManager:
    def __init__(self):
        self._connections: Dict[WebSocket, Set[str]] = {}

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._connections[ws] = set()
        logger.info("[WS] Client connected. Total: %d", len(self._connections))

    def disconnect(self, ws: WebSocket):
        self._connections.pop(ws, None)
        logger.info("[WS] Client disconnected. Total: %d", len(self._connections))

    async def send(self, ws: WebSocket, payload: dict):
        try:
            await ws.send_text(json.dumps(payload))
        except Exception:
            self.disconnect(ws)

    async def broadcast(self, symbol: str, payload: dict):
        for ws, syms in list(self._connections.items()):
            if symbol in syms:
                await self.send(ws, payload)

    def subscribe(self, ws: WebSocket, symbol: str):
        if ws in self._connections:
            self._connections[ws].add(symbol.upper())

    def unsubscribe(self, ws: WebSocket, symbol: str):
        if ws in self._connections:
            self._connections[ws].discard(symbol.upper())

    @property
    def active_symbols(self) -> Set[str]:
        s: Set[str] = set()
        for syms in self._connections.values():
            s.update(syms)
        return s


manager = ConnectionManager()


class StreamScheduler:
    def __init__(self, interval: int = 15):
        self.interval = interval
        self._tasks: Dict[str, asyncio.Task] = {}

    def ensure_running(self, symbol: str, regime_detector=None):
        if symbol not in self._tasks or self._tasks[symbol].done():
            self._tasks[symbol] = asyncio.create_task(
                self._stream(symbol, regime_detector))

    async def _stream(self, symbol: str, regime_detector=None):
        logger.info("[WS] Stream started: %s", symbol)
        while True:
            try:
                if symbol not in manager.active_symbols:
                    logger.info("[WS] No subscribers for %s, stopping", symbol)
                    break
                quote = await _fetch_live_quote(symbol)
                if quote:
                    payload = {
                        "type":   "price_update",
                        "symbol": symbol,
                        **quote,
                        "signal": "HOLD",
                        "ts":     datetime.now(timezone.utc).isoformat(),
                    }
                    if regime_detector:
                        try:
                            from data_loader import DataLoader
                            df = DataLoader().get(symbol)
                            r  = regime_detector.detect(df)
                            payload["regime"]     = r.current_regime
                            payload["regimeProb"] = round(r.probability, 2)
                        except Exception:
                            payload["regime"] = "Unknown"
                    await manager.broadcast(symbol, payload)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("[WS] stream error %s: %s", symbol, e)
            await asyncio.sleep(self.interval)
        logger.info("[WS] Stream stopped: %s", symbol)


scheduler = StreamScheduler(interval=15)


async def ws_endpoint(websocket: WebSocket, regime_detector=None):
    await manager.connect(websocket)
    try:
        await manager.send(websocket, {
            "type":    "connected",
            "message": "StockAnalyzer AI stream ready",
            "ts":      datetime.now(timezone.utc).isoformat(),
        })
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await manager.send(websocket, {"type": "error", "message": "Invalid JSON"})
                continue

            action = msg.get("action", "").lower()
            symbol = msg.get("symbol", "").upper().strip()

            if action == "subscribe" and symbol:
                manager.subscribe(websocket, symbol)
                scheduler.ensure_running(symbol, regime_detector)
                await manager.send(websocket, {
                    "type": "subscribed", "symbol": symbol,
                    "message": f"Subscribed to {symbol}. Updates every 15s.",
                })
            elif action == "unsubscribe" and symbol:
                manager.unsubscribe(websocket, symbol)
                await manager.send(websocket, {"type": "unsubscribed", "symbol": symbol})
            elif action == "ping":
                await manager.send(websocket, {
                    "type": "pong", "ts": datetime.now(timezone.utc).isoformat()})
            else:
                await manager.send(websocket, {
                    "type": "error",
                    "message": f"Unknown action '{action}'. Use subscribe/unsubscribe/ping",
                })
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error("[WS] Unexpected: %s", e)
        manager.disconnect(websocket)