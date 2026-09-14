import asyncio
from decimal import Decimal, getcontext
import json
import logging
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional
import pandas as pd
import websockets

from .models import MarketRewardConfig, OrderbookLevel, OrderbookSnapshot, OrderSide
from .reconciler import BookReconciler
from .watchdog import SystemWatchdog

getcontext().prec = 28
logger = logging.getLogger(__name__)


class OrderbookRAMStore:
    """Maintains real-time L2 orderbooks in memory from WS streams."""

    def __init__(self):
        # token_id -> { "bids": {price: size}, "asks": {price: size}, "updated_ns": int }
        self.raw_books: Dict[str, Dict[str, Any]] = {}

    def apply_snapshot(self, token_id: str, bids: List[Dict[str, Any]], asks: List[Dict[str, Any]], timestamp_ns: int) -> None:
        bid_map = {Decimal(str(b["price"])): Decimal(str(b["size"])) for b in bids if Decimal(str(b["size"])) > 0}
        ask_map = {Decimal(str(a["price"])): Decimal(str(a["size"])) for a in asks if Decimal(str(a["size"])) > 0}
        self.raw_books[token_id] = {
            "bids": bid_map,
            "asks": ask_map,
            "updated_ns": timestamp_ns,
            "valid_from_ns": timestamp_ns,
        }

    def apply_delta(self, token_id: str, side: str, price: Decimal, size: Decimal, timestamp_ns: int) -> None:
        if token_id not in self.raw_books:
            self.raw_books[token_id] = {"bids": {}, "asks": {}, "updated_ns": timestamp_ns, "valid_from_ns": timestamp_ns}

        book = self.raw_books[token_id]
        target_map = book["bids"] if side.lower() == "buy" or side.lower() == "bid" else book["asks"]
        if size <= Decimal("0.0"):
            target_map.pop(price, None)
        else:
            target_map[price] = size
        book["updated_ns"] = timestamp_ns

    def get_snapshot(self, condition_id: str, token_id: str) -> OrderbookSnapshot:
        book = self.raw_books.get(token_id)
        now_ns = time.time_ns()
        if not book:
            return OrderbookSnapshot(
                condition_id=condition_id,
                asset_id=token_id,
                bids=[],
                asks=[],
                timestamp_ns=now_ns,
                valid_from_ns=now_ns,
            )

        bids = [
            OrderbookLevel(price=p, size=s)
            for p, s in sorted(book["bids"].items(), key=lambda x: x[0], reverse=True)
        ]
        asks = [
            OrderbookLevel(price=p, size=s)
            for p, s in sorted(book["asks"].items(), key=lambda x: x[0])
        ]

        return OrderbookSnapshot(
            condition_id=condition_id,
            asset_id=token_id,
            bids=bids,
            asks=asks,
            timestamp_ns=book["updated_ns"],
            valid_from_ns=book.get("valid_from_ns", book["updated_ns"]),
        )


class MarketDataCollector:
    """Manages WebSocket connection, PING/PONG heartbeats, and disk-safe logging."""

    def __init__(
        self,
        ws_endpoint: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market",
        storage_path: str = r"D:\flipoly-research\lp-rewards",
        ping_interval_sec: float = 10.0,
        pong_timeout_sec: float = 20.0,
        watchdog: Optional[SystemWatchdog] = None,
        reconciler: Optional[BookReconciler] = None,
    ):
        self.ws_endpoint = ws_endpoint
        self.storage_path = Path(storage_path)
        self.ping_interval_sec = ping_interval_sec
        self.pong_timeout_sec = pong_timeout_sec
        self.watchdog = watchdog or SystemWatchdog(storage_path)
        self.reconciler = reconciler or BookReconciler()

        self.ram_store = OrderbookRAMStore()
        self.running = False
        self.last_pong_time: float = time.time()
        self.active_markets: Dict[str, MarketRewardConfig] = {}
        self.trade_buffer: List[Dict[str, Any]] = []

    def set_active_markets(self, markets: List[MarketRewardConfig]) -> None:
        self.active_markets = {m.condition_id: m for m in markets}

    async def run(self) -> None:
        self.running = True
        while self.running:
            # Check disk watchdog before connecting
            _, status = self.watchdog.check_disk_space()
            if status == "EMERGENCY_HALT":
                logger.critical("Collector halted due to low disk space on target storage drive.")
                self.running = False
                break

            try:
                async with websockets.connect(self.ws_endpoint, ping_interval=None) as ws:
                    self.last_pong_time = time.time()
                    # Subscribe to active tokens
                    all_tokens = []
                    for m in self.active_markets.values():
                        all_tokens.extend([m.yes_token_id, m.no_token_id])

                    if all_tokens:
                        sub_msg = {
                            "type": "market",
                            "assets_ids": all_tokens,
                        }
                        await ws.send(json.dumps(sub_msg))

                    # Concurrently run message handler and ping loop
                    handler_task = asyncio.create_task(self._message_loop(ws))
                    ping_task = asyncio.create_task(self._ping_loop(ws))

                    done, pending = await asyncio.wait(
                        [handler_task, ping_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in pending:
                        task.cancel()

            except Exception as e:
                logger.warning(f"WS connection error: {e}. Reconnecting in 3s...")
                await asyncio.sleep(3.0)

    async def _ping_loop(self, ws: websockets.WebSocketClientProtocol) -> None:
        while self.running:
            await asyncio.sleep(self.ping_interval_sec)
            now = time.time()
            if now - self.last_pong_time > self.pong_timeout_sec:
                logger.warning("PONG timeout exceeded. Forcing reconnect...")
                await ws.close()
                break
            try:
                pong_waiter = await ws.ping()
                await asyncio.wait_for(pong_waiter, timeout=self.ping_interval_sec)
                self.last_pong_time = time.time()
            except Exception as e:
                logger.warning(f"Ping failed: {e}")
                await ws.close()
                break

    async def _message_loop(self, ws: websockets.WebSocketClientProtocol) -> None:
        async for raw_msg in ws:
            now_ns = time.time_ns()
            try:
                msg = json.loads(raw_msg)
                event_type = msg.get("event_type")

                if event_type == "book":
                    asset_id = msg.get("asset_id")
                    bids = msg.get("bids", [])
                    asks = msg.get("asks", [])
                    self.ram_store.apply_snapshot(asset_id, bids, asks, now_ns)

                elif event_type == "price_change":
                    asset_id = msg.get("asset_id")
                    side = msg.get("side", "")
                    price = Decimal(str(msg.get("price", "0")))
                    size = Decimal(str(msg.get("size", "0")))
                    self.ram_store.apply_delta(asset_id, side, price, size, now_ns)

                elif event_type == "last_trade_price":
                    asset_id = msg.get("asset_id")
                    price = Decimal(str(msg.get("price", "0")))
                    size = Decimal(str(msg.get("size", "0")))
                    side = OrderSide.BUY if msg.get("side", "").upper() == "BUY" else OrderSide.SELL
                    self.trade_buffer.append({
                        "timestamp_ns": now_ns,
                        "asset_id": asset_id,
                        "price": float(price),
                        "size": float(size),
                        "side": side.value,
                    })

            except Exception as e:
                logger.debug(f"Error parsing WS message: {e}")

    def flush_trades_to_disk(self, condition_id: str, date_str: str) -> None:
        """Flush trade buffer to Parquet on target storage path."""
        if not self.trade_buffer:
            return

        free_gb, status = self.watchdog.check_disk_space()
        if status == "EMERGENCY_HALT":
            logger.critical("Cannot flush trades: disk emergency halt threshold reached.")
            return

        target_dir = self.storage_path / "public_trades" / condition_id
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / f"{date_str}.parquet"

        df = pd.DataFrame(self.trade_buffer)
        df.to_parquet(file_path, compression="zstd", index=False)
        self.trade_buffer.clear()
