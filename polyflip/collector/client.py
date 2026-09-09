import asyncio
from dataclasses import dataclass
import httpx
from typing import List, Dict, Any, TypedDict
from datetime import datetime, timezone
import structlog
import json
import time
import pandas as pd
from polyflip.constants import HTTP_TIMEOUT_SEC, VOLUME_WINDOW_MIN
from polyflip.collector.orderbook_depth import OrderbookContract, validate_and_normalize_orderbook

logger = structlog.get_logger(__name__)


class MarketPricesResult(TypedDict, total=False):
    current_yes_price: float | None
    current_no_price: float | None
    current_spread: float | None
    best_bid: float | None
    best_ask: float | None
    best_bid_no: float | None
    best_ask_no: float | None
    tick_size: float | None
    min_order_size: float | None
    error: str
    bids: list[dict[str, float]]
    asks: list[dict[str, float]]
    yes_orderbook: OrderbookContract | None
    no_orderbook: OrderbookContract | None

@dataclass(frozen=True)
class StrikeProvenance:
    strike_value: float | None
    strike_source: str
    strike_effective_at: datetime | None
    strike_received_at: datetime | None
    market_start_at: datetime | None = None
    market_end_at: datetime | None = None
    settlement_price_source: str = "CHAINLINK_ORACLE"

    def __float__(self) -> float:
        if self.strike_value is None:
            raise TypeError("Cannot cast unresolved StrikeProvenance to float")
        return float(self.strike_value)


@dataclass(frozen=True)
class VolumeResult:
    volume: float | None
    status: str  # "VALID", "HTTP_ERROR", "AUTH_REQUIRED", "PARSE_ERROR", "UNAVAILABLE", "VALID_ZERO"
    timestamp: datetime
    source: str = "CLOB_TRADES"

    def __float__(self) -> float:
        if self.volume is None:
            raise TypeError("Cannot cast unavailable VolumeResult to float")
        return float(self.volume)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (int, float)):
            if self.volume is None or self.status not in ("VALID", "VALID_ZERO"):
                return False
            return float(self.volume) == float(other)
        return super().__eq__(other)


def _canonical_strike_provenance(market: Dict[str, Any], event: Dict[str, Any]) -> StrikeProvenance:
    """Extract Polymarket opening/Chainlink strike with full provenance."""
    now = datetime.now(timezone.utc)
    candidates = [
        ("market.underlying_price", market.get("underlying_price")),
        ("market.underlyingPrice", market.get("underlyingPrice")),
        ("market.strike", market.get("strike")),
        ("market.strikePrice", market.get("strikePrice")),
        ("market.priceToBeat", market.get("priceToBeat")),
        ("market.openingPrice", market.get("openingPrice")),
        ("event.underlying_price", event.get("underlying_price")),
        ("event.underlyingPrice", event.get("underlyingPrice")),
        ("event.strike", event.get("strike")),
        ("event.strikePrice", event.get("strikePrice")),
        ("event.priceToBeat", event.get("priceToBeat")),
    ]
    effective_at = None
    start_str = market.get("startDate") or market.get("market_start") or event.get("startDate")
    if start_str:
        try:
            effective_at = pd.to_datetime(start_str, utc=True).to_pydatetime()
        except Exception:
            effective_at = None

    end_dt = None
    end_str = market.get("endDate") or market.get("market_end") or event.get("endDate")
    if end_str:
        try:
            end_dt = pd.to_datetime(end_str, utc=True).to_pydatetime()
        except Exception:
            end_dt = None

    settlement_src = str(
        market.get("resolutionSource")
        or market.get("oracle")
        or event.get("resolutionSource")
        or "CHAINLINK_ORACLE"
    )

    for src_name, candidate in candidates:
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            continue
        if value > 0.0 and value == value and value != float("inf"):
            return StrikeProvenance(
                strike_value=value,
                strike_source=src_name,
                strike_effective_at=effective_at or now,
                strike_received_at=now,
                market_start_at=effective_at,
                market_end_at=end_dt,
                settlement_price_source=settlement_src,
            )
    return StrikeProvenance(
        strike_value=None,
        strike_source="UNKNOWN",
        strike_effective_at=effective_at,
        strike_received_at=now,
        market_start_at=effective_at,
        market_end_at=end_dt,
        settlement_price_source=settlement_src,
    )


def _canonical_strike(market: Dict[str, Any], event: Dict[str, Any]) -> float | None:
    """Extract Polymarket's opening/Chainlink strike without Binance fallbacks."""
    return _canonical_strike_provenance(market, event).strike_value

class PolymarketClient:
    GAMMA_API = "https://gamma-api.polymarket.com"
    CLOB_API = "https://clob.polymarket.com"

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=HTTP_TIMEOUT_SEC)
        self._market_info_cache: dict[str, tuple[float, Dict[str, Any]]] = {}

    async def close(self):
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.client.aclose()

    async def get_clob_market_info(self, condition_id: str) -> Dict[str, Any] | None:
        """Return official CLOB V2 market metadata.

        The V2 endpoint returns market-level fields such as ``mos`` (minimum
        order size), ``mts`` (minimum tick size), and ``fd`` (fee curve).
        Fee metadata is cached briefly because it is market-level data rather
        than a per-tick quote. A failed lookup returns ``None`` and callers
        must use their explicit fallback cost model.
        """
        condition_id = str(condition_id or "").strip()
        if not condition_id:
            return None
        now = time.monotonic()
        cached = self._market_info_cache.get(condition_id)
        if cached and now - cached[0] < 600.0:
            return cached[1]
        try:
            response = await self.client.get(f"{self.CLOB_API}/clob-markets/{condition_id}")
            if response.status_code != 200:
                logger.debug(
                    "clob_market_info_unavailable",
                    condition_id=condition_id,
                    status=response.status_code,
                )
                return None
            payload = response.json()
            if not isinstance(payload, dict):
                return None
            self._market_info_cache[condition_id] = (now, payload)
            return payload
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            logger.debug(
                "clob_market_info_network_error",
                condition_id=condition_id,
                error=str(exc),
            )
            return None
        except Exception as exc:
            logger.debug(
                "clob_market_info_error",
                condition_id=condition_id,
                error=str(exc),
            )
            return None

    async def get_market_fee_schedule(self, condition_id: str) -> Dict[str, Any] | None:
        """Extract the CLOB V2 fee curve and minimum order size.

        CLOB V2 exposes ``fd={r, e, to}``: rate, price-curve exponent and
        taker-only flag. The legacy ``feeSchedule`` shape remains supported
        for old fixtures and transitional deployments.
        """
        info = await self.get_clob_market_info(condition_id)
        if not info:
            return None
        def _number(value: Any, default: float | None = None) -> float | None:
            try:
                parsed = float(value)
            except (TypeError, ValueError, OverflowError):
                return default
            return parsed if parsed == parsed and parsed not in (float("inf"), float("-inf")) else default

        def _rate(value: Any) -> float | None:
            parsed = _number(value)
            if parsed is None:
                return None
            if parsed > 1.0:
                parsed /= 10000.0
            return parsed if 0.0 <= parsed <= 1.0 else None

        def _bool(value: Any, default: bool = False) -> bool:
            if isinstance(value, str):
                return value.strip().lower() not in {"false", "0", "no", "off", "disabled"}
            return default if value is None else bool(value)

        minimum_order_shares = _number(info.get("mos"))
        if "fd" in info:
            fee_details = info.get("fd")
            if fee_details is None:
                return {
                    "fee_rate": 0.0,
                    "fee_exponent": 1.0,
                    "maker_fee_rate": 0.0,
                    "fees_enabled": False,
                    "taker_only": False,
                    "min_order_shares": minimum_order_shares,
                    "source": "CLOB_FD_DISABLED",
                }
            if not isinstance(fee_details, dict):
                return None
            fee_rate = _rate(fee_details.get("r"))
            if fee_rate is None:
                return None
            fee_exponent = _number(fee_details.get("e"), 1.0)
            if fee_exponent is None or fee_exponent < 0.0:
                return None
            return {
                "fee_rate": fee_rate,
                "fee_exponent": fee_exponent,
                "maker_fee_rate": 0.0,
                "fees_enabled": fee_rate > 0.0,
                "taker_only": _bool(fee_details.get("to"), False),
                "min_order_shares": minimum_order_shares,
                "source": "CLOB_FD",
            }

        schedule = info.get("feeSchedule") or info.get("fee_schedule")
        if not isinstance(schedule, dict):
            return None

        raw_rate = (
            schedule.get("r")
            if schedule.get("r") is not None
            else schedule.get("feeRate", schedule.get("fee_rate"))
        )
        try:
            fee_rate = float(raw_rate)
        except (TypeError, ValueError, OverflowError):
            return None
        if fee_rate > 1.0:
            # Some SDK payloads expose the rate in basis points.
            fee_rate /= 10000.0
        if not 0.0 <= fee_rate <= 1.0:
            return None
        maker_rate = schedule.get("makerFeeRate", schedule.get("maker_fee_rate", 0.0))
        try:
            maker_rate = float(maker_rate)
        except (TypeError, ValueError, OverflowError):
            maker_rate = 0.0
        if maker_rate > 1.0:
            maker_rate /= 10000.0
        raw_fees_enabled = info.get(
            "feesEnabled", info.get("fees_enabled", True)
        )
        if isinstance(raw_fees_enabled, str):
            fees_enabled = raw_fees_enabled.strip().lower() not in {
                "false", "0", "no", "off", "disabled",
            }
        else:
            fees_enabled = bool(raw_fees_enabled)
        fee_exponent = _number(schedule.get("e"), 1.0)
        if fee_exponent is None or fee_exponent < 0.0:
            return None
        return {
            "fee_rate": fee_rate if fees_enabled else 0.0,
            "fee_exponent": fee_exponent,
            "maker_fee_rate": max(0.0, maker_rate),
            "fees_enabled": fees_enabled,
            "taker_only": _bool(schedule.get("to"), False),
            "min_order_shares": minimum_order_shares,
            "source": "CLOB_FEE_SCHEDULE" if fees_enabled else "CLOB_FEE_SCHEDULE_DISABLED",
        }

    async def get_active_15m_markets(self, assets: List[str]) -> List[Dict[str, Any]]:
        """
        Получает активные 15-минутные рынки (Up/Down) для заданных активов.
        """
        markets = []
        
        # Маппинг тикеров в полные названия для поиска в тегах и заголовках
        asset_mapping = {
            "BTC": ["BITCOIN", "BTC"],
            "ETH": ["ETHEREUM", "ETH"],
            "SOL": ["SOLANA", "SOL"],
            "XRP": ["XRP"],
            "DOGE": ["DOGECOIN", "DOGE"],
            "BNB": ["BNB"],
            "HYPE": ["HYPERLIQUID", "HYPE"]
        }
        
        try:
            # Запрашиваем конкретно 15-минутные рынки через tag_slug=15m
            response = await self.client.get(
                f"{self.GAMMA_API}/events",
                params={"active": "true", "closed": "false", "tag_slug": "15m", "limit": 100}
            )
            response.raise_for_status()
            events = response.json()

            for event in events:
                title = event.get("title", "").upper()
                tags = [t.get("label", "").upper() for t in event.get("tags", [])]
                
                # Ищем, какому активу из наших настроек принадлежит этот рынок
                matched_asset = None
                for a in assets:
                    search_terms = asset_mapping.get(a.upper(), [a.upper()])
                    if any(term in title or term in tags for term in search_terms):
                        matched_asset = a.upper()
                        break
                        
                if not matched_asset:
                    continue

                for market in event.get("markets", []):
                    if not market.get("active") or market.get("closed"):
                        continue
                        
                    # Нас интересуют бинарные рынки Up/Down (или Yes/No на всякий случай)
                    outcomes = market.get("outcomes", [])
                    if isinstance(outcomes, str):
                        try:
                            outcomes = json.loads(outcomes)
                        except Exception:
                            pass
                            
                    if not isinstance(outcomes, list):
                        logger.debug("skipping_market_invalid_outcomes_type", outcomes=outcomes, market_id=market.get("id"))
                        continue
                        
                    outcomes_lower = [str(o).lower() for o in outcomes]
                    if outcomes_lower != ["up", "down"] and outcomes_lower != ["yes", "no"]:
                        logger.debug("skipping_market_unknown_outcomes", outcomes=outcomes, market_id=market.get("id"))
                        continue

                    clob_token_ids = market.get("clobTokenIds", [])
                    if type(clob_token_ids) is str:
                        clob_token_ids = json.loads(clob_token_ids)
                        
                    if not clob_token_ids or len(clob_token_ids) < 2:
                        continue
                        
                    yes_token_id = clob_token_ids[0] # Up или Yes
                    no_token_id = clob_token_ids[1]  # Down или No

                    markets.append({
                        "market_id": market.get("id"),
                        "condition_id": market.get("conditionId") or market.get("condition_id"),
                        "yes_token_id": yes_token_id,
                        "no_token_id": no_token_id,
                        "question": market.get("question"),
                        "asset": matched_asset,
                        "end_date_iso": market.get("endDate"),
                        "underlying_price": _canonical_strike(market, event),
                        "strike_provenance": _canonical_strike_provenance(market, event),
                    })
                        
        except Exception as e:
            logger.error("error_fetching_gamma_markets", error=str(e))
            
        return markets

    async def get_single_orderbook(
        self,
        token_id: str,
        outcome_side: str = "YES",
        market_id: str = "",
        depth_limit: int | None = None,
    ) -> OrderbookContract:
        """
        Fetches and normalizes raw orderbook from CLOB API for a single token.
        Ensures strict contract verification and explicit quality status (Points 4, 6, 7).
        """
        now = datetime.now(timezone.utc)
        received_at_dt = now
        try:
            response = None
            for attempt in range(3):
                try:
                    response = await self.client.get(
                        f"{self.CLOB_API}/book", params={"token_id": token_id}
                    )
                    received_at_dt = datetime.now(timezone.utc)
                    now = received_at_dt
                except (httpx.TimeoutException, httpx.NetworkError) as exc:
                    if attempt == 2:
                        raise
                    delay = 0.25 * (2 ** attempt)
                    await asyncio.sleep(delay)
                    continue
                if response.status_code in {408, 429, 500, 502, 503, 504} and attempt < 2:
                    delay = 0.25 * (2 ** attempt)
                    await asyncio.sleep(delay)
                    continue
                break

            if response is None or response.status_code != 200:
                status = "API_ERROR" if response else "NO_RESPONSE"
                err_msg = f"HTTP {response.status_code}" if response else "No response from API"
                return OrderbookContract(
                    market_id=str(market_id),
                    token_id=str(token_id),
                    outcome_side=outcome_side,
                    event_at=now,
                    received_at=received_at_dt,
                    bids=[],
                    asks=[],
                    quality_status=status,
                    quality_notes=err_msg,
                )

            book = response.json()
            bids = book.get("bids", [])
            asks = book.get("asks", [])
            seq_id = book.get("sequence_id") or book.get("hash")
            
            raw_ts = book.get("timestamp")
            event_at_dt = None
            if raw_ts:
                try:
                    ts_str = str(raw_ts)
                    if ts_str.replace('.', '', 1).isdigit():
                        ts = float(ts_str)
                        if ts > 1e11:
                            ts /= 1000.0
                        event_at_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                    else:
                        event_at_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                except Exception:
                    event_at_dt = received_at_dt
            else:
                event_at_dt = received_at_dt

            return validate_and_normalize_orderbook(
                raw_bids=bids,
                raw_asks=asks,
                market_id=str(market_id),
                token_id=str(token_id),
                outcome_side=outcome_side,
                event_at=event_at_dt,
                received_at=received_at_dt,
                sequence_id=seq_id,
                depth_limit=depth_limit,
                source="CLOB",
            )
        except Exception as exc:
            return OrderbookContract(
                market_id=str(market_id),
                token_id=str(token_id),
                outcome_side=outcome_side,
                event_at=now,
                received_at=now,
                bids=[],
                asks=[],
                quality_status="EXCEPTION",
                quality_notes=str(exc),
            )

    async def get_both_orderbooks(
        self,
        market_id: str,
        yes_token_id: str,
        no_token_id: str,
        depth_limit: int | None = None,
    ) -> tuple[OrderbookContract, OrderbookContract]:
        """
        Fetches real orderbooks for both YES and NO tokens concurrently.
        Prevents reconstructing NO from YES (Point 6). Errors on one side are NOT masked.
        """
        yes_contract, no_contract = await asyncio.gather(
            self.get_single_orderbook(
                yes_token_id, outcome_side="YES", market_id=market_id, depth_limit=depth_limit
            ),
            self.get_single_orderbook(
                no_token_id, outcome_side="NO", market_id=market_id, depth_limit=depth_limit
            ),
        )
        return yes_contract, no_contract

    async def get_market_prices(
        self,
        yes_token_id: str,
        no_token_id: str | None = None,
        market_id: str = "",
    ) -> MarketPricesResult:
        """
        Получает стакан (orderbook) из CLOB API.
        Если передан no_token_id, запрашивает РЕАЛЬНЫЕ стаканы YES и NO без реконструкции.
        """
        if no_token_id:
            yes_book, no_book = await self.get_both_orderbooks(market_id, yes_token_id, no_token_id)
            res: MarketPricesResult = {
                "yes_orderbook": yes_book,
                "no_orderbook": no_book,
            }

            if yes_book.quality_status in ("VALID", "UNORDERED_LEVELS_NORMALIZED", "EMPTY_BOOK"):
                res["bids"] = yes_book.bids
                res["asks"] = yes_book.asks
                res["best_bid"] = yes_book.best_bid_price
                res["best_ask"] = yes_book.best_ask_price
                res["current_yes_price"] = None
                res["current_spread"] = None
                if yes_book.best_bid_price is not None and yes_book.best_ask_price is not None:
                    res["current_yes_price"] = (yes_book.best_bid_price + yes_book.best_ask_price) / 2.0
                    res["current_spread"] = yes_book.best_ask_price - yes_book.best_bid_price
            else:
                res["error"] = f"YES orderbook error: {yes_book.quality_status}"

            # Real NO book metrics: NOT 1.0 - mid_price! (Point 6)
            if no_book.quality_status in ("VALID", "UNORDERED_LEVELS_NORMALIZED"):
                res["best_bid_no"] = no_book.best_bid_price
                res["best_ask_no"] = no_book.best_ask_price
                if no_book.best_bid_price is not None and no_book.best_ask_price is not None:
                    res["current_no_price"] = (no_book.best_bid_price + no_book.best_ask_price) / 2.0
            else:
                res["current_no_price"] = None

            return res

        # Single-token fallback
        yes_book = await self.get_single_orderbook(yes_token_id, outcome_side="YES", market_id=market_id)
        if yes_book.quality_status not in ("VALID", "UNORDERED_LEVELS_NORMALIZED"):
            return {"error": f"API Error: {yes_book.quality_status}"}

        best_bid = yes_book.best_bid_price
        best_ask = yes_book.best_ask_price
        mid_price = (best_bid + best_ask) / 2.0 if (best_bid is not None and best_ask is not None) else None
        spread = (best_ask - best_bid) if (best_bid is not None and best_ask is not None) else None

        return {
            "current_yes_price": mid_price,
            "current_no_price": (1.0 - mid_price) if mid_price is not None else None,
            "current_spread": spread,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "bids": yes_book.bids,
            "asks": yes_book.asks,
            "yes_orderbook": yes_book,
            "no_orderbook": None,
        }

    async def get_recent_trades_volume(self, yes_token_id: str, minutes: int = VOLUME_WINDOW_MIN) -> VolumeResult:
        """
        Получает историю сделок из CLOB API и суммирует объем за последние N минут.
        Возвращает VolumeResult с явным статусом (VALID, AUTH_REQUIRED, HTTP_ERROR, UNAVAILABLE).
        """
        now = datetime.now(timezone.utc)
        try:
            # Пытаемся получить последние сделки по токену
            response = await self.client.get(f"{self.CLOB_API}/trades", params={"token_id": yes_token_id})
            if response.status_code == 401:
                return VolumeResult(volume=None, status="AUTH_REQUIRED", timestamp=now)
            if response.status_code != 200:
                logger.warning("clob_trades_api_error", token_id=yes_token_id, status=response.status_code)
                return VolumeResult(volume=None, status="HTTP_ERROR", timestamp=now)
                
            trades = response.json()
            if not isinstance(trades, list):
                # Иногда API отдает словарь с ключом data или history
                trades = trades.get("data", []) or trades.get("trades", []) or []
            if not trades:
                return VolumeResult(volume=0.0, status="VALID_ZERO", timestamp=now)

            total_volume = 0.0
            
            for t in trades:
                # Парсим время сделки. Поддерживает как ISO8601, так и unix epoch (сек/мс)
                ts_val = t.get("timestamp") or t.get("created_at")
                if not ts_val:
                    continue
                
                try:
                    if isinstance(ts_val, (int, float)) or (isinstance(ts_val, str) and ts_val.isdigit()):
                        sec = float(ts_val)
                        if sec > 1e11:  # milliseconds
                            sec /= 1000.0
                        trade_time = datetime.fromtimestamp(sec, tz=timezone.utc)
                    else:
                        trade_time = datetime.fromisoformat(str(ts_val).replace("Z", "+00:00"))
                except Exception:
                    continue

                if trade_time.tzinfo is None:
                    trade_time = trade_time.replace(tzinfo=timezone.utc)
                delta_minutes = (now - trade_time).total_seconds() / 60.0
                
                if delta_minutes <= minutes:
                    size = float(t.get("size", 0))
                    price = float(t.get("price", 1.0))
                    total_volume += size * price # Учитываем объем в долларах (USDC)
                    
            status = "VALID_ZERO" if total_volume == 0.0 else "VALID"
            return VolumeResult(volume=total_volume, status=status, timestamp=now)
            
        except Exception as e:
            logger.error("error_fetching_clob_trades", token_id=yes_token_id, error=str(e))
            return VolumeResult(volume=None, status="UNAVAILABLE", timestamp=now)

    async def get_positions(self, market_id: str) -> dict:
        """
        Возвращает балансы (positions) для данного рынка (токенов).
        Реализация зависит от ClobClient.
        """
        from py_clob_client.client import ClobClient
        # Since PolymarketClient might not have auth for ClobClient, we might need a generic way,
        # but let's return {} for now. This should ideally be called on PolyTrader instead!
        return {}
