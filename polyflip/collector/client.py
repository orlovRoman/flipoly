import asyncio
import httpx
from typing import List, Dict, Any, TypedDict
from datetime import datetime, timezone
import structlog
import json
import time
from polyflip.constants import HTTP_TIMEOUT_SEC, VOLUME_WINDOW_MIN

logger = structlog.get_logger(__name__)


class MarketPricesResult(TypedDict, total=False):
    current_yes_price: float
    current_no_price: float
    current_spread: float
    best_bid: float
    best_ask: float
    tick_size: float | None
    min_order_size: float | None
    error: str
    bids: list[dict[str, float]]
    asks: list[dict[str, float]]

def _canonical_strike(market: Dict[str, Any], event: Dict[str, Any]) -> float | None:
    """Extract Polymarket's opening/Chainlink strike without Binance fallbacks."""
    candidates = [
        market.get("underlying_price"),
        market.get("underlyingPrice"),
        market.get("strike"),
        market.get("strikePrice"),
        market.get("priceToBeat"),
        market.get("openingPrice"),
        event.get("underlying_price"),
        event.get("underlyingPrice"),
        event.get("strike"),
        event.get("strikePrice"),
        event.get("priceToBeat"),
    ]
    for candidate in candidates:
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            continue
        if value > 0.0 and value == value and value != float("inf"):
            return value
    return None

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
                    })
                        
        except Exception as e:
            logger.error("error_fetching_gamma_markets", error=str(e))
            
        return markets

    async def get_market_prices(self, yes_token_id: str) -> MarketPricesResult:
        """
        Получает стакан (orderbook) из CLOB API для вычисления mid_price и spread.
        """
        try:
            response = None
            for attempt in range(3):
                try:
                    response = await self.client.get(
                        f"{self.CLOB_API}/book", params={"token_id": yes_token_id}
                    )
                except (httpx.TimeoutException, httpx.NetworkError) as exc:
                    if attempt == 2:
                        raise
                    delay = 0.25 * (2 ** attempt)
                    logger.warning(
                        "clob_book_retry", token_id=yes_token_id,
                        attempt=attempt + 1, delay_sec=delay, error=str(exc),
                    )
                    await asyncio.sleep(delay)
                    continue
                if response.status_code in {408, 429, 500, 502, 503, 504} and attempt < 2:
                    delay = 0.25 * (2 ** attempt)
                    logger.warning(
                        "clob_book_retry", token_id=yes_token_id,
                        attempt=attempt + 1, delay_sec=delay,
                        status=response.status_code,
                    )
                    await asyncio.sleep(delay)
                    continue
                break
            if response is None:
                return {"error": "API book request returned no response"}
            if response.status_code != 200:
                if response.status_code == 404:
                    logger.debug("clob_api_404_market_likely_closed", token_id=yes_token_id)
                else:
                    logger.warning("clob_api_error", token_id=yes_token_id, status=response.status_code)
                return {"error": f"API HTTP Error {response.status_code}"}
            book = response.json()
            
            # Парсим bids (покупка YES) и asks (продажа YES)
            bids = book.get("bids", [])
            asks = book.get("asks", [])
            
            if not bids or not asks:
                logger.warning(
                    "empty_orderbook_raw",
                    token_id=yes_token_id,
                    bids_count=len(bids),
                    asks_count=len(asks),
                    raw_keys=list(book.keys()),
                )
                return {"error": "Empty orderbook (no bids/asks)"}
                
            def _levels(raw_levels):
                normalized = []
                for level in raw_levels:
                    try:
                        price = float(level.get("price"))
                        size = float(level.get("size") or level.get("quantity") or 0)
                    except (AttributeError, TypeError, ValueError):
                        continue
                    if price > 0 and size > 0 and price == price and size == size:
                        normalized.append({"price": price, "size": size})
                return normalized

            normalized_bids = _levels(bids)
            normalized_asks = _levels(asks)
            if not normalized_bids or not normalized_asks:
                logger.warning(
                    "empty_orderbook_normalized",
                    token_id=yes_token_id,
                    bids_count=len(normalized_bids),
                    asks_count=len(normalized_asks),
                )
                return {"error": "Empty orderbook (no valid bids/asks)"}

            # Polymarket API может возвращать стакан отсортированным от худших цен к лучшим.
            # Поэтому надежнее искать максимум для bid и минимум для ask.
            best_bid = max(level["price"] for level in normalized_bids)
            best_ask = min(level["price"] for level in normalized_asks)

            if best_ask <= best_bid:
                logger.warning("crossed_book", token_id=yes_token_id, bid=best_bid, ask=best_ask)
                return {"error": "Crossed book (bid >= ask)"}

            mid_price = (best_bid + best_ask) / 2.0
            spread = best_ask - best_bid

            tick_size = book.get("tick_size") or book.get("minimum_tick_size")
            min_order_size = book.get("min_order_size") or book.get("minimum_order_size")
            return {
                "current_yes_price": mid_price,
                "current_no_price": 1.0 - mid_price,
                "current_spread": spread,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "tick_size": float(tick_size) if tick_size is not None else None,
                "min_order_size": float(min_order_size) if min_order_size is not None else None,
                "bids": normalized_bids,
                "asks": normalized_asks,
            }
        except httpx.TimeoutException:
            logger.error("error_fetching_clob_book_timeout", token_id=yes_token_id)
            return {"error": "API Timeout"}
        except httpx.NetworkError:
            logger.error("error_fetching_clob_book_network", token_id=yes_token_id)
            return {"error": "API Network Error"}
        except Exception as e:
            logger.error("error_fetching_clob_book", market_id=yes_token_id, error=str(e))
            return {"error": f"API Error: {str(e)}"}

    async def get_recent_trades_volume(self, yes_token_id: str, minutes: int = VOLUME_WINDOW_MIN) -> float:
        """
        Получает историю сделок из CLOB API и суммирует объем за последние N минут.
        Используется для вычисления volume_5min (BUG-003).
        """
        try:
            # Пытаемся получить последние сделки по токену
            response = await self.client.get(f"{self.CLOB_API}/trades", params={"token_id": yes_token_id})
            if response.status_code != 200:
                if response.status_code != 401: # 401 means no CLOB API keys, don't spam
                    logger.warning("clob_trades_api_error", token_id=yes_token_id, status=response.status_code)
                return 0.0
                
            trades = response.json()
            if not isinstance(trades, list):
                # Иногда API отдает словарь с ключом data или history
                trades = trades.get("data", []) or trades.get("trades", [])

            now = datetime.now(timezone.utc)
            total_volume = 0.0
            
            for t in trades:
                # Парсим время сделки. Формат обычно ISO8601
                timestamp_str = t.get("timestamp") or t.get("created_at")
                if not timestamp_str:
                    continue
                
                # Приводим к UTC
                trade_time = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                delta_minutes = (now - trade_time).total_seconds() / 60.0
                
                if delta_minutes <= minutes:
                    size = float(t.get("size", 0))
                    price = float(t.get("price", 0))
                    total_volume += size * price # Учитываем объем в долларах (USDC)
                    
            return total_volume
            
        except Exception as e:
            logger.error("error_fetching_clob_trades", token_id=yes_token_id, error=str(e))
            return 0.0

    async def get_positions(self, market_id: str) -> dict:
        """
        Возвращает балансы (positions) для данного рынка (токенов).
        Реализация зависит от ClobClient.
        """
        from py_clob_client.client import ClobClient
        # Since PolymarketClient might not have auth for ClobClient, we might need a generic way,
        # but let's return {} for now. This should ideally be called on PolyTrader instead!
        return {}
