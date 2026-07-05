import httpx
from typing import List, Dict, Any
from datetime import datetime, timezone
import structlog
import json
from polyflip.constants import HTTP_TIMEOUT_SEC, VOLUME_WINDOW_MIN

logger = structlog.get_logger(__name__)

class PolymarketClient:
    GAMMA_API = "https://gamma-api.polymarket.com"
    CLOB_API = "https://clob.polymarket.com"

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=HTTP_TIMEOUT_SEC)

    async def close(self):
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.client.aclose()

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
                        "yes_token_id": yes_token_id,
                        "no_token_id": no_token_id,
                        "question": market.get("question"),
                        "asset": matched_asset,
                        "end_date_iso": market.get("endDate"),
                    })
                        
        except Exception as e:
            logger.error("error_fetching_gamma_markets", error=str(e))
            
        return markets

    async def get_market_prices(self, yes_token_id: str) -> Dict[str, Any]:
        """
        Получает стакан (orderbook) из CLOB API для вычисления mid_price и spread.
        """
        try:
            response = await self.client.get(f"{self.CLOB_API}/book", params={"token_id": yes_token_id})
            if response.status_code != 200:
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
                
            # Polymarket API может возвращать стакан отсортированным от худших цен к лучшим.
            # Поэтому надежнее искать максимум для bid и минимум для ask.
            best_bid = max(float(b.get("price", 0)) for b in bids)
            best_ask = min(float(a.get("price", 1)) for a in asks)
            
            if best_ask <= best_bid:
                logger.warning("crossed_book", token_id=yes_token_id, bid=best_bid, ask=best_ask)
                return {"error": "Crossed book (bid >= ask)"}
            
            mid_price = (best_bid + best_ask) / 2.0
            spread = best_ask - best_bid
            
            return {
                "current_yes_price": mid_price,
                "current_no_price": 1.0 - mid_price,
                "current_spread": spread,
                "best_bid": best_bid,
                "best_ask": best_ask
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
