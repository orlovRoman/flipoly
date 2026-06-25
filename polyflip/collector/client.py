import httpx
from typing import List, Dict, Any
from datetime import datetime, timezone
import structlog

logger = structlog.get_logger(__name__)

class PolymarketClient:
    GAMMA_API = "https://gamma-api.polymarket.com"
    CLOB_API = "https://clob.polymarket.com"

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=10.0)

    async def close(self):
        await self.client.aclose()

    async def get_active_15m_markets(self, assets: List[str]) -> List[Dict[str, Any]]:
        """
        Получает активные 15-минутные рынки для заданных активов.
        Используем фильтрацию по активным событиям.
        """
        markets = []
        try:
            # Polymarket events API - берем активные и незакрытые
            response = await self.client.get(
                f"{self.GAMMA_API}/events",
                params={"active": "true", "closed": "false", "limit": 100}
            )
            response.raise_for_status()
            events = response.json()

            for event in events:
                # Проверяем, относится ли к нужным активам
                # Обычно токены есть в tags или названии
                title = event.get("title", "").upper()
                tags = [t.get("label", "").upper() for t in event.get("tags", [])]
                
                is_target_asset = any(a.upper() in title or a.upper() in tags for a in assets)
                if not is_target_asset:
                    continue

                for market in event.get("markets", []):
                    if not market.get("active") or market.get("closed"):
                        continue
                        
                    # Нас интересуют только бинарные YES/NO рынки
                    outcomes = market.get("outcomes", [])
                    if outcomes != ["Yes", "No"]:
                        continue

                    market_question = market.get("question", "").lower()
                    
                    # Фильтрация 15-минутных рынков (UP/DOWN)
                    # Обычно они содержат "15m", "15 min" или что-то подобное в вопросе или title.
                    is_15m = "15m" in market_question or "15 min" in market_question or "15m" in title.lower()
                    
                    # Также можно проверять end_date - start_date, но строковый фильтр надежнее
                    # Для CLOB API нам нужен token_id (обычно первый элемент - это токен YES)
                    clob_token_ids = market.get("clobTokenIds", [])
                    if not clob_token_ids:
                        continue
                        
                    yes_token_id = clob_token_ids[0]
                    no_token_id = clob_token_ids[1] if len(clob_token_ids) > 1 else ""

                    if is_15m:
                        markets.append({
                            "market_id": market.get("id"), # Уникальный ID рынка
                            "yes_token_id": yes_token_id,
                            "no_token_id": no_token_id,
                            "question": market.get("question"),
                            "asset": next((a for a in assets if a.upper() in title or a.upper() in tags), "UNKNOWN"),
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
                logger.warning("clob_api_error", market_id=market_id, status=response.status_code)
                return {}
                
            book = response.json()
            
            # Парсим bids (покупка YES) и asks (продажа YES)
            bids = book.get("bids", [])
            asks = book.get("asks", [])
            
            if not bids or not asks:
                return {}
                
            best_bid = float(bids[0].get("price", 0))
            best_ask = float(asks[0].get("price", 1))
            
            mid_price = (best_bid + best_ask) / 2.0
            spread = best_ask - best_bid
            
            return {
                "current_yes_price": mid_price,
                "current_no_price": 1.0 - mid_price,
                "current_spread": spread,
                "best_bid": best_bid,
                "best_ask": best_ask
            }
        except Exception as e:
            logger.error("error_fetching_clob_book", market_id=yes_token_id, error=str(e))
            return {}

    async def get_recent_trades_volume(self, yes_token_id: str, minutes: int = 5) -> float:
        """
        Получает историю сделок из CLOB API и суммирует объем за последние N минут.
        Используется для вычисления volume_5min (BUG-003).
        """
        try:
            # Пытаемся получить последние сделки по токену
            response = await self.client.get(f"{self.CLOB_API}/trades", params={"token_id": yes_token_id})
            if response.status_code != 200:
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
