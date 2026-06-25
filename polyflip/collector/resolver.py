import asyncio
import structlog
import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from polyflip.db.models import MarketSnapshot

logger = structlog.get_logger(__name__)

async def resolve_pending_markets(db_session: AsyncSession):
    """
    Находит закрытые рынки со статусом PENDING и обновляет их исход.
    Определяет, произошел ли 'флип' (flip_vs_final).
    """
    # Выбираем уникальные рынки, которые еще PENDING
    stmt = select(MarketSnapshot.market_id).where(
        MarketSnapshot.final_outcome == "PENDING"
    ).distinct()
    
    result = await db_session.execute(stmt)
    pending_market_ids = result.scalars().all()
    
    if not pending_market_ids:
        logger.info("no_pending_markets_to_resolve")
        return

    logger.info("resolving_pending_markets", count=len(pending_market_ids))

    async with httpx.AsyncClient(timeout=10.0) as client:
        for market_id in pending_market_ids:
            try:
                # Получаем инфу о рынке из Gamma API
                response = await client.get(f"https://gamma-api.polymarket.com/markets/{market_id}")
                
                # BUG-009 FIX: Rate limit protection
                await asyncio.sleep(0.2)
                
                if response.status_code != 200:
                    continue
                
                market_data = response.json()
                
                if not market_data.get("closed"):
                    continue # Еще не закрыт
                
                # Обычно Polymarket возвращает answer или winning_outcome для закрытых рынков
                # Если их нет, но рынок закрыт, возможно он INVALID.
                # Для 15m рынков обычно ответ "Yes" или "No" (соответствует outcomes)
                # Поищем поле "answer" или сымитируем через prices, если он жестко зафиксирован на 1 или 0
                answer = market_data.get("answer") 
                
                # Если явного ответа нет, но токены залочены (один стоит 1, другой 0)
                if not answer:
                    prices = market_data.get("outcomePrices", [])
                    outcomes = market_data.get("outcomes", ["Yes", "No"])
                    
                    if type(outcomes) is str:
                        import json
                        outcomes = json.loads(outcomes)
                    if type(prices) is str:
                        import json
                        prices = json.loads(prices)
                        
                    if prices and len(prices) >= 2 and outcomes and len(outcomes) >= 2:
                        if str(prices[0]) in ("1", "1.0"):
                            answer = outcomes[0]
                        elif str(prices[1]) in ("1", "1.0"):
                            answer = outcomes[1]

                if not answer:
                    logger.warning("market_closed_but_no_answer", market_id=market_id)
                    continue
                
                # Нормализуем UP/DOWN в YES/NO для корректного расчета флипа (BUG-005)
                outcome_map = {"UP": "YES", "DOWN": "NO", "YES": "YES", "NO": "NO"}
                final_outcome = outcome_map.get(answer.upper(), answer.upper()) # "YES" или "NO"
                
                # Теперь обновляем все снепшоты этого рынка
                snapshots_stmt = select(MarketSnapshot).where(MarketSnapshot.market_id == market_id)
                snapshots_result = await db_session.execute(snapshots_stmt)
                snapshots = snapshots_result.scalars().all()
                
                for snap in snapshots:
                    snap.final_outcome = final_outcome
                    
                    # Логика флипа:
                    # mid_price > 0.5 означает, что рынок верил в "YES".
                    # Если исход "NO", значит произошел флип.
                    # И наоборот: mid_price < 0.5 -> верили в "NO", если исход "YES" -> флип.
                    market_believed_yes = snap.mid_price > 0.5
                    actual_is_yes = (final_outcome == "YES")
                    
                    # Исключаем идеальные 0.5 для чистоты (неопределенность)
                    if snap.mid_price == 0.5:
                        snap.flip_vs_final = False
                    else:
                        snap.flip_vs_final = (market_believed_yes != actual_is_yes)
                        
                await db_session.commit()
                logger.info("market_resolved", market_id=market_id, outcome=final_outcome)
                
            except Exception as e:
                logger.error("error_resolving_market", market_id=market_id, error=str(e))
                await db_session.rollback()
