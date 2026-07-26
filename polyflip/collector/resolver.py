import asyncio
import structlog
import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import json

from polyflip.db.models import MarketSnapshot

logger = structlog.get_logger(__name__)

def extract_final_outcome(market_data: dict) -> str | None:
    answer = market_data.get("answer") or market_data.get("winnerOutcome")
    
    if not answer:
        # Check outcomePrices if no explicit answer
        prices = market_data.get("outcomePrices", [])
        outcomes = market_data.get("outcomes", [])
        
        if isinstance(outcomes, str):
            import json
            try:
                outcomes = json.loads(outcomes)
            except:
                pass
        if isinstance(prices, str):
            import json
            try:
                prices = json.loads(prices)
            except:
                pass
                
        if prices and len(prices) >= 2 and outcomes and len(outcomes) >= 2:
            try:
                max_price = max(float(p) for p in prices)
                if max_price >= 0.95:
                    idx = [float(p) for p in prices].index(max_price)
                    answer = outcomes[idx]
            except Exception:
                pass

    if not answer:
        return None

    answer_upper = answer.upper()
    if answer_upper in ("YES", "UP"):
        return "YES"
    elif answer_upper in ("NO", "DOWN"):
        return "NO"
    elif answer_upper == "INVALID":
        return "INVALID"
    
    return None

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

    any_resolved = False
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
                
                final_outcome = extract_final_outcome(market_data)
                
                if final_outcome is None:
                    logger.warning("market_closed_but_no_answer", market_id=market_id)
                    continue
                
                # Теперь обновляем все снепшоты этого рынка (BUG-AK)
                try:
                    async with db_session.begin_nested():
                        snapshots_stmt = select(MarketSnapshot).where(MarketSnapshot.market_id == market_id)
                        snapshots_result = await db_session.execute(snapshots_stmt)
                        snapshots = snapshots_result.scalars().all()
                        
                        for snap in snapshots:
                            snap.final_outcome = final_outcome
                            
                            if final_outcome in ("YES", "NO"):
                                if snap.mid_price == 0.5:
                                    snap.flip_vs_final = False
                                else:
                                    market_believed_yes = snap.mid_price > 0.5
                                    actual_is_yes = (final_outcome == "YES")
                                    snap.flip_vs_final = (market_believed_yes != actual_is_yes)
                            else:
                                snap.flip_vs_final = None
                    any_resolved = True
                    logger.info("market_prepared_to_resolve", market_id=market_id, outcome=final_outcome)
                except Exception as exc:
                    await db_session.rollback()
                    logger.exception("resolver_snapshot_failed", market_id=market_id, error=str(exc))
                    raise
                
            except Exception as e:
                logger.error("error_preparing_market_resolve", market_id=market_id, error=str(e))

    if any_resolved:
        try:
            await db_session.commit()
            logger.info("resolved_markets_committed")
        except Exception as e:
            logger.error("error_committing_resolved_markets", error=str(e))
            await db_session.rollback()
