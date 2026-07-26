import asyncio
import structlog
import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import json

from polyflip.db.models import MarketSnapshot

logger = structlog.get_logger(__name__)

from typing import Any

def extract_final_outcome(market_data: dict) -> str | None:
    answer = market_data.get("answer") or market_data.get("winnerOutcome")
    
    def normalize(val: Any) -> str | None:
        if not val: return None
        val_str = str(val).upper()
        if val_str in ("YES", "UP"): return "YES"
        if val_str in ("NO", "DOWN"): return "NO"
        if val_str == "INVALID": return "INVALID"
        return None

    norm_answer = normalize(answer)
    if norm_answer:
        return norm_answer

    # Fallback to terminal prices if answer is missing or unknown, but ONLY if market is closed
    if market_data.get("closed"):
        prices = market_data.get("outcomePrices", [])
        outcomes = market_data.get("outcomes", [])
        
        if isinstance(outcomes, str):
            import json
            try: outcomes = json.loads(outcomes)
            except: pass
        if isinstance(prices, str):
            import json
            try: prices = json.loads(prices)
            except: pass
                
        if isinstance(prices, list) and isinstance(outcomes, list) and len(prices) >= 2 and len(outcomes) >= 2:
            try:
                float_prices = [float(p) for p in prices]
                winners = [i for i, p in enumerate(float_prices) if p >= 0.95]
                if len(winners) == 1:
                    return normalize(outcomes[winners[0]])
            except Exception:
                pass

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
                    logger.exception("resolver_snapshot_failed", market_id=market_id, error=str(exc))
                
            except Exception as e:
                logger.error("error_preparing_market_resolve", market_id=market_id, error=str(e))

    if any_resolved:
        try:
            await db_session.commit()
            logger.info("resolved_markets_committed")
        except Exception as e:
            logger.error("error_committing_resolved_markets", error=str(e))
            await db_session.rollback()
