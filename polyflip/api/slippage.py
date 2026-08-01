from fastapi import APIRouter, Depends
from sqlalchemy import select, func
from polyflip.db.connection import async_session
from polyflip.db.models import SlippageLog
from polyflip.api.auth import verify_api_key

router = APIRouter(prefix="/api/slippage", tags=["Slippage"], dependencies=[Depends(verify_api_key)])

@router.get("/summary")
async def get_slippage_summary():
    """Агрегированная статистика slippage по активу."""
    async with async_session() as session:
        result = await session.execute(
            select(
                SlippageLog.asset,
                func.count(SlippageLog.id).label("count"),
                func.avg(SlippageLog.slippage).label("avg_slippage"),
                func.avg(SlippageLog.slippage_pct).label("avg_slippage_pct"),
                func.sum(SlippageLog.slippage_cost_usdc).label("total_cost_usdc"),
                func.max(SlippageLog.slippage).label("max_slippage"),
            ).group_by(SlippageLog.asset)
        )
        rows = result.all()
    return [dict(r._mapping) for r in rows]

@router.get("/list")
async def get_slippage_list(limit: int = 100):
    """Последние N записей slippage."""
    async with async_session() as session:
        result = await session.execute(
            select(SlippageLog).order_by(SlippageLog.created_at.desc()).limit(limit)
        )
        return result.scalars().all()
