import asyncio
from polyflip.db.connection import async_session
from polyflip.db.models import TradeHistory, MarketSnapshot
from polyflip.trading.decision_logic import decide_ml_trend, decide_favorite
from polyflip.trading.feature_builder import MarketSignal
from sqlalchemy import select

async def verify():
    async with async_session() as session:
        trades = (await session.execute(
            select(TradeHistory).order_by(TradeHistory.id.desc()).limit(10)
        )).scalars().all()
        
        mismatches = 0
        for trade in trades:
            # Reconstruct signal from trade data
            signal = MarketSignal(
                asset=trade.asset,
                mid_price=float(trade.executed_price),  # приближение
                spread=0.02,  # если нет реального spread
                volume_5min=0, price_velocity=0,
                hour_of_day=trade.created_at.hour if trade.created_at else 12,
                time_left_min=5.0,
            )
            print(f"Trade {trade.id}: {trade.outcome_bought} @ {trade.executed_price:.3f} | "
                  f"strategy={trade.active_features}")
        
        print(f"\nVerified {len(trades)} trades, {mismatches} mismatches")

asyncio.run(verify())
