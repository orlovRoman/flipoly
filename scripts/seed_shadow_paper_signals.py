import asyncio
import os
import sys
from datetime import datetime, timezone, timedelta
from decimal import Decimal

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from polyflip.db.models import TradeHistory
from polyflip.db.execution_models import ExecutionRequest
from polyflip.execution.live_mirror_worker import set_mirror_enabled


async def seed() -> None:
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("DATABASE_URL is not set.")
        sys.exit(1)

    print(f"Seeding shadow paper signals to {db_url}...")
    engine = create_async_engine(db_url, echo=False)
    SessionLocal = async_sessionmaker(engine, expire_on_commit=False)

    async with SessionLocal() as session:
        # Включаем mirror switch и сбрасываем timestamp отсечки
        await set_mirror_enabled(session, enabled=True, updated_by="seed_script")
        await session.commit()

        now = datetime.now(timezone.utc)
        
        # Создаем 5 синтетических исполненных PAPER ордеров
        for i in range(1, 6):
            t = TradeHistory(
                market_id=f"MKT-SEED-{i}",
                asset="BTC" if i % 2 == 1 else "ETH",
                outcome_bought="YES" if i % 2 == 1 else "NO",
                amount_usdc=10.0,
                executed_price=0.50,
                predicted_flip_prob=0.65,
                active_features="seed_test_features",
                model_version=1,
                status="SUCCESS",
                mode="PAPER",
                position_status="OPEN",
                entry_filled_shares=20.0,
                entry_cost_usdc=10.0,
                remaining_shares=20.0,
                realized_pnl_usdc=0.0,
                market_role="FAVORITE",
                edge=0.15,
                created_at=now - timedelta(seconds=i * 2),
            )
            session.add(t)
            await session.flush()

            req = ExecutionRequest(
                trade_history_id=t.id,
                market_id=t.market_id,
                asset=t.asset,
                intent="OPEN",
                outcome_to_buy=t.outcome_bought,
                target_amount_usdc=Decimal("10.0"),
                requested_shares=Decimal("20.0"),
                limit_price=Decimal("0.50"),
                filled_shares=Decimal("20.0"),
                filled_cost_usdc=Decimal("10.0"),
                requested_mode="PAPER",
                state="FILLED",
                created_at=t.created_at,
                updated_at=t.created_at,
            )
            session.add(req)

        await session.commit()
        print("Successfully seeded 5 synthetic PAPER trades and requests.")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(seed())
