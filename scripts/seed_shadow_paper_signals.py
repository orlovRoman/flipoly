import asyncio
import os
import sys
import uuid
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from polyflip.db.models import TradeHistory, RuntimeSettings
from polyflip.db.execution_models import ExecutionRequest
from polyflip.execution.live_mirror_worker import set_mirror_enabled


async def upsert_runtime_setting(session: AsyncSession, key: str, value: str) -> None:
    now = datetime.now(timezone.utc)
    setting = (
        await session.execute(select(RuntimeSettings).where(RuntimeSettings.key == key))
    ).scalar_one_or_none()
    if setting:
        setting.value = value
        setting.updated_at = now
        setting.updated_by = "seed_script"
    else:
        session.add(
            RuntimeSettings(
                key=key, value=value, updated_at=now, updated_by="seed_script"
            )
        )


async def seed() -> None:
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("DATABASE_URL is not set.")
        sys.exit(1)

    print(f"Seeding shadow paper signals to {db_url}...")
    engine = create_async_engine(db_url, echo=False)
    SessionLocal = async_sessionmaker(engine, expire_on_commit=False)

    run_id = uuid.uuid4().hex[:8]

    async with SessionLocal() as session:
        # 1. Включаем mirror switch и сбрасываем timestamp отсечки LIVE_MIRROR_STARTED_AT = now
        await set_mirror_enabled(session, enabled=True, updated_by="seed_script")
        await upsert_runtime_setting(session, key="LIVE_RELEASE_MODE", value="AUTO")
        await session.commit()

        # 2. Генерируем 5 синтетических исполненных PAPER ордеров со свежими метками времени (сигналы после включения mirror)
        for i in range(1, 6):
            signal_time = datetime.now(timezone.utc)
            trade = TradeHistory(
                market_id=f"MKT-SEED-{run_id}-{i}",
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
                created_at=signal_time,
            )
            session.add(trade)
            await session.flush()

            request = ExecutionRequest(
                id=uuid.uuid4(),
                idempotency_key=f"PAPER-OPEN-seed-{run_id}-{i}",
                trade_history_id=trade.id,
                market_id=trade.market_id,
                asset=trade.asset,
                intent="OPEN",
                outcome_to_buy=trade.outcome_bought,
                target_amount_usdc=Decimal("10.0"),
                requested_shares=Decimal("20.0"),
                limit_price=Decimal("0.50"),
                max_slippage_pct=0.01,
                filled_shares=Decimal("20.0"),
                filled_cost_usdc=Decimal("10.0"),
                requested_mode="PAPER",
                state="FILLED",
                created_at=signal_time,
                updated_at=signal_time,
            )
            session.add(request)

        await session.commit()
        print("Successfully seeded 5 synthetic PAPER trades and requests.")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(seed())
