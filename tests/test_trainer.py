import pytest
from sqlalchemy import select
from datetime import datetime, timezone
from unittest.mock import patch
from polyflip.config import settings
from polyflip.db.models import MarketSnapshot, ModelRegistry
from polyflip.models.trainer import ModelTrainer

@pytest.mark.asyncio
async def test_trainer_skips_insufficient_data(db_session):
    trainer = ModelTrainer(db_session)
    with patch.object(settings, "MIN_SAMPLES_FOR_MODEL", 10):
        res = await trainer.train_model("BTC")
    assert res is False

    stmt = select(ModelRegistry).where(ModelRegistry.asset == "BTC")
    models = (await db_session.execute(stmt)).scalars().all()
    assert len(models) == 0

@pytest.mark.asyncio
async def test_trainer_creates_model(db_session):
    # Insert 20 dummy snapshots (resolved)
    snaps = []
    for i in range(20):
        snaps.append(MarketSnapshot(
            market_id=f"test_m_{i}", asset="BTC", time_left_min=10.0,
            mid_price=0.8 if i % 2 == 0 else 0.2, spread=0.01,
            volume_5min=100.0, price_velocity=0.0, hour_of_day=12,
            final_outcome="NO", # Both resolved to NO
            flip_vs_final=True if i % 2 == 0 else False, # Even: mid=0.8->YES, final=NO -> flip
            recorded_at=datetime.now(timezone.utc)
        ))
    db_session.add_all(snaps)
    await db_session.commit()

    trainer = ModelTrainer(db_session)
    with patch.object(settings, "MIN_SAMPLES_FOR_MODEL", 10):
        res = await trainer.train_model("BTC")
    assert res is True

    stmt = select(ModelRegistry).where(ModelRegistry.asset == "BTC")
    models = (await db_session.execute(stmt)).scalars().all()
    assert len(models) == 1
    assert models[0].is_active is True
    assert models[0].model_blob is not None

@pytest.mark.asyncio
async def test_trainer_saves_model_even_if_accuracy_is_low(db_session):
    # Создаем 20 снимков, где классы перемешаны случайно, чтобы модель получилась "глупой"
    snaps = []
    for i in range(20):
        snaps.append(MarketSnapshot(
            market_id=f"test_low_m_{i}", asset="BTC", time_left_min=10.0,
            mid_price=0.5, spread=0.01,
            volume_5min=100.0, price_velocity=0.0, hour_of_day=12,
            final_outcome="NO",
            flip_vs_final=True if i % 2 == 0 else False,
            recorded_at=datetime.now(timezone.utc)
        ))
    db_session.add_all(snaps)
    await db_session.commit()

    trainer = ModelTrainer(db_session)
    with patch.object(settings, "MIN_SAMPLES_FOR_MODEL", 10):
        res = await trainer.train_model("BTC")
    
    assert res is True # Модель должна успешно сохраниться, несмотря на низкую точность

    stmt = select(ModelRegistry).where(ModelRegistry.asset == "BTC").order_by(ModelRegistry.version.desc())
    models = (await db_session.execute(stmt)).scalars().all()
    assert len(models) >= 1
    assert models[0].baseline is not None
