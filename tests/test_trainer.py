import pytest
from sqlalchemy import select
from datetime import datetime, timezone
from unittest.mock import patch
from polyflip.config import settings
from polyflip.db.models import MarketSnapshot, ModelRegistry, RuntimeSettings
from polyflip.models.trainer import ModelTrainer

@pytest.mark.asyncio
async def test_trainer_skips_insufficient_data(db_session):
    trainer = ModelTrainer(db_session)
    with patch.object(settings, 'MIN_SAMPLES_FOR_MODEL', 10):
        res = await trainer.train_model('BTC')
    assert res is False
    stmt = select(ModelRegistry).where(ModelRegistry.asset == 'BTC')
    models = (await db_session.execute(stmt)).scalars().all()
    assert len(models) == 0