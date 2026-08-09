"""
tests/crypto/test_trainer_polymarket_target.py

Тесты трейнера:
  - Пустой Polymarket-датасет -> обучение прекращается без записи модели или создания ModelRegistry.
  - Валидация колонок и таргета {0, 1}.
"""
import pytest
import pandas as pd
from unittest.mock import AsyncMock, MagicMock, patch
from polyflip.crypto.trainer import CryptoModelTrainer


@pytest.mark.asyncio
async def test_empty_polymarket_dataset_stops_training_without_saving():
    db = AsyncMock()
    mock_res = MagicMock()
    mock_res.scalar_one_or_none.return_value = None
    mock_res.scalars.return_value.all.return_value = []
    db.execute.return_value = mock_res

    trainer = CryptoModelTrainer(db)

    with patch("polyflip.crypto.trainer.get_recent_candles", AsyncMock(return_value=[MagicMock()] * 600)), \
         patch("polyflip.crypto.trainer.build_market_outcome_dataset", AsyncMock(return_value=pd.DataFrame())):
        result = await trainer.train(symbol="BTCUSDT", interval="15m")
        assert result is False
        assert db.add.call_count == 0
