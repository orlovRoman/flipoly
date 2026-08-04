import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from polyflip.models.trainer import _TRAINING_LOCKS, ModelTrainer

@pytest.fixture(autouse=True)
def clear_locks():
    _TRAINING_LOCKS.clear()
    yield
    _TRAINING_LOCKS.clear()

@pytest.mark.asyncio
async def test_different_assets_run_concurrently():
    """BTC and ETH should not block each other."""
    order = []
    
    original_func = ModelTrainer.train_model.__wrapped__
    
    async def fake_train_inner(self, asset):
        order.append(f"start_{asset}")
        await asyncio.sleep(0.05)
        order.append(f"end_{asset}")
        return True
        
    ModelTrainer.train_model.__wrapped__ = fake_train_inner
    try:
        await asyncio.gather(
            ModelTrainer(MagicMock()).train_model("BTC"),
            ModelTrainer(MagicMock()).train_model("ETH"),
        )
    finally:
        ModelTrainer.train_model.__wrapped__ = original_func

    # Both should start before either ends
    assert order.index("start_ETH") < order.index("end_BTC")
    assert order.index("start_BTC") < order.index("end_ETH")

@pytest.mark.asyncio
async def test_same_asset_serialized():
    """Two calls for BTC should run sequentially."""
    order = []
    
    original_func = ModelTrainer.train_model.__wrapped__
    
    async def fake_train_inner(self, asset):
        order.append(f"start_{asset}")
        await asyncio.sleep(0.05)
        order.append(f"end_{asset}")
        return True
        
    ModelTrainer.train_model.__wrapped__ = fake_train_inner
    try:
        await asyncio.gather(
            ModelTrainer(MagicMock()).train_model("BTC"),
            ModelTrainer(MagicMock()).train_model("BTC"),
        )
    finally:
        ModelTrainer.train_model.__wrapped__ = original_func

    # The second BTC should not start until the first one ends
    # order should be: start_BTC, end_BTC, start_BTC, end_BTC
    assert order == ["start_BTC", "end_BTC", "start_BTC", "end_BTC"]
