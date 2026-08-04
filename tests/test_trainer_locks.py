import asyncio
import pytest
from unittest.mock import MagicMock
from polyflip.models.trainer import serialize_training, _TRAINING_LOCKS

@pytest.fixture(autouse=True)
def clear_locks():
    _TRAINING_LOCKS.clear()
    yield
    _TRAINING_LOCKS.clear()

@pytest.mark.asyncio
async def test_different_assets_run_concurrently():
    """BTC and ETH should not block each other."""
    order = []

    @serialize_training
    async def fake_train(self, asset: str):
        order.append(f"start_{asset}")
        await asyncio.sleep(0.05)
        order.append(f"end_{asset}")
        return True

    obj = object.__new__(object)  # минимальный self
    await asyncio.gather(
        fake_train(obj, asset="BTC"),
        fake_train(obj, asset="ETH"),
    )
    
    assert order.index("start_ETH") < order.index("end_BTC")
    assert order.index("start_BTC") < order.index("end_ETH")

@pytest.mark.asyncio
async def test_same_asset_serialized():
    """Two calls for BTC should run sequentially."""
    order = []

    @serialize_training
    async def fake_train(self, asset: str):
        order.append(f"start_{asset}")
        await asyncio.sleep(0.05)
        order.append(f"end_{asset}")
        return True

    obj = object.__new__(object)  # минимальный self
    await asyncio.gather(
        fake_train(obj, asset="BTC"),
        fake_train(obj, asset="BTC"),
    )

    assert order == ["start_BTC", "end_BTC", "start_BTC", "end_BTC"]
