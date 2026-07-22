import pytest
from unittest.mock import AsyncMock, MagicMock
from polyflip.api.trading_dashboard import get_funnel_stats, get_funnel_detail

@pytest.mark.asyncio
async def test_funnel_stats_empty():
    mock_db = AsyncMock()
    mock_res = MagicMock()
    mock_res.scalars.return_value.all.return_value = []
    mock_db.execute.return_value = mock_res

    res = await get_funnel_stats(hours=24, asset=None, db=mock_db)
    assert res["total"] == 0
    assert res["traded"] == 0
    assert res["by_gate"] == {}

@pytest.mark.asyncio
async def test_funnel_detail_empty():
    mock_db = AsyncMock()
    mock_res = MagicMock()
    mock_res.scalars.return_value.all.return_value = []
    mock_db.execute.return_value = mock_res

    res = await get_funnel_detail(hours=6, asset=None, limit=100, db=mock_db)
    assert res == []
