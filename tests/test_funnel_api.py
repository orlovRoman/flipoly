import pytest
from unittest.mock import AsyncMock, MagicMock
from polyflip.api.trading_dashboard import get_funnel_stats, get_funnel_detail

@pytest.mark.asyncio
async def test_funnel_stats_empty():
    mock_row = MagicMock()
    mock_row.total = 0
    mock_row.traded = 0
    mock_db = AsyncMock()
    mock_res = MagicMock()
    mock_res.one.return_value = mock_row
    mock_db.execute.return_value = mock_res

    res = await get_funnel_stats(hours=24, asset=None, db=mock_db)
    assert res["total"] == 0
    assert res["traded"] == 0
    assert res["by_gate"] == {}

@pytest.mark.asyncio
async def test_funnel_stats_pct_calculation():
    """g4_no_flip заблокировал 40 из 100 → pct должен быть 40.0"""
    mock_row = MagicMock()
    mock_row.total = 100
    mock_row.traded = 10
    for g in ["g1_model_loaded","g2_price_fetched","g3_dead_zone",
              "g4_no_flip","g5_min_edge","g6_price_range",
              "g7_crypto_confirm","g8_combined_vote"]:
        setattr(mock_row, f"blocked_{g}", 40 if g == "g4_no_flip" else 0)

    mock_db = AsyncMock()
    mock_db.execute.side_effect = [
        MagicMock(**{"one.return_value": mock_row}),
        MagicMock(**{"all.return_value": []}),
    ]
    res = await get_funnel_stats(hours=24, asset=None, db=mock_db)
    assert res["by_gate"]["g4_no_flip"]["pct"] == 40.0
    assert res["by_gate"]["g1_model_loaded"]["pct"] == 0.0

@pytest.mark.asyncio
async def test_funnel_detail_empty():
    mock_db = AsyncMock()
    mock_res = MagicMock()
    mock_res.scalars.return_value.all.return_value = []
    mock_db.execute.return_value = mock_res

    res = await get_funnel_detail(hours=6, asset=None, limit=100, db=mock_db)
    assert res == []
