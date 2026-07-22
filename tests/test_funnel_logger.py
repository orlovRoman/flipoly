import pytest
from unittest.mock import AsyncMock, MagicMock
from polyflip.trading.funnel_logger import log_funnel

@pytest.mark.asyncio
async def test_funnel_logger_no_raise():
    """FunnelLogger не бросает исключений при ошибке записи."""
    mock_session = AsyncMock()
    mock_session.commit.side_effect = Exception("DB Connection Error")

    # Должен молча обработать исключение и записать warning в лог, не выбрасывая ошибку
    await log_funnel(
        mock_session, market_id="x" * 200, asset="BTC",
        trading_mode="ML", used_model=None,
        p_flip=0.3, edge=0.05, fresh_price=0.7,
        threshold_lower=0.35, threshold_upper=0.65,
        min_edge_used=0.05, final_action="SKIP",
    )
