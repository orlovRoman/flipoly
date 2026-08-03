import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from decimal import Decimal
from polyflip.db.execution_models import ExecutionRequest
from polyflip.execution.worker import process_ready_requests

@pytest.mark.asyncio
async def test_live_quote_failure_does_not_submit():
    """Если при получении котировки происходит ошибка (в LIVE режиме), запрос должен перейти в READY с ошибкой,
    и не должен создаваться ExecutionAttempt."""
    from polyflip.db.models import LiveMarket
    
    # We patch the necessary functions in worker.py
    with patch("polyflip.execution.worker.ExecutionSettings") as mock_settings:
        mock_settings.return_value.execution_mode.value = "LIVE"
        
        with patch("polyflip.execution.worker.claim_one") as mock_claim:
            req = ExecutionRequest(
                id="req-123",
                intent="OPEN",
                requested_mode="LIVE",
                market_id="m1",
                state="CLAIMED",
                limit_price=Decimal("0.5"),
                max_acceptable_price=Decimal("0.6")
            )
            mock_claim.side_effect = [req, None]
            
            with patch("polyflip.execution.worker.build_execution_gateway") as mock_build_gw:
                mock_gw = AsyncMock()
                mock_build_gw.return_value = mock_gw
                
                with patch("polyflip.execution.worker.check_risk_limits", return_value=None):
                    with patch("polyflip.execution.worker.async_session") as mock_session_ctx:
                        mock_session = AsyncMock()
                        mock_session_ctx.return_value.__aenter__.return_value = mock_session
                        
                        # mock `select` to return a market
                        with patch("polyflip.execution.worker.select") as mock_select:
                            from unittest.mock import MagicMock
                            mock_result = MagicMock()
                            # We just need to mock session.execute().scalar_one_or_none() to return a mock market
                            market = LiveMarket(yes_token_id="yes1", no_token_id="no1")
                            mock_result.scalar_one_or_none.return_value = market
                            mock_session.execute.return_value = mock_result
                            
                            with patch("polyflip.collector.client.PolymarketClient") as mock_client:
                                api_client = AsyncMock()
                                # Simulate quote failure
                                api_client.get_market_prices.side_effect = Exception("Quote failed")
                                mock_client.return_value = api_client
                                
                                await process_ready_requests()
                                
                                # Assertions
                                assert req.state == "READY"
                                assert req.error_reason == "EXECUTION_QUOTE_UNAVAILABLE"
                                mock_gw.submit.assert_not_awaited()
                                # Verify attempt wasn't created
                                assert not any(call[0][0].__class__.__name__ == "ExecutionAttempt" for call in mock_session.add.call_args_list)
