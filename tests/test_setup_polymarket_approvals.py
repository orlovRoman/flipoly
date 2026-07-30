from unittest.mock import AsyncMock, patch, MagicMock
import pytest

from scripts.setup_polymarket_approvals import run_setup


@pytest.mark.asyncio
async def test_setup_polymarket_approvals_success(monkeypatch):
    monkeypatch.setenv("POLYGON_PRIVATE_KEY", "0xabc123")
    monkeypatch.setenv("POLYGON_ADDRESS", "0xdef456")

    mock_client = AsyncMock()
    mock_client.setup_trading_approvals = AsyncMock()

    mock_readiness = MagicMock()
    mock_readiness.ready = True
    mock_readiness.collateral_allowance_ready = True
    mock_readiness.conditional_allowance_ready = True

    mock_gateway = AsyncMock()
    mock_gateway.get_client = AsyncMock(return_value=mock_client)
    mock_gateway.get_readiness = AsyncMock(return_value=mock_readiness)

    with patch(
        "scripts.setup_polymarket_approvals.PolymarketExecutionGateway",
        return_value=mock_gateway,
    ):
        await run_setup(token_ids=["tok_yes_123"])

    mock_client.setup_trading_approvals.assert_awaited_once()
    mock_gateway.get_readiness.assert_awaited_once_with(
        conditional_token_ids=("tok_yes_123",)
    )
