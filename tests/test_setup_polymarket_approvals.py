from unittest.mock import AsyncMock, patch, MagicMock
import pytest

from scripts.setup_polymarket_approvals import run_setup

KNOWN_PRIVATE_KEY = "0x0000000000000000000000000000000000000000000000000000000000000001"
KNOWN_SIGNER_ADDRESS = "0x7E5F4552091A69125d5DfCb7b8C2659029395Bdf"


@pytest.mark.asyncio
async def test_setup_polymarket_approvals_success(monkeypatch):
    monkeypatch.setenv("POLYGON_PRIVATE_KEY", KNOWN_PRIVATE_KEY)
    monkeypatch.setenv("POLYGON_ADDRESS", "0xdef456")
    monkeypatch.setenv("POLYMARKET_RELAYER_API_KEY", "relayer-key")
    monkeypatch.setenv("POLYMARKET_RELAYER_API_KEY_ADDRESS", KNOWN_SIGNER_ADDRESS)

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
        "scripts.setup_polymarket_approvals.PolymarketExecutionGateway"
    ) as gateway_class:
        gateway_class.return_value = mock_gateway

        await run_setup(token_ids=["tok_yes_123"])

    gateway_class.assert_called_once_with(
        private_key=KNOWN_PRIVATE_KEY,
        wallet_address="0xdef456",
        relayer_api_key="relayer-key",
        relayer_api_key_address=KNOWN_SIGNER_ADDRESS,
        host="https://clob.polymarket.com",
    )
    mock_client.setup_trading_approvals.assert_awaited_once()
    mock_gateway.get_readiness.assert_awaited_once_with(
        conditional_token_ids=("tok_yes_123",)
    )


@pytest.mark.asyncio
async def test_approvals_requires_token_id(monkeypatch):
    monkeypatch.setenv("POLYGON_PRIVATE_KEY", KNOWN_PRIVATE_KEY)
    monkeypatch.setenv("POLYGON_ADDRESS", "0xdef456")
    monkeypatch.setenv("POLYMARKET_RELAYER_API_KEY", "relayer-key")
    monkeypatch.setenv("POLYMARKET_RELAYER_API_KEY_ADDRESS", KNOWN_SIGNER_ADDRESS)

    with pytest.raises(SystemExit) as exc:
        await run_setup(token_ids=[])

    assert exc.value.code == 2
