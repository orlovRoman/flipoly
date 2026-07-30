import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from polyflip.execution.gateways.polymarket import PolymarketExecutionGateway


@pytest.mark.asyncio
@patch(
    "polyflip.execution.gateways.polymarket.AsyncSecureClient.create",
    new_callable=AsyncMock,
)
async def test_gateway_passes_configured_environment(create_mock):
    create_mock.return_value = AsyncMock()

    gateway = PolymarketExecutionGateway(
        private_key="key",
        wallet_address="wallet",
        host="https://custom-clob.example",
    )

    await gateway.get_client()

    environment = create_mock.await_args.kwargs["environment"]
    assert environment.clob_url == "https://custom-clob.example"


@pytest.mark.asyncio
async def test_readiness_checks_all_conditional_tokens():
    gateway = PolymarketExecutionGateway(
        private_key="key",
        wallet_address="wallet",
        host="https://clob.polymarket.com",
    )

    client = AsyncMock()
    client.get_balance_allowance.return_value = MagicMock(
        balance=10_000_000,
        allowances={"exchange": 10_000_000},
    )
    gateway.get_client = AsyncMock(return_value=client)
    gateway.get_token_allowance = AsyncMock(
        side_effect=[Decimal("1"), Decimal("2")]
    )

    readiness = await gateway.get_readiness(
        conditional_token_ids=("YES", "NO"),
    )

    assert readiness.ready is True
    assert readiness.conditional_allowance_ready is True
    assert readiness.balance.conditional_allowances_checked == 2
    assert readiness.balance.conditional_allowance_ready is True
