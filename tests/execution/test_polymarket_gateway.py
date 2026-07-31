import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from polyflip.execution.contracts import GatewayOrder
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
        relayer_api_key="relayer-key",
        relayer_api_key_address="0x1111111111111111111111111111111111111111",
        host="https://custom-clob.example",
    )

    await gateway.get_client()

    environment = create_mock.await_args.kwargs["environment"]
    assert environment.clob_url == "https://custom-clob.example"


@pytest.mark.asyncio
@patch(
    "polyflip.execution.gateways.polymarket.AsyncSecureClient.create",
    new_callable=AsyncMock,
)
async def test_gateway_passes_relayer_credentials(create_mock):
    create_mock.return_value = AsyncMock()

    gateway = PolymarketExecutionGateway(
        private_key="0xprivate",
        wallet_address="0xwallet",
        relayer_api_key="relayer-key",
        relayer_api_key_address="0x1111111111111111111111111111111111111111",
    )

    await gateway.get_client()

    api_key = create_mock.await_args.kwargs["api_key"]
    assert api_key.key == "relayer-key"
    assert api_key.address == "0x1111111111111111111111111111111111111111"


@pytest.mark.asyncio
async def test_readiness_checks_all_conditional_tokens():
    gateway = PolymarketExecutionGateway(
        private_key="key",
        wallet_address="wallet",
        relayer_api_key="relayer-key",
        relayer_api_key_address="0x1111111111111111111111111111111111111111",
        host="https://clob.polymarket.com",
    )

    client = AsyncMock()
    client.get_balance_allowance.return_value = MagicMock(
        balance=10_000_000,
        allowances={"exchange": 10_000_000},
    )
    gateway.get_client = AsyncMock(return_value=client)
    gateway.get_token_allowance = AsyncMock(side_effect=[Decimal("1"), Decimal("2")])

    readiness = await gateway.get_readiness(
        conditional_token_ids=("YES", "NO"),
    )

    assert readiness.ready is True
    assert readiness.conditional_allowance_ready is True
    assert readiness.balance.conditional_allowances_checked == 2
    assert readiness.balance.conditional_allowance_ready is True


@pytest.mark.asyncio
async def test_gateway_submits_buy_and_reads_order():
    gateway = PolymarketExecutionGateway(
        private_key="dummy_key",
        wallet_address="0xDummyAddress",
        relayer_api_key="relayer-key",
        relayer_api_key_address="0x1111111111111111111111111111111111111111",
        host="https://clob.polymarket.com",
    )

    client = AsyncMock()
    client.place_market_order.return_value = MagicMock(
        ok=True,
        order_id="order-123",
        status="FILLED",
        trade_ids=[],
    )
    client.get_order.return_value = {
        "status": "FILLED",
    }
    gateway.get_client = AsyncMock(return_value=client)

    order = GatewayOrder(
        attempt_id=uuid4(),
        market_id="market-1",
        asset="BTC",
        outcome_to_buy="YES",
        token_id="token-yes",
        side="BUY",
        requested_shares=Decimal("10"),
        max_spend_usdc=Decimal("5"),
        limit_price=Decimal("0.5"),
    )

    submission = await gateway.submit(order)

    assert submission.accepted is True
    assert submission.provider_status == "FILLED"
    assert submission.provider_order_id == "order-123"

    client.place_market_order.assert_awaited_once_with(
        token_id="token-yes",
        side="BUY",
        amount="5",
        max_spend="5",
        max_price="0.5",
        order_type="FAK",
    )

    observed = await gateway.get_order("order-123")

    assert observed.provider_status == "FILLED"
    client.get_order.assert_awaited_once_with(
        order_id="order-123",
    )
