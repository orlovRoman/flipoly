import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from datetime import datetime, timezone, timedelta

from polyflip.execution.contracts import GatewayOrder
from polyflip.execution.gateways.polymarket import PolymarketExecutionGateway

KNOWN_PRIVATE_KEY = "0x0000000000000000000000000000000000000000000000000000000000000001"
KNOWN_SIGNER_ADDRESS = "0x7E5F4552091A69125d5DfCb7b8C2659029395Bdf"


@pytest.mark.asyncio
@patch(
    "polyflip.execution.gateways.polymarket.AsyncSecureClient.create",
    new_callable=AsyncMock,
)
async def test_gateway_passes_configured_environment(create_mock):
    create_mock.return_value = AsyncMock()

    gateway = PolymarketExecutionGateway(
        private_key=KNOWN_PRIVATE_KEY,
        wallet_address="wallet",
        relayer_api_key="relayer-key",
        relayer_api_key_address=KNOWN_SIGNER_ADDRESS,
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
        private_key=KNOWN_PRIVATE_KEY,
        wallet_address="0xwallet",
        relayer_api_key="relayer-key",
        relayer_api_key_address=KNOWN_SIGNER_ADDRESS,
    )

    await gateway.get_client()

    api_key = create_mock.await_args.kwargs["api_key"]
    assert api_key.key == "relayer-key"
    assert api_key.address == KNOWN_SIGNER_ADDRESS


@pytest.mark.asyncio
async def test_readiness_checks_all_conditional_tokens():
    gateway = PolymarketExecutionGateway(
        private_key=KNOWN_PRIVATE_KEY,
        wallet_address="wallet",
        relayer_api_key="relayer-key",
        relayer_api_key_address=KNOWN_SIGNER_ADDRESS,
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
        private_key=KNOWN_PRIVATE_KEY,
        wallet_address="0xDummyAddress",
        relayer_api_key="relayer-key",
        relayer_api_key_address=KNOWN_SIGNER_ADDRESS,
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


def test_rejects_mismatched_signer_private_key():
    from polyflip.execution.contracts import GatewayUnavailable

    gateway = PolymarketExecutionGateway(
        private_key=KNOWN_PRIVATE_KEY,
        wallet_address="0xPolymarketWallet",
        relayer_api_key="relayer-key",
        relayer_api_key_address="0x1111111111111111111111111111111111111111",
    )

    with pytest.raises(
        GatewayUnavailable,
        match="does not match",
    ):
        gateway._validate_credentials()


@pytest.mark.asyncio
async def test_readiness_reports_signer_mismatch():
    gateway = PolymarketExecutionGateway(
        private_key=KNOWN_PRIVATE_KEY,
        wallet_address="0xPolymarketWallet",
        relayer_api_key="relayer-key",
        relayer_api_key_address="0x1111111111111111111111111111111111111111",
    )

    result = await gateway.get_readiness(conditional_token_ids=("token-1",))

    assert result.ready is False
    assert result.credentials_loaded is True
    assert result.client_initialized is False
    assert "does not match" in result.error_message


@pytest.mark.asyncio
async def test_client_close_failure_does_not_break_invalidation():
    gateway = PolymarketExecutionGateway(
        private_key=KNOWN_PRIVATE_KEY,
        wallet_address="0xPolymarketWallet",
        relayer_api_key="relayer-key",
        relayer_api_key_address=KNOWN_SIGNER_ADDRESS,
    )

    mock_client = AsyncMock()
    # Force close to raise an exception
    mock_client.close.side_effect = Exception("Test close error")
    gateway._client_cache = mock_client

    # invalidation should suppress the exception and still set _client_cache to None
    await gateway.invalidate_client()

    assert gateway._client_cache is None
    mock_client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_fak_no_liquidity_is_deterministic_rejection():
    from polyflip.execution.contracts import GatewayOrderRejected

    gateway = PolymarketExecutionGateway(
        private_key=KNOWN_PRIVATE_KEY,
        wallet_address="0xPolymarketWallet",
        relayer_api_key="relayer-key",
        relayer_api_key_address=KNOWN_SIGNER_ADDRESS,
    )
    mock_client = AsyncMock()
    gateway._client_cache = mock_client

    mock_client.place_market_order.side_effect = Exception(
        "no orders found to match with FAK order. "
        "FAK orders are partially filled or killed if no match is found."
    )

    order = GatewayOrder(
        attempt_id="00000000-0000-0000-0000-000000000001",
        market_id="m1",
        asset="BTC",
        outcome_to_buy="YES",
        side="BUY",
        intent="OPEN",
        token_id="token-yes",
        requested_shares=Decimal("5"),
        requested_amount_usdc=Decimal("5"),
        limit_price=Decimal("0.5"),
    )

    with pytest.raises(GatewayOrderRejected, match="NO_LIQUIDITY_FAK"):
        await gateway.submit(order)

@pytest.mark.asyncio
async def test_gateway_routes_gtc_to_limit_order():
    gateway = PolymarketExecutionGateway(
        private_key=KNOWN_PRIVATE_KEY,
        wallet_address="0xDummyAddress",
        relayer_api_key="relayer-key",
        relayer_api_key_address=KNOWN_SIGNER_ADDRESS,
    )
    client = AsyncMock()
    client.place_limit_order.return_value = MagicMock(
        ok=True,
        order_id="gtc-order-1",
        status="LIVE",
        trade_ids=[],
    )
    gateway.get_client = AsyncMock(return_value=client)

    order = GatewayOrder(
        attempt_id=uuid4(),
        market_id="market-gtc",
        asset="BTC",
        outcome_to_buy="YES",
        token_id="token-yes",
        side="BUY",
        requested_shares=Decimal("10"),
        max_spend_usdc=Decimal("5"),
        limit_price=Decimal("0.5"),
    )

    submission = await gateway.submit(order, order_type="GTC")

    assert submission.accepted is True
    assert submission.provider_order_id == "gtc-order-1"
    client.place_limit_order.assert_awaited_once_with(
        token_id="token-yes",
        price="0.5",
        size="10",
        side="BUY",
        post_only=False,
        expiration=None,
    )
    client.place_market_order.assert_not_awaited()


@pytest.mark.asyncio
async def test_gateway_does_not_apply_market_minimum_locally():
    from polyflip.execution.gateways.exceptions import GatewayOrderRejected

    gateway = PolymarketExecutionGateway(
        private_key=KNOWN_PRIVATE_KEY,
        wallet_address="0xDummyAddress",
        relayer_api_key="relayer-key",
        relayer_api_key_address=KNOWN_SIGNER_ADDRESS,
        host="https://clob.polymarket.com",
    )
    client = AsyncMock()
    client.place_limit_order.return_value = MagicMock(
        ok=True,
        order_id="small-order-1",
        status="LIVE",
        trade_ids=[],
    )
    gateway.get_client = AsyncMock(return_value=client)

    order = GatewayOrder(
        attempt_id=uuid4(),
        market_id="market-min-size",
        asset="XRP",
        outcome_to_buy="NO",
        token_id="token-no",
        side="BUY",
        requested_shares=Decimal("3.66"),
        max_spend_usdc=Decimal("1.10"),
        limit_price=Decimal("0.30"),
        post_only=True,
        minimum_shares=Decimal("5"),
    )

    submission = await gateway.submit(order, order_type="GTC")

    assert submission.accepted is True
    assert submission.provider_order_id == "small-order-1"
    client.place_limit_order.assert_awaited_once_with(
        token_id="token-no",
        price="0.30",
        size="3.66",
        side="BUY",
        post_only=True,
        expiration=None,
    )

@pytest.mark.asyncio
async def test_gateway_routes_gtd_to_limit_order_with_expiration():
    gateway = PolymarketExecutionGateway(
        private_key=KNOWN_PRIVATE_KEY,
        wallet_address="0xDummyAddress",
        relayer_api_key="relayer-key",
        relayer_api_key_address=KNOWN_SIGNER_ADDRESS,
    )
    client = AsyncMock()
    client.place_limit_order.return_value = MagicMock(
        ok=True,
        order_id="gtd-order-1",
        status="LIVE",
        trade_ids=[],
    )
    gateway.get_client = AsyncMock(return_value=client)

    gtd_expiration = int(datetime.now(timezone.utc).timestamp()) + 600
    order = GatewayOrder(
        attempt_id=uuid4(),
        market_id="market-gtd",
        asset="ETH",
        outcome_to_buy="NO",
        token_id="token-no",
        side="SELL",
        requested_shares=Decimal("5"),
        limit_price=Decimal("0.4"),
        expiration=gtd_expiration,
    )

    submission = await gateway.submit(order, order_type="GTD")

    assert submission.accepted is True
    kwargs = client.place_limit_order.await_args.kwargs
    assert kwargs["token_id"] == "token-no"
    assert kwargs["price"] == "0.4"
    assert kwargs["size"] == "5"
    assert kwargs["side"] == "SELL"
    assert kwargs["post_only"] is False
    assert kwargs["expiration"] == gtd_expiration
    client.place_market_order.assert_not_awaited()


@pytest.mark.asyncio
async def test_gateway_normalizes_post_only_cross_rejection():
    gateway = PolymarketExecutionGateway(
        private_key=KNOWN_PRIVATE_KEY,
        wallet_address="0xDummyAddress",
        relayer_api_key="relayer-key",
        relayer_api_key_address=KNOWN_SIGNER_ADDRESS,
    )
    client = AsyncMock()
    client.place_limit_order.side_effect = Exception(
        "invalid post-only order: order crosses book"
    )
    gateway.get_client = AsyncMock(return_value=client)

    order = GatewayOrder(
        attempt_id=uuid4(), market_id="market-cross", asset="BTC",
        outcome_to_buy="YES", token_id="token-yes", side="BUY",
        requested_shares=Decimal("2"), limit_price=Decimal("0.8"),
        post_only=True,
    )
    submission = await gateway.submit(order, order_type="GTC")

    assert submission.accepted is False
    assert submission.provider_status == "POST_ONLY_REJECTED"
    assert submission.rejection_code == "POST_ONLY_REJECTED"
    assert "POST_ONLY_REJECTED" in (submission.error_message or "")
