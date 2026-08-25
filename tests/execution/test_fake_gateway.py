import pytest
from uuid import uuid4
from decimal import Decimal
from polyflip.execution.gateways.fake import FakeExecutionGateway
from polyflip.execution.contracts import GatewayOrder

@pytest.mark.asyncio
async def test_fake_execution_gateway_submit():
    gateway = FakeExecutionGateway()
    attempt_id = uuid4()
    
    order = GatewayOrder(
        attempt_id=attempt_id,
        market_id="0x123",
        asset="BTC",
        outcome_to_buy="YES",
        token_id="0x456",
        side="BUY",
        limit_price=Decimal("0.5"),
        requested_shares=Decimal("100.0")
    )
    
    res = await gateway.submit(order)
    assert res.provider_status == "MATCHED"
    assert res.provider_order_id == f"PAPER:{attempt_id}"
    assert len(res.provider_trade_ids) == 1
    
@pytest.mark.asyncio
async def test_fake_execution_gateway_get_order():
    gateway = FakeExecutionGateway()
    res = await gateway.get_order("PAPER:123")
    assert res.provider_status == "MATCHED"
    assert res.provider_order_id == "PAPER:123"

@pytest.mark.asyncio
async def test_fake_execution_gateway_get_balance_allowance():
    gateway = FakeExecutionGateway()
    bal = await gateway.get_balance_allowance("COLLATERAL")
    assert bal == Decimal("1000000.0")


def _parity_order(*, shares: str = "10", price: str = "0.50", post_only: bool = False):
    return GatewayOrder(
        attempt_id=uuid4(),
        market_id="market",
        asset="BTC",
        outcome_to_buy="YES",
        token_id="token",
        side="BUY",
        limit_price=Decimal(price),
        requested_shares=Decimal(shares),
        max_spend_usdc=Decimal("10"),
        post_only=post_only,
    )


@pytest.mark.asyncio
async def test_paper_live_parity_consumes_depth_and_applies_costs():
    quote_calls = []
    sleeps = []

    async def quote_provider(token_id):
        quote_calls.append(token_id)
        return {
            "best_bid": 0.29,
            "best_ask": 0.30,
            "asks": [
                {"price": 0.30, "size": 4},
                {"price": 0.40, "size": 10},
            ],
            "bids": [{"price": 0.29, "size": 20}],
        }

    async def sleep_fn(seconds):
        sleeps.append(seconds)

    gateway = FakeExecutionGateway(
        profile="LIVE_PARITY",
        quote_provider=quote_provider,
        delay_sec=2.0,
        slippage_pct="0.5",
        fee_rate="0.002",
        sleep_fn=sleep_fn,
    )
    result = await gateway.submit(_parity_order(), order_type="FAK")

    assert result.accepted is True
    assert result.provider_status == "FILLED"
    assert sum((fill.shares for fill in result.fills), Decimal("0")) == Decimal("10")
    assert result.paper_quote_price == Decimal("0.3")
    assert result.paper_available_shares == Decimal("14")
    assert result.paper_slippage_usdc is not None and result.paper_slippage_usdc > 0
    assert result.paper_fee_usdc is not None and result.paper_fee_usdc > 0
    assert quote_calls == ["token"]
    assert sleeps == [2.0]


@pytest.mark.asyncio
async def test_paper_live_parity_returns_partial_fill_when_depth_is_short():
    async def quote_provider(token_id):
        return {
            "best_bid": 0.29,
            "best_ask": 0.30,
            "asks": [{"price": 0.30, "size": 6}],
            "bids": [{"price": 0.29, "size": 10}],
        }

    gateway = FakeExecutionGateway(
        profile="LIVE_PARITY", quote_provider=quote_provider, slippage_pct=0, fee_rate=0
    )
    result = await gateway.submit(_parity_order(shares="10"), order_type="FAK")

    assert result.accepted is True
    assert result.provider_status == "PARTIALLY_FILLED"
    assert sum((fill.shares for fill in result.fills), Decimal("0")) == Decimal("6")


@pytest.mark.asyncio
async def test_paper_live_parity_rejects_small_resting_order_before_quote():
    quote_calls = []

    async def quote_provider(token_id):
        quote_calls.append(token_id)
        return {}

    gateway = FakeExecutionGateway(profile="LIVE_PARITY", quote_provider=quote_provider)
    result = await gateway.submit(_parity_order(shares="4"), order_type="GTC")

    assert result.accepted is False
    assert result.rejection_code == "PAPER_MIN_ORDER_SHARES"
    assert quote_calls == []


@pytest.mark.asyncio
async def test_paper_live_parity_allows_small_fak_retry_order():
    async def quote_provider(token_id):
        return {
            "best_bid": 0.29,
            "best_ask": 0.30,
            "asks": [{"price": 0.30, "size": 4}],
            "bids": [{"price": 0.29, "size": 10}],
        }

    gateway = FakeExecutionGateway(
        profile="LIVE_PARITY", quote_provider=quote_provider, slippage_pct=0, fee_rate=0
    )
    result = await gateway.submit(_parity_order(shares="4"), order_type="FAK")

    assert result.accepted is True
    assert result.provider_status == "FILLED"
    assert sum((fill.shares for fill in result.fills), Decimal("0")) == Decimal("4")


@pytest.mark.asyncio
async def test_paper_live_parity_rejects_crossing_post_only_order():
    async def quote_provider(token_id):
        return {
            "best_bid": 0.29,
            "best_ask": 0.30,
            "asks": [{"price": 0.30, "size": 10}],
            "bids": [{"price": 0.29, "size": 10}],
        }

    gateway = FakeExecutionGateway(profile="LIVE_PARITY", quote_provider=quote_provider)
    result = await gateway.submit(
        _parity_order(price="0.30", post_only=True), order_type="GTC"
    )

    assert result.accepted is False
    assert result.rejection_code == "PAPER_POST_ONLY_CROSSES_BOOK"
