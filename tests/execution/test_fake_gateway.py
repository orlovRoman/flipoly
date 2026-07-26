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
    assert res.status == "MATCHED"
    assert res.provider_order_id == f"PAPER:{attempt_id}"
    assert len(res.fills) == 1
    
    fill = res.fills[0]
    assert fill.price == Decimal("0.5")
    assert fill.shares == Decimal("100.0")
    assert fill.fee_usdc == Decimal("0")
    
@pytest.mark.asyncio
async def test_fake_execution_gateway_get_order():
    gateway = FakeExecutionGateway()
    res = await gateway.get_order("PAPER:123")
    assert res.status == "MATCHED"
    assert res.provider_order_id == "PAPER:123"

@pytest.mark.asyncio
async def test_fake_execution_gateway_get_balance():
    gateway = FakeExecutionGateway()
    bal = await gateway.get_balance()
    assert bal == Decimal("1000000.0")
