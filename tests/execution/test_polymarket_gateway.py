import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from decimal import Decimal
from datetime import datetime, timezone
from polyflip.execution.gateways.polymarket import PolymarketExecutionGateway
from polyflip.execution.contracts import GatewayOrder

@pytest.mark.asyncio
async def test_gateway_uses_supported_sdk_methods():
    # Setup
    gateway = PolymarketExecutionGateway()
    client_mock = AsyncMock()
    
    # Mocking SDK methods
    client_mock.place_market_order = AsyncMock(return_value=MagicMock(ok=True, order_id="123", status="FILLED"))
    client_mock.get_order = AsyncMock(return_value={"status": "FILLED", "size_matched": "100", "price": "0.5"})
    client_mock.get_balance_allowance = AsyncMock(return_value=[{"asset_type": "collateral", "balance": "1000000000"}])
    
    # Bypass get_client
    gateway.get_client = AsyncMock(return_value=client_mock)
    
    # Test submit BUY
    from uuid import uuid4
    order = GatewayOrder(
        attempt_id=uuid4(), market_id="m1", asset="Yes", outcome_to_buy="Yes",
        token_id="0x123", side="BUY", requested_shares=Decimal("10"), 
        max_spend_usdc=Decimal("5"), limit_price=Decimal("0.5")
    )
    res = await gateway.submit(order)
    assert res.status == "FILLED"
    client_mock.place_market_order.assert_called_once_with(
        token_id="0x123", side="BUY", amount=5.0, max_spend=5.0, max_price=0.5, order_type="FAK"
    )
    
    # Test get_order
    res_order = await gateway.get_order("123")
    assert res_order.status == "FILLED"
    assert len(res_order.fills) == 1
    client_mock.get_order.assert_called_once_with(order_id="123")
    
    # Test balance
    bal = await gateway.get_balance()
    assert bal == Decimal("1000000000")
    client_mock.get_balance_allowance.assert_called_once_with(asset_type="COLLATERAL")
