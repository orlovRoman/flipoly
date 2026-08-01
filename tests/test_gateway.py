import pytest
import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, patch, MagicMock

from polyflip.execution.gateways.polymarket import PolymarketExecutionGateway
from polyflip.execution.contracts import GatewayOrder
from polyflip.db.execution_models import ExecutionRequest


@pytest.mark.asyncio
async def test_gateway_full_cycle():
    """
    Тест должен проходить полный цикл: submit (ACCEPTED) -> fills (2 fills, sum=requested) -> reconcile -> FILLED
    """
    gateway = PolymarketExecutionGateway(
        private_key="0x0000000000000000000000000000000000000000000000000000000000000001",
        wallet_address="0xtest",
        relayer_api_key="relayer_key",
        relayer_api_key_address="0x7E5F4552091A69125d5DfCb7b8C2659029395Bdf",
        host="https://gamma-api.polymarket.com",
    )

    mock_client = AsyncMock()
    gateway._client_cache = mock_client

    # 1. submit
    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.order_id = "order_123"
    mock_resp.status = "ACCEPTED"
    mock_resp.trade_ids = []

    mock_client.place_market_order.return_value = mock_resp

    order = GatewayOrder(
        attempt_id="00000000-0000-0000-0000-000000000000",
        market_id="market_123",
        asset="BTC",
        outcome_to_buy="YES",
        side="BUY",
        token_id="token_123",
        max_spend_usdc=Decimal("10.0"),
        limit_price=Decimal("0.5"),
        requested_shares=Decimal("20.0"),
    )

    submit_result = await gateway.submit(order)
    assert submit_result.accepted is True
    assert submit_result.provider_order_id == "order_123"
    assert submit_result.provider_status == "ACCEPTED"

    # 2. fills (2 fills, sum=requested)
    class MockTrade:
        def __init__(self, tid, t_oid, price, size, matched_at):
            self.id = tid
            self.taker_order_id = t_oid
            self.price = price
            self.size = size
            self.status = "CONFIRMED"
            self.fee_rate_bps = 0
            self.matched_at = matched_at
            self.maker_orders = []

    class MockPage:
        def __init__(self, items):
            self.items = items

    async def mock_list_account_trades(*args, **kwargs):
        yield MockPage(
            [
                MockTrade(
                    "trade_1", "order_123", 0.5, 10.0, datetime.now(timezone.utc)
                ),
                MockTrade(
                    "trade_2", "order_123", 0.5, 10.0, datetime.now(timezone.utc)
                ),
            ]
        )

    mock_client.list_account_trades = mock_list_account_trades

    fills = await gateway.fetch_order_fills("order_123", "token_123")
    assert len(fills) == 2

    total_shares = sum(f.shares for f in fills)
    assert total_shares == Decimal("20.0")

    # 3. reconcile mock response
    mock_client.get_order.return_value = {"status": "FILLED"}
    reconcile_res = await gateway.get_order("order_123")
    assert reconcile_res.provider_status == "FILLED"
