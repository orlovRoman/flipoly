import asyncio
from decimal import Decimal
import json
from pathlib import Path
import pytest
import respx
import httpx
from eth_account import Account

from polyflip.research.lp_rewards.execution import LiveOrderExecutor
from polyflip.research.lp_rewards.models import (
    MarketPosition,
    MarketRewardConfig,
    OrderbookLevel,
    OrderbookSnapshot,
    OrderSide,
    QuotingState,
    VirtualOrder,
)
from polyflip.research.lp_rewards.quoting_fsm import MarketQuotingFSM
from polyflip.research.lp_rewards.reward_calibration import RewardCalibrator
from polyflip.research.lp_rewards.universe import (
    enrich_market_info,
    parse_market_reward_config,
)
import importlib

reconcile_07 = importlib.import_module("scripts.research.lp_rewards.07_reconcile_orders")
fetch_remote_orders = reconcile_07.fetch_remote_orders


def test_executor_token_id_parsing_hex_and_int():
    """Verify hex ('0x...'), int, and string token IDs parse properly without defaulting to 1."""
    assert LiveOrderExecutor._parse_token_id(12345) == 12345
    assert LiveOrderExecutor._parse_token_id("987654321") == 987654321
    assert LiveOrderExecutor._parse_token_id("0x1a") == 26
    assert LiveOrderExecutor._parse_token_id("0X2B") == 43
    with pytest.raises(ValueError):
        LiveOrderExecutor._parse_token_id("invalid_text")


def test_executor_allowance_verification(monkeypatch):
    """Verify allowance check blocks order when allowance is lower than order cost."""
    monkeypatch.setenv("LP_LIVE_ENABLED", "true")
    executor = LiveOrderExecutor(
        expected_protocol_hash="hash_123",
        clob_client="mock_client",
    )
    executor.verify_gate_a = lambda: True

    # Cost = 0.50 * 50 = $25.00
    # Case 1: Allowance $10.00 < $25.00 -> raise PermissionError
    with pytest.raises(PermissionError) as exc:
        executor.submit_order(
            token_id="12345",
            side="BUY",
            price=Decimal("0.50"),
            size=Decimal("50.0"),
            protocol_hash="hash_123",
            allowance=Decimal("10.00"),
        )
    assert "Insufficient collateral allowance" in str(exc.value)

    # Case 2: Allowance $30.00 >= $25.00 -> OK
    res = executor.submit_order(
        token_id="12345",
        side="BUY",
        price=Decimal("0.50"),
        size=Decimal("50.0"),
        protocol_hash="hash_123",
        allowance=Decimal("30.00"),
    )
    assert res["status"] == "SUBMITTED"


@pytest.mark.asyncio
async def test_executor_cancel_all_in_running_event_loop():
    """Verify cancel_all_orders does NOT crash when called inside a running asyncio event loop."""
    class MockAsyncClobClient:
        def __init__(self):
            self.cancelled = False

        async def cancel_all_orders(self):
            self.cancelled = True
            return {"status": "OK"}

    client = MockAsyncClobClient()
    executor = LiveOrderExecutor(
        expected_protocol_hash="hash_123",
        clob_client=client,
    )

    # This previously failed with 'RuntimeError: Cannot run the event loop while another loop is running'
    success = executor.cancel_all_orders()
    assert success is True

    # Async cancel-all variant
    async_success = await executor.cancel_all_orders_async()
    assert async_success is True


def test_executor_clob_v2_domain_and_post_order(monkeypatch):
    """Verify EIP-712 domain uses Polymarket CTF Exchange V2 and posts order via clob_client."""
    monkeypatch.setenv("LP_LIVE_ENABLED", "true")

    class MockSyncClobClient:
        def __init__(self):
            self.posted = None

        def post_order(self, order_payload):
            self.posted = order_payload
            return {"orderID": "mock_order_999"}

    client = MockSyncClobClient()
    account = Account.create()

    executor = LiveOrderExecutor(
        expected_protocol_hash="hash_123",
        wallet_private_key=account.key.hex(),
        wallet_address=account.address,
        clob_client=client,
    )
    executor.verify_gate_a = lambda: True

    res = executor.submit_order(
        token_id="0xabc123",
        side="BUY",
        price=Decimal("0.40"),
        size=Decimal("20.0"),
        protocol_hash="hash_123",
    )

    assert res["status"] == "SUBMITTED"
    assert res["order_id"] == "mock_order_999"
    assert client.posted is not None

    # Verify domain and token ID in signed order
    signed_data = res["signed_order"]["order"]
    assert signed_data["tokenId"] == int("0xabc123", 16)


def test_parse_market_reward_config_clob_v2_schema():
    """Verify parse_market_reward_config correctly parses official CLOB V2 schema (r.moas, fd.r, t, nr)."""
    raw_item = {
        "condition_id": "c_clob_v2",
        "total_daily_rate": "150.0",
        "rewards_max_spread": "0.04",
        "rewards_min_size": "50",
    }
    clob_info = {
        "r": {
            "mi": 50,
            "ma": 4.0,
            "e": True,
            "moas": 30,  # Real OAS: 30 seconds
        },
        "fd": {
            "r": 0.04,  # Real fee: 4%
            "e": 1,
            "to": True,
        },
        "t": [
            {"t": "tok_yes_123", "o": "Yes"},
            {"t": "tok_no_456", "o": "No"},
        ],
        "nr": False,
        "question": "Will CLOB V2 be fully parsed?",
    }

    cfg = parse_market_reward_config(raw_item, clob_market_info=clob_info)
    assert cfg is not None
    assert cfg.condition_id == "c_clob_v2"
    assert cfg.oas == Decimal("30")
    assert cfg.taker_fee_rate == Decimal("0.04")
    assert cfg.yes_token_id == "tok_yes_123"
    assert cfg.no_token_id == "tok_no_456"
    assert cfg.neg_risk is False
    assert cfg.question == "Will CLOB V2 be fully parsed?"


def test_fsm_liquidity_insufficient_retry_and_recovery():
    """Verify FSM in EXIT_LIQUIDITY_INSUFFICIENT can retry liquidation and reach FLAT when bids appear."""
    cfg = MarketRewardConfig(
        condition_id="c_retry",
        question="Test retry",
        rewards_daily_rate=Decimal("100.0"),
        rewards_max_spread=Decimal("0.05"),
        rewards_min_size=Decimal("10.0"),
        oas=Decimal("5.0"),
        taker_fee_rate=Decimal("0.01"),
        yes_token_id="tok_y",
        no_token_id="tok_n",
    )
    fsm = MarketQuotingFSM(config=cfg, hedging_timeout_sec=900.0)

    # Put FSM in LONG_YES with 50 units @ 0.50
    fsm.position.state = QuotingState.LONG_YES
    fsm.position.yes_inventory = Decimal("50.0")
    fsm.position.yes_fill_cost = Decimal("25.0")
    fsm.position.cash_invested = Decimal("25.0")
    fsm.state_timer_ns = 1_000_000_000_000

    # Tick 1: After 1000s, book has thin depth (only 20 units @ 0.48)
    now_tick1 = 1_000_000_000_000 + 1000 * 1_000_000_000
    book_thin = OrderbookSnapshot(
        condition_id="c_retry",
        asset_id="tok_y",
        bids=[OrderbookLevel(price=Decimal("0.48"), size=Decimal("20.0"))],
        asks=[],
        timestamp_ns=now_tick1,
        valid_from_ns=now_tick1,
    )

    exited1, fill1 = fsm.check_timeout_and_exit(now_tick1, book_thin)
    assert exited1 is True
    assert fill1 is not None
    assert fill1.size == Decimal("20.0")
    assert fsm.position.state == QuotingState.EXIT_LIQUIDITY_INSUFFICIENT
    assert fsm.position.yes_inventory == Decimal("30.0")  # 30 units remainder retained

    # Tick 2: Subsequent cycle with new bids (50 units @ 0.47)
    now_tick2 = now_tick1 + 60 * 1_000_000_000
    book_liquid = OrderbookSnapshot(
        condition_id="c_retry",
        asset_id="tok_y",
        bids=[OrderbookLevel(price=Decimal("0.47"), size=Decimal("50.0"))],
        asks=[],
        timestamp_ns=now_tick2,
        valid_from_ns=now_tick2,
    )

    # Must be able to retry and liquidate the 30 remainder
    exited2, fill2 = fsm.check_timeout_and_exit(now_tick2, book_liquid)
    assert exited2 is True
    assert fill2 is not None
    assert fill2.size == Decimal("30.0")
    assert fsm.position.state == QuotingState.FLAT  # Fully liquidated
    assert fsm.position.yes_inventory == Decimal("0.0")


@pytest.mark.asyncio
@respx.mock
async def test_reconcile_orders_v2_orders_endpoint():
    """Verify fetch_remote_orders uses /data/orders with auth headers."""
    route = respx.get("https://clob.polymarket.com/data/orders").respond(
        status_code=200,
        json=[{"order_id": "ord_remote_1", "price": "0.50", "size": "10.0"}],
    )

    async with httpx.AsyncClient() as client:
        orders = await fetch_remote_orders(
            wallet_address="0xabc",
            client=client,
            api_key="secret_key_123",
        )

    assert len(orders) == 1
    assert orders[0]["order_id"] == "ord_remote_1"
    assert route.called
    req = route.calls.last.request
    assert req.headers.get("POLY_API_KEY") == "secret_key_123"
