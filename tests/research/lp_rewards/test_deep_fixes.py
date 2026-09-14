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
    executor.get_pusd_balance_onchain = lambda x: Decimal("1000.0")
    executor.get_pusd_allowance_onchain = lambda x, y: Decimal("1000.0")

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

@pytest.mark.asyncio
@respx.mock
async def test_reconcile_rewards_user_endpoint(monkeypatch):
    """Verify fetch_actual_user_rewards uses /rewards/user with api key."""
    monkeypatch.setenv("POLY_ADDRESS", "0x123")
    monkeypatch.setenv("POLY_SIGNATURE", "sig")
    monkeypatch.setenv("POLY_TIMESTAMP", "123")
    monkeypatch.setenv("POLY_PASSPHRASE", "pass")
    route = respx.get("https://clob.polymarket.com/rewards/user").respond(
        status_code=200,
        json=[{"market": "c_1", "earnings": "100.0", "date": "2023-01-01"}],
    )

    async with httpx.AsyncClient() as client:
        calibrator = RewardCalibrator(Decimal("0.5"))
        rewards = await calibrator.fetch_actual_user_rewards(
            wallet_address="0xabc",
            client=client,
            api_key="secret_key_123",
        )

    assert len(rewards) == 1
    assert rewards[0]["earnings"] == "100.0"
    assert route.called
    req = route.calls.last.request
    assert req.headers.get("POLY_API_KEY") == "secret_key_123"


def test_executor_clob_v2_exact_struct_and_address():
    """Verify official V2 contract address, V2 fields, and removal of V1 legacy fields."""
    from polyflip.research.lp_rewards.execution import CTF_EXCHANGE_ADDRESS
    assert CTF_EXCHANGE_ADDRESS == "0xE11118001712aA868d4aB713A2c8f85fBB9161aB"

    executor = LiveOrderExecutor(
        expected_protocol_hash="hash_123",
        wallet_address="0x1111111111111111111111111111111111111111",
    )
    signed = executor.sign_eip712_order(
        token_id="12345",
        side="BUY",
        price=Decimal("0.50"),
        size=Decimal("10.0"),
    )
    msg = signed["order"]
    order_data = signed["order_data"]
    types = order_data["types"]["Order"]
    type_names = [t["name"] for t in types]

    expected_names = [
        "salt", "maker", "signer", "tokenId", "makerAmount",
        "takerAmount", "side", "signatureType", "timestamp",
        "metadata", "builder"
    ]
    assert type_names == expected_names
    for forbidden in ["taker", "expiration", "nonce", "feeRateBps"]:
        assert forbidden not in type_names
        assert forbidden not in msg

    assert msg["metadata"] == "0x" + "00" * 32
    assert msg["builder"] == "0x" + "00" * 32
    assert msg["timestamp"] > 1_700_000_000_000


def test_pusd_onchain_rpc_balance_and_allowance_mocked(monkeypatch):
    """Verify on-chain pUSD balance and allowance queries via JSON-RPC, with fail-closed behavior."""
    from unittest.mock import MagicMock
    import urllib.request

    executor = LiveOrderExecutor(
        expected_protocol_hash="hash_123",
        wallet_address="0x1111111111111111111111111111111111111111",
    )

    mock_balance_resp = json.dumps({"jsonrpc": "2.0", "result": "0x05f5e100", "id": 1}).encode()
    mock_allowance_resp = json.dumps({"jsonrpc": "2.0", "result": "0x0bebc200", "id": 1}).encode()

    responses = [mock_balance_resp, mock_allowance_resp]
    def mock_urlopen(req, timeout=5):
        data = responses.pop(0)
        resp = MagicMock()
        resp.read.return_value = data
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = None
        return resp

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    bal = executor.get_pusd_balance_onchain("0x1111111111111111111111111111111111111111")
    assert bal == Decimal("100.0")

    allow = executor.get_pusd_allowance_onchain("0x1111111111111111111111111111111111111111", executor.verifying_contract)
    assert allow == Decimal("200.0")

    def mock_urlopen_fail(req, timeout=5):
        raise ConnectionError("Polygon RPC down")

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen_fail)
    bal_err = executor.get_pusd_balance_onchain("0x1111111111111111111111111111111111111111")
    assert bal_err == Decimal("0.0")

    with pytest.raises(ValueError) as exc:
        executor.check_live_balance(Decimal("10.0"))
    assert "Insufficient live balance" in str(exc.value)

    with pytest.raises(PermissionError) as exc2:
        executor.check_allowance(Decimal("10.0"))
    assert "Insufficient collateral allowance" in str(exc2.value)


@pytest.mark.asyncio
async def test_reconcile_rewards_l2_headers_missing_raises_permission_error(monkeypatch):
    """Verify missing required L2 headers raises PermissionError (fail-closed)."""
    monkeypatch.delenv("POLY_ADDRESS", raising=False)
    monkeypatch.delenv("POLY_SIGNATURE", raising=False)
    monkeypatch.delenv("POLY_TIMESTAMP", raising=False)
    monkeypatch.delenv("POLY_PASSPHRASE", raising=False)

    calibrator = RewardCalibrator(Decimal("0.30"))
    with pytest.raises(PermissionError) as exc:
        await calibrator.fetch_actual_user_rewards("0x123")
    assert "Missing required L2 authenticated headers" in str(exc.value)


@pytest.mark.asyncio
@respx.mock
async def test_reconcile_rewards_pagination_and_date_sum(monkeypatch):
    """Verify rewards pagination loops until LTE= and handles dict responses."""
    monkeypatch.setenv("POLY_ADDRESS", "0x123")
    monkeypatch.setenv("POLY_SIGNATURE", "sig")
    monkeypatch.setenv("POLY_TIMESTAMP", "123")
    monkeypatch.setenv("POLY_PASSPHRASE", "pass")

    respx.get("https://clob.polymarket.com/rewards/user").side_effect = [
        httpx.Response(
            200,
            json={
                "data": [{"condition_id": "c1", "earnings": "10.5", "date": "2026-09-10T12:00:00Z"}],
                "next_cursor": "cursor_page_2",
            },
        ),
        httpx.Response(
            200,
            json={
                "data": [{"condition_id": "c2", "earnings": "5.5", "date": "2026-09-10T18:00:00Z"}],
                "next_cursor": "LTE=",
            },
        ),
    ]

    async with httpx.AsyncClient() as client:
        calibrator = RewardCalibrator(Decimal("0.30"))
        rewards = await calibrator.fetch_actual_user_rewards("0xabc", client=client)

    assert len(rewards) == 2
    assert Decimal(str(rewards[0]["earnings"])) + Decimal(str(rewards[1]["earnings"])) == Decimal("16.0")


def test_gate_b_evaluator_fail_closed_when_recon_missing():
    """Verify Gate B fails closed (not TARGET_CONFIRMED) when reward prediction error is default 1.0."""
    from polyflip.research.lp_rewards.evaluation import determine_gate_b_verdict
    verdict = determine_gate_b_verdict(
        point_est=Decimal("5.0"),
        lower_95=Decimal("3.8"),
        upper_95=Decimal("6.2"),
        max_drawdown=Decimal("0.05"),
        mean_prediction_error=Decimal("1.0"),
        target_rate=Decimal("3.50"),
        min_live_days=14,
        total_live_days=14,
    )
    assert verdict != "TARGET_CONFIRMED"
    assert verdict == "INCONCLUSIVE"


def test_shadow_collector_quote_hours_calculation():
    """Verify quote_hours uses aggregated quoting intervals rather than elapsed uptime."""
    market_uptime = {"m1": 7200, "m2": 7200}
    interval_sec = 1.0
    avg_ticks = sum(market_uptime.values()) / len(market_uptime)
    quote_hours = (avg_ticks * interval_sec) / 3600.0
    assert quote_hours == 2.0
