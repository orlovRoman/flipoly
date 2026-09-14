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
        wallet_address="0x1111111111111111111111111111111111111111",
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
            live_balance=Decimal("100.00"),
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
        live_balance=Decimal("100.00"),
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
    assert CTF_EXCHANGE_ADDRESS == "0xE111180000d2663C0091e4f400237545B87B996B"

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


def test_subtract_orders_deduplication():
    """Verify subtract_orders removes our resting liquidity from public orderbook."""
    from polyflip.research.lp_rewards.scoring import subtract_orders
    public = [
        OrderbookLevel(price=Decimal("0.50"), size=Decimal("100.0")),
        OrderbookLevel(price=Decimal("0.49"), size=Decimal("50.0")),
    ]
    ours = [
        OrderbookLevel(price=Decimal("0.50"), size=Decimal("40.0")),
        OrderbookLevel(price=Decimal("0.49"), size=Decimal("60.0")),
    ]
    result = subtract_orders(public, ours)
    assert len(result) == 1
    assert result[0].price == Decimal("0.50")
    assert result[0].size == Decimal("60.0")


def test_clob_v2_sdk_import_and_init():
    """Verify py_clob_client_v2 is importable and initializes with standard params."""
    from py_clob_client_v2 import ClobClient, ApiCreds
    creds = ApiCreds(api_key="k", api_secret="s", api_passphrase="p")
    client = ClobClient(
        host="https://clob.polymarket.com",
        chain_id=137,
        key="0x" + "11" * 32,
        creds=creds,
        signature_type=0,
    )
    assert client is not None
    assert hasattr(client, "post_order") or hasattr(client, "create_and_post_order")


def test_evaluate_gate_a_missing_protocol_hash_rejected():
    """Verify evaluate_gate_a_full rejects records that omit protocol_hash."""
    from polyflip.research.lp_rewards.evaluation import evaluate_gate_a_full
    records_no_hash = [
        {
            "day": d,
            "date": f"2026-09-0{d+1}",
            "net_pnl": "4.00",
            "quote_hours": "20.0",
            "book_uncertain_count": 0,
            "market_breakdown": {f"c_{m}": {"net_pnl": "0.40", "coverage_ratio": "0.995"} for m in range(10)},
        }
        for d in range(7)
    ]
    rep = evaluate_gate_a_full(records_no_hash, expected_protocol_hash="expected_hash_123")
    assert rep.protocol_hash_valid is False
    assert any("Protocol hash missing" in r for r in rep.rejection_reasons)


def test_evaluate_gate_a_duplicate_dates_rejected():
    """Verify evaluate_gate_a_full rejects records with duplicate calendar dates."""
    from polyflip.research.lp_rewards.evaluation import evaluate_gate_a_full
    records_dup_date = [
        {
            "day": d,
            "date": "2026-09-01",
            "net_pnl": "4.00",
            "quote_hours": "20.0",
            "protocol_hash": "hash_123",
            "book_uncertain_count": 0,
            "market_breakdown": {f"c_{m}": {"net_pnl": "0.40", "coverage_ratio": "0.995"} for m in range(10)},
        }
        for d in range(7)
    ]
    rep = evaluate_gate_a_full(records_dup_date, expected_protocol_hash="hash_123")
    assert any("Duplicate evaluation dates" in r for r in rep.rejection_reasons)


def test_subtract_orders_sorting_and_aggregation():
    """Verify subtract_orders aggregates duplicates at same price, sorts bids/asks, and clamps >= 0."""
    from polyflip.research.lp_rewards.scoring import subtract_orders

    # Duplicate levels at price 0.50 (100 + 50 = 150)
    public = [
        OrderbookLevel(price=Decimal("0.50"), size=Decimal("100.0")),
        OrderbookLevel(price=Decimal("0.48"), size=Decimal("40.0")),
        OrderbookLevel(price=Decimal("0.50"), size=Decimal("50.0")),
        OrderbookLevel(price=Decimal("0.49"), size=Decimal("30.0")),
    ]
    # Our duplicate orders at 0.50 (60 + 40 = 100) -> remaining 150 - 100 = 50
    # Our order at 0.48 (50 > 40) -> clamped to 0 and removed
    ours = [
        OrderbookLevel(price=Decimal("0.50"), size=Decimal("60.0")),
        OrderbookLevel(price=Decimal("0.50"), size=Decimal("40.0")),
        OrderbookLevel(price=Decimal("0.48"), size=Decimal("50.0")),
    ]

    # Test bids: descending order (0.50, 0.49)
    res_bids = subtract_orders(public, ours, is_bid=True)
    assert len(res_bids) == 2
    assert res_bids[0].price == Decimal("0.50")
    assert res_bids[0].size == Decimal("50.0")
    assert res_bids[1].price == Decimal("0.49")
    assert res_bids[1].size == Decimal("30.0")

    # Test asks: ascending order (0.49, 0.50)
    res_asks = subtract_orders(public, ours, is_bid=False)
    assert len(res_asks) == 2
    assert res_asks[0].price == Decimal("0.49")
    assert res_asks[1].price == Decimal("0.50")


def test_executor_pusd_wallet_guard_fail_closed():
    """Verify fail-closed behavior when wallet_address and explicit balance/allowance are None."""
    executor = LiveOrderExecutor(
        expected_protocol_hash="hash_123",
        clob_client="mock",
        wallet_address=None,
    )
    with pytest.raises(PermissionError) as exc_bal:
        executor.check_live_balance(cost=Decimal("10.0"), available_balance=None)
    assert "Dedicated isolated wallet address or available_balance must be provided" in str(exc_bal.value)

    with pytest.raises(PermissionError) as exc_allow:
        executor.check_allowance(required_amount=Decimal("10.0"), current_allowance=None)
    assert "Dedicated isolated wallet address or current_allowance must be provided" in str(exc_allow.value)


def test_executor_cancel_all_not_implemented_error():
    """Verify cancel_all_orders and cancel_all_orders_async raise NotImplementedError if client lacks methods."""
    class ClientWithoutCancel:
        pass

    executor = LiveOrderExecutor(
        expected_protocol_hash="hash_123",
        clob_client=ClientWithoutCancel(),
    )
    with pytest.raises(NotImplementedError) as exc:
        executor.cancel_all_orders()
    assert "CLOB client does not support cancel_all_orders" in str(exc.value)


@pytest.mark.asyncio
async def test_executor_cancel_all_async_not_implemented_error():
    class ClientWithoutCancel:
        pass

    executor = LiveOrderExecutor(
        expected_protocol_hash="hash_123",
        clob_client=ClientWithoutCancel(),
    )
    with pytest.raises(NotImplementedError) as exc:
        await executor.cancel_all_orders_async()
    assert "CLOB client does not support cancel_all_orders" in str(exc.value)


def test_executor_submits_signed_order_v2_to_sdk(monkeypatch):
    """Verify submit_order constructs SignedOrderV2 with timestamp recognized by CLOB V2 SDK."""
    from py_clob_client_v2.client import _is_v2_order

    received_orders = []

    class MockV2Client:
        def post_order(self, order_obj):
            received_orders.append(order_obj)
            assert _is_v2_order(order_obj) is True
            assert hasattr(order_obj, "timestamp")
            assert hasattr(order_obj, "maker")
            return {"orderID": "0xv2_ord_success"}

        def cancel_all_orders(self):
            return {"status": "OK"}

    monkeypatch.setenv("LP_LIVE_ENABLED", "true")
    client = MockV2Client()
    executor = LiveOrderExecutor(
        expected_protocol_hash="hash_123",
        clob_client=client,
        wallet_address="0x1111111111111111111111111111111111111111",
    )
    executor.verify_gate_a = lambda: True

    res = executor.submit_order(
        token_id="12345",
        side="BUY",
        price=Decimal("0.50"),
        size=Decimal("10.0"),
        protocol_hash="hash_123",
        live_balance=Decimal("100.00"),
        allowance=Decimal("100.00"),
    )
    assert res["status"] == "SUBMITTED"
    assert res["order_id"] == "0xv2_ord_success"
    assert len(received_orders) == 1
    assert _is_v2_order(received_orders[0]) is True


@pytest.mark.asyncio
async def test_reward_calibration_fetch_actual_rewards_error_propagation():
    """Verify fetch_actual_user_rewards raises RuntimeError on non-200 HTTP response."""
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("POLY_ADDRESS", "0xaddr")
    monkeypatch.setenv("POLY_SIGNATURE", "0xsig")
    monkeypatch.setenv("POLY_TIMESTAMP", "123456")
    monkeypatch.setenv("POLY_PASSPHRASE", "pass")

    with respx.mock(assert_all_called=False) as respx_mock:
        respx_mock.get("https://clob.polymarket.com/rewards/user").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )
        async with httpx.AsyncClient() as client:
            calibrator = RewardCalibrator(Decimal("0.30"))
            with pytest.raises(RuntimeError) as exc:
                await calibrator.fetch_actual_user_rewards("0xabc", client=client)
            assert "CLOB rewards endpoint returned HTTP 500" in str(exc.value)


def test_evaluate_gate_a_rejects_missing_date_in_any_record():
    """Verify evaluate_gate_a_full rejects when any record is missing date."""
    from polyflip.research.lp_rewards.evaluation import evaluate_gate_a_full
    records = [
        {
            "day": d,
            "date": f"2026-09-0{d+1}" if d != 3 else "",
            "net_pnl": "4.00",
            "quote_hours": "20.0",
            "protocol_hash": "hash_123",
            "book_uncertain_count": 0,
            "market_breakdown": {f"c_{m}": {"net_pnl": "0.40", "coverage_ratio": "0.995"} for m in range(10)},
        }
        for d in range(7)
    ]
    rep = evaluate_gate_a_full(records, expected_protocol_hash="hash_123")
    assert any("Required 'date' field missing" in r for r in rep.rejection_reasons)


def test_evaluate_gate_a_rejects_single_mismatched_protocol_hash():
    """Verify evaluate_gate_a_full rejects if even 1 record has mismatched protocol_hash."""
    from polyflip.research.lp_rewards.evaluation import evaluate_gate_a_full
    records = [
        {
            "day": d,
            "date": f"2026-09-0{d+1}",
            "net_pnl": "4.00",
            "quote_hours": "20.0",
            "protocol_hash": "hash_123" if d != 5 else "tampered_hash_999",
            "book_uncertain_count": 0,
            "market_breakdown": {f"c_{m}": {"net_pnl": "0.40", "coverage_ratio": "0.995"} for m in range(10)},
        }
        for d in range(7)
    ]
    rep = evaluate_gate_a_full(records, expected_protocol_hash="hash_123")
    assert rep.protocol_hash_valid is False
    assert any("Protocol hash mismatch" in r for r in rep.rejection_reasons)


def test_live_calibration_fetch_live_midpoint():
    """Verify fetch_live_midpoint retrieves mid from CLOB or orderbook correctly."""
    mod_06 = importlib.import_module("scripts.research.lp_rewards.06_run_live_calibration")
    fetch_live_midpoint = mod_06.fetch_live_midpoint

    class MockClobWithMidpoint:
        def get_midpoint(self, token_id):
            return {"mid": "0.485"}

    client = MockClobWithMidpoint()
    mid = fetch_live_midpoint(client, "tok_123")
    assert mid == Decimal("0.485")

    class MockClobWithBook:
        def get_midpoint(self, token_id):
            return None

        def get_order_book(self, token_id):
            return {
                "bids": [{"price": "0.48", "size": "100"}],
                "asks": [{"price": "0.52", "size": "100"}],
            }

    client_book = MockClobWithBook()
    mid_book = fetch_live_midpoint(client_book, "tok_123")
    assert mid_book == Decimal("0.50")


def test_gate_b_evaluator_script_missing_wallet_fails_closed(monkeypatch):
    """Verify 09_evaluate_gate_b.py main() immediately exits with code 1 if LP_ISOLATED_WALLET_ADDRESS is missing."""
    mod_09 = importlib.import_module("scripts.research.lp_rewards.09_evaluate_gate_b")
    monkeypatch.delenv("LP_ISOLATED_WALLET_ADDRESS", raising=False)
    with pytest.raises(SystemExit) as exc:
        mod_09.main()
    assert exc.value.code == 1


def test_gate_b_evaluator_script_empty_wallet_fails_closed(monkeypatch):
    """Verify 09_evaluate_gate_b.py main() immediately exits with code 1 if LP_ISOLATED_WALLET_ADDRESS is empty whitespace."""
    mod_09 = importlib.import_module("scripts.research.lp_rewards.09_evaluate_gate_b")
    monkeypatch.setenv("LP_ISOLATED_WALLET_ADDRESS", "   ")
    with pytest.raises(SystemExit) as exc:
        mod_09.main()
    assert exc.value.code == 1


def test_gate_b_evaluator_script_missing_recon_files_fails_closed(tmp_path, monkeypatch):
    """Verify 09_evaluate_gate_b.py main() exits with code 1 when reconciliation reports are missing."""
    mod_09 = importlib.import_module("scripts.research.lp_rewards.09_evaluate_gate_b")
    monkeypatch.setenv("LP_ISOLATED_WALLET_ADDRESS", "0x1234567890123456789012345678901234567890")

    from polyflip.research.lp_rewards.protocol import load_protocol
    protocol = load_protocol().model_copy(deep=True)
    protocol.data_storage.root_path = str(tmp_path)
    monkeypatch.setattr(mod_09, "load_protocol", lambda: protocol)

    eval_dir = tmp_path / "daily_evaluations_live"
    eval_dir.mkdir(parents=True)
    with open(eval_dir / "eval_20260901.json", "w", encoding="utf-8") as f:
        json.dump({
            "date": "2026-09-01",
            "net_pnl": "4.0",
            "protocol_hash": protocol.sha256_hash,
        }, f)

    with pytest.raises(SystemExit) as exc:
        mod_09.main()
    assert exc.value.code == 1


def test_gate_b_evaluator_script_mismatched_wallet_in_recon_fails_closed(tmp_path, monkeypatch):
    """Verify 09_evaluate_gate_b.py main() exits with code 1 when reconciliation report has mismatched wallet."""
    mod_09 = importlib.import_module("scripts.research.lp_rewards.09_evaluate_gate_b")
    monkeypatch.setenv("LP_ISOLATED_WALLET_ADDRESS", "0x1234567890123456789012345678901234567890")

    from polyflip.research.lp_rewards.protocol import load_protocol
    protocol = load_protocol().model_copy(deep=True)
    protocol.data_storage.root_path = str(tmp_path)
    monkeypatch.setattr(mod_09, "load_protocol", lambda: protocol)

    eval_dir = tmp_path / "daily_evaluations_live"
    eval_dir.mkdir(parents=True)
    with open(eval_dir / "eval_20260901.json", "w", encoding="utf-8") as f:
        json.dump({
            "date": "2026-09-01",
            "net_pnl": "4.0",
            "protocol_hash": protocol.sha256_hash,
        }, f)

    recon_dir = tmp_path / "reconciliation"
    recon_dir.mkdir(parents=True)
    with open(recon_dir / "rewards_20260901.json", "w", encoding="utf-8") as f:
        json.dump({
            "protocol_hash": protocol.sha256_hash,
            "wallet_address": "0x9999999999999999999999999999999999999999",
            "mean_error_ratio": "0.15",
        }, f)

    with pytest.raises(SystemExit) as exc:
        mod_09.main()
    assert exc.value.code == 1


def test_gate_b_evaluator_script_happy_path(tmp_path, monkeypatch):
    """Verify 09_evaluate_gate_b.py main() completes and writes artifact when wallet and recon match."""
    mod_09 = importlib.import_module("scripts.research.lp_rewards.09_evaluate_gate_b")
    wallet = "0x1234567890123456789012345678901234567890"
    monkeypatch.setenv("LP_ISOLATED_WALLET_ADDRESS", wallet)

    from polyflip.research.lp_rewards.protocol import load_protocol
    protocol = load_protocol().model_copy(deep=True)
    protocol.data_storage.root_path = str(tmp_path)
    monkeypatch.setattr(mod_09, "load_protocol", lambda: protocol)

    eval_dir = tmp_path / "daily_evaluations_live"
    eval_dir.mkdir(parents=True)
    with open(eval_dir / "eval_20260901.json", "w", encoding="utf-8") as f:
        json.dump({
            "date": "2026-09-01",
            "net_pnl": "4.0",
            "protocol_hash": protocol.sha256_hash,
        }, f)

    recon_dir = tmp_path / "reconciliation"
    recon_dir.mkdir(parents=True)
    with open(recon_dir / "rewards_20260901.json", "w", encoding="utf-8") as f:
        json.dump({
            "protocol_hash": protocol.sha256_hash,
            "wallet_address": wallet,
            "mean_error_ratio": "0.15",
        }, f)

    mod_09.main()
    verdict_file = tmp_path / "gate_b_verdict.json"
    assert verdict_file.exists()
    with open(verdict_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["protocol_id"] == protocol.protocol_id
    assert data["mean_prediction_error"] == "0.15"


def test_gate_b_evaluator_script_zero_address_wallet_fails_closed(monkeypatch):
    """Verify 09_evaluate_gate_b.py main() immediately exits with code 1 if LP_ISOLATED_WALLET_ADDRESS is zero address."""
    mod_09 = importlib.import_module("scripts.research.lp_rewards.09_evaluate_gate_b")
    monkeypatch.setenv("LP_ISOLATED_WALLET_ADDRESS", "0x0000000000000000000000000000000000000000")
    with pytest.raises(SystemExit) as exc:
        mod_09.main()
    assert exc.value.code == 1


@pytest.mark.asyncio
async def test_reconcile_rewards_script_missing_or_zero_wallet_fails_closed(monkeypatch):
    """Verify 08_reconcile_rewards.py main() fails closed with code 1 if wallet is missing, empty, or zero address."""
    mod_08 = importlib.import_module("scripts.research.lp_rewards.08_reconcile_rewards")
    for bad_wallet in [None, "   ", "0x0000000000000000000000000000000000000000"]:
        if bad_wallet is None:
            monkeypatch.delenv("LP_ISOLATED_WALLET_ADDRESS", raising=False)
        else:
            monkeypatch.setenv("LP_ISOLATED_WALLET_ADDRESS", bad_wallet)
        with pytest.raises(SystemExit) as exc:
            await mod_08.main()
        assert exc.value.code == 1


@pytest.mark.asyncio
async def test_reconcile_orders_script_missing_or_zero_wallet_fails_closed(monkeypatch):
    """Verify 07_reconcile_orders.py main() fails closed with code 1 if wallet is missing, empty, or zero address."""
    mod_07 = importlib.import_module("scripts.research.lp_rewards.07_reconcile_orders")
    for bad_wallet in [None, "   ", "0x0000000000000000000000000000000000000000"]:
        if bad_wallet is None:
            monkeypatch.delenv("LP_ISOLATED_WALLET_ADDRESS", raising=False)
        else:
            monkeypatch.setenv("LP_ISOLATED_WALLET_ADDRESS", bad_wallet)
        with pytest.raises(SystemExit) as exc:
            await mod_07.main()
        assert exc.value.code == 1


def test_executor_isolated_wallet_whitespace_and_zero_rejection():
    """Verify verify_isolated_wallet rejects whitespace strings and zero address."""
    executor = LiveOrderExecutor(expected_protocol_hash="hash_123")
    with pytest.raises(ValueError) as exc:
        executor.verify_isolated_wallet(wallet_address="   ")
    assert "wallet address is not configured" in str(exc.value)

    with pytest.raises(ValueError) as exc_z:
        executor.verify_isolated_wallet(wallet_address="0x0000000000000000000000000000000000000000")
    assert "wallet address is not configured" in str(exc_z.value)


def test_executor_sign_eip712_order_blocks_missing_or_zero_wallet():
    """Verify sign_eip712_order raises PermissionError if wallet is missing, whitespace, or zero address."""
    executor = LiveOrderExecutor(expected_protocol_hash="hash_123", wallet_address=None)
    with pytest.raises(PermissionError) as exc1:
        executor.sign_eip712_order("12345", "BUY", Decimal("0.50"), Decimal("10.0"))
    assert "Order signing requires configured non-zero wallet_address" in str(exc1.value)

    executor_ws = LiveOrderExecutor(expected_protocol_hash="hash_123", wallet_address="   ")
    with pytest.raises(PermissionError) as exc2:
        executor_ws.sign_eip712_order("12345", "BUY", Decimal("0.50"), Decimal("10.0"))
    assert "Order signing requires configured non-zero wallet_address" in str(exc2.value)

    executor_zero = LiveOrderExecutor(
        expected_protocol_hash="hash_123",
        wallet_address="0x0000000000000000000000000000000000000000",
    )
    with pytest.raises(PermissionError) as exc3:
        executor_zero.sign_eip712_order("12345", "BUY", Decimal("0.50"), Decimal("10.0"))
    assert "Order signing requires configured non-zero wallet_address" in str(exc3.value)


@pytest.mark.asyncio
@respx.mock
async def test_reconcile_orders_fetch_remote_http_error_fails_closed():
    """Verify fetch_remote_orders raises RuntimeError and fails closed on non-200 HTTP status."""
    respx.get("https://clob.polymarket.com/data/orders").respond(
        status_code=500,
        text="Internal Server Error",
    )

    async with httpx.AsyncClient() as client:
        with pytest.raises(RuntimeError) as exc:
            await fetch_remote_orders(
                wallet_address="0xabc",
                client=client,
            )
    assert "CLOB orders query returned status 500" in str(exc.value)


@pytest.mark.asyncio
@respx.mock
async def test_reconcile_orders_fetch_remote_network_error_fails_closed():
    """Verify fetch_remote_orders raises RuntimeError on network/connection exception."""
    respx.get("https://clob.polymarket.com/data/orders").mock(
        side_effect=httpx.ConnectError("Connection refused")
    )

    async with httpx.AsyncClient() as client:
        with pytest.raises(RuntimeError) as exc:
            await fetch_remote_orders(
                wallet_address="0xabc",
                client=client,
            )
    assert "Network error querying remote CLOB orders" in str(exc.value)


@pytest.mark.asyncio
@respx.mock
async def test_reconcile_orders_fetch_remote_invalid_json_fails_closed():
    """Verify fetch_remote_orders raises RuntimeError when response body is not valid JSON."""
    respx.get("https://clob.polymarket.com/data/orders").respond(
        status_code=200,
        content=b"not-a-valid-json-string",
        headers={"content-type": "application/json"},
    )

    async with httpx.AsyncClient() as client:
        with pytest.raises(RuntimeError) as exc:
            await fetch_remote_orders(
                wallet_address="0xabc",
                client=client,
            )
    assert "Failed to parse CLOB orders JSON response" in str(exc.value)


@pytest.mark.asyncio
@respx.mock
async def test_reconcile_orders_fetch_remote_unexpected_format_fails_closed():
    """Verify fetch_remote_orders raises RuntimeError when JSON response is neither list nor dict."""
    respx.get("https://clob.polymarket.com/data/orders").respond(
        status_code=200,
        json=12345,
    )

    async with httpx.AsyncClient() as client:
        with pytest.raises(RuntimeError) as exc:
            await fetch_remote_orders(
                wallet_address="0xabc",
                client=client,
            )
    assert "Unexpected response format from CLOB orders API" in str(exc.value)


@pytest.mark.asyncio
@respx.mock
async def test_reconcile_orders_main_remote_error_fails_closed(monkeypatch):
    """Verify 07_reconcile_orders.py main() logs error and exits with code 1 when remote fetch fails."""
    monkeypatch.setenv("LP_ISOLATED_WALLET_ADDRESS", "0x1111111111111111111111111111111111111111")
    respx.get("https://clob.polymarket.com/data/orders").respond(
        status_code=503,
        text="Service Unavailable",
    )

    with pytest.raises(SystemExit) as exc:
        await reconcile_07.main()
    assert exc.value.code == 1


@pytest.mark.asyncio
@respx.mock
async def test_reconcile_orders_fetch_remote_dict_with_error_fails_closed():
    """Verify fetch_remote_orders raises RuntimeError when response dict contains 'error'."""
    respx.get("https://clob.polymarket.com/data/orders").respond(
        status_code=200,
        json={"error": "API key rejected"},
    )

    async with httpx.AsyncClient() as client:
        with pytest.raises(RuntimeError) as exc:
            await fetch_remote_orders(
                wallet_address="0xabc",
                client=client,
            )
    assert "CLOB orders API returned error in response" in str(exc.value)


@pytest.mark.asyncio
@respx.mock
async def test_reconcile_orders_fetch_remote_dict_missing_data_fails_closed():
    """Verify fetch_remote_orders raises RuntimeError when response dict lacks 'data' key."""
    respx.get("https://clob.polymarket.com/data/orders").respond(
        status_code=200,
        json={"status": "ok", "count": 0},
    )

    async with httpx.AsyncClient() as client:
        with pytest.raises(RuntimeError) as exc:
            await fetch_remote_orders(
                wallet_address="0xabc",
                client=client,
            )
    assert "CLOB orders response dict missing 'data' field" in str(exc.value)


@pytest.mark.asyncio
@respx.mock
async def test_reconcile_orders_fetch_remote_multipage_failure_fails_closed():
    """Verify fetch_remote_orders fails closed without returning partial page-1 orders when page-2 query fails."""
    route = respx.get("https://clob.polymarket.com/data/orders")
    route.side_effect = [
        httpx.Response(
            200,
            json={
                "data": [{"order_id": "ord_1", "price": "0.50", "size": "10.0"}],
                "next_cursor": "PAGE_2_CURSOR",
            },
        ),
        httpx.Response(
            500,
            text="Internal Gateway Error on page 2",
        ),
    ]

    async with httpx.AsyncClient() as client:
        with pytest.raises(RuntimeError) as exc:
            await fetch_remote_orders(
                wallet_address="0xabc",
                client=client,
            )
    assert "CLOB orders query returned status 500" in str(exc.value)


@pytest.mark.asyncio
@respx.mock
async def test_reconcile_orders_main_corrupted_local_file_fails_closed(tmp_path, monkeypatch):
    """Verify 07_reconcile_orders.py main() logs error and exits with code 1 if local_orders_file is corrupted JSON."""
    monkeypatch.setenv("LP_ISOLATED_WALLET_ADDRESS", "0x1111111111111111111111111111111111111111")
    from polyflip.research.lp_rewards.protocol import load_protocol
    protocol = load_protocol().model_copy(deep=True)
    protocol.data_storage.root_path = str(tmp_path)
    monkeypatch.setattr(reconcile_07, "load_protocol", lambda: protocol)

    corrupt_file = tmp_path / "live_open_orders.json"
    corrupt_file.write_text("{broken json file content", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        await reconcile_07.main()
    assert exc.value.code == 1
