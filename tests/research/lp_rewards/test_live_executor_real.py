from decimal import Decimal
import json
from pathlib import Path
import pytest
from eth_account import Account

from polyflip.research.lp_rewards.execution import LiveOrderExecutor


class MockClobClient:
    def __init__(self):
        self.cancelled = False

    def cancel_all_orders(self):
        self.cancelled = True
        return {"status": "OK"}


def test_executor_gate_a_rejection(tmp_path, monkeypatch):
    monkeypatch.setenv("LP_LIVE_ENABLED", "true")

    # Gate A file does not exist
    missing_file = tmp_path / "missing_gate_a.json"
    executor = LiveOrderExecutor(
        clob_client=MockClobClient(),
        expected_protocol_hash="hash_123",
        gate_a_verdict_path=missing_file,
        require_gate_a=True,
    )
    with pytest.raises(PermissionError) as exc:
        executor.submit_order(
            token_id="12345",
            side="BUY",
            price=Decimal("0.50"),
            size=Decimal("10.0"),
            protocol_hash="hash_123",
        )
    assert "Gate A verdict artifact not found" in str(exc.value)

    # Gate A verdict is not PROCEED_LIVE
    gate_a_file = tmp_path / "gate_a_verdict.json"
    gate_a_file.write_text(json.dumps({
        "verdict": "TARGET_REJECTED",
        "protocol_hash": "hash_123",
    }), encoding="utf-8")

    executor = LiveOrderExecutor(
        clob_client=MockClobClient(),
        expected_protocol_hash="hash_123",
        gate_a_verdict_path=gate_a_file,
        require_gate_a=True,
    )
    with pytest.raises(PermissionError) as exc:
        executor.submit_order(
            token_id="12345",
            side="BUY",
            price=Decimal("0.50"),
            size=Decimal("10.0"),
            protocol_hash="hash_123",
        )
    assert "Live trading forbidden" in str(exc.value)


def test_executor_isolated_wallet_rejection(monkeypatch):
    monkeypatch.setenv("LP_LIVE_ENABLED", "true")
    # Missing wallet
    executor = LiveOrderExecutor(
        clob_client=MockClobClient(),
        expected_protocol_hash="hash_123",
        wallet_address="",
        wallet_private_key="",
    )
    with pytest.raises(ValueError) as exc:
        executor.verify_isolated_wallet()
    assert "wallet address is not configured" in str(exc.value)

    # Isolated wallet equals main production wallet
    shared_address = "0x1111111111111111111111111111111111111111"
    executor_shared = LiveOrderExecutor(
        clob_client=MockClobClient(),
        expected_protocol_hash="hash_123",
        wallet_address=shared_address,
        main_wallet_address=shared_address,
    )
    executor_shared.verify_gate_a = lambda: True
    with pytest.raises(PermissionError) as exc:
        executor_shared.submit_order(
            token_id="12345",
            side="BUY",
            price=Decimal("0.50"),
            size=Decimal("10.0"),
            protocol_hash="hash_123",
        )
    assert "matches main production trading wallet" in str(exc.value)


def test_executor_working_capital_limit(monkeypatch):
    monkeypatch.setenv("LP_LIVE_ENABLED", "true")
    executor = LiveOrderExecutor(
        clob_client=MockClobClient(),
        expected_protocol_hash="hash_123",
        allocated_capital_limit=Decimal("100.00"),
    )
    executor.verify_gate_a = lambda: True

    # First order: $80 (price 0.80 * 100 = 80) -> OK
    res1 = executor.submit_order(
        token_id="12345",
        side="BUY",
        price=Decimal("0.80"),
        size=Decimal("100.0"),
        protocol_hash="hash_123",
        live_balance=Decimal("1000.0"),
        allowance=Decimal("1000.0"),
    )
    assert res1["status"] == "SUBMITTED"

    # Second order: $30 (total 80 + 30 = 110 > 100) -> Rejected
    with pytest.raises(ValueError) as exc:
        executor.submit_order(
            token_id="12345",
            side="BUY",
            price=Decimal("0.30"),
            size=Decimal("100.0"),
            protocol_hash="hash_123",
            live_balance=Decimal("1000.0"),
            allowance=Decimal("1000.0"),
        )
    assert "Working capital limit exceeded" in str(exc.value)


def test_executor_token_allowlist_rejection(monkeypatch):
    monkeypatch.setenv("LP_LIVE_ENABLED", "true")
    executor = LiveOrderExecutor(
        clob_client=MockClobClient(),
        expected_protocol_hash="hash_123",
        allowlist_tokens={"approved_tok_a", "approved_tok_b"},
    )
    executor.verify_gate_a = lambda: True
    with pytest.raises(ValueError) as exc:
        executor.submit_order(
            token_id="unapproved_tok_c",
            side="BUY",
            price=Decimal("0.50"),
            size=Decimal("10.0"),
            protocol_hash="hash_123",
        )
    assert "not in approved active universe allowlist" in str(exc.value)


def test_executor_tick_size_and_min_size(monkeypatch):
    monkeypatch.setenv("LP_LIVE_ENABLED", "true")
    executor = LiveOrderExecutor(
        clob_client=MockClobClient(),
        expected_protocol_hash="hash_123",
        min_size=Decimal("10.0"),
        tick_size=Decimal("0.01"),
    )
    executor.verify_gate_a = lambda: True

    # Size too small
    with pytest.raises(ValueError) as exc:
        executor.submit_order(
            token_id="12345",
            side="BUY",
            price=Decimal("0.50"),
            size=Decimal("5.0"),
            protocol_hash="hash_123",
        )
    assert "below minimum allowable size" in str(exc.value)

    # Price invalid tick (0.505 with tick_size 0.01)
    with pytest.raises(ValueError) as exc:
        executor.submit_order(
            token_id="12345",
            side="BUY",
            price=Decimal("0.505"),
            size=Decimal("10.0"),
            protocol_hash="hash_123",
        )
    assert "does not conform to tick size" in str(exc.value)


def test_executor_eip712_signing_and_cancel_all(monkeypatch):
    monkeypatch.setenv("LP_LIVE_ENABLED", "true")
    # Generate real test key
    account = Account.create()
    test_key = account.key.hex()
    test_addr = account.address

    mock_client = MockClobClient()
    executor = LiveOrderExecutor(
        clob_client=mock_client,
        expected_protocol_hash="hash_123",
        wallet_private_key=test_key,
        wallet_address=test_addr,
    )
    executor.verify_gate_a = lambda: True
    executor.get_pusd_balance_onchain = lambda x: Decimal("1000.0")
    executor.get_pusd_allowance_onchain = lambda x, y: Decimal("1000.0")

    res = executor.submit_order(
        token_id="12345",
        side="BUY",
        price=Decimal("0.45"),
        size=Decimal("20.0"),
        protocol_hash="hash_123",
    )
    assert res["status"] == "SUBMITTED"
    assert "signed_order" in res
    assert res["signed_order"]["signature"].startswith("0x")
    assert len(res["signed_order"]["signature"]) > 2
    assert res["signed_order"]["order"]["maker"] == test_addr

    # Test cancel all
    assert len(executor.submitted_orders) == 1
    assert executor.cancel_all_orders() is True
    assert len(executor.submitted_orders) == 0
    assert executor.current_committed_capital == Decimal("0.0")
    assert mock_client.cancelled is True
