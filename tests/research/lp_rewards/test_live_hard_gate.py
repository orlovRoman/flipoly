from decimal import Decimal
import os
import pytest

from polyflip.research.lp_rewards.execution import LiveOrderExecutor


def test_live_hard_gate_blocks_when_env_disabled(monkeypatch):
    monkeypatch.setenv("LP_LIVE_ENABLED", "false")
    executor = LiveOrderExecutor(clob_client='mock', expected_protocol_hash="valid_hash_123")

    with pytest.raises(PermissionError) as exc_info:
        executor.submit_order(
            token_id="12345",
            side="BUY",
            price=Decimal("0.50"),
            size=Decimal("10.0"),
            protocol_hash="valid_hash_123",
        )
    assert "LP Live trading is disabled" in str(exc_info.value)


def test_live_hard_gate_blocks_protocol_hash_mismatch(monkeypatch):
    monkeypatch.setenv("LP_LIVE_ENABLED", "true")
    executor = LiveOrderExecutor(clob_client='mock', expected_protocol_hash="approved_hash_abc")
    executor.verify_gate_a = lambda: True

    with pytest.raises(ValueError) as exc_info:
        executor.submit_order(
            token_id="12345",
            side="BUY",
            price=Decimal("0.50"),
            size=Decimal("10.0"),
            protocol_hash="different_unapproved_hash",
        )
    assert "Protocol hash mismatch" in str(exc_info.value)


def test_live_order_succeeds_when_all_gates_pass(monkeypatch):
    monkeypatch.setenv("LP_LIVE_ENABLED", "true")
    executor = LiveOrderExecutor(
        clob_client='mock',
        expected_protocol_hash="approved_hash_abc",
        wallet_address="0x1111111111111111111111111111111111111111",
    )
    executor.verify_gate_a = lambda: True

    res = executor.submit_order(
        token_id="12345",
        side="BUY",
        price=Decimal("0.50"),
        size=Decimal("10.0"),
        protocol_hash="approved_hash_abc",
        live_balance=Decimal("100.00"),
        allowance=Decimal("100.00"),
    )
    assert res["status"] == "SUBMITTED"


def test_live_order_blocks_when_wallet_missing(monkeypatch):
    monkeypatch.setenv("LP_LIVE_ENABLED", "true")
    executor = LiveOrderExecutor(
        clob_client='mock',
        expected_protocol_hash="approved_hash_abc",
        wallet_address=None,
    )
    executor.verify_gate_a = lambda: True

    with pytest.raises(PermissionError) as exc_info:
        executor.submit_order(
            token_id="12345",
            side="BUY",
            price=Decimal("0.50"),
            size=Decimal("10.0"),
            protocol_hash="approved_hash_abc",
            live_balance=Decimal("100.00"),
            allowance=Decimal("100.00"),
        )
    assert "Live execution requires configured non-zero wallet_address" in str(exc_info.value)


def test_live_order_blocks_when_wallet_is_zero_address(monkeypatch):
    monkeypatch.setenv("LP_LIVE_ENABLED", "true")
    executor = LiveOrderExecutor(
        clob_client='mock',
        expected_protocol_hash="approved_hash_abc",
        wallet_address="0x0000000000000000000000000000000000000000",
    )
    executor.verify_gate_a = lambda: True

    with pytest.raises(PermissionError) as exc_info:
        executor.submit_order(
            token_id="12345",
            side="BUY",
            price=Decimal("0.50"),
            size=Decimal("10.0"),
            protocol_hash="approved_hash_abc",
            live_balance=Decimal("100.00"),
            allowance=Decimal("100.00"),
        )
    assert "Live execution requires configured non-zero wallet_address" in str(exc_info.value)
