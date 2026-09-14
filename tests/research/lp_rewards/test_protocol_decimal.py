from decimal import Decimal, getcontext
from pathlib import Path
import pytest

from polyflip.research.lp_rewards.protocol import load_protocol, compute_file_sha256


def test_protocol_loading_and_decimal_precision():
    protocol = load_protocol()
    assert protocol.version == "0.1.0"
    assert protocol.hypothesis.target_r100_calendar == Decimal("3.50")
    assert protocol.capital_allocation.allocated_working_capital == Decimal("100.00")
    assert protocol.capital_allocation.max_unhedged_per_market == Decimal("25.00")
    assert protocol.capital_allocation.max_unhedged_total == Decimal("50.00")
    assert getcontext().prec >= 28


def test_protocol_sha256_verification():
    protocol = load_protocol()
    assert protocol.sha256_hash is not None
    assert len(protocol.sha256_hash) == 64
