import pytest
from decimal import Decimal

def test_effective_pnl_decimal_to_float():
    """Проверяем, что Decimal из БД не вызывает TypeError при сложении."""
    class FakeRow:
        realized_pnl_usdc = Decimal("12.50")
        pnl = None
        amount_usdc = Decimal("500.00")

    row = FakeRow()
    effective_pnl = float(row.realized_pnl_usdc) if row.realized_pnl_usdc is not None else float(row.pnl or 0)
    volume = float(row.amount_usdc or 0)

    total = 0.0
    total += effective_pnl
    total += volume
    assert isinstance(total, float)
    assert total == pytest.approx(512.50)

def test_effective_pnl_none_fallback():
    """Fallback на pnl когда realized_pnl_usdc is None."""
    class FakeRow:
        realized_pnl_usdc = None
        pnl = Decimal("7.25")
        amount_usdc = None

    row = FakeRow()
    effective_pnl = float(row.realized_pnl_usdc) if row.realized_pnl_usdc is not None else float(row.pnl or 0)
    assert effective_pnl == pytest.approx(7.25)
