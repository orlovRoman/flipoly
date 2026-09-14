from decimal import Decimal
import pytest

from polyflip.research.lp_rewards.evaluation import determine_gate_b_verdict
from polyflip.research.lp_rewards.ledger import PortfolioLedger
from polyflip.research.lp_rewards.models import MarketPosition, OrderbookLevel


def test_gate_b_verdicts():
    # 1. TARGET_CONFIRMED
    v1 = determine_gate_b_verdict(
        point_est=Decimal("4.50"),
        lower_95=Decimal("3.60"),  # >= 3.50
        upper_95=Decimal("5.40"),
        max_drawdown=Decimal("0.05"),  # <= 0.10
        mean_prediction_error=Decimal("0.15"),  # <= 0.30
    )
    assert v1 == "TARGET_CONFIRMED"

    # 2. TARGET_NOT_CONFIRMED due to drawdown breach (> 10%)
    v2 = determine_gate_b_verdict(
        point_est=Decimal("4.50"),
        lower_95=Decimal("3.60"),
        upper_95=Decimal("5.40"),
        max_drawdown=Decimal("0.12"),  # > 0.10!
        mean_prediction_error=Decimal("0.15"),
    )
    assert v2 == "TARGET_NOT_CONFIRMED"

    # 3. TARGET_NOT_CONFIRMED due to return below target (upper < 3.50)
    v3 = determine_gate_b_verdict(
        point_est=Decimal("2.50"),
        lower_95=Decimal("1.80"),
        upper_95=Decimal("3.20"),  # < 3.50!
        max_drawdown=Decimal("0.04"),
        mean_prediction_error=Decimal("0.10"),
    )
    assert v3 == "TARGET_NOT_CONFIRMED"

    # 4. INCONCLUSIVE (spans across 3.50)
    v4 = determine_gate_b_verdict(
        point_est=Decimal("3.80"),
        lower_95=Decimal("2.90"),  # < 3.50
        upper_95=Decimal("4.70"),  # > 3.50
        max_drawdown=Decimal("0.05"),
        mean_prediction_error=Decimal("0.20"),
    )
    assert v4 == "INCONCLUSIVE"


def test_ledger_executable_mtm_liquidation():
    ledger = PortfolioLedger(allocated_capital=Decimal("100.00"))

    # Position: 50 YES tokens on market m1
    pos = MarketPosition(condition_id="m1", yes_inventory=Decimal("50.0"), cash_invested=Decimal("25.0"))
    positions = {"m1": pos}

    # Bids available for m1_YES: 30 @ 0.48, 40 @ 0.45
    current_bids = {
        "m1_YES": [
            OrderbookLevel(price=Decimal("0.48"), size=Decimal("30.0")),
            OrderbookLevel(price=Decimal("0.45"), size=Decimal("40.0")),
        ]
    }
    # Walking 50 tokens: 30 @ 0.48 ($14.40) + 20 @ 0.45 ($9.00) = $23.40
    mtm = ledger.calculate_executable_mtm(positions, current_bids)
    assert mtm == Decimal("23.40")
