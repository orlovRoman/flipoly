from polyflip.trading.schemas import TradeExecution, ExecutionFees
from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4


def make_dummy_execution(
    status="FILLED",
    mode="LIVE",
    executed_price=0.5,
    executed_usdc=5.0,
    filled_shares=None,
    error_msg=None,
    attempt_id=None,
):
    if attempt_id is None:
        attempt_id = uuid4()
    fees = ExecutionFees(
        platform_fee_usdc=Decimal("0.0"),
        builder_fee_usdc=Decimal("0.0"),
        network_fee_native=Decimal("0.0"),
        network_fee_symbol="POL",
        network_fee_usdc=Decimal("0.0"),
        fee_source="CONFIRMED_ZERO",
    )
    return TradeExecution(
        attempt_id=attempt_id,
        provider_order_id="dummy_order",
        provider_status="FILLED",
        status="PAPER_FILLED" if mode == "PAPER" else "FILLED",
        side="BUY",
        order_type="FOK",
        token_id="dummy_token",
        original_requested_shares=Decimal(
            str(filled_shares if filled_shares is not None else 20.0)
        ),
        submitted_shares=Decimal(
            str(
                filled_shares
                if filled_shares is not None
                else (0.0 if status in ("FAILED", "REJECTED") else 20.0)
            )
        ),
        filled_shares=Decimal(
            str(
                filled_shares
                if filled_shares is not None
                else (0.0 if status in ("FAILED", "REJECTED") else 20.0)
            )
        ),
        net_position_delta_shares=Decimal(
            str(
                filled_shares
                if filled_shares is not None
                else (0.0 if status in ("FAILED", "REJECTED") else 20.0)
            )
        ),
        average_price=Decimal(str(executed_price)),
        gross_quote_usdc=Decimal(str(executed_usdc)),
        net_quote_usdc=Decimal(str(executed_usdc)),
        liquidity_role="UNKNOWN",
        fees=fees,
        trade_ids=("t1",),
        transaction_hashes=("h1",),
        submitted_at=datetime.now(timezone.utc),
        observed_at=datetime.now(timezone.utc),
        error_message=error_msg,
    )
