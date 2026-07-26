from datetime import datetime, timezone
from decimal import Decimal
from polyflip.execution.contracts import GatewayOrder, TradeExecution, SubmissionResult, ProviderOrderState

class FakeExecutionGateway:
    name = "FAKE"
    
    async def submit(self, order: GatewayOrder) -> SubmissionResult:
        now = datetime.now(timezone.utc)
        fill = TradeExecution(
            provider_trade_id=f"PAPER:{order.attempt_id}",
            gateway="FAKE",
            gross_quote_usdc=order.limit_price * order.requested_shares,
            price=order.limit_price,
            shares=order.requested_shares,
            fee_usdc=Decimal("0"),
            matched_at=now,
        )
        state = ProviderOrderState(
            provider_order_id=f"PAPER:{order.attempt_id}",
            status="MATCHED",
            filled_shares=order.requested_shares,
            filled_usdc=order.limit_price * order.requested_shares,
            remaining_shares=Decimal("0"),
            fee_usdc=Decimal("0"),
            created_at=now,
            updated_at=now
        )
        return SubmissionResult(
            provider_order_id=f"PAPER:{order.attempt_id}",
            status="MATCHED",
            state=state,
            fills=(fill,),
        )
        
    async def get_order(self, provider_order_id: str) -> SubmissionResult:
        now = datetime.now(timezone.utc)
        state = ProviderOrderState(
            provider_order_id=provider_order_id,
            status="MATCHED",
            filled_shares=Decimal("0"),
            filled_usdc=Decimal("0"),
            remaining_shares=Decimal("0"),
            fee_usdc=Decimal("0"),
            created_at=now,
            updated_at=now
        )
        return SubmissionResult(
            provider_order_id=provider_order_id,
            status="MATCHED",
            state=state
        )
        
    async def get_balance(self) -> Decimal:
        return Decimal("1000000.0")
