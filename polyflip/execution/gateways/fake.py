from datetime import datetime, timezone
from decimal import Decimal
from polyflip.execution.contracts import GatewayOrder, ProviderFill, SubmissionResult

class FakeExecutionGateway:
    name = "FAKE"
    
    async def submit(self, order: GatewayOrder) -> SubmissionResult:
        fill = ProviderFill(
            provider_trade_id=f"PAPER:{order.attempt_id}",
            price=order.limit_price,
            shares=order.requested_shares,
            fee_usdc=Decimal("0"),
            matched_at=datetime.now(timezone.utc),
        )
        return SubmissionResult(
            provider_order_id=f"PAPER:{order.attempt_id}",
            status="MATCHED",
            fills=(fill,),
        )
        
    async def get_order(self, provider_order_id: str) -> SubmissionResult:
        return SubmissionResult(
            provider_order_id=provider_order_id,
            status="MATCHED",
        )
        
    async def get_balance(self) -> Decimal:
        return Decimal("1000000.0")
