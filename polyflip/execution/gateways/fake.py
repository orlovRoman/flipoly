from datetime import datetime, timezone
from decimal import Decimal
from polyflip.execution.contracts import GatewayOrder, TradeExecution, SubmissionResult, GatewayReadiness, BalanceResult

class FakeExecutionGateway:
    name = "FAKE"
    
    async def submit(self, order: GatewayOrder) -> SubmissionResult:
        return SubmissionResult(
            accepted=True,
            provider_order_id=f"PAPER:{order.attempt_id}",
            provider_status="MATCHED",
            provider_trade_ids=(f"TRADE:{order.attempt_id}",)
        )
        
    async def get_order(self, provider_order_id: str) -> SubmissionResult:
        return SubmissionResult(
            accepted=True,
            provider_order_id=provider_order_id,
            provider_status="MATCHED"
        )
        
    async def get_balance_allowance(self, asset_type: str = "COLLATERAL", token_id: str | None = None) -> Decimal:
        return Decimal("1000000.0")
        
    async def fetch_order_fills(self, provider_order_id: str, token_id: str, after: str = "0") -> tuple[TradeExecution, ...]:
        from polyflip.db.connection import async_session
        from sqlalchemy import select
        from polyflip.db.execution_models import ExecutionAttempt, ExecutionRequest
        import uuid
        
        # In PAPER mode, provider_order_id = PAPER:attempt_id
        if not provider_order_id.startswith("PAPER:"):
            return ()
            
        try:
            attempt_id = uuid.UUID(provider_order_id.split(":")[1])
        except Exception:
            return ()
            
        async with async_session() as session:
            attempt = (await session.execute(
                select(ExecutionAttempt).where(ExecutionAttempt.id == attempt_id)
            )).scalar_one_or_none()
            
            if not attempt:
                return ()
                
            req = (await session.execute(
                select(ExecutionRequest).where(ExecutionRequest.id == attempt.request_id)
            )).scalar_one_or_none()
            
            if not req:
                return ()
                
            price = Decimal(str(req.limit_price or "0"))
            shares = Decimal(str(req.requested_shares or "0"))
            
            return (TradeExecution(
                provider_trade_id=f"TRADE:{attempt.id}",
                gateway=self.name,
                gross_quote_usdc=price * shares,
                price=price,
                shares=shares,
                fee_usdc=Decimal("0"),
                matched_at=datetime.now(timezone.utc)
            ),)

    async def get_readiness(self, conditional_token_ids: tuple[str, ...] = ()) -> GatewayReadiness:
        return GatewayReadiness(
            ready=True,
            gateway=self.name,
            wallet_address="0xFAKE",
            balance=BalanceResult(
                balance_usdc=Decimal("1000000.0"),
                checked_at=datetime.now(timezone.utc)
            ),
            credentials_loaded=True,
            client_initialized=True,
            collateral_allowance_ready=True,
            conditional_allowance_ready=True,
            checked_at=datetime.now(timezone.utc)
        )
