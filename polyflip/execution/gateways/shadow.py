"""
SHADOW gateway — симулирует исполнение без отправки реальных ордеров.

Детерминированно генерирует fill на основе attempt_id:
- Идентификаторы зависят только от attempt_id (идемпотентность повторной обработки).
- Заполняет поле SubmissionResult.fills синхронно (worker пропускает fetch_order_fills).
- settlement_state = "CONFIRMED" немедленно (как PAPER/FAKE).
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from polyflip.execution.contracts import (
    BalanceResult,
    GatewayOrder,
    GatewayReadiness,
    SubmissionResult,
    TradeExecution,
)


class ShadowExecutionGateway:
    name = "SHADOW"

    async def submit(self, order: GatewayOrder) -> SubmissionResult:
        order_id = f"SHADOW:{order.attempt_id}"
        trade_id = f"SHADOW-TRADE:{order.attempt_id}"

        fill = TradeExecution(
            provider_trade_id=trade_id,
            gateway=self.name,
            gross_quote_usdc=order.limit_price * order.requested_shares,
            price=order.limit_price,
            shares=order.requested_shares,
            fee_usdc=Decimal("0"),
            matched_at=datetime.now(timezone.utc),
        )

        return SubmissionResult(
            accepted=True,
            provider_order_id=order_id,
            provider_status="MATCHED",
            provider_trade_ids=(trade_id,),
            settlement_state="CONFIRMED",
            fills=(fill,),
        )

    async def get_order(self, provider_order_id: str) -> SubmissionResult:
        return SubmissionResult(
            accepted=True,
            provider_order_id=provider_order_id,
            provider_status="MATCHED",
            settlement_state="CONFIRMED",
        )

    async def fetch_order_fills(
        self,
        provider_order_id: str,
        token_id: str,
        after: str = "0",
    ) -> tuple[TradeExecution, ...]:
        # SHADOW fills уже возвращаются синхронно в submit();
        # этот метод вызывается только при ручном reconcile.
        return ()

    async def get_token_allowance(self, token_id: str) -> Decimal:
        return Decimal("1_000_000_000")

    async def approve_token(self, token_id: str) -> None:
        pass

    async def get_readiness(
        self, conditional_token_ids: tuple[str, ...] = ()
    ) -> GatewayReadiness:
        return GatewayReadiness(
            ready=True,
            gateway=self.name,
            wallet_address="0xSHADOW",
            balance=BalanceResult(
                balance_usdc=Decimal("1_000_000"),
                checked_at=datetime.now(timezone.utc),
            ),
            credentials_loaded=True,
            client_initialized=True,
            collateral_allowance_ready=True,
            conditional_allowance_ready=True,
            checked_at=datetime.now(timezone.utc),
        )
