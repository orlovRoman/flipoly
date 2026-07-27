"""
ShadowGateway: логирует все вызовы submit(), но не отправляет ордера в сеть.
Используется как безопасный переходный режим между PAPER и LIVE.
"""
import uuid
import structlog
from decimal import Decimal
from datetime import datetime, timezone

from polyflip.execution.contracts import (
    GatewayOrder,
    GatewayReadiness,
    BalanceResult,
    SubmissionResult,
    TradeExecution,
    GatewayUnavailable,
)

logger = structlog.get_logger(__name__)

_SHADOW_FILL_PRICE_SLIPPAGE = Decimal("0.001")  # 0.1% симулируемый slippage


class ShadowExecutionGateway:
    """
    Shadow gateway: отправка ордеров не производится, все операции
    логируются и возвращают фиктивный успешный результат.

    Назначение: убедиться, что весь контур исполнения работает корректно
    ДО включения реальных LIVE-ордеров.
    """

    name = "SHADOW"

    async def submit(self, order: GatewayOrder) -> SubmissionResult:
        shadow_order_id = f"SHADOW-{uuid.uuid4()}"
        shadow_trade_id = f"SHADOW-TRADE-{uuid.uuid4()}"

        fill_price = order.limit_price * (
            Decimal("1") + _SHADOW_FILL_PRICE_SLIPPAGE
            if order.side.upper() == "BUY"
            else Decimal("1") - _SHADOW_FILL_PRICE_SLIPPAGE
        )

        logger.info(
            "shadow_gateway_submit",
            order_id=shadow_order_id,
            market_id=order.market_id,
            side=order.side,
            token_id=order.token_id,
            requested_shares=str(order.requested_shares),
            limit_price=str(order.limit_price),
            simulated_fill_price=str(fill_price),
        )

        return SubmissionResult(
            accepted=True,
            provider_order_id=shadow_order_id,
            provider_status="MATCHED",
            provider_trade_ids=(shadow_trade_id,),
            settlement_state="CONFIRMED",
            transaction_hashes=(),
        )

    async def get_order(self, provider_order_id: str) -> SubmissionResult:
        logger.info("shadow_gateway_get_order", provider_order_id=provider_order_id)
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
        logger.info(
            "shadow_gateway_fetch_fills",
            provider_order_id=provider_order_id,
        )
        # Возвращаем пустой кортеж — воркер обработает это как RECONCILING
        # и не запишет фиктивные fills. Для SHADOW это допустимо:
        # позиция не откроется, но ошибки не будет.
        return ()

    async def get_token_allowance(self, token_id: str) -> Decimal:
        logger.info("shadow_gateway_get_allowance", token_id=token_id)
        # Shadow всегда "разрешает" — нет реальных ERC-1155 апрувов
        return Decimal("999999")

    async def approve_token(self, token_id: str) -> None:
        logger.info("shadow_gateway_approve_token", token_id=token_id)
        # no-op

    async def get_readiness(
        self,
        conditional_token_ids: tuple[str, ...] = (),
    ) -> GatewayReadiness:
        now = datetime.now(timezone.utc)
        balance = BalanceResult(
            balance_usdc=Decimal("0"),
            collateral_allowances={},
            conditional_allowances_checked=0,
            checked_at=now,
            raw_asset_type="SHADOW",
        )
        return GatewayReadiness(
            ready=True,
            gateway=self.name,
            wallet_address="SHADOW_WALLET",
            balance=balance,
            credentials_loaded=True,
            client_initialized=True,
            collateral_allowance_ready=True,
            conditional_allowance_ready=True,
            checked_at=now,
        )
