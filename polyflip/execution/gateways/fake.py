from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Awaitable, Callable, Mapping

from polyflip.execution.config import POLYMARKET_MIN_ORDER_SHARES
from polyflip.execution.contracts import (
    BalanceResult,
    GatewayOrder,
    GatewayReadiness,
    SubmissionResult,
    TradeExecution,
)


class FakeExecutionGateway:
    """Deterministic PAPER gateway with an optional LIVE_PARITY profile.

    ``INSTANT`` is retained for focused unit tests and legacy callers.  The
    factory uses ``LIVE_PARITY`` in production PAPER mode: it waits for a
    configurable submission delay, consumes the supplied order-book levels,
    enforces the venue minimum size, applies slippage and fees, and returns
    partial fills when available depth is insufficient.
    """

    name = "FAKE"

    def __init__(
        self,
        *,
        profile: str = "INSTANT",
        quote_provider: Callable[[str], Awaitable[Mapping[str, Any]]] | None = None,
        delay_sec: float = 0.0,
        slippage_pct: Decimal | float | str = Decimal("0"),
        fee_rate: Decimal | float | str = Decimal("0"),
        min_order_shares: Decimal | float | str = POLYMARKET_MIN_ORDER_SHARES,
        sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        normalized_profile = str(profile or "INSTANT").strip().upper()
        if normalized_profile not in {"INSTANT", "LIVE_PARITY"}:
            raise ValueError("PAPER profile must be INSTANT or LIVE_PARITY")
        self.profile = normalized_profile
        self.quote_provider = quote_provider
        self.delay_sec = max(0.0, float(delay_sec or 0.0))
        self.slippage_pct = self._decimal(slippage_pct, "0")
        self.fee_rate = self._decimal(fee_rate, "0")
        self.min_order_shares = self._decimal(
            min_order_shares, str(POLYMARKET_MIN_ORDER_SHARES)
        )
        if self.slippage_pct < 0 or self.fee_rate < 0 or self.min_order_shares <= 0:
            raise ValueError("PAPER execution parameters must be non-negative")
        self._sleep = sleep_fn
        self._orders: dict[str, SubmissionResult] = {}

    @staticmethod
    def _decimal(value: Any, default: str = "0") -> Decimal:
        try:
            parsed = Decimal(str(value if value is not None else default))
        except (ArithmeticError, TypeError, ValueError):
            parsed = Decimal(default)
        return parsed if parsed.is_finite() else Decimal(default)

    @staticmethod
    def _levels(value: Any, *, reverse: bool = False) -> list[tuple[Decimal, Decimal]]:
        levels: list[tuple[Decimal, Decimal]] = []
        if not isinstance(value, (list, tuple)):
            return levels
        for raw in value:
            if not isinstance(raw, Mapping):
                continue
            price = FakeExecutionGateway._decimal(raw.get("price"))
            size = FakeExecutionGateway._decimal(
                raw.get("size", raw.get("quantity", raw.get("shares")))
            )
            if price > 0 and size > 0:
                levels.append((price, size))
        return sorted(levels, key=lambda item: item[0], reverse=reverse)

    def _remember(self, result: SubmissionResult) -> SubmissionResult:
        if result.provider_order_id:
            self._orders[result.provider_order_id] = result
        return result

    def _instant_result(self, order: GatewayOrder) -> SubmissionResult:
        now = datetime.now(timezone.utc)
        price = order.limit_price
        shares = order.requested_shares
        trade_id = f"TRADE:{order.attempt_id}"
        fee = (price * shares * self.fee_rate).quantize(Decimal("0.00000001"))
        fill = TradeExecution(
            provider_trade_id=trade_id,
            gateway=self.name,
            gross_quote_usdc=price * shares,
            price=price,
            shares=shares,
            fee_usdc=fee,
            matched_at=now,
        )
        return self._remember(
            SubmissionResult(
                accepted=True,
                provider_order_id=f"PAPER:{order.attempt_id}",
                provider_status="MATCHED",
                provider_trade_ids=(trade_id,),
                settlement_state="CONFIRMED",
                transaction_hashes=(),
                fills=(fill,),
                submitted_limit_price=price,
                submitted_requested_shares=shares,
                paper_quote_price=price,
                paper_available_shares=shares,
                paper_delay_seconds=0.0,
                paper_slippage_usdc=Decimal("0"),
                paper_fee_usdc=fee,
            )
        )

    def _rejected(
        self,
        order: GatewayOrder,
        code: str,
        message: str,
        *,
        quote_price: Decimal | None = None,
        available_shares: Decimal | None = None,
    ) -> SubmissionResult:
        return self._remember(
            SubmissionResult(
                accepted=False,
                provider_order_id=f"PAPER:{order.attempt_id}",
                provider_status=code,
                rejection_code=code,
                error_message=message,
                settlement_state="FAILED",
                submitted_limit_price=order.limit_price,
                submitted_requested_shares=order.requested_shares,
                paper_quote_price=quote_price,
                paper_available_shares=available_shares,
                paper_delay_seconds=self.delay_sec,
            )
        )

    async def submit(
        self, order: GatewayOrder, order_type: str = "FAK"
    ) -> SubmissionResult:
        if self.profile == "INSTANT":
            return self._instant_result(order)

        if order.requested_shares < self.min_order_shares:
            return self._rejected(
                order,
                "PAPER_MIN_ORDER_SHARES",
                f"Order size {order.requested_shares} is below minimum {self.min_order_shares} shares",
            )
        if self.quote_provider is None:
            return self._rejected(
                order,
                "PAPER_QUOTE_UNAVAILABLE",
                "LIVE_PARITY PAPER requires a fresh order-book quote",
            )

        if self.delay_sec > 0:
            await self._sleep(self.delay_sec)
        try:
            quote = await self.quote_provider(order.token_id)
        except Exception as exc:
            return self._rejected(
                order, "PAPER_QUOTE_UNAVAILABLE", f"Order-book request failed: {exc}"
            )
        if not isinstance(quote, Mapping) or quote.get("error"):
            return self._rejected(
                order,
                "PAPER_QUOTE_UNAVAILABLE",
                str((quote or {}).get("error") if isinstance(quote, Mapping) else quote),
            )

        side = order.side.upper()
        if side == "BUY":
            levels = self._levels(quote.get("asks"))
            best_bid = self._decimal(quote.get("best_bid"))
            best_ask = self._decimal(quote.get("best_ask"))
        elif side == "SELL":
            levels = self._levels(quote.get("bids"), reverse=True)
            best_bid = self._decimal(quote.get("best_bid"))
            best_ask = self._decimal(quote.get("best_ask"))
        else:
            return self._rejected(order, "PAPER_INVALID_SIDE", f"Unsupported side: {order.side}")

        if not levels:
            return self._rejected(
                order,
                "PAPER_NO_LIQUIDITY",
                "Fresh order book has no executable depth",
                quote_price=(best_ask if side == "BUY" else best_bid),
                available_shares=Decimal("0"),
            )

        quote_price = levels[0][0]
        crosses = (
            side == "BUY" and order.limit_price >= quote_price
        ) or (side == "SELL" and order.limit_price <= quote_price)
        is_resting = str(order_type or "FAK").upper() in {"GTC", "GTD", "GTC_TTL"} or order.post_only
        if is_resting and crosses:
            return self._rejected(
                order,
                "PAPER_POST_ONLY_CROSSES_BOOK",
                "invalid post-only order: order crosses book",
                quote_price=quote_price,
            )
        if is_resting:
            result = self._remember(
                SubmissionResult(
                    accepted=True,
                    provider_order_id=f"PAPER:{order.attempt_id}",
                    provider_status="RESTING",
                    settlement_state="PENDING",
                    submitted_limit_price=order.limit_price,
                    submitted_requested_shares=order.requested_shares,
                    maker_status="RESTING",
                    maker_best_bid=best_bid,
                    maker_best_ask=best_ask,
                    paper_quote_price=quote_price,
                    paper_available_shares=sum((size for _, size in levels), Decimal("0")),
                    paper_delay_seconds=self.delay_sec,
                )
            )
            return result

        fills: list[TradeExecution] = []
        remaining = order.requested_shares
        remaining_budget = order.max_spend_usdc if side == "BUY" else None
        slippage_factor = self.slippage_pct / Decimal("100")
        total_slippage = Decimal("0")
        total_fee = Decimal("0")
        available = Decimal("0")
        for raw_price, depth in levels:
            if side == "BUY" and raw_price > order.limit_price:
                break
            if side == "SELL" and raw_price < order.limit_price:
                break
            available += depth
            execution_price = raw_price * (
                Decimal("1") + slippage_factor
                if side == "BUY"
                else Decimal("1") - slippage_factor
            )
            if execution_price <= 0:
                continue
            if side == "BUY" and execution_price > order.limit_price:
                break
            if side == "SELL" and execution_price < order.limit_price:
                break
            take = min(remaining, depth)
            if remaining_budget is not None:
                denominator = execution_price * (Decimal("1") + self.fee_rate)
                if denominator > 0:
                    take = min(take, remaining_budget / denominator)
            if take <= 0:
                break
            gross = execution_price * take
            fee = (gross * self.fee_rate).quantize(Decimal("0.00000001"))
            trade_id = f"TRADE:{order.attempt_id}:{len(fills) + 1}"
            fills.append(
                TradeExecution(
                    provider_trade_id=trade_id,
                    gateway=self.name,
                    gross_quote_usdc=gross,
                    price=execution_price,
                    shares=take,
                    fee_usdc=fee,
                    matched_at=datetime.now(timezone.utc),
                )
            )
            total_slippage += abs(execution_price - raw_price) * take
            total_fee += fee
            remaining -= take
            if remaining_budget is not None:
                remaining_budget -= gross + fee
            if remaining <= Decimal("0.0000000001"):
                break

        filled_shares = sum((fill.shares for fill in fills), Decimal("0"))
        if not fills:
            return self._rejected(
                order,
                "NO_LIQUIDITY_FAK",
                "No order-book depth matched the limit price after slippage",
                quote_price=quote_price,
                available_shares=available,
            )
        if str(order_type or "FAK").upper() in {"FOK", "IOC"} and filled_shares < order.requested_shares:
            return self._rejected(
                order,
                "PAPER_PARTIAL_FOK",
                "Available depth was insufficient for a full FOK fill",
                quote_price=quote_price,
                available_shares=available,
            )

        status = "FILLED" if filled_shares >= order.requested_shares else "PARTIALLY_FILLED"
        result = SubmissionResult(
            accepted=True,
            provider_order_id=f"PAPER:{order.attempt_id}",
            provider_status=status,
            provider_trade_ids=tuple(fill.provider_trade_id for fill in fills),
            settlement_state="CONFIRMED",
            fills=tuple(fills),
            submitted_limit_price=order.limit_price,
            submitted_requested_shares=order.requested_shares,
            paper_quote_price=quote_price,
            paper_available_shares=available,
            paper_delay_seconds=self.delay_sec,
            paper_slippage_usdc=total_slippage,
            paper_fee_usdc=total_fee,
        )
        return self._remember(result)

    async def get_order(self, provider_order_id: str) -> SubmissionResult:
        return self._orders.get(
            provider_order_id,
            SubmissionResult(
                accepted=True,
                provider_order_id=provider_order_id,
                provider_status="MATCHED",
                settlement_state="CONFIRMED",
            ),
        )

    async def cancel_order(self, provider_order_id: str) -> SubmissionResult:
        current = self._orders.get(provider_order_id)
        if current is None:
            return SubmissionResult(
                accepted=True,
                provider_order_id=provider_order_id,
                provider_status="CANCELLED",
                settlement_state="CONFIRMED",
            )
        cancelled = current.model_copy(
            update={
                "accepted": False,
                "provider_status": "CANCELLED",
                "settlement_state": "FAILED",
                "error_message": "PAPER resting order cancelled after TTL",
            }
        )
        self._orders[provider_order_id] = cancelled
        return cancelled

    async def get_balance_allowance(
        self, asset_type: str = "COLLATERAL", token_id: str | None = None
    ) -> Decimal:
        return Decimal("1000000.0")

    async def get_token_allowance(self, token_id: str) -> Decimal:
        return Decimal("1000000000")

    async def approve_token(self, token_id: str) -> None:
        pass

    async def fetch_order_fills(
        self, provider_order_id: str, token_id: str, after: str = "0"
    ) -> tuple[TradeExecution, ...]:
        current = self._orders.get(provider_order_id)
        if current and current.fills:
            return current.fills
        # A worker restart creates a new gateway instance. Keep the durable
        # PAPER recovery contract instead of relying only on the in-memory map.
        if not provider_order_id.startswith("PAPER:"):
            return ()
        try:
            import uuid
            from sqlalchemy import select
            from polyflip.db.connection import async_session
            from polyflip.db.execution_models import ExecutionAttempt, ExecutionFill, ExecutionRequest
            attempt_id = uuid.UUID(provider_order_id.split(":", 1)[1])
        except (ImportError, ValueError, TypeError, IndexError):
            return ()

        try:
            async with async_session() as session:
                attempt = await session.scalar(
                    select(ExecutionAttempt).where(ExecutionAttempt.id == attempt_id)
                )
                if attempt is None:
                    return ()
                stored_fills = (
                    await session.execute(
                        select(ExecutionFill)
                        .where(ExecutionFill.attempt_id == attempt.id)
                        .order_by(ExecutionFill.timestamp.asc())
                    )
                ).scalars().all()
                if stored_fills:
                    return tuple(
                        TradeExecution(
                            provider_trade_id=f.provider_trade_id or f"TRADE:{attempt.id}:{index}",
                            gateway=f.gateway or self.name,
                            gross_quote_usdc=Decimal(str(f.gross_quote_usdc or (f.price * f.shares))),
                            price=Decimal(str(f.price)),
                            shares=Decimal(str(f.shares)),
                            fee_usdc=Decimal(str(f.fee_usdc or 0)),
                            matched_at=f.timestamp,
                        )
                        for index, f in enumerate(stored_fills, start=1)
                    )

                raw_response = attempt.raw_response if isinstance(attempt.raw_response, dict) else {}
                paper_telemetry = raw_response.get("paper_telemetry")
                raw_fills = paper_telemetry.get("fills") if isinstance(paper_telemetry, dict) else None
                if isinstance(raw_fills, list):
                    recovered: list[TradeExecution] = []
                    for index, raw_fill in enumerate(raw_fills, start=1):
                        if not isinstance(raw_fill, dict):
                            continue
                        try:
                            price = Decimal(str(raw_fill["price"]))
                            shares = Decimal(str(raw_fill["shares"]))
                            gross = Decimal(str(raw_fill.get("gross_quote_usdc", price * shares)))
                            fee = Decimal(str(raw_fill.get("fee_usdc", "0")))
                            matched_at = datetime.fromisoformat(str(raw_fill["matched_at"]))
                        except (KeyError, TypeError, ValueError, ArithmeticError):
                            continue
                        recovered.append(
                            TradeExecution(
                                provider_trade_id=str(raw_fill.get("provider_trade_id") or f"TRADE:{attempt.id}:{index}"),
                                gateway=self.name,
                                gross_quote_usdc=gross,
                                price=price,
                                shares=shares,
                                fee_usdc=fee,
                                matched_at=matched_at,
                            )
                        )
                    if recovered:
                        return tuple(recovered)

                req = await session.scalar(
                    select(ExecutionRequest).where(ExecutionRequest.id == attempt.request_id)
                )
                if req is None or attempt.status in {"FAILED", "REJECTED"}:
                    return ()
                if str(attempt.provider_status or "").upper() not in {"MATCHED", "FILLED", "PARTIALLY_FILLED"}:
                    return ()
                price_value = req.submitted_limit_price or req.limit_price
                shares_value = req.filled_shares or req.requested_shares
                if price_value is None or shares_value is None:
                    return ()
                price = Decimal(str(price_value))
                shares = Decimal(str(shares_value))
                if price <= 0 or shares <= 0:
                    return ()
                fee = Decimal("0")
                if isinstance(paper_telemetry, dict) and paper_telemetry.get("fee_usdc") is not None:
                    fee = Decimal(str(paper_telemetry["fee_usdc"]))
                gross = price * shares
                return (
                    TradeExecution(
                        provider_trade_id=f"TRADE:{attempt.id}",
                        gateway=self.name,
                        gross_quote_usdc=gross,
                        price=price,
                        shares=shares,
                        fee_usdc=fee,
                        matched_at=attempt.finished_at or datetime.now(timezone.utc),
                    ),
                )
        except Exception:
            return ()

    async def get_readiness(
        self, conditional_token_ids: tuple[str, ...] = ()
    ) -> GatewayReadiness:
        return GatewayReadiness(
            ready=True,
            gateway=self.name,
            wallet_address="0xFAKE",
            balance=BalanceResult(
                balance_usdc=Decimal("1000000.0"), checked_at=datetime.now(timezone.utc)
            ),
            credentials_loaded=True,
            client_initialized=True,
            collateral_allowance_ready=True,
            conditional_allowance_ready=True,
            checked_at=datetime.now(timezone.utc),
        )
