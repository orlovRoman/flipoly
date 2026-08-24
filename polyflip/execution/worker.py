import asyncio
import argparse
import sys
import structlog
import os
import socket
import ssl
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Optional

from sqlalchemy import select, or_, and_, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.db.connection import async_session
from polyflip.db.execution_models import (
    ExecutionRequest,
    ExecutionAttempt,
    ExecutionFill,
    ExecutionWorkerStatus,
)
from polyflip.db.models import LiveMarket, TradeHistory, RuntimeSettings
from polyflip.execution.order_strategies import execute_gtc_ttl, execute_fak_retry, execute_maker_limit
from polyflip.execution.config import (
    POLYMARKET_MIN_ORDER_SHARES,
    ExecutionSettings,
)
from polyflip.execution.gateways.factory import build_execution_gateway
from polyflip.execution.contracts import GatewayOrder, GatewayUnavailable
from polyflip.execution.gateways.exceptions import (
    GatewayOrderRejected,
    GatewaySubmissionUnknown,
)
from polyflip.execution.outbox import enqueue_close_request, finalize_request
from polyflip.execution.states import (
    ACTIVE_REQUEST_STATES,
    FAILURE_TERMINAL_STATES,
    RECONCILABLE_REQUEST_STATES,
)
from polyflip.execution.risk_checks import check_risk_limits

logger = structlog.get_logger(__name__)

# Advisory lock namespace: 2001 — один lock на режим исполнения,
# а не на рынок, чтобы глобальные лимиты не обходились параллельно.
_MODE_LOCK_KEYS = {
    "PAPER": 1,
    "SHADOW": 2,
    "LIVE": 3,
}
_ADVISORY_LOCK_NAMESPACE = 2001


def _resolve_requested_shares(
    *,
    requested_shares: Decimal | None,
    max_spend_usdc: Decimal | None,
    limit_price: Decimal | None,
    side: str,
) -> Decimal:
    """Return a positive BUY size when a request omitted its token quantity.

    GTC/GTD orders are size-based at the Polymarket CLOB. Some release-gate
    requests intentionally leave requested_shares empty because they only
    carry a USDC budget. Derive that size from the same limit price used for
    the order, while keeping SELL requests and invalid budgets fail-closed.
    """
    if requested_shares is not None and requested_shares > 0:
        return requested_shares

    spend = max_spend_usdc or Decimal("0")
    price = limit_price or Decimal("0")
    if side.upper() == "BUY" and spend > 0 and price > 0:
        return spend / price
    return Decimal("0")


# Через сколько секунд неопределённого состояния переходим в MANUAL_REVIEW_REQUIRED
async def _enqueue_gtd_take_profit_after_fill(session, req) -> None:
    """Place a native GTD TP order after a LIVE BUY is fully filled.

    The close request is created in the same transaction as the BUY
    accounting update.  This prevents a scheduler race from creating a
    second TP request before the position becomes visible as EXIT_REQUESTED.
    """
    if req.intent != "OPEN" or req.state != "FILLED":
        return
    if str(req.requested_mode).upper() != "LIVE":
        return

    mode_row = await session.scalar(
        select(RuntimeSettings.value).where(
            RuntimeSettings.key == "TAKE_PROFIT_ORDER_MODE"
        )
    )
    if str(mode_row or "GTD").strip().upper() != "GTD":
        return

    trade = await session.get(TradeHistory, req.trade_history_id, with_for_update=True)
    if not trade or not trade.take_profit_enabled:
        return
    # Both the fill path and reconciliation can observe the same BUY fill.
    # Once a TP request is queued or terminal, this helper must be idempotent.
    if trade.take_profit_status in {"QUEUED", "FILLED", "EXPIRED"}:
        return
    if trade.position_status == "CLOSED" or not trade.take_profit_price:
        return
    remaining = Decimal(str(trade.remaining_shares or 0))
    if remaining <= 0:
        return

    market_end = trade.market_end_time
    if market_end is None:
        market_end = await session.scalar(
            select(LiveMarket.end_time_est).where(
                LiveMarket.market_id == trade.market_id
            )
        )
    if market_end is None:
        logger.warning("gtd_take_profit_missing_market_end", trade_id=trade.id)
        return
    if market_end.tzinfo is None:
        market_end = market_end.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    if market_end <= now + timedelta(seconds=180):
        # Polymarket rejects GTD expirations inside its three-minute window.
        # Leave the position ACTIVE so the legacy trigger worker can use a
        # marketable close when the target is actually reached.
        logger.info(
            "gtd_take_profit_too_close_to_expiry",
            trade_id=trade.id,
            market_end=market_end.isoformat(),
        )
        return

    result = await enqueue_close_request(
        session,
        trade_id=trade.id,
        trigger_reason="TAKE_PROFIT",
        limit_price=float(trade.take_profit_price),
        expires_at=market_end,
    )
    if result.disposition.value == "CREATED":
        trade.take_profit_status = "QUEUED"
        trade.take_profit_sell_price = trade.take_profit_price
        logger.info(
            "gtd_take_profit_request_created",
            trade_id=trade.id,
            request_id=str(result.request_id),
            limit_price=trade.take_profit_price,
            expires_at=market_end.isoformat(),
        )


MAX_RECONCILIATION_AGE_SEC = 900  # 15 минут


async def _get_dialect(session) -> str:
    """
    Возвращает имя диалекта БД через connection, а не session.bind.
    AsyncSession.bind всегда None при async_sessionmaker — использование
    session.bind.dialect.name вызывает AttributeError в продакшене.
    """
    conn = await session.connection()
    return conn.dialect.name


async def _acquire_mode_lock(session, requested_mode: str) -> None:
    """
    Берёт PostgreSQL advisory lock на уровне режима исполнения.
    Гарантирует, что глобальные лимиты (MAX_OPEN_POSITIONS, MAX_TOTAL_EXPOSURE)
    проверяются и изменяются атомарно: два воркера одного режима не могут
    одновременно пройти risk-check.
    На SQLite — no-op (тесты).
    """
    # Bug #3 fix: получаем диалект через connection, не через session.bind
    dialect_name = await _get_dialect(session)
    if dialect_name != "postgresql":
        return
    mode_key = _MODE_LOCK_KEYS.get(requested_mode, 0)
    await session.execute(
        text("SELECT pg_advisory_xact_lock(:namespace, :mode_key)"),
        {"namespace": _ADVISORY_LOCK_NAMESPACE, "mode_key": mode_key},
    )


async def _persist_fills(
    session,
    attempt: ExecutionAttempt,
    fills,
) -> None:
    """
    Сохраняет fills идемпотентно через ON CONFLICT DO NOTHING
    по (gateway, provider_trade_id).
    Совместимо с PostgreSQL и SQLite (index_elements вместо constraint=).
    """
    # Bug #4 fix: используем _get_dialect вместо session.bind.dialect.name
    dialect_name = await _get_dialect(session)
    insert_func = sqlite_insert if dialect_name == "sqlite" else pg_insert

    for f in fills:
        stmt = insert_func(ExecutionFill).values(
            attempt_id=attempt.id,
            provider_trade_id=f.provider_trade_id,
            gateway=f.gateway,
            gross_quote_usdc=f.gross_quote_usdc,
            price=f.price,
            shares=f.shares,
            fee_usdc=f.fee_usdc,
            timestamp=f.matched_at,
        )

        # В PostgreSQL явно ссылаемся на constraint, который создаётся
        # миграцией c4d5e6f7a8b9. Так drift между ORM и production-схемой
        # обнаруживается сразу и не маскируется несовпадающим conflict target.
        if dialect_name == "postgresql":
            stmt = stmt.on_conflict_do_nothing(constraint="uq_execution_provider_trade")
        else:
            stmt = stmt.on_conflict_do_nothing(
                index_elements=["gateway", "provider_trade_id"],
            )
        await session.execute(stmt)


async def _finish_submit_exception(
    session,
    *,
    request_id,
    attempt_id,
    attempt_no: int,
    requested_mode: str,
    error: str,
    is_deterministic_rejection: bool = False,
) -> None:
    """
    Восстанавливает сессию после любой ошибки submit/persist/accounting.

    SQL-ошибка переводит транзакцию SQLAlchemy в failed state. Поэтому перед
    изменением ExecutionRequest обязателен rollback и повторная загрузка строк.
    PAPER можно безопасно повторить: Fake gateway не создаёт внешнего ордера.
    LIVE/SHADOW детерминированно отклоняются при is_deterministic_rejection=True.
    В противном случае требуют ручной проверки (MANUAL_REVIEW_REQUIRED).
    """
    await session.rollback()

    req = (
        await session.execute(
            select(ExecutionRequest)
            .where(ExecutionRequest.id == request_id)
            .with_for_update()
        )
    ).scalar_one_or_none()
    attempt = await session.get(ExecutionAttempt, attempt_id)

    if not req:
        return

    now = datetime.now(timezone.utc)
    if attempt:
        attempt.status = "FAILED"
        attempt.error_msg = error[:2000]
        attempt.finished_at = now

    if requested_mode == "PAPER" and attempt_no < 3 and not is_deterministic_rejection:
        req.state = "READY"
        req.claimed_by = None
        req.claimed_at = None
        req.lease_expires_at = None
        req.expires_at = now + timedelta(seconds=max(req.ttl_seconds or 60, 60))
        req.error_reason = f"PAPER retry after submit error: {error}"[:2000]
        req.updated_at = now
    else:
        err_lower = error.lower()
        is_deterministic = (
            is_deterministic_rejection
            or ("invalid token" in err_lower)
            or ("not enough balance" in err_lower)
            or (
                "allowance" in err_lower
            )  # covers: insufficient allowance, allowance exceeded, erc20: insufficient allowance
        )
        terminal_state = (
            "REJECTED"
            if (is_deterministic or requested_mode == "PAPER")
            else "MANUAL_REVIEW_REQUIRED"
        )
        await finalize_request(
            session,
            req,
            state=terminal_state,
            error=f"Submit error: {error}"[:2000],
        )

    await session.commit()


async def claim_one(session, worker_mode: str) -> ExecutionRequest | None:
    now = datetime.now(timezone.utc)
    # Bug #4 fix: используем _get_dialect вместо session.bind.dialect.name
    dialect = await _get_dialect(session)

    where_clause = and_(
        ExecutionRequest.requested_mode == worker_mode,
        or_(
            ExecutionRequest.state == "READY",
            and_(
                ExecutionRequest.state == "CLAIMED",
                ExecutionRequest.lease_expires_at < now,
            ),
        ),
    )

    stmt = select(ExecutionRequest).where(where_clause).limit(1)
    if dialect != "sqlite":
        stmt = stmt.with_for_update(skip_locked=True)

    result = await session.execute(stmt)
    req = result.scalar_one_or_none()

    if req:
        expires_at = req.expires_at
        if expires_at is not None:
            # SQLite возвращает naive datetime — нормализуем к UTC
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            if expires_at < now:
                await finalize_request(
                    session, req, state="EXPIRED", error="TTL expired"
                )
                await session.commit()
                return None

        req.state = "CLAIMED"
        req.claimed_by = f"worker-{os.getpid()}"
        req.claimed_at = now
        req.lease_expires_at = now + timedelta(seconds=30)
        req.updated_at = now
        await session.commit()
        return req
    return None


async def _load_paper_execution_config(session, settings: ExecutionSettings) -> dict[str, str]:
    """Read PAPER parity knobs without changing the request transaction."""
    keys = (
        "PAPER_EXECUTION_PROFILE",
        "PAPER_LIVE_DELAY_SEC",
        "PAPER_SLIPPAGE_PCT",
        "PAPER_FEE_RATE",
        "PAPER_MIN_ORDER_SHARES",
    )
    try:
        result = await session.execute(
            select(RuntimeSettings.key, RuntimeSettings.value, RuntimeSettings.updated_by).where(
                RuntimeSettings.key.in_(keys)
            )
        )
        rows = result.all()
    except Exception as exc:
        logger.warning("paper_execution_settings_read_failed", error=str(exc))
        return {}
    values = {row.key: row.value for row in rows}
    owner_by_key = {row.key: row.updated_by for row in rows}
    profile = str(values.get("PAPER_EXECUTION_PROFILE", settings.paper_execution_profile)).strip().upper()
    if profile == "INSTANT" and str(owner_by_key.get("PAPER_EXECUTION_PROFILE", "")).strip().lower() in {"", "system"}:
        profile = "LIVE_PARITY"
    return {
        "profile": profile,
        "delay_sec": str(values.get("PAPER_LIVE_DELAY_SEC", settings.paper_live_delay_sec)),
        "slippage_pct": str(values.get("PAPER_SLIPPAGE_PCT", settings.paper_slippage_pct)),
        "fee_rate": str(values.get("PAPER_FEE_RATE", settings.paper_fee_rate)),
        "min_order_shares": str(values.get("PAPER_MIN_ORDER_SHARES", settings.paper_min_order_shares)),
    }


async def process_ready_requests():
    settings = ExecutionSettings()
    worker_mode = settings.execution_mode.value

    async with async_session() as session:
        req = await claim_one(session, worker_mode)
        if not req:
            return

        logger.info(
            "execution_request_claimed",
            request_id=str(req.id),
            intent=req.intent,
            requested_mode=req.requested_mode,
        )

        # Bug #1 fix: дублирующий kill-switch блок удалён.
        # Kill-switch для LIVE OPEN полностью обрабатывается внутри
        # check_risk_limits() — единая точка проверки без двойного SELECT.

        paper_config = await _load_paper_execution_config(session, settings) if req.requested_mode == "PAPER" else None
        gateway = build_execution_gateway(settings, paper_config=paper_config)

        if req.requested_mode == "LIVE" and gateway.name == "FAKE":
            await finalize_request(
                session,
                req,
                state="REJECTED",
                error="LIVE mode cannot be executed via fake gateway",
            )
            await session.commit()
            return

        market_stmt = select(LiveMarket).where(LiveMarket.market_id == req.market_id)
        market = (await session.execute(market_stmt)).scalar_one_or_none()

        if not market:
            await finalize_request(
                session, req, state="REJECTED", error="Market not found"
            )
            await session.commit()
            return

        token_id = (
            market.yes_token_id if req.outcome_to_buy == "YES" else market.no_token_id
        )
        side = "BUY" if req.intent == "OPEN" else "SELL"

        limit_price = req.limit_price or Decimal("0")
        max_spend_usdc = req.max_spend_usdc or Decimal("0")

        # --- Advisory lock на режим (глобальный) ---
        await _acquire_mode_lock(session, req.requested_mode)

        risk_err = await check_risk_limits(
            session,
            intent=req.intent,
            max_spend_usdc=max_spend_usdc,
            requested_mode=req.requested_mode,
            request_id=req.id,
            trade_history_id=req.trade_history_id,
        )
        if risk_err:
            logger.warning(
                "risk_limit_breached", request_id=str(req.id), error=risk_err
            )
            await finalize_request(
                session, req, state="REJECTED", error=f"Risk check failed: {risk_err}"
            )
            await session.commit()
            return

        attempt_count_stmt = select(ExecutionAttempt).where(
            ExecutionAttempt.request_id == req.id
        )
        attempt_count = len((await session.execute(attempt_count_stmt)).scalars().all())
        attempt_no = attempt_count + 1

        # --- CLOSE: проверка экспирации и allowance ---
        if req.intent == "CLOSE":
            now_utc = datetime.now(timezone.utc)
            if market and market.end_time_est and now_utc >= market.end_time_est:
                logger.info(
                    "market_expired_skipping_close_submit",
                    request_id=str(req.id),
                    market_id=req.market_id,
                    end_time_est=str(market.end_time_est),
                )
                await finalize_request(
                    session,
                    req,
                    state="REJECTED",
                    error=(
                        f"MARKET_EXPIRED_AWAITING_RESOLUTION: "
                        f"Market ended at {market.end_time_est}"
                    ),
                )
                await session.commit()
                return

            try:
                allowance = await gateway.get_token_allowance(token_id)
                if allowance < (req.requested_shares or Decimal("0")):
                    await finalize_request(
                        session,
                        req,
                        state="MANUAL_REVIEW_REQUIRED",
                        error="CONDITIONAL_ALLOWANCE_NOT_READY: run setup_approvals.py",
                    )
                    await session.commit()
                    return
            except GatewayUnavailable as e:
                await finalize_request(
                    session,
                    req,
                    state="MANUAL_REVIEW_REQUIRED",
                    error=f"Cannot verify allowance: {e}",
                )
                await session.commit()
                return

        # --- Проверка цены перед попыткой исполнения ---
        api_client = None
        executable_price = float(limit_price)
        if req.intent == "OPEN":
            if req.requested_mode == "LIVE":
                fresh_quote_unavailable = False
                try:
                    from polyflip.collector.client import PolymarketClient

                    api_client = PolymarketClient()
                    prices = await asyncio.wait_for(
                        api_client.get_market_prices(token_id), timeout=3.0
                    )
                    if prices and prices.get("best_ask") is not None:
                        executable_price = float(prices["best_ask"])
                    else:
                        fresh_quote_unavailable = True
                except Exception as e:
                    logger.warning("worker_fetch_price_failed", error=str(e))
                    fresh_quote_unavailable = True

                if fresh_quote_unavailable:
                    await finalize_request(
                        session,
                        req,
                        state="READY",
                        error="EXECUTION_QUOTE_UNAVAILABLE",
                    )
                    await session.commit()
                    return

            # CLOSE/TAKE_PROFIT requests do not use the OPEN price guard, but
            # maker reprice still needs the same live quote client after a
            # post-only cross.
            if req.requested_mode == "LIVE" and api_client is None:
                try:
                    from polyflip.collector.client import PolymarketClient
                    api_client = PolymarketClient()
                except Exception as client_err:
                    logger.warning("worker_maker_quote_client_unavailable", error=str(client_err))
            req.submit_quote_price = executable_price
            req.submit_quote_at = datetime.now(timezone.utc)

            if req.max_acceptable_price is not None and executable_price > float(
                req.max_acceptable_price
            ):
                logger.warning(
                    "max_acceptable_price_exceeded",
                    request_id=str(req.id),
                    limit_price=float(limit_price),
                    executable_price=executable_price,
                    max_price=float(req.max_acceptable_price),
                )
                await finalize_request(
                    session,
                    req,
                    state="REJECTED",
                    error="MAX_ACCEPTABLE_PRICE_EXCEEDED",
                )
                await session.commit()
                return

        submission_key = f"{req.idempotency_key}:{attempt_no}"

        attempt = ExecutionAttempt(
            request_id=req.id,
            gateway=gateway.name,
            attempt_no=attempt_no,
            submission_key=submission_key,
            started_at=datetime.now(timezone.utc),
        )
        session.add(attempt)

        req.state = "SUBMITTING"
        req.updated_at = datetime.now(timezone.utc)
        await session.commit()

        request_id = req.id
        attempt_id = attempt.id
        requested_mode = req.requested_mode

        try:
            # Конструирование заказа тоже должно находиться внутри try.
            # Иначе Pydantic ValidationError оставляет заявку в SUBMITTING.
            resolved_requested_shares = _resolve_requested_shares(
                requested_shares=req.requested_shares,
                max_spend_usdc=max_spend_usdc,
                limit_price=limit_price,
                side=side,
            )
            if (
                req.requested_shares is None or req.requested_shares <= 0
            ) and resolved_requested_shares > 0:
                req.requested_shares = resolved_requested_shares
                req.updated_at = datetime.now(timezone.utc)
                await session.flush()
                logger.info(
                    "requested_shares_derived_from_budget",
                    request_id=str(req.id),
                    max_spend_usdc=str(max_spend_usdc),
                    limit_price=str(limit_price),
                    requested_shares=str(resolved_requested_shares),
                )

            request_expiration = req.expires_at
            if request_expiration is not None and request_expiration.tzinfo is None:
                request_expiration = request_expiration.replace(tzinfo=timezone.utc)
            order = GatewayOrder(
                attempt_id=attempt_id,
                market_id=req.market_id,
                asset=req.asset,
                outcome_to_buy=req.outcome_to_buy,
                token_id=token_id,
                side=side,
                limit_price=limit_price,
                requested_shares=resolved_requested_shares,
                max_spend_usdc=max_spend_usdc,
                expiration=(
                    int(request_expiration.timestamp())
                    if req.trigger_reason == "TAKE_PROFIT" and request_expiration
                    else None
                ),
            )
            order_mode = "FAK"
            gtc_ttl_sec = 5.0
            retry_attempts = 3
            retry_delay = 0.75
            settings_dict = {}

            if req.requested_mode == "LIVE":
                try:
                    settings_res = await session.execute(
                        select(RuntimeSettings.key, RuntimeSettings.value).where(
                            RuntimeSettings.key.in_(
                                [
                                    "LIVE_ORDER_MODE",
                                    "LIVE_GTC_TTL_SECONDS",
                                    "LIVE_FAK_RETRY_MAX_ATTEMPTS",
                                    "LIVE_FAK_RETRY_DELAY_SEC",
                                    "LIVE_MAKER_REPRICE_ON_CROSS",
                                    "LIVE_MAKER_REPRICE_MAX_RETRIES",
                                    "LIVE_MAKER_TICK_SIZE",
                                    "TAKE_PROFIT_ORDER_MODE",
                                ]
                            )
                        )
                    )
                    settings_dict = {row.key: row.value for row in settings_res.all()}

                    if (
                        "LIVE_ORDER_MODE" in settings_dict
                        and settings_dict["LIVE_ORDER_MODE"]
                    ):
                        order_mode = settings_dict["LIVE_ORDER_MODE"].strip().upper()
                    if (
                        "LIVE_GTC_TTL_SECONDS" in settings_dict
                        and settings_dict["LIVE_GTC_TTL_SECONDS"]
                    ):
                        gtc_ttl_sec = float(settings_dict["LIVE_GTC_TTL_SECONDS"])
                    if (
                        "LIVE_FAK_RETRY_MAX_ATTEMPTS" in settings_dict
                        and settings_dict["LIVE_FAK_RETRY_MAX_ATTEMPTS"]
                    ):
                        retry_attempts = int(
                            settings_dict["LIVE_FAK_RETRY_MAX_ATTEMPTS"]
                        )
                    if (
                        "LIVE_FAK_RETRY_DELAY_SEC" in settings_dict
                        and settings_dict["LIVE_FAK_RETRY_DELAY_SEC"]
                    ):
                        retry_delay = float(settings_dict["LIVE_FAK_RETRY_DELAY_SEC"])
                except Exception as setting_err:
                    logger.warning(
                        "order_mode_settings_read_failed", error=str(setting_err)
                    )

            # SMART_MAKER: auto-select GTC_TTL if shares >= CLOB minimum, else FAK_RETRY.
            # Keep the threshold in execution.config so PAPER and LIVE use the
            # same venue contract and there is no second hard-coded value.
            if order_mode == "SMART_MAKER":
                _effective_shares = _resolve_requested_shares(
                    requested_shares=req.requested_shares,
                    max_spend_usdc=req.max_spend_usdc,
                    limit_price=req.limit_price,
                    side=req.side,
                )
                if _effective_shares >= POLYMARKET_MIN_ORDER_SHARES:
                    order_mode = "GTC_TTL"
                    logger.info("smart_maker_chose_gtc", shares=str(_effective_shares))
                else:
                    order_mode = "FAK_RETRY"
                    logger.info("smart_maker_chose_fak_retry", shares=str(_effective_shares))

            if order_mode in {"MAKER_TTL", "LIMIT_TTL"}:
                order_mode = "GTC_TTL"
            maker_reprice_enabled = str(settings_dict.get("LIVE_MAKER_REPRICE_ON_CROSS", "true")).strip().lower() in {"1", "true", "yes", "on"}
            try:
                maker_reprice_attempts = min(1, max(0, int(settings_dict.get("LIVE_MAKER_REPRICE_MAX_RETRIES", "1"))))
            except (TypeError, ValueError):
                maker_reprice_attempts = 1
            try:
                maker_tick_size = Decimal(str(settings_dict.get("LIVE_MAKER_TICK_SIZE", "0.01")))
                if maker_tick_size <= 0:
                    raise ValueError
            except (ArithmeticError, TypeError, ValueError):
                maker_tick_size = Decimal("0.01")
            req.execution_order_mode = order_mode
            req.post_only = order_mode in {"GTC_TTL", "GTD"}
            order = order.model_copy(update={"post_only": req.post_only})

            is_gtd_take_profit = (
                req.intent == "CLOSE"
                and req.trigger_reason == "TAKE_PROFIT"
                and req.requested_mode == "LIVE"
                and str(settings_dict.get("TAKE_PROFIT_ORDER_MODE", "GTD"))
                .strip()
                .upper()
                == "GTD"
            )
            if is_gtd_take_profit:
                order_mode = "GTD"
                req.execution_order_mode = order_mode
                req.post_only = True
                order = order.model_copy(update={"post_only": True})

            if order_mode == "GTC_TTL":
                sub_res = await execute_gtc_ttl(
                    gateway,
                    order,
                    ttl_seconds=gtc_ttl_sec,
                    api_client=api_client,
                    max_acceptable_price=req.max_acceptable_price,
                    max_reprice_attempts=(maker_reprice_attempts if maker_reprice_enabled else 0),
                    tick_size=maker_tick_size,
                )
            elif order_mode == "GTD":
                sub_res = await execute_maker_limit(
                    gateway,
                    order,
                    order_type="GTD",
                    api_client=api_client,
                    max_acceptable_price=req.max_acceptable_price,
                    max_reprice_attempts=(maker_reprice_attempts if maker_reprice_enabled else 0),
                    tick_size=maker_tick_size,
                )
            elif order_mode == "FAK_RETRY":
                api_client_retry = api_client if api_client else None
                sub_res = await execute_fak_retry(
                    gateway,
                    order,
                    api_client=api_client_retry,
                    max_attempts=retry_attempts,
                    delay_seconds=retry_delay,
                )
            else:
                sub_res = await gateway.submit(order)

            attempt.finished_at = datetime.now(timezone.utc)
            attempt.provider_order_id = sub_res.provider_order_id
            attempt.provider_status = sub_res.provider_status
            attempt.provider_trade_ids = list(sub_res.provider_trade_ids)
            attempt.transaction_hashes = list(sub_res.transaction_hashes)
            attempt.settlement_state = sub_res.settlement_state
            req.submitted_limit_price = float(sub_res.submitted_limit_price or order.limit_price)
            if sub_res.submitted_requested_shares:
                req.requested_shares = sub_res.submitted_requested_shares
            if sub_res.paper_quote_price is not None:
                # The quote belongs to the parity gateway snapshot, not to the
                # original release decision. Keep its timestamp beside the
                # price so PAPER/LIVE latency analysis remains meaningful.
                req.submit_quote_price = float(sub_res.paper_quote_price)
                req.submit_quote_at = datetime.now(timezone.utc)
            maker_telemetry = {
                "maker_status": sub_res.maker_status,
                "maker_attempts": sub_res.maker_attempts,
                "maker_best_bid": (str(sub_res.maker_best_bid) if sub_res.maker_best_bid is not None else None),
                "maker_best_ask": (str(sub_res.maker_best_ask) if sub_res.maker_best_ask is not None else None),
                "submitted_limit_price": str(req.submitted_limit_price),
            }
            existing_response = attempt.raw_response if isinstance(attempt.raw_response, dict) else {}
            paper_telemetry = {
                "profile": getattr(gateway, "profile", None),
                "quote_price": (str(sub_res.paper_quote_price) if sub_res.paper_quote_price is not None else None),
                "quote_at": (req.submit_quote_at.isoformat() if req.submit_quote_at is not None else None),
                "available_shares": (str(sub_res.paper_available_shares) if sub_res.paper_available_shares is not None else None),
                "delay_seconds": sub_res.paper_delay_seconds,
                "slippage_usdc": (str(sub_res.paper_slippage_usdc) if sub_res.paper_slippage_usdc is not None else None),
                "fee_usdc": (str(sub_res.paper_fee_usdc) if sub_res.paper_fee_usdc is not None else None),
                "provider_status": sub_res.provider_status,
                "fills": [
                    {
                        "provider_trade_id": fill.provider_trade_id,
                        "price": str(fill.price),
                        "shares": str(fill.shares),
                        "gross_quote_usdc": str(fill.gross_quote_usdc),
                        "fee_usdc": str(fill.fee_usdc),
                        "matched_at": fill.matched_at.isoformat(),
                    }
                    for fill in sub_res.fills
                ],
            }
            attempt.raw_response = {
                **existing_response,
                "maker_telemetry": maker_telemetry,
                "paper_telemetry": paper_telemetry,
            }

            if not sub_res.accepted or sub_res.provider_status in ("REJECTED", "ERROR"):
                attempt.status = "FAILED"
                attempt.error_msg = sub_res.provider_status
                await finalize_request(
                    session,
                    req,
                    state="REJECTED",
                    error=sub_res.error_message or sub_res.provider_status,
                )
            elif sub_res.settlement_state == "CONFIRMED":
                # Шлюзы SHADOW/FAKE возвращают fills синхронно в sub_res.fills.
                # LIVE: делаем fetch_order_fills как обычно.
                if sub_res.fills:
                    fills = sub_res.fills
                else:
                    fills = await gateway.fetch_order_fills(
                        attempt.provider_order_id, token_id
                    )
                if len(fills) == 0:
                    attempt.status = "FAILED"
                    attempt.error_msg = (
                        "Order cancelled or expired on exchange without fills"
                    )
                    await finalize_request(
                        session,
                        req,
                        state="REJECTED",
                        error="NO_FILLS_UNFILLED: 0 shares matched on exchange",
                    )
                else:
                    attempt.status = "SUCCESS"
                    await _persist_fills(session, attempt, fills)
                    filled_shares = sum((fill.shares for fill in fills), Decimal("0"))
                    filled_quote = sum(
                        (fill.gross_quote_usdc for fill in fills), Decimal("0")
                    )

                    req.filled_shares = filled_shares
                    req.filled_cost_usdc = filled_quote

                    if filled_shares < (req.requested_shares or Decimal("0")):
                        await finalize_request(
                            session, req, state="PARTIALLY_FILLED_FINAL"
                        )
                    else:
                        await finalize_request(session, req, state="FILLED")

            elif sub_res.settlement_state == "FAILED":
                attempt.status = "FAILED"
                attempt.error_msg = "Settlement failed on chain"
                await finalize_request(
                    session,
                    req,
                    state="REJECTED",
                    error="Settlement failed on chain",
                )
            else:
                # PENDING или UNKNOWN — отправляем в RECONCILING
                attempt.status = "UNKNOWN"
                req.state = "RECONCILING"

            req.updated_at = datetime.now(timezone.utc)

            # rebuild_trade_accounting НЕ делает commit сам.
            # Единый commit ниже фиксирует fills + request state + TradeHistory атомарно.
            if req.trade_history_id:
                await rebuild_trade_accounting(session, req.trade_history_id)
                await _enqueue_gtd_take_profit_after_fill(session, req)

            await session.commit()

        except GatewayOrderRejected as e:
            logger.warning(
                "gateway_order_rejected", error=str(e), attempt_id=str(attempt_id)
            )
            attempt.status = "FAILED"
            attempt.error_msg = str(e)
            attempt.finished_at = datetime.now(timezone.utc)
            await finalize_request(session, req, state="REJECTED", error=str(e))
            req.updated_at = datetime.now(timezone.utc)
            if req.trade_history_id:
                await rebuild_trade_accounting(session, req.trade_history_id)
            await session.commit()

        except (GatewaySubmissionUnknown, GatewayUnavailable) as e:
            logger.warning(
                "gateway_submission_unknown", error=str(e), attempt_id=str(attempt_id)
            )
            await _finish_submit_exception(
                session,
                request_id=request_id,
                attempt_id=attempt_id,
                attempt_no=attempt_no,
                requested_mode=requested_mode,
                error=f"Submission unknown: {e}",
                is_deterministic_rejection=False,
            )

        except Exception as e:
            logger.exception(
                "gateway_submit_failed", error=str(e), attempt_id=str(attempt_id)
            )
            await _finish_submit_exception(
                session,
                request_id=request_id,
                attempt_id=attempt_id,
                attempt_no=attempt_no,
                requested_mode=requested_mode,
                error=str(e),
                is_deterministic_rejection=False,
            )


async def rebuild_trade_accounting(session, trade_id: int) -> Optional[TradeHistory]:
    """
    Пересчитывает бухгалтерию позиции по всем fills.

    Правила:
    - PnL считается per-fill, без вычитания ВСЕХ entry-комиссий при первом
      частичном выходе.
    - realized_pnl = Σ(close_proceeds) - Σ(allocated_entry_basis) - Σ(close_fees)
    - allocated_entry_basis = avg_entry_cost_per_share * close_shares
    - remaining_shares не может быть отрицательным.
    - take_profit_price вычисляется из реальной средней цены входа.
    """
    trade = (
        await session.execute(
            select(TradeHistory).where(TradeHistory.id == trade_id).with_for_update()
        )
    ).scalar_one_or_none()

    from polyflip.execution.states import FINAL_POSITION_STATES, ExitReason

    if not trade:
        return None

    if trade.position_status in FINAL_POSITION_STATES:
        logger.warning(
            "rebuild_accounting_on_final_trade",
            trade_id=trade_id,
            status=trade.position_status,
        )
        return trade

    reqs_result = await session.execute(
        select(ExecutionRequest).where(ExecutionRequest.trade_history_id == trade_id)
    )
    reqs = reqs_result.scalars().all()

    if not reqs and trade.position_status in ("OPEN", "CLOSED", "PARTIALLY_CLOSED"):
        return

    # --- Собираем все fills, разделяя entry и exit ---
    open_shares = Decimal("0")
    open_gross = Decimal("0")
    open_fees = Decimal("0")
    close_shares = Decimal("0")
    close_gross = Decimal("0")
    close_fees = Decimal("0")
    latest_close_time = None

    for req in reqs:
        attempts = (
            (
                await session.execute(
                    select(ExecutionAttempt).where(
                        ExecutionAttempt.request_id == req.id
                    )
                )
            )
            .scalars()
            .all()
        )

        for attempt in attempts:
            fills = (
                (
                    await session.execute(
                        select(ExecutionFill).where(
                            ExecutionFill.attempt_id == attempt.id
                        )
                    )
                )
                .scalars()
                .all()
            )

            for fill in fills:
                gross = fill.gross_quote_usdc or (fill.shares * fill.price)
                fee = fill.fee_usdc or Decimal("0")
                if req.intent == "OPEN":
                    open_shares += fill.shares
                    open_gross += gross
                    open_fees += fee
                elif req.intent == "CLOSE":
                    close_shares += fill.shares
                    close_gross += gross
                    close_fees += fee
                    if latest_close_time is None or fill.timestamp > latest_close_time:
                        latest_close_time = fill.timestamp

    # Fallback: если fills по OPEN нет, берём данные самой сделки
    if (
        open_shares == Decimal("0")
        and trade.entry_filled_shares
        and trade.entry_filled_shares > 0
    ):
        open_shares = Decimal(str(trade.entry_filled_shares))
        open_gross = Decimal(str(trade.entry_cost_usdc or trade.amount_usdc or "0"))

    # Invariant: нельзя продать больше, чем куплено
    if close_shares > open_shares + Decimal("0.000001") and open_shares > Decimal("0"):
        logger.error(
            "accounting_invariant_violated",
            trade_id=trade_id,
            open_shares=str(open_shares),
            close_shares=str(close_shares),
        )
        # Не зажимаем — переводим в MANUAL_REVIEW_REQUIRED и прерываем
        trade.position_status = "MANUAL_REVIEW_REQUIRED"
        trade.last_exit_error = (
            f"over-close: close_shares={close_shares} > open_shares={open_shares}"
        )
        await session.commit()
        return

    # Repair legacy rows where a terminal OPEN request was marked FAILED but
    # accounting left the position in OPENING and dropped the provider reason.
    # This can happen when a worker lease expires between finalize and a later
    # accounting rebuild. Do not repair while another OPEN request is active.
    failed_open_requests = [
        req
        for req in reqs
        if req.intent == "OPEN"
        and req.state in FAILURE_TERMINAL_STATES
        and Decimal(str(req.filled_shares or 0)) <= Decimal("0")
    ]
    active_open_requests = [
        req
        for req in reqs
        if req.intent == "OPEN" and req.state in ACTIVE_REQUEST_STATES
    ]
    if (
        open_shares <= Decimal("0")
        and not trade.entry_filled_shares
        and failed_open_requests
        and not active_open_requests
    ):
        failed_req = max(
            failed_open_requests,
            key=lambda item: (
                item.updated_at
                or item.created_at
                or datetime.min.replace(tzinfo=timezone.utc)
            ),
        )
        trade.status = "FAILED"
        trade.position_status = "ENTRY_FAILED"
        if not trade.error_msg:
            trade.error_msg = (
                failed_req.error_reason
                or failed_req.terminal_code
                or f"Execution request {failed_req.state}"
            )
        trade.entry_filled_shares = Decimal("0")
        trade.entry_cost_usdc = Decimal("0")
        trade.remaining_shares = Decimal("0")
        trade.executed_price = 0.0
        trade.realized_pnl_usdc = Decimal("0")
        trade.pnl = 0.0
        trade.position_accounting_version = (trade.position_accounting_version or 0) + 1
        return trade

    # --- PnL по формуле частичного закрытия ---
    # entry_basis включает gross + fees (полная стоимость входа)
    entry_basis = open_gross + open_fees
    avg_entry_cost_per_share = (
        entry_basis / open_shares if open_shares > Decimal("0") else Decimal("0")
    )
    allocated_basis = avg_entry_cost_per_share * close_shares
    realized_pnl = close_gross - close_fees - allocated_basis

    remaining_shares = open_shares - close_shares

    trade.entry_filled_shares = open_shares
    # entry_cost_usdc = gross + entry fees (полная себестоимость входа)
    trade.entry_cost_usdc = open_gross + open_fees
    trade.amount_usdc = open_gross  # legacy: только gross
    if open_shares > Decimal("0"):
        trade.executed_price = float(open_gross / open_shares)
    else:
        trade.executed_price = 0.0
    trade.remaining_shares = max(Decimal("0"), remaining_shares)
    trade.realized_pnl_usdc = realized_pnl
    trade.pnl = float(realized_pnl)  # явное приведение: колонка pnl имеет тип Float

    if close_shares > Decimal("0"):
        avg_close_price = close_gross / close_shares
        trade.close_price = avg_close_price

    # Статус позиции
    if open_shares > Decimal("0") and remaining_shares <= Decimal("0"):
        trade.position_status = "CLOSED"
        if latest_close_time:
            trade.closed_at = latest_close_time
        if trade.exit_reason == "TAKE_PROFIT":
            trade.take_profit_status = "TRIGGERED"
        elif trade.exit_reason == "STOP_LOSS":
            trade.stop_loss_status = "TRIGGERED"
    elif remaining_shares > Decimal("0") and remaining_shares < open_shares:
        trade.position_status = "PARTIALLY_CLOSED"
    elif remaining_shares > Decimal("0"):
        trade.position_status = "OPEN"
        if close_shares <= Decimal("0"):
            if trade.exit_reason != ExitReason.SETTLEMENT:
                trade.exit_reason = None

    # Stop Loss / Take Profit цены из реальной средней цены входа
    if trade.position_status in ("OPEN", "PARTIALLY_CLOSED") and open_shares > Decimal(
        "0"
    ):
        close_reqs = [r for r in reqs if r.intent == "CLOSE"]
        all_close_failed = len(close_reqs) > 0 and all(
            r.state in ("REJECTED", "FAILED") for r in close_reqs
        )

        if trade.stop_loss_pct is not None:
            if trade.stop_loss_status not in ("TRIGGERED", "ACTIVE") or (
                trade.stop_loss_status == "TRIGGERED" and all_close_failed
            ):
                trade.stop_loss_status = "ACTIVE"
        if trade.take_profit_enabled:
            if trade.take_profit_status not in ("TRIGGERED", "ACTIVE") or (
                trade.take_profit_status == "TRIGGERED" and all_close_failed
            ):
                trade.take_profit_status = "ACTIVE"

        if avg_entry_cost_per_share > Decimal("0"):
            if trade.stop_loss_pct is not None and trade.stop_loss_price is None:
                trade.stop_loss_price = float(
                    avg_entry_cost_per_share
                    * (
                        Decimal("1")
                        - Decimal(str(trade.stop_loss_pct)) / Decimal("100")
                    )
                )
            if trade.take_profit_enabled and trade.take_profit_price is None:
                multiplier = Decimal(
                    str(getattr(trade, "take_profit_multiplier", 1.5) or 1.5)
                )
                raw_target = avg_entry_cost_per_share * multiplier
                if raw_target > Decimal("0.99"):
                    # Цель выходит за пределы рынка (>0.99). Отключаем TP, чтобы позиция
                    # спокойно доходила до разрешения/экспирации и получала полные $1.00.
                    trade.take_profit_status = "SKIPPED"
                    trade.take_profit_price = None
                else:
                    trade.take_profit_price = float(raw_target)

    # Общий статус
    if open_shares > Decimal("0"):
        trade.status = "SUCCESS"
        if remaining_shares > Decimal("0") and trade.position_status not in (
            "PARTIALLY_CLOSED",
            "CLOSED",
        ):
            trade.position_status = "OPEN"
    else:
        # Защита от установки OPEN/SUCCESS при zero-fills.
        # Учитываем, что finalize_request может перевести позицию в FAILED / ENTRY_FAILED
        if trade.status != "FAILED" and trade.position_status != "ENTRY_FAILED":
            trade.status = "PENDING"
            trade.position_status = "OPENING"

    trade.position_accounting_version = (trade.position_accounting_version or 0) + 1
    # НЕ делаем session.commit() здесь — вызывающая функция владеет транзакцией.
    # Это гарантирует атомарность: fills + request state + TradeHistory
    # фиксируются в одном commit, а не в двух независимых.
    return trade


async def reconcile_active_requests():
    """
    Опрашивает gateway для заявок в RECONCILABLE_REQUEST_STATES.
    При таймауте переходит в MANUAL_REVIEW_REQUIRED (не REJECTED),
    чтобы не снять резерв раньше времени.
    """
    settings = ExecutionSettings()
    async with async_session() as session:
        now = datetime.now(timezone.utc)
        # Выбираем все reconcilable заявки своего режима
        stmt = select(ExecutionRequest).where(
            ExecutionRequest.state.in_(RECONCILABLE_REQUEST_STATES),
            ExecutionRequest.requested_mode == settings.execution_mode.value,
        )
        result = await session.execute(stmt)
        reqs = result.scalars().all()

        for req in reqs:
            updated_at = req.updated_at
            if updated_at is not None and updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=timezone.utc)
            if not updated_at or (now - updated_at).total_seconds() < 60:
                continue

            logger.info("reconciling_request", request_id=str(req.id), state=req.state)

            attempt_stmt = (
                select(ExecutionAttempt)
                .where(ExecutionAttempt.request_id == req.id)
                .order_by(ExecutionAttempt.attempt_no.desc())
                .limit(1)
            )
            attempt = (await session.execute(attempt_stmt)).scalar_one_or_none()

            time_in_reconciling = (now - updated_at).total_seconds()
            request_expiry = req.expires_at
            if request_expiry is not None and request_expiry.tzinfo is None:
                request_expiry = request_expiry.replace(tzinfo=timezone.utc)
            is_gtd_take_profit = (
                req.intent == "CLOSE"
                and req.trigger_reason == "TAKE_PROFIT"
                and req.requested_mode == "LIVE"
                and request_expiry is not None
                and request_expiry <= now
            )

            # Таймаут: неизвестность != отсутствие сделки.
            # MANUAL_REVIEW_REQUIRED сохраняет резерв.
            if (
                time_in_reconciling > MAX_RECONCILIATION_AGE_SEC
                and not is_gtd_take_profit
            ):
                logger.warning("request_timed_out_in_unknown", request_id=str(req.id))
                if attempt:
                    attempt.status = "UNKNOWN"
                    attempt.error_msg = (
                        "Automatic reconciliation timed out; "
                        "provider evidence remains unresolved"
                    )
                await finalize_request(
                    session,
                    req,
                    state="MANUAL_REVIEW_REQUIRED",
                    error=(
                        "Settlement status still unknown after "
                        f"{MAX_RECONCILIATION_AGE_SEC}s, manual review required"
                    ),
                )
                await session.commit()
                continue

            if is_gtd_take_profit and (not attempt or not attempt.provider_order_id):
                if attempt:
                    attempt.status = "FAILED"
                    attempt.error_msg = "GTD order expired without provider order id"
                    attempt.finished_at = now
                await finalize_request(
                    session,
                    req,
                    state="EXPIRED",
                    error="GTD take-profit expired without provider order id",
                )
                if req.trade_history_id:
                    await rebuild_trade_accounting(session, req.trade_history_id)
                    trade = await session.get(TradeHistory, req.trade_history_id)
                    if trade and trade.position_status != "CLOSED":
                        trade.take_profit_status = "EXPIRED"
                await session.commit()
                continue

            if not attempt or not attempt.provider_order_id:
                if settings.execution_mode.value == "PAPER":
                    # У PAPER нет внешнего ордера, поэтому SUBMITTING без
                    # provider_order_id можно безопасно вернуть в READY.
                    if attempt:
                        attempt.status = "FAILED"
                        attempt.finished_at = now
                        attempt.error_msg = (
                            "Recovered stale PAPER attempt without provider_order_id"
                        )
                    req.state = "READY"
                    req.claimed_by = None
                    req.claimed_at = None
                    req.lease_expires_at = None
                    req.expires_at = now + timedelta(
                        seconds=max(req.ttl_seconds or 60, 60)
                    )
                    req.error_reason = (
                        "Recovered stale PAPER request without provider_order_id"
                    )
                    req.updated_at = now
                    await session.commit()
                    logger.warning(
                        "paper_request_requeued_without_provider_id",
                        request_id=str(req.id),
                    )
                    continue
                logger.warning(
                    "cannot_reconcile_no_provider_id", request_id=str(req.id)
                )
                continue

            gateway = build_execution_gateway(settings)

            try:
                market = await session.scalar(
                    select(LiveMarket).where(LiveMarket.market_id == req.market_id)
                )
                token_id = (
                    market.yes_token_id
                    if market and req.outcome_to_buy == "YES"
                    else market.no_token_id if market else ""
                )

                fills = await gateway.fetch_order_fills(
                    attempt.provider_order_id,
                    token_id,
                )
                if fills:
                    await _persist_fills(session, attempt, fills)

                    filled_shares = sum(
                        (fill.shares for fill in fills),
                        Decimal("0"),
                    )
                    filled_cost = sum(
                        (fill.gross_quote_usdc for fill in fills),
                        Decimal("0"),
                    )
                    req.filled_shares = filled_shares
                    req.filled_cost_usdc = filled_cost
                    attempt.status = "SUCCESS"
                    attempt.provider_status = "MATCHED"
                    attempt.provider_trade_ids = [
                        fill.provider_trade_id for fill in fills
                    ]
                    attempt.transaction_hashes = list(
                        {
                            fill.transaction_hash
                            for fill in fills
                            if fill.transaction_hash
                        }
                    )
                    attempt.settlement_state = "CONFIRMED"
                    attempt.finished_at = now

                    if filled_shares < (req.requested_shares or Decimal("0")):
                        await finalize_request(
                            session, req, state="PARTIALLY_FILLED_FINAL"
                        )
                    else:
                        await finalize_request(session, req, state="FILLED")

                    if req.trade_history_id:
                        await rebuild_trade_accounting(
                            session,
                            req.trade_history_id,
                        )
                        await _enqueue_gtd_take_profit_after_fill(session, req)

                    await session.commit()
                    continue

                # A native GTD order is allowed to remain pending until the
                # market closes. Once its expiry is reached, cancel the
                # provider order (after the fill lookup above) and release
                # the position back to the normal accounting state.
                if is_gtd_take_profit:
                    if attempt and attempt.provider_order_id:
                        try:
                            await gateway.cancel_order(attempt.provider_order_id)
                        except Exception as cancel_error:
                            logger.warning(
                                "gtd_take_profit_cancel_failed",
                                request_id=str(req.id),
                                error=str(cancel_error),
                            )
                    if attempt:
                        attempt.status = "FAILED"
                        attempt.error_msg = "GTD order expired at market end"
                        attempt.finished_at = now
                    await finalize_request(
                        session,
                        req,
                        state="EXPIRED",
                        error="GTD take-profit order expired at market end",
                    )
                    if req.trade_history_id:
                        await rebuild_trade_accounting(session, req.trade_history_id)
                        trade = await session.get(TradeHistory, req.trade_history_id)
                        if trade and trade.position_status != "CLOSED":
                            trade.take_profit_status = "EXPIRED"
                    await session.commit()
                    continue

                sub_res = await gateway.get_order(attempt.provider_order_id)
                attempt.provider_status = sub_res.provider_status
                if sub_res.settlement_state:
                    attempt.settlement_state = sub_res.settlement_state
                PENDING_ORDER_STATUSES = frozenset(
                    {"ACCEPTED", "UNKNOWN", "PENDING", "LIVE", "DELAYED"}
                )

                if sub_res.provider_status.upper() == "MATCHED":
                    attempt.status = "UNKNOWN"
                    attempt.provider_status = "MATCHED"
                    req.state = "RECONCILING"
                    await session.commit()
                    continue

                if (
                    sub_res.provider_status.upper() in PENDING_ORDER_STATUSES
                    or sub_res.settlement_state in ("PENDING", "UNKNOWN", "")
                ):
                    logger.info(
                        "reconcile_still_pending",
                        request_id=str(req.id),
                        provider_status=sub_res.provider_status,
                    )
                    req.state = "RECONCILING"
                    await session.commit()
                    continue

                if sub_res.settlement_state == "CONFIRMED":
                    logger.info(
                        "reconcile_confirmed_no_fills_waiting",
                        request_id=str(req.id),
                    )
                    req.state = "RECONCILING"
                    req.updated_at = now
                    await session.commit()
                    continue

                elif sub_res.settlement_state in (
                    "FAILED",
                    "REJECTED",
                    "EXPIRED",
                    "CANCELED",
                ) or sub_res.provider_status.upper() in (
                    "REJECTED",
                    "FAILED",
                    "EXPIRED",
                    "CANCELED",
                    "UNMATCHED",
                ):
                    attempt.status = "FAILED"
                    await finalize_request(
                        session,
                        req,
                        state="REJECTED",
                        error=f"Gateway returned terminal status: "
                        f"{sub_res.settlement_state or sub_res.provider_status}",
                    )
                    await session.commit()
                else:
                    logger.info(
                        "reconcile_unknown_status",
                        request_id=str(req.id),
                        gateway_status=sub_res.provider_status,
                    )
                    req.state = "RECONCILING"
                    req.updated_at = now
                    await session.commit()

            except GatewayUnavailable as e:
                logger.warning("gateway_unavailable_reconcile", error=str(e))
            except Exception as e:
                logger.exception(
                    "reconcile_failed", request_id=str(req.id), error=str(e)
                )


_last_auto_resolve_check_by_mode: dict[str, datetime] = {}
AUTO_RESOLVE_CHECK_INTERVAL_SEC = 60


async def _auto_resolve_stuck_manual_reviews(execution_mode: str) -> None:
    """
    Автоматически закрывает зависшие MANUAL_REVIEW_REQUIRED CLOSE-заявки старше 15 минут.
    Перед REJECTED проверяются возможные fills через gateway.
    Использует собственную изолированную сессию БД с троттлингом 60 сек на каждый режим.
    """
    now = datetime.now(timezone.utc)
    last_check = _last_auto_resolve_check_by_mode.get(execution_mode)
    if (
        last_check
        and (now - last_check).total_seconds() < AUTO_RESOLVE_CHECK_INTERVAL_SEC
    ):
        return
    _last_auto_resolve_check_by_mode[execution_mode] = now

    settings = ExecutionSettings()
    async with async_session() as session:
        manual_cutoff = now - timedelta(minutes=15)
        stmt_manual = select(ExecutionRequest).where(
            ExecutionRequest.state == "MANUAL_REVIEW_REQUIRED",
            ExecutionRequest.requested_mode == execution_mode,
            ExecutionRequest.intent == "CLOSE",
            ExecutionRequest.updated_at <= manual_cutoff,
        )
        stuck_manual_reqs = (await session.execute(stmt_manual)).scalars().all()
        if not stuck_manual_reqs:
            return

        gateway = build_execution_gateway(settings)
        for req in stuck_manual_reqs:
            logger.info(
                "auto_resolving_stuck_manual_review_close_request",
                request_id=str(req.id),
                trade_history_id=req.trade_history_id,
            )
            attempt_stmt = (
                select(ExecutionAttempt)
                .where(ExecutionAttempt.request_id == req.id)
                .order_by(ExecutionAttempt.attempt_no.desc())
                .limit(1)
            )
            attempt = (await session.execute(attempt_stmt)).scalar_one_or_none()

            fills = []
            if attempt and attempt.provider_order_id:
                try:
                    market = await session.scalar(
                        select(LiveMarket).where(LiveMarket.market_id == req.market_id)
                    )
                    token_id = (
                        market.yes_token_id
                        if market and req.outcome_to_buy == "YES"
                        else (market.no_token_id if market else "")
                    )
                    if token_id:
                        fills = await gateway.fetch_order_fills(
                            attempt.provider_order_id, token_id
                        )
                except Exception as e:
                    logger.warning(
                        "auto_resolve_fills_fetch_failed",
                        request_id=str(req.id),
                        error=str(e),
                    )

            if fills and attempt:
                await _persist_fills(session, attempt, fills)
                await finalize_request(session, req, state="FILLED")
            else:
                await finalize_request(
                    session,
                    req,
                    state="REJECTED",
                    error="Auto-resolved: no fills confirmed after MANUAL_REVIEW timeout",
                )

            if req.trade_history_id:
                await rebuild_trade_accounting(session, req.trade_history_id)
            await session.commit()


def classify_readiness_error(exc: Exception) -> str:
    message = str(exc).lower()

    if isinstance(exc, asyncio.TimeoutError):
        return "READINESS_TIMEOUT"

    if isinstance(exc, ssl.SSLError):
        return "TLS_TRANSPORT_ERROR"

    if "ssl" in message or "bad record mac" in message:
        return "TLS_TRANSPORT_ERROR"

    if "connection" in message or "transport" in message:
        return "NETWORK_TRANSPORT_ERROR"

    return "READINESS_UNKNOWN_ERROR"


async def publish_liveness_once(
    session: AsyncSession,
    worker_id: str,
    execution_mode: str,
):
    now = datetime.now(timezone.utc)
    dialect_name = await _get_dialect(session)
    insert_func = sqlite_insert if dialect_name == "sqlite" else pg_insert

    stmt = (
        insert_func(ExecutionWorkerStatus)
        .values(
            worker_id=worker_id,
            execution_mode=execution_mode,
            heartbeat_at=now,
            gateway_ready=False,
        )
        .on_conflict_do_update(
            index_elements=["worker_id"],
            set_={
                "heartbeat_at": now,
                "execution_mode": execution_mode,
            },
        )
    )

    await session.execute(stmt)
    await session.commit()


async def publish_liveness(worker_id: str, execution_mode: str):
    while True:
        try:
            async with async_session() as session:
                await publish_liveness_once(
                    session,
                    worker_id,
                    execution_mode,
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("heartbeat_failed")

        await asyncio.sleep(10)


async def refresh_gateway_readiness_once(
    session: AsyncSession,
    worker_id: str,
    execution_mode: str,
    gateway,
):
    now = datetime.now(timezone.utc)

    ws = await session.get(ExecutionWorkerStatus, worker_id)
    if not ws:
        return

    probe_token_id = await session.scalar(
        select(LiveMarket.yes_token_id)
        .where(
            LiveMarket.end_time_est > now,
            LiveMarket.yes_token_id.is_not(None),
        )
        .order_by(LiveMarket.last_updated.desc())
        .limit(1)
    )

    if not probe_token_id:
        ws.gateway_ready = False
        ws.conditional_allowance_ready = None
        ws.last_error_code = "NO_APPROVAL_PROBE_TOKEN"
        ws.last_error_message = (
            "Нет активного рынка для проверки Conditional Token Approval"
        )
        ws.readiness_checked_at = now
        await session.commit()
        return

    conditional_token_ids = (probe_token_id,)
    READINESS_TIMEOUT_SECONDS = 15

    try:
        readiness = await asyncio.wait_for(
            gateway.get_readiness(
                conditional_token_ids=conditional_token_ids,
            ),
            timeout=READINESS_TIMEOUT_SECONDS,
        )

        if getattr(readiness, "error_message", None):
            error_code = getattr(
                readiness, "error_code", None
            ) or classify_readiness_error(Exception(readiness.error_message))

            if error_code in {
                "TLS_TRANSPORT_ERROR",
                "NETWORK_TRANSPORT_ERROR",
                "READINESS_TIMEOUT",
            }:
                try:
                    await gateway.invalidate_client()
                except Exception as exc:
                    logger.warning(
                        "gateway_client_invalidation_failed",
                        error=str(exc),
                    )

            ws.gateway_ready = False
            ws.readiness_checked_at = now
            ws.last_error_code = error_code
            ws.last_error_message = readiness.error_message
        else:
            ws.gateway_ready = readiness.ready
            ws.readiness_checked_at = now
            ws.readiness_success_at = now
            ws.last_error_code = None
            ws.last_error_message = None

            if readiness.balance:
                ws.balance_usdc = Decimal(str(readiness.balance.balance_usdc))
            if readiness.collateral_allowance_ready is not None:
                ws.collateral_allowance_ready = readiness.collateral_allowance_ready
            if readiness.conditional_allowance_ready is not None:
                ws.conditional_allowance_ready = readiness.conditional_allowance_ready

            ws.credentials_loaded = readiness.credentials_loaded
            ws.wallet_address = readiness.wallet_address
            ws.network_chain_id = readiness.network_chain_id

    except Exception as exc:
        if isinstance(exc, asyncio.CancelledError):
            raise

        error_code = classify_readiness_error(exc)

        # Сбросить клиент при сетевой/SSL ошибке
        if error_code in {
            "TLS_TRANSPORT_ERROR",
            "NETWORK_TRANSPORT_ERROR",
            "READINESS_TIMEOUT",
        }:
            try:
                await gateway.invalidate_client()
            except Exception as exc:
                logger.warning(
                    "gateway_client_invalidation_failed",
                    error=str(exc),
                )

        ws.gateway_ready = False
        ws.readiness_checked_at = now
        ws.last_error_code = error_code
        ws.last_error_message = str(exc)

    await session.commit()


async def refresh_gateway_readiness(
    worker_id: str,
    execution_mode: str,
    gateway,
):
    while True:
        try:
            async with async_session() as session:
                await refresh_gateway_readiness_once(
                    session,
                    worker_id,
                    execution_mode,
                    gateway,
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("gateway_readiness_refresh_failed")

        await asyncio.sleep(30)


async def execution_worker_loop():
    logger.info("execution_worker_started")
    settings = ExecutionSettings()
    gateway = build_execution_gateway(settings)
    execution_mode = settings.execution_mode.value
    worker_id = f"{execution_mode}:{socket.gethostname()}:{os.getpid()}"

    heartbeat_task = asyncio.create_task(publish_liveness(worker_id, execution_mode))
    readiness_task = asyncio.create_task(
        refresh_gateway_readiness(worker_id, execution_mode, gateway)
    )

    try:
        while True:
            try:
                await process_ready_requests()
                await reconcile_active_requests()
                await _auto_resolve_stuck_manual_reviews(execution_mode)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("execution_worker_error", error=str(exc))

            await asyncio.sleep(1)
    finally:
        heartbeat_task.cancel()
        readiness_task.cancel()
        await asyncio.gather(heartbeat_task, readiness_task, return_exceptions=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", action="store_true", help="Run in dry-run mode and exit 0"
    )
    args = parser.parse_args()

    if args.dry_run:
        print("Dry run successful.")
        sys.exit(0)

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]
    )
    asyncio.run(execution_worker_loop())
