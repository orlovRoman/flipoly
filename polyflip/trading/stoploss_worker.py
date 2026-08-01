"""Фоновый воркер: мониторит открытые позиции и триггерит стоп-лосс."""

import asyncio
from datetime import datetime, timezone, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
import structlog

from polyflip.db.models import TradeHistory, RuntimeSettings, LiveMarket, SlippageLog
from polyflip.services.settings_service import get_float
from polyflip.collector.client import PolymarketClient
from polyflip.trading.stoploss import evaluate_stop_loss

logger = structlog.get_logger(__name__)


async def _process_single_stoploss(
    db_session: AsyncSession,
    api_client: PolymarketClient,
    trade: TradeHistory,
    fee_rate: float,
    now: datetime,
) -> None:
    """
    Обрабатывает одну открытую позицию стоп-лосса.
    Фаза 1: Чтение данных и HTTP вызовы выполняются ВНЕ savepoint.
    Фаза 2: Изменение состояния БД происходит внутри короткого savepoint.
    """
    # ── Фаза 1: чтение данных и HTTP-запросы (ВНЕ savepoint) ──────────────────────────
    if trade.market_end_time is not None:
        market_end = trade.market_end_time
        if market_end.tzinfo is None:
            market_end = market_end.replace(tzinfo=timezone.utc)
        if now >= market_end:
            trade.stop_loss_status = "EXPIRED"
            logger.info(
                "stoploss_market_expired",
                trade_id=trade.id,
                market_end=market_end.isoformat(),
            )
            return

    mkt_result = await db_session.execute(
        select(LiveMarket).where(LiveMarket.market_id == trade.market_id)
    )
    market = mkt_result.scalar_one_or_none()
    if not market:
        trade.stop_loss_status = "EXPIRED"
        logger.warning(
            "stoploss_market_not_in_live", trade_id=trade.id, market_id=trade.market_id
        )
        return

    token_id = (
        market.yes_token_id if trade.outcome_bought == "YES" else market.no_token_id
    )

    # HTTP — ВНЕ транзакции
    prices = await api_client.get_market_prices(token_id)
    if not prices or "error" in prices or prices.get("best_bid") is None:
        logger.warning(
            "stoploss_no_bid",
            trade_id=trade.id,
            error=prices.get("error") if prices else "No response",
        )
        return

    current_bid = float(prices["best_bid"])

    if trade.stop_loss_pct is None:
        trade.stop_loss_status = "EXPIRED"
        logger.warning("stoploss_missing_pct", trade_id=trade.id)
        return

    decision = evaluate_stop_loss(
        entry_price=trade.executed_price,
        stop_loss_pct=trade.stop_loss_pct,
        current_bid=current_bid,
    )

    if not decision.should_sell:
        return

    # HTTP — ВНЕ транзакции
    logger.warning(
        "stoploss_triggered",
        trade_id=trade.id,
        market_id=trade.market_id,
        entry=trade.executed_price,
        stop_price=decision.stop_price,
        current_bid=current_bid,
    )

    from polyflip.execution.outbox import enqueue_close_request

    sell_floor = max(0.01, current_bid - 0.01)

    res = await enqueue_close_request(
        db_session,
        trade_id=trade.id,
        trigger_reason="STOP_LOSS",
        limit_price=sell_floor,
    )
    if res is None or res.disposition != "CREATED":
        return

    # После успешной постановки меняем статус стоп-лосса
    trade.stop_loss_status = "QUEUED"
    trade.stop_loss_hit_at = now
    trade.stop_loss_sell_price = decision.stop_price
    await db_session.commit()

    logger.info(
        "stoploss_outbox_request_created",
        trade_id=trade.id,
        sell_price=current_bid,
        request_id=str(res.request_id),
    )


async def stoploss_worker_cycle(
    db_session: AsyncSession,
    api_client: PolymarketClient,
) -> None:
    """Один цикл проверки стоп-лоссов."""

    # 1. Проверяем включён ли стоп-лосс
    result = await db_session.execute(
        select(RuntimeSettings).where(RuntimeSettings.key == "STOP_LOSS_ENABLED")
    )
    setting = result.scalar_one_or_none()
    if not setting or setting.value.lower() != "true":
        return

    # Загружаем POLYMARKET_FEE_RATE из RuntimeSettings через settings_service
    fee_rate = await get_float(db_session, "POLYMARKET_FEE_RATE")

    # ── Восстановление: обработать зависшие TRIGGERING позиции ──────────
    stuck_stmt = select(TradeHistory).where(
        and_(
            TradeHistory.stop_loss_status == "TRIGGERING",
            TradeHistory.stop_loss_hit_at.is_(None),
        )
    )
    stuck_trades = (await db_session.execute(stuck_stmt)).scalars().all()
    for t in stuck_trades:
        mkt_res = await db_session.execute(
            select(LiveMarket).where(LiveMarket.market_id == t.market_id)
        )
        mkt = mkt_res.scalar_one_or_none()
        token_id = (
            (mkt.yes_token_id if t.outcome_bought == "YES" else mkt.no_token_id)
            if mkt
            else None
        )

        positions = await api_client.get_positions(t.market_id) if token_id else None
        if not positions or positions.get("size", 0) == 0:
            t.stop_loss_status = "TRIGGERED"
            t.stop_loss_hit_at = datetime.now(timezone.utc)
            logger.warning("stoploss_recovering_triggering", trade_id=t.id)
        else:
            t.stop_loss_status = "ACTIVE"
            logger.warning("stoploss_reset_to_active", trade_id=t.id)
    if stuck_trades:
        await db_session.commit()

    # 2. Загружаем ACTIVE позиции с выставленным stop_loss_price
    stmt = select(TradeHistory).where(
        and_(
            TradeHistory.position_status.in_(
                ["OPEN", "EXIT_FAILED", "PARTIALLY_CLOSED"]
            ),
            TradeHistory.exit_attempts < 10,
            TradeHistory.stop_loss_status == "ACTIVE",
            TradeHistory.stop_loss_price.is_not(None),
        )
    )
    open_trades = (await db_session.execute(stmt)).scalars().all()

    if not open_trades:
        return

    now = datetime.now(timezone.utc)

    for trade in open_trades:
        try:
            await _process_single_stoploss(db_session, api_client, trade, fee_rate, now)
            await db_session.commit()
        except Exception as e:
            logger.exception("stoploss_worker_error", trade_id=trade.id, error=str(e))
            await db_session.rollback()
