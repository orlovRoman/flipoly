"""Фоновый воркер: мониторит открытые позиции и триггерит тейк-профит."""

from datetime import datetime, timezone, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
import structlog

from polyflip.db.models import TradeHistory, RuntimeSettings, LiveMarket, SlippageLog
from polyflip.services.settings_service import get_float
from polyflip.collector.client import PolymarketClient
from polyflip.trading.takeprofit import evaluate_take_profit

logger = structlog.get_logger(__name__)


async def takeprofit_worker_cycle(
    db_session: AsyncSession,
    api_client: PolymarketClient,
) -> None:
    """Один цикл проверки тейк-профитов."""

    # 1. Проверяем включён ли тейк-профит
    result = await db_session.execute(
        select(RuntimeSettings).where(RuntimeSettings.key == "TAKE_PROFIT_ENABLED")
    )
    setting = result.scalar_one_or_none()
    if not setting or setting.value.lower() != "true":
        return

    # Загружаем POLYMARKET_FEE_RATE из RuntimeSettings
    fee_rate = await get_float(db_session, "POLYMARKET_FEE_RATE")

    # 2. Загружаем ACTIVE позиции с выставленным take_profit_price
    stmt = select(TradeHistory).where(
        and_(
            TradeHistory.position_status.in_(
                ["OPEN", "EXIT_FAILED", "PARTIALLY_CLOSED"]
            ),
            TradeHistory.exit_attempts < 10,
            TradeHistory.take_profit_status == "ACTIVE",
            TradeHistory.take_profit_price.is_not(None),
        )
    )
    open_trades = (await db_session.execute(stmt)).scalars().all()

    if not open_trades:
        return

    now = datetime.now(timezone.utc)

    for trade in open_trades:
        try:
            # Проверяем не истёк ли рынок
            if trade.market_end_time is not None:
                market_end = trade.market_end_time
                if market_end.tzinfo is None:
                    market_end = market_end.replace(tzinfo=timezone.utc)
                if now >= market_end:
                    trade.take_profit_status = "EXPIRED"
                    logger.info(
                        "takeprofit_market_expired",
                        trade_id=trade.id,
                        market_end=market_end.isoformat(),
                    )
                    await db_session.commit()
                    continue

            # Дополнительная проверка на наличие в LiveMarket
            mkt_result = await db_session.execute(
                select(LiveMarket).where(LiveMarket.market_id == trade.market_id)
            )
            market = mkt_result.scalar_one_or_none()
            if not market:
                trade.take_profit_status = "EXPIRED"
                logger.warning(
                    "takeprofit_market_not_in_live",
                    trade_id=trade.id,
                    market_id=trade.market_id,
                )
                await db_session.commit()
                continue

            token_id = (
                market.yes_token_id
                if trade.outcome_bought == "YES"
                else market.no_token_id
            )

            # Получаем текущий bid (цена, по которой покупатели готовы выкупить токен — цена продажи)
            prices = await api_client.get_market_prices(token_id)
            if not prices or "error" in prices or prices.get("best_bid") is None:
                logger.warning(
                    "takeprofit_no_bid",
                    trade_id=trade.id,
                    error=prices.get("error") if prices else "No response",
                )
                continue

            current_bid = float(prices["best_bid"])

            if trade.take_profit_multiplier is None:
                logger.warning("takeprofit_missing_multiplier", trade_id=trade.id)
                trade.take_profit_status = "EXPIRED"
                await db_session.commit()
                continue

            decision = evaluate_take_profit(
                entry_price=trade.executed_price,
                tp_multiplier=trade.take_profit_multiplier,
                current_bid=current_bid,
            )

            if not decision.should_sell:
                continue

            # Триггер: продаём
            from polyflip.execution.outbox import enqueue_close_request

            sell_floor = max(0.01, current_bid - 0.01)

            res = await enqueue_close_request(
                db_session,
                trade_id=trade.id,
                trigger_reason="TAKE_PROFIT",
                limit_price=sell_floor,
            )
            if res is None or res.disposition != "CREATED":
                continue

            # После успешной постановки меняем статус тейк-профита
            trade.take_profit_status = "QUEUED"
            trade.take_profit_hit_at = now
            trade.take_profit_sell_price = decision.tp_price
            await db_session.commit()

            logger.info(
                "takeprofit_outbox_request_created",
                trade_id=trade.id,
                sell_price=current_bid,
                request_id=str(res.request_id),
            )

        except Exception as e:
            logger.exception("takeprofit_worker_error", trade_id=trade.id, error=str(e))
            await db_session.rollback()
