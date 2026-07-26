"""Фоновый воркер: восстанавливает зависшие (EXIT_FAILED) или частично закрытые (PARTIALLY_CLOSED) позиции."""
import asyncio
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
import structlog

from polyflip.db.models import TradeHistory, RuntimeSettings, LiveMarket
from polyflip.services.settings_service import get_float
from polyflip.trading.trader import PolyTrader
from polyflip.collector.client import PolymarketClient
from polyflip.trading.position_closer import execute_position_exit

logger = structlog.get_logger(__name__)


async def _process_single_recovery(
    db_session: AsyncSession,
    trader: PolyTrader,
    api_client: PolymarketClient,
    trade: TradeHistory,
    fee_rate: float,
    now: datetime,
) -> None:
    """
    Проверяет реальный баланс на бирже и закрывает остатки позиции.
    """
    mkt_result = await db_session.execute(
        select(LiveMarket).where(LiveMarket.market_id == trade.market_id)
    )
    market = mkt_result.scalar_one_or_none()
    if not market:
        logger.warning("recovery_market_not_found", trade_id=trade.id, market_id=trade.market_id)
        return

    token_id = market.yes_token_id if trade.outcome_bought == "YES" else market.no_token_id

    # Проверяем реальный баланс
    actual_balance = await trader.get_balance(token_id)
    
    if actual_balance <= 0.01: # Мелкие пылинки игнорируем
        # Баланс нулевой, значит позиция полностью закрыта
        logger.info("recovery_balance_zero_closing", trade_id=trade.id)
        trade.position_status = "CLOSED"
        trade.closed_at = now
        trade.updated_at = now
        return

    # Если баланс есть, нужно допродать
    # Получаем текущую цену (можно продавать по рынку)
    prices = await api_client.get_market_prices(token_id)
    if not prices or "error" in prices or prices.get("best_bid") is None:
        logger.warning("recovery_no_bid", trade_id=trade.id)
        return

    current_bid = float(prices["best_bid"])
    
    logger.info("recovery_executing_exit", trade_id=trade.id, remaining_balance=actual_balance, current_bid=current_bid)
    
    # Пытаемся продать остаток. Важно: execute_position_exit использует trade.amount_usdc / trade.executed_price 
    # для определения размера. Нужно временно подменить?
    # Т.к. мы вызываем execute_position_exit, он попробует продать shares_held = trade.amount_usdc / executed_price
    # Это может быть больше actual_balance. В trader.execute_trade нужно передать правильный размер!
    # Поскольку execute_position_exit жёстко берет размер из trade, мы можем просто напрямую дёрнуть трейдера 
    # и reconcile_position_exit, или переписать execute_position_exit чтобы принимал size.
    # Давайте просто вызовем execute_position_exit, а в нём подправим логику.
    
    # Но для начала, давайте просто передадим. execute_position_exit мы изменим чтобы он принимал optional size.
    await execute_position_exit(db_session, trader, trade, market, current_bid, fee_rate, actual_size=actual_balance)


async def recovery_worker_cycle(
    db_session: AsyncSession,
    trader: PolyTrader,
    api_client: PolymarketClient,
) -> None:
    """Один цикл проверки восстанавливаемых позиций."""
    fee_rate = await get_float(db_session, "POLYMARKET_FEE_RATE")

    stmt = select(TradeHistory).where(
        and_(
            TradeHistory.position_status.in_(["EXIT_FAILED", "PARTIALLY_CLOSED"]),
            TradeHistory.exit_attempts < 10,
        )
    )
    stuck_trades = (await db_session.execute(stmt)).scalars().all()

    if not stuck_trades:
        return

    now = datetime.now(timezone.utc)

    for trade in stuck_trades:
        try:
            await _process_single_recovery(db_session, trader, api_client, trade, fee_rate, now)
            await db_session.commit()
        except Exception as e:
            logger.exception("recovery_worker_error", trade_id=trade.id, error=str(e))
            await db_session.rollback()
