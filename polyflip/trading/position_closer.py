import structlog
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from polyflip.db.models import TradeHistory, LiveMarket
from polyflip.trading.trader import PolyTrader

logger = structlog.get_logger(__name__)


async def claim_position_for_exit(db_session: AsyncSession, trade_id: int, reason: str) -> bool:
    """
    Атомарно захватывает позицию для закрытия.
    Возвращает True, если захват успешен.
    """
    stmt = (
        update(TradeHistory)
        .where(TradeHistory.id == trade_id)
        .where(TradeHistory.position_status.in_(["OPEN", "EXIT_FAILED"]))
        .values(
            position_status="CLOSING",
            exit_reason=reason,
            exit_attempts=TradeHistory.exit_attempts + 1,
            updated_at=datetime.now(timezone.utc)
        )
        .returning(TradeHistory.id)
    )
    result = await db_session.execute(stmt)
    claimed_id = result.scalar()
    await db_session.commit()
    return claimed_id is not None


async def execute_position_exit(
    db_session: AsyncSession,
    trader: PolyTrader,
    trade: TradeHistory,
    market: LiveMarket,
    target_price: float,
    fee_rate: float
):
    """
    Выполняет продажу позиции и вызывает reconcile_position_exit.
    """
    token_to_sell = market.yes_token_id if trade.outcome_bought == "YES" else market.no_token_id
    
    # Пытаемся продать купленный размер
    try:
        size_shares = trade.amount_usdc / trade.executed_price if trade.executed_price > 0 else 0
        
        exit_result = await trader.execute_trade(
            market_id=market.market_id,
            token_id=token_to_sell,
            side="SELL",
            price=target_price,
            size=size_shares
        )
    except Exception as e:
        logger.error("position_exit_exception", trade_id=trade.id, error=str(e))
        exit_result = {"status": "FAILED", "error_msg": str(e)}

    await reconcile_position_exit(db_session, trade.id, exit_result, target_price, fee_rate, trade.market_id, trade.asset, trade.outcome_bought)


async def reconcile_position_exit(
    db_session: AsyncSession, 
    trade_id: int, 
    exit_result: dict,
    target_price: float,
    fee_rate: float,
    market_id: str,
    asset: str,
    outcome_bought: str
):
    """
    Обновляет статус позиции после попытки продажи.
    Если успешно - CLOSED (или PARTIALLY_CLOSED если будет реализовано).
    Если ошибка - EXIT_FAILED.
    """
    now = datetime.now(timezone.utc)
    
    stmt = select(TradeHistory).where(TradeHistory.id == trade_id)
    result = await db_session.execute(stmt)
    trade = result.scalar_one_or_none()
    
    if not trade:
        return
        
    if exit_result.get("status") == "SUCCESS":
        # Успешная продажа
        executed_price = exit_result.get("executed_price", 0.0)
        executed_usdc = exit_result.get("executed_usdc", 0.0)
        
        trade.position_status = "CLOSED"
        trade.closed_at = now
        trade.close_price = executed_price
        
        shares_held = trade.amount_usdc / trade.executed_price if trade.executed_price > 0 else 0
        
        if trade.exit_reason == "TAKE_PROFIT":
            trade.take_profit_sell_size = round(shares_held, 2)
        elif trade.exit_reason == "STOP_LOSS":
            trade.stop_loss_sell_size = round(shares_held, 2)
        
        if trade.amount_usdc > 0 and executed_usdc > 0:
            # PnL (чистый профит/убыток в USDC)
            gross = executed_price * shares_held
            net = gross * (1.0 - fee_rate)
            trade.pnl = net - trade.amount_usdc
            
        trade.exit_order_id = exit_result.get("order_id", "")
        trade.updated_at = now
        
        # Записываем проскальзывание
        from polyflip.db.models import SlippageLog
        slip = target_price - executed_price
        slip_pct = (slip / target_price * 100) if target_price > 0 else 0.0
        slip_cost = slip * shares_held
        
        db_session.add(SlippageLog(
            trade_id=trade.id,
            market_id=market_id,
            asset=asset,
            outcome_bought=outcome_bought,
            expected_price=target_price,
            executed_price=executed_price,
            slippage=slip,
            slippage_pct=slip_pct,
            bet_size_usdc=trade.amount_usdc,
            slippage_cost_usdc=slip_cost,
            mode=exit_result.get("mode", "PAPER"),
            created_at=now,
        ))
        
    else:
        # Неуспешная продажа
        trade.position_status = "EXIT_FAILED"
        
        # Сброс триггера, чтобы воркер мог повторить попытку
        if trade.exit_reason == "STOP_LOSS":
            trade.stop_loss_status = "ACTIVE"
        elif trade.exit_reason == "TAKE_PROFIT":
            trade.take_profit_status = "ACTIVE"
            
        # Сохраняем ошибку, если была
        error_msg = exit_result.get("error_msg", "unknown error")
        logger.warning("position_exit_failed", trade_id=trade_id, error=error_msg)
        trade.updated_at = now

    await db_session.commit()
