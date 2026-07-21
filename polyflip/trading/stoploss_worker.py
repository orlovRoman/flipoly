"""Фоновый воркер: мониторит открытые позиции и триггерит стоп-лосс."""
import asyncio
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
import structlog

from polyflip.db.models import TradeHistory, RuntimeSettings, LiveMarket, SlippageLog
from polyflip.constants import POLYMARKET_FEE_RATE
from polyflip.trading.trader import PolyTrader
from polyflip.collector.client import PolymarketClient
from polyflip.trading.stoploss import evaluate_stop_loss

logger = structlog.get_logger(__name__)


async def stoploss_worker_cycle(
    db_session: AsyncSession,
    trader: PolyTrader,
    api_client: PolymarketClient,
) -> None:
    """Один цикл проверки стоп-лоссов."""

    # 1. Проверяем включён ли стоп-лосс
    result = await db_session.execute(
        select(RuntimeSettings).where(
            RuntimeSettings.key == "STOP_LOSS_ENABLED"
        )
    )
    setting = result.scalar_one_or_none()
    if not setting or setting.value.lower() != "true":
        return

    # Загружаем POLYMARKET_FEE_RATE из RuntimeSettings
    fee_row = await db_session.execute(
        select(RuntimeSettings).where(RuntimeSettings.key == "POLYMARKET_FEE_RATE")
    )
    fee_row_val = fee_row.scalar_one_or_none()
    fee_rate = float(fee_row_val.value) if fee_row_val else POLYMARKET_FEE_RATE

    # 2. Загружаем ACTIVE позиции с выставленным stop_loss_price
    stmt = select(TradeHistory).where(
        and_(
            TradeHistory.status == "SUCCESS",
            TradeHistory.stop_loss_status == "ACTIVE",
            TradeHistory.stop_loss_price.is_not(None),
            TradeHistory.pnl.is_(None),  # позиция ещё не закрыта
        )
    )
    open_trades = (await db_session.execute(stmt)).scalars().all()

    if not open_trades:
        return

    now = datetime.now(timezone.utc)

    for trade in open_trades:
        try:
            # Проверяем не истёк ли рынок по зафиксированному времени окончания
            if trade.market_end_time is not None:
                market_end = trade.market_end_time
                if market_end.tzinfo is None:
                    market_end = market_end.replace(tzinfo=timezone.utc)
                if now >= market_end:
                    trade.stop_loss_status = "EXPIRED"
                    logger.info("stoploss_market_expired", trade_id=trade.id,
                                market_end=market_end.isoformat())
                    await db_session.commit()
                    continue

            # Дополнительная проверка на наличие в LiveMarket
            mkt_result = await db_session.execute(
                select(LiveMarket).where(LiveMarket.market_id == trade.market_id)
            )
            market = mkt_result.scalar_one_or_none()
            if not market:
                trade.stop_loss_status = "EXPIRED"
                logger.warning("stoploss_market_not_in_live", trade_id=trade.id,
                               market_id=trade.market_id)
                await db_session.commit()
                continue

            # Определяем token_id продаваемого токена
            token_id = market.yes_token_id if trade.outcome_bought == "YES" else market.no_token_id

            # Получаем текущий bid
            prices = await api_client.get_market_prices(token_id)
            if not prices or "error" in prices or prices.get("best_bid") is None:
                logger.warning("stoploss_no_bid", trade_id=trade.id, error=prices.get("error") if prices else "No response")
                continue

            current_bid = float(prices["best_bid"])

            if trade.stop_loss_pct is None:
                logger.warning("stoploss_missing_pct", trade_id=trade.id)
                trade.stop_loss_status = "EXPIRED"
                await db_session.commit()
                continue

            decision = evaluate_stop_loss(
                entry_price=trade.executed_price,
                stop_loss_pct=trade.stop_loss_pct,
                current_bid=current_bid,
            )

            if not decision.should_sell:
                continue

            # Триггер: продаём
            logger.warning(
                "stoploss_triggered",
                trade_id=trade.id,
                market_id=trade.market_id,
                entry=trade.executed_price,
                stop_price=decision.stop_price,
                current_bid=current_bid,
            )

            shares_held = round(trade.amount_usdc / trade.executed_price, 2)
            sell_res = await trader.execute_trade(
                market_id=trade.market_id,
                token_id=token_id,
                side="SELL",
                price=current_bid,
                size=shares_held,
            )

            executed_price = sell_res.get("executed_price", current_bid)
            
            # Обновляем запись сделки
            trade.stop_loss_status   = "TRIGGERED"
            trade.stop_loss_hit_at   = now
            trade.stop_loss_sell_price = executed_price
            
            # PnL с учётом комиссии продажи POLYMARKET_FEE_RATE (0.002)
            from polyflip.constants import POLYMARKET_FEE_RATE
            net_sell = executed_price * shares_held * (1.0 - POLYMARKET_FEE_RATE)
            trade.pnl = round(net_sell - trade.amount_usdc, 4)

            # Записываем проскальзывание в SlippageLog
            # Для продажи: slip = expected_price (наш bid триггера) - executed_price (цена исполнения)
            slip = round(current_bid - executed_price, 6)
            slip_pct = round(slip / current_bid * 100, 4) if current_bid > 0 else 0.0
            slip_cost = round(slip * shares_held, 4)

            slippage_record = SlippageLog(
                trade_id=trade.id,
                market_id=trade.market_id,
                asset=trade.asset,
                outcome_bought=trade.outcome_bought,
                expected_price=current_bid,
                executed_price=executed_price,
                slippage=slip,
                slippage_pct=slip_pct,
                bet_size_usdc=trade.amount_usdc,
                slippage_cost_usdc=slip_cost,
                mode=sell_res.get("mode", "PAPER"),
                created_at=now,
            )
            db_session.add(slippage_record)

            await db_session.commit()

            logger.info(
                "stoploss_executed",
                trade_id=trade.id,
                pnl=trade.pnl,
                sell_price=executed_price,
                slippage=slip
            )

        except Exception as e:
            logger.exception("stoploss_worker_error", trade_id=trade.id, error=str(e))
            await db_session.rollback()
