"""Фоновый воркер: мониторит открытые позиции и триггерит тейк-профит."""
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
import structlog

from polyflip.db.models import TradeHistory, RuntimeSettings, LiveMarket, SlippageLog
from polyflip.trading.trader import PolyTrader
from polyflip.collector.client import PolymarketClient
from polyflip.trading.takeprofit import evaluate_take_profit

logger = structlog.get_logger(__name__)


async def takeprofit_worker_cycle(
    db_session: AsyncSession,
    trader: PolyTrader,
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

    # 2. Загружаем ACTIVE позиции с выставленным take_profit_price
    stmt = select(TradeHistory).where(
        and_(
            TradeHistory.status == "SUCCESS",
            TradeHistory.take_profit_status == "ACTIVE",
            TradeHistory.take_profit_price.is_not(None),
            TradeHistory.pnl.is_(None),  # позиция ещё не закрыта
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
                    logger.info("takeprofit_market_expired", trade_id=trade.id,
                                market_end=market_end.isoformat())
                    await db_session.commit()
                    continue

            # Дополнительная проверка на наличие в LiveMarket
            mkt_result = await db_session.execute(
                select(LiveMarket).where(LiveMarket.market_id == trade.market_id)
            )
            market = mkt_result.scalar_one_or_none()
            if not market:
                trade.take_profit_status = "EXPIRED"
                logger.warning("takeprofit_market_not_in_live", trade_id=trade.id,
                               market_id=trade.market_id)
                await db_session.commit()
                continue

            token_id = market.yes_token_id if trade.outcome_bought == "YES" else market.no_token_id

            # Получаем текущий bid (цена, по которой покупатели готовы выкупить токен — цена продажи)
            prices = await api_client.get_market_prices(token_id)
            if not prices or "error" in prices or prices.get("best_bid") is None:
                logger.warning("takeprofit_no_bid", trade_id=trade.id,
                               error=prices.get("error") if prices else "No response")
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
            logger.info(
                "takeprofit_triggered",
                trade_id=trade.id,
                market_id=trade.market_id,
                entry=trade.executed_price,
                tp_price=decision.tp_price,
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

            # PnL с вычетом комиссии Polymarket 0.2%
            gross = executed_price * shares_held
            net   = gross * (1.0 - 0.002)
            trade.pnl = round(net - trade.amount_usdc, 4)

            # Обновляем поля тейк-профита (status сделки остаётся "SUCCESS")
            trade.take_profit_status    = "TRIGGERED"
            trade.take_profit_hit_at    = now
            trade.take_profit_sell_price = executed_price

            # Записываем проскальзывание
            slip      = round(current_bid - executed_price, 6)
            slip_pct  = round(slip / current_bid * 100, 4) if current_bid > 0 else 0.0
            slip_cost = round(slip * shares_held, 4)

            db_session.add(SlippageLog(
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
            ))

            await db_session.commit()

            logger.info(
                "takeprofit_executed",
                trade_id=trade.id,
                pnl=trade.pnl,
                sell_price=executed_price,
                slippage=slip,
            )

        except Exception as e:
            logger.exception("takeprofit_worker_error", trade_id=trade.id, error=str(e))
            await db_session.rollback()
