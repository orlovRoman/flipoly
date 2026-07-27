"""
Единый сервис закрытия позиций по итогам разрешения рынка.

Используется в:
- polyflip/scheduler/jobs.py (resolve_trades_job)
- scripts/reconstruct_history.py

Правила:
- remaining_basis = avg_entry_cost_per_share * remaining_shares
- payout = remaining_shares * payout_per_share (только если winning outcome)
- realized_pnl += payout - settlement_fee - remaining_basis
- Идемпотентен: повторный вызов на уже закрытой позиции не меняет PnL.
"""
from __future__ import annotations

import structlog
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.db.models import TradeHistory

logger = structlog.get_logger(__name__)

_OUTCOME_ALIASES: dict[str, str] = {
    "UP": "YES",
    "DOWN": "NO",
    "1": "YES",
    "0": "NO",
}


def _normalize_outcome(outcome: str) -> str:
    raw = (outcome or "").upper().strip()
    return _OUTCOME_ALIASES.get(raw, raw)


def _calculate_remaining_basis(trade: TradeHistory) -> Decimal:
    """
    Оставшаяся себестоимость незакрытой части позиции.
    Включает пропорциональную долю entry-комиссий.
    """
    entry_filled = Decimal(str(trade.entry_filled_shares or 0))
    entry_cost = Decimal(str(trade.entry_cost_usdc or trade.amount_usdc or 0))
    remaining = Decimal(str(trade.remaining_shares or 0))

    if entry_filled <= Decimal("0"):
        return Decimal("0")

    # entry_cost уже включает gross + entry fees (см. rebuild_trade_accounting)
    avg_cost_per_share = entry_cost / entry_filled
    return avg_cost_per_share * remaining


async def settle_resolved_position(
    session: AsyncSession,
    *,
    trade_id: int,
    winning_outcome: str,
    payout_per_share: Decimal,
    settlement_fee_usdc: Decimal = Decimal("0"),
) -> None:
    """
    Закрывает позицию по итогам разрешения рынка.

    Параметры
    ----------
    trade_id : int
        ID записи TradeHistory.
    winning_outcome : str
        Итог рынка (YES / NO / INVALID и т.д.). Нормализуется автоматически.
    payout_per_share : Decimal
        Выплата за 1 share победителю (обычно 1.0, при INVALID может быть другой).
    settlement_fee_usdc : Decimal
        Комиссия за settlement (если есть).
    """
    trade = (
        await session.scalar(
            select(TradeHistory)
            .where(TradeHistory.id == trade_id)
            .with_for_update()
        )
    )

    if trade is None:
        logger.warning("settle_resolved_position_trade_not_found", trade_id=trade_id)
        return

    if trade.position_status == "CLOSED":
        logger.info(
            "settle_resolved_position_already_closed",
            trade_id=trade_id,
            pnl=str(trade.realized_pnl_usdc),
        )
        return  # идемпотентность

    normalized_winning = _normalize_outcome(winning_outcome)
    normalized_bought = _normalize_outcome(str(trade.outcome_bought or ""))

    remaining_shares = Decimal(str(trade.remaining_shares or 0))
    remaining_basis = _calculate_remaining_basis(trade)

    if normalized_winning == "INVALID":
        # При INVALID рынок возвращает стоимость позиции по payout_per_share
        payout = remaining_shares * payout_per_share
    elif normalized_bought == normalized_winning:
        payout = remaining_shares * payout_per_share
    else:
        payout = Decimal("0")

    delta_pnl = payout - settlement_fee_usdc - remaining_basis

    prior_realized = Decimal(str(trade.realized_pnl_usdc or 0))
    new_realized = prior_realized + delta_pnl

    trade.realized_pnl_usdc = new_realized
    trade.pnl = new_realized  # синхронизация legacy-колонки
    trade.remaining_shares = Decimal("0")
    trade.position_status = "CLOSED"
    trade.closed_at = datetime.now(timezone.utc)

    logger.info(
        "settle_resolved_position_done",
        trade_id=trade_id,
        winning_outcome=normalized_winning,
        outcome_bought=normalized_bought,
        payout=str(payout),
        settlement_fee=str(settlement_fee_usdc),
        remaining_basis=str(remaining_basis),
        delta_pnl=str(delta_pnl),
        total_realized_pnl=str(new_realized),
    )
