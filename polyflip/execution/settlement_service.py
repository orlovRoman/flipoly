"""
Единый сервис закрытия позиций по итогам разрешения рынка.

Правила бухгалтерии:
- entry_cost_usdc = open_gross + open_entry_fees  (включает комиссии входа)
- remaining_basis = avg_entry_cost_per_share * remaining_shares
- payout (YES-победитель) = remaining_shares * 1.0
- payout (INVALID) = remaining_shares * 0.5  (50/50 redemption по Polymarket)
- realized_pnl += payout - remaining_basis
  (Polymarket-комиссии взимаются при match, а не при разрешении рынка.
   settlement_fee_usdc оставлен для будущих gas-charges из chain_transactions.)

Идемпотентен: повторный вызов на CLOSED позиции → no-op.
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


class AccountingInvariantError(Exception):
    """Критическая ошибка бухгалтерских инвариантов — требует ручного разбора."""


def _reconstruct_missing_fields(trade: TradeHistory) -> None:
    """
    Восстанавливает nullable legacy-поля для старых записей.
    Выбрасывает AccountingInvariantError если восстановить невозможно надёжно.
    """
    if trade.remaining_shares is None:
        if trade.entry_filled_shares is not None:
            trade.remaining_shares = trade.entry_filled_shares
        else:
            raise AccountingInvariantError(
                f"Trade {trade.id}: remaining_shares is NULL and cannot be reconstructed"
            )

    if trade.entry_filled_shares is None:
        if trade.executed_price and Decimal(str(trade.executed_price)) > 0:
            trade.entry_filled_shares = Decimal(str(trade.amount_usdc or 0)) / Decimal(
                str(trade.executed_price)
            )
        else:
            raise AccountingInvariantError(
                f"Trade {trade.id}: entry_filled_shares is NULL and executed_price is zero"
            )

    if trade.entry_cost_usdc is None:
        # Для старых записей: basis = gross amount (без fees, т.к. они неизвестны)
        trade.entry_cost_usdc = Decimal(str(trade.amount_usdc or 0))


def _calculate_remaining_basis(trade: TradeHistory) -> Decimal:
    """
    Оставшаяся себестоимость незакрытой части позиции.
    entry_cost_usdc уже включает entry-комиссии.
    """
    entry_filled = Decimal(str(trade.entry_filled_shares or 0))
    entry_cost = Decimal(str(trade.entry_cost_usdc or 0))
    remaining = Decimal(str(trade.remaining_shares or 0))

    if entry_filled <= Decimal("0"):
        return Decimal("0")

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
        Итог рынка (YES / NO / INVALID). Нормализуется автоматически.
    payout_per_share : Decimal
        Выплата за 1 share победителю:
        - 1.0 для нормального исхода
        - 0.5 для INVALID (50/50 redemption)
    settlement_fee_usdc : Decimal
        Gas/network fee (если оплачен пользователем из chain_transactions).
        При match-комиссиях Polymarket — передавать Decimal("0").
    """
    trade = await session.scalar(
        select(TradeHistory).where(TradeHistory.id == trade_id).with_for_update()
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

    # Восстановление null-полей для legacy-записей
    try:
        _reconstruct_missing_fields(trade)
    except AccountingInvariantError as exc:
        logger.error(
            "settle_resolved_position_accounting_error",
            trade_id=trade_id,
            error=str(exc),
        )
        # Переводим в MANUAL_REVIEW_REQUIRED без изменения PnL
        trade.position_status = "MANUAL_REVIEW_REQUIRED"
        return

    normalized_winning = _normalize_outcome(winning_outcome)
    normalized_bought = _normalize_outcome(str(trade.outcome_bought or ""))

    remaining_shares = Decimal(str(trade.remaining_shares or 0))
    remaining_basis = _calculate_remaining_basis(trade)

    if normalized_winning == "INVALID":
        payout = remaining_shares * payout_per_share  # 0.5 per share
    elif normalized_bought == normalized_winning:
        payout = remaining_shares * payout_per_share  # 1.0 per share
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
        payout_per_share=str(payout_per_share),
        settlement_fee=str(settlement_fee_usdc),
        remaining_basis=str(remaining_basis),
        delta_pnl=str(delta_pnl),
        total_realized_pnl=str(new_realized),
    )
