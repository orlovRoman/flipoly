"""
SettlementSnapshot — результат опроса settlement-состояния ордера и его trades.
"""
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any


@dataclass(frozen=True)
class SettlementSnapshot:
    """Снимок состояния settlement для одного ордера."""

    order_status: str
    """Статус самого ордера (live, matched, delayed, unmatched…)."""

    confirmed_trades: tuple[Any, ...] = field(default_factory=tuple)
    """Объекты trades со статусом CONFIRMED."""

    is_terminal: bool = False
    """True если все trades достигли терминального статуса."""

    all_failed: bool = False
    """True если is_terminal и ни одного CONFIRMED trade."""
