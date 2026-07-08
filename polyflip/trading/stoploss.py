"""Stop-loss логика: расчёт порога и принятие решения о срабатывании."""
from dataclasses import dataclass
from typing import Optional


def compute_stop_price(entry_price: float, stop_loss_pct: float) -> float:
    """stop_price = entry_price * (1 - pct/100). Clamp в [0.01, 0.99]."""
    if not (0 < stop_loss_pct < 100):
        raise ValueError(f"stop_loss_pct must be in (0, 100), got {stop_loss_pct}")
    raw = entry_price * (1.0 - stop_loss_pct / 100.0)
    return max(0.01, min(0.99, raw))


@dataclass
class StopLossDecision:
    should_sell: bool
    current_price: float
    stop_price: float
    reason: str


def evaluate_stop_loss(
    entry_price: float,
    stop_loss_pct: float,
    current_bid: float,   # лучший bid в стакане (цена продажи)
) -> StopLossDecision:
    """Возвращает решение: продавать или нет."""
    stop_price = compute_stop_price(entry_price, stop_loss_pct)
    should_sell = current_bid <= stop_price
    return StopLossDecision(
        should_sell=should_sell,
        current_price=current_bid,
        stop_price=stop_price,
        reason=f"bid={current_bid:.4f} <= stop={stop_price:.4f}" if should_sell else "above_stop"
    )
