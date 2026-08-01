"""Take-profit логика: расчёт целевой цены и принятие решения о срабатывании."""
from dataclasses import dataclass


def compute_take_profit_price(entry_price: float, multiplier: float) -> float:
    """tp_price = entry_price * multiplier. Clamp в [0.01, 0.99]."""
    if multiplier <= 1.0:
        raise ValueError(f"take_profit_multiplier must be > 1.0, got {multiplier}")
    raw = entry_price * multiplier
    return max(0.01, min(0.99, round(raw, 4)))


@dataclass
class TakeProfitDecision:
    should_sell: bool
    current_price: float
    tp_price: float
    reason: str


def evaluate_take_profit(
    entry_price: float,
    tp_multiplier: float,
    current_bid: float,   # лучший bid в стакане (цена, по которой покупатели готовы выкупить токен)
) -> TakeProfitDecision:
    """Возвращает решение: зафиксировать прибыль или нет.

    Использует best_bid (аналогично stoploss_worker), так как продажа
    токенов на Polymarket исполняется по цене лучшего покупателя (bid).
    """
    tp_price = compute_take_profit_price(entry_price, tp_multiplier)
    should_sell = current_bid >= tp_price
    return TakeProfitDecision(
        should_sell=should_sell,
        current_price=current_bid,
        tp_price=tp_price,
        reason=f"bid={current_bid:.4f} >= tp={tp_price:.4f}" if should_sell else "below_target"
    )
