"""Take-profit логика: расчёт целевой цены и принятие решения о срабатывании."""
from dataclasses import dataclass


_TP_MIN_PRICE = 0.01
_TP_MAX_PRICE = 0.99


def compute_take_profit_price(entry_price: float, multiplier: float) -> float | None:
    """tp_price = entry_price * multiplier.
    Если цель > 0.99 (недостижима на Polymarket), возвращает None.
    """
    if multiplier <= 1.0:
        raise ValueError(f"take_profit_multiplier must be > 1.0, got {multiplier}")
    raw = entry_price * multiplier
    if raw > _TP_MAX_PRICE:
        return None
    return max(_TP_MIN_PRICE, round(raw, 4))


@dataclass
class TakeProfitDecision:
    should_sell: bool
    current_price: float
    tp_price: float | None
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
    if tp_price is None:
        return TakeProfitDecision(
            should_sell=False,
            current_price=current_bid,
            tp_price=None,
            reason="target_exceeds_max_price",
        )

    should_sell = current_bid >= tp_price
    return TakeProfitDecision(
        should_sell=should_sell,
        current_price=current_bid,
        tp_price=tp_price,
        reason=f"bid={current_bid:.4f} >= tp={tp_price:.4f}" if should_sell else "below_target"
    )
