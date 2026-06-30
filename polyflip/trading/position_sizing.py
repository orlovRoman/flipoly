"""
Расчёт размера позиции: Kelly criterion, bet sizing, dead zone.
Все функции — чистые (без БД, API, логгера).
"""
from __future__ import annotations


def compute_kelly_fraction(win_prob: float, buy_price: float) -> float:
    """
    Kelly criterion для бинарного рынка.
    
    win_prob: вероятность выигрыша (например, mid_price для YES)
    buy_price: цена покупки (ask-цена)
    
    Формула: f* = (p*b - q) / b, где b = (1/price - 1)
    Возвращает долю капитала [0.0, 1.0].
    """
    if buy_price <= 0 or buy_price >= 1:
        return 0.0
    b = (1.0 / buy_price) - 1.0  # payoff ratio
    q = 1.0 - win_prob
    if b <= 0:
        return 0.0
    kelly = (win_prob * b - q) / b
    return max(0.0, min(1.0, kelly))


def compute_bet_size(
    kelly_fraction: float,
    capital_usdc: float,
    kelly_multiplier: float,  # дробный Kelly, обычно 0.25
    min_bet_usdc: float,
    max_bet_usdc: float,
) -> float:
    """
    Итоговый размер ставки с учётом Kelly, капитала и лимитов.
    Возвращает 0.0 если Kelly = 0 (нет преимущества).
    """
    if kelly_fraction <= 0:
        return 0.0
    raw = capital_usdc * kelly_fraction * kelly_multiplier
    return round(max(min_bet_usdc, min(raw, max_bet_usdc)), 2)


def compute_edge(win_prob: float, buy_price: float) -> float:
    """
    Математическое преимущество: EV/bet - 1.
    edge > 0 → положительное ожидание.
    """
    if buy_price <= 0:
        return -1.0
    return round(win_prob - buy_price, 4)


def is_in_dead_zone(mid_price: float, dead_zone_width: float) -> bool:
    """
    True если рынок слишком близко к 0.5 — нет явного фаворита.
    dead_zone_width=0.1 означает [0.45, 0.55] — мёртвая зона.
    """
    return abs(mid_price - 0.5) < dead_zone_width / 2


def apply_polymarket_fee(gross_pnl: float, fee_rate: float = 0.002) -> float:
    """
    Применяет комиссию Polymarket (0.2% от выплаты).
    Используется только для расчёта PnL в бэктесте.
    """
    return gross_pnl * (1.0 - fee_rate)
