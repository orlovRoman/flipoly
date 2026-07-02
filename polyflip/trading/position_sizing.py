"""
Расчёт размера позиции: Kelly criterion, bet sizing, dead zone.
Все функции — чистые (без БД, API, логгера).
"""
from __future__ import annotations


def compute_bet_size_edge_scaled(
    edge: float,
    min_bet_usdc: float,
    max_bet_usdc: float,
    min_edge: float = 0.05,
    max_edge: float = 0.40,
) -> float:
    """
    Линейное масштабирование ставки по силе edge.
    edge=min_edge → min_bet, edge>=max_edge → max_bet.
    Без Kelly, без предположений о калиброванности модели.
    """
    if edge <= 0:
        return 0.0
    t = min(max((edge - min_edge) / (max_edge - min_edge), 0.0), 1.0)
    raw = min_bet_usdc + t * (max_bet_usdc - min_bet_usdc)
    return round(raw, 2)


def compute_bet_size_with_liquidity(
    edge: float,
    volume_5min: float,
    min_bet_usdc: float,
    max_bet_usdc: float,
    min_edge: float = 0.05,
    max_edge: float = 0.20,
    liquidity_fraction: float = 0.05,
) -> float:
    """
    Масштабирует ставку по edge И ограничивает её по ликвидности рынка.
    
    liquidity_fraction: максимальная доля от volume_5min.
    Например 0.05 = не более 5% от объёма за последние 5 минут.
    
    Гарантирует: min_bet_usdc <= result <= min(max_bet_usdc, liquidity_cap).
    """
    raw_bet = compute_bet_size_edge_scaled(
        edge=edge,
        min_bet_usdc=min_bet_usdc,
        max_bet_usdc=max_bet_usdc,
        min_edge=min_edge,
        max_edge=max_edge,
    )
    # liquidity_cap: не менее min_bet чтобы не заблокировать торговлю на тихих рынках
    liquidity_cap = max(volume_5min * liquidity_fraction, min_bet_usdc)
    return round(min(raw_bet, liquidity_cap), 2)

def compute_edge(win_prob: float, buy_price: float) -> float:
    """
    Математическое преимущество: EV/bet - 1.
    edge > 0 → положительное ожидание.
    """
    if buy_price <= 0:
        return -1.0
    return round((win_prob / buy_price) - 1.0, 4)


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
