"""
Расчёт размера позиции: Kelly criterion, bet sizing, dead zone.
Все функции — чистые (без БД, API, логгера).
"""
from __future__ import annotations

from polyflip.constants import (
    MIN_EDGE, MAX_EDGE_SCALING, MAX_EDGE_FILTER,
    LIQUIDITY_FRACTION, POLYMARKET_FEE_RATE,
    INVALID_EDGE_SENTINEL, FLIP_MIDPOINT
)
def compute_bet_size_edge_scaled(
    edge: float,
    min_bet_usdc: float,
    max_bet_usdc: float,
    min_edge: float = MIN_EDGE,
    max_edge: float = MAX_EDGE_SCALING,
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
    min_edge: float = MIN_EDGE,
    max_edge: float = MAX_EDGE_FILTER,
    liquidity_fraction: float = LIQUIDITY_FRACTION,
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
        return INVALID_EDGE_SENTINEL
    return round((win_prob / buy_price) - 1.0, 4)


def is_in_dead_zone(mid_price: float, dead_zone_width: float) -> bool:
    """
    True если рынок слишком близко к 0.5 — нет явного фаворита.
    dead_zone_width=0.1 означает [0.45, 0.55] — мёртвая зона.
    """
    return abs(mid_price - FLIP_MIDPOINT) < dead_zone_width / 2


def apply_polymarket_fee(gross_pnl: float, fee_rate: float = POLYMARKET_FEE_RATE) -> float:
    """
    Применяет комиссию Polymarket (0.2% от выплаты).
    Используется только для расчёта PnL в бэктесте.
    """
    return gross_pnl * (1.0 - fee_rate)


def apply_ece_correction(p: float, ece: float) -> float:
    """
    Консервативная коррекция: если ECE высокий — сжимаем уверенность к 0.5.
    Формула: p_corrected = 0.5 + (p - 0.5) * (1 - min(ece / 0.1, 1.0))
    При ECE=0.0 → без изменений. При ECE>=0.1 → всё схлопывается к 0.5.
    """
    if ece is None or ece <= 0.0:
        return p
    shrink = max(0.0, 1.0 - ece / 0.10)
    return 0.5 + (p - 0.5) * shrink
