"""
Чистые функции принятия торговых решений.
НЕТ обращений к БД, API, логгеру.
Используется: engine.py (production), backtesting/strategy.py (backtest).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional

from polyflip.trading.feature_builder import MarketSignal, build_feature_vector
from polyflip.trading.position_sizing import (
    compute_bet_size_edge_scaled,
    compute_edge, is_in_dead_zone
)
import structlog

logger = structlog.get_logger(__name__)

def _resolve_final_bet(edge: float, volume_5min: float, config: dict) -> float:
    from polyflip.trading.position_sizing import compute_bet_size_with_liquidity
    min_bet = float(config.get("TRADE_BET_SIZE_USDC", 5))
    bet = compute_bet_size_with_liquidity(
        edge=edge,
        volume_5min=volume_5min,
        min_bet_usdc=min_bet,
        max_bet_usdc=float(config.get("MAX_BET_SIZE_USDC", 50)),
        min_edge=float(config.get("MIN_EDGE", 0.05)),
        max_edge=float(config.get("MAX_EDGE", 0.40)),
        liquidity_fraction=float(config.get("LIQUIDITY_FRACTION", 0.05)),
    )
    # При edge < min_edge_scaled, compute_bet_size_edge_scaled возвращает 0.
    # Это ожидаемо для PURE_FAVORITE где FAVORITE_MIN_EDGE может быть отрицательным:
    # edge уже прошёл фильтр decide_favorite, значит он > FAVORITE_MIN_EDGE,
    # но может быть < MIN_EDGE (который используется при масштабировании).
    # В этом случае ставим min_bet — минимально допустимый размер позиции.
    if bet < min_bet:
        bet = min_bet
    return bet

StrategyType = Literal["PURE_FAVORITE", "ML_TREND", "OUTSIDER", "SKIP"]
ActionType = Literal["BUY_YES", "BUY_NO", "SKIP"]


@dataclass(frozen=True)
class TradeDecision:
    action: ActionType
    buy_price: float
    bet_size_usdc: float
    reason: str
    strategy_type: StrategyType
    p_flip: Optional[float] = None
    edge: Optional[float] = None


def decide_favorite(signal: MarketSignal, config: dict) -> TradeDecision:
    """
    PURE_FAVORITE стратегия.
    Покупает фаворита (YES если mid_price > threshold, NO если < 1-threshold).
    config ожидает ключи:
      - FAVORITE_THRESHOLD: float (напр. 0.65)
      - MIN_EDGE: float (напр. 0.02)
      - MAX_EDGE: float (напр. 0.15)
      - FAVORITE_MIN_PRICE / FAVORITE_MAX_PRICE: float
      - AUTO_DEAD_ZONE_WIDTH: float
      - INITIAL_CAPITAL: float
      - KELLY_MULTIPLIER: float
      - TRADE_BET_SIZE_USDC: float (min bet)
      - MAX_BET_SIZE_USDC: float
    """
    threshold = float(config.get("FAVORITE_THRESHOLD", 0.55))
    if "FAVORITE_THRESHOLD" not in config:
        logger.warning(
            "favorite_threshold_default_used",
            threshold=threshold,
            note="Default changed from 0.65 to 0.55 in v1.x — set FAVORITE_THRESHOLD explicitly"
        )
    dead_zone = float(config.get("AUTO_DEAD_ZONE_WIDTH", 0.10))

    if is_in_dead_zone(signal.mid_price, dead_zone):
        return TradeDecision("SKIP", 0, 0, "dead zone", "SKIP")

    fav_min = float(config.get("FAVORITE_MIN_PRICE", 0.55))
    fav_max = float(config.get("FAVORITE_MAX_PRICE", 0.95))

    # --- YES side ---
    if signal.mid_price >= threshold:
        if not (fav_min <= signal.yes_ask <= fav_max):
            return TradeDecision("SKIP", 0, 0,
                f"YES price {signal.yes_ask:.3f} out of bounds [{fav_min},{fav_max}]", "SKIP")
        p_win_yes = signal.mid_price
        edge = compute_edge(p_win_yes, signal.yes_ask)
        min_edge = float(config.get("FAVORITE_MIN_EDGE", config.get("MIN_EDGE", -0.01)))
        if edge < min_edge:
            return TradeDecision("SKIP", 0, 0,
                f"favorite YES edge={edge:.4f} < min_edge={min_edge:.4f}", "SKIP",
                edge=edge)
        bet = _resolve_final_bet(edge, signal.volume_5min, config)
        return TradeDecision("BUY_YES", signal.yes_ask, bet,
            f"favorite YES edge={edge:.4f}", "PURE_FAVORITE",
            edge=edge)

    # --- NO side ---
    if signal.mid_price <= (1.0 - threshold):
        no_prob = 1.0 - signal.mid_price
        if not (fav_min <= signal.no_ask <= fav_max):
            return TradeDecision("SKIP", 0, 0,
                f"NO price {signal.no_ask:.3f} out of bounds [{fav_min},{fav_max}]", "SKIP")
        edge = compute_edge(no_prob, signal.no_ask)
        min_edge = float(config.get("FAVORITE_MIN_EDGE", config.get("MIN_EDGE", -0.01)))
        if edge < min_edge:
            return TradeDecision("SKIP", 0, 0,
                f"favorite NO edge={edge:.4f} < min_edge={min_edge:.4f}", "SKIP",
                edge=edge)
        bet = _resolve_final_bet(edge, signal.volume_5min, config)
        return TradeDecision("BUY_NO", signal.no_ask, bet,
            f"favorite NO edge={edge:.4f}", "PURE_FAVORITE",
            edge=edge)

    return TradeDecision("SKIP", 0, 0, "no clear favorite", "SKIP")


def decide_ml_trend(
    signal: MarketSignal,
    p_flip: float,
    config: dict,
) -> TradeDecision:
    """
    ML Trend стратегия.
    Если P(flip) < no_flip_threshold → рынок не флипнет → покупаем фаворита.
    config дополнительно ожидает:
      - NO_FLIP_THRESHOLD: float (напр. 0.35)
      - FAVORITE_MIN_PRICE / FAVORITE_MAX_PRICE: float
        (применяются через вызов decide_favorite)
      - MIN_EDGE / MAX_EDGE: float  ← ML-edge фильтр, отдельный от FAVORITE_MIN_EDGE
    """
    no_flip_thresh = float(config.get("NO_FLIP_THRESHOLD", 0.35))
    # NOTE: default должен совпадать с BacktestConfig.no_flip_threshold (0.35)
    # Если меняешь дефолт — меняй в обоих местах.

    if p_flip >= no_flip_thresh:
        return TradeDecision("SKIP", 0, 0,
            f"p_flip={p_flip:.3f} >= threshold={no_flip_thresh:.3f}", "SKIP",
            p_flip=p_flip)

    # Логика выбора стороны такая же как у PURE_FAVORITE,
    # но стратегия помечается как ML_TREND
    decision = decide_favorite(signal, config)
    if decision.action == "SKIP":
        return TradeDecision("SKIP", 0, 0, decision.reason, "SKIP", p_flip=p_flip)

    p_win = 1.0 - p_flip
    buy_price = decision.buy_price

    edge = compute_edge(p_win, buy_price)
    
    min_edge = float(config.get("MIN_EDGE", 0.05))
    max_edge = float(config.get("MAX_EDGE", 0.40))
    if edge < min_edge or edge > max_edge:
        return TradeDecision("SKIP", 0, 0, f"Edge out of bounds (edge={edge:.4f})", "SKIP", p_flip=p_flip, edge=edge)

    bet = _resolve_final_bet(edge, signal.volume_5min, config)
    bypass = str(config.get("BYPASS_BET_SIZE_CHECK", "false")).lower() == "true"
    if bet <= 0 and not bypass:
        return TradeDecision("SKIP", 0, 0, "Bet size 0", "SKIP", p_flip=p_flip, edge=edge)

    return TradeDecision(
        decision.action, buy_price, bet,
        f"ML_TREND p_flip={p_flip:.3f} < {no_flip_thresh:.3f}, {decision.reason}",
        "ML_TREND",
        p_flip=p_flip, edge=edge
    )


def decide_outsider(
    signal: MarketSignal,
    p_flip: float,
    config: dict,
) -> TradeDecision:
    """
    Outsider стратегия (TRADE_ON_FLIP).
    Если P(flip) >= flip_threshold → рынок флипнет → покупаем аутсайдера.
    config дополнительно ожидает:
      - FLIP_THRESHOLD: float (напр. 0.60)
      - OUTSIDER_MAX_PRICE: float (напр. 0.45) — не брать аутсайдера дороже этой цены
    """
    flip_thresh = float(config.get("FLIP_THRESHOLD", 0.60))

    if p_flip < flip_thresh:
        return TradeDecision("SKIP", 0, 0,
            f"p_flip={p_flip:.3f} < threshold={flip_thresh:.3f}", "SKIP",
            p_flip=p_flip)

    max_outsider_price = float(config.get("OUTSIDER_MAX_PRICE", 0.45))
    min_edge = float(config.get("MIN_EDGE", 0.02))
    dead_zone = float(config.get("AUTO_DEAD_ZONE_WIDTH", 0.10))

    if is_in_dead_zone(signal.mid_price, dead_zone):
        return TradeDecision("SKIP", 0, 0, "dead zone", "SKIP", p_flip=p_flip)

    # Аутсайдер: если YES дорогой — покупаем NO, и наоборот
    if signal.mid_price >= 0.5:
        # YES — фаворит, покупаем NO (аутсайдера)
        if signal.no_ask > max_outsider_price:
            return TradeDecision("SKIP", 0, 0,
                f"NO ask {signal.no_ask:.3f} > max_outsider_price {max_outsider_price}", "SKIP",
                p_flip=p_flip)
        edge = compute_edge(p_flip, signal.no_ask)
        if edge < min_edge:
            return TradeDecision("SKIP", 0, 0, f"edge {edge:.3f} < min", "SKIP", p_flip=p_flip)

        bet = _resolve_final_bet(edge, signal.volume_5min, config)
        bypass = str(config.get("BYPASS_BET_SIZE_CHECK", "false")).lower() == "true"
        if bet <= 0 and not bypass:
            return TradeDecision("SKIP", 0, 0, "Bet size 0", "SKIP", p_flip=p_flip)

        return TradeDecision("BUY_NO", signal.no_ask, bet,
            f"outsider NO, p_flip={p_flip:.3f}", "OUTSIDER",
            p_flip=p_flip, edge=edge)
    else:
        # NO — фаворит, покупаем YES (аутсайдера)
        if signal.yes_ask > max_outsider_price:
            return TradeDecision("SKIP", 0, 0,
                f"YES ask {signal.yes_ask:.3f} > max_outsider_price {max_outsider_price}", "SKIP",
                p_flip=p_flip)
        edge = compute_edge(p_flip, signal.yes_ask)
        if edge < min_edge:
            return TradeDecision("SKIP", 0, 0, f"edge {edge:.3f} < min", "SKIP", p_flip=p_flip)

        bet = _resolve_final_bet(edge, signal.volume_5min, config)
        bypass = str(config.get("BYPASS_BET_SIZE_CHECK", "false")).lower() == "true"
        if bet <= 0 and not bypass:
            return TradeDecision("SKIP", 0, 0, "Bet size 0", "SKIP", p_flip=p_flip)

        return TradeDecision("BUY_YES", signal.yes_ask, bet,
            f"outsider YES, p_flip={p_flip:.3f}", "OUTSIDER",
            p_flip=p_flip, edge=edge)

