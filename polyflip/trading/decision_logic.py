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
    compute_kelly_fraction, compute_bet_size,
    compute_edge, is_in_dead_zone
)

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
    kelly_fraction: Optional[float] = None


def decide_favorite(signal: MarketSignal, config: dict) -> TradeDecision:
    """
    PURE_FAVORITE стратегия.
    Покупает фаворита (YES если mid_price > threshold, NO если < 1-threshold).
    config ожидает ключи:
      - FAVORITE_THRESHOLD: float (напр. 0.65)
      - MIN_EDGE: float (напр. 0.02)
      - MAX_EDGE: float (напр. 0.15)
      - YES_MIN_PRICE / YES_MAX_PRICE: float
      - NO_MIN_PRICE / NO_MAX_PRICE: float
      - AUTO_DEAD_ZONE_WIDTH: float
      - INITIAL_CAPITAL: float
      - KELLY_MULTIPLIER: float
      - TRADE_BET_SIZE_USDC: float (min bet)
      - MAX_BET_SIZE_USDC: float
    """
    threshold = float(config.get("FAVORITE_THRESHOLD", 0.55))
    min_edge  = float(config.get("MIN_EDGE", 0.05))
    dead_zone = float(config.get("AUTO_DEAD_ZONE_WIDTH", 0.10))

    if is_in_dead_zone(signal.mid_price, dead_zone):
        return TradeDecision("SKIP", 0, 0, "dead zone", "SKIP")

    # --- YES side ---
    if signal.mid_price >= threshold:
        yes_min = float(config.get("YES_MIN_PRICE", 0.55))
        yes_max = float(config.get("YES_MAX_PRICE", 0.95))
        if not (yes_min <= signal.yes_ask <= yes_max):
            return TradeDecision("SKIP", 0, 0,
                f"YES price {signal.yes_ask:.3f} out of bounds [{yes_min},{yes_max}]", "SKIP")
        bet = float(config.get("TRADE_BET_SIZE_USDC", 5))
        return TradeDecision("BUY_YES", signal.yes_ask, bet,
            f"favorite YES", "PURE_FAVORITE",
            edge=0.0, kelly_fraction=0.0)

    # --- NO side ---
    if signal.mid_price <= (1.0 - threshold):
        no_min = float(config.get("NO_MIN_PRICE", 0.55))
        no_max = float(config.get("NO_MAX_PRICE", 0.95))
        no_prob = 1.0 - signal.mid_price
        if not (no_min <= signal.no_ask <= no_max):
            return TradeDecision("SKIP", 0, 0,
                f"NO price {signal.no_ask:.3f} out of bounds [{no_min},{no_max}]", "SKIP")
        bet = float(config.get("TRADE_BET_SIZE_USDC", 5))
        return TradeDecision("BUY_NO", signal.no_ask, bet,
            f"favorite NO", "PURE_FAVORITE",
            edge=0.0, kelly_fraction=0.0)

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
    """
    no_flip_thresh = float(config.get("NO_FLIP_THRESHOLD", 0.35))

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

    kelly_enabled = config.get("KELLY_ENABLED", True)
    if isinstance(kelly_enabled, str):
        kelly_enabled = kelly_enabled.lower() == "true"

    if kelly_enabled:
        kf = compute_kelly_fraction(p_win, buy_price)
        bet = compute_bet_size(
            kf,
            float(config.get("INITIAL_CAPITAL", 1000)),
            float(config.get("KELLY_MULTIPLIER", 0.25)),
            float(config.get("TRADE_BET_SIZE_USDC", 5)),
            float(config.get("MAX_BET_SIZE_USDC", 50)),
        )
        if bet <= 0:
            return TradeDecision("SKIP", 0, 0, "Kelly=0", "SKIP", p_flip=p_flip, edge=edge)
    else:
        kf = None
        bet = float(config.get("TRADE_BET_SIZE_USDC", 5))

    return TradeDecision(
        decision.action, buy_price, bet,
        f"ML_TREND p_flip={p_flip:.3f} < {no_flip_thresh:.3f}, {decision.reason}",
        "ML_TREND",
        p_flip=p_flip, edge=edge, kelly_fraction=kf,
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
      - OUTSIDER_NO_MIN_PRICE / OUTSIDER_NO_MAX_PRICE
    """
    flip_thresh = float(config.get("FLIP_THRESHOLD", 0.60))

    if p_flip < flip_thresh:
        return TradeDecision("SKIP", 0, 0,
            f"p_flip={p_flip:.3f} < threshold={flip_thresh:.3f}", "SKIP",
            p_flip=p_flip)

    min_edge = float(config.get("MIN_EDGE", 0.02))
    dead_zone = float(config.get("AUTO_DEAD_ZONE_WIDTH", 0.10))

    if is_in_dead_zone(signal.mid_price, dead_zone):
        return TradeDecision("SKIP", 0, 0, "dead zone", "SKIP", p_flip=p_flip)

    # Аутсайдер: если YES дорогой — покупаем NO, и наоборот
    if signal.mid_price >= 0.5:
        # YES — фаворит, покупаем NO (аутсайдера)
        no_min = float(config.get("OUTSIDER_NO_MIN_PRICE", config.get("NO_MIN_PRICE", 0.10)))
        no_max = float(config.get("OUTSIDER_NO_MAX_PRICE", config.get("NO_MAX_PRICE", 0.50)))
        no_prob = 1.0 - signal.mid_price
        if not (no_min <= signal.no_ask <= no_max):
            return TradeDecision("SKIP", 0, 0,
                f"outsider NO price {signal.no_ask:.3f} out of [{no_min},{no_max}]", "SKIP",
                p_flip=p_flip)
        edge = compute_edge(no_prob, signal.no_ask)
        if edge < min_edge:
            return TradeDecision("SKIP", 0, 0, f"edge {edge:.3f} < min", "SKIP", p_flip=p_flip)
        kelly_enabled = config.get("KELLY_ENABLED", True)
        if isinstance(kelly_enabled, str):
            kelly_enabled = kelly_enabled.lower() == "true"
        if kelly_enabled:
            kf = compute_kelly_fraction(no_prob, signal.no_ask)
            bet = compute_bet_size(
                kf,
                float(config.get("INITIAL_CAPITAL", 1000)),
                float(config.get("KELLY_MULTIPLIER", 0.25)),
                float(config.get("TRADE_BET_SIZE_USDC", 5)),
                float(config.get("MAX_BET_SIZE_USDC", 50)),
            )
            if bet <= 0:
                return TradeDecision("SKIP", 0, 0, "Kelly=0", "SKIP", p_flip=p_flip)
        else:
            kf = None
            bet = float(config.get("TRADE_BET_SIZE_USDC", 5))

        return TradeDecision("BUY_NO", signal.no_ask, bet,
            f"outsider NO, p_flip={p_flip:.3f}", "OUTSIDER",
            p_flip=p_flip, edge=edge, kelly_fraction=kf)
    else:
        # NO — фаворит, покупаем YES (аутсайдера)
        yes_min = float(config.get("OUTSIDER_YES_MIN_PRICE", config.get("YES_MIN_PRICE", 0.05)))
        yes_max = float(config.get("OUTSIDER_YES_MAX_PRICE", config.get("YES_MAX_PRICE", 0.45)))
        yes_prob = signal.mid_price
        if not (yes_min <= signal.yes_ask <= yes_max):
            return TradeDecision("SKIP", 0, 0,
                f"outsider YES price {signal.yes_ask:.3f} out of [{yes_min},{yes_max}]", "SKIP",
                p_flip=p_flip)
        edge = compute_edge(yes_prob, signal.yes_ask)
        if edge < min_edge:
            return TradeDecision("SKIP", 0, 0, f"edge {edge:.3f} < min", "SKIP", p_flip=p_flip)
        kelly_enabled = config.get("KELLY_ENABLED", True)
        if isinstance(kelly_enabled, str):
            kelly_enabled = kelly_enabled.lower() == "true"
        if kelly_enabled:
            kf = compute_kelly_fraction(yes_prob, signal.yes_ask)
            bet = compute_bet_size(
                kf,
                float(config.get("INITIAL_CAPITAL", 1000)),
                float(config.get("KELLY_MULTIPLIER", 0.25)),
                float(config.get("TRADE_BET_SIZE_USDC", 5)),
                float(config.get("MAX_BET_SIZE_USDC", 50)),
            )
            if bet <= 0:
                return TradeDecision("SKIP", 0, 0, "Kelly=0", "SKIP", p_flip=p_flip)
        else:
            kf = None
            bet = float(config.get("TRADE_BET_SIZE_USDC", 5))

        return TradeDecision("BUY_YES", signal.yes_ask, bet,
            f"outsider YES, p_flip={p_flip:.3f}", "OUTSIDER",
            p_flip=p_flip, edge=edge, kelly_fraction=kf)
