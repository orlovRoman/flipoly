"""
Чистые функции принятия торговых решений.
НЕТ обращений к БД, API, логгеру.
Используется: engine.py (production), backtesting/strategy.py (backtest).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional

from polyflip.trading.feature_builder import MarketSignal, build_feature_vector
from polyflip.crypto.predictor import CryptoSignal

from polyflip.trading.position_sizing import (
    compute_bet_size_edge_scaled,
    compute_edge, is_in_dead_zone,
    apply_ece_correction
)
from polyflip.constants import FLIP_MIDPOINT, ECE_WARN_THRESHOLD
import structlog


logger = structlog.get_logger(__name__)

def _resolve_final_bet(edge: float, volume_5min: float, config: dict) -> float:
    from polyflip.trading.position_sizing import compute_bet_size_with_liquidity
    min_bet = float(config.get("TRADE_BET_SIZE_USDC", 5.0))
    if config.get("BET_SIZING_MODE") and str(config.get("BET_SIZING_MODE")).lower() == "fixed":
        return min_bet
    bet = compute_bet_size_with_liquidity(
        edge=edge,
        volume_5min=volume_5min,
        min_bet_usdc=min_bet,
        max_bet_usdc=float(config.get("MAX_BET_SIZE_USDC", 50.0)),
        min_edge=float(config.get("MIN_EDGE", 0.05)),
        max_edge=float(config.get("MAX_BET_EDGE", 0.40)),  # масштабирование ставки
        liquidity_fraction=float(config.get("LIQUIDITY_FRACTION", 0.05)),
    )
    if bet < min_bet:
        bet = min_bet
    return bet

StrategyType = Literal["PURE_FAVORITE", "ML_TREND", "OUTSIDER", "LIGHTGBM_TREND", "COMBINED", "SKIP"]
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
    p_up: Optional[float] = None
    strike: Optional[float] = None



def decide_favorite(signal: MarketSignal, config: dict) -> TradeDecision:
    """
    PURE_FAVORITE стратегия.
    Покупает фаворита (YES если mid_price > threshold, NO если < 1-threshold).
    
    Важно: обе стороны проверяются независимо.
    YES-side out-of-bounds НЕ блокирует проверку NO-side.
    Если обе стороны подходят — выбирается с бо́льшим edge.
    """
    raw_fav = str(config.get("FAVORITE_THRESHOLD", "")).strip()
    if not raw_fav:
        threshold = 0.55
        logger.warning(
            "favorite_threshold_default_used",
            threshold=threshold,
            note="Default changed from 0.65 to 0.55 in v1.x — set FAVORITE_THRESHOLD explicitly"
        )
    else:
        try:
            threshold = float(raw_fav)
        except ValueError:
            threshold = 0.55
            logger.warning("favorite_threshold_invalid", raw=raw_fav, fallback=threshold)

    dead_zone = float(config.get("DEAD_ZONE_WIDTH", 0.10))

    if is_in_dead_zone(signal.mid_price, dead_zone):
        return TradeDecision("SKIP", 0, 0, "dead zone", "SKIP")

    fav_min  = float(config.get("FAVORITE_MIN_PRICE", 0.55))
    fav_max  = float(config.get("FAVORITE_MAX_PRICE", 0.95))
    min_edge = float(config.get("FAVORITE_MIN_EDGE", config.get("MIN_EDGE", -0.01)))

    candidates: list[TradeDecision] = []

    # --- YES side ---
    if signal.mid_price >= threshold:
        if fav_min <= signal.yes_ask <= fav_max:
            p_win_yes = signal.mid_price
            edge = compute_edge(p_win_yes, signal.yes_ask)
            if edge >= min_edge:
                bet = _resolve_final_bet(edge, signal.volume_5min, config)
                candidates.append(TradeDecision(
                    "BUY_YES", signal.yes_ask, bet,
                    f"favorite YES edge={edge:.4f}", "PURE_FAVORITE",
                    edge=edge, p_up=p_win_yes,
                ))

    # --- NO side --- проверяется НЕЗАВИСИМО от YES-side
    if signal.mid_price <= (1.0 - threshold):
        if fav_min <= signal.no_ask <= fav_max:
            no_prob = 1.0 - signal.mid_price
            edge = compute_edge(no_prob, signal.no_ask)
            if edge >= min_edge:
                bet = _resolve_final_bet(edge, signal.no_ask)
                candidates.append(TradeDecision(
                    "BUY_NO", signal.no_ask, bet,
                    f"favorite NO edge={edge:.4f}", "PURE_FAVORITE",
                    edge=edge, p_up=1.0 - no_prob,
                ))

    if not candidates:
        if signal.mid_price >= threshold and not (fav_min <= signal.yes_ask <= fav_max):
            reason = f"YES price {signal.yes_ask:.3f} out of bounds [{fav_min},{fav_max}]"
        elif signal.mid_price <= (1.0 - threshold) and not (fav_min <= signal.no_ask <= fav_max):
            reason = f"NO price {signal.no_ask:.3f} out of bounds [{fav_min},{fav_max}]"
        else:
            reason = "no clear favorite"
        return TradeDecision("SKIP", 0.0, 0.0, reason, "SKIP")

    return max(candidates, key=lambda d: d.edge or 0.0)


def decide_ml_trend(
    signal: MarketSignal,
    p_flip: float,
    config: dict,
    ece: float = 0.0,
) -> TradeDecision:
    """
    ML Trend стратегия.
    Если P(flip) < no_flip_threshold → рынок не флипнет → покупаем фаворита.
    config дополнительно ожидает:
      - NO_FLIP_THRESHOLD: float (напр. 0.35)
      - FAVORITE_MIN_PRICE / FAVORITE_MAX_PRICE: float
      - MIN_EDGE / MAX_EDGE: float  ← ML-edge фильтр
    """
    no_flip_thresh = float(config.get("NO_FLIP_THRESHOLD", 0.35))

    p_flip_calibrated = apply_ece_correction(p_flip, ece)
    p_win = 1.0 - p_flip_calibrated

    # 1. Проверяем dead zone
    dead_zone = float(config.get("DEAD_ZONE_WIDTH", 0.10))
    if is_in_dead_zone(signal.mid_price, dead_zone):
        return TradeDecision("SKIP", 0, 0, "dead zone", "SKIP", p_flip=p_flip)

    # 2. Порог P(flip) < no_flip_threshold
    if p_flip_calibrated >= no_flip_thresh:
        return TradeDecision("SKIP", 0, 0,
            f"p_flip_calibrated={p_flip_calibrated:.3f} >= threshold={no_flip_thresh:.3f}", "SKIP",
            p_flip=p_flip)

    fav_min = float(config.get("FAVORITE_MIN_PRICE", 0.55))
    fav_max = float(config.get("FAVORITE_MAX_PRICE", 0.95))

    # 3. Определяем сторону и цену входа по фавориту
    if signal.mid_price >= FLIP_MIDPOINT:
        action: ActionType = "BUY_YES"
        buy_price = signal.yes_ask
        if not (fav_min <= buy_price <= fav_max):
            return TradeDecision("SKIP", 0, 0,
                f"YES price {buy_price:.3f} out of [{fav_min},{fav_max}]", "SKIP", p_flip=p_flip)
    else:
        action: ActionType = "BUY_NO"
        buy_price = signal.no_ask
        if not (fav_min <= buy_price <= fav_max):
            return TradeDecision("SKIP", 0, 0,
                f"NO price {buy_price:.3f} out of [{fav_min},{fav_max}]", "SKIP", p_flip=p_flip)

    if ece and ece > ECE_WARN_THRESHOLD:
        logger.warning("poor_calibration_model", asset=signal.asset, ece=ece, note="p_flip estimates may be unreliable")

    # 4. Единый ML-edge
    edge = compute_edge(p_win, buy_price)
    min_edge = float(config.get("MIN_EDGE", 0.05))
    if edge < min_edge:
        return TradeDecision("SKIP", 0, 0,
            f"Edge={edge:.4f} < min={min_edge:.4f}", "SKIP", p_flip=p_flip, edge=edge)

    # 5. Ставка на основе ML-edge
    bet = _resolve_final_bet(edge, signal.volume_5min, config)
    bypass = str(config.get("BYPASS_BET_SIZE_CHECK", "false")).lower() == "true"
    if bet <= 0 and not bypass:
        return TradeDecision("SKIP", 0, 0, "Bet size 0", "SKIP", p_flip=p_flip, edge=edge)

    return TradeDecision(
        action, buy_price, bet,
        f"ML_TREND p_flip={p_flip:.3f} < {no_flip_thresh:.3f}",
        "ML_TREND",
        p_flip=p_flip, edge=edge
    )


def decide_outsider(
    signal: MarketSignal,
    p_flip: float,
    config: dict,
    ece: float = 0.0,
) -> TradeDecision:
    """
    Outsider стратегия (TRADE_ON_FLIP).
    Если P(flip) >= flip_threshold → рынок флипнет → покупаем аутсайдера.
    """
    flip_thresh = float(config.get("FLIP_THRESHOLD", 0.60))
    p_flip_calibrated = apply_ece_correction(p_flip, ece)

    is_yes_fav = signal.mid_price >= FLIP_MIDPOINT
    outsider_ask = signal.no_ask if is_yes_fav else signal.yes_ask
    outsider_action: ActionType = "BUY_NO" if is_yes_fav else "BUY_YES"

    outsider_edge = compute_edge(p_flip_calibrated, outsider_ask) if outsider_ask > 0 else None

    if p_flip_calibrated < flip_thresh:
        return TradeDecision("SKIP", 0, 0,
            f"p_flip_calibrated={p_flip_calibrated:.3f} < threshold={flip_thresh:.3f}", "SKIP",
            p_flip=p_flip, edge=outsider_edge)

    max_outsider_price = float(config.get("OUTSIDER_MAX_PRICE", 0.45))
    min_edge = float(config.get("NO_MIN_EDGE", config.get("MIN_EDGE", 0.04)))
    dead_zone = float(config.get("DEAD_ZONE_WIDTH", 0.10))

    if is_in_dead_zone(signal.mid_price, dead_zone):
        return TradeDecision("SKIP", 0, 0, "dead zone", "SKIP", p_flip=p_flip, edge=outsider_edge)

    if outsider_ask <= 0:
        return TradeDecision("SKIP", 0, 0, "outsider_ask=0", "SKIP", p_flip=p_flip)

    edge = compute_edge(p_flip_calibrated, outsider_ask)

    if outsider_ask > max_outsider_price:
        return TradeDecision("SKIP", 0, 0,
            f"{outsider_action} ask {outsider_ask:.3f} > max {max_outsider_price}", "SKIP",
            p_flip=p_flip, edge=edge)

    if ece and ece > ECE_WARN_THRESHOLD:
        logger.warning("poor_calibration_model", asset=signal.asset, ece=ece, note="p_flip estimates may be unreliable")

    if edge < min_edge:
        return TradeDecision("SKIP", 0, 0,
            f"edge={edge:.3f} < min={min_edge:.3f}", "SKIP", p_flip=p_flip, edge=edge)

    bet = _resolve_final_bet(edge, signal.volume_5min, config)
    bypass = str(config.get("BYPASS_BET_SIZE_CHECK", "false")).lower() == "true"
    if bet <= 0 and not bypass:
        return TradeDecision("SKIP", 0, 0, "Bet size 0", "SKIP", p_flip=p_flip, edge=edge)

    return TradeDecision(
        outsider_action, outsider_ask, bet,
        f"outsider {outsider_action.split('_')[1]}, p_flip={p_flip:.3f}", "OUTSIDER",
        p_flip=p_flip, edge=edge
    )


def decide_crypto_trend(
    crypto: CryptoSignal,
    entry_price: float,       # Текущая цена YES токена Polymarket рынка
    volume_5min: float,
    config: dict,
) -> TradeDecision:
    """
    Торговая логика для LIGHTGBM_TREND.
    Сигнал UP (рост) -> покупаем YES.
    Сигнал DOWN (падение) -> покупаем NO.
    """
    if entry_price <= 0.0:
        return TradeDecision(
            action="SKIP", buy_price=0.0, bet_size_usdc=0.0,
            reason=f"entry_price={entry_price} invalid",
            strategy_type="LIGHTGBM_TREND",
            p_up=crypto.p_up, strike=crypto.strike
        )

    if not crypto.features_ok:
        return TradeDecision(
            action="SKIP", buy_price=0.0, bet_size_usdc=0.0, 
            reason="Invalid crypto features", strategy_type="LIGHTGBM_TREND", 
            p_up=crypto.p_up, strike=crypto.strike
        )

    min_edge = float(config.get("MIN_EDGE", 0.05))

    if crypto.direction == "NONE" or crypto.edge < min_edge:
        return TradeDecision(
            action="SKIP", buy_price=0.0, bet_size_usdc=0.0,
            reason=f"crypto edge={crypto.edge:.4f} < min_edge={min_edge:.4f}",
            strategy_type="LIGHTGBM_TREND", p_up=crypto.p_up, strike=crypto.strike, edge=crypto.edge
        )

    bet = _resolve_final_bet(crypto.edge, volume_5min, config)
    bypass = str(config.get("BYPASS_BET_SIZE_CHECK", "false")).lower() == "true"
    if bet <= 0 and not bypass:
        return TradeDecision(
            action="SKIP", buy_price=0.0, bet_size_usdc=0.0, 
            reason="Bet size 0", strategy_type="LIGHTGBM_TREND", 
            p_up=crypto.p_up, strike=crypto.strike, edge=crypto.edge
        )

    action: ActionType = "BUY_YES" if crypto.direction == "UP" else "BUY_NO"
    actual_buy_price = entry_price if action == "BUY_YES" else round(1.0 - entry_price, 4)
    
    return TradeDecision(
        action=action,
        buy_price=actual_buy_price,
        bet_size_usdc=bet,
        reason=f"LIGHTGBM_TREND {crypto.symbol} p_up={crypto.p_up:.3f} edge={crypto.edge:.4f}",
        strategy_type="LIGHTGBM_TREND",
        p_up=crypto.p_up,
        strike=crypto.strike,
        edge=crypto.edge
    )


