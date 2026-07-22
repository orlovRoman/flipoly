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
from polyflip.constants import FLIP_MIDPOINT
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
    raw_fav = config.get("FAVORITE_THRESHOLD")
    if raw_fav is None or str(raw_fav).strip() == "":
        threshold = 0.55
    else:
        try:
            threshold = float(raw_fav)
        except ValueError:
            threshold = 0.55
            
    if "FAVORITE_THRESHOLD" not in config or config.get("FAVORITE_THRESHOLD") == "":
        logger.warning(
            "favorite_threshold_default_used",
            threshold=threshold,
            note="Default changed from 0.65 to 0.55 in v1.x — set FAVORITE_THRESHOLD explicitly"
        )
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
                bet = _resolve_final_bet(edge, signal.volume_5min, config)
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
        (применяются через вызов decide_favorite)
      - MIN_EDGE / MAX_EDGE: float  ← ML-edge фильтр, отдельный от FAVORITE_MIN_EDGE
    """
    no_flip_thresh = float(config.get("NO_FLIP_THRESHOLD", 0.35))
    # NOTE: default должен совпадать с BacktestConfig.no_flip_threshold (0.35)
    # Если меняешь дефолт — меняй в обоих местах.

    p_flip_calibrated = apply_ece_correction(p_flip, ece)
    p_win = 1.0 - p_flip_calibrated
    fav_ask = signal.yes_ask if signal.mid_price >= 0.50 else signal.no_ask
    trend_edge = compute_edge(p_win, fav_ask) if fav_ask > 0 else None

    if p_flip >= no_flip_thresh:
        return TradeDecision("SKIP", 0, 0,
            f"p_flip={p_flip:.3f} >= threshold={no_flip_thresh:.3f}", "SKIP",
            p_flip=p_flip, edge=trend_edge)

    # Логика выбора стороны такая же как у PURE_FAVORITE,
    # но стратегия помечается как ML_TREND
    decision = decide_favorite(signal, config)
    if decision.action == "SKIP":
        return TradeDecision("SKIP", 0, 0, decision.reason, "SKIP", p_flip=p_flip, edge=trend_edge)

    ECE_WARN_THRESHOLD = 0.07
    if ece and ece > ECE_WARN_THRESHOLD:
        logger.warning("poor_calibration_model", asset=signal.asset, ece=ece, note="p_flip estimates may be unreliable")

    buy_price = decision.buy_price
    edge = compute_edge(p_win, buy_price)
    
    min_edge = float(config.get("MIN_EDGE", 0.05))
    # MAX_EDGE_FILTER как фильтр аномального edge (SKIP если edge > filter)
    max_edge = float(config.get("MAX_EDGE_FILTER", config.get("MAX_BET_EDGE", 0.75)))
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
    ece: float = 0.0,
) -> TradeDecision:
    """
    Outsider стратегия (TRADE_ON_FLIP).
    Если P(flip) >= flip_threshold → рынок флипнет → покупаем аутсайдера.
    config дополнительно ожидает:
      - FLIP_THRESHOLD: float (напр. 0.60)
      - OUTSIDER_MAX_PRICE: float (напр. 0.45) — не брать аутсайдера дороже этой цены
    """
    flip_thresh = float(config.get("FLIP_THRESHOLD", 0.60))

    p_flip_calibrated = apply_ece_correction(p_flip, ece)
    outsider_ask = signal.no_ask if signal.mid_price >= FLIP_MIDPOINT else signal.yes_ask
    outsider_edge = compute_edge(p_flip_calibrated, outsider_ask) if outsider_ask > 0 else None

    if p_flip < flip_thresh:
        return TradeDecision("SKIP", 0, 0,
            f"p_flip={p_flip:.3f} < threshold={flip_thresh:.3f}", "SKIP",
            p_flip=p_flip, edge=outsider_edge)

    max_outsider_price = float(config.get("OUTSIDER_MAX_PRICE", 0.45))
    min_edge = float(config.get("NO_MIN_EDGE", config.get("MIN_EDGE", 0.04)))
    # DEAD_ZONE_WIDTH — единый параметр ширины мёртвой зоны
    dead_zone = float(config.get("DEAD_ZONE_WIDTH", 0.10))

    if is_in_dead_zone(signal.mid_price, dead_zone):
        return TradeDecision("SKIP", 0, 0, "dead zone", "SKIP", p_flip=p_flip, edge=outsider_edge)

    # Аутсайдер: если YES дорогой — покупаем NO, и наоборот
    if signal.mid_price >= FLIP_MIDPOINT:
        # YES — фаворит, покупаем NO (аутсайдера)
        edge = compute_edge(p_flip_calibrated, signal.no_ask)
        if signal.no_ask > max_outsider_price:
            return TradeDecision("SKIP", 0, 0,
                f"NO ask {signal.no_ask:.3f} > max_outsider_price {max_outsider_price}", "SKIP",
                p_flip=p_flip, edge=edge)
        ECE_WARN_THRESHOLD = 0.07
        if ece and ece > ECE_WARN_THRESHOLD:
            logger.warning("poor_calibration_model", asset=signal.asset, ece=ece, note="p_flip estimates may be unreliable")
        if edge < min_edge:
            return TradeDecision("SKIP", 0, 0, f"edge {edge:.3f} < min", "SKIP", p_flip=p_flip, edge=edge)

        bet = _resolve_final_bet(edge, signal.volume_5min, config)
        bypass = str(config.get("BYPASS_BET_SIZE_CHECK", "false")).lower() == "true"
        if bet <= 0 and not bypass:
            return TradeDecision("SKIP", 0, 0, "Bet size 0", "SKIP", p_flip=p_flip, edge=edge)

        return TradeDecision("BUY_NO", signal.no_ask, bet,
            f"outsider NO, p_flip={p_flip:.3f}", "OUTSIDER",
            p_flip=p_flip, edge=edge)
    else:
        # NO — фаворит, покупаем YES (аутсайдера)
        edge = compute_edge(p_flip_calibrated, signal.yes_ask)
        if signal.yes_ask > max_outsider_price:
            return TradeDecision("SKIP", 0, 0,
                f"YES ask {signal.yes_ask:.3f} > max_outsider_price {max_outsider_price}", "SKIP",
                p_flip=p_flip, edge=edge)
        ECE_WARN_THRESHOLD = 0.07
        if ece and ece > ECE_WARN_THRESHOLD:
            logger.warning("poor_calibration_model", asset=signal.asset, ece=ece, note="p_flip estimates may be unreliable")
        if edge < min_edge:
            return TradeDecision("SKIP", 0, 0, f"edge {edge:.3f} < min", "SKIP", p_flip=p_flip, edge=edge)

        bet = _resolve_final_bet(edge, signal.volume_5min, config)
        bypass = str(config.get("BYPASS_BET_SIZE_CHECK", "false")).lower() == "true"
        if bet <= 0 and not bypass:
            return TradeDecision("SKIP", 0, 0, "Bet size 0", "SKIP", p_flip=p_flip, edge=edge)

        return TradeDecision("BUY_YES", signal.yes_ask, bet,
            f"outsider YES, p_flip={p_flip:.3f}", "OUTSIDER",
            p_flip=p_flip, edge=edge)


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

    min_edge = float(config.get("CRYPTO_MIN_EDGE", config.get("MIN_EDGE", 0.04)))
    max_edge = float(config.get("MAX_BET_EDGE", config.get("MAX_EDGE_FILTER", 0.35)))

    if crypto.direction == "NONE" or crypto.edge < min_edge:
        return TradeDecision(
            action="SKIP", buy_price=0.0, bet_size_usdc=0.0,
            reason=f"crypto edge={crypto.edge:.4f} < min_edge={min_edge:.4f}",
            strategy_type="LIGHTGBM_TREND", p_up=crypto.p_up, strike=crypto.strike, edge=crypto.edge
        )

    if crypto.edge > max_edge:
        return TradeDecision(
            action="SKIP", buy_price=0.0, bet_size_usdc=0.0,
            reason=f"crypto edge={crypto.edge:.4f} > max_edge={max_edge:.4f} (suspicious)",
            strategy_type="LIGHTGBM_TREND", p_up=crypto.p_up, strike=crypto.strike, edge=crypto.edge
        )

    # Стандартизованный сайзинг ставок с подменой MIN_EDGE на CRYPTO_MIN_EDGE
    crypto_config = {**config, "MIN_EDGE": min_edge}
    bet = _resolve_final_bet(crypto.edge, volume_5min, crypto_config)
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


