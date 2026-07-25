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
    yes_bid = getattr(signal, "yes_bid", None)
    yes_ask = getattr(signal, "yes_ask", None)
    if yes_bid is not None and yes_bid > 0 and yes_ask is not None and signal.mid_price > 0:
        spread_pct = (yes_ask - yes_bid) / signal.mid_price
        max_spread = float(config.get("MAX_SPREAD_PCT", 0.08))
        if spread_pct > max_spread:
            return TradeDecision("SKIP", 0.0, 0.0, f"spread too wide: {spread_pct:.2%}", "SKIP", edge=0.0)

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
        return TradeDecision("SKIP", 0, 0, "dead zone", "SKIP", edge=0.0)

    fav_min  = float(config.get("FAVORITE_MIN_PRICE", 0.55))
    fav_max  = float(config.get("FAVORITE_MAX_PRICE", 0.95))
    global_min = float(config.get("MIN_EDGE", 0.05))
    fav_raw = config.get("FAVORITE_MIN_EDGE")
    if fav_raw is not None and str(fav_raw).strip() != "":
        fav_override = float(fav_raw)
        if 0.0 <= fav_override < global_min:
            logger.warning(
                "favorite_min_edge_below_global_floor",
                favorite_min_edge=fav_override,
                global_min=global_min,
                note="FAVORITE_MIN_EDGE < global MIN_EDGE — using global floor"
            )
            min_edge = max(global_min, fav_override)
        else:
            min_edge = fav_override
    else:
        min_edge = global_min

    # ── Spread check ────────────────────────────────────────────────────────
    yes_bid = signal.yes_bid
    yes_ask = signal.yes_ask
    if yes_bid is not None and yes_ask is not None and yes_bid > 0 and signal.mid_price > 0:
        spread_pct = (yes_ask - yes_bid) / signal.mid_price
    elif signal.mid_price > 0 and getattr(signal, "spread", None) is not None:
        spread_pct = signal.spread / signal.mid_price
    else:
        spread_pct = 0.0

    max_spread = float(config.get("MAX_SPREAD_PCT", 0.08))
    if spread_pct > max_spread:
        return TradeDecision("SKIP", 0.0, 0.0, f"spread too wide ({spread_pct:.2%})", "SKIP", edge=0.0)

    candidates: list[TradeDecision] = []

    # --- YES side ---
    if signal.mid_price >= threshold:
        eff_yes_ask = signal.get_yes_ask()
        if fav_min <= eff_yes_ask <= fav_max:
            if signal.yes_bid is not None and float(signal.yes_bid) > 0:
                p_win_yes = float(signal.yes_bid)
            else:
                p_win_yes = signal.mid_price - (eff_yes_ask - signal.mid_price)
            edge = compute_edge(p_win_yes, eff_yes_ask)
            if edge >= min_edge:
                bet = _resolve_final_bet(edge, signal.volume_5min, config)
                candidates.append(TradeDecision(
                    "BUY_YES", eff_yes_ask, bet,
                    f"favorite YES edge={edge:.4f}", "PURE_FAVORITE",
                    edge=edge, p_up=p_win_yes,
                ))

    # --- NO side --- проверяется НЕЗАВИСИМО от YES-side
    if signal.mid_price <= (1.0 - threshold):
        eff_no_ask = signal.get_no_ask()
        if fav_min <= eff_no_ask <= fav_max:
            if signal.no_bid is not None and float(signal.no_bid) > 0:
                no_prob = float(signal.no_bid)
            else:
                no_prob = (1.0 - signal.mid_price) - (eff_no_ask - (1.0 - signal.mid_price))
            edge = compute_edge(no_prob, eff_no_ask)
            if edge >= min_edge:
                bet = _resolve_final_bet(edge, signal.volume_5min, config)
                candidates.append(TradeDecision(
                    "BUY_NO", eff_no_ask, bet,
                    f"favorite NO edge={edge:.4f}", "PURE_FAVORITE",
                    edge=edge, p_up=1.0 - no_prob,
                ))

    if not candidates:
        eff_yes = signal.get_yes_ask()
        eff_no = signal.get_no_ask()
        if signal.mid_price >= threshold and not (fav_min <= eff_yes <= fav_max):
            reason = f"YES price {eff_yes:.3f} out of bounds [{fav_min},{fav_max}]"
        elif signal.mid_price <= (1.0 - threshold) and not (fav_min <= eff_no <= fav_max):
            reason = f"NO price {eff_no:.3f} out of bounds [{fav_min},{fav_max}]"
        else:
            reason = "no clear favorite"
        return TradeDecision("SKIP", 0.0, 0.0, reason, "SKIP", edge=0.0)

    # Выбираем кандидата с наибольшим edge
    best_candidate = max(candidates, key=lambda c: c.edge if c.edge is not None else -999.0)
    return best_candidate


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
        return TradeDecision("SKIP", 0, 0, "dead zone", "SKIP", p_flip=p_flip, edge=0.0)

    # 2. Порог P(flip) < no_flip_threshold
    if p_flip_calibrated >= no_flip_thresh:
        return TradeDecision("SKIP", 0, 0,
            f"p_flip_calibrated={p_flip_calibrated:.3f} >= threshold={no_flip_thresh:.3f}", "SKIP",
            p_flip=p_flip, edge=0.0)

    fav_min = float(config.get("FAVORITE_MIN_PRICE", 0.55))
    fav_max = float(config.get("FAVORITE_MAX_PRICE", 0.95))

    # 3. Определяем сторону и цену входа по фавориту
    if signal.mid_price >= FLIP_MIDPOINT:
        action: ActionType = "BUY_YES"
        buy_price = signal.get_yes_ask()
        if not (fav_min <= buy_price <= fav_max):
            return TradeDecision("SKIP", 0, 0,
                f"YES price {buy_price:.3f} out of [{fav_min},{fav_max}]", "SKIP", p_flip=p_flip, edge=0.0)
    else:
        action: ActionType = "BUY_NO"
        buy_price = signal.get_no_ask()
        if not (fav_min <= buy_price <= fav_max):
            return TradeDecision("SKIP", 0, 0,
                f"NO price {buy_price:.3f} out of [{fav_min},{fav_max}]", "SKIP", p_flip=p_flip, edge=0.0)

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
    dead_zone = float(config.get("DEAD_ZONE_WIDTH", 0.10))
    p_flip_calibrated = apply_ece_correction(p_flip, ece)

    # 1. Сначала проверяем dead zone
    if is_in_dead_zone(signal.mid_price, dead_zone):
        return TradeDecision("SKIP", 0, 0, "dead zone", "SKIP", p_flip=p_flip, edge=0.0)

    is_yes_fav = signal.mid_price >= FLIP_MIDPOINT
    outsider_ask = signal.get_no_ask() if is_yes_fav else signal.get_yes_ask()
    outsider_action: ActionType = "BUY_NO" if is_yes_fav else "BUY_YES"

    if outsider_ask <= 0:
        return TradeDecision("SKIP", 0, 0, "outsider_ask=0", "SKIP", p_flip=p_flip, edge=0.0)

    outsider_pwin_discount = float(config.get("OUTSIDER_PWIN_DISCOUNT", 0.65))
    p_win_outsider = p_flip_calibrated * outsider_pwin_discount
    outsider_edge = compute_edge(p_win_outsider, outsider_ask)

    logger.debug(
        "outsider_p_win_calc",
        p_flip_calibrated=round(p_flip_calibrated, 4),
        discount=outsider_pwin_discount,
        p_win_adjusted=round(p_win_outsider, 4),
        outsider_ask=outsider_ask,
        edge=round(outsider_edge, 4),
    )

    # 2. Потом проверяем порог p_flip
    if p_flip_calibrated < flip_thresh:
        return TradeDecision("SKIP", 0, 0,
            f"p_flip_calibrated={p_flip_calibrated:.3f} < threshold={flip_thresh:.3f}", "SKIP",
            p_flip=p_flip, edge=outsider_edge)

    max_outsider_price = float(config.get("OUTSIDER_MAX_PRICE", 0.45))
    global_min = float(config.get("MIN_EDGE", 0.05))
    no_min_raw = config.get("NO_MIN_EDGE")
    no_min = float(no_min_raw) if no_min_raw is not None and str(no_min_raw).strip() != "" else 0.0
    min_edge = max(global_min, no_min)

    if no_min_raw is not None and str(no_min_raw).strip() != "" and float(no_min_raw) < global_min:
        logger.warning(
            "no_min_edge_overridden_by_global_min",
            no_min_edge=no_min,
            global_min_edge=global_min,
            effective_min_edge=min_edge,
            note="NO_MIN_EDGE in DB is below global MIN_EDGE floor — using global MIN_EDGE"
        )

    edge = outsider_edge

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
    no_ask: Optional[float] = None,
    p_flip_ml: Optional[float] = None,
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
            p_up=crypto.p_up, strike=crypto.strike, edge=0.0
        )

    if not crypto.features_ok:
        return TradeDecision(
            action="SKIP", buy_price=0.0, bet_size_usdc=0.0, 
            reason="Invalid crypto features", strategy_type="LIGHTGBM_TREND", 
            p_up=crypto.p_up, strike=crypto.strike, edge=0.0
        )

    flip_thresh = float(config.get("FLIP_THRESHOLD", 0.60))
    if flip_thresh > 1.0:
        flip_thresh = flip_thresh / 100.0

    if p_flip_ml is not None and crypto.direction == "DOWN":
        if p_flip_ml < flip_thresh:
            return TradeDecision(
                action="SKIP", buy_price=0.0, bet_size_usdc=0.0,
                reason=f"p_flip={p_flip_ml:.3f} < FLIP_THRESHOLD ({flip_thresh:.2f})",
                strategy_type="LIGHTGBM_TREND", p_up=crypto.p_up, strike=crypto.strike, edge=0.0
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
    if action == "BUY_YES":
        actual_buy_price = entry_price
    else:
        actual_buy_price = no_ask if no_ask is not None and no_ask > 0 else round(1.0 - entry_price, 4)
    
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


