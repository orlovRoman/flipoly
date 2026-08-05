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
    compute_edge,
    is_in_dead_zone,
    apply_ece_correction
)
from polyflip.constants import FLIP_MIDPOINT, ECE_WARN_THRESHOLD
from polyflip.trading.trading_config import parse_trading_settings
import structlog


logger = structlog.get_logger(__name__)

def _resolve_final_bet(edge: float, volume_5min: float, config: dict, is_outsider: bool = False) -> float:
    """Рассчитывает размер ставки. config — сырой dict (для обратной совместимости с бэктестом)."""
    from polyflip.trading.position_sizing import compute_bet_size_with_liquidity
    cfg = parse_trading_settings(config)
    if cfg.bet_sizing_mode == "fixed":
        return cfg.bet_size
    bet = compute_bet_size_with_liquidity(
        edge=edge,
        volume_5min=volume_5min,
        min_bet_usdc=cfg.bet_size,
        max_bet_usdc=cfg.max_bet_size_usdc,
        min_edge=cfg.get_min_edge(is_outsider),
        max_edge=cfg.max_bet_edge,
        liquidity_fraction=cfg.liquidity_fraction,
    )
    if bet < cfg.bet_size:
        bet = cfg.bet_size
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
    p_win_effective: Optional[float] = None
    p_win_raw: Optional[float] = None
    probability_adjustment: Optional[str] = None
    decision_details: Optional[dict] = None
    direction_value: Optional[str] = None



def decide_favorite(signal: MarketSignal, config: dict, time_left_sec: float = 0.0) -> TradeDecision:
    """
    PURE_FAVORITE стратегия.
    Покупает фаворита (YES если mid_price > threshold, NO если < 1-threshold).
    
    Важно: обе стороны проверяются независимо.
    YES-side out-of-bounds НЕ блокирует проверку NO-side.
    Если обе стороны подходят — выбирается с бо́льшим edge.
    """
    yes_bid = getattr(signal, "yes_bid", None)
    yes_ask = getattr(signal, "yes_ask", None)
    cfg = parse_trading_settings(config)
    
    is_valid_time, time_reason = cfg.is_time_valid(time_left_sec, is_outsider=False)
    if not is_valid_time and time_left_sec > 0:
        return TradeDecision("SKIP", 0, 0, time_reason, "PURE_FAVORITE")
        
    if yes_bid is not None and yes_bid > 0 and yes_ask is not None and signal.mid_price > 0:
        spread_pct = (yes_ask - yes_bid) / signal.mid_price
        if spread_pct > cfg.max_spread_pct:
            return TradeDecision("SKIP", 0.0, 0.0, f"spread too wide: {spread_pct:.2%}", "SKIP", edge=0.0)
    threshold = cfg.favorite_threshold
    # PURE_FAVORITE использует свой порог edge (может быть отрицательным)
    min_edge = cfg.favorite_min_edge
    
    if is_in_dead_zone(signal.mid_price, cfg.dead_zone):
        return TradeDecision("SKIP", 0, 0, "dead zone", "SKIP", edge=0.0)


    candidates: list[TradeDecision] = []

    # --- YES side ---
    if signal.mid_price >= threshold:
        eff_yes_ask = signal.get_yes_ask()
        is_valid_price, price_reason = cfg.is_price_valid(eff_yes_ask, is_outsider=False)
        if is_valid_price:
            if signal.yes_bid is not None and float(signal.yes_bid) > 0:
                p_win_yes = float(signal.yes_bid)
            else:
                p_win_yes = signal.mid_price
            edge = compute_edge(p_win_yes, eff_yes_ask)
            if edge >= min_edge:
                bet = _resolve_final_bet(edge, signal.volume_5min, config, is_outsider=False)
                candidates.append(TradeDecision(
                    "BUY_YES", eff_yes_ask, bet,
                    f"favorite YES edge={edge:.4f}", "PURE_FAVORITE",
                    edge=edge, p_up=p_win_yes,
                    p_win_effective=p_win_yes, p_win_raw=p_win_yes
                ))

    # --- NO side --- проверяется НЕЗАВИСИМО от YES-side
    if signal.mid_price <= (1.0 - threshold):
        eff_no_ask = signal.get_no_ask()
        is_valid_price, price_reason = cfg.is_price_valid(eff_no_ask, is_outsider=False)
        if is_valid_price:
            if signal.no_bid is not None and float(signal.no_bid) > 0:
                no_prob = float(signal.no_bid)
            else:
                no_prob = 1.0 - signal.mid_price
            edge = compute_edge(no_prob, eff_no_ask)
            if edge >= min_edge:
                bet = _resolve_final_bet(edge, signal.volume_5min, config, is_outsider=False)
                candidates.append(TradeDecision(
                    "BUY_NO", eff_no_ask, bet,
                    f"favorite NO edge={edge:.4f}", "PURE_FAVORITE",
                    edge=edge, p_up=1.0 - no_prob,
                    p_win_effective=no_prob, p_win_raw=no_prob
                ))

    if not candidates:
        eff_yes = signal.get_yes_ask()
        eff_no = signal.get_no_ask()
        skipped_edge = 0.0
        if signal.mid_price >= threshold:
            is_valid_price, price_reason = cfg.is_price_valid(eff_yes, is_outsider=False)
            if not is_valid_price:
                reason = price_reason
            else:
                p_win_yes = float(signal.yes_bid) if signal.yes_bid is not None and float(signal.yes_bid) > 0 else signal.mid_price
                skipped_edge = compute_edge(p_win_yes, eff_yes)
                reason = f"favorite YES edge={skipped_edge:.4f} < min_edge={min_edge:.4f}"
        elif signal.mid_price <= (1.0 - threshold):
            is_valid_price, price_reason = cfg.is_price_valid(eff_no, is_outsider=False)
            if not is_valid_price:
                reason = price_reason
            else:
                no_prob = float(signal.no_bid) if signal.no_bid is not None and float(signal.no_bid) > 0 else (1.0 - signal.mid_price)
                skipped_edge = compute_edge(no_prob, eff_no)
                reason = f"favorite NO edge={skipped_edge:.4f} < min_edge={min_edge:.4f}"
        else:
            reason = "no clear favorite"
        return TradeDecision("SKIP", 0.0, 0.0, reason, "SKIP", edge=skipped_edge)

    # Выбираем кандидата с наибольшим edge
    best_candidate = max(candidates, key=lambda c: c.edge if c.edge is not None else -999.0)
    return best_candidate


def decide_outsider(
    signal: MarketSignal,
    p_flip: float,
    config: dict,
    ece: float = 0.0,
    time_left_sec: float = 0.0,
) -> TradeDecision:
    """
    Outsider стратегия (TRADE_ON_FLIP).
    Если P(flip) >= flip_threshold → рынок флипнет → покупаем аутсайдера.
    """
    cfg = parse_trading_settings(config)
    
    is_valid_time, time_reason = cfg.is_time_valid(time_left_sec, is_outsider=True)
    if not is_valid_time and time_left_sec > 0:
        return TradeDecision("SKIP", 0, 0, time_reason, "OUTSIDER", p_flip=p_flip)
        
    flip_thresh = cfg.flip_threshold
    if flip_thresh > 1.0:
        flip_thresh = flip_thresh / 100.0
    p_flip_calibrated = apply_ece_correction(p_flip, ece)
    p_flip_effective = min(p_flip, p_flip_calibrated)

    # 1. Сначала проверяем dead zone
    if is_in_dead_zone(signal.mid_price, cfg.dead_zone):
        return TradeDecision("SKIP", 0, 0, "dead zone", "SKIP", p_flip=p_flip, edge=0.0)

    is_yes_fav = signal.mid_price >= FLIP_MIDPOINT
    outsider_ask = signal.get_no_ask() if is_yes_fav else signal.get_yes_ask()
    outsider_action: ActionType = "BUY_NO" if is_yes_fav else "BUY_YES"

    if outsider_ask <= 0:
        return TradeDecision("SKIP", 0, 0, "outsider_ask=0", "SKIP", p_flip=p_flip, edge=0.0)

    outsider_pwin_discount = cfg.outsider_pwin_discount
    p_win_outsider = p_flip_effective * outsider_pwin_discount
    outsider_edge = compute_edge(p_win_outsider, outsider_ask)

    logger.debug(
        "outsider_p_win_calc",
        p_flip_effective=round(p_flip_effective, 4),
        discount=outsider_pwin_discount,
        p_win_adjusted=round(p_win_outsider, 4),
        outsider_ask=outsider_ask,
        edge=round(outsider_edge, 4),
    )

    # 2. Потом проверяем порог p_flip
    if p_flip_effective < flip_thresh:
        return TradeDecision("SKIP", 0, 0,
            f"p_flip_effective={p_flip_effective:.3f} < threshold={flip_thresh:.3f}", "SKIP",
            p_flip=p_flip, edge=outsider_edge)

    edge = outsider_edge
    min_edge = cfg.get_min_edge(is_outsider=True)

    is_valid_price, price_reason = cfg.is_price_valid(outsider_ask, is_outsider=True)
    if not is_valid_price:
        return TradeDecision("SKIP", 0, 0, price_reason, "SKIP", p_flip=p_flip, edge=edge)

    if ece and ece > ECE_WARN_THRESHOLD:
        logger.warning("poor_calibration_model", asset=signal.asset, ece=ece, note="p_flip estimates may be unreliable")

    if edge < min_edge:
        return TradeDecision("SKIP", 0, 0,
            f"edge={edge:.3f} < min={min_edge:.3f}", "SKIP", p_flip=p_flip, edge=edge)

    bet = _resolve_final_bet(edge, signal.volume_5min, config, is_outsider=True)
    if bet <= 0 and not cfg.bypass_bet_size_check:
        return TradeDecision("SKIP", 0, 0, "Bet size 0", "SKIP", p_flip=p_flip, edge=edge)

    decision_details = {
        "market_role": "OUTSIDER",
        "signal_type": "FLIP",
        "p_flip_raw": round(p_flip, 4),
        "p_flip_effective": round(p_flip_effective, 4),
        "ece_used": round(ece, 4),
        "threshold_upper_applied": round(flip_thresh, 4),
        "bet_size_before_multiplier": round(bet, 4),
        "outsider_discount": round(outsider_pwin_discount, 4),
    }

    return TradeDecision(
        outsider_action, outsider_ask, bet,
        f"OUTSIDER p_flip_effective={p_flip_effective:.3f} >= {flip_thresh:.3f}",
        "OUTSIDER",
        p_flip=p_flip, edge=edge,
        p_win_effective=p_win_outsider, p_win_raw=p_flip * outsider_pwin_discount,
        decision_details=decision_details
    )

