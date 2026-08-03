"""
polyflip/trading/combined_voting.py

COMBINED-режим принятия решений:
1. Direction Model (LightGBM):
   Определяет направление базового актива (UP / DOWN).
   Без валидного направления (UP/DOWN) сделка в COMBINED не создаётся (SKIP).

2. Candidate Side:
   UP   -> BUY_YES
   DOWN -> BUY_NO

3. Entry Model (LogReg):
   Фазовая модель (contested / leaning / decided) оценивает p_flip.
   Рассчитывается p_candidate_win, gross_edge, cost_buffer и net_edge.
   Вход разрешается только при net_edge >= min_net_edge.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Any
import structlog

from polyflip.crypto.predictor import CryptoSignal
from polyflip.trading.position_sizing import compute_edge, is_in_dead_zone

logger = structlog.get_logger(__name__)

ActionType = Literal["BUY_YES", "BUY_NO", "SKIP"]


@dataclass(frozen=True)
class CryptoSignalProxy:
    direction: Optional[Literal["UP", "DOWN", "NONE"]]
    features_ok: bool
    model_version: Optional[int] = None
    risk_vetoed: bool = False


@dataclass(frozen=True)
class CombinedEntryResult:
    action: ActionType
    reason: str
    direction_status: str
    direction_model_key: Optional[str] = None
    direction_model_version: Optional[int] = None
    direction_regime: Optional[str] = None
    direction_probability: Optional[float] = None
    direction_p_up: Optional[float] = None
    direction_p_down: Optional[float] = None
    direction_value: Optional[str] = None
    # P0: детальная причина сбоя Direction Model (INFERENCE_FAILED, REGIME_UNAVAILABLE)
    direction_error_detail: Optional[str] = None
    entry_requested_key: Optional[str] = None
    entry_model_key: Optional[str] = None
    entry_model_version: Optional[int] = None
    entry_model_phase: Optional[str] = None
    entry_model_source: Optional[str] = None  # PHASE | BASE | GLOBAL | NONE
    entry_status: str = "READY"
    fallback_reason: Optional[str] = None
    p_candidate_win: Optional[float] = None
    candidate_side: Optional[str] = None
    candidate_ask: Optional[float] = None
    gross_edge: Optional[float] = None
    cost_buffer: float = 0.02
    net_edge: Optional[float] = None
    max_acceptable_price: Optional[float] = None
    bet_size_usdc: float = 0.0
    strike_source: Optional[str] = None
    strike_proxy: Optional[float] = None
    underlying_price: Optional[float] = None
    distance_to_strike_pct: Optional[float] = None
    p_flip: Optional[float] = None


def evaluate_combined_entry(
    crypto_sig: CryptoSignal,
    market_phase: str,
    entry_requested_key: Optional[str],
    entry_model_key: Optional[str],
    entry_model_version: Optional[int],
    entry_model_source: str,
    p_flip: Optional[float],
    fresh_yes_price: float,
    yes_ask: Optional[float],
    no_ask: Optional[float],
    cost_buffer: float = 0.02,
    min_net_edge: float = 0.03,
    min_price: float = 0.05,
    max_price: float = 0.95,
    volume_5min: float = 0.0,
    config_dict: Optional[dict] = None,
    underlying_price: Optional[float] = None,
    fallback_reason: Optional[str] = None,
    min_direction_prob: float = 0.55,
    min_win_prob: float = 0.55,
) -> CombinedEntryResult:
    """
    Чистая функция оценки входа в Combined-режиме.
    """
    config_dict = config_dict or {}

    dir_prob = crypto_sig.p_up if crypto_sig.direction == "UP" else (crypto_sig.p_down if crypto_sig.direction == "DOWN" else 0.5)
    dir_val = crypto_sig.direction
    dir_status = crypto_sig.status or ("READY" if crypto_sig.features_ok else "INVALID_FEATURES")

    strike = crypto_sig.strike if crypto_sig.strike > 0 else None
    und_price = underlying_price if underlying_price is not None else strike
    dist_pct = None
    if und_price and strike and strike > 0:
        dist_pct = round((und_price - strike) / strike * 100.0, 4)

    # 1. Валидация LightGBM Direction
    if not crypto_sig.features_ok or crypto_sig.model_version is None or crypto_sig.model_version < 0:
        # P0: формируем информативный reason
        symbol = crypto_sig.symbol or ""
        regime = crypto_sig.regime or ""
        error_detail = getattr(crypto_sig, "risk_reason", "") or ""

        if dir_status == "REGIME_UNAVAILABLE" and symbol and regime:
            required_key = f"{symbol}_{regime}"
            reason_str = f"Direction Model unavailable: required={required_key} status=REGIME_UNAVAILABLE"
        elif dir_status == "MODEL_NOT_LOADED":
            reason_str = f"Direction Model unavailable: no active models loaded for {symbol} status=MODEL_NOT_LOADED"
        elif dir_status == "INFERENCE_FAILED" and error_detail:
            reason_str = f"Direction Model error ({symbol}): {error_detail}"
        else:
            reason_str = f"Direction Model unavailable (status={dir_status})"

        return CombinedEntryResult(
            action="SKIP",
            reason=reason_str,
            direction_status=dir_status,
            direction_error_detail=error_detail or None,
            direction_model_key=crypto_sig.model_key or None,
            direction_model_version=crypto_sig.model_version if crypto_sig.model_version >= 0 else None,
            direction_regime=crypto_sig.regime or None,
            direction_probability=dir_prob,
            direction_p_up=getattr(crypto_sig, 'p_up', None),
            direction_p_down=getattr(crypto_sig, 'p_down', None),
            direction_value=dir_val,
            entry_requested_key=entry_requested_key,
            entry_model_key=entry_model_key,
            entry_model_version=entry_model_version,
            entry_model_phase=market_phase,
            entry_model_source=entry_model_source,
            entry_status="DIRECTION_UNAVAILABLE",
            fallback_reason=fallback_reason,
            cost_buffer=cost_buffer,
            strike_source="BINANCE_LAST_CANDLE" if strike else None,
            strike_proxy=strike,
            underlying_price=und_price,
            distance_to_strike_pct=dist_pct,
            p_flip=p_flip,
        )

    if crypto_sig.risk_vetoed:
        return CombinedEntryResult(
            action="SKIP",
            reason=f"Direction Model funding veto: {crypto_sig.risk_reason}",
            direction_status="FUNDING_VETOED",
            direction_model_key=crypto_sig.model_key or None,
            direction_model_version=crypto_sig.model_version,
            direction_regime=crypto_sig.regime or None,
            direction_probability=dir_prob,
            direction_p_up=getattr(crypto_sig, 'p_up', None),
            direction_p_down=getattr(crypto_sig, 'p_down', None),
            direction_value=dir_val,
            entry_requested_key=entry_requested_key,
            entry_model_key=entry_model_key,
            entry_model_version=entry_model_version,
            entry_model_phase=market_phase,
            entry_model_source=entry_model_source,
            entry_status="DIRECTION_VETOED",
            fallback_reason=fallback_reason,
            cost_buffer=cost_buffer,
            strike_source="BINANCE_LAST_CANDLE" if strike else None,
            strike_proxy=strike,
            underlying_price=und_price,
            distance_to_strike_pct=dist_pct,
            p_flip=p_flip,
        )

    if crypto_sig.direction not in ("UP", "DOWN"):
        return CombinedEntryResult(
            action="SKIP",
            reason=f"Direction Model gave no clear trend ({crypto_sig.direction})",
            direction_status="DIRECTION_NONE",
            direction_model_key=crypto_sig.model_key or None,
            direction_model_version=crypto_sig.model_version,
            direction_regime=crypto_sig.regime or None,
            direction_probability=dir_prob,
            direction_p_up=getattr(crypto_sig, 'p_up', None),
            direction_p_down=getattr(crypto_sig, 'p_down', None),
            direction_value=dir_val,
            entry_requested_key=entry_requested_key,
            entry_model_key=entry_model_key,
            entry_model_version=entry_model_version,
            entry_model_phase=market_phase,
            entry_model_source=entry_model_source,
            entry_status="DIRECTION_NONE",
            fallback_reason=fallback_reason,
            cost_buffer=cost_buffer,
            strike_source="BINANCE_LAST_CANDLE" if strike else None,
            strike_proxy=strike,
            underlying_price=und_price,
            distance_to_strike_pct=dist_pct,
            p_flip=p_flip,
        )

    if dir_prob < min_direction_prob:
        return CombinedEntryResult(
            action="SKIP",
            reason=f"Direction prob {dir_prob:.4f} < min {min_direction_prob}",
            direction_status="LOW_DIRECTION_PROB",
            direction_model_key=crypto_sig.model_key or None,
            direction_model_version=crypto_sig.model_version,
            direction_regime=crypto_sig.regime or None,
            direction_probability=dir_prob,
            direction_p_up=getattr(crypto_sig, 'p_up', None),
            direction_p_down=getattr(crypto_sig, 'p_down', None),
            direction_value=dir_val,
            entry_requested_key=entry_requested_key,
            entry_model_key=entry_model_key,
            entry_model_version=entry_model_version,
            entry_model_phase=market_phase,
            entry_model_source=entry_model_source,
            entry_status="DIRECTION_UNAVAILABLE",
            fallback_reason=fallback_reason,
            cost_buffer=cost_buffer,
            strike_source="BINANCE_LAST_CANDLE" if strike else None,
            strike_proxy=strike,
            underlying_price=und_price,
            distance_to_strike_pct=dist_pct,
            p_flip=p_flip,
        )

    # 2. Определение стороны кандидата и цены предложения
    if crypto_sig.direction == "UP":
        candidate_side: ActionType = "BUY_YES"
        candidate_ask = yes_ask if (yes_ask is not None and yes_ask > 0) else fresh_yes_price
    else:
        candidate_side = "BUY_NO"
        candidate_ask = no_ask if (no_ask is not None and no_ask > 0) else round(1.0 - fresh_yes_price, 4)

    # 3. Валидация LogReg Entry Model
    if p_flip is None or entry_model_key is None:
        return CombinedEntryResult(
            action="SKIP",
            reason="Entry Model (LogReg) evaluation failed or unavailable",
            direction_status=dir_status,
            direction_model_key=crypto_sig.model_key or None,
            direction_model_version=crypto_sig.model_version,
            direction_regime=crypto_sig.regime or None,
            direction_probability=dir_prob,
            direction_p_up=getattr(crypto_sig, 'p_up', None),
            direction_p_down=getattr(crypto_sig, 'p_down', None),
            direction_value=dir_val,
            entry_requested_key=entry_requested_key,
            entry_model_key=entry_model_key,
            entry_model_version=entry_model_version,
            entry_model_phase=market_phase,
            entry_model_source=entry_model_source,
            entry_status="MODEL_NOT_FOUND",
            fallback_reason=fallback_reason,
            candidate_side=candidate_side,
            candidate_ask=candidate_ask,
            cost_buffer=cost_buffer,
            strike_source="BINANCE_LAST_CANDLE" if strike else None,
            strike_proxy=strike,
            underlying_price=und_price,
            distance_to_strike_pct=dist_pct,
            p_flip=p_flip,
        )

    # 4. Расчет вероятности победы кандидата (p_candidate_win)
    # p_flip = вероятность смены лидера (флипа от фаворита к аутсайдеру)
    if crypto_sig.direction == "UP":
        # Кандидат YES. Если YES - фаворит (price >= 0.50), win prob = 1 - p_flip.
        # Если YES - аутсайдер (price < 0.50), win prob = p_flip.
        p_candidate_win = (1.0 - p_flip) if fresh_yes_price >= 0.50 else p_flip
    else:
        # Кандидат NO. Если YES - фаворит (price >= 0.50), NO - аутсайдер, win prob = p_flip.
        # Если YES - аутсайдер (price < 0.50), NO - фаворит, win prob = 1 - p_flip.
        p_candidate_win = p_flip if fresh_yes_price >= 0.50 else (1.0 - p_flip)

    p_candidate_win = round(max(0.0, min(1.0, p_candidate_win)), 4)

    if p_candidate_win < min_win_prob:
        return CombinedEntryResult(
            action="SKIP",
            reason=f"Candidate win prob {p_candidate_win:.4f} < min {min_win_prob}",
            direction_status=dir_status,
            direction_model_key=crypto_sig.model_key or None,
            direction_model_version=crypto_sig.model_version,
            direction_regime=crypto_sig.regime or None,
            direction_probability=dir_prob,
            direction_p_up=getattr(crypto_sig, 'p_up', None),
            direction_p_down=getattr(crypto_sig, 'p_down', None),
            direction_value=dir_val,
            entry_requested_key=entry_requested_key,
            entry_model_key=entry_model_key,
            entry_model_version=entry_model_version,
            entry_model_phase=market_phase,
            entry_model_source=entry_model_source,
            entry_status="LOW_WIN_PROB",
            fallback_reason=fallback_reason,
            p_candidate_win=p_candidate_win,
            candidate_side=candidate_side,
            candidate_ask=candidate_ask,
            cost_buffer=cost_buffer,
            strike_source="BINANCE_LAST_CANDLE" if strike else None,
            strike_proxy=strike,
            underlying_price=und_price,
            distance_to_strike_pct=dist_pct,
            p_flip=p_flip,
        )

    # 5. Проверка диапазона цен покупки
    if not (min_price <= candidate_ask <= max_price):
        return CombinedEntryResult(
            action="SKIP",
            reason=f"Candidate ask {candidate_ask:.3f} out of allowed bounds [{min_price}, {max_price}]",
            direction_status=dir_status,
            direction_model_key=crypto_sig.model_key or None,
            direction_model_version=crypto_sig.model_version,
            direction_regime=crypto_sig.regime or None,
            direction_probability=dir_prob,
            direction_p_up=getattr(crypto_sig, 'p_up', None),
            direction_p_down=getattr(crypto_sig, 'p_down', None),
            direction_value=dir_val,
            entry_requested_key=entry_requested_key,
            entry_model_key=entry_model_key,
            entry_model_version=entry_model_version,
            entry_model_phase=market_phase,
            entry_model_source=entry_model_source,
            entry_status="PRICE_OUT_OF_BOUNDS",
            fallback_reason=fallback_reason,
            p_candidate_win=p_candidate_win,
            candidate_side=candidate_side,
            candidate_ask=candidate_ask,
            cost_buffer=cost_buffer,
            strike_source="BINANCE_LAST_CANDLE" if strike else None,
            strike_proxy=strike,
            underlying_price=und_price,
            distance_to_strike_pct=dist_pct,
            p_flip=p_flip,
        )

    # 6. Расчет Gross Edge и Net Edge
    gross_edge = compute_edge(p_candidate_win, candidate_ask)
    net_edge = round(gross_edge - cost_buffer, 4)

    if net_edge < min_net_edge:
        return CombinedEntryResult(
            action="SKIP",
            reason=f"Insufficient net edge: {net_edge:.4f} < min {min_net_edge:.4f} (gross={gross_edge:.4f}, buffer={cost_buffer:.4f})",
            direction_status=dir_status,
            direction_model_key=crypto_sig.model_key or None,
            direction_model_version=crypto_sig.model_version,
            direction_regime=crypto_sig.regime or None,
            direction_probability=dir_prob,
            direction_p_up=getattr(crypto_sig, 'p_up', None),
            direction_p_down=getattr(crypto_sig, 'p_down', None),
            direction_value=dir_val,
            entry_requested_key=entry_requested_key,
            entry_model_key=entry_model_key,
            entry_model_version=entry_model_version,
            entry_model_phase=market_phase,
            entry_model_source=entry_model_source,
            entry_status="INSUFFICIENT_NET_EDGE",
            fallback_reason=fallback_reason,
            p_candidate_win=p_candidate_win,
            candidate_side=candidate_side,
            candidate_ask=candidate_ask,
            gross_edge=gross_edge,
            cost_buffer=cost_buffer,
            net_edge=net_edge,
            strike_source="BINANCE_LAST_CANDLE" if strike else None,
            strike_proxy=strike,
            underlying_price=und_price,
            distance_to_strike_pct=dist_pct,
            p_flip=p_flip,
        )

    # 7. Расчет размера ставки
    from polyflip.trading.decision_logic import _resolve_final_bet
    bet_size = _resolve_final_bet(net_edge, volume_5min, config_dict)

    bypass_bet = str(config_dict.get("BYPASS_BET_SIZE_CHECK", "false")).lower() == "true"
    if bet_size <= 0 and not bypass_bet:
        return CombinedEntryResult(
            action="SKIP",
            reason="Calculated bet size is 0.0 USDC",
            direction_status=dir_status,
            direction_model_key=crypto_sig.model_key or None,
            direction_model_version=crypto_sig.model_version,
            direction_regime=crypto_sig.regime or None,
            direction_probability=dir_prob,
            direction_p_up=getattr(crypto_sig, 'p_up', None),
            direction_p_down=getattr(crypto_sig, 'p_down', None),
            direction_value=dir_val,
            entry_requested_key=entry_requested_key,
            entry_model_key=entry_model_key,
            entry_model_version=entry_model_version,
            entry_model_phase=market_phase,
            entry_model_source=entry_model_source,
            entry_status="ZERO_BET",
            fallback_reason=fallback_reason,
            p_candidate_win=p_candidate_win,
            candidate_side=candidate_side,
            candidate_ask=candidate_ask,
            gross_edge=gross_edge,
            cost_buffer=cost_buffer,
            net_edge=net_edge,
            bet_size_usdc=0.0,
            strike_source="BINANCE_LAST_CANDLE" if strike else None,
            strike_proxy=strike,
            underlying_price=und_price,
            distance_to_strike_pct=dist_pct,
            p_flip=p_flip,
        )

    # 8. Расчет max_acceptable_price (защита от дрейфа и спреда)
    max_drift = float(config_dict.get("MAX_PRICE_DRIFT", 0.03))
    # Максимально допустимая цена исполнения не должна снижать net_edge ниже min_net_edge
    max_price_by_edge = round(p_candidate_win - cost_buffer - min_net_edge, 3)
    max_price_by_drift = round(candidate_ask + max_drift, 3)
    max_acceptable_price = min(max_price_by_edge, max_price_by_drift, max_price)

    return CombinedEntryResult(
        action=candidate_side,
        reason=f"COMBINED {candidate_side}: net_edge={net_edge:.4f} (gross={gross_edge:.4f}, dir={crypto_sig.direction}, p_win={p_candidate_win:.3f})",
        direction_status=dir_status,
        direction_model_key=crypto_sig.model_key or None,
        direction_model_version=crypto_sig.model_version,
        direction_regime=crypto_sig.regime or None,
        direction_probability=dir_prob,
            direction_p_up=getattr(crypto_sig, 'p_up', None),
            direction_p_down=getattr(crypto_sig, 'p_down', None),
        direction_value=dir_val,
        entry_requested_key=entry_requested_key,
        entry_model_key=entry_model_key,
        entry_model_version=entry_model_version,
        entry_model_phase=market_phase,
        entry_model_source=entry_model_source,
        entry_status="READY",
        fallback_reason=fallback_reason,
        p_candidate_win=p_candidate_win,
        candidate_side=candidate_side,
        candidate_ask=candidate_ask,
        gross_edge=gross_edge,
        cost_buffer=cost_buffer,
        net_edge=net_edge,
        max_acceptable_price=max_acceptable_price,
        bet_size_usdc=bet_size,
        strike_source="BINANCE_LAST_CANDLE" if strike else None,
        strike_proxy=strike,
        underlying_price=und_price,
        distance_to_strike_pct=dist_pct,
        p_flip=p_flip,
    )


# ── Legacy compatibility ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class VotingResult:
    action: Literal["BUY_YES", "BUY_NO", "SKIP"]
    reason: str
    confidence: float
    ml_action: str
    lgbm_direction: Optional[str]
    lgbm_features_ok: bool
    bet_size_multiplier: float = 1.0


def combine_votes(
    ml_action: str,
    ml_edge: float,
    crypto_sig: Any,
    asset: str,
    none_bet_multiplier: float = 0.5,
    ml_skip_reason: str = "",
) -> VotingResult:
    """Legacy helper maintained for backward compatibility with older tests."""
    if not getattr(crypto_sig, "features_ok", False):
        return VotingResult(
            action=ml_action,  # type: ignore
            reason="LightGBM features invalid, fallback to ML-only",
            confidence=ml_edge,
            ml_action=ml_action,
            lgbm_direction=getattr(crypto_sig, "direction", None),
            lgbm_features_ok=False,
            bet_size_multiplier=1.0,
        )

    if getattr(crypto_sig, "risk_vetoed", False):
        return VotingResult(
            action="SKIP",
            reason="Hard Veto: LightGBM risk veto",
            confidence=1.0,
            ml_action=ml_action,
            lgbm_direction=getattr(crypto_sig, "direction", None),
            lgbm_features_ok=True,
            bet_size_multiplier=0.0,
        )

    if ml_action == "SKIP":
        return VotingResult(
            action="SKIP",
            reason=f"ML hard-SKIP: {ml_skip_reason or 'ML voted SKIP'}",
            confidence=0.0,
            ml_action=ml_action,
            lgbm_direction=getattr(crypto_sig, "direction", None),
            lgbm_features_ok=True,
            bet_size_multiplier=0.0,
        )

    ml_direction = "UP" if ml_action == "BUY_YES" else "DOWN"
    sig_dir = getattr(crypto_sig, "direction", "NONE")

    if sig_dir == "NONE":
        if none_bet_multiplier <= 0.0:
            return VotingResult(
                action="SKIP",
                reason="LightGBM flat (NONE) with zero multiplier: veto",
                confidence=0.0,
                ml_action=ml_action,
                lgbm_direction="NONE",
                lgbm_features_ok=True,
                bet_size_multiplier=0.0,
            )
        return VotingResult(
            action=ml_action,  # type: ignore
            reason=f"LightGBM flat (NONE): ML={ml_action}, reduced bet size",
            confidence=ml_edge * 0.7,
            ml_action=ml_action,
            lgbm_direction="NONE",
            lgbm_features_ok=True,
            bet_size_multiplier=none_bet_multiplier,
        )

    if sig_dir == ml_direction:
        return VotingResult(
            action=ml_action,  # type: ignore
            reason=f"Both models agree: ML={ml_action}, LightGBM={sig_dir}",
            confidence=min(1.0, ml_edge * 1.2),
            ml_action=ml_action,
            lgbm_direction=sig_dir,
            lgbm_features_ok=True,
            bet_size_multiplier=1.0,
        )

    return VotingResult(
        action="SKIP",
        reason=f"LightGBM veto: ML={ml_action} but LightGBM={sig_dir}",
        confidence=0.0,
        ml_action=ml_action,
        lgbm_direction=sig_dir,
        lgbm_features_ok=True,
        bet_size_multiplier=0.0,
    )
