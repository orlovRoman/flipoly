"""
polyflip/trading/combined_voting.py

COMBINED-режим принятия решений:
1. Direction Model (LightGBM):
   Определяет направление базового актива (UP / DOWN).
   Без валидного направления (UP/DOWN) сделка в COMBINED не создаётся (SKIP).

2. Candidate Side:
   LGBM=UP    -> BUY_YES
   LGBM=DOWN  -> BUY_NO
   LGBM=NONE  -> fallback to LogReg (если combined_fallback_to_logreg_on_none=True)
   При разногласии моделей: SKIP (если combined_require_consensus=True)

3. Entry Model (LogReg):
   Фазовая модель (contested / leaning / decided) оценивает p_flip.
   Рассчитывается p_candidate_win, gross_edge, cost_buffer и net_edge.
   Вход разрешается только при net_edge >= min_net_edge.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Optional, Any, TYPE_CHECKING, cast
import structlog

if TYPE_CHECKING:
    from polyflip.trading.trading_config import TradingConfig

from polyflip.crypto.predictor import CryptoSignal
from polyflip.trading.position_sizing import compute_edge

logger = structlog.get_logger(__name__)

ActionType = Literal["BUY_YES", "BUY_NO", "SKIP"]

_LOGREG_ABSTAIN_BAND: float = 0.05


@dataclass(frozen=True)
class CryptoSignalProxy:
    direction: Optional[Literal["UP", "DOWN", "NONE"]]
    features_ok: bool
    model_version: Optional[int] = None
    risk_vetoed: bool = False



@dataclass(frozen=True)
class DirectionConsensus:
    final_side: Literal["BUY_YES", "BUY_NO", "SKIP"]
    consensus_type: str
    reason: str

def logreg_direction_vote(
    p_flip: Optional[float],
    fresh_yes_price: float,
    flip_threshold: float = 0.50,
    abstain_band: float = _LOGREG_ABSTAIN_BAND,
) -> Literal["BUY_YES", "BUY_NO", "ABSTAIN"]:
    if p_flip is None:
        return "ABSTAIN"
    # p_flip близкий к flip_threshold — нет уверенного сигнала, воздерживаемся (|p_flip - flip_threshold| < abstain_band)
    if abs(p_flip - flip_threshold) < abstain_band:
        return "ABSTAIN"
    is_yes_fav = fresh_yes_price >= 0.50
    if p_flip > flip_threshold:
        return "BUY_NO" if is_yes_fav else "BUY_YES"
    else:
        return "BUY_YES" if is_yes_fav else "BUY_NO"

def resolve_direction_consensus(
    lgbm_vote: str,
    lr_vote: str,
    require_consensus: bool,
    fallback_to_logreg_on_none: bool,
) -> DirectionConsensus:
    lgbm_side = "BUY_YES" if lgbm_vote == "UP" else ("BUY_NO" if lgbm_vote == "DOWN" else "ABSTAIN")
    
    if lgbm_side == "ABSTAIN" and lr_vote == "ABSTAIN":
        return DirectionConsensus("SKIP", "BOTH_ABSTAIN", "No directional signal from any model")
        
    if lgbm_side == "ABSTAIN":
        if fallback_to_logreg_on_none:
            return DirectionConsensus(cast(Literal["BUY_YES", "BUY_NO", "SKIP"], lr_vote), "PARTIAL_LR", "LightGBM is NONE, fallback to LogReg")
        else:
            return DirectionConsensus("SKIP", "PARTIAL_LR", "LightGBM is NONE, fallback disabled")
            
    if lr_vote == "ABSTAIN":
        return DirectionConsensus(cast(Literal["BUY_YES", "BUY_NO", "SKIP"], lgbm_side), "PARTIAL_LGBM", "LogReg is missing, using LightGBM")
        
    if lgbm_side == lr_vote:
        return DirectionConsensus(cast(Literal["BUY_YES", "BUY_NO", "SKIP"], lgbm_side), "AGREE", f"Both models agree on {lgbm_side}")
        
    if require_consensus:
        return DirectionConsensus("SKIP", "CONFLICT", f"Conflict: LGBM={lgbm_side}, LR={lr_vote}")
    else:
        return DirectionConsensus(cast(Literal["BUY_YES", "BUY_NO", "SKIP"], lgbm_side), "CONFLICT", f"Conflict resolved to LightGBM: {lgbm_side}")

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
    direction_threshold_up: Optional[float] = None
    direction_threshold_down: Optional[float] = None
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
    p_logreg_win: Optional[float] = None
    direction_discount_applied: float = 1.0
    combined_dir_discount_weight: float = 0.0
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
    lr_direction_vote: Optional[str] = None
    lgbm_direction_vote: Optional[str] = None
    consensus_type: Optional[str] = None
    lgbm_inverted: bool = False
    lgbm_p_up_raw: float = 0.0
    lgbm_p_down_raw: float = 0.0


def apply_direction_confidence_discount(
    p_logreg_win: float,
    dir_prob: float,
    min_direction_prob: float,
    strong_threshold: float,
    discount_weight: float,
) -> float:
    """
    Дисконтирует вероятность победы LogReg за неуверенность LightGBM.

    Формула:
      weakness = (strong_threshold - dir_prob) / (strong_threshold - min_direction_prob)
      weakness = max(0.0, min(1.0, weakness))
      multiplier = 1.0 - (discount_weight * weakness)
      p_candidate_win = p_logreg_win * multiplier

    Свойства:
      - При dir_prob >= strong_threshold: weakness = 0.0 -> multiplier = 1.0 (дисконта нет)
      - При dir_prob <= min_direction_prob: weakness = 1.0 -> multiplier = 1.0 - discount_weight (макс. дисконт)
      - При discount_weight <= 0.0: multiplier = 1.0 (дисконт отключен)
      - При strong_threshold <= min_direction_prob: fallback, дисконт не применяется
    """
    if discount_weight <= 0.0:
        return round(max(0.0, min(1.0, p_logreg_win)), 4)

    band = strong_threshold - min_direction_prob
    if band <= 0:
        return round(max(0.0, min(1.0, p_logreg_win)), 4)

    weakness = (strong_threshold - dir_prob) / band
    weakness = max(0.0, min(1.0, weakness))
    multiplier = 1.0 - (discount_weight * weakness)
    discounted = p_logreg_win * multiplier
    return round(max(0.0, min(1.0, discounted)), 4)


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
    cfg: "TradingConfig",
    cost_buffer: float = 0.02,
    volume_5min: float = 0.0,
    underlying_price: Optional[float] = None,
    fallback_reason: Optional[str] = None,
    time_left_sec: float = 0.0,
) -> CombinedEntryResult:
    """Обёртка для переноса флагов LightGBM в результат."""
    result = _evaluate_combined_entry_inner(
        crypto_sig=crypto_sig,
        market_phase=market_phase,
        entry_requested_key=entry_requested_key,
        entry_model_key=entry_model_key,
        entry_model_version=entry_model_version,
        entry_model_source=entry_model_source,
        p_flip=p_flip,
        fresh_yes_price=fresh_yes_price,
        yes_ask=yes_ask,
        no_ask=no_ask,
        cfg=cfg,
        cost_buffer=cost_buffer,
        volume_5min=volume_5min,
        underlying_price=underlying_price,
        fallback_reason=fallback_reason,
        time_left_sec=time_left_sec,
    )
    if crypto_sig:
        result = replace(
            result,
            lgbm_inverted=getattr(crypto_sig, "inverted", False),
            lgbm_p_up_raw=getattr(crypto_sig, "p_up_raw", 0.0),
            lgbm_p_down_raw=getattr(crypto_sig, "p_down_raw", 0.0)
        )
    return result


def _evaluate_combined_entry_inner(
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
    cfg: "TradingConfig",
    cost_buffer: float = 0.02,
    volume_5min: float = 0.0,
    underlying_price: Optional[float] = None,
    fallback_reason: Optional[str] = None,
    time_left_sec: float = 0.0,
) -> CombinedEntryResult:
    """Внутренняя логика оценки."""

    if crypto_sig.direction == "UP":
        dir_prob = crypto_sig.p_up or 0.0
    elif crypto_sig.direction == "DOWN":
        dir_prob = crypto_sig.p_down or 0.0
    else:
        # NONE: берём max(p_up, p_down) — диагностически корректно
        dir_prob = max(crypto_sig.p_up or 0.0, crypto_sig.p_down or 0.0)
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
            direction_threshold_up=getattr(crypto_sig, 'threshold_up', None),
            direction_threshold_down=getattr(crypto_sig, 'threshold_down', None),
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
            direction_threshold_up=getattr(crypto_sig, 'threshold_up', None),
            direction_threshold_down=getattr(crypto_sig, 'threshold_down', None),
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
            direction_threshold_up=getattr(crypto_sig, 'threshold_up', None),
            direction_threshold_down=getattr(crypto_sig, 'threshold_down', None),
            direction_value=dir_val,
            entry_requested_key=entry_requested_key,
            entry_model_key=entry_model_key,
            entry_model_version=entry_model_version,
            entry_model_phase=market_phase,
            entry_model_source=entry_model_source,
            entry_status="MODEL_NOT_FOUND",
            fallback_reason=fallback_reason,
            candidate_side=None,
            candidate_ask=None,
            cost_buffer=cost_buffer,
            strike_source="BINANCE_LAST_CANDLE" if strike else None,
            strike_proxy=strike,
            underlying_price=und_price,
            distance_to_strike_pct=dist_pct,
            p_flip=p_flip,
        )

    # 3.5 Валидация минимальной уверенности LightGBM (для UP / DOWN)
    min_direction_prob_cfg = getattr(cfg, "min_direction_prob", 0.505)
    if crypto_sig.direction in ("UP", "DOWN") and dir_prob < min_direction_prob_cfg:
        return CombinedEntryResult(
            action="SKIP",
            reason=f"Direction prob {dir_prob:.4f} < min {min_direction_prob_cfg:.4f} (floor)",
            direction_status="LOW_DIRECTION_PROB",
            direction_model_key=crypto_sig.model_key or None,
            direction_model_version=crypto_sig.model_version,
            direction_regime=crypto_sig.regime or None,
            direction_probability=dir_prob,
            direction_p_up=getattr(crypto_sig, 'p_up', None),
            direction_p_down=getattr(crypto_sig, 'p_down', None),
            direction_threshold_up=getattr(crypto_sig, 'threshold_up', None),
            direction_threshold_down=getattr(crypto_sig, 'threshold_down', None),
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

    # 3.6 Consensus
    lr_abstain_band = getattr(cfg, "combined_logreg_abstain_band", _LOGREG_ABSTAIN_BAND)
    lr_vote = logreg_direction_vote(p_flip, fresh_yes_price, cfg.flip_threshold, abstain_band=lr_abstain_band)
    lgbm_vote = crypto_sig.direction or "NONE"
    
    consensus = resolve_direction_consensus(
        lgbm_vote=lgbm_vote,
        lr_vote=lr_vote,
        require_consensus=getattr(cfg, "combined_require_consensus", True),
        fallback_to_logreg_on_none=getattr(cfg, "combined_fallback_to_logreg_on_none", True),
    )
    
    if consensus.final_side == "SKIP":
        return CombinedEntryResult(
            action="SKIP",
            reason=f"Consensus failed: {consensus.consensus_type} ({consensus.reason})",
            direction_status=dir_status,
            direction_model_key=crypto_sig.model_key or None,
            direction_model_version=crypto_sig.model_version,
            direction_regime=crypto_sig.regime or None,
            direction_probability=dir_prob,
            direction_p_up=getattr(crypto_sig, 'p_up', None),
            direction_p_down=getattr(crypto_sig, 'p_down', None),
            direction_threshold_up=getattr(crypto_sig, 'threshold_up', None),
            direction_threshold_down=getattr(crypto_sig, 'threshold_down', None),
            direction_value=dir_val,
            entry_requested_key=entry_requested_key,
            entry_model_key=entry_model_key,
            entry_model_version=entry_model_version,
            entry_model_phase=market_phase,
            entry_model_source=entry_model_source,
            entry_status="CONSENSUS_FAILED",
            fallback_reason=fallback_reason,
            candidate_side=None,
            candidate_ask=None,
            cost_buffer=cost_buffer,
            strike_source="BINANCE_LAST_CANDLE" if strike else None,
            strike_proxy=strike,
            underlying_price=und_price,
            distance_to_strike_pct=dist_pct,
            p_flip=p_flip,
            lr_direction_vote=lr_vote,
            lgbm_direction_vote=lgbm_vote,
            consensus_type=consensus.consensus_type,
        )

    if consensus.consensus_type == "PARTIAL_LR":
        logger.info(
            "combined_none_fallback_to_logreg",
            asset=crypto_sig.symbol,
            lgbm_direction=lgbm_vote,
            lr_vote=lr_vote,
            p_up=getattr(crypto_sig, 'p_up', None),
            p_down=getattr(crypto_sig, 'p_down', None),
            threshold_up=getattr(crypto_sig, 'threshold_up', None),
            threshold_down=getattr(crypto_sig, 'threshold_down', None),
        )

    if crypto_sig.direction not in ("UP", "DOWN"):
        dir_status_for_result = "DIRECTION_NONE_FALLBACK_LR"
    else:
        dir_status_for_result = dir_status

    candidate_side: Literal["BUY_YES", "BUY_NO", "SKIP"] = consensus.final_side
    if candidate_side == "BUY_YES":
        candidate_ask = yes_ask if (yes_ask is not None and yes_ask > 0) else fresh_yes_price
    else:
        candidate_ask = no_ask if (no_ask is not None and no_ask > 0) else round(1.0 - fresh_yes_price, 4)

    is_outsider = (candidate_side == "BUY_YES" and fresh_yes_price < 0.50) or (candidate_side == "BUY_NO" and fresh_yes_price >= 0.50)

    # ВАЖНО: Если TRADE_ON_FAVORITE выключен, мы отсекаем фаворитов ДО выполнения лишних вычислений discount/probabilities
    if not is_outsider and not cfg.trade_on_favorite:
        return CombinedEntryResult(
            action="SKIP",
            reason=f"TRADE_ON_FAVORITE is disabled, skipping favorite candidate {candidate_side}",
            direction_status=dir_status_for_result,
            direction_model_key=crypto_sig.model_key or None,
            direction_model_version=crypto_sig.model_version,
            direction_regime=crypto_sig.regime or None,
            direction_probability=dir_prob,
            direction_p_up=getattr(crypto_sig, 'p_up', None),
            direction_p_down=getattr(crypto_sig, 'p_down', None),
            direction_threshold_up=getattr(crypto_sig, 'threshold_up', None),
            direction_threshold_down=getattr(crypto_sig, 'threshold_down', None),
            direction_value=dir_val,
            entry_requested_key=entry_requested_key,
            entry_model_key=entry_model_key,
            entry_model_version=entry_model_version,
            entry_model_phase=market_phase,
            entry_model_source=entry_model_source,
            entry_status="FAVORITE_DISABLED",
            fallback_reason=fallback_reason,
            lr_direction_vote=lr_vote,
            lgbm_direction_vote=lgbm_vote,
            consensus_type=consensus.consensus_type,
            candidate_side=candidate_side,
            candidate_ask=candidate_ask,
            strike_source="BINANCE_LAST_CANDLE" if strike else None,
            strike_proxy=strike,
            underlying_price=und_price,
            distance_to_strike_pct=dist_pct,
            p_flip=p_flip,
        )

    # 4. Probabilities & Discountвероятности победы кандидата (p_candidate_win)
    # p_flip = вероятность смены лидера (флипа от фаворита к аутсайдеру)
    if candidate_side == "BUY_YES":
        # Кандидат YES. Если YES - фаворит (price >= 0.50), win prob = 1 - p_flip.
        # Если YES - аутсайдер (price < 0.50), win prob = p_flip.
        p_logreg_win = (1.0 - p_flip) if fresh_yes_price >= 0.50 else p_flip
    else:
        # Кандидат NO. Если YES - фаворит (price >= 0.50), NO - аутсайдер, win prob = p_flip.
        # Если YES - аутсайдер (price < 0.50), NO - фаворит, win prob = 1 - p_flip.
        p_logreg_win = p_flip if fresh_yes_price >= 0.50 else (1.0 - p_flip)

    p_logreg_win = round(max(0.0, min(1.0, p_logreg_win)), 4)

    # Применяем дисконт за неуверенность LightGBM
    discount_weight = getattr(cfg, "combined_dir_discount_weight", 0.0)
    strong_thresh = getattr(cfg, "combined_dir_strong_threshold", 0.65)
    min_dir_prob_val = getattr(cfg, "min_direction_prob", 0.505)

    p_candidate_win = apply_direction_confidence_discount(
        p_logreg_win=p_logreg_win,
        dir_prob=dir_prob,
        min_direction_prob=min_dir_prob_val,
        strong_threshold=strong_thresh,
        discount_weight=discount_weight,
    )
    if p_logreg_win > 0:
        discount_mult = round(p_candidate_win / p_logreg_win, 4)
    else:
        discount_mult = 0.0
        logger.warning("combined_discount_mult_zero_logreg", asset=crypto_sig.symbol, p_flip=p_flip)

    if discount_weight > 0.0:
        logger.info(
            "combined_direction_discount_applied",
            p_logreg_win=p_logreg_win,
            dir_prob=dir_prob,
            min_direction_prob=min_dir_prob_val,
            strong_threshold=strong_thresh,
            discount_weight=discount_weight,
            p_candidate_win=p_candidate_win,
            multiplier=discount_mult,
        )

    min_win_prob_cfg = getattr(cfg, "min_win_prob", 0.51)
    if p_candidate_win < min_win_prob_cfg:
        return CombinedEntryResult(
            action="SKIP",
            reason=f"Candidate win prob {p_candidate_win:.4f} < min {min_win_prob_cfg:.4f}",
            direction_status=dir_status_for_result,
            direction_model_key=crypto_sig.model_key or None,
            direction_model_version=crypto_sig.model_version,
            direction_regime=crypto_sig.regime or None,
            direction_probability=dir_prob,
            direction_p_up=getattr(crypto_sig, 'p_up', None),
            direction_p_down=getattr(crypto_sig, 'p_down', None),
            direction_threshold_up=getattr(crypto_sig, 'threshold_up', None),
            direction_threshold_down=getattr(crypto_sig, 'threshold_down', None),
            direction_value=dir_val,
            entry_requested_key=entry_requested_key,
            entry_model_key=entry_model_key,
            entry_model_version=entry_model_version,
            entry_model_phase=market_phase,
            entry_model_source=entry_model_source,
            entry_status="LOW_WIN_PROB",
            fallback_reason=fallback_reason,
            p_candidate_win=p_candidate_win,
            p_logreg_win=p_logreg_win,
            direction_discount_applied=discount_mult,
            combined_dir_discount_weight=discount_weight,
            lr_direction_vote=lr_vote,
            lgbm_direction_vote=lgbm_vote,
            consensus_type=consensus.consensus_type,
            candidate_side=candidate_side,
            candidate_ask=candidate_ask,
            cost_buffer=cost_buffer,
            strike_source="BINANCE_LAST_CANDLE" if strike else None,
            strike_proxy=strike,
            underlying_price=und_price,
            distance_to_strike_pct=dist_pct,
            p_flip=p_flip,
        )

    is_valid_time, time_reason = cfg.is_time_valid(time_left_sec, is_outsider)
    if not is_valid_time:
        return CombinedEntryResult(
            action="SKIP",
            reason=f"{time_reason}",
            direction_status=dir_status_for_result,
            direction_model_key=crypto_sig.model_key or None,
            direction_model_version=crypto_sig.model_version,
            direction_regime=crypto_sig.regime or None,
            direction_probability=dir_prob,
            direction_p_up=getattr(crypto_sig, 'p_up', None),
            direction_p_down=getattr(crypto_sig, 'p_down', None),
            direction_threshold_up=getattr(crypto_sig, 'threshold_up', None),
            direction_threshold_down=getattr(crypto_sig, 'threshold_down', None),
            direction_value=dir_val,
            entry_requested_key=entry_requested_key,
            entry_model_key=entry_model_key,
            entry_model_version=entry_model_version,
            entry_model_phase=market_phase,
            entry_model_source=entry_model_source,
            entry_status="INVALID_TIME",
            fallback_reason=fallback_reason,
            p_candidate_win=p_candidate_win,
            p_flip=p_flip,
            strike_source="BINANCE_LAST_CANDLE" if strike else None,
            strike_proxy=strike,
            underlying_price=und_price,
            distance_to_strike_pct=dist_pct,
        )
    
    if is_outsider and not cfg.trade_on_flip:
        return CombinedEntryResult(
            action="SKIP",
            reason=f"TRADE_ON_FLIP is disabled, skipping outsider candidate {candidate_side}",
            direction_status=dir_status_for_result,
            direction_model_key=crypto_sig.model_key or None,
            direction_model_version=crypto_sig.model_version,
            direction_regime=crypto_sig.regime or None,
            direction_probability=dir_prob,
            direction_p_up=getattr(crypto_sig, 'p_up', None),
            direction_p_down=getattr(crypto_sig, 'p_down', None),
            direction_threshold_up=getattr(crypto_sig, 'threshold_up', None),
            direction_threshold_down=getattr(crypto_sig, 'threshold_down', None),
            direction_value=dir_val,
            entry_requested_key=entry_requested_key,
            entry_model_key=entry_model_key,
            entry_model_version=entry_model_version,
            entry_model_phase=market_phase,
            entry_model_source=entry_model_source,
            entry_status="OUTSIDER_DISABLED",
            fallback_reason=fallback_reason,
            p_candidate_win=p_candidate_win,
            p_logreg_win=p_logreg_win,
            direction_discount_applied=discount_mult,
            combined_dir_discount_weight=discount_weight,
            lr_direction_vote=lr_vote,
            lgbm_direction_vote=lgbm_vote,
            consensus_type=consensus.consensus_type,
            candidate_side=candidate_side,
            candidate_ask=candidate_ask,
            cost_buffer=cost_buffer,
            strike_source="BINANCE_LAST_CANDLE" if strike else None,
            strike_proxy=strike,
            underlying_price=und_price,
            distance_to_strike_pct=dist_pct,
            p_flip=p_flip,
        )

    if is_outsider:
        flip_thresh_val = cfg.flip_threshold / 100.0 if cfg.flip_threshold > 1.0 else cfg.flip_threshold
        if p_flip is not None and p_flip < flip_thresh_val:
            return CombinedEntryResult(
                action="SKIP",
                reason=f"p_flip {p_flip:.4f} < FLIP_THRESHOLD {flip_thresh_val:.4f}",
                direction_status=dir_status_for_result,
                direction_model_key=crypto_sig.model_key or None,
                direction_model_version=crypto_sig.model_version,
                direction_regime=crypto_sig.regime or None,
                direction_probability=dir_prob,
                direction_p_up=getattr(crypto_sig, 'p_up', None),
                direction_p_down=getattr(crypto_sig, 'p_down', None),
                direction_threshold_up=getattr(crypto_sig, 'threshold_up', None),
                direction_threshold_down=getattr(crypto_sig, 'threshold_down', None),
                direction_value=dir_val,
                entry_requested_key=entry_requested_key,
                entry_model_key=entry_model_key,
                entry_model_version=entry_model_version,
                entry_model_phase=market_phase,
                entry_model_source=entry_model_source,
                entry_status="PFLIP_BELOW_FLIP_THRESHOLD",
                fallback_reason=fallback_reason,
                p_candidate_win=p_candidate_win,
                p_logreg_win=p_logreg_win,
                direction_discount_applied=discount_mult,
                combined_dir_discount_weight=discount_weight,
                lr_direction_vote=lr_vote,
                lgbm_direction_vote=lgbm_vote,
                consensus_type=consensus.consensus_type,
                candidate_side=candidate_side,
                candidate_ask=candidate_ask,
                cost_buffer=cost_buffer,
                strike_source="BINANCE_LAST_CANDLE" if strike else None,
                strike_proxy=strike,
                underlying_price=und_price,
                distance_to_strike_pct=dist_pct,
                p_flip=p_flip,
            )

    # (Перенесено наверх, до шага 4)

    # 5. Проверка диапазона цен покупки
    is_valid_price, price_reason = cfg.is_price_valid(candidate_ask, is_outsider)
    if not is_valid_price:
        return CombinedEntryResult(
            action="SKIP",
            reason=f"{price_reason}",
            direction_status=dir_status_for_result,
            direction_model_key=crypto_sig.model_key or None,
            direction_model_version=crypto_sig.model_version,
            direction_regime=crypto_sig.regime or None,
            direction_probability=dir_prob,
            direction_p_up=getattr(crypto_sig, 'p_up', None),
            direction_p_down=getattr(crypto_sig, 'p_down', None),
            direction_threshold_up=getattr(crypto_sig, 'threshold_up', None),
            direction_threshold_down=getattr(crypto_sig, 'threshold_down', None),
            direction_value=dir_val,
            entry_requested_key=entry_requested_key,
            entry_model_key=entry_model_key,
            entry_model_version=entry_model_version,
            entry_model_phase=market_phase,
            entry_model_source=entry_model_source,
            entry_status="PRICE_OUT_OF_BOUNDS",
            fallback_reason=fallback_reason,
            p_candidate_win=p_candidate_win,
            p_logreg_win=p_logreg_win,
            direction_discount_applied=discount_mult,
            combined_dir_discount_weight=discount_weight,
            lr_direction_vote=lr_vote,
            lgbm_direction_vote=lgbm_vote,
            consensus_type=consensus.consensus_type,
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
    min_net_edge = cfg.get_min_edge(is_outsider)

    if net_edge < min_net_edge:
        return CombinedEntryResult(
            action="SKIP",
            reason=f"Insufficient net edge: {net_edge:.4f} < min {min_net_edge:.4f} (gross={gross_edge:.4f}, buffer={cost_buffer:.4f})",
            direction_status=dir_status_for_result,
            direction_model_key=crypto_sig.model_key or None,
            direction_model_version=crypto_sig.model_version,
            direction_regime=crypto_sig.regime or None,
            direction_probability=dir_prob,
            direction_p_up=getattr(crypto_sig, 'p_up', None),
            direction_p_down=getattr(crypto_sig, 'p_down', None),
            direction_threshold_up=getattr(crypto_sig, 'threshold_up', None),
            direction_threshold_down=getattr(crypto_sig, 'threshold_down', None),
            direction_value=dir_val,
            entry_requested_key=entry_requested_key,
            entry_model_key=entry_model_key,
            entry_model_version=entry_model_version,
            entry_model_phase=market_phase,
            entry_model_source=entry_model_source,
            entry_status="INSUFFICIENT_NET_EDGE",
            fallback_reason=fallback_reason,
            p_candidate_win=p_candidate_win,
            p_logreg_win=p_logreg_win,
            direction_discount_applied=discount_mult,
            combined_dir_discount_weight=discount_weight,
            lr_direction_vote=lr_vote,
            lgbm_direction_vote=lgbm_vote,
            consensus_type=consensus.consensus_type,
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
    bet_size = _resolve_final_bet(net_edge, volume_5min, cfg, is_outsider)

    bypass_bet = cfg.bypass_bet_size_check
    if bet_size <= 0 and not bypass_bet:
        return CombinedEntryResult(
            action="SKIP",
            reason="Calculated bet size is 0.0 USDC",
            direction_status=dir_status_for_result,
            direction_model_key=crypto_sig.model_key or None,
            direction_model_version=crypto_sig.model_version,
            direction_regime=crypto_sig.regime or None,
            direction_probability=dir_prob,
            direction_p_up=getattr(crypto_sig, 'p_up', None),
            direction_p_down=getattr(crypto_sig, 'p_down', None),
            direction_threshold_up=getattr(crypto_sig, 'threshold_up', None),
            direction_threshold_down=getattr(crypto_sig, 'threshold_down', None),
            direction_value=dir_val,
            entry_requested_key=entry_requested_key,
            entry_model_key=entry_model_key,
            entry_model_version=entry_model_version,
            entry_model_phase=market_phase,
            entry_model_source=entry_model_source,
            entry_status="ZERO_BET",
            fallback_reason=fallback_reason,
            p_candidate_win=p_candidate_win,
            p_logreg_win=p_logreg_win,
            direction_discount_applied=discount_mult,
            combined_dir_discount_weight=discount_weight,
            lr_direction_vote=lr_vote,
            lgbm_direction_vote=lgbm_vote,
            consensus_type=consensus.consensus_type,
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
    max_drift = cfg.max_price_drift
    # Максимально допустимая цена исполнения не должна снижать net_edge ниже min_net_edge
    max_price_by_edge = round(p_candidate_win - cost_buffer - min_net_edge, 3)
    max_price_by_drift = round(candidate_ask + max_drift, 3)
    max_acceptable_price = min(max_price_by_edge, max_price_by_drift, cfg.trade_max_price)

    return CombinedEntryResult(
        action=candidate_side,
        reason=f"COMBINED {candidate_side}: net_edge={net_edge:.4f} (gross={gross_edge:.4f}, dir={crypto_sig.direction}, p_win={p_candidate_win:.3f})",
        direction_status=dir_status_for_result,
        direction_model_key=crypto_sig.model_key or None,
        direction_model_version=crypto_sig.model_version,
        direction_regime=crypto_sig.regime or None,
        direction_probability=dir_prob,
        direction_p_up=getattr(crypto_sig, 'p_up', None),
        direction_p_down=getattr(crypto_sig, 'p_down', None),
        direction_threshold_up=getattr(crypto_sig, 'threshold_up', None),
        direction_threshold_down=getattr(crypto_sig, 'threshold_down', None),
        direction_value=dir_val,
        entry_requested_key=entry_requested_key,
        entry_model_key=entry_model_key,
        entry_model_version=entry_model_version,
        entry_model_phase=market_phase,
        entry_model_source=entry_model_source,
        entry_status="READY",
        fallback_reason=fallback_reason,
        p_candidate_win=p_candidate_win,
        p_logreg_win=p_logreg_win,
        direction_discount_applied=discount_mult,
        combined_dir_discount_weight=discount_weight,
        lr_direction_vote=lr_vote,
        lgbm_direction_vote=lgbm_vote,
        consensus_type=consensus.consensus_type,
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
    """Legacy helper maintained for backward compatibility with older tests.
    
    .. deprecated::
        Use evaluate_combined_entry instead.
    """
    import warnings
    warnings.warn(
        "combine_votes is deprecated. Use evaluate_combined_entry instead.",
        DeprecationWarning,
        stacklevel=2,
    )
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
