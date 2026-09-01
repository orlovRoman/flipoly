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
from polyflip.trading.weighted_policy import (
    WeightedPolicyConfig,
    WeightedSelection,
    logreg_flip_to_yes_probability,
    market_yes_probability,
    probability_for_side,
    select_weighted_side,
)

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
    direction_raw_opinion: Optional[str] = None
    direction_p_up_raw: Optional[float] = None
    direction_p_down_raw: Optional[float] = None
    p_flip_raw: Optional[float] = None
    p_flip_effective: Optional[float] = None
    entry_model_ece: float = 0.0
    would_live_accept: Optional[bool] = None
    # Weighted policy telemetry.  These fields are also populated in shadow
    weighted_market_contribution_logodds: Optional[float] = None
    weighted_logreg_contribution_logodds: Optional[float] = None
    weighted_lgbm_contribution_logodds: Optional[float] = None
    weighted_mrf_contribution_logodds: Optional[float] = None
    weighted_intercept_contribution_logodds: Optional[float] = None
    weighted_models_agree: Optional[bool] = None
    # mode, while LEGACY behavior remains unchanged.
    weighted_policy_id: str = "UNVERSIONED"
    weighted_policy_mode: str = "LEGACY"
    weighted_p_market_yes: Optional[float] = None
    weighted_p_logreg_yes: Optional[float] = None
    weighted_p_lgbm_yes: Optional[float] = None
    weighted_p_final_yes: Optional[float] = None
    weighted_market_weight: Optional[float] = None
    weighted_logreg_weight: Optional[float] = None
    weighted_lgbm_weight: Optional[float] = None
    weighted_mrf_evidence: Optional[float] = None
    weighted_selected_side: Optional[str] = None
    weighted_yes_net_ev: Optional[float] = None
    weighted_no_net_ev: Optional[float] = None
    weighted_net_ev_per_share: Optional[float] = None
    weighted_cost_per_share: Optional[float] = None
    weighted_fee_rate: Optional[float] = None
    weighted_maker_fee_rate: Optional[float] = None
    weighted_execution_role: Optional[str] = None
    weighted_fee_exponent: Optional[float] = None
    weighted_fee_per_share: Optional[float] = None
    weighted_maker_fee_per_share: Optional[float] = None
    weighted_taker_fee_per_share: Optional[float] = None
    weighted_slippage_per_share: Optional[float] = None
    weighted_spread_per_share: Optional[float] = None
    weighted_latency_buffer_per_share: Optional[float] = None
    weighted_expected_execution_price: Optional[float] = None
    weighted_missing_components: Optional[str] = None
    weighted_selection_reason: Optional[str] = None
    weighted_fee_source: Optional[str] = None


def _normalize_threshold(value: float) -> float:
    """Нормализует порог: если передан в процентах (> 1.0), делит на 100."""
    return value / 100.0 if value > 1.0 else value


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


def _build_weighted_selection(
    *,
    crypto_sig: CryptoSignal,
    p_flip: Optional[float],
    fresh_yes_price: float,
    yes_ask: Optional[float],
    no_ask: Optional[float],
    cfg: "TradingConfig",
    mrf_evidence: Optional[float] = None,
    fee_rate: Optional[float] = None,
    maker_fee_rate: Optional[float] = None,
    taker_only: bool = False,
    fee_exponent: Optional[float] = None,
    fee_source: str = "CONFIG_DEFAULT",
    spread: float = 0.0,
    spread_cost: Optional[float] = None,
) -> WeightedSelection:
    """Build the shared weighted-policy result for active or shadow mode."""
    lgbm_available = bool(
        getattr(crypto_sig, "features_ok", False)
        and getattr(crypto_sig, "model_version", None) is not None
        and getattr(crypto_sig, "model_version", -1) >= 0
    )
    p_lgbm_yes = getattr(crypto_sig, "p_up", None) if lgbm_available else None
    policy_cfg = WeightedPolicyConfig(
        market_weight=float(getattr(cfg, "weighted_market_weight", 0.90)),
        logreg_weight=float(getattr(cfg, "weighted_logreg_weight", 0.05)),
        lgbm_weight=float(getattr(cfg, "weighted_lgbm_weight", 0.05)),
        mrf_beta=float(getattr(cfg, "weighted_mrf_beta", 0.0)),
        intercept=float(getattr(cfg, "weighted_intercept", 0.0)),
        fee_rate=(
            float(fee_rate)
            if fee_rate is not None
            else float(getattr(cfg, "weighted_fee_rate", 0.07))
        ),
        maker_fee_rate=(
            float(maker_fee_rate)
            if maker_fee_rate is not None
            else float(getattr(cfg, "weighted_maker_fee_rate", 0.0))
        ),
        fee_exponent=(
            float(fee_exponent)
            if fee_exponent is not None
            else float(getattr(cfg, "weighted_fee_exponent", 1.0))
        ),
        slippage_rate=float(getattr(cfg, "weighted_slippage_rate", 0.005)),
        latency_buffer=float(getattr(cfg, "weighted_latency_buffer", 0.0)),
        execution_role=(
            "TAKER"
            if taker_only
            else str(getattr(cfg, "weighted_execution_role", "TAKER"))
        ),
        policy_id=str(getattr(cfg, "weighted_policy_id", "UNVERSIONED") or "UNVERSIONED")[:64],
    )
    return select_weighted_side(
        p_market_yes=market_yes_probability(
            yes_ask=yes_ask,
            no_ask=no_ask,
            fallback_yes=fresh_yes_price,
        ),
        p_logreg_yes=logreg_flip_to_yes_probability(p_flip, fresh_yes_price),
        p_lgbm_yes=p_lgbm_yes,
        yes_ask=yes_ask,
        no_ask=no_ask,
        config=policy_cfg,
        mrf_evidence=mrf_evidence,
        min_net_ev=0.0,
        fee_source=fee_source,
        spread=spread if spread_cost is None else spread_cost,
        mrf_extreme_veto_threshold=getattr(cfg, "weighted_mrf_extreme_veto_threshold", -1.0),
    )


def _weighted_result_fields(
    policy_mode: str,
    selection: WeightedSelection,
    policy_id: str,
) -> dict[str, Any]:
    """Flatten weighted policy telemetry into ``CombinedEntryResult`` fields."""
    probability = selection.probability
    selected = selection.selected
    quoted = selected or selection.best_quote
    return {
        "weighted_policy_mode": policy_mode,
        "weighted_p_market_yes": probability.p_market_yes,
        "weighted_policy_id": policy_id,
        "weighted_p_logreg_yes": probability.p_logreg_yes,
        "weighted_p_lgbm_yes": probability.p_lgbm_yes,
        "weighted_p_final_yes": probability.p_final_yes,
        "weighted_market_weight": probability.market_weight,
        "weighted_logreg_weight": probability.logreg_weight,
        "weighted_lgbm_weight": probability.lgbm_weight,
        "weighted_market_contribution_logodds": probability.market_contribution_logodds,
        "weighted_logreg_contribution_logodds": probability.logreg_contribution_logodds,
        "weighted_lgbm_contribution_logodds": probability.lgbm_contribution_logodds,
        "weighted_mrf_contribution_logodds": probability.mrf_adjustment_logodds,
        "weighted_intercept_contribution_logodds": probability.intercept_contribution_logodds,
        "weighted_models_agree": probability.models_agree,
        "weighted_mrf_evidence": probability.mrf_evidence,
        "weighted_selected_side": selected.side if selected else None,
        "weighted_yes_net_ev": selection.yes_quote.net_ev_per_share if selection.yes_quote else None,
        "weighted_no_net_ev": selection.no_quote.net_ev_per_share if selection.no_quote else None,
        "weighted_net_ev_per_share": selected.net_ev_per_share if selected else None,
        "weighted_cost_per_share": quoted.cost.total_per_share if quoted else None,
        "weighted_fee_rate": quoted.cost.fee_rate if quoted else None,
        "weighted_maker_fee_rate": quoted.cost.maker_fee_rate if quoted else None,
        "weighted_execution_role": quoted.cost.role if quoted else None,
        "weighted_fee_exponent": quoted.cost.fee_exponent if quoted else None,
        "weighted_fee_per_share": quoted.cost.fee_per_share if quoted else None,
        "weighted_maker_fee_per_share": quoted.cost.maker_fee_per_share if quoted else None,
        "weighted_taker_fee_per_share": quoted.cost.taker_fee_per_share if quoted else None,
        "weighted_slippage_per_share": quoted.cost.slippage_per_share if quoted else None,
        "weighted_spread_per_share": quoted.cost.spread_per_share if quoted else None,
        "weighted_latency_buffer_per_share": quoted.cost.latency_buffer_per_share if quoted else None,
        "weighted_expected_execution_price": quoted.cost.expected_execution_price if quoted else None,
        "weighted_missing_components": ",".join(probability.missing_components) or None,
        "weighted_selection_reason": selection.reason,
        "weighted_fee_source": quoted.cost.source if quoted else None,
    }


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
    weighted_maker_fee_rate: Optional[float] = None,
    weighted_taker_only: bool = False,
    fallback_reason: Optional[str] = None,
    time_left_sec: float = 0.0,
    entry_model_ece: float = 0.0,
    flip_threshold: Optional[float] = None,
    mrf_evidence: Optional[float] = None,
    weighted_fee_rate: Optional[float] = None,
    weighted_fee_exponent: Optional[float] = None,
    weighted_fee_source: str = "CONFIG_DEFAULT",
    spread: Optional[float] = None,
    spread_cost: Optional[float] = None,
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
        entry_model_ece=entry_model_ece,
        flip_threshold=flip_threshold,
        mrf_evidence=mrf_evidence,
        weighted_fee_rate=weighted_fee_rate,
        weighted_maker_fee_rate=weighted_maker_fee_rate,
        weighted_taker_only=weighted_taker_only,
        weighted_fee_exponent=weighted_fee_exponent,
        weighted_fee_source=weighted_fee_source,
        spread=spread,
        spread_cost=spread_cost,
    )
    policy_mode = str(getattr(cfg, "trading_policy_mode", "LEGACY") or "LEGACY").upper()
    if policy_mode in {"WEIGHTED_SHADOW", "WEIGHTED_ACTIVE"}:
        # Compute once at the wrapper boundary so SHADOW has identical maths
        # to ACTIVE without changing the legacy result returned by ``inner``.
        shadow_p_flip = result.p_flip_effective if result.p_flip_effective is not None else p_flip
        weighted_selection = _build_weighted_selection(
            crypto_sig=crypto_sig,
            p_flip=shadow_p_flip,
            fresh_yes_price=fresh_yes_price,
            yes_ask=yes_ask,
            no_ask=no_ask,
            cfg=cfg,
            mrf_evidence=mrf_evidence,
            fee_rate=weighted_fee_rate,
            maker_fee_rate=weighted_maker_fee_rate,
            taker_only=weighted_taker_only,
            fee_exponent=weighted_fee_exponent,
            fee_source=weighted_fee_source,
            spread=spread or 0.0,
            spread_cost=spread_cost,
        )
        result = replace(result, **_weighted_result_fields(policy_mode, weighted_selection, str(getattr(cfg, "weighted_policy_id", "UNVERSIONED") or "UNVERSIONED")[:64]))

    if crypto_sig:
        result = replace(
            result,
            lgbm_inverted=getattr(crypto_sig, "inverted", False),
            lgbm_p_up_raw=getattr(crypto_sig, "p_up_raw", 0.0),
            lgbm_p_down_raw=getattr(crypto_sig, "p_down_raw", 0.0),
            direction_raw_opinion=getattr(crypto_sig, "raw_opinion", None),
            direction_p_up_raw=getattr(crypto_sig, "p_up_raw", None),
            direction_p_down_raw=getattr(crypto_sig, "p_down_raw", None),
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
    weighted_maker_fee_rate: Optional[float] = None,
    weighted_taker_only: bool = False,
    entry_model_ece: float = 0.0,
    flip_threshold: Optional[float] = None,
    mrf_evidence: Optional[float] = None,
    weighted_fee_rate: Optional[float] = None,
    weighted_fee_exponent: Optional[float] = None,
    weighted_fee_source: str = "CONFIG_DEFAULT",
    spread: Optional[float] = None,
    spread_cost: Optional[float] = None,
) -> CombinedEntryResult:
    """Внутренняя логика оценки."""

    # ECE-коррекция p_flip (управляется параметром enable_ece_correction)
    p_flip_raw = p_flip
    ece_val = float(entry_model_ece) if isinstance(entry_model_ece, (int, float)) and not isinstance(entry_model_ece, bool) else 0.0
    if p_flip is not None and ece_val > 0.0 and getattr(cfg, "enable_ece_correction", True):
        from polyflip.trading.position_sizing import apply_ece_correction
        p_flip_effective = apply_ece_correction(p_flip, entry_model_ece)
        logger.info(
            "ece_correction_applied",
            p_flip_raw=round(p_flip_raw, 4),
            p_flip_effective=round(p_flip_effective, 4),
            entry_model_ece=round(entry_model_ece, 4),
        )
    else:
        p_flip_effective = p_flip

    p_flip = p_flip_effective

    configured_flip_threshold = cfg.flip_threshold if flip_threshold is None else flip_threshold
    try:
        flip_threshold_value = _normalize_threshold(float(configured_flip_threshold))
    except (TypeError, ValueError):
        flip_threshold_value = _normalize_threshold(float(cfg.flip_threshold))
    if not 0.0 <= flip_threshold_value <= 1.0:
        flip_threshold_value = _normalize_threshold(float(cfg.flip_threshold))

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

    lgbm_mode = getattr(cfg, "lightgbm_decision_mode", "SHADOW")
    policy_mode = str(getattr(cfg, "trading_policy_mode", "LEGACY") or "LEGACY").upper()
    weighted_active = policy_mode == "WEIGHTED_ACTIVE"
    logreg_only = lgbm_mode in {"SHADOW", "OFF"} and not weighted_active

    weighted_selection: Optional[WeightedSelection] = None
    if weighted_active:
        weighted_selection = _build_weighted_selection(
            crypto_sig=crypto_sig,
            p_flip=p_flip,
            fresh_yes_price=fresh_yes_price,
            yes_ask=yes_ask,
            no_ask=no_ask,
            cfg=cfg,
            mrf_evidence=mrf_evidence,
            fee_rate=weighted_fee_rate,
            maker_fee_rate=weighted_maker_fee_rate,
            taker_only=weighted_taker_only,
            fee_exponent=weighted_fee_exponent,
            fee_source=weighted_fee_source,
            spread=spread or 0.0,
            spread_cost=spread_cost,
        )
        if weighted_selection.selected is None:
            consensus = DirectionConsensus(
                "SKIP",
                "WEIGHTED_SCORE",
                weighted_selection.reason,
            )
        else:
            consensus = DirectionConsensus(
                cast(Literal["BUY_YES", "BUY_NO", "SKIP"], weighted_selection.candidate_side),
                "WEIGHTED_SCORE",
                "Selected side with highest cost-aware expected value",
            )
    if weighted_active and weighted_selection is not None and weighted_selection.selected is None:
        return CombinedEntryResult(
            action="SKIP",
            reason=f"Weighted policy: {weighted_selection.reason}",
            direction_status="WEIGHTED_NO_SELECTION",
            direction_model_key=crypto_sig.model_key or None,
            direction_model_version=crypto_sig.model_version,
            direction_regime=crypto_sig.regime or None,
            direction_probability=dir_prob,
            direction_p_up=getattr(crypto_sig, "p_up", None),
            direction_p_down=getattr(crypto_sig, "p_down", None),
            direction_value=dir_val,
            entry_requested_key=entry_requested_key,
            entry_model_key=entry_model_key,
            entry_model_version=entry_model_version,
            entry_model_phase=market_phase,
            entry_model_source=entry_model_source,
            entry_status="WEIGHTED_SKIP",
            fallback_reason=fallback_reason,
            p_flip=p_flip,
            p_flip_raw=p_flip_raw,
            p_flip_effective=p_flip_effective,
            entry_model_ece=entry_model_ece,
            would_live_accept=False,
        )
    elif logreg_only:
        dir_status = "SHADOW_NOT_APPLIED" if lgbm_mode == "SHADOW" else "DISABLED_BY_OPERATOR"

    # 1. Валидация LightGBM Direction (пропускается в режиме logreg_only)
    if not weighted_active and not logreg_only and (not crypto_sig.features_ok or crypto_sig.model_version is None or crypto_sig.model_version < 0):
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

        unavailable_statuses = {"MODEL_NOT_LOADED", "REGIME_UNAVAILABLE", "INFERENCE_FAILED"}
        policy = getattr(cfg, "lgbm_unavailable_policy", "SKIP")
        if dir_status in unavailable_statuses and policy == "LOGREG_FALLBACK":
            logger.warning(
                "lgbm_unavailable_logreg_fallback",
                status=dir_status,
                symbol=crypto_sig.symbol,
                reason=reason_str,
            )
            crypto_sig = replace(
                crypto_sig,
                direction="NONE",
                features_ok=True,
                model_version=-1,
                status="LGBM_FALLBACK",
            )
            dir_status = "LGBM_FALLBACK"
            dir_val = "NONE"
        else:
            return CombinedEntryResult(
                action="SKIP",
                reason=reason_str,
                direction_status=dir_status,
                direction_error_detail=error_detail or None,
                direction_model_key=crypto_sig.model_key or None,
                direction_model_version=crypto_sig.model_version if (crypto_sig.model_version is not None and crypto_sig.model_version >= 0) else None,
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
                p_flip_raw=p_flip_raw,
                p_flip_effective=p_flip_effective,
                entry_model_ece=entry_model_ece,
                would_live_accept=False,
            )

    if not logreg_only and crypto_sig.risk_vetoed:
        return CombinedEntryResult(
            action="SKIP",
            reason=f"Direction Model funding veto: {crypto_sig.risk_reason}",
            direction_status="FUNDING_VETOED",
            direction_model_key=crypto_sig.model_key or None,
            direction_model_version=crypto_sig.model_version if (crypto_sig.model_version is not None and crypto_sig.model_version >= 0) else None,
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
            p_flip_raw=p_flip_raw,
            p_flip_effective=p_flip_effective,
            entry_model_ece=entry_model_ece,
            would_live_accept=False,
        )

    # 3. Валидация LogReg Entry Model
    if not weighted_active and (p_flip is None or entry_model_key is None):
        return CombinedEntryResult(
            action="SKIP",
            reason="Entry Model (LogReg) evaluation failed or unavailable",
            direction_status=dir_status,
            direction_model_key=crypto_sig.model_key or None,
            direction_model_version=crypto_sig.model_version if (crypto_sig.model_version is not None and crypto_sig.model_version >= 0) else None,
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
            p_flip_raw=p_flip_raw,
            p_flip_effective=p_flip_effective,
            entry_model_ece=entry_model_ece,
        )

    # 3.5 Валидация минимальной уверенности LightGBM (для UP / DOWN)
    min_direction_prob_cfg = getattr(cfg, "min_direction_prob", 0.505)
    # 3.5 Валидация минимальной уверенности LightGBM (для UP / DOWN) - пропускается в logreg_only
    min_direction_prob_cfg = getattr(cfg, "min_direction_prob", 0.505)
    if not weighted_active and not logreg_only and crypto_sig.direction in ("UP", "DOWN") and dir_prob < min_direction_prob_cfg:
        return CombinedEntryResult(
            action="SKIP",
            reason=f"Direction prob {dir_prob:.4f} < min {min_direction_prob_cfg:.4f} (floor)",
            direction_status="LOW_DIRECTION_PROB",
            direction_model_key=crypto_sig.model_key or None,
            direction_model_version=crypto_sig.model_version if (crypto_sig.model_version is not None and crypto_sig.model_version >= 0) else None,
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
            p_flip_raw=p_flip_raw,
            p_flip_effective=p_flip_effective,
            entry_model_ece=entry_model_ece,
        )

    # 3.6 Consensus / LogReg-Only Direction Selection
    lr_abstain_band = getattr(cfg, "combined_logreg_abstain_band", _LOGREG_ABSTAIN_BAND)
    lr_vote = logreg_direction_vote(
        p_flip,
        fresh_yes_price,
        flip_threshold_value,
        abstain_band=lr_abstain_band,
    )
    lgbm_vote = crypto_sig.direction or "NONE"
    
    if weighted_active:
        # ``consensus`` was already selected by the cost-aware weighted
        # scorer above.  Do not let the legacy hard-vote resolver overwrite
        # that choice later in the function.
        assert weighted_selection is not None
    elif logreg_only:
        if lr_vote == "ABSTAIN":
            return CombinedEntryResult(
                action="SKIP",
                reason="LogReg did not yield a confident direction (ABSTAIN)",
                direction_status=dir_status,
                direction_model_key=crypto_sig.model_key or None,
                direction_model_version=crypto_sig.model_version if (crypto_sig.model_version is not None and crypto_sig.model_version >= 0) else None,
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
                entry_status="LOGREG_ABSTAIN",
                fallback_reason=fallback_reason,
                candidate_side=None,
                candidate_ask=None,
                cost_buffer=cost_buffer,
                strike_source="BINANCE_LAST_CANDLE" if strike else None,
                strike_proxy=strike,
                underlying_price=und_price,
                distance_to_strike_pct=dist_pct,
                p_flip=p_flip,
                p_flip_raw=p_flip_raw,
                p_flip_effective=p_flip_effective,
                entry_model_ece=entry_model_ece,
                lr_direction_vote=lr_vote,
                lgbm_direction_vote=lgbm_vote,
                consensus_type="LOGREG_ONLY",
            )
        consensus = DirectionConsensus(final_side=lr_vote, consensus_type="LOGREG_ONLY", reason="LogReg-only mode active")
    else:
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
            direction_model_version=crypto_sig.model_version if (crypto_sig.model_version is not None and crypto_sig.model_version >= 0) else None,
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
            p_flip_raw=p_flip_raw,
            p_flip_effective=p_flip_effective,
            entry_model_ece=entry_model_ece,
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

    if not logreg_only and not weighted_active and crypto_sig.direction not in ("UP", "DOWN"):
        dir_status_for_result = "DIRECTION_NONE_FALLBACK_LR"
    elif weighted_active:
        dir_status_for_result = (
            "WEIGHTED_LGBM_USED"
            if weighted_selection is not None
            and weighted_selection.probability.p_lgbm_yes is not None
            else "WEIGHTED_LGBM_MISSING"
        )
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

    # Spread is an execution-quality guard, not a directional model vote.
    # The old setting existed but was never enforced in the combined path.
    if spread is not None:
        try:
            spread_value = abs(float(spread))
            max_spread_pct = float(getattr(cfg, "max_spread_pct", 0.0))
            spread_ratio = spread_value / max(candidate_ask, 1e-9)
        except (TypeError, ValueError, OverflowError):
            spread_value = 0.0
            max_spread_pct = 0.0
            spread_ratio = 0.0
        if max_spread_pct > 0.0 and spread_ratio > max_spread_pct:
            return CombinedEntryResult(
                action="SKIP",
                reason=(
                    f"Spread {spread_value:.4f} / selected ask {candidate_ask:.4f} "
                    f"= {spread_ratio:.4f} > max {max_spread_pct:.4f}"
                ),
                direction_status=dir_status_for_result,
                direction_model_key=crypto_sig.model_key or None,
                direction_model_version=crypto_sig.model_version,
                direction_regime=crypto_sig.regime or None,
                direction_probability=dir_prob,
                direction_p_up=getattr(crypto_sig, "p_up", None),
                direction_p_down=getattr(crypto_sig, "p_down", None),
                direction_value=dir_val,
                entry_requested_key=entry_requested_key,
                entry_model_key=entry_model_key,
                entry_model_version=entry_model_version,
                entry_model_phase=market_phase,
                entry_model_source=entry_model_source,
                entry_status="SPREAD_TOO_WIDE",
                fallback_reason=fallback_reason,
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
                p_flip_raw=p_flip_raw,
                p_flip_effective=p_flip_effective,
                entry_model_ece=entry_model_ece,
            )

    # 4. Probability of the candidate side winning.
    if weighted_active:
        assert weighted_selection is not None and weighted_selection.selected is not None
        p_candidate_win = weighted_selection.p_candidate_win
        if p_candidate_win is None:
            p_candidate_win = 0.0
        p_logreg_win = probability_for_side(
            weighted_selection.probability.p_logreg_yes,
            candidate_side,
        )
        if p_logreg_win is None:
            p_logreg_win = p_candidate_win
        p_candidate_win = round(max(0.0, min(1.0, p_candidate_win)), 4)
        p_logreg_win = round(max(0.0, min(1.0, p_logreg_win)), 4)
        # The weighted score already accounts for the model blend; do not
        # apply the legacy LightGBM discount a second time.
        discount_weight = 0.0
        discount_mult = 1.0
    else:
        # Legacy p_flip conversion: probability that the selected side wins.
        if candidate_side == "BUY_YES":
            p_logreg_win = (1.0 - p_flip) if fresh_yes_price >= 0.50 else p_flip
        else:
            p_logreg_win = p_flip if fresh_yes_price >= 0.50 else (1.0 - p_flip)

        p_logreg_win = round(max(0.0, min(1.0, p_logreg_win)), 4)

        # Применяем дисконт за неуверенность LightGBM (отключен в logreg_only режиме)
        if logreg_only:
            discount_weight = 0.0
            p_candidate_win = p_logreg_win
            discount_mult = 1.0
        else:
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
    if not weighted_active and p_candidate_win < min_win_prob_cfg:
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
            p_flip_raw=p_flip_raw,
            p_flip_effective=p_flip_effective,
            entry_model_ece=entry_model_ece,
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
            underlying_price=und_price,
            distance_to_strike_pct=dist_pct,
            p_flip=p_flip,
            p_flip_raw=p_flip_raw,
            p_flip_effective=p_flip_effective,
            entry_model_ece=entry_model_ece,
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
            p_flip_raw=p_flip_raw,
            p_flip_effective=p_flip_effective,
            entry_model_ece=entry_model_ece,
        )

    if not weighted_active and is_outsider:  # legacy predictive gate
        flip_thresh_val = flip_threshold_value
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
            p_flip_raw=p_flip_raw,
            p_flip_effective=p_flip_effective,
            entry_model_ece=entry_model_ece,
        )

    # 6. Расчет Gross Edge и Net Edge.  Weighted mode uses the same
    # per-share cost estimate that selected the side; legacy keeps its
    # historical configurable buffer for backward compatibility.
    gross_edge = compute_edge(p_candidate_win, candidate_ask)
    if weighted_active and weighted_selection is not None and weighted_selection.selected is not None:
        cost_buffer = weighted_selection.selected.cost.total_per_share
        net_edge = weighted_selection.selected.net_ev_per_share
    else:
        net_edge = round(gross_edge - cost_buffer, 4)
    min_net_edge = (
        cfg.get_weighted_min_net_ev(is_outsider)
        if weighted_active
        else cfg.get_min_edge(is_outsider)
    )

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
            p_flip_raw=p_flip_raw,
            p_flip_effective=p_flip_effective,
            entry_model_ece=entry_model_ece,
        )

    # 7. Расчет размера ставки
    if weighted_active:
        bet_size = float(getattr(cfg, "weighted_fixed_bet_usdc", 1.0))
    else:
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
            p_flip_raw=p_flip_raw,
            p_flip_effective=p_flip_effective,
            entry_model_ece=entry_model_ece,
        )

    # 8. Расчет max_acceptable_price (защита от дрейфа и спреда)
    max_drift = cfg.max_price_drift
    # Максимально допустимая цена исполнения не должна снижать net_edge ниже min_net_edge
    max_price_by_edge = round(p_candidate_win - cost_buffer - min_net_edge, 3)
    max_price_by_drift = round(candidate_ask + max_drift, 3)
    max_acceptable_price = min(max_price_by_edge, max_price_by_drift, cfg.trade_max_price)

    would_live_accept = (entry_model_source == "PHASE")
    if dir_status == "LGBM_FALLBACK":
        would_live_accept = False

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
        p_flip_raw=p_flip_raw,
        p_flip_effective=p_flip_effective,
        entry_model_ece=entry_model_ece,
        would_live_accept=would_live_accept,
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
