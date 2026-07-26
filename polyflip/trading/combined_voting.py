"""
Таблица голосования для COMBINED-режима.

ML-модель (LogReg) предсказывает p_flip (вероятность флипа рынка).
LightGBM-модель предсказывает направление крипто-цены (UP/DOWN).

Правила объединения:
┌──────────────┬────────────────┬──────────────────────────────────┐
│  ML-сигнал   │ LightGBM-сигн. │   Решение                        │
├──────────────┼────────────────┼──────────────────────────────────┤
│ BUY_YES      │ UP             │ BUY_YES (полный размер)           │
│ BUY_YES      │ DOWN           │ SKIP (вето)                       │
│ BUY_YES      │ NONE           │ BUY_YES (50% размер, без буста)   │
│ BUY_NO       │ DOWN           │ BUY_NO (полный размер)            │
│ BUY_NO       │ UP             │ SKIP (вето)                       │
│ BUY_NO       │ NONE           │ BUY_NO (50% размер, без буста)    │
│ SKIP         │ любой          │ SKIP                              │
│ любой        │ features_ok=F  │ fallback → ML                     │
└──────────────┴────────────────┴──────────────────────────────────┘
"""
from dataclasses import dataclass
from typing import Literal, Optional
import structlog

logger = structlog.get_logger(__name__)

@dataclass(frozen=True)
class CryptoSignalProxy:
    direction: Optional[Literal["UP", "DOWN", "NONE"]]
    features_ok: bool
    model_version: Optional[int] = None
    risk_vetoed: bool = False

@dataclass(frozen=True)
class VotingResult:
    action: Literal["BUY_YES", "BUY_NO", "SKIP"]
    reason: str
    confidence: float            # 0.0–1.0, для логирования
    ml_action: str
    lgbm_direction: Optional[str]
    lgbm_features_ok: bool
    bet_size_multiplier: float = 1.0  # 1.0 = полный, 0.5 = уменьшенный, 0.0 = вето

def combine_votes(
    ml_action: str,
    ml_edge: float,
    crypto_sig: "CryptoSignal",
    asset: str,
    none_bet_multiplier: float = 0.5,
    ml_skip_reason: str = "",
) -> VotingResult:
    """
    Основная таблица голосования с поддержкой уменьшенного размера ставки при NONE
    и автономной торговлей LightGBM при мягких пропусках ML (Soft SKIP).
    """
    if not crypto_sig.features_ok:
        # LightGBM-фичи недоступны → fallback на ML-решение без вето
        logger.warning("combined_lgbm_features_invalid_fallback", asset=asset)
        return VotingResult(
            action=ml_action,
            reason="LightGBM features invalid, fallback to ML-only",
            confidence=ml_edge,
            ml_action=ml_action,
            lgbm_direction=crypto_sig.direction,
            lgbm_features_ok=crypto_sig.features_ok,
            bet_size_multiplier=1.0,
        )

    if crypto_sig.risk_vetoed:
        logger.warning("combined_lgbm_risk_vetoed", asset=asset, direction=crypto_sig.direction)
        return VotingResult(
            action="SKIP",
            reason="Hard Veto: LightGBM risk veto",
            confidence=1.0,
            ml_action=ml_action,
            lgbm_direction=crypto_sig.direction,
            lgbm_features_ok=crypto_sig.features_ok,
            bet_size_multiplier=0.0,
        )


    if ml_action == "SKIP":
        skip_reason_lower = (ml_skip_reason or "").lower()

        # Soft SKIP = технические причины (зона, цена вне диапазона)
        # LightGBM может торговать автономно с 50% ставкой
        SOFT_SKIP_PATTERNS = ("dead zone", "dead_zone", "price out of", "out of bounds", "мёртвая зона", "мёртвая")
        is_soft = any(p in skip_reason_lower for p in SOFT_SKIP_PATTERNS)

        # Если ML скипнул из-за FLIP_THRESHOLD / p_flip (и не из-за dead zone) — Hard Veto
        flip_threshold_veto = (
            ("threshold" in skip_reason_lower or "p_flip" in skip_reason_lower) and not is_soft
        )

        if flip_threshold_veto:
            return VotingResult(
                action="SKIP",
                reason=f"FLIP_THRESHOLD veto: {ml_skip_reason or 'p_flip failed threshold'}",
                confidence=0.0,
                ml_action=ml_action,
                lgbm_direction=crypto_sig.direction,
                lgbm_features_ok=True,
                bet_size_multiplier=0.0,
            )

        # Soft SKIP = технические причины (зона, цена вне диапазона)
        # LightGBM может торговать автономно с 50% ставкой
        # ВРЕМЕННО ОТКЛЮЧЕНО
        SOFT_SKIP_PATTERNS = ("dead zone", "price out of", "out of bounds", "мёртвая зона")
        is_soft = False # any(p in skip_reason_lower for p in SOFT_SKIP_PATTERNS)

        if is_soft and crypto_sig.features_ok and crypto_sig.direction not in (None, "NONE"):
            lgbm_action = "BUY_YES" if crypto_sig.direction == "UP" else "BUY_NO"
            return VotingResult(
                action=lgbm_action,
                reason=f"ML soft-SKIP ({ml_skip_reason}), LightGBM autonomous",
                confidence=0.35,
                ml_action=ml_action,
                lgbm_direction=crypto_sig.direction,
                lgbm_features_ok=True,
                bet_size_multiplier=0.5,
            )

        return VotingResult(
            action="SKIP",
            reason=f"ML hard-SKIP: {ml_skip_reason or 'ML voted SKIP'}",
            confidence=0.0,
            ml_action=ml_action,
            lgbm_direction=crypto_sig.direction,
            lgbm_features_ok=True,
            bet_size_multiplier=0.0,
        )

    # Согласование направлений
    ml_direction = "UP" if ml_action == "BUY_YES" else "DOWN"

    if crypto_sig.direction == "NONE":
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
        # LGBM во флэте — не знает, но и не против. Уменьшаем ставку вместо вето.
        return VotingResult(
            action=ml_action,
            reason=f"LightGBM flat (NONE): ML={ml_action}, reduced bet size",
            confidence=ml_edge * 0.7,
            ml_action=ml_action,
            lgbm_direction="NONE",
            lgbm_features_ok=True,
            bet_size_multiplier=none_bet_multiplier,
        )

    if crypto_sig.direction == ml_direction:
        return VotingResult(
            action=ml_action,
            reason=f"Both models agree: ML={ml_action}, LightGBM={crypto_sig.direction}",
            confidence=min(1.0, ml_edge * 1.2),  # небольшой буст при согласии (только для логов)
            ml_action=ml_action,
            lgbm_direction=crypto_sig.direction,
            lgbm_features_ok=True,
            bet_size_multiplier=1.0,
        )
    else:
        return VotingResult(
            action="SKIP",
            reason=f"LightGBM veto: ML={ml_action} but LightGBM={crypto_sig.direction}",
            confidence=0.0,
            ml_action=ml_action,
            lgbm_direction=crypto_sig.direction,
            lgbm_features_ok=True,
            bet_size_multiplier=0.0,
        )

