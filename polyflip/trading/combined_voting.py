"""
Таблица голосования для COMBINED-режима.

ML-модель (LogReg) предсказывает p_flip (вероятность флипа рынка).
LightGBM-модель предсказывает направление крипто-цены (UP/DOWN).

Правила объединения:
┌──────────────┬────────────────┬──────────────┐
│  ML-сигнал   │ LightGBM-сигн. │   Решение    │
├──────────────┼────────────────┼──────────────┤
│ BUY_YES      │ UP             │ BUY_YES (полн. размер) │
│ BUY_YES      │ DOWN           │ SKIP (вето)  │
│ BUY_NO       │ DOWN           │ BUY_NO (полн. размер)  │
│ BUY_NO       │ UP             │ SKIP (вето)  │
│ SKIP         │ любой          │ SKIP         │
│ любой        │ features_ok=F  │ fallback → ML│
└──────────────┴────────────────┴──────────────┘
"""
from dataclasses import dataclass
from typing import Literal, Optional
import structlog

logger = structlog.get_logger(__name__)

@dataclass(frozen=True)
class CryptoSignalProxy:
    direction: Optional[Literal["UP", "DOWN"]]
    features_ok: bool
    model_version: Optional[int] = None

@dataclass(frozen=True)
class VotingResult:
    action: Literal["BUY_YES", "BUY_NO", "SKIP"]
    reason: str
    confidence: float            # 0.0–1.0, для логирования
    ml_action: str
    lgbm_direction: Optional[str]
    lgbm_features_ok: bool

def combine_votes(
    ml_action: str,
    ml_edge: float,
    crypto_sig: CryptoSignalProxy,
    asset: str,
) -> VotingResult:
    """
    Основная таблица голосования.
    ml_action: "BUY_YES" | "BUY_NO" | "SKIP"
    """
    if not crypto_sig.features_ok:
        # LightGBM-фичи недоступны → fallback на ML-решение без вето
        logger.warning("combined_lgbm_features_invalid_fallback", asset=asset)
        return VotingResult(
            action=ml_action,
            reason="LightGBM features invalid, fallback to ML-only",
            confidence=ml_edge,
            ml_action=ml_action,
            lgbm_direction=None,
            lgbm_features_ok=False,
        )

    if ml_action == "SKIP":
        return VotingResult(
            action="SKIP",
            reason="ML voted SKIP",
            confidence=0.0,
            ml_action=ml_action,
            lgbm_direction=crypto_sig.direction,
            lgbm_features_ok=True,
        )

    # Согласование направлений
    ml_direction = "UP" if ml_action == "BUY_YES" else "DOWN"
    if crypto_sig.direction == ml_direction:
        return VotingResult(
            action=ml_action,
            reason=f"Both models agree: ML={ml_action}, LightGBM={crypto_sig.direction}",
            confidence=min(1.0, ml_edge * 1.2),  # небольшой буст при согласии (только для логов)
            ml_action=ml_action,
            lgbm_direction=crypto_sig.direction,
            lgbm_features_ok=True,
        )
    else:
        return VotingResult(
            action="SKIP",
            reason=f"LightGBM veto: ML={ml_action} but LightGBM={crypto_sig.direction}",
            confidence=0.0,
            ml_action=ml_action,
            lgbm_direction=crypto_sig.direction,
            lgbm_features_ok=True,
        )
