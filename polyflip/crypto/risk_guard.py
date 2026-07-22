# polyflip/crypto/risk_guard.py
from __future__ import annotations
from dataclasses import dataclass
import structlog

logger = structlog.get_logger(__name__)

FUNDING_EXTREME_THRESHOLD = 0.0005   # 0.05% — аномальная зона (Binance стандарт)
FUNDING_HIGH_THRESHOLD    = 0.0003   # 0.03% — повышенный риск, снижаем ставку


@dataclass(frozen=True)
class RiskVeto:
    vetoed: bool
    reason: str
    stake_multiplier: float = 1.0   # 1.0 = норма, 0.5 = снизить, 0.0 = запрет

    def __bool__(self) -> bool:
        return self.vetoed


def check_funding_veto(funding_rate: float, direction: str) -> RiskVeto:
    """
    Внешнее ВЕТО на основе funding rate.
    direction: "UP" | "DOWN"
    
    Логика:
    - Экстремально положительный фандинг → толпа в лонгах → ожидается шорт-сквиз.
      Ставка UP (с толпой) = ЗАПРЕТ. Ставка DOWN (против толпы) = снизить.
    - Экстремально отрицательный фандинг → толпа в шортах → ожидается лонг-сквиз.
      Ставка DOWN (с толпой) = ЗАПРЕТ. Ставка UP (против толпы) = снизить.
    """
    abs_fr = abs(funding_rate)

    if abs_fr < FUNDING_HIGH_THRESHOLD:
        return RiskVeto(vetoed=False, reason="normal_funding", stake_multiplier=1.0)

    crowd_direction = "UP" if funding_rate > 0 else "DOWN"
    is_with_crowd   = (direction == crowd_direction)
    is_extreme      = abs_fr >= FUNDING_EXTREME_THRESHOLD

    if is_extreme and is_with_crowd:
        logger.warning(
            "funding_veto_applied",
            funding_rate=round(funding_rate, 5),
            direction=direction,
            crowd=crowd_direction,
        )
        return RiskVeto(
            vetoed=True,
            reason=f"extreme_funding={funding_rate:.5f}_crowd_{crowd_direction}",
            stake_multiplier=0.0,
        )

    if is_extreme and not is_with_crowd:
        return RiskVeto(
            vetoed=False,
            reason="against_crowd_extreme",
            stake_multiplier=0.75,
        )

    # Повышенный риск, не экстремальный
    multiplier = 0.85 if is_with_crowd else 1.0
    return RiskVeto(
        vetoed=False,
        reason="elevated_funding",
        stake_multiplier=multiplier,
    )
