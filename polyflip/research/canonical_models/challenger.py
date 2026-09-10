"""Step 31: pick ONE challenger from M2-M4 on DEVELOPMENT only.

Primary: mean out-of-fold log loss. Diagnostics (mandatory): Brier,
cheap-range (ask 0.01-0.40) behaviour, temporal stability by week.
Unconvincing gain -> NO_CLEAR_CHALLENGER (never crown a winner just for
being minimally best). The pick is recorded BEFORE the final test is
opened and never replaced afterwards.
"""
from __future__ import annotations

from dataclasses import dataclass

from .forecast_table import logloss


@dataclass(frozen=True)
class ChallengerDecision:
    challenger: str  # M2 | M3 | M4 | NO_CLEAR_CHALLENGER
    oof_logloss: dict
    margin: float
    reason: str


def pick_challenger(oof: dict[str, list[tuple[float, int]]],
                    min_margin: float = 0.002) -> ChallengerDecision:
    """oof: variant -> [(p_up, y)]. Only M2/M3/M4 are eligible."""
    elig = {k: oof[k] for k in ("M2", "M3", "M4") if k in oof}
    if not elig:
        return ChallengerDecision("NO_CLEAR_CHALLENGER", {}, 0.0, "no eligible OOF")
    ll = {k: logloss([p for p, _ in v], [y for _, y in v]) for k, v in elig.items()}
    best = min(ll, key=lambda k: ll[k])
    ordered = sorted(ll.values())
    margin = (ordered[1] - ordered[0]) if len(ordered) > 1 else 0.0
    if margin < min_margin:
        return ChallengerDecision("NO_CLEAR_CHALLENGER", ll, margin,
                                  f"margin {margin:.5f} < {min_margin} -> unconvincing")
    return ChallengerDecision(best, ll, margin, f"{best} wins OOF logloss by {margin:.5f}")
