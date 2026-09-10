"""Step 17: CT as a separate policy feature (fixed implementation).

Contract:
- Uses the FIXED CT realization and the REAL history of the CHOSEN token.
- Persists version, params, n_obs, history source.
- A known historical example must reproduce (golden test).
- Missing NO-history yields status NO_DATA (never a 'trend' regime).

The trading strategy wires the real CT profile here via ``register_ct``;
until then ``compute_ct`` returns NO_DATA instead of fabricating a regime.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

CT_VERSION = "ct-v1-placeholder"
CT_PARAMS = {"profile": "existing-CT-profile-not-optimized", "window": "as-fixed"}

_Impl = None  # type: ignore


@dataclass(frozen=True)
class CTResult:
    signal: float | None  # e.g. trend score in [-1, 1]
    regime: str           # TREND_UP | TREND_DOWN | RANGE | NO_DATA
    version: str
    params: dict
    n_obs: int
    source: str


def register_ct(fn: Callable[[list[float], dict], CTResult]) -> None:
    global _Impl
    _Impl = fn


def compute_ct(history: list[float] | None, meta: dict | None = None) -> CTResult:
    meta = meta or {}
    if not history:
        return CTResult(None, "NO_DATA", CT_VERSION, dict(CT_PARAMS),
                        0, meta.get("source", "unknown"))
    if _Impl is None:
        # No real CT wired in this scaffold: refuse to invent a regime.
        return CTResult(None, "NO_DATA", CT_VERSION, dict(CT_PARAMS),
                        len(history), meta.get("source", "unknown"))
    return _Impl(list(history), dict(meta))
