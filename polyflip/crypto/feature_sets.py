"""Feature-set contracts shared by LightGBM training and inference.

The model registry stores the exact feature order used by an artifact.  Keeping
the control set in a small dependency-free module prevents ``trainer`` and
``predictor`` from silently growing different global lists as experiments are
added.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

from polyflip.crypto.feature_builder import CRYPTO_FEATURE_COLUMNS
from polyflip.models.sequence_features import (
    SEQUENCE_CANDLE_FEATURES,
    SEQUENCE_DIRECTION_FEATURES,
)


@dataclass(frozen=True)
class CryptoFeatureSet:
    key: str
    version: str
    features: tuple[str, ...]


# The existing LightGBM schema.  It is intentionally frozen as the A/control
# baseline so experiment metrics remain comparable across training runs.
CONTROL_FEATURES: tuple[str, ...] = (
    "ret_1", "ret_3", "ret_6",
    "vol_6", "vol_24",
    "vol_z_1", "taker_buy_ratio", "cvd_1", "cvd_6",
    "rsi_14", "ema_ratio_9_21", "bb_width", "bb_position",
    "dist_to_high_24", "dist_to_low_24",
    "range_1", "range_avg_24",
    "consec_balance",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "strike_gap_pct", "log_moneyness",
)

# D is a deliberately small, audit-derived schema.  It keeps features that
# were repeatedly useful across recent asset/regime audits and removes the
# sequence fields that were usually zero-gain.  E adds the canonical contract
# context back; F adds only context that is available at the decision boundary.
STABLE_FEATURES: tuple[str, ...] = (
    "ret_1", "ret_3", "ret_6",
    "vol_6", "vol_24", "vol_z_1",
    "taker_buy_ratio", "cvd_1", "cvd_6",
    "rsi_14", "ema_ratio_9_21", "bb_width", "bb_position",
    "dist_to_high_24", "dist_to_low_24",
    "range_1", "range_avg_24", "consec_balance",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
)
CANONICAL_STRIKE_FEATURES: tuple[str, ...] = ("strike_gap_pct", "log_moneyness")
MARKET_CONTEXT_FEATURES: tuple[str, ...] = (
    "pm_momentum_5m",
    "pm_volume_5m",
    "pm_spread_pct",
    "pm_quote_pressure",
    # Top-of-book prices persisted by MarketSnapshot. These are execution
    # context features, not fabricated depth imbalance.
    "pm_best_bid",
    "pm_best_ask",
)

FEATURE_SETS: dict[str, CryptoFeatureSet] = {
    "A": CryptoFeatureSet("A", "A-control-v1", CONTROL_FEATURES),
    "B": CryptoFeatureSet(
        "B",
        "B-direction-sequence-v1",
        (*CONTROL_FEATURES, *SEQUENCE_DIRECTION_FEATURES),
    ),
    "C": CryptoFeatureSet(
        "C",
        "C-candle-structure-v1",
        (*CONTROL_FEATURES, *SEQUENCE_CANDLE_FEATURES),
    ),
    "D": CryptoFeatureSet("D", "D-stable-audit-v1", STABLE_FEATURES),
    "E": CryptoFeatureSet(
        "E", "E-stable-strike-v1",
        (*STABLE_FEATURES, *CANONICAL_STRIKE_FEATURES),
    ),
    "F": CryptoFeatureSet(
        "F", "F-stable-market-context-v1",
        (*STABLE_FEATURES, *CANONICAL_STRIKE_FEATURES, *MARKET_CONTEXT_FEATURES),
    ),
}

EXPERIMENTAL_FEATURES: tuple[str, ...] = tuple(
    dict.fromkeys((
        *SEQUENCE_DIRECTION_FEATURES,
        *SEQUENCE_CANDLE_FEATURES,
        *MARKET_CONTEXT_FEATURES,
    ))
)
ALL_CRYPTO_FEATURES: tuple[str, ...] = tuple(
    dict.fromkeys((*CRYPTO_FEATURE_COLUMNS, *EXPERIMENTAL_FEATURES))
)

_CONTROL_SCHEMA_DRIFT = sorted(
    set(CONTROL_FEATURES) - set(CRYPTO_FEATURE_COLUMNS)
)
if _CONTROL_SCHEMA_DRIFT:
    raise RuntimeError(
        "CONTROL_FEATURES contains columns absent from CRYPTO_FEATURE_COLUMNS: "
        f"{_CONTROL_SCHEMA_DRIFT}"
    )


def normalize_feature_set(value: str | None) -> str:
    """Return a canonical feature-set key for API and trainer inputs."""
    key = (value or "A").strip().upper()
    if key == "AUTO":
        return "A"
    if key not in FEATURE_SETS:
        raise ValueError(f"Unknown LightGBM feature set: {value!r}")
    return key


def get_feature_set(value: str | None = "A") -> CryptoFeatureSet:
    return FEATURE_SETS[normalize_feature_set(value)]


def parse_feature_names(value: str | Iterable[str] | None) -> tuple[str, ...]:
    """Parse ModelRegistry.features without accepting duplicate columns."""
    if value is None:
        return ()
    if isinstance(value, str):
        names = tuple(item.strip() for item in value.split(",") if item.strip())
    else:
        names = tuple(str(item).strip() for item in value if str(item).strip())
    if len(names) != len(set(names)):
        raise ValueError("Model feature schema contains duplicate columns")
    return names


def feature_schema_hash(features: Iterable[str]) -> str:
    payload = "\x1f".join(features).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def validate_feature_schema(features: Iterable[str]) -> tuple[str, ...]:
    names = parse_feature_names(features)
    if not names:
        raise ValueError("Model feature schema is empty")
    unknown = sorted(set(names) - set(ALL_CRYPTO_FEATURES))
    if unknown:
        raise ValueError(f"Unknown crypto model features: {unknown}")
    return names
