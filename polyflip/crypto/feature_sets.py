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


@dataclass(frozen=True)
class CryptoFeatureSet:
    key: str
    version: str
    features: tuple[str, ...]


# The existing LightGBM schema.  B and C are deliberately added in a later
# phase so that this control remains a stable baseline for the experiments.
CONTROL_FEATURES: tuple[str, ...] = (
    "ret_1", "ret_3", "ret_6",
    "vol_6", "vol_24",
    "vol_z_1", "taker_buy_ratio", "cvd_1", "cvd_6",
    "rsi_14", "ema_ratio_9_21", "bb_width", "bb_position",
    "dist_to_high_24", "dist_to_low_24",
    "range_1", "range_avg_24",
    "consec_balance",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
)

FEATURE_SETS: dict[str, CryptoFeatureSet] = {
    "A": CryptoFeatureSet("A", "A-control-v1", CONTROL_FEATURES),
}


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
    unknown = sorted(set(names) - set(CRYPTO_FEATURE_COLUMNS))
    if unknown:
        raise ValueError(f"Unknown crypto model features: {unknown}")
    return names
