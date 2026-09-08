"""
polyflip/models/outsider_feature_sets.py

Explicit, immutable feature contracts for Outsider LogReg models.
Guarantees that selecting a compact feature set (e.g. Model A) never
automatically expands to 16/26 features behind the scenes.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

OUTSIDER_BUILDER_VERSION = "2.0.0"


@dataclass(frozen=True)
class OutsiderFeatureSet:
    key: str
    version: str
    features: tuple[str, ...]
    description: str

    @property
    def schema_hash(self) -> str:
        return feature_schema_hash(self.features)


def feature_schema_hash(features: Sequence[str]) -> str:
    """Compute deterministic sha256 hash of ordered feature names."""
    canonical = ",".join(features)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# Model A (v1 backward-compatible compact contract)
MODEL_A_FEATURES: tuple[str, ...] = (
    "mid_price",
    "time_left_min",
    "spread",
)

# Model A1: Explicit transformed features (Item 2.6)
MODEL_A1_FEATURES: tuple[str, ...] = (
    "logit_mid_price",
    "log_time_left",
    "candidate_spread",
    "logit_price_x_log_time",
)

# Model B1: Model A1 + normalized strike distance + 30s/120s directional momentum (Items 2.2, 2.3, 2.7)
MODEL_B1_FEATURES: tuple[str, ...] = (
    "logit_mid_price",
    "log_time_left",
    "candidate_spread",
    "logit_price_x_log_time",
    "z_outsider",
    "ret_outsider_30s",
    "ret_outsider_120s",
)

# Legacy LogReg feature set
LEGACY_LOGREG_FEATURES: tuple[str, ...] = (
    "time_left_min",
    "mid_price",
    "spread",
    "volume_5min",
    "price_velocity",
    "hour_of_day",
    "day_of_week",
    "price_distance_from_max",
    "price_momentum",
    "spread_trend",
    "volume_trend",
)

OUTSIDER_FEATURE_SETS: dict[str, OutsiderFeatureSet] = {
    "MODEL_A": OutsiderFeatureSet(
        key="MODEL_A",
        version="model-a-compact-v1",
        features=MODEL_A_FEATURES,
        description="Compact outsider win probability based on mid_price, time_left_min, spread",
    ),
    "MODEL_A1": OutsiderFeatureSet(
        key="MODEL_A1",
        version="model-a1-compact-v2",
        features=MODEL_A1_FEATURES,
        description="Model A1: logit(mid_price), log1p(time_left_min), candidate_spread, interaction",
    ),
    "MODEL_B": OutsiderFeatureSet(
        key="MODEL_B",
        version="model-b1-momentum-v1",
        features=MODEL_B1_FEATURES,
        description="Model B: Model A1 + z_outsider strike distance + 30s/120s directional momentum",
    ),
    "MODEL_B1": OutsiderFeatureSet(
        key="MODEL_B1",
        version="model-b1-momentum-v1",
        features=MODEL_B1_FEATURES,
        description="Model B1: Model A1 + z_outsider strike distance + 30s/120s directional momentum",
    ),
    "LEGACY": OutsiderFeatureSet(
        key="LEGACY",
        version="legacy-logreg-v1",
        features=LEGACY_LOGREG_FEATURES,
        description="Legacy expanded LogReg features with derived lags",
    ),
}


def get_outsider_feature_set(key_or_name: str) -> OutsiderFeatureSet:
    normalized = str(key_or_name).strip().upper()
    if normalized in OUTSIDER_FEATURE_SETS:
        return OUTSIDER_FEATURE_SETS[normalized]
    # Fallback to Model A if unknown
    return OUTSIDER_FEATURE_SETS["MODEL_A"]

