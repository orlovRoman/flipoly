"""
polyflip/crypto/volatility.py

Единая логика классификации режимов волатильности (VolatilityRegimePolicy)
для синхронизации между trainer, predictor, backtester и аналитикой.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class VolatilityRegimePolicy:
    low_boundary: float
    high_boundary: float

    def classify(self, vol_trend: float) -> str:
        """
        Классифицирует vol_trend по границам P33/P67:
          - "low_vol":  vol_trend <= low_boundary
          - "mid_vol":  low_boundary < vol_trend <= high_boundary
          - "high_vol": vol_trend > high_boundary
        """
        if vol_trend <= self.low_boundary:
            return "low_vol"
        elif vol_trend <= self.high_boundary:
            return "mid_vol"
        else:
            return "high_vol"
