"""
polyflip/research/outsider_price_path

Research modules for studying outsider price path trajectory:
- protocol and dataset builders
- trajectory feature extractors (peak, trough, drawdown, rebound)
- mutually exclusive cohorts and orthogonal rebound indicators
- stratified statistical evaluation and bootstrap uncertainty estimation
"""
from __future__ import annotations

__all__ = [
    "dataset",
    "features",
    "cohorts",
    "evaluation",
]
