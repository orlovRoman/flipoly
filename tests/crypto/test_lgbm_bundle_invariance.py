"""Test regime bundle invariance (Item 1.11).

Ensures that training a candidate regime bundle does not alter active predictor
volatility tertile boundaries or predictions of already loaded active models.
"""
import pytest
from tests.crypto.test_lgbm_bundle import test_regime_bundle_versioning_and_prediction_invariance

__all__ = ["test_regime_bundle_versioning_and_prediction_invariance"]
