# tests/crypto/test_feature_importance_guard.py
import pytest
import warnings
from polyflip.crypto.feature_builder import CRYPTO_FEATURE_COLUMNS
from polyflip.crypto.trainer import CRYPTO_FEATURES
from polyflip.crypto.predictor import CryptoFeaturesValidator

def test_final_feature_set_matches_everywhere():
    """Финальная синхронность: все три источника совпадают."""
    validator_fields = set(CryptoFeaturesValidator.model_fields.keys())
    features_set = set(CRYPTO_FEATURES)
    columns_set = set(CRYPTO_FEATURE_COLUMNS)

    assert validator_fields == features_set, \
        f"Validator ≠ CRYPTO_FEATURES: {validator_fields ^ features_set}"
    assert features_set == columns_set, \
        f"CRYPTO_FEATURES ≠ COLUMNS: {features_set ^ columns_set}"
