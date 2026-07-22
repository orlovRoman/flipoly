# tests/crypto/test_risk_guard.py
import pytest
from polyflip.crypto.risk_guard import check_funding_veto, FUNDING_EXTREME_THRESHOLD

class TestFundingVeto:
    def test_normal_funding_no_veto(self):
        v = check_funding_veto(0.0001, "UP")
        assert not v.vetoed
        assert v.stake_multiplier == 1.0

    def test_extreme_positive_with_crowd_vetoed(self):
        v = check_funding_veto(0.0006, "UP")   # positive → crowd UP
        assert v.vetoed
        assert v.stake_multiplier == 0.0

    def test_extreme_positive_against_crowd_allowed(self):
        v = check_funding_veto(0.0006, "DOWN")
        assert not v.vetoed
        assert v.stake_multiplier == 0.75

    def test_extreme_negative_with_crowd_vetoed(self):
        v = check_funding_veto(-0.0006, "DOWN") # negative → crowd DOWN
        assert v.vetoed

    def test_extreme_negative_against_crowd_allowed(self):
        v = check_funding_veto(-0.0006, "UP")
        assert not v.vetoed

    def test_boundary_exactly_at_threshold(self):
        v = check_funding_veto(FUNDING_EXTREME_THRESHOLD, "UP")
        assert v.vetoed   # >= threshold → veto

    def test_none_direction_does_not_crash(self):
        v = check_funding_veto(0.001, "NONE")
        assert not v.vetoed   # NONE не совпадает с crowd → разрешаем

def test_validator_fields_match_crypto_features():
    from polyflip.crypto.predictor import CryptoFeaturesValidator
    from polyflip.crypto.trainer import CRYPTO_FEATURES
    validator_fields = set(CryptoFeaturesValidator.model_fields.keys())
    assert validator_fields == set(CRYPTO_FEATURES), \
        f"Mismatch: {validator_fields.symmetric_difference(set(CRYPTO_FEATURES))}"
