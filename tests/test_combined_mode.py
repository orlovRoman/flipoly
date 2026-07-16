import pytest
import math
from polyflip.trading.combined_voting import CryptoSignalProxy, combine_votes

def test_combined_mode_agreement():
    """ML и LightGBM согласны — берем сделку с бустом confidence"""
    crypto = CryptoSignalProxy(direction="UP", features_ok=True)
    res = combine_votes("BUY_YES", 0.10, crypto, "BTC")
    assert res.action == "BUY_YES"
    assert math.isclose(res.confidence, 0.12, rel_tol=1e-9)  # 0.10 * 1.2

    crypto = CryptoSignalProxy(direction="DOWN", features_ok=True)
    res = combine_votes("BUY_NO", 0.05, crypto, "ETH")
    assert res.action == "BUY_NO"
    assert math.isclose(res.confidence, 0.06, rel_tol=1e-9)

def test_combined_mode_veto():
    """ML и LightGBM не согласны — вето (SKIP)"""
    # ML говорит YES, LGBM говорит DOWN (вето)
    crypto = CryptoSignalProxy(direction="DOWN", features_ok=True)
    res = combine_votes("BUY_YES", 0.15, crypto, "BTC")
    assert res.action == "SKIP"
    assert "veto" in res.reason.lower()

    # ML говорит NO, LGBM говорит UP (вето)
    crypto = CryptoSignalProxy(direction="UP", features_ok=True)
    res = combine_votes("BUY_NO", 0.10, crypto, "ETH")
    assert res.action == "SKIP"
    assert "veto" in res.reason.lower()

    # ML говорит YES, LGBM говорит NONE (нет тренда) -> вето
    crypto = CryptoSignalProxy(direction="NONE", features_ok=True)
    res = combine_votes("BUY_YES", 0.10, crypto, "BTC")
    assert res.action == "SKIP"
    assert "veto" in res.reason.lower()
    # Если LGBM вернул None при features_ok=True (баг предиктора),
    # combine_votes обработает это как вето (None != "UP")
    crypto = CryptoSignalProxy(direction=None, features_ok=True)
    res = combine_votes("BUY_YES", 0.10, crypto, "BTC")
    assert res.action == "SKIP"
    assert "veto" in res.reason.lower()

def test_combined_mode_ml_skip():
    """ML уже SKIP -> оставляем SKIP, независим от LightGBM"""
    crypto = CryptoSignalProxy(direction="UP", features_ok=True)
    res = combine_votes("SKIP", 0.0, crypto, "BTC")
    assert res.action == "SKIP"
    assert "ML (in Combined mode) voted SKIP" in res.reason

def test_combined_mode_fallback():
    """LightGBM features_ok = False -> Fallback на ML"""
    crypto = CryptoSignalProxy(direction=None, features_ok=False)
    
    # ML говорит YES
    res = combine_votes("BUY_YES", 0.15, crypto, "BTC")
    assert res.action == "BUY_YES"
    assert res.confidence == 0.15  # без буста
    assert res.lgbm_features_ok is False
    assert "fallback" in res.reason.lower()

    # ML говорит NO
    res = combine_votes("BUY_NO", 0.10, crypto, "ETH")
    assert res.action == "BUY_NO"
    assert res.confidence == 0.10
    assert res.lgbm_features_ok is False
    assert "fallback" in res.reason.lower()
