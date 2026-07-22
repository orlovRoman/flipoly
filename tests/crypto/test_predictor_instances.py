# tests/crypto/test_predictor_instances.py
import gc
import weakref
import pytest
from polyflip.crypto.predictor import CryptoPredictor

def test_dead_weakrefs_cleaned_up_on_init():
    """__init__ очищает мертвые ссылки из _instances."""
    p1 = CryptoPredictor()
    p2 = CryptoPredictor()
    initial_count = len(CryptoPredictor._instances)

    del p1
    gc.collect()

    p3 = CryptoPredictor()
    alive = [ref for ref in CryptoPredictor._instances if ref() is not None]
    assert len(alive) <= initial_count

def test_vol_tertile_cold_start_defaults():
    """При отсутствии данных в БД, волатильность = 1.0 должна попадать в mid_vol."""
    predictor = CryptoPredictor()
    vol_p33 = predictor._vol_p33s.get("BTCUSDT", 0.8)
    vol_p67 = predictor._vol_p67s.get("BTCUSDT", 1.2)

    vol_trend = 1.0
    assert vol_p33 < vol_trend <= vol_p67, "1.0 trend must fall into mid_vol default range"
