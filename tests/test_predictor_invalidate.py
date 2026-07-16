import pytest
from polyflip.crypto.predictor import CryptoPredictor


@pytest.fixture(autouse=True)
def clean_predictor_instances():
    """Очищаем _instances до и после каждого теста."""
    original = CryptoPredictor._instances.copy()
    CryptoPredictor._instances.clear()
    yield
    CryptoPredictor._instances.clear()
    CryptoPredictor._instances.extend(original)


def test_invalidate_all_clears_cache():
    p1, p2 = CryptoPredictor(), CryptoPredictor()
    for p in [p1, p2]:
        p._loaded_symbols.add("BTCUSDT")
        p._models["BTCUSDT"]         = {"low_vol": object()}
        p._vol_p33s["BTCUSDT"]       = 1.0
        p._vol_p67s["BTCUSDT"]       = 2.0
        p._thresholds["BTCUSDT"]     = {"low_vol": (0.55, 0.45)}
        p._model_versions["BTCUSDT"] = {"low_vol": 1}

    CryptoPredictor.invalidate_all("BTCUSDT")

    for p in [p1, p2]:
        assert "BTCUSDT" not in p._loaded_symbols
        assert "BTCUSDT" not in p._models
        assert "BTCUSDT" not in p._vol_p33s
        assert "BTCUSDT" not in p._vol_p67s


def test_invalidate_other_symbol_untouched():
    p = CryptoPredictor()
    p._loaded_symbols.update({"BTCUSDT", "ETHUSDT"})
    p._models["ETHUSDT"] = {"low_vol": object()}

    CryptoPredictor.invalidate_all("BTCUSDT")

    assert "ETHUSDT" in p._loaded_symbols
    assert "ETHUSDT" in p._models
