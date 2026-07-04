def test_invalidate_all_clears_cache():
    from polyflip.crypto.predictor import CryptoPredictor
    p1, p2 = CryptoPredictor(), CryptoPredictor()

    for p in [p1, p2]:
        p._loaded_symbols.add("BTCUSDT")
        p._models["BTCUSDT"]         = {"low_vol": object()}
        p._vol_medians["BTCUSDT"]    = 1.0
        p._thresholds["BTCUSDT"]     = {"low_vol": (0.55, 0.45)}
        p._model_versions["BTCUSDT"] = {"low_vol": 1}

    CryptoPredictor.invalidate_all("BTCUSDT")

    for p in [p1, p2]:
        assert "BTCUSDT" not in p._loaded_symbols
        assert "BTCUSDT" not in p._models
        assert "BTCUSDT" not in p._vol_medians


def test_invalidate_other_symbol_untouched():
    from polyflip.crypto.predictor import CryptoPredictor
    p = CryptoPredictor()
    p._loaded_symbols.update({"BTCUSDT", "ETHUSDT"})
    p._models["ETHUSDT"] = {"low_vol": object()}

    CryptoPredictor.invalidate_all("BTCUSDT")

    # ETHUSDT должен остаться нетронутым
    assert "ETHUSDT" in p._loaded_symbols
    assert "ETHUSDT" in p._models
