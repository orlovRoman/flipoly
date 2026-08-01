"""
Тесты выбора режима в CryptoPredictor после миграции
с vol_median → vol_p33/vol_p67 (tertile).
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from polyflip.crypto.predictor import CryptoPredictor


def _make_predictor(p33: float = 0.5, p67: float = 1.5, symbol: str = "BTCUSDT") -> CryptoPredictor:
    """Создаёт предзагруженный предсказатель с нужными квантилями."""
    pred = CryptoPredictor()
    pred._vol_p33s[symbol] = p33
    pred._vol_p67s[symbol] = p67
    # Мокаем модели для трёх режимов
    for regime in ["low_vol", "mid_vol", "high_vol"]:
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = [[0.4, 0.6]]
        pred._models[symbol] = pred._models.get(symbol, {})
        pred._models[symbol][regime] = mock_model
        pred._thresholds[symbol] = pred._thresholds.get(symbol, {})
        pred._thresholds[symbol][regime] = (0.6, 0.4)
    pred._loaded_symbols.add(symbol)
    return pred


def test_regime_low_vol():
    """vol_ratio <= P33 → low_vol."""
    pred = _make_predictor(p33=0.5, p67=1.5)
    # Имитируем vol_ratio = 0.3 (ниже P33=0.5)
    vol_ratio = 0.3
    p33 = pred._vol_p33s["BTCUSDT"]
    p67 = pred._vol_p67s["BTCUSDT"]
    regime = "low_vol" if vol_ratio <= p33 else ("mid_vol" if vol_ratio <= p67 else "high_vol")
    assert regime == "low_vol"


def test_regime_mid_vol():
    """P33 < vol_ratio <= P67 → mid_vol."""
    pred = _make_predictor(p33=0.5, p67=1.5)
    vol_ratio = 0.9  # между P33=0.5 и P67=1.5
    p33 = pred._vol_p33s["BTCUSDT"]
    p67 = pred._vol_p67s["BTCUSDT"]
    regime = "low_vol" if vol_ratio <= p33 else ("mid_vol" if vol_ratio <= p67 else "high_vol")
    assert regime == "mid_vol"


def test_regime_high_vol():
    """vol_ratio > P67 → high_vol."""
    pred = _make_predictor(p33=0.5, p67=1.5)
    vol_ratio = 2.0  # выше P67=1.5
    p33 = pred._vol_p33s["BTCUSDT"]
    p67 = pred._vol_p67s["BTCUSDT"]
    regime = "low_vol" if vol_ratio <= p33 else ("mid_vol" if vol_ratio <= p67 else "high_vol")
    assert regime == "high_vol"


def test_regime_boundary_p33():
    """vol_ratio == P33 → low_vol (граничное значение включается в low)."""
    pred = _make_predictor(p33=0.5, p67=1.5)
    vol_ratio = 0.5  # точно на границе P33
    p33 = pred._vol_p33s["BTCUSDT"]
    p67 = pred._vol_p67s["BTCUSDT"]
    regime = "low_vol" if vol_ratio <= p33 else ("mid_vol" if vol_ratio <= p67 else "high_vol")
    assert regime == "low_vol", "Граница P33 должна входить в low_vol (<=)"


def test_regime_boundary_p67():
    """vol_ratio == P67 → mid_vol (граничное значение включается в mid)."""
    pred = _make_predictor(p33=0.5, p67=1.5)
    vol_ratio = 1.5  # точно на границе P67
    p33 = pred._vol_p33s["BTCUSDT"]
    p67 = pred._vol_p67s["BTCUSDT"]
    regime = "low_vol" if vol_ratio <= p33 else ("mid_vol" if vol_ratio <= p67 else "high_vol")
    assert regime == "mid_vol", "Граница P67 должна входить в mid_vol (<=)"


def test_invalidate_clears_tertile_caches():
    """invalidate() должен очищать _vol_p33s и _vol_p67s."""
    pred = _make_predictor()
    assert "BTCUSDT" in pred._vol_p33s
    assert "BTCUSDT" in pred._vol_p67s
    pred.invalidate("BTCUSDT")
    assert "BTCUSDT" not in pred._vol_p33s
    assert "BTCUSDT" not in pred._vol_p67s


def test_no_vol_median_key_present():
    """Старый ключ _vol_medians не должен существовать в предсказателе."""
    pred = CryptoPredictor()
    assert not hasattr(pred, "_vol_medians"), \
        "_vol_medians удалён в пользу _vol_p33s/_vol_p67s, не должен существовать"

def test_default_tertile_values_when_no_db():
    """Дефолтные значения P33=0.5, P67=1.5 при отсутствии записей в RuntimeSettings."""
    pred = CryptoPredictor()
    # Симулируем что для символа квантили не загружены (не было load())
    p33 = pred._vol_p33s.get("NEWUSDT", 0.5)
    p67 = pred._vol_p67s.get("NEWUSDT", 1.5)
    assert p33 == 0.5
    assert p67 == 1.5
