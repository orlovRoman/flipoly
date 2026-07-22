# tests/crypto/test_no_redundant_features.py

def test_cvd_trend_removed():
    from polyflip.crypto.feature_builder import CRYPTO_FEATURE_COLUMNS
    assert "cvd_trend" not in CRYPTO_FEATURE_COLUMNS, \
        "cvd_trend is a deterministic linear function of cvd_6 - cvd_1, must be removed"

def test_no_deterministic_derived_features():
    """
    Гарантия что в будущем не добавят другие детерминированные функции.
    """
    from polyflip.crypto.feature_builder import CRYPTO_FEATURE_COLUMNS
    derived = {"cvd_trend"}
    found = derived & set(CRYPTO_FEATURE_COLUMNS)
    assert not found, f"Deterministic derived features found: {found}"

def test_feature_count_after_cvd_trim():
    from polyflip.crypto.feature_builder import CRYPTO_FEATURE_COLUMNS
    assert len(CRYPTO_FEATURE_COLUMNS) == 23, f"Expected 23 features, got {len(CRYPTO_FEATURE_COLUMNS)}"
