from polyflip.models.trainer import DERIVED_FEATURES
from polyflip.models.feature_lags import LAG_FEATURE_NAMES
from polyflip.trading.feature_builder import FEATURE_COLUMNS

def test_feature_columns_contains_all_derived():
    """Все производные фичи должны присутствовать в FEATURE_COLUMNS."""
    base_features = {
        "time_left_min", "mid_price", "spread",
        "volume_5min", "price_velocity", "hour_of_day",
        "day_of_week",
    }
    expected = base_features | set(DERIVED_FEATURES) | set(LAG_FEATURE_NAMES)
    actual = set(FEATURE_COLUMNS)
    diff = expected - actual
    assert not diff, f"Отсутствуют в FEATURE_COLUMNS: {diff}"
