"""
Тесты валидации BacktestConfig.
Запуск: pytest tests/test_backtest_schemas.py -v
"""
import pytest
from pydantic import ValidationError
from polyflip.api.backtest_schemas import BacktestConfig


# ─── min_time_left_min ───────────────────────────────────────────────────────

class TestTimeWindow:
    def test_min_time_zero_allowed(self):
        """ge=0.0 — теперь 0 допустим по схеме."""
        cfg = BacktestConfig(min_time_left_min=0.0, max_time_left_min=10.0)
        assert cfg.min_time_left_min == 0.0

    def test_min_time_negative_rejected(self):
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            BacktestConfig(min_time_left_min=-1.0)

    def test_min_must_be_less_than_max(self):
        """min >= max → невалидный диапазон."""
        with pytest.raises(ValidationError, match="must be < max_time_left_min"):
            BacktestConfig(min_time_left_min=60.0, max_time_left_min=60.0)

    def test_min_equal_to_max_rejected(self):
        with pytest.raises(ValidationError):
            BacktestConfig(min_time_left_min=30.0, max_time_left_min=30.0)

    def test_valid_time_window(self):
        cfg = BacktestConfig(min_time_left_min=5.0, max_time_left_min=60.0)
        assert cfg.min_time_left_min < cfg.max_time_left_min


# ─── favorite_threshold ──────────────────────────────────────────────────────

class TestFavoriteThreshold:
    def test_default_is_065(self):
        cfg = BacktestConfig()
        assert cfg.favorite_threshold == pytest.approx(0.65)

    def test_above_half_valid(self):
        cfg = BacktestConfig(favorite_threshold=0.70)
        assert cfg.favorite_threshold == pytest.approx(0.70)

    def test_below_half_allowed_with_warning(self):
        """ge=0.01 — значения < 0.5 допустимы, но должны генерировать предупреждение."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg = BacktestConfig(favorite_threshold=0.30)
            assert cfg.favorite_threshold == pytest.approx(0.30)
            # Проверяем что предупреждение выдано
            assert any("0.5" in str(warning.message) for warning in w), (
                "Expected UserWarning when favorite_threshold < 0.5"
            )

    def test_zero_rejected(self):
        with pytest.raises(ValidationError, match="greater than or equal to"):
            BacktestConfig(favorite_threshold=0.0)

    def test_above_099_rejected(self):
        with pytest.raises(ValidationError):
            BacktestConfig(favorite_threshold=1.0)

    def test_099_valid(self):
        cfg = BacktestConfig(favorite_threshold=0.99)
        assert cfg.favorite_threshold == pytest.approx(0.99)


# ─── to_runner_config round-trip ─────────────────────────────────────────────

class TestRunnerConfigConversion:
    def test_favorite_threshold_in_runner_config(self):
        cfg = BacktestConfig(favorite_threshold=0.70)
        rc = cfg.to_runner_config()
        assert rc["FAVORITE_THRESHOLD"] == pytest.approx(0.70)

    def test_min_time_zero_in_runner_config(self):
        cfg = BacktestConfig(min_time_left_min=0.0, max_time_left_min=10.0)
        rc = cfg.to_runner_config()
        assert rc["MIN_TIME_LEFT_MIN"] == pytest.approx(0.0)

    def test_trade_on_flip_is_bool(self):
        """Проверяем что TRADE_ON_FLIP не стал строкой после рефакторинга."""
        cfg = BacktestConfig(trade_on_flip=True)
        rc = cfg.to_runner_config()
        assert isinstance(rc["TRADE_ON_FLIP"], bool), (
            f"TRADE_ON_FLIP should be bool, got {type(rc['TRADE_ON_FLIP'])}"
        )
        assert rc["TRADE_ON_FLIP"] is True

    def test_entry_strategy_default(self):
        cfg = BacktestConfig()
        rc = cfg.to_runner_config()
        assert rc["ENTRY_STRATEGY"] == "first"

    def test_entry_strategy_confirmed(self):
        cfg = BacktestConfig(entry_strategy="confirmed")
        rc = cfg.to_runner_config()
        assert rc["ENTRY_STRATEGY"] == "confirmed"
