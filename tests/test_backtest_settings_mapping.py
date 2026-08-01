"""
Тесты маппинга Live-настроек → BacktestConfig.

Проверяют:
  1. Правильность конвертации процентов → дробей (45 → 0.45)
  2. Корректность BacktestConfig при пограничных значениях
  3. Маппинг TRADING_MODE → strategy_mode
  4. Фикс поля max_bet_edge (не max_edge)
  5. Базовую валидацию Pydantic-схемы
"""
import pytest
from pydantic import ValidationError
from polyflip.api.backtest_schemas import BacktestConfig


# ── Вспомогательная функция: симулирует readConfig() из JS ────────────────
def live_settings_to_backtest_config(live: dict) -> dict:
    """
    Симулирует логику applyLiveSettings() + readConfig() из backtest.js.
    Live-настройки (проценты 0-100) → BacktestConfig (дроби 0-1).
    """
    mode_map = {"ml": "ML", "favorite": "PURE_FAVORITE", "CRYPTO": "ML"}

    def pct_to_frac(value, default):
        """Конвертирует % значение из Live в дробь для Pydantic."""
        v = live.get(value)
        return float(v) / 100 if v is not None else default

    return {
        "strategy_mode": mode_map.get(live.get("TRADING_MODE", "ml"), "ML"),
        "no_flip_threshold":    pct_to_frac("TRADE_NO_FLIP_THRESHOLD", 0.35),
        "flip_threshold":       pct_to_frac("FLIP_THRESHOLD", 0.60),
        "auto_dead_zone_width": pct_to_frac("DEAD_ZONE_WIDTH", 0.10),
        "favorite_threshold":   float(live.get("FAVORITE_THRESHOLD", 0.65)),
        "min_time_left_min":    float(live.get("TRADE_MIN_TIME_LEFT_SEC", 60)) / 60,
        "max_time_left_min":    float(live.get("TRADE_MAX_TIME_LEFT_SEC", 3600)) / 60,
        "trade_bet_size_usdc":  float(live.get("TRADE_BET_SIZE_USDC", 5)),
        "max_bet_size_usdc":    float(live.get("MAX_BET_SIZE_USDC", 50)),
        "min_edge":             float(live.get("MIN_EDGE", -0.05)),
        "max_bet_edge":         float(live.get("MAX_BET_EDGE", 0.50)),  # ← правильное поле
        "slippage_pct":         float(live.get("SLIPPAGE_PCT", 0.005)),
        "trade_on_flip":        live.get("TRADE_ON_FLIP") == "true",
        "bet_sizing_mode":      live.get("BET_SIZING_MODE", "scaled"),
    }


# ── Фикстуры ──────────────────────────────────────────────────────────────
@pytest.fixture
def typical_live_settings():
    """Типичные Live-настройки торгового бота (проценты 0-100)."""
    return {
        "TRADING_MODE": "ml",
        "TRADE_NO_FLIP_THRESHOLD": "45",   # → 0.45
        "FLIP_THRESHOLD": "65",            # → 0.65
        "DEAD_ZONE_WIDTH": "12",           # → 0.12
        "FAVORITE_THRESHOLD": "0.68",
        "TRADE_MIN_TIME_LEFT_SEC": "60",   # → 1.0 мин
        "TRADE_MAX_TIME_LEFT_SEC": "3600", # → 60.0 мин
        "TRADE_BET_SIZE_USDC": "10",
        "MAX_BET_SIZE_USDC": "50",
        "MIN_EDGE": "-0.03",
        "MAX_BET_EDGE": "0.40",
        "SLIPPAGE_PCT": "0.005",
        "TRADE_ON_FLIP": "false",
        "BET_SIZING_MODE": "scaled",
    }


# ── Тест 1: Корректная конвертация % → дроби ─────────────────────────────
class TestPercentToFractionConversion:

    def test_no_flip_threshold_45_becomes_0_45(self, typical_live_settings):
        cfg_dict = live_settings_to_backtest_config(typical_live_settings)
        assert cfg_dict["no_flip_threshold"] == pytest.approx(0.45)

    def test_flip_threshold_65_becomes_0_65(self, typical_live_settings):
        cfg_dict = live_settings_to_backtest_config(typical_live_settings)
        assert cfg_dict["flip_threshold"] == pytest.approx(0.65)

    def test_dead_zone_12_becomes_0_12(self, typical_live_settings):
        cfg_dict = live_settings_to_backtest_config(typical_live_settings)
        assert cfg_dict["auto_dead_zone_width"] == pytest.approx(0.12)

    def test_time_sec_to_min_conversion(self, typical_live_settings):
        cfg_dict = live_settings_to_backtest_config(typical_live_settings)
        assert cfg_dict["min_time_left_min"] == pytest.approx(1.0)
        assert cfg_dict["max_time_left_min"] == pytest.approx(60.0)


# ── Тест 2: Pydantic ValidationError НЕ возникает при правильной конвертации
class TestPydanticValidationPassesAfterConversion:

    def test_no_validation_error_with_converted_values(self, typical_live_settings):
        cfg_dict = live_settings_to_backtest_config(typical_live_settings)
        # Не должно бросать ValidationError
        cfg = BacktestConfig(**cfg_dict)
        assert cfg.no_flip_threshold == pytest.approx(0.45)
        assert cfg.flip_threshold == pytest.approx(0.65)

    def test_validation_error_with_raw_percent_values(self):
        """
        Воспроизводит ОРИГИНАЛЬНЫЙ БАГ: значение 45 (процент) без деления на 100
        должно нарушать Pydantic-ограничение le=1.0.
        """
        with pytest.raises(ValidationError, match="less than or equal to 1"):
            BacktestConfig(no_flip_threshold=45.0)  # ← это и был баг

    def test_validation_error_dead_zone_over_1(self):
        with pytest.raises(ValidationError):
            BacktestConfig(auto_dead_zone_width=10.0)  # 10% без деления


# ── Тест 3: Маппинг TRADING_MODE ─────────────────────────────────────────
class TestTradingModeMapping:

    @pytest.mark.parametrize("live_mode,expected_bt_mode", [
        ("ml",       "ML"),
        ("favorite", "PURE_FAVORITE"),
        ("CRYPTO",   "ML"),          # CRYPTO маппится в ML как наиболее близкий
        ("unknown",  "ML"),          # неизвестный режим → ML (дефолт)
    ])
    def test_mode_mapping(self, typical_live_settings, live_mode, expected_bt_mode):
        typical_live_settings["TRADING_MODE"] = live_mode
        cfg_dict = live_settings_to_backtest_config(typical_live_settings)
        assert cfg_dict["strategy_mode"] == expected_bt_mode

    def test_mode_accepted_by_pydantic(self, typical_live_settings):
        for mode in ("ML", "PURE_FAVORITE"):
            cfg = BacktestConfig(
                **{**live_settings_to_backtest_config(typical_live_settings),
                   "strategy_mode": mode}
            )
            assert cfg.strategy_mode == mode


# ── Тест 4: Поле max_bet_edge (не max_edge) ──────────────────────────────
class TestFieldNameMaxBetEdge:

    def test_max_bet_edge_field_present_in_schema(self):
        """Поле в схеме называется max_bet_edge, не max_edge."""
        cfg = BacktestConfig()
        assert hasattr(cfg, "max_bet_edge"), "BacktestConfig должна иметь поле max_bet_edge"
        assert not hasattr(cfg, "max_edge"), (
            "Поле max_edge не должно существовать — это старое имя из JS"
        )

    def test_max_bet_edge_from_live(self, typical_live_settings):
        cfg_dict = live_settings_to_backtest_config(typical_live_settings)
        assert "max_bet_edge" in cfg_dict
        assert "max_edge" not in cfg_dict
        cfg = BacktestConfig(**cfg_dict)
        assert cfg.max_bet_edge == pytest.approx(0.40)


# ── Тест 5: Пограничные значения ─────────────────────────────────────────
class TestBoundaryValues:

    def test_threshold_0_percent(self):
        live = {"TRADE_NO_FLIP_THRESHOLD": "0", "FLIP_THRESHOLD": "0",
                "DEAD_ZONE_WIDTH": "0"}
        cfg_dict = live_settings_to_backtest_config(live)
        cfg = BacktestConfig(**cfg_dict)
        assert cfg.no_flip_threshold == pytest.approx(0.0)

    def test_threshold_100_percent(self):
        live = {"TRADE_NO_FLIP_THRESHOLD": "100", "FLIP_THRESHOLD": "100"}
        cfg_dict = live_settings_to_backtest_config(live)
        cfg = BacktestConfig(**cfg_dict)
        assert cfg.no_flip_threshold == pytest.approx(1.0)

    def test_dead_zone_50_percent_max(self):
        live = {"DEAD_ZONE_WIDTH": "50"}
        cfg_dict = live_settings_to_backtest_config(live)
        cfg = BacktestConfig(**cfg_dict)
        assert cfg.auto_dead_zone_width == pytest.approx(0.50)

    def test_dead_zone_over_50_percent_fails(self):
        live = {"DEAD_ZONE_WIDTH": "51"}
        cfg_dict = live_settings_to_backtest_config(live)
        with pytest.raises(ValidationError):
            BacktestConfig(**cfg_dict)

    def test_time_window_conflict_raises(self):
        """min_time >= max_time должно бросать ValidationError."""
        with pytest.raises(ValidationError, match="must be <"):
            BacktestConfig(min_time_left_min=60.0, max_time_left_min=60.0)
            
        with pytest.raises(ValidationError, match="must be <"):
            BacktestConfig(min_time_left_min=65.0, max_time_left_min=60.0)

    def test_none_values_use_defaults(self):
        """None в Live → используются дефолтные значения."""
        cfg_dict = live_settings_to_backtest_config({})
        cfg = BacktestConfig(**cfg_dict)
        assert cfg.no_flip_threshold == pytest.approx(0.35)
        assert cfg.flip_threshold    == pytest.approx(0.60)


# ── Тест 6: to_runner_config() полнота ───────────────────────────────────
class TestRunnerConfigCompleteness:

    REQUIRED_RUNNER_KEYS = {
        "MIN_TIME_LEFT_MIN", "MAX_TIME_LEFT_MIN", "STRATEGY_MODE",
        "ENTRY_STRATEGY", "TRADE_ON_FLIP", "NO_FLIP_THRESHOLD",
        "FLIP_THRESHOLD", "FAVORITE_THRESHOLD", "FAVORITE_MIN_PRICE",
        "FAVORITE_MAX_PRICE", "OUTSIDER_MAX_PRICE", "AUTO_DEAD_ZONE_WIDTH",
        "INITIAL_CAPITAL", "BET_SIZING_MODE", "TRADE_BET_SIZE_USDC",
        "MAX_BET_SIZE_USDC", "MIN_EDGE", "MAX_BET_EDGE", "SLIPPAGE_PCT",
    }

    def test_runner_config_has_all_required_keys(self, typical_live_settings):
        cfg = BacktestConfig(**live_settings_to_backtest_config(typical_live_settings))
        runner_cfg = cfg.to_runner_config()
        missing = self.REQUIRED_RUNNER_KEYS - runner_cfg.keys()
        assert not missing, f"to_runner_config() не содержит ключи: {missing}"

    def test_runner_config_thresholds_are_fractions(self, typical_live_settings):
        cfg = BacktestConfig(**live_settings_to_backtest_config(typical_live_settings))
        runner_cfg = cfg.to_runner_config()
        assert 0.0 <= runner_cfg["NO_FLIP_THRESHOLD"] <= 1.0
        assert 0.0 <= runner_cfg["FLIP_THRESHOLD"] <= 1.0
        assert 0.0 <= runner_cfg["AUTO_DEAD_ZONE_WIDTH"] <= 1.0
