"""
tests/trading/test_phase_models.py

Тесты: фазовые модели (contested/leaning/decided) корректно
загружаются в кэш и выбираются при инференсе.
"""
import pickle
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass, field
from typing import Any

from polyflip.constants import PRICE_PHASE_BOUNDARIES, get_price_phase
from polyflip.trading.ml_inference import (
    ModelsCache,
    populate_models_cache,
    get_models_cache,
    clear_models_cache,
)
from polyflip.trading.decision_runners import DecisionResult


# ─── Fixtures ────────────────────────────────────────────────────────────────

class DummyModel:
    def __init__(self):
        self.feature_names_in_ = ["mid_price", "spread", "time_left_min"]

    def predict_proba(self, X):
        return [[0.4, 0.6]]

def _make_dummy_model(asset_key: str):
    """Возвращает заглушку модели с predict_proba."""
    return DummyModel()


def _make_model_registry_row(asset: str, version: int = 1):
    row = MagicMock()
    row.asset   = asset
    row.version = version
    row.is_active = True
    row.features  = "mid_price,spread,time_left_min"
    row.ece       = 0.05
    row.model_blob = pickle.dumps(_make_dummy_model(asset))
    return row


# ─── 1. Константы ─────────────────────────────────────────────────────────────

class TestPricePhaseConstants:
    """Единый источник истины: PRICE_PHASE_BOUNDARIES ↔ get_price_phase."""

    def test_all_phase_names_match_boundaries_keys(self):
        """get_price_phase возвращает только ключи из PRICE_PHASE_BOUNDARIES + fallback 'decided'."""
        valid_phases = set(PRICE_PHASE_BOUNDARIES.keys())
        for price in [0.01, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99]:
            phase = get_price_phase(price)
            assert phase in valid_phases, (
                f"get_price_phase({price}) = '{phase}' не входит в PRICE_PHASE_BOUNDARIES: {valid_phases}"
            )

    def test_contested_range(self):
        """mid_price 0.41–0.59 → contested (отклонение 0.00–0.09)."""
        for price in [0.41, 0.45, 0.50, 0.55, 0.59]:
            assert get_price_phase(price) == "contested", f"price={price} должен быть contested"

    def test_leaning_range(self):
        """mid_price 0.25–0.40 и 0.60–0.75 → leaning (отклонение 0.10–0.25)."""
        for price in [0.26, 0.30, 0.39, 0.61, 0.70, 0.74]:
            assert get_price_phase(price) == "leaning", f"price={price} должен быть leaning"

    def test_decided_range(self):
        """mid_price < 0.25 или > 0.75 → decided (отклонение 0.25–0.50)."""
        for price in [0.01, 0.10, 0.24, 0.76, 0.90, 0.99]:
            assert get_price_phase(price) == "decided", f"price={price} должен быть decided"

    def test_boundary_exactness(self):
        """Граничные значения попадают в правильную фазу."""
        assert get_price_phase(0.40) == "leaning"    # dev=0.10 → leaning (lo=0.10)
        assert get_price_phase(0.25) == "decided"    # dev=0.25 → decided (lo=0.25)
        assert get_price_phase(0.251) == "leaning"   # dev=0.249 → leaning (dev < 0.25)

    def test_phase_boundaries_coverage(self):
        """Диапазоны фаз не оставляют пробелов в [0.0, 0.5]."""
        for dev_int in range(0, 50):  # deviate 0.00 до 0.49
            dev = dev_int / 100.0
            price = round(0.5 + dev, 4)
            phase = get_price_phase(price)
            assert phase in PRICE_PHASE_BOUNDARIES, (
                f"dev={dev:.2f} price={price:.4f} → '{phase}' не в PRICE_PHASE_BOUNDARIES"
            )

    def test_trainer_keys_match_phase_names(self):
        """Ключи, которые trainer.py записывает в ModelRegistry, совпадают с get_price_phase."""
        asset = "BTC"
        trainer_keys = {f"{asset}_{p}" for p in PRICE_PHASE_BOUNDARIES}
        
        # Симулируем все возможные выходы get_price_phase для BTC
        inference_keys = set()
        for price_int in range(1, 100):
            price = price_int / 100.0
            phase = get_price_phase(price)
            inference_keys.add(f"{asset}_{phase}")
        
        # Все ключи инференса должны быть подмножеством ключей тренера
        assert inference_keys.issubset(trainer_keys), (
            f"Инференс генерирует ключи {inference_keys - trainer_keys}, "
            f"которых нет в тренере {trainer_keys}"
        )


# ─── 2. populate_models_cache ──────────────────────────────────────────────

class TestPopulateModelsCache:
    """populate_models_cache загружает фазовые модели в кэш."""

    @pytest.fixture(autouse=True)
    def reset_cache(self):
        clear_models_cache()
        yield
        clear_models_cache()

    @pytest.mark.asyncio
    async def test_phase_models_loaded_into_cache(self):
        """Фазовые модели BTC_contested, BTC_leaning, BTC_decided попадают в кэш."""
        assets = ["BTC", "BTC_contested", "BTC_leaning", "BTC_decided", "ETH"]
        rows = [_make_model_registry_row(a, v) for v, a in enumerate(assets, 1)]

        call_count = [0]
        async def side_effect(stmt):
            call_count[0] += 1
            result = MagicMock()
            if call_count[0] == 1:
                result.all.return_value = [
                    MagicMock(asset=r.asset, version=r.version) for r in rows
                ]
            else:
                result.scalars.return_value.all.return_value = rows
            return result

        mock_db = AsyncMock()
        mock_db.execute = side_effect

        await populate_models_cache(mock_db)

        cache = get_models_cache()
        for asset in assets:
            assert asset in cache.models, f"Модель '{asset}' не попала в кэш"
            assert asset in cache.versions

    @pytest.mark.asyncio
    async def test_cache_log_separates_base_and_phase(self, caplog):
        """Лог models_cache_populated содержит base_models и phase_models раздельно."""
        import structlog

        assets = ["BTC", "BTC_contested", "ETH", "ETH_decided"]
        rows = [_make_model_registry_row(a, i) for i, a in enumerate(assets, 1)]

        call_count = [0]
        async def side_effect(stmt):
            call_count[0] += 1
            result = MagicMock()
            if call_count[0] == 1:
                result.all.return_value = [
                    MagicMock(asset=r.asset, version=r.version) for r in rows
                ]
            else:
                result.scalars.return_value.all.return_value = rows
            return result

        mock_db = AsyncMock()
        mock_db.execute = side_effect

        with structlog.testing.capture_logs() as captured:
            await populate_models_cache(mock_db)

        populated_events = [e for e in captured if e.get("event") == "models_cache_populated"]
        assert len(populated_events) == 1, "Лог models_cache_populated должен быть ровно один раз"

        evt = populated_events[0]
        assert "base_models" in evt
        assert "phase_models" in evt
        assert "total" in evt

        # Проверяем разделение
        for key in ["BTC", "ETH"]:
            assert key in evt["base_models"], f"'{key}' должен быть в base_models"
        for key in ["BTC_contested", "ETH_decided"]:
            assert key in evt["phase_models"], f"'{key}' должен быть в phase_models"


# ─── 3. decide_ml_mode — выбор фазовой модели ─────────────────────────────

class TestDecideMlModePhaseSelection:
    """decide_ml_mode выбирает фазовую модель при наличии, базовую при отсутствии."""

    def _make_cache_with_phases(self) -> ModelsCache:
        cache = ModelsCache(models={}, versions={}, features={}, eces={})
        for key in ["BTC", "BTC_contested", "BTC_leaning", "BTC_decided"]:
            cache.models[key]   = _make_dummy_model(key)
            cache.versions[key] = 1
            cache.features[key] = ["mid_price", "spread", "time_left_min"]
            cache.eces[key]     = 0.05
        return cache

    def _make_market(self, asset="BTC"):
        m = MagicMock()
        m.asset         = asset
        m.market_id     = f"{asset}_001"
        m.yes_token_id  = "tok_001"
        m.current_yes_price = 0.55
        m.current_spread    = 0.02
        m.volume_5min       = 1000.0
        m.price_velocity    = 0.0
        return m

    @pytest.mark.asyncio
    async def test_phase_model_selected_when_available(self):
        """При fresh_price=0.55 (contested) выбирается BTC_contested, а не BTC."""
        from polyflip.trading.decision_runners import decide_ml_mode

        cache  = self._make_cache_with_phases()
        market = self._make_market("BTC")

        mock_api = AsyncMock()
        mock_api.get_market_prices = AsyncMock(return_value={
            "current_yes_price": 0.55,  # → phase=contested
            "current_spread": 0.02,
        })

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=MagicMock(
            scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
        ))

        cfg = MagicMock()
        cfg.trade_on_favorite   = True
        cfg.trade_on_flip       = False
        cfg.no_flip_threshold   = 0.35
        cfg.dead_zone           = 0.10
        cfg.auto_dead_zone      = True
        cfg.bet_size            = 5.0
        cfg.max_bet_size_usdc   = 50.0
        cfg.bet_sizing_mode     = "scaled"
        cfg.liquidity_fraction  = 0.05
        cfg.use_crypto_confirm  = False

        import structlog.testing
        with structlog.testing.capture_logs() as logs:
            result = await decide_ml_mode(
                db_session=mock_db,
                api_client=mock_api,
                market=market,
                cfg=cfg,
                raw_settings={},
                models_cache=cache,
                crypto_predictor=None,
            )

        selected_events = [e for e in logs if e.get("event") == "ml_model_selected"]
        assert len(selected_events) == 1

        evt = selected_events[0]
        assert evt["phase"] == "contested"
        assert evt["phase_available"] is True
        assert evt["used_model"] == "BTC_contested"

    @pytest.mark.asyncio
    async def test_base_model_fallback_when_phase_not_in_cache(self):
        """Если фазовая модель отсутствует в кэше — используется базовая BTC."""
        from polyflip.trading.decision_runners import decide_ml_mode

        # Кэш без фазовых моделей
        cache = ModelsCache(models={}, versions={}, features={}, eces={})
        cache.models["BTC"]   = _make_dummy_model("BTC")
        cache.versions["BTC"] = 1
        cache.features["BTC"] = ["mid_price", "spread", "time_left_min"]
        cache.eces["BTC"]     = 0.05

        market = self._make_market("BTC")
        mock_api = AsyncMock()
        mock_api.get_market_prices = AsyncMock(return_value={
            "current_yes_price": 0.90,  # → phase=decided, но BTC_decided не в кэше
            "current_spread": 0.02,
        })

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=MagicMock(
            scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
        ))

        cfg = MagicMock()
        cfg.trade_on_favorite  = True
        cfg.trade_on_flip      = False
        cfg.no_flip_threshold  = 0.35
        cfg.dead_zone          = 0.10
        cfg.auto_dead_zone     = True
        cfg.bet_size           = 5.0
        cfg.max_bet_size_usdc  = 50.0
        cfg.bet_sizing_mode    = "scaled"
        cfg.liquidity_fraction = 0.05
        cfg.use_crypto_confirm = False

        import structlog.testing
        with structlog.testing.capture_logs() as logs:
            result = await decide_ml_mode(
                db_session=mock_db, api_client=mock_api,
                market=market, cfg=cfg,
                raw_settings={}, models_cache=cache,
                crypto_predictor=None,
            )

        evt = next((e for e in logs if e.get("event") == "ml_model_selected"), None)
        assert evt is not None
        assert evt["phase_available"] is False
        assert evt["used_model"] == "BTC"

    @pytest.mark.asyncio
    async def test_used_model_key_in_decision_result(self):
        """DecisionResult.used_model_key содержит ключ выбранной модели."""
        from polyflip.trading.decision_runners import decide_ml_mode

        cache  = self._make_cache_with_phases()
        market = self._make_market("BTC")

        mock_api = AsyncMock()
        mock_api.get_market_prices = AsyncMock(return_value={
            "current_yes_price": 0.55,
            "current_spread": 0.02,
        })

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=MagicMock(
            scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
        ))

        cfg = MagicMock()
        cfg.trade_on_favorite  = True
        cfg.trade_on_flip      = False
        cfg.no_flip_threshold  = 0.35
        cfg.dead_zone          = 0.10
        cfg.auto_dead_zone     = True
        cfg.bet_size           = 5.0
        cfg.max_bet_size_usdc  = 50.0
        cfg.bet_sizing_mode    = "scaled"
        cfg.liquidity_fraction = 0.05
        cfg.use_crypto_confirm = False

        result = await decide_ml_mode(
            db_session=mock_db, api_client=mock_api,
            market=market, cfg=cfg,
            raw_settings={}, models_cache=cache,
            crypto_predictor=None,
        )

        assert hasattr(result, "used_model_key"), (
            "DecisionResult должен иметь поле used_model_key"
        )
        assert result.used_model_key == "BTC_contested", (
            f"Ожидали BTC_contested, получили {result.used_model_key}"
        )


# ─── 4. DecisionResult — поле used_model_key ──────────────────────────────

class TestDecisionResultStructure:
    """DecisionResult имеет поле used_model_key с правильным дефолтом."""

    def test_used_model_key_field_exists_with_default_none(self):
        """used_model_key по умолчанию None — обратная совместимость."""
        result = DecisionResult(
            decision_obj=None,
            p_flip=0.0,
            model_ver=None,
            edge=None,
            skip_reason="test",
        )
        assert hasattr(result, "used_model_key")
        assert result.used_model_key is None

    def test_used_model_key_can_be_set(self):
        """used_model_key корректно принимает значение."""
        result = DecisionResult(
            decision_obj=None,
            p_flip=0.5,
            model_ver=3,
            edge=0.07,
            skip_reason=None,
            used_model_key="BTC_decided",
        )
        assert result.used_model_key == "BTC_decided"


# ─── 5. decide_combined_mode — ml_phase_model в lgbm_metadata ─────────────

class TestCombinedModePhaseMetadata:
    """В COMBINED-режиме lgbm_metadata содержит ml_phase_model."""

    @pytest.mark.asyncio
    async def test_ml_phase_model_in_lgbm_metadata(self):
        """Поле ml_phase_model появляется в JSON lgbm_metadata при COMBINED-режиме."""
        import json
        from polyflip.trading.decision_runners import decide_combined_mode

        cache  = ModelsCache(models={}, versions={}, features={}, eces={})
        for key in ["BTC", "BTC_contested"]:
            cache.models[key]   = _make_dummy_model(key)
            cache.versions[key] = 1
            cache.features[key] = ["mid_price", "spread", "time_left_min"]
            cache.eces[key]     = 0.05

        market = MagicMock()
        market.asset        = "BTC"
        market.market_id    = "BTC_001"
        market.yes_token_id = "tok_001"
        market.current_yes_price = 0.55
        market.current_spread    = 0.02
        market.volume_5min       = 1000.0
        market.price_velocity    = 0.0

        mock_api = AsyncMock()
        mock_api.get_market_prices = AsyncMock(return_value={
            "current_yes_price": 0.55,
            "current_spread": 0.02,
        })

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=MagicMock(
            scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
        ))

        cfg = MagicMock()
        cfg.trade_on_favorite  = True
        cfg.trade_on_flip      = False
        cfg.no_flip_threshold  = 0.35
        cfg.dead_zone          = 0.10
        cfg.auto_dead_zone     = True
        cfg.bet_size           = 5.0
        cfg.max_bet_size_usdc  = 50.0
        cfg.bet_sizing_mode    = "scaled"
        cfg.liquidity_fraction = 0.05
        cfg.use_crypto_confirm = False

        result = await decide_combined_mode(
            db_session=mock_db, api_client=mock_api,
            market=market, cfg=cfg, raw_settings={},
            models_cache=cache, crypto_predictor=None,
            start_time=None, time_left_sec=300.0,
        )

        assert result.lgbm_metadata is not None, "lgbm_metadata не должен быть None"
        meta = json.loads(result.lgbm_metadata)

        assert "ml_phase_model" in meta, (
            f"lgbm_metadata не содержит 'ml_phase_model'. Ключи: {list(meta.keys())}"
        )
        assert meta["ml_phase_model"] in (
            "BTC_contested", "BTC_leaning", "BTC_decided", "BTC", None
        ), f"Неожиданное значение ml_phase_model: {meta['ml_phase_model']}"
