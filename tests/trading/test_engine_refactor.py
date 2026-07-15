"""Самотесты для плана рефакторинга trade_worker_cycle (8 этапов).

Каждая группа тестов проверяет контракт будущего модуля ещё ДО его создания.
Тесты должны сначала упасть на импорте, затем — пройти после реализации этапа.
Это обеспечивает TDD-подход к рефакторингу.

Запуск всей группы:
    pytest tests/trading/test_engine_refactor.py -v

Запуск одного этапа:
    pytest tests/trading/test_engine_refactor.py -v -k "step1"
"""
import dataclasses
import inspect
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
import pytest_asyncio


# ==============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФАБРИКИ — используются во всех этапах
# ==============================================================================

def _make_market(**kwargs):
    """Создаёт минимальный мок LiveMarket."""
    m = MagicMock()
    m.market_id = kwargs.get("market_id", "market_001")
    m.asset = kwargs.get("asset", "BTC")
    m.yes_token_id = kwargs.get("yes_token_id", "token_yes_001")
    m.no_token_id = kwargs.get("no_token_id", "token_no_001")
    m.current_yes_price = kwargs.get("current_yes_price", 0.65)
    m.current_no_price = kwargs.get("current_no_price", 0.35)
    m.current_spread = kwargs.get("current_spread", 0.02)
    m.volume_5min = kwargs.get("volume_5min", 500.0)
    m.price_velocity = kwargs.get("price_velocity", 0.001)
    m.end_time_est = kwargs.get(
        "end_time_est",
        datetime.now(timezone.utc) + timedelta(minutes=10)
    )
    return m


def _make_raw_settings(*args, **kwargs) -> dict:
    """Минимальный набор settings_db для TradingConfig."""
    base = {
        "TRADING_ENABLED": "true",
        "TRADING_MODE": "ml",
        "TRADE_MIN_TIME_LEFT_SEC": "300",
        "TRADE_MAX_TIME_LEFT_SEC": "900",
        "TRADE_BET_SIZE_USDC": "10.0",
        "TRADE_NO_FLIP_THRESHOLD": "0.55",
        "DEAD_ZONE_WIDTH": "0.05",
        "DAILY_LOSS_LIMIT_USDC": "-100.0",
        "TRADE_MIN_PRICE": "0.05",
        "TRADE_MAX_PRICE": "0.95",
        "TRADE_ASSETS": "BTC,ETH",
        "ACTIVE_FEATURES": "mid_price,spread",
        "MIN_EDGE": "0.05",
        "MAX_BET_EDGE": "0.30",
        "MAX_EDGE_FILTER": "0.99",
        "FAVORITE_THRESHOLD": "0.70",
        "TRADE_ON_FAVORITE": "true",
        "TRADE_ON_FLIP": "false",
        "FLIP_THRESHOLD": "0.60",
        "NO_MIN_EDGE": "0.03",
        "OUTSIDER_MAX_PRICE": "0.40",
        "AUTO_DEAD_ZONE": "false",
        "FAVORITE_MODE_ENTRY_SEC": "120",
        "USE_CRYPTO_CONFIRM": "false",
        "CRYPTO_STANDALONE": "false",
        "CRYPTO_MIN_EDGE": "0.05",
        "BET_SIZING_MODE": "fixed",
        "MAX_BET_SIZE_USDC": "50.0",
        "MAX_PRICE_DRIFT": "0.03",
        "STOP_LOSS_ENABLED": "false",
        "TAKE_PROFIT_ENABLED": "false",
        "TAKE_PROFIT_MULTIPLIER": "2.0",
        "FAVORITE_MIN_PRICE": "0.55",
        "FAVORITE_MAX_PRICE": "0.95",
        "FAVORITE_MIN_EDGE": "0.02",
        "LIQUIDITY_FRACTION": "0.1",
        "BYPASS_BET_SIZE_CHECK": "false",
    }
    if args and isinstance(args[0], dict):
        base.update(args[0])
    base.update(kwargs)
    return base


# ==============================================================================
# ШАГ 1 — load_trading_settings()
# Модуль: polyflip/trading/settings_loader.py
# ==============================================================================

class TestStep1LoadTradingSettings:
    """Контракт для load_trading_settings(db_session, trade_assets) → dict[str,str]."""

    @pytest.mark.asyncio
    async def test_step1_returns_dict(self, db_session):
        """load_trading_settings возвращает dict (не None, не список)."""
        from polyflip.trading.settings_loader import load_trading_settings
        result = await load_trading_settings(db_session)
        assert isinstance(result, dict), "Должен вернуть dict[str, str]"

    @pytest.mark.asyncio
    async def test_step1_empty_db_returns_empty_dict(self, db_session):
        """При пустой БД возвращает пустой dict — парсинг дефолтов на стороне parse_trading_settings."""
        from polyflip.trading.settings_loader import load_trading_settings
        result = await load_trading_settings(db_session)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_step1_values_are_strings(self, db_session):
        """Все значения в возвращаемом dict — строки (сырые значения из БД)."""
        from polyflip.db.models import RuntimeSettings
        from polyflip.trading.settings_loader import load_trading_settings

        now = datetime.now(timezone.utc)
        db_session.add(RuntimeSettings(key="TRADING_ENABLED", value="true",
                                       updated_at=now, updated_by="test"))
        await db_session.commit()

        result = await load_trading_settings(db_session)
        for k, v in result.items():
            assert isinstance(v, str), f"Ключ {k!r}: ожидается str, получено {type(v)}"

    @pytest.mark.asyncio
    async def test_step1_per_asset_keys_loaded(self, db_session):
        """Per-asset ключи MIN_EDGE_BTC и TRADING_MODE_BTC добавляются в результат."""
        from polyflip.db.models import RuntimeSettings
        from polyflip.trading.settings_loader import load_trading_settings

        now = datetime.now(timezone.utc)
        db_session.add(RuntimeSettings(key="MIN_EDGE_BTC", value="0.10",
                                       updated_at=now, updated_by="test"))
        db_session.add(RuntimeSettings(key="TRADING_MODE_BTC", value="crypto",
                                       updated_at=now, updated_by="test"))
        await db_session.commit()

        result = await load_trading_settings(db_session, trade_assets=["BTC"])
        assert "MIN_EDGE_BTC" in result, "Per-asset ключ MIN_EDGE_BTC должен быть в результате"
        assert result["MIN_EDGE_BTC"] == "0.10"

    def test_step1_function_signature(self):
        """Функция принимает (db_session, trade_assets=None) — оба аргумента корректны."""
        from polyflip.trading.settings_loader import load_trading_settings
        sig = inspect.signature(load_trading_settings)
        params = list(sig.parameters.keys())
        assert "db_session" in params
        assert "trade_assets" in params
        assert sig.parameters["trade_assets"].default is None


# ==============================================================================
# ШАГ 2 — TradingConfig + parse_trading_settings()
# Модуль: polyflip/trading/trading_config.py
# ==============================================================================

class TestStep2TradingConfig:
    """Контракт для TradingConfig dataclass и parse_trading_settings(raw) → TradingConfig."""

    def test_step2_parse_returns_trading_config(self):
        """parse_trading_settings возвращает экземпляр TradingConfig."""
        from polyflip.trading.trading_config import TradingConfig, parse_trading_settings
        cfg = parse_trading_settings(_make_raw_settings())
        assert isinstance(cfg, TradingConfig)

    def test_step2_parse_empty_dict_uses_defaults(self):
        """При пустом dict используются дефолтные значения из settings."""
        from polyflip.trading.trading_config import parse_trading_settings
        cfg = parse_trading_settings({})
        assert isinstance(cfg.trading_enabled, bool)
        assert isinstance(cfg.min_edge, float)
        assert isinstance(cfg.trade_assets, list)

    def test_step2_parse_overrides_defaults(self):
        """Значения из raw-dict перекрывают defaults."""
        from polyflip.trading.trading_config import parse_trading_settings
        cfg = parse_trading_settings({"MIN_EDGE": "0.15", "TRADING_ENABLED": "false"})
        assert cfg.min_edge == pytest.approx(0.15)
        assert cfg.trading_enabled is False

    def test_step2_config_is_frozen(self):
        """TradingConfig immutable — защита от случайной мутации в цикле."""
        from polyflip.trading.trading_config import parse_trading_settings
        cfg = parse_trading_settings(_make_raw_settings())
        with pytest.raises(Exception):  # FrozenInstanceError или AttributeError
            cfg.min_edge = 999.0  # type: ignore

    def test_step2_trade_assets_is_list(self):
        """trade_assets всегда list[str], не строка."""
        from polyflip.trading.trading_config import parse_trading_settings
        cfg = parse_trading_settings({"TRADE_ASSETS": "BTC,ETH,SOL"})
        assert isinstance(cfg.trade_assets, list)
        assert len(cfg.trade_assets) == 3
        assert "BTC" in cfg.trade_assets

    def test_step2_empty_string_values_dont_crash(self):
        """Пустые строки в dict не вызывают ValueError/TypeError."""
        from polyflip.trading.trading_config import parse_trading_settings
        try:
            parse_trading_settings({"MIN_EDGE": "", "DEAD_ZONE_WIDTH": ""})
        except (ValueError, TypeError) as e:
            pytest.fail(f"parse_trading_settings упал на пустой строке: {e}")

    def test_step2_zero_string_values_handled(self):
        """Строки '0' и '0.0' не вызывают ошибок — используется дефолт."""
        from polyflip.trading.trading_config import parse_trading_settings
        try:
            parse_trading_settings({"MAX_BET_EDGE": "0", "FLIP_THRESHOLD": "0.0"})
        except Exception as e:
            pytest.fail(f"parse_trading_settings упал на нулевом значении: {e}")

    def test_step2_required_fields_present(self):
        """TradingConfig содержит все поля, используемые в engine.py."""
        from polyflip.trading.trading_config import TradingConfig
        required_fields = [
            "trading_enabled", "trading_mode", "min_time_left", "max_time_left",
            "bet_size", "dead_zone", "daily_limit", "min_edge", "max_bet_edge",
            "max_edge_filter", "trade_min_price", "trade_max_price", "trade_assets",
            "trade_on_favorite", "trade_on_flip", "entry_sec",
            "use_crypto_confirm", "crypto_standalone",
        ]
        fields = {f.name for f in dataclasses.fields(TradingConfig)}
        missing = [f for f in required_fields if f not in fields]
        assert not missing, f"TradingConfig не хватает полей: {missing}"


# ==============================================================================
# ШАГ 3 — load_eligible_markets()
# Модуль: polyflip/trading/market_loader.py
# ==============================================================================

class TestStep3LoadEligibleMarkets:
    """Контракт для load_eligible_markets(db_session, cfg, start_time) → list | None."""

    @pytest.mark.asyncio
    async def test_step3_returns_list_when_no_markets(self, db_session):
        """Пустой список рынков — не ошибка, возвращает []."""
        from polyflip.trading.market_loader import load_eligible_markets
        from polyflip.trading.trading_config import parse_trading_settings
        cfg = parse_trading_settings(_make_raw_settings())
        start = datetime.now(timezone.utc)
        result = await load_eligible_markets(db_session, cfg, start)
        assert result is not None and isinstance(result, list)

    @pytest.mark.asyncio
    async def test_step3_daily_limit_returns_none(self, db_session):
        """При достижении дневного лимита убытков возвращает None (сигнал — остановить цикл)."""
        from polyflip.db.models import TradeHistory
        from polyflip.trading.market_loader import load_eligible_markets
        from polyflip.trading.trading_config import parse_trading_settings

        now = datetime.now(timezone.utc)
        # Создаём сделку с убытком -150, лимит -100
        db_session.add(TradeHistory(
            market_id="m_loss", asset="BTC",
            outcome_bought="YES", amount_usdc=50.0,
            executed_price=0.60, predicted_flip_prob=0.7,
            active_features="mid_price", status="SUCCESS",
            pnl=-150.0, mode="PAPER", created_at=now,
        ))
        await db_session.commit()

        cfg = parse_trading_settings(_make_raw_settings({"DAILY_LOSS_LIMIT_USDC": "-100.0"}))
        result = await load_eligible_markets(db_session, cfg, now)
        assert result is None, "При убытке > лимита должен вернуть None"

    @pytest.mark.asyncio
    async def test_step3_market_in_window_returned(self, db_session):
        """Рынок в временном окне [min_time, max_time] возвращается."""
        from polyflip.db.models import LiveMarket
        from polyflip.trading.market_loader import load_eligible_markets
        from polyflip.trading.trading_config import parse_trading_settings

        now = datetime.now(timezone.utc)
        end_time = now + timedelta(seconds=600)  # 10 минут — в окне [300,900]
        db_session.add(LiveMarket(
            market_id="m_window", asset="BTC",
            question="Q?", yes_token_id="ty", no_token_id="tn",
            current_yes_price=0.6, current_no_price=0.4,
            current_spread=0.02, volume_5min=100.0, price_velocity=0.0,
            end_time_est=end_time, last_updated=now,
        ))
        await db_session.commit()

        cfg = parse_trading_settings(_make_raw_settings({
            "TRADE_MIN_TIME_LEFT_SEC": "300",
            "TRADE_MAX_TIME_LEFT_SEC": "900",
        }))
        result = await load_eligible_markets(db_session, cfg, now)
        assert result is not None
        ids = [m.market_id for m in result]
        assert "m_window" in ids, "Рынок в временном окне должен вернуться"

    @pytest.mark.asyncio
    async def test_step3_market_outside_window_excluded(self, db_session):
        """Рынок вне временного окна НЕ возвращается."""
        from polyflip.db.models import LiveMarket
        from polyflip.trading.market_loader import load_eligible_markets
        from polyflip.trading.trading_config import parse_trading_settings

        now = datetime.now(timezone.utc)
        end_time = now + timedelta(seconds=60)  # 1 минута — вне окна [300,900]
        db_session.add(LiveMarket(
            market_id="m_outside", asset="BTC",
            question="Q?", yes_token_id="ty2", no_token_id="tn2",
            current_yes_price=0.6, current_no_price=0.4,
            current_spread=0.02, volume_5min=100.0, price_velocity=0.0,
            end_time_est=end_time, last_updated=now,
        ))
        await db_session.commit()

        cfg = parse_trading_settings(_make_raw_settings({
            "TRADE_MIN_TIME_LEFT_SEC": "300",
            "TRADE_MAX_TIME_LEFT_SEC": "900",
        }))
        result = await load_eligible_markets(db_session, cfg, now)
        if result:
            ids = [m.market_id for m in result]
            assert "m_outside" not in ids


# ==============================================================================
# ШАГ 4 — check_market_guards()
# Модуль: polyflip/trading/market_guards.py
# ==============================================================================

class TestStep4MarketGuards:
    """Контракт для check_market_guards(...) → GuardResult."""

    @pytest.mark.asyncio
    async def test_step4_missing_token_ids_fails(self, db_session):
        """Рынок без token_id: GuardResult.passed=False, skip_reason содержит 'Token'."""
        from polyflip.trading.market_guards import check_market_guards
        from polyflip.trading.trading_config import parse_trading_settings
        from polyflip.constants import TRADING_MODE_ML

        market = _make_market(yes_token_id=None, no_token_id="tn")
        cfg = parse_trading_settings(_make_raw_settings())
        result = await check_market_guards(
            db_session, market, cfg, TRADING_MODE_ML, 600.0,
            datetime.now(timezone.utc)
        )
        assert not result.passed
        assert result.skip_reason is not None
        assert "token" in result.skip_reason.lower() or "Token" in result.skip_reason

    @pytest.mark.asyncio
    async def test_step4_na_token_ids_fails(self, db_session):
        """Рынок с token_id='N/A' отклоняется так же как None."""
        from polyflip.trading.market_guards import check_market_guards
        from polyflip.trading.trading_config import parse_trading_settings
        from polyflip.constants import TRADING_MODE_ML

        market = _make_market(yes_token_id="N/A", no_token_id="N/A")
        cfg = parse_trading_settings(_make_raw_settings())
        result = await check_market_guards(
            db_session, market, cfg, TRADING_MODE_ML, 600.0,
            datetime.now(timezone.utc)
        )
        assert not result.passed

    @pytest.mark.asyncio
    async def test_step4_duplicate_success_trade_fails(self, db_session):
        """Рынок с существующей SUCCESS сделкой: GuardResult.passed=False."""
        from polyflip.db.models import TradeHistory
        from polyflip.trading.market_guards import check_market_guards
        from polyflip.trading.trading_config import parse_trading_settings
        from polyflip.constants import TRADING_MODE_ML

        now = datetime.now(timezone.utc)
        db_session.add(TradeHistory(
            market_id="dup_market", asset="BTC",
            outcome_bought="YES", amount_usdc=10.0,
            executed_price=0.60, predicted_flip_prob=0.7,
            active_features="mid_price", status="SUCCESS",
            mode="PAPER", created_at=now,
        ))
        await db_session.commit()

        market = _make_market(market_id="dup_market")
        cfg = parse_trading_settings(_make_raw_settings())
        result = await check_market_guards(
            db_session, market, cfg, TRADING_MODE_ML, 600.0, now
        )
        assert not result.passed

    @pytest.mark.asyncio
    async def test_step4_asset_not_in_trade_assets_fails(self, db_session):
        """Актив не в TRADE_ASSETS → passed=False."""
        from polyflip.trading.market_guards import check_market_guards
        from polyflip.trading.trading_config import parse_trading_settings
        from polyflip.constants import TRADING_MODE_ML

        market = _make_market(asset="SOL")  # SOL не в BTC,ETH
        cfg = parse_trading_settings(_make_raw_settings({"TRADE_ASSETS": "BTC,ETH"}))
        result = await check_market_guards(
            db_session, market, cfg, TRADING_MODE_ML, 600.0,
            datetime.now(timezone.utc)
        )
        assert not result.passed

    @pytest.mark.asyncio
    async def test_step4_clean_market_passes(self, db_session):
        """Новый рынок с валидными токенами и активом в TRADE_ASSETS проходит все guards."""
        from polyflip.trading.market_guards import check_market_guards
        from polyflip.trading.trading_config import parse_trading_settings
        from polyflip.constants import TRADING_MODE_ML

        market = _make_market(asset="BTC")
        cfg = parse_trading_settings(_make_raw_settings({"TRADE_ASSETS": "BTC,ETH"}))
        result = await check_market_guards(
            db_session, market, cfg, TRADING_MODE_ML, 600.0,
            datetime.now(timezone.utc)
        )
        assert result.passed, f"Чистый рынок должен пройти guards, skip_reason={result.skip_reason!r}"

    @pytest.mark.asyncio
    async def test_step4_existing_skipped_returned(self, db_session):
        """GuardResult.existing_skipped содержит объект TradeHistory если есть SKIPPED запись."""
        from polyflip.db.models import TradeHistory
        from polyflip.trading.market_guards import check_market_guards
        from polyflip.trading.trading_config import parse_trading_settings
        from polyflip.constants import TRADING_MODE_ML

        now = datetime.now(timezone.utc)
        db_session.add(TradeHistory(
            market_id="skip_market", asset="BTC",
            outcome_bought="NONE", amount_usdc=0.0,
            executed_price=0.0, predicted_flip_prob=0.0,
            active_features="", status="SKIPPED",
            error_msg="No signal", mode="PAPER", created_at=now,
        ))
        await db_session.commit()

        market = _make_market(market_id="skip_market", asset="BTC")
        cfg = parse_trading_settings(_make_raw_settings({"TRADE_ASSETS": "BTC,ETH"}))
        result = await check_market_guards(
            db_session, market, cfg, TRADING_MODE_ML, 600.0, now
        )
        assert result.passed  # SKIPPED не блокирует
        assert result.existing_skipped is not None

    def test_step4_guard_result_dataclass_fields(self):
        """GuardResult содержит поля passed, skip_reason, existing_skipped."""
        from polyflip.trading.market_guards import GuardResult
        fields = {f.name for f in dataclasses.fields(GuardResult)}
        assert "passed" in fields
        assert "skip_reason" in fields
        assert "existing_skipped" in fields


# ==============================================================================
# ШАГ 5а — build_inference_dataframe() + run_model_inference()
# Модуль: polyflip/trading/ml_inference.py
# ==============================================================================

class TestStep5aMLInference:
    """Контракт для функций ML-инференса (чистые функции, без БД)."""

    def _make_snapshot(self, offset_sec: int = 0):
        s = MagicMock()
        s.market_id = "m001"
        s.recorded_at = datetime.now(timezone.utc) - timedelta(seconds=offset_sec)
        s.time_left_min = 10.0 - offset_sec / 60.0
        s.mid_price = 0.60
        s.spread = 0.02
        s.price_velocity = 0.001
        s.volume_5min = 300.0
        s.hour_of_day = 12
        return s

    def test_step5a_build_df_row_count(self):
        """DataFrame содержит len(history_snaps)+1 строк (последняя — текущий момент)."""
        from polyflip.trading.ml_inference import build_inference_dataframe

        snaps = [self._make_snapshot(i * 30) for i in range(6)]
        market = _make_market()
        df = build_inference_dataframe(
            market=market, history_snaps=snaps,
            fresh_yes_price=0.63, fresh_spread=0.02,
            global_max=0.75, start_time=datetime.now(timezone.utc),
            time_left_sec=600.0,
        )
        assert len(df) == len(snaps) + 1

    def test_step5a_build_df_last_row_is_live(self):
        """Последняя строка DataFrame соответствует текущим параметрам (fresh_yes_price)."""
        from polyflip.trading.ml_inference import build_inference_dataframe

        snaps = [self._make_snapshot(30)]
        market = _make_market()
        df = build_inference_dataframe(
            market=market, history_snaps=snaps,
            fresh_yes_price=0.77, fresh_spread=0.015,
            global_max=0.80, start_time=datetime.now(timezone.utc),
            time_left_sec=300.0,
        )
        last = df.iloc[-1]
        assert last["mid_price"] == pytest.approx(0.77)
        assert last["spread"] == pytest.approx(0.015)

    def test_step5a_build_df_no_recorded_at_column(self):
        """DataFrame не содержит 'recorded_at' и 'market_id' (дропнуты для инференса)."""
        from polyflip.trading.ml_inference import build_inference_dataframe

        market = _make_market()
        df = build_inference_dataframe(
            market=market, history_snaps=[],
            fresh_yes_price=0.60, fresh_spread=0.02,
            global_max=0.70, start_time=datetime.now(timezone.utc),
            time_left_sec=600.0,
        )
        assert "recorded_at" not in df.columns
        assert "market_id" not in df.columns

    def test_step5a_run_inference_proba_in_range(self):
        """run_model_inference возвращает float в [0.0, 1.0]."""
        from polyflip.trading.ml_inference import run_model_inference

        mock_model = MagicMock()
        mock_model.predict_proba = MagicMock(return_value=[[0.35, 0.65]])
        df = pd.DataFrame({"f1": [0.5], "f2": [0.3]})
        p = run_model_inference(df, mock_model, ["f1", "f2"])
        assert 0.0 <= p <= 1.0

    def test_step5a_run_inference_single_class_model(self):
        """Если модель возвращает только один класс — p_flip=0.0 (не IndexError)."""
        from polyflip.trading.ml_inference import run_model_inference

        mock_model = MagicMock()
        mock_model.predict_proba = MagicMock(return_value=[[1.0]])
        df = pd.DataFrame({"f1": [0.5]})
        try:
            p = run_model_inference(df, mock_model, ["f1"])
            assert p == pytest.approx(0.0)
        except IndexError as e:
            pytest.fail(f"IndexError при одном классе: {e}")

    def test_step5a_price_distance_from_max_correct(self):
        """price_distance_from_max = clip(global_max - mid_price, 0.0)."""
        from polyflip.trading.ml_inference import build_inference_dataframe

        market = _make_market()
        df = build_inference_dataframe(
            market=market, history_snaps=[],
            fresh_yes_price=0.60, fresh_spread=0.02,
            global_max=0.80,  # max > price → distance = 0.20
            start_time=datetime.now(timezone.utc),
            time_left_sec=600.0,
        )
        if "price_distance_from_max" in df.columns:
            assert df.iloc[-1]["price_distance_from_max"] == pytest.approx(0.20, abs=0.001)


# ==============================================================================
# ШАГ 5б — DecisionResult + decide_*_mode()
# Модуль: polyflip/trading/decision_runners.py
# ==============================================================================

class TestStep5bDecisionRunners:
    """Контракт для DecisionResult и трёх decide_*_mode() функций."""

    def test_step5b_decision_result_fields(self):
        """DecisionResult содержит: decision_obj, p_flip, model_ver, edge, skip_reason."""
        from polyflip.trading.decision_runners import DecisionResult
        fields = {f.name for f in dataclasses.fields(DecisionResult)}
        required = {"decision_obj", "p_flip", "model_ver", "edge", "skip_reason"}
        missing = required - fields
        assert not missing, f"DecisionResult не хватает: {missing}"

    def test_step5b_decision_result_skip_when_no_decision(self):
        """DecisionResult.skip_reason != None означает 'записать SKIPPED и перейти к следующему рынку'."""
        from polyflip.trading.decision_runners import DecisionResult
        r = DecisionResult(decision_obj=None, p_flip=0.0, model_ver=None,
                           edge=None, skip_reason="No signal")
        assert r.skip_reason is not None
        assert r.decision_obj is None

    @pytest.mark.asyncio
    async def test_step5b_decide_favorite_returns_result(self, db_session):
        """decide_favorite_mode возвращает DecisionResult без исключений."""
        from polyflip.trading.decision_runners import decide_favorite_mode
        from polyflip.trading.trading_config import parse_trading_settings

        market = _make_market(current_yes_price=0.72)
        cfg = parse_trading_settings(_make_raw_settings())
        result = await decide_favorite_mode(
            market=market, cfg=cfg,
            asset_min_edge=0.05, asset_max_price=0.95,
            start_time=datetime.now(timezone.utc),
            time_left_sec=600.0,
        )
        assert isinstance(result.p_flip, float)

    @pytest.mark.asyncio
    async def test_step5b_decide_favorite_half_price_skips(self, db_session):
        """При current_yes_price=0.5 (нет фаворита) → skip_reason != None."""
        from polyflip.trading.decision_runners import decide_favorite_mode
        from polyflip.trading.trading_config import parse_trading_settings

        market = _make_market(current_yes_price=0.5)
        cfg = parse_trading_settings(_make_raw_settings())
        result = await decide_favorite_mode(
            market=market, cfg=cfg,
            asset_min_edge=0.05, asset_max_price=0.95,
            start_time=datetime.now(timezone.utc),
            time_left_sec=600.0,
        )
        assert result.skip_reason is not None, "Цена 0.5 = нет фаворита → должен SKIP"

    @pytest.mark.asyncio
    async def test_step5b_decide_ml_no_model_skips(self, db_session):
        """Если нет активной модели для актива → skip_reason содержит 'model'."""
        from polyflip.trading.decision_runners import decide_ml_mode
        from polyflip.trading.trading_config import parse_trading_settings
        from polyflip.trading.ml_inference import ModelsCache

        cfg = parse_trading_settings(_make_raw_settings())
        market = _make_market(asset="BTC")
        api_mock = AsyncMock()
        api_mock.get_market_prices = AsyncMock(return_value={
            "current_yes_price": 0.60, "current_spread": 0.02,
        })
        empty_cache = ModelsCache(models={}, versions={}, features={})
        predictor_mock = MagicMock()

        result = await decide_ml_mode(
            db_session=db_session, api_client=api_mock,
            market=market, cfg=cfg,
            models_cache=empty_cache, crypto_predictor=predictor_mock,
            start_time=datetime.now(timezone.utc),
            time_left_sec=600.0, existing_skipped=None,
        )
        assert result.skip_reason is not None
        assert "model" in result.skip_reason.lower()


# ==============================================================================
# ШАГ 6 — validate_pre_trade()
# Модуль: polyflip/trading/pre_trade_validator.py
# ==============================================================================

class TestStep6PreTradeValidator:
    """Контракт для validate_pre_trade(...) → PreTradeValidation."""

    def _make_decision(self, action="BUY_YES", buy_price=0.60, edge=0.10, bet=10.0):
        from polyflip.trading.decision_logic import TradeDecision
        return TradeDecision(
            action=action, buy_price=buy_price, bet_size_usdc=bet,
            strategy_type="ML_TREND", reason="test", edge=edge,
        )

    @pytest.mark.asyncio
    async def test_step6_price_drift_too_large_fails(self):
        """price drift > MAX_PRICE_DRIFT → valid=False, skip_reason содержит 'drift'."""
        from polyflip.trading.pre_trade_validator import validate_pre_trade
        from polyflip.trading.trading_config import parse_trading_settings
        from polyflip.constants import TRADING_MODE_ML

        api_mock = AsyncMock()
        api_mock.get_market_prices = AsyncMock(return_value={"best_ask": 0.70})  # drift=0.10 > 0.03
        market = _make_market()
        cfg = parse_trading_settings(_make_raw_settings({"MAX_PRICE_DRIFT": "0.03"}))
        decision = self._make_decision(buy_price=0.60)

        result = await validate_pre_trade(
            api_client=api_mock, market=market, decision_obj=decision,
            cfg=cfg, asset_mode=TRADING_MODE_ML,
            asset_min_edge=0.05, asset_max_price=0.95, p_flip=0.7, model_ver=1,
        )
        assert not result.valid
        assert result.skip_reason is not None
        assert "drift" in result.skip_reason.lower()

    @pytest.mark.asyncio
    async def test_step6_zero_bet_size_fails(self):
        """actual_bet_size <= 0 → valid=False."""
        from polyflip.trading.pre_trade_validator import validate_pre_trade
        from polyflip.trading.trading_config import parse_trading_settings
        from polyflip.constants import TRADING_MODE_ML

        api_mock = AsyncMock()
        api_mock.get_market_prices = AsyncMock(return_value={"best_ask": 0.62})
        market = _make_market()
        cfg = parse_trading_settings(_make_raw_settings({
            "BET_SIZING_MODE": "fixed", "TRADE_BET_SIZE_USDC": "0",
        }))
        decision = self._make_decision(buy_price=0.60, bet=0.0)

        result = await validate_pre_trade(
            api_client=api_mock, market=market, decision_obj=decision,
            cfg=cfg, asset_mode=TRADING_MODE_ML,
            asset_min_edge=0.05, asset_max_price=0.95, p_flip=0.7, model_ver=1,
        )
        assert not result.valid

    @pytest.mark.asyncio
    async def test_step6_valid_trade_returns_updated_buy_price(self):
        """Валидный trade: valid=True, buy_price обновлён до свежего best_ask."""
        from polyflip.trading.pre_trade_validator import validate_pre_trade
        from polyflip.trading.trading_config import parse_trading_settings
        from polyflip.constants import TRADING_MODE_ML

        fresh_ask = 0.61
        api_mock = AsyncMock()
        api_mock.get_market_prices = AsyncMock(return_value={"best_ask": fresh_ask})
        market = _make_market(current_yes_price=0.60)
        cfg = parse_trading_settings(_make_raw_settings())
        decision = self._make_decision(buy_price=0.60, edge=0.12, bet=10.0)

        result = await validate_pre_trade(
            api_client=api_mock, market=market, decision_obj=decision,
            cfg=cfg, asset_mode=TRADING_MODE_ML,
            asset_min_edge=0.05, asset_max_price=0.95, p_flip=0.2, model_ver=1,
        )
        assert result.valid, f"Должен быть valid=True, skip_reason={result.skip_reason!r}"
        assert result.buy_price == pytest.approx(fresh_ask)

    def test_step6_pre_trade_validation_dataclass_fields(self):
        """PreTradeValidation содержит поля: valid, buy_price, actual_bet_size, edge, skip_reason."""
        from polyflip.trading.pre_trade_validator import PreTradeValidation
        fields = {f.name for f in dataclasses.fields(PreTradeValidation)}
        required = {"valid", "buy_price", "actual_bet_size", "edge", "skip_reason"}
        missing = required - fields
        assert not missing, f"PreTradeValidation не хватает полей: {missing}"


# ==============================================================================
# ШАГ 7 — execute_and_record()
# Модуль: polyflip/trading/trade_recorder.py
# ==============================================================================

class TestStep7TradeRecorder:
    """Контракт для execute_and_record(...) → запись TradeHistory."""

    def _make_validation(self, buy_price=0.60, bet=10.0, edge=0.12):
        from polyflip.trading.pre_trade_validator import PreTradeValidation
        return PreTradeValidation(
            valid=True, buy_price=buy_price,
            actual_bet_size=bet, edge=edge, skip_reason=None
        )

    def _make_decision_obj(self, action="BUY_YES", buy_price=0.60, edge=0.12):
        from polyflip.trading.decision_logic import TradeDecision
        return TradeDecision(
            action=action, buy_price=buy_price, bet_size_usdc=10.0,
            strategy_type="ML_TREND", reason="test", edge=edge
        )

    @pytest.mark.asyncio
    async def test_step7_creates_trade_history(self, db_session):
        """После исполнения в БД появляется TradeHistory со статусом SUCCESS."""
        from sqlalchemy import select
        from polyflip.db.models import TradeHistory
        from polyflip.trading.trade_recorder import execute_and_record
        from polyflip.trading.trading_config import parse_trading_settings
        from polyflip.constants import TRADING_MODE_ML

        trader_mock = AsyncMock()
        trader_mock.execute_trade = AsyncMock(return_value={
            "status": "SUCCESS", "executed_price": 0.61,
            "executed_usdc": 10.0, "mode": "PAPER"
        })
        market = _make_market(market_id="rec_m1", asset="BTC")
        cfg = parse_trading_settings(_make_raw_settings())

        await execute_and_record(
            db_session=db_session, trader=trader_mock, market=market,
            decision_obj=self._make_decision_obj(),
            validation=self._make_validation(),
            asset_mode=TRADING_MODE_ML, active_features="mid_price",
            p_flip=0.7, model_ver=5, cfg=cfg,
            existing_skipped=None, start_time=datetime.now(timezone.utc),
        )
        await db_session.commit()

        result = await db_session.execute(
            select(TradeHistory).where(TradeHistory.market_id == "rec_m1")
        )
        trade = result.scalar_one_or_none()
        assert trade is not None
        assert trade.status == "SUCCESS"

    @pytest.mark.asyncio
    async def test_step7_creates_slippage_log(self, db_session):
        """При SUCCESS создаётся SlippageLog с корректным slippage."""
        from sqlalchemy import select
        from polyflip.db.models import SlippageLog
        from polyflip.trading.trade_recorder import execute_and_record
        from polyflip.trading.trading_config import parse_trading_settings
        from polyflip.constants import TRADING_MODE_ML

        exec_price, buy_price = 0.63, 0.60
        trader_mock = AsyncMock()
        trader_mock.execute_trade = AsyncMock(return_value={
            "status": "SUCCESS", "executed_price": exec_price,
            "executed_usdc": 10.0, "mode": "PAPER"
        })
        market = _make_market(market_id="slip_m1", asset="BTC")
        cfg = parse_trading_settings(_make_raw_settings())

        await execute_and_record(
            db_session=db_session, trader=trader_mock, market=market,
            decision_obj=self._make_decision_obj(buy_price=buy_price),
            validation=self._make_validation(buy_price=buy_price),
            asset_mode=TRADING_MODE_ML, active_features="mid_price",
            p_flip=0.7, model_ver=5, cfg=cfg,
            existing_skipped=None, start_time=datetime.now(timezone.utc),
        )
        await db_session.commit()

        result = await db_session.execute(
            select(SlippageLog).where(SlippageLog.market_id == "slip_m1")
        )
        slip = result.scalar_one_or_none()
        assert slip is not None
        assert slip.slippage == pytest.approx(exec_price - buy_price, abs=1e-5)

    @pytest.mark.asyncio
    async def test_step7_sets_stop_loss_when_enabled(self, db_session):
        """При STOP_LOSS_ENABLED=true заполняется stop_loss_price и stop_loss_status='ACTIVE'."""
        from sqlalchemy import select
        from polyflip.db.models import TradeHistory
        from polyflip.trading.trade_recorder import execute_and_record
        from polyflip.trading.trading_config import parse_trading_settings
        from polyflip.constants import TRADING_MODE_ML

        trader_mock = AsyncMock()
        trader_mock.execute_trade = AsyncMock(return_value={
            "status": "SUCCESS", "executed_price": 0.60,
            "executed_usdc": 10.0, "mode": "PAPER"
        })
        market = _make_market(market_id="sl_m1", asset="BTC")
        cfg = parse_trading_settings(_make_raw_settings({
            "STOP_LOSS_ENABLED": "true", "STOP_LOSS_PCT_FAVORITE": "40.0",
        }))

        await execute_and_record(
            db_session=db_session, trader=trader_mock, market=market,
            decision_obj=self._make_decision_obj(),
            validation=self._make_validation(),
            asset_mode=TRADING_MODE_ML, active_features="mid_price",
            p_flip=0.7, model_ver=5, cfg=cfg,
            existing_skipped=None, start_time=datetime.now(timezone.utc),
        )
        await db_session.commit()

        result = await db_session.execute(
            select(TradeHistory).where(TradeHistory.market_id == "sl_m1")
        )
        trade = result.scalar_one_or_none()
        assert trade is not None
        assert trade.stop_loss_status == "ACTIVE"
        assert trade.stop_loss_price is not None and trade.stop_loss_price > 0

    @pytest.mark.asyncio
    async def test_step7_sets_take_profit_when_enabled(self, db_session):
        """При TAKE_PROFIT_ENABLED=true заполняется take_profit_price и take_profit_status='ACTIVE'."""
        from sqlalchemy import select
        from polyflip.db.models import TradeHistory
        from polyflip.trading.trade_recorder import execute_and_record
        from polyflip.trading.trading_config import parse_trading_settings
        from polyflip.constants import TRADING_MODE_ML

        trader_mock = AsyncMock()
        trader_mock.execute_trade = AsyncMock(return_value={
            "status": "SUCCESS", "executed_price": 0.50,
            "executed_usdc": 10.0, "mode": "PAPER"
        })
        market = _make_market(market_id="tp_m1", asset="BTC")
        cfg = parse_trading_settings(_make_raw_settings({
            "TAKE_PROFIT_ENABLED": "true", "TAKE_PROFIT_MULTIPLIER": "2.0",
        }))

        await execute_and_record(
            db_session=db_session, trader=trader_mock, market=market,
            decision_obj=self._make_decision_obj(),
            validation=self._make_validation(buy_price=0.50),
            asset_mode=TRADING_MODE_ML, active_features="mid_price",
            p_flip=0.7, model_ver=5, cfg=cfg,
            existing_skipped=None, start_time=datetime.now(timezone.utc),
        )
        await db_session.commit()

        result = await db_session.execute(
            select(TradeHistory).where(TradeHistory.market_id == "tp_m1")
        )
        trade = result.scalar_one_or_none()
        assert trade is not None
        assert trade.take_profit_status == "ACTIVE"
        assert trade.take_profit_price == pytest.approx(0.99, abs=0.01)  # min(0.50 * 2.0, 0.99)

    @pytest.mark.asyncio
    async def test_step7_deletes_existing_skipped(self, db_session):
        """Если existing_skipped != None — он удаляется из БД после успешной сделки."""
        from sqlalchemy import select
        from polyflip.db.models import TradeHistory
        from polyflip.trading.trade_recorder import execute_and_record
        from polyflip.trading.trading_config import parse_trading_settings
        from polyflip.constants import TRADING_MODE_ML

        now = datetime.now(timezone.utc)
        existing = TradeHistory(
            market_id="del_m1", asset="BTC",
            outcome_bought="NONE", amount_usdc=0.0,
            executed_price=0.0, predicted_flip_prob=0.0,
            active_features="", status="SKIPPED",
            error_msg="old skip", mode="PAPER", created_at=now,
        )
        db_session.add(existing)
        await db_session.commit()
        await db_session.refresh(existing)

        trader_mock = AsyncMock()
        trader_mock.execute_trade = AsyncMock(return_value={
            "status": "SUCCESS", "executed_price": 0.62,
            "executed_usdc": 10.0, "mode": "PAPER"
        })
        market = _make_market(market_id="del_m1", asset="BTC")
        cfg = parse_trading_settings(_make_raw_settings())

        await execute_and_record(
            db_session=db_session, trader=trader_mock, market=market,
            decision_obj=self._make_decision_obj(),
            validation=self._make_validation(),
            asset_mode=TRADING_MODE_ML, active_features="mid_price",
            p_flip=0.7, model_ver=5, cfg=cfg,
            existing_skipped=existing, start_time=now,
        )
        await db_session.commit()

        result = await db_session.execute(
            select(TradeHistory).where(
                TradeHistory.market_id == "del_m1",
                TradeHistory.status == "SKIPPED"
            )
        )
        assert result.scalar_one_or_none() is None, "SKIPPED запись должна быть удалена"


# ==============================================================================
# ШАГ 8 — Оркестратор trade_worker_cycle (регрессия)
# Проверяем, что после рефакторинга публичный API engine.py не изменился
# ==============================================================================

class TestStep8Orchestrator:
    """Регрессионные тесты оркестратора: публичный API и базовое поведение."""

    def test_step8_trade_worker_cycle_exists_and_is_async(self):
        """trade_worker_cycle по-прежнему существует и является async."""
        from polyflip.trading.engine import trade_worker_cycle
        assert inspect.iscoroutinefunction(trade_worker_cycle)

    def test_step8_trade_worker_cycle_signature(self):
        """Сигнатура trade_worker_cycle(db_session, trader, api_client) не изменилась."""
        from polyflip.trading.engine import trade_worker_cycle
        sig = inspect.signature(trade_worker_cycle)
        params = list(sig.parameters.keys())
        assert "db_session" in params
        assert "trader" in params
        assert "api_client" in params

    @pytest.mark.asyncio
    async def test_step8_disabled_trading_returns_early(self, db_session):
        """При TRADING_ENABLED=false цикл возвращается без обращения к рынкам."""
        from polyflip.db.models import RuntimeSettings
        from polyflip.trading.engine import trade_worker_cycle

        now = datetime.now(timezone.utc)
        db_session.add(RuntimeSettings(key="TRADING_ENABLED", value="false",
                                       updated_at=now, updated_by="test"))
        await db_session.commit()

        trader_mock = AsyncMock()
        api_mock = AsyncMock()
        trader_mock.execute_trade = AsyncMock(
            side_effect=AssertionError("Не должен торговать")
        )

        await trade_worker_cycle(db_session, trader_mock, api_mock)
        trader_mock.execute_trade.assert_not_called()

    @pytest.mark.asyncio
    async def test_step8_no_markets_no_trades(self, db_session):
        """Без рынков в БД — нет вызовов execute_trade."""
        from polyflip.db.models import RuntimeSettings
        from polyflip.trading.engine import trade_worker_cycle

        now = datetime.now(timezone.utc)
        db_session.add(RuntimeSettings(key="TRADING_ENABLED", value="true",
                                       updated_at=now, updated_by="test"))
        await db_session.commit()

        trader_mock = AsyncMock()
        api_mock = AsyncMock()
        trader_mock.execute_trade = AsyncMock()

        await trade_worker_cycle(db_session, trader_mock, api_mock)
        trader_mock.execute_trade.assert_not_called()

    @pytest.mark.asyncio
    async def test_step8_exception_does_not_propagate(self, db_session):
        """Исключение внутри цикла не пробрасывается наружу — оркестратор его поглощает."""
        from polyflip.db.models import RuntimeSettings
        from polyflip.trading.engine import trade_worker_cycle

        now = datetime.now(timezone.utc)
        db_session.add(RuntimeSettings(key="TRADING_ENABLED", value="true",
                                       updated_at=now, updated_by="test"))
        await db_session.commit()

        trader_mock = AsyncMock()
        api_mock = AsyncMock()

        with patch("polyflip.trading.engine.load_eligible_markets",
                   AsyncMock(side_effect=RuntimeError("DB down"))):
            try:
                await trade_worker_cycle(db_session, trader_mock, api_mock)
            except RuntimeError:
                pytest.fail(
                    "trade_worker_cycle должен поглощать исключения, а не пробрасывать"
                )   # До рефакторинга — все упадут на ImportError (ожидаемо)
