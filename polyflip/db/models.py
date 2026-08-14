from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    LargeBinary,
    Index,
    UniqueConstraint,
    ForeignKey,
    CheckConstraint,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import JSON
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class CryptoCandle(Base):
    """OHLCV-????? ?? Binance, ???????????? ????????????????? ??????."""

    __tablename__ = "crypto_candles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(32), nullable=False)
    interval = Column(String(8), nullable=False)
    open_time = Column(DateTime(timezone=True), nullable=False)
    close_time = Column(DateTime(timezone=True), nullable=True)
    is_closed = Column(Boolean, nullable=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    taker_buy_volume = Column(Float, nullable=True)
    source = Column(String(16), nullable=False, default="binance")

    __table_args__ = (
        UniqueConstraint(
            "symbol", "interval", "open_time",
            name="uix_crypto_candle",
        ),
        Index("idx_crypto_candles_symbol_interval", "symbol", "interval"),
        Index("idx_crypto_candles_open_time", "open_time"),
    )


class MarketSnapshot(Base):
    """
    Таблица для хранения снапшотов цен с Polymarket и Binance.
    """

    __tablename__ = "market_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset = Column(String(32), nullable=False)
    polymarket_id = Column(String(128), nullable=True)

    # Polymarket цены (Up и Down токены)
    poly_up_best_bid = Column(Float, nullable=True)
    poly_up_best_ask = Column(Float, nullable=True)
    poly_up_mid = Column(Float, nullable=True)
    poly_down_best_bid = Column(Float, nullable=True)
    poly_down_best_ask = Column(Float, nullable=True)
    poly_down_mid = Column(Float, nullable=True)

    # Binance цены
    binance_spot_bid = Column(Float, nullable=True)
    binance_spot_ask = Column(Float, nullable=True)
    binance_spot_mid = Column(Float, nullable=True)
    binance_perp_bid = Column(Float, nullable=True)
    binance_perp_ask = Column(Float, nullable=True)
    binance_perp_mid = Column(Float, nullable=True)

    # Относительное время до закрытия
    time_left_seconds = Column(Integer, nullable=True)

    # Временные метки
    market_timestamp = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    received_timestamp = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("idx_snapshots_asset_time", "asset", "market_timestamp"),
        Index("idx_snapshots_market_timestamp", "market_timestamp"),
    )


class TradeHistory(Base):
    """
    Таблица для хранения истории совершенных сделок.
    """

    __tablename__ = "trade_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name = Column(String(64), nullable=False)
    asset = Column(String(32), nullable=False)
    side = Column(String(8), nullable=False)  # BUY, SELL
    direction = Column(String(8), nullable=True)  # UP, DOWN
    price = Column(Float, nullable=False)
    size = Column(Float, nullable=False)
    cost = Column(Float, nullable=False)
    fee = Column(Float, nullable=True, default=0.0)
    pnl = Column(Float, nullable=True)
    tx_hash = Column(String(128), nullable=True)
    order_id = Column(String(128), nullable=True)
    status = Column(
        String(32), nullable=False, default="FILLED"
    )  # FILLED, CANCELLED, FAILED, PARTIAL
    market_slug = Column(String(256), nullable=True)
    trade_role = Column(
        String(16), nullable=True, server_default="MAKER"
    )  # MAKER, TAKER
    closed_at = Column(DateTime(timezone=True), nullable=True)
    close_reason = Column(String(64), nullable=True)
    close_price = Column(Float, nullable=True)
    model_version = Column(Integer, nullable=True)
    model_registry_id = Column(
        Integer,
        ForeignKey("model_registry.id", ondelete="SET NULL"),
        nullable=True,
    )
    is_reconstructed = Column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    timestamp = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("idx_trades_strategy_asset", "strategy_name", "asset"),
        Index("idx_trades_timestamp", "timestamp"),
        Index("idx_trades_asset_model_version", "asset", "model_version"),
        Index("idx_trades_model_registry_id", "model_registry_id"),
    )


class CollectorStatus(Base):
    """
    Таблица для мониторинга здоровья и задержек компонентов сбора данных.
    """

    __tablename__ = "collector_status"

    id = Column(Integer, primary_key=True, autoincrement=True)
    service_name = Column(
        String(64), nullable=False
    )  # polymarket_ws, binance_ws, etc.
    status = Column(
        String(32), nullable=False
    )  # HEALTHY, DEGRADED, DOWN, RECONNECTING
    latency_ms = Column(Float, nullable=True)
    last_event_timestamp = Column(DateTime(timezone=True), nullable=True)
    error_message = Column(String(512), nullable=True)
    details = Column(
        JSON().with_variant(JSONB, "postgresql"), nullable=True
    )  # Дополнительные метрики (например, reconnect_count)
    timestamp = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("idx_status_service_time", "service_name", "timestamp"),
    )


class LiveMarket(Base):
    """
    Таблица текущих активных 15-минутных рынков с Polymarket.
    """

    __tablename__ = "live_markets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    condition_id = Column(String(128), unique=True, nullable=False)
    slug = Column(String(256), nullable=False)
    asset = Column(String(32), nullable=False)
    question = Column(String(512), nullable=True)
    end_date = Column(DateTime(timezone=True), nullable=False)
    clob_token_id = Column(String(256), nullable=True)
    clob_token_down_id = Column(String(256), nullable=True)
    status = Column(
        String(32), default="ACTIVE"
    )  # ACTIVE, RESOLVED, EXPIRED, CANCELLED

    # Текущие цены для быстрого доступа
    current_price_up = Column(Float, nullable=True)
    current_price_down = Column(Float, nullable=True)
    rewards_min_size = Column(Float, nullable=True)
    rewards_max_spread = Column(Float, nullable=True)

    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        Index("idx_live_markets_asset_status", "asset", "status"),
        Index("idx_live_markets_end_date", "end_date"),
    )


class OpenPosition(Base):
    """
    Таблица для отслеживания текущих открытых позиций.
    """

    __tablename__ = "open_positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market_id = Column(
        Integer,
        ForeignKey("live_markets.id", ondelete="CASCADE"),
        nullable=False,
    )
    condition_id = Column(String(128), nullable=False)
    asset = Column(String(32), nullable=False)
    direction = Column(String(8), nullable=False)  # UP, DOWN
    entry_price = Column(Float, nullable=False)
    current_price = Column(Float, nullable=True)
    unrealized_pnl = Column(Float, nullable=True)
    size = Column(Float, nullable=False)
    cost = Column(Float, nullable=False)
    peak_price = Column(Float, nullable=True)
    strategy_name = Column(String(64), nullable=False)
    status = Column(
        String(32), default="OPEN"
    )  # OPEN, CLOSING, CLOSED, EXPIRED
    order_id = Column(String(128), nullable=True)
    opened_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        Index("idx_open_positions_market_id", "market_id"),
        Index("idx_open_positions_condition_id", "condition_id"),
        Index("idx_open_positions_asset_status", "asset", "status"),
    )


class Order(Base):
    """
    Таблица для отслеживания жизненного цикла ордеров (от создания до исполнения/отмены).
    """

    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String(128), unique=True, nullable=False)
    market_id = Column(
        Integer,
        ForeignKey("live_markets.id", ondelete="CASCADE"),
        nullable=False,
    )
    strategy_name = Column(String(64), nullable=False)
    side = Column(String(8), nullable=False)  # BUY, SELL
    direction = Column(String(8), nullable=False)  # UP, DOWN
    price = Column(Float, nullable=False)
    size = Column(Float, nullable=False)
    filled_size = Column(Float, default=0.0)
    remaining_size = Column(Float, nullable=False)
    status = Column(
        String(32), default="PENDING"
    )  # PENDING, OPEN, FILLED, CANCELLED, REJECTED, EXPIRED
    order_type = Column(String(16), default="GTC")  # GTC, FOK, GTD
    trade_role = Column(
        String(16), nullable=True, server_default="MAKER"
    )  # MAKER, TAKER
    raw_response = Column(
        JSON().with_variant(JSONB, "postgresql"), nullable=True
    )
    error_message = Column(String(512), nullable=True)
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        Index("idx_orders_market_id", "market_id"),
        Index(
            "idx_orders_status",
            "status",
            postgresql_where=text("status IN ('QUEUED', 'RUNNING')"),
            sqlite_where=text("status IN ('QUEUED', 'RUNNING')"),
        ),
        Index("idx_orders_created_at", "created_at"),
    )


class ModelRegistry(Base):
    """
    Таблица для хранения версий обученных моделей ML.
    """

    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset = Column(String(32), nullable=False)
    version = Column(Integer, nullable=False)
    model_type = Column(
        String(64), nullable=False
    )  # LogisticRegression, LightGBM, etc.
    features = Column(
        String(256), nullable=False
    )  # comma-separated feature list (e.g. FS_D0)
    decision_threshold = Column(Float, nullable=False, default=0.55)
    decision_threshold_down = Column(Float, nullable=True, default=0.45)
    accuracy = Column(Float, nullable=True)
    backtest_pnl = Column(Float, nullable=True)
    backtest_trades = Column(Integer, nullable=True)
    is_active = Column(Boolean, default=False)
    model_weights = Column(LargeBinary, nullable=True)
    model_metadata = Column(
        JSON().with_variant(JSONB, "postgresql"), nullable=True
    )  # гиперпараметры, train/test периоды
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "asset", "version", name="uix_model_registry_asset_version"
        ),
        Index("idx_model_registry_asset_active", "asset", "is_active"),
    )


class ExecutionSession(Base):
    """
    Таблица для хранения сессий исполнения торговли.
    """

    __tablename__ = "execution_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), unique=True, nullable=False)
    asset = Column(String(32), nullable=False)
    mode = Column(String(16), nullable=False, default="PAPER")  # PAPER, LIVE
    status = Column(
        String(16), nullable=False, default="RUNNING"
    )  # RUNNING, STOPPED, FAILED
    strategy_name = Column(String(64), nullable=False)
    started_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    stopped_at = Column(DateTime(timezone=True), nullable=True)
    initial_balance = Column(Float, nullable=False, default=1000.0)
    current_balance = Column(Float, nullable=False, default=1000.0)
    total_pnl = Column(Float, nullable=False, default=0.0)
    total_trades = Column(Integer, nullable=False, default=0)
    winning_trades = Column(Integer, nullable=False, default=0)
    losing_trades = Column(Integer, nullable=False, default=0)
    config = Column(
        JSON().with_variant(JSONB, "postgresql"), nullable=True
    )  # Runtime settings snapshot
    error_message = Column(String(512), nullable=True)

    __table_args__ = (
        Index("idx_exec_sessions_asset_status", "asset", "status"),
        Index("idx_exec_sessions_started_at", "started_at"),
    )


class ExecutionTrade(Base):
    """
    Таблица для сделок, совершенных в рамках сессии исполнения (Paper/Live).
    """

    __tablename__ = "execution_trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(
        String(64),
        ForeignKey("execution_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
    )
    trade_id = Column(String(64), unique=True, nullable=False)
    condition_id = Column(String(128), nullable=False)
    market_slug = Column(String(256), nullable=False)
    asset = Column(String(32), nullable=False)
    direction = Column(String(8), nullable=False)  # UP, DOWN
    side = Column(String(8), nullable=False)  # BUY, SELL
    trade_role = Column(
        String(16), nullable=True, server_default="MAKER"
    )  # MAKER, TAKER
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    size = Column(Float, nullable=False)
    cost = Column(Float, nullable=False)
    pnl = Column(Float, nullable=True)
    pnl_percent = Column(Float, nullable=True)
    opened_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    closed_at = Column(DateTime(timezone=True), nullable=True)
    close_reason = Column(
        String(64), nullable=True
    )  # RESOLUTION, STOP_LOSS, TAKE_PROFIT, MANUAL
    model_version = Column(Integer, nullable=True)
    predicted_probability = Column(Float, nullable=True)
    features_snapshot = Column(
        JSON().with_variant(JSONB, "postgresql"), nullable=True
    )
    raw_order_id = Column(String(128), nullable=True)
    market_end_date = Column(DateTime(timezone=True), nullable=True)
    clob_token_id = Column(String(256), nullable=True)
    clob_token_down_id = Column(String(256), nullable=True)
    entry_binance_price = Column(Float, nullable=True)
    entry_strike_price = Column(Float, nullable=True)
    entry_time_left_sec = Column(Integer, nullable=True)
    exit_binance_price = Column(Float, nullable=True)
    exit_resolved_direction = Column(String(8), nullable=True)
    is_won = Column(Boolean, nullable=True)
    status = Column(String(16), nullable=False, default="OPEN")  # OPEN, CLOSED

    __table_args__ = (
        Index("idx_exec_trades_session_id", "session_id"),
        Index("idx_exec_trades_asset_status", "asset", "status"),
        Index("idx_exec_trades_opened_at", "opened_at"),
    )


class RuntimeSettings(Base):
    """
    Таблица для хранения глобальных настроек торгового бота.
    """

    __tablename__ = "runtime_settings"

    key = Column(String(64), primary_key=True)
    value = Column(String(256), nullable=False)
    description = Column(String(512), nullable=True)
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class DecisionFunnelLog(Base):
    """
    Таблица для детального аудита каждого шага воронки принятия решений (Decision Funnel).
    Фиксирует причину отказа (REJECT) или пропуска (PASS) для каждого тика/рынка.
    """

    __tablename__ = "decision_funnel_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    asset = Column(String(32), nullable=False)
    condition_id = Column(String(128), nullable=True)
    step_name = Column(
        String(64), nullable=False
    )  # e.g., 'TIME_WINDOW', 'SPREAD_CHECK', 'ML_INFERENCE', 'ORDER_EXECUTION'
    decision = Column(String(16), nullable=False)  # 'PASS', 'REJECT', 'EXECUTE'
    reason = Column(
        String(256), nullable=True
    )  # e.g., 'time_left_seconds 895 > 840', 'p_up 0.52 < 0.55'
    details = Column(
        JSON().with_variant(JSONB, "postgresql"), nullable=True
    )  # Полный снапшот фичей/метрик на момент шага

    __table_args__ = (
        Index("idx_funnel_asset_timestamp", "asset", "timestamp"),
        Index("idx_funnel_decision_step", "decision", "step_name"),
    )


class Binance15mBar(Base):
    """
    Таблица для хранения исторических 15-минутных свечей (OHLCV) с Binance.
    Служит источником данных для обучения и бэктестирования моделей LightGBM.
    """

    __tablename__ = "binance_15m_bars"

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset = Column(
        String(32), nullable=False
    )  # 'BTCUSDT', 'ETHUSDT', 'SOLUSDT'
    open_time = Column(
        DateTime(timezone=True), nullable=False
    )  # Время открытия 15м бара
    close_time = Column(
        DateTime(timezone=True), nullable=False
    )  # Время закрытия 15м бара

    # OHLCV данные
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)

    # Дополнительные метрики из Binance Klines
    quote_asset_volume = Column(Float, nullable=True)
    number_of_trades = Column(Integer, nullable=True)
    taker_buy_base_volume = Column(Float, nullable=True)
    taker_buy_quote_volume = Column(Float, nullable=True)

    # Метаданные
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "asset", "open_time", name="uix_binance_15m_bars_asset_time"
        ),
        Index("idx_binance_15m_bars_asset_close", "asset", "close_time"),
    )


class BinanceFundingRate(Base):
    """
    Таблица для хранения исторических ставок финансирования (Funding Rate) с Binance Futures.
    Фандинг начисляется каждые 8 часов (00:00, 08:00, 16:00 UTC).
    """

    __tablename__ = "binance_funding_rates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset = Column(
        String(32), nullable=False
    )  # 'BTCUSDT', 'ETHUSDT', 'SOLUSDT'
    funding_time = Column(
        DateTime(timezone=True), nullable=False
    )  # Время фиксации ставки (UTC)
    funding_rate = Column(Float, nullable=False)  # Значение ставки (например, 0.0001 = 0.01%)
    mark_price = Column(Float, nullable=True)  # Цена маркировки в момент фиксации

    # Метаданные
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "asset", "funding_time", name="uix_binance_funding_rates_asset_time"
        ),
        Index("idx_binance_funding_rates_asset_time", "asset", "funding_time"),
    )


class FeatureSetRegistry(Base):
    """
    Таблица реестра наборов признаков (Feature Sets) для ML моделей.
    Хранит определения версий фичей (FS_D0, FS_D1, FS_D2, FS_D3) и их метаданные.
    """

    __tablename__ = "feature_set_registry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(32), unique=True, nullable=False)  # 'FS_D0', 'FS_D1', 'FS_D2', 'FS_D3'
    version = Column(Integer, nullable=False, default=1)
    description = Column(String(512), nullable=True)

    # Список фичей через запятую (для быстрого чтения)
    features_list = Column(Text, nullable=False)

    # Количество фичей в наборе
    feature_count = Column(Integer, nullable=False)

    # Метаданные набора: описание источников (OHLCV, Funding, Orderbook, etc.),
    # используемые периоды SMA/RSI/ATR и другие параметры
    metadata_json = Column(
        JSON().with_variant(JSONB, "postgresql"), nullable=True
    )

    # Флаг готовности к использованию в боевом инференсе
    is_production_ready = Column(Boolean, default=False, server_default="false")

    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        Index("idx_feature_set_name_version", "name", "version"),
    )


class Binance15mFeature(Base):
    """
    Таблица для хранения предрассчитанных фичей на 15-минутных барах.
    Ускоряет обучение моделей и бэктестинг, исключая повторный расчет индикаторов.
    """

    __tablename__ = "binance_15m_features"

    id = Column(Integer, primary_key=True, autoincrement=True)
    bar_id = Column(
        Integer,
        ForeignKey("binance_15m_bars.id", ondelete="CASCADE"),
        nullable=False,
    )
    asset = Column(
        String(32), nullable=False
    )  # 'BTCUSDT', 'ETHUSDT', 'SOLUSDT'
    open_time = Column(
        DateTime(timezone=True), nullable=False
    )  # Время бара (для быстрой фильтрации)
    feature_set = Column(
        String(32), nullable=False, default="FS_D0"
    )  # 'FS_D0', 'FS_D1', 'FS_D2', 'FS_D3'

    # Базовые технические индикаторы (FS_D0)
    ret_1 = Column(Float, nullable=True)  # Доходность за 1 бар
    ret_3 = Column(Float, nullable=True)  # Доходность за 3 бара
    ret_6 = Column(Float, nullable=True)  # Доходность за 6 баров
    ret_12 = Column(Float, nullable=True)  # Доходность за 12 баров
    ret_24 = Column(Float, nullable=True)  # Доходность за 24 бара

    volatility_6 = Column(Float, nullable=True)  # Волатильность (std ret_1) за 6 баров
    volatility_24 = Column(Float, nullable=True)  # Волатильность за 24 бара

    rsi_14 = Column(Float, nullable=True)  # RSI 14
    atr_14 = Column(Float, nullable=True)  # ATR 14

    volume_ratio_6 = Column(Float, nullable=True)  # Отношение текущего объема к среднему за 6 баров
    volume_ratio_24 = Column(Float, nullable=True)  # Отношение к среднему за 24 бара

    # Продвинутые фичи (FS_D1, FS_D2, FS_D3) — расширяемый JSON-контейнер
    extra_features = Column(
        JSON().with_variant(JSONB, "postgresql"), nullable=True
    )

    # Целевая переменная: направление следующего 15м бара (1 = UP / close > open, 0 = DOWN)
    target_direction = Column(Integer, nullable=True)

    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "asset", "open_time", "feature_set",
            name="uix_binance_15m_features_asset_time_fset"
        ),
        Index("idx_binance_features_asset_time", "asset", "open_time"),
        Index("idx_binance_features_fset", "feature_set"),
    )


class StrategyPreset(Base):
    """
    Таблица для сохранения именованных пресетов настроек стратегии.
    Позволяет сохранять, загружать и версионировать конфигурации параметров.
    """

    __tablename__ = "strategy_presets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(64), unique=True, nullable=False)
    description = Column(String(256), nullable=True)
    settings = Column(
        JSON().with_variant(JSONB, "postgresql"), nullable=False
    )
    is_default = Column(Boolean, default=False, server_default="false")
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


# ---------------------------------------------------------------------------
# AI Lab: Autonomous Optimization Architecture
# ---------------------------------------------------------------------------


class AIOptimizationRun(Base):
    """Execution session for one autonomous AI Lab optimization loop."""

    __tablename__ = "ai_optimization_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    objective = Column(String(4000), nullable=False)
    scope = Column(JSON().with_variant(JSONB, "postgresql"), nullable=False)
    autonomy_level = Column(String(32), nullable=False, server_default="EXPERIMENT")
    status = Column(String(32), nullable=False, server_default="DRAFT")
    agent_thread_id = Column(String(128), nullable=True)
    agent_type = Column(String(64), nullable=True)
    permission_id = Column(
        Integer,
        ForeignKey("ai_permissions.id", ondelete="SET NULL"),
        nullable=True,
    )
    experiment_budget = Column(Integer, nullable=False, server_default="0")
    experiments_completed = Column(Integer, nullable=False, server_default="0")
    budget_seconds = Column(Integer, nullable=False, server_default="0")
    created_by = Column(String(128), nullable=False, server_default="system")
    summary = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("idx_ai_runs_status_created", "status", "created_at"),
        Index("idx_ai_runs_permission_id", "permission_id"),
        CheckConstraint(
            "status IN ('DRAFT', 'PLANNING', 'RUNNING', 'EVALUATING', 'SHADOW', "
            "'PENDING_APPROVAL', 'ACTIVE', 'INSUFFICIENT_DATA', 'FAILED', "
            "'REJECTED', 'CANCELLED', 'ROLLED_BACK')",
            name="ck_ai_runs_status",
        ),
        CheckConstraint(
            "autonomy_level IN ('OBSERVE', 'EXPERIMENT', 'SHADOW', 'LIVE_PROPOSE', 'AUTONOMOUS_SHADOW', 'DIRECTED')",
            name="ck_ai_runs_autonomy_level",
        ),
    )


class AIExperimentConfig(Base):
    """Immutable recipe for training and evaluating an ML candidate."""

    __tablename__ = "ai_experiment_configs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(128), nullable=False)
    asset = Column(String(32), nullable=True)
    regime = Column(String(32), nullable=True)
    model_family = Column(String(32), nullable=False)
    feature_set = Column(String(32), nullable=False)
    feature_pipeline_version = Column(String(64), nullable=False)
    model_params = Column(JSON().with_variant(JSONB, "postgresql"), nullable=False)
    strategy_params = Column(JSON().with_variant(JSONB, "postgresql"), nullable=False)
    backtest_params = Column(JSON().with_variant(JSONB, "postgresql"), nullable=False)
    config_hash = Column(String(64), nullable=False, unique=True)
    parent_id = Column(
        Integer,
        ForeignKey("ai_experiment_configs.id", ondelete="SET NULL"),
        nullable=True,
    )
    description = Column(Text, nullable=True)
    created_by = Column(String(128), nullable=False, server_default="system")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("idx_ai_configs_asset_regime", "asset", "regime"),
        Index("idx_ai_configs_model_family", "model_family"),
    )


class AIRunStep(Base):
    """Atomic task step within an optimization run."""

    __tablename__ = "ai_run_steps"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(
        Integer,
        ForeignKey("ai_optimization_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    step_index = Column(Integer, nullable=False)
    step_type = Column(String(32), nullable=False)
    status = Column(String(16), nullable=False, server_default="PENDING")
    hypothesis = Column(Text, nullable=True)
    action = Column(String(64), nullable=True)
    input_payload = Column(JSON().with_variant(JSONB, "postgresql"), nullable=True)
    output_payload = Column(JSON().with_variant(JSONB, "postgresql"), nullable=True)
    summary = Column(Text, nullable=True)
    error_code = Column(String(64), nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        UniqueConstraint("run_id", "step_index", name="uix_ai_run_step_index"),
        Index("idx_ai_run_steps_run_status", "run_id", "status"),
        CheckConstraint(
            "status IN ('PENDING', 'RUNNING', 'SUCCEEDED', 'FAILED', 'SKIPPED')",
            name="ck_ai_run_steps_status",
        ),
    )


class AIStepAuditLog(Base):
    """Audit record for executor failures that cannot reference ExperimentResult."""

    __tablename__ = "ai_step_audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(
        Integer,
        ForeignKey("ai_optimization_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    step_id = Column(
        Integer,
        ForeignKey("ai_run_steps.id", ondelete="CASCADE"),
        nullable=False,
    )
    config_id = Column(
        Integer,
        ForeignKey("ai_experiment_configs.id", ondelete="SET NULL"),
        nullable=True,
    )
    action = Column(String(64), nullable=False)
    error_code = Column(String(64), nullable=False)
    error_message = Column(Text, nullable=False)
    payload = Column(JSON().with_variant(JSONB, "postgresql"), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("idx_ai_step_audit_run_step", "run_id", "step_id"),
        Index("idx_ai_step_audit_error_code", "error_code"),
    )


class AIModelArtifact(Base):
    """Serialized model artifacts produced by offline experiment jobs."""

    __tablename__ = "ai_model_artifacts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    config_id = Column(
        Integer,
        ForeignKey("ai_experiment_configs.id", ondelete="CASCADE"),
        nullable=False,
    )
    run_id = Column(
        Integer,
        ForeignKey("ai_optimization_runs.id", ondelete="SET NULL"),
        nullable=True,
    )
    step_id = Column(
        Integer,
        ForeignKey("ai_run_steps.id", ondelete="SET NULL"),
        nullable=True,
    )
    model_registry_id = Column(
        Integer,
        ForeignKey("model_registry.id", ondelete="SET NULL"),
        nullable=True,
    )
    artifact_uri = Column(String(512), nullable=True)
    artifact_bytes = Column(LargeBinary, nullable=True)
    artifact_hash = Column(String(64), nullable=False)
    schema_version = Column(String(32), nullable=False)
    feature_pipeline_version = Column(String(64), nullable=False)
    artifact_metadata = Column("metadata", JSON().with_variant(JSONB, "postgresql"), nullable=False)
    loadability_status = Column(String(16), nullable=False, server_default="UNVERIFIED")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("idx_ai_artifacts_registry", "model_registry_id"),
        CheckConstraint(
            "loadability_status IN ('UNVERIFIED', 'VALID', 'INVALID')",
            name="ck_ai_artifacts_loadability_status",
        ),
    )


class ExperimentResult(Base):
    """Metrics for one immutable experiment/configuration evaluation."""

    __tablename__ = "experiment_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(
        Integer,
        ForeignKey("ai_optimization_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    config_id = Column(
        Integer,
        ForeignKey("experiment_configs.id", ondelete="RESTRICT"),
        nullable=False,
    )
    artifact_id = Column(
        Integer,
        ForeignKey("ai_model_artifacts.id", ondelete="SET NULL"),
        nullable=True,
    )
    evaluation_kind = Column(String(24), nullable=False)
    status = Column(String(24), nullable=False, server_default="PENDING")
    code_sha = Column(String(64), nullable=True)
    dataset_fingerprint = Column(String(128), nullable=True)
    train_window_start = Column(DateTime(timezone=True), nullable=True)
    train_window_end = Column(DateTime(timezone=True), nullable=True)
    oot_window_start = Column(DateTime(timezone=True), nullable=True)
    oot_window_end = Column(DateTime(timezone=True), nullable=True)
    trade_count = Column(Integer, nullable=True)
    net_pnl = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    metrics = Column(JSON().with_variant(JSONB, "postgresql"), nullable=True)
    slices = Column(JSON().with_variant(JSONB, "postgresql"), nullable=True)
    summary = Column(String(4000), nullable=True)
    error_code = Column(String(64), nullable=True)
    error_message = Column(String(4000), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("idx_experiment_results_run_id", "run_id"),
        Index("idx_experiment_results_config_kind", "config_id", "evaluation_kind"),
        CheckConstraint(
            "evaluation_kind IN ('TRAIN', 'OOT', 'POLYMARKET_OOT', 'SHADOW')",
            name="ck_ai_results_kind",
        ),
        CheckConstraint(
            "status IN ('PENDING', 'SUCCEEDED', 'FAILED', 'INSUFFICIENT_DATA')",
            name="ck_ai_results_status",
        ),
    )


class DeploymentRevision(Base):
    """Immutable deployment bundle; rollback changes only the active pointer."""

    __tablename__ = "deployment_revisions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    revision_key = Column(String(64), nullable=False, unique=True)
    parent_id = Column(Integer, ForeignKey("deployment_revisions.id"), nullable=True)
    manifest = Column(JSON().with_variant(JSONB, "postgresql"), nullable=False)
    manifest_hash = Column(String(64), nullable=False, unique=True)
    status = Column(String(24), nullable=False, server_default="DRAFT")
    created_by = Column(String(128), nullable=False, server_default="system")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    activated_at = Column(DateTime(timezone=True), nullable=True)
    rolled_back_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("idx_deployment_revisions_status_created", "status", "created_at"),
        CheckConstraint(
            "status IN ('DRAFT', 'SHADOW', 'PENDING_APPROVAL', 'ACTIVE', 'REJECTED', 'ROLLED_BACK')",
            name="ck_deployment_revisions_status",
        ),
    )


class DeploymentEvent(Base):
    """Append-only activation, approval and rollback audit event."""

    __tablename__ = "deployment_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    revision_id = Column(
        Integer,
        ForeignKey("deployment_revisions.id", ondelete="RESTRICT"),
        nullable=False,
    )
    event_type = Column(String(32), nullable=False)
    actor = Column(String(128), nullable=False)
    reason = Column(Text, nullable=True)
    payload = Column(JSON().with_variant(JSONB, "postgresql"), nullable=True)
    previous_hash = Column(String(64), nullable=True)
    event_hash = Column(String(64), nullable=False, unique=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("idx_deployment_events_revision_created", "revision_id", "created_at"),
        Index("idx_deployment_events_revision_id_desc", "revision_id", text("id DESC")),
        CheckConstraint(
            "event_type IN ('CREATED', 'SHADOW_ASSIGNED', 'APPROVED', 'ACTIVATED', "
            "'REJECTED', 'ROLLED_BACK')",
            name="ck_deployment_events_type",
        ),
    )


class AIShadowAssignment(Base):
    """Binds a candidate artifact to real-time passive shadow evaluation."""

    __tablename__ = "ai_shadow_assignments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(
        Integer,
        ForeignKey("ai_optimization_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    candidate_artifact_id = Column(
        Integer,
        ForeignKey("ai_model_artifacts.id", ondelete="RESTRICT"),
        nullable=False,
    )
    baseline_artifact_id = Column(
        Integer,
        ForeignKey("ai_model_artifacts.id", ondelete="RESTRICT"),
        nullable=True,
    )
    asset = Column(String(32), nullable=False)
    regime = Column(String(32), nullable=True)
    status = Column(String(16), nullable=False, server_default="PENDING")
    metrics = Column(JSON().with_variant(JSONB, "postgresql"), nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    ended_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("idx_ai_shadow_run_id", "run_id"),
        Index("idx_ai_shadow_scope_status", "asset", "regime", "status"),
        CheckConstraint(
            "status IN ('PENDING', 'RUNNING', 'COMPLETED', 'STOPPED', 'FAILED')",
            name="ck_ai_shadow_status",
        ),
    )


class AIPermission(Base):
    """Versioned allow-list for autonomous actions and resource limits."""

    __tablename__ = "ai_permissions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    profile_name = Column(String(64), nullable=False)
    version = Column(Integer, nullable=False, server_default="1")
    is_current = Column(Boolean, nullable=False, server_default="true")
    allowed_actions = Column(JSON().with_variant(JSONB, "postgresql"), nullable=False)
    scope = Column(JSON().with_variant(JSONB, "postgresql"), nullable=False)
    limits = Column(JSON().with_variant(JSONB, "postgresql"), nullable=False)
    enabled = Column(Boolean, nullable=False, server_default="true")
    updated_by = Column(String(128), nullable=False, server_default="system")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("profile_name", "version", name="uix_ai_permissions_profile_version"),
        Index("idx_ai_permissions_current", "profile_name", "is_current"),
    )


class AIApprovalRequest(Base):
    """Human approval request for actions outside the autonomous profile."""

    __tablename__ = "ai_approval_requests"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(
        Integer,
        ForeignKey("ai_optimization_runs.id", ondelete="CASCADE"),
        nullable=True,
    )
    target_type = Column(String(32), nullable=False)
    target_id = Column(String(64), nullable=False)
    requested_action = Column(String(32), nullable=False)
    diff = Column(JSON().with_variant(JSONB, "postgresql"), nullable=False)
    status = Column(String(16), nullable=False, server_default="PENDING")
    requested_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    decided_at = Column(DateTime(timezone=True), nullable=True)
    decided_by = Column(String(128), nullable=True)
    decision_reason = Column(Text, nullable=True)

    __table_args__ = (
        Index("idx_ai_approval_status_requested", "status", "requested_at"),
        CheckConstraint(
            "status IN ('PENDING', 'APPROVED', 'REJECTED', 'EXPIRED')",
            name="ck_ai_approval_status",
        ),
    )
