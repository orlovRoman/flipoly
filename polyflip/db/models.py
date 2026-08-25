from sqlalchemy import (
    Column,
    Integer,
    String,
    SmallInteger,
    Float,
    Boolean,
    DateTime,
    LargeBinary,
    Index,
    UniqueConstraint,
    ForeignKey,
    CheckConstraint,
    Text,
    Numeric,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.types import JSON
from sqlalchemy.orm import declarative_base, validates
from sqlalchemy.schema import FetchedValue

Base = declarative_base()


class CryptoCandle(Base):
    """OHLCV-свечи с Binance, агрегированные полировщиком тиков."""

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

    market_id = Column(String(128), nullable=False)
    time_left_min = Column(Float, nullable=False)
    mid_price = Column(Float, nullable=False)
    spread = Column(Float, nullable=False)
    best_bid = Column(Float, nullable=True)
    best_ask = Column(Float, nullable=True)
    volume_5min = Column(Float, nullable=False)
    price_velocity = Column(Float, nullable=False)
    hour_of_day = Column(Integer, nullable=False)
    final_outcome = Column(String(16), nullable=False)  # "YES", "NO", "INVALID", "PENDING"
    flip_vs_final = Column(Boolean, nullable=True)
    recorded_at = Column(DateTime(timezone=True), nullable=False)


class TradeHistory(Base):
    """
    Таблица для хранения истории совершенных сделок.
    """

    __tablename__ = "trade_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # Legacy rows/tests may not carry the newer strategy metadata.
    strategy_name = Column(String(64), nullable=True)
    asset = Column(String(32), nullable=False)
    side = Column(String(8), nullable=True)  # BUY, SELL
    direction = Column(String(8), nullable=True)  # UP, DOWN
    price = Column(Float, nullable=True)
    size = Column(Float, nullable=True)
    cost = Column(Float, nullable=True)
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

    @validates("exit_reason")
    def _validate_exit_reason(self, key, value):
        if value is not None:
            from polyflip.execution.states import ExitReason
            if value not in ExitReason.values():
                raise ValueError(f"Invalid exit_reason: {value!r}")
        return value

    @validates("position_status")
    def _validate_position_status(self, key, value):
        if value in ("RESOLVED_REDEEMABLE", "RESOLVED_LOST") and getattr(self, "exit_reason", None) is None:
            import structlog
            structlog.get_logger(__name__).warning(
                "trade_resolved_without_exit_reason",
                trade_id=getattr(self, "id", None),
                status=value,
            )
        return value

    __table_args__ = (
        Index("idx_trades_strategy_asset", "strategy_name", "asset"),
        Index("idx_trades_timestamp", "timestamp"),
        Index("idx_trades_asset_model_version", "asset", "model_version", "status", "created_at"),
        Index("idx_trades_model_registry_id", "model_registry_id"),
        Index("idx_trade_history_market_id", "market_id"),
        Index("idx_trade_model_analytics", "mode", "model_key", "model_version", "position_status", "closed_at"),
        CheckConstraint(
            "position_accounting_version = 0 OR (entry_filled_shares IS NOT NULL AND entry_cost_usdc IS NOT NULL AND remaining_shares IS NOT NULL AND realized_pnl_usdc IS NOT NULL)",
            name="ck_trade_position_accounting_initialized",
        ),
    )

    market_id = Column(String(128), nullable=False)
    outcome_bought = Column(String(16), nullable=False)  # "YES" or "NO"
    amount_usdc = Column(Float, nullable=False)
    executed_price = Column(Float, nullable=False)
    predicted_flip_prob = Column(Float, nullable=False)
    active_features = Column(String, nullable=False)
    error_msg = Column(String, nullable=True)
    mode = Column(String(16), nullable=False, default="LIVE")
    source_paper_trade_id = Column(
        Integer, ForeignKey("trade_history.id", ondelete="SET NULL"), nullable=True
    )
    live_session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("live_trading_sessions.id", ondelete="SET NULL"),
        nullable=True,
    )
    edge = Column(Float, nullable=True)
    p_up = Column(Float, nullable=True)
    strike = Column(Float, nullable=True)
    lgbm_metadata = Column(String, nullable=True)
    updated_at = Column(DateTime(timezone=True), nullable=True)
    strategy_type = Column(String(32), nullable=True)
    market_role = Column(String(16), nullable=True)
    p_flip_effective = Column(Float, nullable=True)
    p_win_effective = Column(Float, nullable=True)
    market_end_time      = Column(DateTime(timezone=True), nullable=True)
    stop_loss_pct        = Column(Float, nullable=True)
    stop_loss_price      = Column(Float, nullable=True)
    stop_loss_status     = Column(String(20), nullable=True, default="ACTIVE")
    stop_loss_hit_at     = Column(DateTime(timezone=True), nullable=True)
    stop_loss_sell_price = Column(Float, nullable=True)
    stop_loss_sell_size  = Column(Numeric(38, 18), nullable=True)
    take_profit_enabled    = Column(Boolean, nullable=True, default=False)
    take_profit_multiplier = Column(Float, nullable=True)
    take_profit_price      = Column(Float, nullable=True)
    take_profit_status     = Column(String(20), nullable=True, default="ACTIVE")
    take_profit_hit_at     = Column(DateTime(timezone=True), nullable=True)
    take_profit_sell_price = Column(Float, nullable=True)
    take_profit_sell_size  = Column(Numeric(38, 18), nullable=True)
    position_accounting_version = Column(SmallInteger, nullable=False, server_default="0")
    position_version = Column(Integer, nullable=False, default=1, server_default="1")
    entry_filled_shares = Column(Numeric(38, 18), nullable=True)
    entry_cost_usdc = Column(Numeric(38, 18), nullable=True)
    remaining_shares = Column(Numeric(38, 18), nullable=True)
    realized_pnl_usdc = Column(Numeric(38, 18), nullable=True)
    position_status = Column(String(32), nullable=False, default="OPEN")
    exit_reason     = Column(String(32), nullable=True)
    exit_order_id   = Column(String(128), nullable=True)
    exit_attempt_id = Column(UUID(as_uuid=True), nullable=True)
    exit_claimed_at = Column(DateTime(timezone=True), nullable=True)
    last_exit_error = Column(Text, nullable=True)
    exit_attempts   = Column(Integer, nullable=False, default=0)
    settlement_outcome = Column(String(16), nullable=True)
    expected_payout_usdc = Column(Numeric(38, 18), nullable=True)
    redeemable_shares = Column(Numeric(38, 18), nullable=True)
    redemption_status = Column(String(32), nullable=False, server_default="NOT_REQUIRED")
    redemption_tx_hash = Column(String(128), nullable=True)
    redeemed_payout_usdc = Column(Numeric(38, 18), nullable=True)
    redeemed_at = Column(DateTime(timezone=True), nullable=True)
    config_snapshot = Column(Text, nullable=True)   # JSON паспорт настроек на момент сделки
    model_key = Column(String(64), nullable=True)
    confirm_model_key = Column(String(64), nullable=True)
    confirm_model_version = Column(Integer, nullable=True)
    model_attribution_source = Column(String(16), nullable=True)
    direction_model_key = Column(String(64), nullable=True)
    direction_model_version = Column(Integer, nullable=True)
    entry_model_key = Column(String(64), nullable=True)
    entry_model_version = Column(Integer, nullable=True)
    entry_model_source = Column(String(32), nullable=True)
    p_candidate_win = Column(Float, nullable=True)
    p_logreg_win = Column(Float, nullable=True)
    direction_discount_mult = Column(Float, nullable=True)
    combined_dir_discount_weight = Column(Float, nullable=True)
    gross_edge = Column(Float, nullable=True)
    cost_buffer = Column(Float, nullable=True)
    net_edge = Column(Float, nullable=True)
    decision_run_id = Column(String(64), nullable=True)
    direction_value = Column(String(16), nullable=True)
    would_live_accept = Column(Boolean, nullable=True)
    p_flip_raw = Column(Float, nullable=True)
    entry_model_ece = Column(Float, nullable=True)


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

    run_at = Column(DateTime(timezone=True), nullable=False)
    markets_found = Column(Integer, nullable=False, default=0)
    markets_saved = Column(Integer, nullable=False, default=0)
    duration_sec = Column(Float, nullable=False)


class LiveMarket(Base):
    """
    Таблица текущих активных 15-минутных рынков с Polymarket.
    """

    __tablename__ = "live_markets"

    id = Column(Integer, nullable=True, server_default=FetchedValue())
    condition_id = Column(String(128), unique=True, nullable=True)
    slug = Column(String(256), nullable=True)
    asset = Column(String(32), nullable=False)
    question = Column(String(512), nullable=True)
    end_date = Column(DateTime(timezone=True), nullable=True)
    clob_token_id = Column(String(256), nullable=True)
    clob_token_down_id = Column(String(256), nullable=True)
    status = Column(
        String(32), default="ACTIVE"
    )  # ACTIVE, RESOLVED, EXPIRED, CANCELLED

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

    market_id = Column(String(128), primary_key=True)
    yes_token_id = Column(String(128), nullable=False)
    no_token_id = Column(String(128), nullable=False)
    end_time_est = Column(DateTime(timezone=True), nullable=False)
    current_yes_price = Column(Float, nullable=False)
    current_no_price = Column(Float, nullable=False)
    current_spread = Column(Float, nullable=False)
    volume_5min = Column(Float, nullable=False, default=0.0)
    price_velocity = Column(Float, nullable=False, default=0.0)
    last_updated = Column(DateTime(timezone=True), nullable=False)
    underlying_price = Column(Float, nullable=True)
    trading_status = Column(String(24), nullable=False, server_default="UNKNOWN")
    accepting_orders = Column(Boolean, nullable=True)
    resolution_status = Column(String(24), nullable=False, server_default="PENDING")
    final_outcome = Column(String(16), nullable=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    resolution_checked_at = Column(DateTime(timezone=True), nullable=True)
    resolution_source = Column(String(32), nullable=True)


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
        String(64), nullable=True
    )  # LogisticRegression, LightGBM, etc.
    features = Column(
        String(256), nullable=True
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
    )
    __table_args__ = (
        UniqueConstraint(
            "asset", "version", name="uix_model_registry_asset_version"
        ),
        Index("idx_model_registry_asset_active", "asset", "is_active"),
    )

    model_blob = Column(LargeBinary, nullable=True)
    baseline = Column(Float, nullable=True)
    training_params = Column(JSON().with_variant(JSONB, 'postgresql'), nullable=True)
    feature_importance = Column(JSON().with_variant(JSONB, 'postgresql'), nullable=True)
    train_samples = Column(Integer, nullable=True)
    validation_samples = Column(Integer, nullable=True)
    positive_rate = Column(Float, nullable=True)
    precision_at_threshold = Column(Float, nullable=True)
    recall_at_threshold = Column(Float, nullable=True)
    f1_at_threshold = Column(Float, nullable=True)
    brier_score = Column(Float, nullable=True)
    training_window_start = Column(DateTime(timezone=True), nullable=True)
    training_window_end = Column(DateTime(timezone=True), nullable=True)
    ece = Column(Float, nullable=True)
    backtest_wr = Column(Float, nullable=True)
    interval = Column(String(5), nullable=False, server_default='15m')
    dataset_fingerprint = Column(String(32), nullable=True)
    trained_at = Column(DateTime(timezone=True), nullable=False)
    quality_gate_passed = Column(Boolean, nullable=True)
    quality_gate_reasons = Column(JSON, nullable=True)
    activation_source = Column(String(16), nullable=True)
    quality_override = Column(Boolean, nullable=True, default=False)
    activated_at = Column(DateTime(timezone=True), nullable=True)
    activated_by = Column(String(128), nullable=True)
    activation_reason = Column(Text, nullable=True)


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
    )
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

    updated_by = Column(String(64), nullable=False)


class DecisionFunnelLog(Base):
    """
    Таблица для детального аудита каждого шага воронки принятия решений (Decision Funnel).
    Фиксирует причину отказа (REJECT) или пропуска (PASS) для каждого тика/рынка.
    """

    __tablename__ = "decision_funnel_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    asset = Column(String(32), nullable=False)
    condition_id = Column(String(128), nullable=True)
    step_name = Column(
        String(64), nullable=True
    )  # e.g., 'TIME_WINDOW', 'SPREAD_CHECK', 'ML_INFERENCE', 'ORDER_EXECUTION'
    decision = Column(String(16), nullable=True)  # 'PASS', 'REJECT', 'EXECUTE'
    reason = Column(
        String(256), nullable=True
    )  # e.g., 'time_left_seconds 895 > 840', 'p_up 0.52 < 0.55'
    details = Column(
        JSON().with_variant(JSONB, "postgresql"), nullable=True
    )

    __table_args__ = (
        Index("idx_funnel_asset_timestamp", "asset", "timestamp"),
        Index("idx_funnel_decision_step", "decision", "step_name"),
    )

    created_at   = Column(DateTime(timezone=True), nullable=False)
    market_id    = Column(String(128), nullable=False)
    trading_mode = Column(String(16))  # ML / COMBINED / EXPERT
    execution_mode = Column(String(16), nullable=True) # PAPER / LIVE / SHADOW
    used_model   = Column(String(64),  nullable=True)   # "BTC_contested", "ETH" и т.д.
    p_flip       = Column(Float, nullable=True)
    p_flip_raw   = Column(Float, nullable=True)
    entry_model_ece = Column(Float, nullable=True)
    edge         = Column(Float, nullable=True)
    fresh_price  = Column(Float, nullable=True)
    would_live_accept = Column(Boolean, nullable=True)
    threshold_lower = Column(Float, nullable=True)   # NO_FLIP_THRESHOLD (lower)
    threshold_upper = Column(Float, nullable=True)   # FLIP_THRESHOLD (upper)
    min_edge_used   = Column(Float, nullable=True)
    g1_model_loaded     = Column(Boolean, nullable=True)  # модель в кеше
    g2_price_fetched    = Column(Boolean, nullable=True)  # API цена получена
    g3_dead_zone        = Column(Boolean, nullable=True)  # НЕ в dead zone → True
    g4_no_flip          = Column(Boolean, nullable=True)  # p_flip < lower (тренд)
    g5_min_edge         = Column(Boolean, nullable=True)  # edge >= MIN_EDGE
    g6_price_range      = Column(Boolean, nullable=True)  # цена в [MIN_PRICE, MAX_PRICE]
    g7_crypto_confirm   = Column(Boolean, nullable=True)  # LightGBM согласен
    g8_combined_vote    = Column(Boolean, nullable=True)  # финальный голос COMBINED
    final_action = Column(String(16), nullable=False)   # BUY_YES, BUY_NO, SKIP
    skip_reason  = Column(String(256), nullable=True)   # краткая причина если SKIP
    primary_model_key = Column(String(64), nullable=True)
    primary_model_version = Column(Integer, nullable=True)
    confirm_model_key = Column(String(64), nullable=True)
    confirm_model_version = Column(Integer, nullable=True)
    proposed_action = Column(String(16), nullable=True)
    proposed_price = Column(Float, nullable=True)
    proposed_amount_usdc = Column(Float, nullable=True)
    confirm_direction = Column(String(16), nullable=True)
    confirm_passed = Column(Boolean, nullable=True)
    decision_run_id = Column(String(64), nullable=True)
    direction_model_key = Column(String(64), nullable=True)
    direction_model_version = Column(Integer, nullable=True)
    required_direction_model_key = Column(String(64), nullable=True)
    direction_regime = Column(String(32), nullable=True)
    direction_status = Column(String(32), nullable=True)
    direction_probability = Column(Float, nullable=True)
    direction_p_up = Column(Float, nullable=True)
    direction_p_down = Column(Float, nullable=True)
    direction_threshold_up = Column(Float, nullable=True)
    direction_threshold_down = Column(Float, nullable=True)
    direction_value = Column(String(16), nullable=True)
    direction_raw_opinion = Column(String(16), nullable=True)
    direction_p_up_raw = Column(Float, nullable=True)
    direction_p_down_raw = Column(Float, nullable=True)
    entry_requested_key = Column(String(64), nullable=True)
    entry_model_key = Column(String(64), nullable=True)
    entry_model_version = Column(Integer, nullable=True)
    entry_model_phase = Column(String(32), nullable=True)
    entry_model_source = Column(String(32), nullable=True)
    entry_status = Column(String(32), nullable=True)
    fallback_reason = Column(String(128), nullable=True)
    p_candidate_win = Column(Float, nullable=True)
    p_logreg_win = Column(Float, nullable=True)
    direction_discount_mult = Column(Float, nullable=True)
    combined_dir_discount_weight = Column(Float, nullable=True)
    candidate_side = Column(String(16), nullable=True)
    candidate_ask = Column(Float, nullable=True)
    gross_edge = Column(Float, nullable=True)
    cost_buffer = Column(Float, nullable=True)
    net_edge = Column(Float, nullable=True)
    strike_source = Column(String(32), nullable=True)
    strike_proxy = Column(Float, nullable=True)
    underlying_price = Column(Float, nullable=True)
    distance_to_strike_pct = Column(Float, nullable=True)
    max_acceptable_price = Column(Float, nullable=True)
    direction_error_detail = Column(String(512), nullable=True)

    # MRF telemetry (added in v2)
    mrf_mode = Column(String(16), nullable=True)        # OFF / SHADOW / ACTIVE
    mrf_phase = Column(String(32), nullable=True)       # global MarketPhase
    mrf_asset_phase = Column(String(32), nullable=True) # per-asset MarketPhase
    mrf_strength = Column(Float, nullable=True)         # global strength (0-1)
    mrf_confidence = Column(Float, nullable=True)       # global confidence (0-1)
    mrf_multiplier = Column(Float, nullable=True)       # policy stake multiplier
    mrf_applied = Column(Boolean, nullable=True)        # whether filter actually applied
    # MRF v2 expanded audit
    mrf_evaluated = Column(Boolean, nullable=True)
    mrf_as_of = Column(DateTime(timezone=True), nullable=True)
    mrf_failure_reason = Column(String(256), nullable=True)
    mrf_audit_json = Column(Text, nullable=True)
    mrf_original_action = Column(String(16), nullable=True)
    mrf_original_bet = Column(Float, nullable=True)
    mrf_final_action = Column(String(16), nullable=True)
    mrf_final_bet = Column(Float, nullable=True)


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
    mode = Column(String(16), nullable=False, server_default="STANDARD")
    autonomy_level = Column(String(32), nullable=False, server_default="EXPERIMENT")
    status = Column(String(32), nullable=False, server_default="DRAFT")
    agent_thread_id = Column(String(128), nullable=True)
    llm_provider = Column(String(32), nullable=True)
    llm_research_model = Column(String(128), nullable=True)
    llm_summary_model = Column(String(128), nullable=True)
    llm_snapshot = Column(
        JSON().with_variant(JSONB, "postgresql"), nullable=True
    )
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
            "status IN ('DRAFT', 'QUEUED', 'PLANNING', 'RUNNING', 'EVALUATING', "
            "'PAUSED', 'SHADOW', 'PENDING_APPROVAL', 'ACTIVE', 'COMPLETED', "
            "'INSUFFICIENT_DATA', 'RESEARCH_PROVISIONAL', 'INSUFFICIENT_EVIDENCE', "
            "'TECHNICAL_INVALID', 'FAILED', 'REJECTED', 'CANCELLED', 'ROLLED_BACK')",
            name="ck_ai_runs_status",
        ),
        CheckConstraint(
            "mode IN ('STANDARD', 'RESEARCH')",
            name="ck_ai_runs_mode",
        ),
        CheckConstraint(
            "autonomy_level IN ('OBSERVE', 'EXPERIMENT', 'SHADOW', 'AUTONOMOUS_SHADOW', 'AUTONOMOUS_CONFIG', 'LIVE_PROPOSE', 'AUTONOMOUS_LIVE', 'DIRECTED')",
            name="ck_ai_runs_autonomy_level",
        ),
    )

    budget_experiments = Column(Integer, nullable=False, server_default="0")


class ExperimentConfig(Base):
    """Canonical experiment recipe created by the AI optimizer migration.

    The AI Lab branch also contains AIExperimentConfig for the newer
    immutable configuration API. Both tables exist in production, so this
    legacy table must remain represented in the shared ORM metadata as well;
    otherwise Alembic cannot resolve experiment_results.config_id.
    """

    __tablename__ = "experiment_configs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(128), nullable=False)
    description = Column(Text, nullable=True)
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
        ForeignKey("experiment_configs.id"),
        nullable=True,
    )
    created_by = Column(String(128), nullable=False, server_default="system")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index(
            "idx_ai_experiment_configs_scope",
            "asset",
            "regime",
            "model_family",
        ),
        Index("idx_ai_experiment_configs_created_at", "created_at"),
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

    client_request_id = Column(String(64), nullable=True)

    __table_args__ = (
        UniqueConstraint("run_id", "step_index", name="uix_ai_run_step_index"),
        UniqueConstraint(
            "run_id", "client_request_id", name="uix_ai_run_step_client_id"
        ),
        Index("idx_ai_run_steps_run_status", "run_id", "status"),
        Index("idx_ai_run_steps_client_id", "run_id", "client_request_id"),
        CheckConstraint(
            "status IN ('PENDING', 'RUNNING', 'SUCCEEDED', 'FAILED', 'SKIPPED')",
            name="ck_ai_run_steps_status",
        ),
    )

    retry_count = Column(Integer, nullable=False, server_default="0")


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

    sha256 = Column(String(64), nullable=False, unique=True)


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
        ForeignKey("ai_experiment_configs.id", ondelete="RESTRICT"),
        nullable=True,
    )
    # Results written before the AI Lab config split remain addressable without
    # pretending that their legacy config ID belongs to ai_experiment_configs.
    legacy_config_id = Column(
        Integer,
        ForeignKey("experiment_configs.id", ondelete="SET NULL"),
        nullable=True,
    )
    step_id = Column(
        Integer,
        ForeignKey("ai_run_steps.id", ondelete="SET NULL"),
        nullable=True,
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
            "evaluation_kind IN ('TRAIN', 'OOT', 'POLYMARKET_OOT', 'SHADOW', 'LIVE')",
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
        ForeignKey("ai_optimization_runs.id", ondelete="SET NULL"),
        nullable=True,
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


class AIShadowObservation(Base):
    """Same-snapshot active/candidate shadow comparison and counterfactual PnL."""

    __tablename__ = "ai_shadow_observations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    assignment_id = Column(Integer, ForeignKey("ai_shadow_assignments.id", ondelete="CASCADE"), nullable=False)
    run_id = Column(Integer, ForeignKey("ai_optimization_runs.id", ondelete="SET NULL"), nullable=True)
    market_id = Column(String(128), nullable=False)
    snapshot_at = Column(DateTime(timezone=True), nullable=False)
    active_model_key = Column(String(64), nullable=True)
    candidate_model_key = Column(String(64), nullable=False)
    active_action = Column(String(32), nullable=True)
    candidate_action = Column(String(32), nullable=True)
    active_probability = Column(Float, nullable=True)
    candidate_probability = Column(Float, nullable=True)
    active_ask = Column(Float, nullable=True)
    candidate_ask = Column(Float, nullable=True)
    active_net_edge = Column(Float, nullable=True)
    candidate_net_edge = Column(Float, nullable=True)
    market_outcome = Column(String(16), nullable=True)
    active_pnl = Column(Float, nullable=True)
    candidate_pnl = Column(Float, nullable=True)
    lr_direction_vote = Column(String(16), nullable=True)
    lgbm_direction_vote = Column(String(16), nullable=True)
    consensus_type = Column(String(32), nullable=True)
    shadow_logreg_action = Column(String(32), nullable=True)
    actual_combined_action = Column(String(32), nullable=True)
    shadow_logreg_net_edge = Column(Float, nullable=True)
    actual_net_edge = Column(Float, nullable=True)
    status = Column(String(24), nullable=False, server_default="PENDING")
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    idempotency_key = Column(String(256), nullable=False, unique=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("idx_ai_shadow_obs_assignment_market", "assignment_id", "market_id"),
        Index("idx_ai_shadow_obs_status", "status", "created_at"),
        CheckConstraint(
            "status IN ('PENDING', 'RESOLVED', 'ABSTAINED', 'INVALID')",
            name="ck_ai_shadow_observation_status",
        ),
    )


class AIExperimentJob(Base):
    """Durable idempotent job record for restart-safe AI Lab execution."""

    __tablename__ = "ai_experiment_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("ai_optimization_runs.id", ondelete="CASCADE"), nullable=False)
    step_id = Column(Integer, ForeignKey("ai_run_steps.id", ondelete="CASCADE"), nullable=False)
    operation = Column(String(64), nullable=False)
    status = Column(String(16), nullable=False, server_default="QUEUED")
    attempt = Column(Integer, nullable=False, server_default="0")
    idempotency_key = Column(String(256), nullable=False, unique=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    heartbeat_at = Column(DateTime(timezone=True), nullable=True)
    # Token of the worker that owns the current attempt.  Keeping this on the
    # job (in addition to the run-level lease) makes stale/retry transitions
    # auditable and prevents a different worker from completing an old claim.
    owner_token = Column(String(128), nullable=True)
    error = Column(Text, nullable=True)
    traceback = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("idx_ai_jobs_status_heartbeat", "status", "heartbeat_at"),
        Index("idx_ai_jobs_run_step", "run_id", "step_id"),
        CheckConstraint(
            "status IN ('QUEUED', 'RUNNING', 'RETRY_WAIT', 'SUCCEEDED', 'FAILED', 'STALE', 'CANCELLED')",
            name="ck_ai_experiment_jobs_status",
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

# Canonical ORM models restored while preserving the AI Lab schema.
class ModelRegistryOOFArtifact(Base):
    """Immutable OOF rows and Polymarket quotes for one saved LGBM candidate."""

    __tablename__ = "model_registry_oof_artifacts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_registry_id = Column(
        Integer,
        ForeignKey("model_registry.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    schema_version = Column(Integer, nullable=False, server_default="2")
    row_count = Column(Integer, nullable=False)
    artifact_blob = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (Index("idx_model_registry_oof_model", "model_registry_id"),)

class LGBMExperimentConfig(Base):
    """Immutable, versioned configuration used by a LightGBM experiment."""

    __tablename__ = "lgbm_experiment_configs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(128), nullable=False)
    description = Column(Text, nullable=True)
    asset = Column(String(32), nullable=True)
    volatility_regime = Column(String(32), nullable=True)
    feature_set = Column(String(8), nullable=False)
    feature_set_version = Column(String(64), nullable=False)
    model_params = Column(JSON().with_variant(JSONB, "postgresql"), nullable=False)
    calibration_params = Column(JSON().with_variant(JSONB, "postgresql"), nullable=False)
    threshold_params = Column(JSON().with_variant(JSONB, "postgresql"), nullable=False)
    backtest_params = Column(JSON().with_variant(JSONB, "postgresql"), nullable=False)
    config_hash = Column(String(64), nullable=False)
    parent_id = Column(Integer, ForeignKey("lgbm_experiment_configs.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False)
    created_by = Column(String(128), nullable=False, default="dashboard")
    is_archived = Column(Boolean, nullable=False, default=False, server_default="false")

    __table_args__ = (
        Index("idx_lgbm_experiment_configs_scope", "asset", "volatility_regime"),
        Index("idx_lgbm_experiment_configs_created_at", "created_at"),
        Index("idx_lgbm_experiment_configs_hash", "config_hash"),
    )

class LGBMTrainingJob(Base):
    """Durable queue entry for resource-intensive LightGBM training."""

    __tablename__ = "lgbm_training_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(32), nullable=False)
    interval = Column(String(8), nullable=False, server_default="15m")
    feature_set = Column(String(8), nullable=False)
    activate_after_train = Column(Boolean, nullable=False, default=False, server_default="false")
    experiment_config_id = Column(
        Integer,
        ForeignKey("lgbm_experiment_configs.id", ondelete="SET NULL"),
        nullable=True,
    )
    status = Column(String(16), nullable=False, default="QUEUED", server_default="QUEUED")
    created_at = Column(DateTime(timezone=True), nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    worker_pid = Column(Integer, nullable=True)
    result = Column(JSON().with_variant(JSONB, "postgresql"), nullable=True)
    error = Column(Text, nullable=True)
    error_traceback = Column(Text, nullable=True)

    __table_args__ = (
        Index("idx_lgbm_training_jobs_status_created", "status", "created_at"),
        Index("idx_lgbm_training_jobs_symbol_created", "symbol", "created_at"),
        Index(
            "uq_lgbm_training_jobs_symbol_active",
            "symbol",
            unique=True,
            postgresql_where=text("status IN ('QUEUED', 'RUNNING')"),
            sqlite_where=text("status IN ('QUEUED', 'RUNNING')"),
        ),
        CheckConstraint(
            "status IN ('QUEUED', 'RUNNING', 'SUCCESS', 'FAILED')",
            name="ck_lgbm_training_jobs_status",
        ),
    )

class SlippageLog(Base):
    __tablename__ = "slippage_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(Integer, nullable=False)          # FK → trade_history.id
    market_id = Column(String(128), nullable=False)
    asset = Column(String(32), nullable=False)
    outcome_bought = Column(String(16), nullable=False)  # YES / NO
    expected_price = Column(Float, nullable=False)       # buy_price в момент решения
    executed_price = Column(Float, nullable=False)       # реально исполненная цена
    slippage = Column(Float, nullable=False)             # executed - expected
    slippage_pct = Column(Float, nullable=False)         # slippage / expected * 100
    bet_size_usdc = Column(Float, nullable=False)
    slippage_cost_usdc = Column(Float, nullable=False)   # slippage * (bet / executed_price)
    mode = Column(String(16), nullable=False)             # LIVE / PAPER
    created_at = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("idx_slippage_log_asset", "asset"),
        Index("idx_slippage_log_created_at", "created_at"),
        Index("idx_slippage_log_trade_id", "trade_id"),
    )

class StrategyConfig(Base):
    __tablename__ = "strategy_config"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(64), nullable=False)
    old_value = Column(String, nullable=True)      # предыдущее значение (None при первом set)
    new_value = Column(String, nullable=False)
    changed_at = Column(DateTime(timezone=True), nullable=False)
    changed_by = Column(String(64), nullable=False)  # "user", "trainer", "system"
    source_ip = Column(String(64), nullable=True)    # IP дашборда при ручном изменении
    note = Column(String, nullable=True)             # опциональный комментарий

    __table_args__ = (
        Index("idx_strategy_config_key", "key"),
        Index("idx_strategy_config_changed_at", "changed_at"),
    )

class ConfigPreset(Base):
    """
    Слепок всех параметров торгового движка на момент сохранения.
    preset_type: 'manual' | 'ath_capital' | 'ath_pnl'
    """
    __tablename__ = "config_presets"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    name            = Column(String(128), nullable=False)          # "BTC_Profitable_July10"
    description     = Column(String(512), nullable=True)
    preset_type     = Column(String(32),  nullable=False, default="manual")
    snapshot        = Column(Text, nullable=False)                 # JSON-дамп всех RuntimeSettings
    capital_at_save = Column(Float, nullable=True)            # баланс на момент сохранения
    pnl_at_save     = Column(Float, nullable=True)            # суммарный PnL на момент
    created_at      = Column(DateTime(timezone=True), nullable=False)
    created_by      = Column(String(64), nullable=False, default="user")
    is_active       = Column(Boolean, nullable=False, default=True)  # False = удалён

    __table_args__ = (
        Index("idx_config_presets_created_at", "created_at"),
        Index("idx_config_presets_type",       "preset_type"),
    )

class MarketDirectionSignal(Base):
    """
    Таблица для атомарной фиксации единого прогноза LightGBM на 15-минутный рынок.
    Исключает пересчеты и дрейф сигналов при повторных вызовах в рамках одного market_id.
    """
    __tablename__ = "market_direction_signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market_id = Column(String(128), nullable=False, unique=True)
    asset = Column(String(32), nullable=False)
    symbol = Column(String(32), nullable=False)
    regime = Column(String(32), nullable=False)
    direction = Column(String(16), nullable=False)  # "UP", "DOWN", "NONE"
    p_up = Column(Float, nullable=False)
    p_down = Column(Float, nullable=False)
    signal_strength = Column(Float, nullable=False)
    strike = Column(Float, nullable=True)
    threshold_up = Column(Float, nullable=False)
    threshold_down = Column(Float, nullable=False)
    model_key = Column(String(64), nullable=False)
    model_version = Column(Integer, nullable=False)
    features_ok = Column(Boolean, nullable=False, default=True)
    risk_vetoed = Column(Boolean, nullable=False, default=False)
    risk_reason = Column(String(256), nullable=True)
    stake_multiplier = Column(Float, nullable=False, default=1.0)
    funding_rate = Column(Float, nullable=False, default=0.0)
    ece = Column(Float, nullable=False, default=0.0)
    status = Column(String(32), nullable=False, default="READY")
    inverted = Column(Boolean, nullable=False, default=False)
    p_up_raw = Column(Float, nullable=False, default=0.0)
    p_down_raw = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("idx_direction_signal_market", "market_id"),
        Index("idx_direction_signal_asset_created", "asset", "created_at"),
    )

class AIWorkerLease(Base):
    """Short-lived cross-process lease for one autonomous worker run."""

    __tablename__ = "ai_worker_leases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(
        Integer,
        ForeignKey("ai_optimization_runs.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    worker_id = Column(String(128), nullable=False, server_default="external-ai-research-agent")
    owner_token = Column(String(128), nullable=False)
    acquired_at = Column(DateTime(timezone=True), nullable=False)
    heartbeat_at = Column(DateTime(timezone=True), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("idx_ai_worker_leases_expires", "expires_at"),
        Index("idx_ai_worker_leases_worker", "worker_id"),
    )


class AIConfigOverlay(Base):
    """Versioned runtime settings overlay proposed and applied by AI Lab."""

    __tablename__ = "ai_config_overlays"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(
        Integer,
        ForeignKey("ai_optimization_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    parent_overlay_id = Column(
        Integer,
        ForeignKey("ai_config_overlays.id", ondelete="SET NULL"),
        nullable=True,
    )
    scope = Column(JSON().with_variant(JSONB, "postgresql"), nullable=False, default=dict)
    changes = Column(JSON().with_variant(JSONB, "postgresql"), nullable=False, default=dict)
    base_settings_hash = Column(String(64), nullable=False)
    resulting_settings_hash = Column(String(64), nullable=False)
    status = Column(String(32), nullable=False, server_default="PENDING")
    created_by = Column(String(128), nullable=False, server_default="ai_agent")
    expires_at = Column(DateTime(timezone=True), nullable=True)
    rollback_payload = Column(JSON().with_variant(JSONB, "postgresql"), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("idx_ai_overlay_run_status", "run_id", "status"),
        Index("idx_ai_overlay_expires", "expires_at"),
        CheckConstraint(
            "status IN ('PENDING', 'APPLIED', 'EXPIRED', 'ROLLED_BACK', 'REJECTED')",
            name="ck_ai_config_overlays_status",
        ),
    )


class AILLMModelCatalog(Base):
    """Cached snapshot of one provider's LLM models discovered dynamically.

    The dashboard must offer every model the configured provider actually
    returns, so this table caches discovery results per ``(provider,
    model_id)`` and keeps serving the last known catalog when the provider
    endpoint is temporarily unreachable (``is_available``/``expires_at``
    drive staleness).
    """

    __tablename__ = "ai_llm_model_catalog"

    id = Column(Integer, primary_key=True, autoincrement=True)
    provider = Column(String(32), nullable=False)
    model_id = Column(String(128), nullable=False)
    display_name = Column(String(256), nullable=True)
    protocol = Column(String(32), nullable=False, server_default="responses")
    supports_structured_output = Column(
        Boolean, nullable=False, server_default="true"
    )
    is_available = Column(Boolean, nullable=False, server_default="true")
    is_discovered = Column(Boolean, nullable=False, server_default="true")
    probe_status = Column(String(16), nullable=False, server_default="UNCHECKED")
    last_checked_at = Column(DateTime(timezone=True), nullable=True)
    discovered_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    expires_at = Column(DateTime(timezone=True), nullable=True)
    raw_metadata = Column(
        "metadata", JSON().with_variant(JSONB, "postgresql"), nullable=True
    )

    __table_args__ = (
        UniqueConstraint(
            "provider", "model_id", name="uix_ai_llm_catalog_provider_model"
        ),
        Index("idx_ai_llm_catalog_provider", "provider"),
    )
