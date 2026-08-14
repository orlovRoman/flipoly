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
    Text,
    JSON,
    CheckConstraint,
    Numeric,
    SmallInteger,
    ForeignKey,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base, validates

Base = declarative_base()

class MarketSnapshot(Base):
    __tablename__ = "market_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset = Column(String(32), nullable=False)
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

    __table_args__ = (
        UniqueConstraint("market_id", "recorded_at", name="uix_market_recorded"),
        Index("idx_market_snapshots_asset", "asset"),
        Index("idx_market_snapshots_asset_time", "asset", "time_left_min"),
        Index("idx_market_snapshots_recorded_at", "recorded_at"),
        CheckConstraint("final_outcome IN ('PENDING', 'YES', 'NO', 'INVALID')", name="ck_market_snapshot_outcome"),
    )

class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset = Column(String(32), nullable=False)
    version = Column(Integer, nullable=False)
    model_blob = Column(LargeBinary, nullable=False)
    accuracy = Column(Float, nullable=False)
    baseline = Column(Float, nullable=True)

    decision_threshold = Column(Float, nullable=True)
    decision_threshold_down = Column(Float, nullable=True)
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

    features = Column(String, nullable=True)
    ece = Column(Float, nullable=True)
    backtest_pnl = Column(Float, nullable=True)
    backtest_trades = Column(Integer, nullable=True)
    backtest_wr = Column(Float, nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)
    interval = Column(String(5), nullable=False, server_default='15m')
    model_type = Column(String(20), nullable=False, server_default='logreg')
    dataset_fingerprint = Column(String(32), nullable=True)
    trained_at = Column(DateTime(timezone=True), nullable=False)

    # --- Quality Gate ---
    # Результат автоматической проверки качества при обучении
    quality_gate_passed = Column(Boolean, nullable=True)
    # Детали провала: {"auc": 0.49, "ece": 0.21, "reasons": [...]}
    quality_gate_reasons = Column(JSON, nullable=True)

    # --- Activation Audit ---
    # TRAINER = активирована автоматически после обучения; DASHBOARD = активирована вручную из дашборда
    # quality_override = True означает, что Quality Gate был обойдён принудительно
    activation_source = Column(String(16), nullable=True)
    quality_override = Column(Boolean, nullable=True, default=False)
    activated_at = Column(DateTime(timezone=True), nullable=True)
    activated_by = Column(String(128), nullable=True)
    activation_reason = Column(Text, nullable=True)

    __table_args__ = (
        Index("idx_model_registry_asset_active", "asset", "is_active"),
        CheckConstraint("model_type IN ('logreg', 'lgbm')", name="ck_model_registry_model_type"),
    )



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


class CollectorStatus(Base):
    __tablename__ = "collector_status"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_at = Column(DateTime(timezone=True), nullable=False)
    status = Column(String(16), nullable=False)  # "success", "partial", "error"
    markets_found = Column(Integer, nullable=False, default=0)
    markets_saved = Column(Integer, nullable=False, default=0)
    error_message = Column(String, nullable=True)
    duration_sec = Column(Float, nullable=False)

class RuntimeSettings(Base):
    __tablename__ = "runtime_settings"

    key = Column(String(64), primary_key=True)
    value = Column(String, nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)
    updated_by = Column(String(64), nullable=False)

class LiveMarket(Base):
    __tablename__ = "live_markets"

    market_id = Column(String(128), primary_key=True)
    asset = Column(String(32), nullable=False)
    question = Column(String, nullable=False)
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

    # --- Состояние торговли и результат ---
    trading_status = Column(String(24), nullable=False, server_default="UNKNOWN")
    accepting_orders = Column(Boolean, nullable=True)
    resolution_status = Column(String(24), nullable=False, server_default="PENDING")
    final_outcome = Column(String(16), nullable=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    resolution_checked_at = Column(DateTime(timezone=True), nullable=True)
    resolution_source = Column(String(32), nullable=True)

    @property
    def resolved(self) -> bool:
        return self.resolution_status in {"RESOLVED", "INVALID"}

    __table_args__ = (
        Index("idx_live_markets_asset", "asset"),
    )

class TradeHistory(Base):
    __tablename__ = "trade_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market_id = Column(String(128), nullable=False)
    asset = Column(String(32), nullable=False)
    outcome_bought = Column(String(16), nullable=False)  # "YES" or "NO"
    amount_usdc = Column(Float, nullable=False)
    executed_price = Column(Float, nullable=False)
    predicted_flip_prob = Column(Float, nullable=False)
    active_features = Column(String, nullable=False)
    model_version = Column(Integer, nullable=True)
    status = Column(String(32), nullable=False) # "SUCCESS", "FAILED"
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
    pnl = Column(Float, nullable=True)
    edge = Column(Float, nullable=True)
    p_up = Column(Float, nullable=True)
    strike = Column(Float, nullable=True)
    lgbm_metadata = Column(String, nullable=True)
    updated_at = Column(DateTime(timezone=True), nullable=True)
    
    strategy_type = Column(String(32), nullable=True)
    market_role = Column(String(16), nullable=True)
    p_flip_effective = Column(Float, nullable=True)
    p_win_effective = Column(Float, nullable=True)
    
    # --- Stop-loss ---
    market_end_time      = Column(DateTime(timezone=True), nullable=True)
    stop_loss_pct        = Column(Float, nullable=True)
    stop_loss_price      = Column(Float, nullable=True)
    stop_loss_status     = Column(String(20), nullable=True, default="ACTIVE")
    stop_loss_hit_at     = Column(DateTime(timezone=True), nullable=True)
    stop_loss_sell_price = Column(Float, nullable=True)
    stop_loss_sell_size  = Column(Numeric(38, 18), nullable=True)
    
    # --- Take Profit ---
    take_profit_enabled    = Column(Boolean, nullable=True, default=False)
    take_profit_multiplier = Column(Float, nullable=True)
    take_profit_price      = Column(Float, nullable=True)
    take_profit_status     = Column(String(20), nullable=True, default="ACTIVE")
    take_profit_hit_at     = Column(DateTime(timezone=True), nullable=True)
    take_profit_sell_price = Column(Float, nullable=True)
    take_profit_sell_size  = Column(Numeric(38, 18), nullable=True)
    
    # --- Financial Fields & Accounting ---
    position_accounting_version = Column(SmallInteger, nullable=False, server_default="0")
    position_version = Column(Integer, nullable=False, default=1, server_default="1")
    entry_filled_shares = Column(Numeric(38, 18), nullable=True)
    entry_cost_usdc = Column(Numeric(38, 18), nullable=True)
    remaining_shares = Column(Numeric(38, 18), nullable=True)
    realized_pnl_usdc = Column(Numeric(38, 18), nullable=True)
    
    # --- Unified Exit Status ---
    position_status = Column(String(32), nullable=False, default="OPEN")
    exit_reason     = Column(String(32), nullable=True)
    exit_order_id   = Column(String(128), nullable=True)
    exit_attempt_id = Column(UUID(as_uuid=True), nullable=True)
    exit_claimed_at = Column(DateTime(timezone=True), nullable=True)
    last_exit_error = Column(Text, nullable=True)
    exit_attempts   = Column(Integer, nullable=False, default=0)
    closed_at       = Column(DateTime(timezone=True), nullable=True)
    close_price     = Column(Float, nullable=True)

    # --- Settlement & Redemption ---
    settlement_outcome = Column(String(16), nullable=True)
    expected_payout_usdc = Column(Numeric(38, 18), nullable=True)
    redeemable_shares = Column(Numeric(38, 18), nullable=True)
    redemption_status = Column(String(32), nullable=False, server_default="NOT_REQUIRED")
    redemption_tx_hash = Column(String(128), nullable=True)
    redeemed_payout_usdc = Column(Numeric(38, 18), nullable=True)
    redeemed_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), nullable=False)
    config_snapshot = Column(Text, nullable=True)   # JSON паспорт настроек на момент сделки
    model_key = Column(String(64), nullable=True)
    confirm_model_key = Column(String(64), nullable=True)
    confirm_model_version = Column(Integer, nullable=True)
    model_attribution_source = Column(String(16), nullable=True)

    # COMBINED Direction Architecture fields
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
        Index("idx_trade_history_market_id", "market_id"),
                Index("idx_trade_history_model_version", "asset", "model_version", "status", "created_at"),
        Index("idx_trade_model_analytics", "mode", "model_key", "model_version", "position_status", "closed_at"),
        Index("idx_trade_confirm_model_analytics", "mode", "confirm_model_key", "confirm_model_version", "position_status", "closed_at"),
        Index("idx_trade_history_exact_model", "model_key", "model_version", "mode", "position_status", "closed_at"),
        CheckConstraint(
            "position_accounting_version = 0 OR (entry_filled_shares IS NOT NULL AND entry_cost_usdc IS NOT NULL AND remaining_shares IS NOT NULL AND realized_pnl_usdc IS NOT NULL)",
            name="ck_trade_position_accounting_initialized",
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

class CryptoCandle(Base):
    """
    OHLCV-свеча из Binance /api/v3/klines.
    interval: '1m' | '5m' | '15m' | '1h' | '4h'
    symbol:   'BTCUSDT' | 'ETHUSDT'
    """
    __tablename__ = "crypto_candles"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    symbol     = Column(String(32), nullable=False)   # 'BTCUSDT', 'ETHUSDT'
    interval   = Column(String(8),  nullable=False)   # '15m', '1h', etc.
    open_time  = Column(DateTime(timezone=True), nullable=False)
    close_time = Column(DateTime(timezone=True), nullable=True) # Добавлено для Stage 2
    is_closed  = Column(Boolean, nullable=True)                 # Добавлено для Stage 2
    open       = Column(Float, nullable=False)
    high       = Column(Float, nullable=False)
    low        = Column(Float, nullable=False)
    close      = Column(Float, nullable=False)
    volume     = Column(Float, nullable=False)          # base asset volume
    taker_buy_volume = Column(Float, nullable=True)    # агрессивные покупки
    source     = Column(String(16), nullable=False, default="binance")

    __table_args__ = (
        UniqueConstraint("symbol", "interval", "open_time",
                         name="uix_crypto_candle"),
        Index("idx_crypto_candles_symbol_interval", "symbol", "interval"),
        Index("idx_crypto_candles_open_time", "open_time"),
    )


class DecisionFunnelLog(Base):
    """
    Одна запись = один проход через decide_ml_mode / decide_combined_mode.
    Каждый гейт: True = прошёл, False = заблокировал, None = не применялся.
    """
    __tablename__ = "decision_funnel_log"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    created_at   = Column(DateTime(timezone=True), nullable=False)

    # Контекст рынка
    market_id    = Column(String(128), nullable=False)
    asset        = Column(String(32),  nullable=False)
    trading_mode = Column(String(16))  # ML / COMBINED / EXPERT
    execution_mode = Column(String(16), nullable=True) # PAPER / LIVE / SHADOW
    used_model   = Column(String(64),  nullable=True)   # "BTC_contested", "ETH" и т.д.

    # ML-метрики
    p_flip       = Column(Float, nullable=True)
    p_flip_raw   = Column(Float, nullable=True)
    entry_model_ece = Column(Float, nullable=True)
    edge         = Column(Float, nullable=True)
    fresh_price  = Column(Float, nullable=True)
    would_live_accept = Column(Boolean, nullable=True)

    # Пороги, применявшиеся в этом прогоне (для дебага изменений настроек)
    threshold_lower = Column(Float, nullable=True)   # NO_FLIP_THRESHOLD (lower)
    threshold_upper = Column(Float, nullable=True)   # FLIP_THRESHOLD (upper)
    min_edge_used   = Column(Float, nullable=True)

    # Гейты (True=passed, False=blocked, None=not_reached)
    g1_model_loaded     = Column(Boolean, nullable=True)  # модель в кеше
    g2_price_fetched    = Column(Boolean, nullable=True)  # API цена получена
    g3_dead_zone        = Column(Boolean, nullable=True)  # НЕ в dead zone → True
    g4_no_flip          = Column(Boolean, nullable=True)  # p_flip < lower (тренд)
    g5_min_edge         = Column(Boolean, nullable=True)  # edge >= MIN_EDGE
    g6_price_range      = Column(Boolean, nullable=True)  # цена в [MIN_PRICE, MAX_PRICE]
    g7_crypto_confirm   = Column(Boolean, nullable=True)  # LightGBM согласен
    g8_combined_vote    = Column(Boolean, nullable=True)  # финальный голос COMBINED

    # Итог
    final_action = Column(String(16), nullable=False)   # BUY_YES, BUY_NO, SKIP
    skip_reason  = Column(String(256), nullable=True)   # краткая причина если SKIP

    # Паспорт VETO / Подтверждения (Legacy P5)
    primary_model_key = Column(String(64), nullable=True)
    primary_model_version = Column(Integer, nullable=True)
    confirm_model_key = Column(String(64), nullable=True)
    confirm_model_version = Column(Integer, nullable=True)
    proposed_action = Column(String(16), nullable=True)
    proposed_price = Column(Float, nullable=True)
    proposed_amount_usdc = Column(Float, nullable=True)
    confirm_direction = Column(String(16), nullable=True)
    confirm_passed = Column(Boolean, nullable=True)

    # COMBINED Direction Architecture fields
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

    # P0: детальная причина сбоя Direction Model (INFERENCE_FAILED / REGIME_UNAVAILABLE)
    # Содержит: текст исключения, missing regime key, или текст risk_reason
    direction_error_detail = Column(String(512), nullable=True)

    __table_args__ = (
        Index("idx_funnel_asset_created", "asset", "created_at"),
        Index("idx_funnel_market_id",     "market_id"),
        Index("idx_funnel_trading_mode",  "trading_mode", "created_at"),
        Index("idx_funnel_confirm_model_analytics", "confirm_model_key", "confirm_model_version", "created_at"),
        Index("idx_funnel_direction_model", "direction_model_key", "direction_model_version", "created_at"),
        Index("idx_funnel_entry_model", "entry_model_key", "entry_model_version", "created_at"),
        Index("idx_funnel_decision_run", "decision_run_id"),
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

class AIOptimizationRun(Base):
    """Durable lifecycle record for one autonomous optimization run."""

    __tablename__ = "ai_optimization_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    objective = Column(Text, nullable=False)
    scope = Column(JSON().with_variant(JSONB, "postgresql"), nullable=False)
    autonomy_level = Column(String(24), nullable=False, server_default="EXPERIMENT")
    status = Column(String(24), nullable=False, server_default="DRAFT")
    agent_type = Column(String(32), nullable=False, server_default="CODEX")
    agent_thread_id = Column(String(128), nullable=True)
    budget_experiments = Column(Integer, nullable=False, server_default="0")
    budget_seconds = Column(Integer, nullable=False, server_default="0")
    created_by = Column(String(128), nullable=False, server_default="system")
    permission_id = Column(
        Integer,
        ForeignKey("ai_permissions.id", ondelete="SET NULL"),
        nullable=True,
    )
    summary = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("idx_ai_runs_status_created", "status", "created_at"),
        CheckConstraint(
            "autonomy_level IN ('OBSERVE', 'EXPERIMENT', 'SHADOW', 'LIVE_PROPOSE')",
            name="ck_ai_runs_autonomy_level",
        ),
        CheckConstraint(
            "status IN ('DRAFT', 'PLANNING', 'RUNNING', 'EVALUATING', 'SHADOW', "
            "'PENDING_APPROVAL', 'ACTIVE', 'INSUFFICIENT_DATA', 'FAILED', "
            "'REJECTED', 'CANCELLED', 'ROLLED_BACK')",
            name="ck_ai_runs_status",
        ),
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
    owner_token = Column(String(128), nullable=False)
    acquired_at = Column(DateTime(timezone=True), nullable=False)
    heartbeat_at = Column(DateTime(timezone=True), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("idx_ai_worker_leases_expires", "expires_at"),
    )


class AIRunStep(Base):
    """Append-only human-readable and structured audit record for a run step."""

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
    retry_count = Column(Integer, nullable=False, server_default="0")
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
    """Audit record for executor failures that cannot reference ExperimentResult.

    Invalid config IDs or unknown actions cannot satisfy ExperimentResult's
    foreign-key/evaluation-kind constraints. They still need a durable trail.
    """

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
    config_id = Column(Integer, nullable=True)
    action = Column(String(64), nullable=True)
    error_code = Column(String(64), nullable=False)
    error_message = Column(Text, nullable=False)
    payload = Column(JSON().with_variant(JSONB, "postgresql"), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("idx_ai_step_audit_run_created", "run_id", "created_at"),
        Index("idx_ai_step_audit_code", "error_code", "created_at"),
    )


class AIExperimentConfig(Base):
    """Immutable configuration snapshot shared by model families and strategies."""

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
    parent_id = Column(Integer, ForeignKey("experiment_configs.id"), nullable=True)
    created_by = Column(String(128), nullable=False, server_default="system")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("idx_ai_experiment_configs_scope", "asset", "regime", "model_family"),
        Index("idx_ai_experiment_configs_created_at", "created_at"),
    )


class AIModelArtifact(Base):
    """Content-addressed model artifact metadata; the model bytes stay immutable."""

    __tablename__ = "ai_model_artifacts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_registry_id = Column(
        Integer,
        ForeignKey("model_registry.id", ondelete="SET NULL"),
        nullable=True,
    )
    artifact_uri = Column(String(512), nullable=False)
    sha256 = Column(String(64), nullable=False, unique=True)
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
    metrics = Column(JSON().with_variant(JSONB, "postgresql"), nullable=True)
    slices = Column(JSON().with_variant(JSONB, "postgresql"), nullable=True)
    trade_count = Column(Integer, nullable=True)
    net_pnl = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("idx_ai_results_run_status", "run_id", "status"),
        Index("idx_ai_results_config_kind", "config_id", "evaluation_kind"),
        CheckConstraint(
            "evaluation_kind IN ('TRAIN', 'OOT', 'POLYMARKET_OOT', 'SHADOW', 'LIVE')",
            name="ck_ai_results_evaluation_kind",
        ),
        CheckConstraint(
            "status IN ('PENDING', 'RUNNING', 'SUCCEEDED', 'FAILED', 'INSUFFICIENT_DATA')",
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
    manifest_hash = Column(String(64), nullable=False)
    status = Column(String(24), nullable=False, server_default="DRAFT")
    created_by = Column(String(128), nullable=False, server_default="system")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    activated_at = Column(DateTime(timezone=True), nullable=True)
    rolled_back_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("idx_deployment_revisions_status_created", "status", "created_at"),
        CheckConstraint(
            "status IN ('DRAFT', 'SHADOW', 'PENDING_APPROVAL', 'ACTIVE', 'SUPERSEDED', 'REJECTED', 'ROLLED_BACK')",
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
        CheckConstraint(
            "event_type IN ('CREATED', 'SHADOW_ASSIGNED', 'APPROVED', 'ACTIVATED', "
            "'REJECTED', 'ROLLED_BACK')",
            name="ck_deployment_events_type",
        ),
    )


class AIShadowAssignment(Base):
    """Candidate/baseline assignment for passive production observation."""

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

