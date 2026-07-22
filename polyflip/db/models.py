from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, LargeBinary, Index, UniqueConstraint
from sqlalchemy.orm import declarative_base

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
    final_outcome = Column(String(16), nullable=False)  # "YES", "NO", "INVALID"
    flip_vs_final = Column(Boolean, nullable=False)
    recorded_at = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        UniqueConstraint("market_id", "recorded_at", name="uix_market_recorded"),
        Index("idx_market_snapshots_asset", "asset"),
        Index("idx_market_snapshots_asset_time", "asset", "time_left_min"),
        Index("idx_market_snapshots_recorded_at", "recorded_at"),
    )

class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset = Column(String(32), nullable=False)
    version = Column(Integer, nullable=False)
    model_blob = Column(LargeBinary, nullable=False)
    accuracy = Column(Float, nullable=False)
    baseline = Column(Float, nullable=True)
    features = Column(String, nullable=True)
    ece = Column(Float, nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)
    interval = Column(String(5), nullable=False, server_default='15m')
    trained_at = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("idx_model_registry_asset_active", "asset", "is_active"),
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
    pnl = Column(Float, nullable=True)
    edge = Column(Float, nullable=True)
    p_up = Column(Float, nullable=True)
    strike = Column(Float, nullable=True)
    lgbm_metadata = Column(String, nullable=True)
    updated_at = Column(DateTime(timezone=True), nullable=True)
    
    # --- Stop-loss ---
    market_end_time      = Column(DateTime(timezone=True), nullable=True)
    stop_loss_pct        = Column(Float, nullable=True)
    stop_loss_price      = Column(Float, nullable=True)
    stop_loss_status     = Column(String(20), nullable=True, default="ACTIVE")
    stop_loss_hit_at     = Column(DateTime(timezone=True), nullable=True)
    stop_loss_sell_price = Column(Float, nullable=True)
    
    # --- Take Profit ---
    take_profit_enabled    = Column(Boolean, nullable=True, default=False)
    take_profit_multiplier = Column(Float, nullable=True)
    take_profit_price      = Column(Float, nullable=True)
    take_profit_status     = Column(String(20), nullable=True, default="ACTIVE")
    take_profit_hit_at     = Column(DateTime(timezone=True), nullable=True)
    take_profit_sell_price = Column(Float, nullable=True)
    
    created_at = Column(DateTime(timezone=True), nullable=False)
    
    __table_args__ = (
        Index("idx_trade_history_market_id", "market_id"),
        Index("idx_trade_history_model_version", "asset", "model_version", "status", "created_at"),
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
    trading_mode = Column(String(32),  nullable=False)  # ML, COMBINED, CRYPTO, FAVORITE
    used_model   = Column(String(64),  nullable=True)   # "BTC_contested", "ETH" и т.д.

    # ML-метрики
    p_flip       = Column(Float, nullable=True)
    edge         = Column(Float, nullable=True)
    fresh_price  = Column(Float, nullable=True)

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

    __table_args__ = (
        Index("idx_funnel_asset_created", "asset", "created_at"),
        Index("idx_funnel_market_id",     "market_id"),
        Index("idx_funnel_trading_mode",  "trading_mode", "created_at"),
    )
