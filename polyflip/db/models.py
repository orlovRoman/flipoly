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
    is_active = Column(Boolean, nullable=False, default=True)
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
