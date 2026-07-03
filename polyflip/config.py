from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
from polyflip.constants import (
    DEAD_ZONE_WIDTH as _DEFAULT_DEAD_ZONE,
    LIVE_POLL_INTERVAL_SECONDS as _DEFAULT_POLL_INTERVAL,
    MIN_EDGE as _DEFAULT_MIN_EDGE,
    MAX_EDGE_FILTER as _DEFAULT_MAX_BET_EDGE,
    FAVORITE_THRESHOLD as _DEFAULT_FAVORITE_THRESHOLD
)

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://polyflip:secret@db/polyflip"
    API_KEY: str = "test-key"
    ASSETS: str = "BTC,ETH"
    
    # Дефолтные настройки, которые могут быть переопределены в БД
    LIVE_POLL_INTERVAL_SECONDS: int = _DEFAULT_POLL_INTERVAL
    RETRAIN_INTERVAL_HOURS: int = 24
    MIN_SAMPLES_FOR_MODEL: int = 50
    
    # Новые параметры Фазы 5
    ACTIVE_FEATURES: str = "time_left_min,mid_price,spread,volume_5min,price_velocity,hour_of_day"
    TRADE_EXECUTION_TIME_SEC: int = 30
    TRADE_MIN_TIME_LEFT_SEC: int = 10
    TRADE_MAX_TIME_LEFT_SEC: int = 360
    TRADE_BET_SIZE_USDC: float = 10.0
    TRADE_NO_FLIP_THRESHOLD: float = 0.15
    DEAD_ZONE_WIDTH: float = _DEFAULT_DEAD_ZONE
    TRADING_ENABLED: bool = False
    TRADE_ASSETS: str = "BTC,ETH"
    INITIAL_CAPITAL: float = 1000.0
    TRADE_ONLY_FAVORITE: bool = False
    TRADE_MIN_PRICE: float = 0.05
    TRADE_MAX_PRICE: float = 0.95
    TRADING_MODE: str = "ml"
    FAVORITE_MODE_ENTRY_SEC: int = 180
    FAVORITE_THRESHOLD: float = _DEFAULT_FAVORITE_THRESHOLD
    MAX_BET_EDGE: float = _DEFAULT_MAX_BET_EDGE
    
    # Legacy параметры (оставил на всякий случай)
    DRIFT_THRESHOLD: float = 0.05
    MIN_EDGE: float = _DEFAULT_MIN_EDGE
    BET_FRACTION: float = 0.02
    ALERT_WEBHOOK_URL: str = ""
    COLLECTOR_STALE_HOURS: int = 2
    RATE_LIMIT: str = "60/minute"
    SENTRY_DSN: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def asset_list(self) -> List[str]:
        return [a.strip() for a in self.ASSETS.split(",") if a.strip()]

settings = Settings()

if settings.SENTRY_DSN:
    import sentry_sdk
    import os
    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        send_default_pii=False,
        traces_sample_rate=0.1,
        environment=os.getenv("APP_ENV", "production"),
    )
