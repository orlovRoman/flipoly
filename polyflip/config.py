from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
from polyflip.constants import (
    DEAD_ZONE_WIDTH as _DEFAULT_DEAD_ZONE,
    LIVE_POLL_INTERVAL_SECONDS as _DEFAULT_POLL_INTERVAL,
    MIN_EDGE as _DEFAULT_MIN_EDGE,
    MAX_EDGE_SCALING as _DEFAULT_MAX_BET_EDGE,    # для масштабирования ставки (0.40)
    MAX_EDGE_FILTER as _DEFAULT_MAX_EDGE_FILTER,  # для фильтра аномалий (0.20)
    FAVORITE_THRESHOLD as _DEFAULT_FAVORITE_THRESHOLD,
    DAILY_LOSS_LIMIT_USDC as _DEFAULT_DAILY_LOSS_LIMIT,
    FLIP_THRESHOLD as _DEFAULT_FLIP_THRESHOLD,
    OUTSIDER_MAX_PRICE as _DEFAULT_OUTSIDER_MAX_PRICE,
    NO_MIN_EDGE as _DEFAULT_NO_MIN_EDGE,
    CRYPTO_MIN_EDGE as _DEFAULT_CRYPTO_MIN_EDGE,
    FAVORITE_MIN_EDGE as _DEFAULT_FAVORITE_MIN_EDGE,
    COMBINED_NONE_BET_MULTIPLIER as _DEFAULT_COMBINED_NONE_BET_MULTIPLIER,
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
    DEAD_ZONE_WIDTH: float = _DEFAULT_DEAD_ZONE  # единственный параметр ширины зоны
    TRADING_ENABLED: bool = False
    TRADE_ASSETS: str = "BTC,ETH"
    INITIAL_CAPITAL: float = 1000.0
    TRADE_MIN_PRICE: float = 0.05
    TRADE_MAX_PRICE: float = 0.95
    TRADING_MODE: str = "ml"
    FAVORITE_MODE_ENTRY_SEC: int = 180
    FAVORITE_THRESHOLD: float = _DEFAULT_FAVORITE_THRESHOLD
    # MAX_BET_EDGE = потолок масштабирования ставки (не фильтр аномалий)
    MAX_BET_EDGE: float = _DEFAULT_MAX_BET_EDGE
    # MAX_EDGE_FILTER = фильтр "подозрительный edge → SKIP"
    MAX_EDGE_FILTER: float = _DEFAULT_MAX_EDGE_FILTER
    
    # Unified Fallbacks
    BET_SIZING_MODE: str = "scaled"
    MAX_BET_SIZE_USDC: float = 50.0
    DAILY_LOSS_LIMIT_USDC: float = _DEFAULT_DAILY_LOSS_LIMIT
    FLIP_THRESHOLD: float = _DEFAULT_FLIP_THRESHOLD
    MIN_EDGE: float = _DEFAULT_MIN_EDGE
    TRADE_ON_FLIP: bool = False
    AUTO_DEAD_ZONE: bool = True
    OUTSIDER_MAX_PRICE: float = _DEFAULT_OUTSIDER_MAX_PRICE
    NO_MIN_EDGE: float = _DEFAULT_NO_MIN_EDGE
    FAVORITE_MIN_EDGE: float = _DEFAULT_FAVORITE_MIN_EDGE
    MAX_PRICE_DRIFT: float = 0.03
    CRYPTO_MIN_EDGE: float = _DEFAULT_CRYPTO_MIN_EDGE
    USE_CRYPTO_CONFIRM: bool = False
    CRYPTO_STANDALONE: bool = False
    COMBINED_NONE_BET_MULTIPLIER: float = _DEFAULT_COMBINED_NONE_BET_MULTIPLIER
    
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
