from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
from polyflip.constants import LIVE_POLL_INTERVAL_SECONDS as _DEFAULT_POLL_INTERVAL


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://polyflip:secret@db/polyflip"
    API_KEY: str = "test-key"
    ASSETS: str = "BTC,ETH"
    
    # Инфраструктура / Мониторинг
    ALERT_WEBHOOK_URL: str = ""
    COLLECTOR_STALE_HOURS: int = 2
    RATE_LIMIT: str = "60/minute"
    SENTRY_DSN: str = ""

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
    DEAD_ZONE_WIDTH: float = 0.10  # единственный параметр ширины зоны
    TRADING_ENABLED: bool = False
    TRADE_ASSETS: str = "BTC,ETH"
    INITIAL_CAPITAL: float = 1000.0
    TRADE_MIN_PRICE: float = 0.05
    TRADE_MAX_PRICE: float = 0.95
    TRADING_MODE: str = "ml"
    FAVORITE_MODE_ENTRY_SEC: int = 180
    FAVORITE_THRESHOLD: float = 0.55
    # MAX_BET_EDGE = потолок масштабирования ставки (не фильтр аномалий)
    MAX_BET_EDGE: float = 0.40
    # MAX_EDGE_FILTER = фильтр "подозрительный edge → SKIP"
    MAX_EDGE_FILTER: float = 0.20
    
    # Unified Fallbacks
    BET_SIZING_MODE: str = "scaled"
    MAX_BET_SIZE_USDC: float = 50.0
    DAILY_LOSS_LIMIT_USDC: float = -100.0
    FLIP_THRESHOLD: float = 0.60
    MIN_EDGE: float = 0.05
    TRADE_ON_FLIP: bool = False
    AUTO_DEAD_ZONE: bool = True
    OUTSIDER_MAX_PRICE: float = 0.45
    NO_MIN_EDGE: float = 0.04
    FAVORITE_MIN_EDGE: float = -0.01
    CRYPTO_MIN_EDGE: float = 0.05
    COMBINED_NONE_BET_MULTIPLIER: float = 0.5

    @property
    def asset_list(self) -> List[str]:
        return [a.strip() for a in self.ASSETS.split(",") if a.strip()]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
