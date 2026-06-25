from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://polyflip:secret@db/polyflip"
    API_KEY: str = "test-key"
    ASSETS: str = "BTC,ETH"
    
    # Дефолтные настройки, которые могут быть переопределены в БД
    LIVE_POLL_INTERVAL_SECONDS: int = 10
    RETRAIN_INTERVAL_HOURS: int = 24
    MIN_SAMPLES_FOR_MODEL: int = 500
    
    # Новые параметры Фазы 5
    ACTIVE_FEATURES: str = "time_left_min,mid_price,spread,volume_5min,price_velocity,hour_of_day"
    TRADE_EXECUTION_TIME_SEC: int = 30
    TRADE_BET_SIZE_USDC: float = 10.0
    TRADE_NO_FLIP_THRESHOLD: float = 0.15
    TRADE_FLIP_THRESHOLD: float = 0.85
    TRADING_ENABLED: bool = False
    
    # Legacy параметры (оставил на всякий случай)
    DRIFT_THRESHOLD: float = 0.05
    MIN_EDGE: float = 0.05
    BET_FRACTION: float = 0.02
    ALERT_WEBHOOK_URL: str = ""
    COLLECTOR_STALE_HOURS: int = 2
    RATE_LIMIT: str = "60/minute"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def asset_list(self) -> List[str]:
        return [a.strip() for a in self.ASSETS.split(",") if a.strip()]

settings = Settings()
