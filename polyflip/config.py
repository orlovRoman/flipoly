from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
from polyflip.constants import LIVE_POLL_INTERVAL_SECONDS as _DEFAULT_POLL_INTERVAL


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://polyflip:secret@db/polyflip"
    API_KEY: str = "test-key"
    ASSETS: str = "BTC,ETH"

    # Monitoring / Alerts
    ALERT_WEBHOOK_URL: str = ""
    COLLECTOR_STALE_HOURS: int = 2
    RATE_LIMIT: str = "60/minute"
    SENTRY_DSN: str = ""

    # Polling & Retrain Intervals
    LIVE_POLL_INTERVAL_SECONDS: int = _DEFAULT_POLL_INTERVAL
    TRADE_JOB_INTERVAL_SECONDS: int = 15
    RETRAIN_INTERVAL_HOURS: int = 24
    MIN_SAMPLES_FOR_MODEL: int = 50

    # Trading Execution Parameters
    ACTIVE_FEATURES: str = "time_left_min,mid_price,spread,volume_5min,price_velocity,hour_of_day"
    TRADE_EXECUTION_TIME_SEC: int = 30
    TRADE_MIN_TIME_LEFT_SEC: int = 10
    TRADE_MAX_TIME_LEFT_SEC: int = 360
    TRADE_BET_SIZE_USDC: float = 10.0
    TRADE_NO_FLIP_THRESHOLD: float = 0.15
    DEAD_ZONE_WIDTH: float = 0.10
    TRADING_ENABLED: bool = False
    TRADE_ASSETS: str = "BTC,ETH"
    INITIAL_CAPITAL: float = 1000.0
    TRADE_MIN_PRICE: float = 0.05
    TRADE_MAX_PRICE: float = 0.95
    TRADING_MODE: str = "ml"
    FAVORITE_THRESHOLD: float = 0.55

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
    COMBINED_NONE_BET_MULTIPLIER: float = 0.0

    # AI Lab LLM & Autonomous Loop (Phase 10)
    OPENAI_API_KEY: str = ""
    AI_LAB_LLM_PROVIDER: str = "openai"
    AI_LAB_MODEL_RESEARCH: str = "gpt-4o"
    AI_LAB_MODEL_SUMMARY: str = "gpt-4o-mini"
    AI_LAB_LLM_STORE: bool = False
    # STANDARD keeps evidence gates strict; RESEARCH permits provisional SHADOW
    # candidates while never enabling live execution.
    AI_LAB_MODE: str = "STANDARD"
    AI_LAB_MAX_RUNTIME_SECONDS: int = 3600
    AI_LAB_MAX_COST_USD: float = 10.0

    @property
    def asset_list(self) -> List[str]:
        return [a.strip() for a in self.ASSETS.split(",") if a.strip()]

    @property
    def ai_lab_research_mode(self) -> bool:
        return str(self.AI_LAB_MODE or "STANDARD").strip().upper() == "RESEARCH"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
