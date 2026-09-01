from typing import List

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from polyflip.constants import LIVE_POLL_INTERVAL_SECONDS as _DEFAULT_POLL_INTERVAL


AI_LAB_MODES = frozenset({"STANDARD", "RESEARCH"})


def normalize_ai_lab_mode(value: str) -> str:
    mode = str(value).strip().upper()
    if mode not in AI_LAB_MODES:
        raise ValueError(
            f"AI_LAB_MODE must be one of {sorted(AI_LAB_MODES)}, got {value!r}"
        )
    return mode


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
    LIVE_TRADING_ENABLED: bool = False
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

    # Weighted trading policy.  LEGACY preserves the current hard-vote
    # behavior; shadow/active are explicit rollout switches.
    TRADING_POLICY_MODE: str = "LEGACY"
    WEIGHTED_POLICY_ID: str = "UNVERSIONED"
    WEIGHTED_MARKET_WEIGHT: float = 0.90
    WEIGHTED_LOGREG_WEIGHT: float = 0.05
    WEIGHTED_LGBM_WEIGHT: float = 0.05
    WEIGHTED_MRF_BETA: float = 0.0
    WEIGHTED_INTERCEPT: float = 0.0
    WEIGHTED_FEE_RATE: float = 0.07
    WEIGHTED_MAKER_FEE_RATE: float = 0.0
    WEIGHTED_FEE_EXPONENT: float = 1.0
    WEIGHTED_SLIPPAGE_RATE: float = 0.005
    WEIGHTED_LATENCY_BUFFER: float = 0.0
    WEIGHTED_EXECUTION_ROLE: str = "TAKER"
    WEIGHTED_MIN_NET_EV_FAVORITE: float = 0.03
    WEIGHTED_MIN_NET_EV_OUTSIDER: float = 0.03
    WEIGHTED_FIXED_BET_USDC: float = 1.0
    WEIGHTED_MRF_EXTREME_VETO_THRESHOLD: float = -1.0
    WEIGHTED_MODELS_AGREE_BETA: float = 0.0
    WEIGHTED_MRF_APPLICATION: str = "PROBABILITY"
    WEIGHTED_MRF_SIZING_GAMMA: float = 0.0
    WEIGHTED_POLICY_ARTIFACT_PATH: str = ""
    WEIGHTED_SIZING_MODE: str = "FIXED"
    WEIGHTED_STANDARD_ERROR: float = 0.0
    WEIGHTED_KELLY_FRACTION: float = 0.025
    WEIGHTED_SIZE_CAP_USDC: float = 3.0

    # AI Lab LLM & Autonomous Loop (Phase 10)
    OPENAI_API_KEY: str = ""
    AI_LAB_LLM_PROVIDER: str = "openai"
    AI_LAB_MODEL_RESEARCH: str = "gpt-4o"
    AI_LAB_MODEL_SUMMARY: str = "gpt-4o-mini"
    AI_LAB_LLM_STORE: bool = False
    AI_LAB_LLM_API_KEY: str = ""
    AI_LAB_LLM_ENDPOINT: str = ""
    AI_LAB_LLM_AVAILABLE_PROVIDERS: str = "mock,openai,opencode"
    AI_LAB_ALLOWED_MODELS: str = ""
    # Optional Codex thread lifecycle; "none" keeps research usable without SDK.
    AI_LAB_THREAD_PROVIDER: str = "none"
    AI_LAB_MAX_RUNTIME_SECONDS: int = 3600
    AI_LAB_MAX_COST_USD: float = 10.0
    AI_LAB_MODE: str = "STANDARD"
    # Research remains a bounded, offline schedule even when enabled through
    # environment configuration. These values mirror the scheduler hard caps.
    AI_LAB_RESEARCH_MAX_ITERATIONS: int = 1
    AI_LAB_RESEARCH_MAX_STEPS: int = 1
    AI_LAB_RESEARCH_INTERVAL_SECONDS: float = 0.0
    AI_LAB_RESEARCH_LEASE_TTL_SECONDS: float = 120.0
    AI_LAB_SCHEDULE_ENABLED: bool = False
    AI_LAB_SCHEDULE_CRON: str = ""
    AI_LAB_MAX_DAILY_RUNS: int = 1
    AI_LAB_MAX_CONCURRENT_RUNS: int = 1
    # Dynamic OpenCode model discovery for the independent research agent.
    AI_LAB_OPENCODE_MODELS_ENDPOINT: str = ""
    AI_LAB_OPENCODE_CATALOG_TTL_SECONDS: int = 3600
    AI_LAB_OPENCODE_MODELS_FALLBACK: str = ""
    AI_LAB_OPENCODE_RESPONSES_ENDPOINT: str = "https://opencode.ai/zen/v1/responses"
    AI_LAB_OPENCODE_CHAT_ENDPOINT: str = "https://opencode.ai/zen/v1/chat/completions"
    AI_LAB_OPENCODE_CHAT_MODELS: str = "big-pickle,nemotron-3-ultra-free"
    AI_LAB_OPENCODE_PROBE_TTL_SECONDS: int = 86400
    # External ai_research_agent container authentication (falls back to API_KEY).
    AI_LAB_AGENT_TOKEN: str = ""
    AI_LAB_AGENT_LEASE_TTL_SECONDS: int = 120

    @field_validator("AI_LAB_MODE")
    @classmethod
    def validate_ai_lab_mode(cls, value: str) -> str:
        return normalize_ai_lab_mode(value)

    @model_validator(mode="after")
    def validate_ai_lab_safety(self) -> "Settings":
        # PAPER/SHADOW may run research even when the general trading switch
        # is on; only the explicit LIVE gate blocks RESEARCH at startup.
        if self.AI_LAB_MODE == "RESEARCH" and self.LIVE_TRADING_ENABLED:
            raise ValueError(
                "AI_LAB_MODE=RESEARCH cannot be used while LIVE_TRADING_ENABLED=true"
            )
        if self.AI_LAB_MAX_DAILY_RUNS < 1:
            raise ValueError("AI_LAB_MAX_DAILY_RUNS must be at least 1")
        if self.AI_LAB_MAX_CONCURRENT_RUNS < 1:
            raise ValueError("AI_LAB_MAX_CONCURRENT_RUNS must be at least 1")
        return self

    @property
    def asset_list(self) -> List[str]:
        return [a.strip() for a in self.ASSETS.split(",") if a.strip()]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
