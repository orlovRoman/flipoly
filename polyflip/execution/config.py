from decimal import Decimal
from enum import StrEnum

LIVE_MIN_GROSS_BUY_USDC = Decimal("1.10")
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ExecutionMode(StrEnum):
    PAPER = "PAPER"
    SHADOW = "SHADOW"
    LIVE = "LIVE"


class ExecutionSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False,
    )

    execution_mode: ExecutionMode = ExecutionMode.PAPER
    polymarket_host: str = "https://clob.polymarket.com"
    polygon_private_key: str | None = None
    polygon_address: str | None = None
    polymarket_relayer_api_key: str | None = None
    polymarket_relayer_api_key_address: str | None = None

    @model_validator(mode="after")
    def validate_live_credentials(self):
        if self.execution_mode == ExecutionMode.LIVE:
            required = {
                "POLYGON_PRIVATE_KEY": self.polygon_private_key,
                "POLYGON_ADDRESS": self.polygon_address,
                "POLYMARKET_RELAYER_API_KEY": self.polymarket_relayer_api_key,
                "POLYMARKET_RELAYER_API_KEY_ADDRESS": self.polymarket_relayer_api_key_address,
            }
            missing = [name for name, value in required.items() if not value]
            if missing:
                raise ValueError(f"Missing LIVE credentials: {', '.join(missing)}")
        return self
