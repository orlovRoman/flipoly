from enum import StrEnum
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

    @model_validator(mode="after")
    def validate_live_credentials(self):
        if self.execution_mode == ExecutionMode.LIVE:
            if not self.polygon_private_key:
                raise ValueError("POLYGON_PRIVATE_KEY is required in LIVE mode")
            if not self.polygon_address:
                raise ValueError("POLYGON_ADDRESS is required in LIVE mode")
        return self
