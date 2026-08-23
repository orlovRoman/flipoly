from decimal import Decimal
from enum import StrEnum
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LIVE_MIN_GROSS_BUY_USDC = Decimal("1.10")
# Polymarket CLOB rejects orders smaller than five outcome tokens.  Keep this
# as a gateway-level invariant so both FAK and resting-limit submissions fail
# locally instead of consuming an API request and being reported as an
# opaque provider error.  The value is deliberately expressed in shares, not
# USDC: the required notional therefore depends on the current limit price.
POLYMARKET_MIN_ORDER_SHARES = Decimal("5")


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
    # The live worker is intentionally kept alive in deployments where the
    # global kill switch is OFF and credentials are not mounted.  Keep this
    # optional so local/tests without the setting retain strict validation.
    live_trading_enabled: bool = False
    polymarket_host: str = "https://clob.polymarket.com"
    polygon_private_key: str | None = None
    polygon_address: str | None = None
    polymarket_relayer_api_key: str | None = None
    polymarket_relayer_api_key_address: str | None = None

    @model_validator(mode="after")
    def validate_live_credentials(self):
        if self.execution_mode == ExecutionMode.LIVE and self.live_trading_enabled is not False:
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
